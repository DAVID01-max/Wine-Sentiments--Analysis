"""
Wine Review Sentiment API
==========================
A Flask app that serves the model saved by your notebook
(artifacts/best_model.joblib).

What it does, in plain words:
    1. When the server starts, it loads the saved model ONCE.
    2. Someone sends it a wine review (text), or a WHOLE FILE of reviews.
    3. The model says whether each review sounds "high quality" (positive
       sentiment) or "low quality" (negative sentiment), and how confident
       it is.
    4. The home page DRAWS the results: a bar chart for a single review, a
       running trend line of recent confidence, and a drag-and-drop file
       upload that scores a whole batch of reviews at once and shows a
       sentiment breakdown + a sortable results table.

Endpoints:
    GET  /               -> the web page (text box + file upload + charts)
    GET  /health          -> "is the server up, is the model loaded, since when"
    POST /predict         -> score ONE review
    POST /predict_batch   -> score MANY reviews, sent as JSON
    POST /predict_file    -> score MANY reviews, sent as an uploaded .csv/.txt file
    GET  /stats            -> recent prediction history (used by the trend chart)

Run locally:
    python app.py

Production notes:
    - Configuration is environment-driven (see .env.example). Nothing here
      needs to be hand-edited to move between local/staging/prod.
    - Structured logging replaces print() so output is timestamped and
      filterable by level (LOG_LEVEL=DEBUG|INFO|WARNING|ERROR).
    - PREDICTION_HISTORY and the request-rate tracker are protected by a
      lock because Flask/gunicorn can serve requests from multiple threads
      in the same process. They are still per-process/in-memory (not
      shared across worker processes) -- see the History note below.
    - A simple in-memory rate limiter guards the scoring endpoints. It is
      per-process, not distributed -- fine for a single instance behind a
      load balancer with sticky sessions, or as a basic safety net;
      swap in Redis/nginx rate limiting for a multi-instance deployment.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
from flask import Flask, Response, g, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
# Everything tunable lives in one place and is read from the environment,
# so the same code runs unmodified in dev / staging / prod -- you configure
# it with env vars instead of editing source. See .env.example for the
# full list with explanations.
@dataclass(frozen=True)
class Config:
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    model_path: str = field(default_factory=lambda: os.environ.get(
        "MODEL_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "best_model.joblib"),
    ))
    max_text_length: int = int(os.environ.get("MAX_TEXT_LENGTH", 5000))
    max_batch_size: int = int(os.environ.get("MAX_BATCH_SIZE", 100))
    max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_BYTES", 2 * 1024 * 1024))  # 2 MB
    history_maxlen: int = int(os.environ.get("HISTORY_MAXLEN", 20))
    rate_limit_per_minute: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", 60))
    allowed_origins: str = os.environ.get("ALLOWED_ORIGINS", "*")
    log_level: str = os.environ.get("LOG_LEVEL", "INFO").upper()
    debug: bool = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port: int = int(os.environ.get("PORT", 5000))


CONFIG = Config()
APP_VERSION = "2.0.0"
START_TIME = time.time()

# Column names we'll look for in an uploaded CSV (checked in this order,
# case-insensitive). If none of these exist, we fall back to the first column.
CSV_TEXT_COLUMNS = ["description", "review", "text", "sentiment_text", "comment"]

# Human-friendly names for the model's 0 / 1 output.
# (Same wording as the notebook: this is a PROXY for quality, not the truth.)
LABEL_NAMES = {0: "LOW_QUALITY_PROXY", 1: "HIGH_QUALITY_PROXY"}


# ---------------------------------------------------------------------------
# 2. LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, CONFIG.log_level, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
logger = logging.getLogger("wine_sentiment_api")


# ---------------------------------------------------------------------------
# 3. THREAD-SAFE SHARED STATE
# ---------------------------------------------------------------------------
# A deque with maxlen automatically drops the oldest entry once it's full,
# so this never grows without bound. This is IN-MEMORY ONLY: it resets
# whenever the server restarts, and if you run multiple server processes
# each one has its own separate history. That's fine for a demo / single
# instance app; swap in a real database (or Redis) if you need it to
# persist or be shared across workers.
_history_lock = threading.Lock()
PREDICTION_HISTORY: Deque[Dict[str, Any]] = deque(maxlen=CONFIG.history_maxlen)

REQUEST_COUNT = 0
_request_count_lock = threading.Lock()

# Naive fixed-window rate limiter: {ip: [timestamps within the last 60s]}.
_rate_lock = threading.Lock()
_rate_window: Dict[str, Deque[float]] = {}


def _is_rate_limited(ip: str, limit_per_minute: int) -> bool:
    """Return True if `ip` has exceeded `limit_per_minute` requests in the
    trailing 60 seconds. Cheap, dependency-free, good enough for a single
    process; not a substitute for a real rate limiter at scale."""
    now = time.time()
    with _rate_lock:
        window = _rate_window.setdefault(ip, deque())
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit_per_minute:
            return True
        window.append(now)
        return False


# ---------------------------------------------------------------------------
# 4. LOAD THE MODEL (once, at startup)
# ---------------------------------------------------------------------------
# Loading is slow-ish, so we do it a single time here rather than on every
# request. The saved object is a scikit-learn Pipeline, which means the TF-IDF
# step and the classifier travel together: we can feed it raw text directly.
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = CONFIG.max_upload_bytes  # Flask returns 413 automatically past this


def load_model(model_path: str):
    """Load the trained pipeline, failing loudly and clearly if it can't."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Create it by running:  python train_model.py\n"
            "(or run the whole notebook, then copy artifacts/best_model.joblib here)."
        )
    try:
        loaded = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - we want to catch and re-raise with context
        raise RuntimeError(f"Model file at {model_path} could not be loaded: {exc}") from exc
    logger.info("Model loaded from %s", model_path)
    return loaded


try:
    model = load_model(CONFIG.model_path)
    MODEL_LOAD_ERROR: Optional[str] = None
except (FileNotFoundError, RuntimeError) as exc:
    # We don't crash the process on import: this lets /health report a clear
    # "model not loaded" status instead of the server failing to boot with a
    # stack trace that's hard to see if you're running behind a supervisor.
    model = None
    MODEL_LOAD_ERROR = str(exc)
    logger.error("Model failed to load: %s", MODEL_LOAD_ERROR)


# ---------------------------------------------------------------------------
# 5. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def score_reviews(texts: List[str]) -> List[Dict[str, Any]]:
    """Run the model on a list of review strings.
    Returns a list of dicts, one per review."""
    # predict()       -> the winning class for each text: 0 or 1
    # predict_proba() -> probability of each class, e.g. [0.12, 0.88]
    labels = model.predict(texts)
    probabilities = model.predict_proba(texts)

    results = []
    new_history_entries = []
    for text, label, probs in zip(texts, labels, probabilities):
        label = int(label)
        result = {
            "label": LABEL_NAMES[label],
            # How sure the model is about the label it chose:
            "confidence": round(float(probs[label]), 3),
            # Probability of "high quality" specifically (handy as a score):
            "high_quality_probability": round(float(probs[1]), 3),
        }
        results.append(result)

        # Remember this prediction so the /stats endpoint (and the charts on
        # the home page) can show a running history. We only keep a short
        # text preview, not the full review, to keep each entry small.
        new_history_entries.append(
            {
                "timestamp": time.time(),
                "text_preview": text[:60] + ("..." if len(text) > 60 else ""),
                "label": result["label"],
                "confidence": result["confidence"],
                "high_quality_probability": result["high_quality_probability"],
            }
        )

    with _history_lock:
        PREDICTION_HISTORY.extend(new_history_entries)

    return results


def check_text(value: Any) -> Optional[str]:
    """Return an error message if `value` is not a usable review, else None."""
    if not isinstance(value, str) or not value.strip():
        return "Review text must be a non-empty string."
    if len(value) > CONFIG.max_text_length:
        return f"Review is too long (max {CONFIG.max_text_length} characters)."
    return None


def error(message: str, status: int = 400, code: Optional[str] = None) -> Tuple[Response, int]:
    """Build a consistent JSON error response.

    Shape: {"error": {"message": "...", "code": "SOME_CODE"}}
    `code` is a short machine-readable string a client can branch on
    without parsing the human-readable message.
    """
    return jsonify({"error": {"message": message, "code": code or "BAD_REQUEST"}}), status


def extract_texts_from_upload(file_storage) -> Tuple[List[str], Optional[str]]:
    """Pull a list of review strings out of an uploaded file.

    Supports two simple formats:
      - .txt  -> one review per line
      - .csv  -> looks for a column named like "description"/"review"/"text"
                 (see CSV_TEXT_COLUMNS above); if none match, uses the first
                 column. A header row is optional -- if the first row doesn't
                 look like a known column name, we just treat it as data too.

    Returns (texts, error_message). Exactly one of them will be "empty"
    (texts=[] on error, error_message=None on success).
    """
    filename = (file_storage.filename or "").lower()
    raw_bytes = file_storage.read()

    try:
        raw_text = raw_bytes.decode("utf-8-sig")  # handles Excel's UTF-8 BOM too
    except UnicodeDecodeError:
        return [], "Could not read the file as text (expected UTF-8 encoding)."

    if filename.endswith(".csv"):
        reader = csv.reader(io.StringIO(raw_text))
        rows = [row for row in reader if row]  # drop blank lines
        if not rows:
            return [], "The CSV file appears to be empty."

        header = [cell.strip().lower() for cell in rows[0]]
        column_index = 0
        has_header = False
        for wanted in CSV_TEXT_COLUMNS:
            if wanted in header:
                column_index = header.index(wanted)
                has_header = True
                break

        data_rows = rows[1:] if has_header else rows
        texts = [row[column_index].strip() for row in data_rows if len(row) > column_index]

    elif filename.endswith(".txt"):
        texts = [line.strip() for line in raw_text.splitlines()]

    else:
        return [], "Unsupported file type. Please upload a .csv or .txt file."

    # Drop empty lines/cells.
    texts = [t for t in texts if t]

    if not texts:
        return [], "No review text was found in the file."
    if len(texts) > CONFIG.max_batch_size:
        return [], f"Too many reviews in the file (max {CONFIG.max_batch_size})."

    return texts, None


def require_model(handler):
    """Decorator: short-circuit with a clear 503 if the model never loaded,
    instead of every route re-checking `model is None` and instead of a
    500 with a confusing AttributeError."""

    def wrapped(*args, **kwargs):
        if model is None:
            return error(
                f"Model is not available: {MODEL_LOAD_ERROR}",
                status=503,
                code="MODEL_UNAVAILABLE",
            )
        return handler(*args, **kwargs)

    wrapped.__name__ = handler.__name__
    return wrapped


def rate_limited(handler):
    """Decorator: apply the per-IP rate limit to a route."""

    def wrapped(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
        if _is_rate_limited(ip, CONFIG.rate_limit_per_minute):
            return error(
                "Too many requests. Please slow down and try again shortly.",
                status=429,
                code="RATE_LIMITED",
            )
        return handler(*args, **kwargs)

    wrapped.__name__ = handler.__name__
    return wrapped


# ---------------------------------------------------------------------------
# 6. REQUEST LOGGING + SECURITY HEADERS
# ---------------------------------------------------------------------------
@app.before_request
def _start_timer():
    g.start_time = time.time()
    global REQUEST_COUNT
    with _request_count_lock:
        REQUEST_COUNT += 1


@app.after_request
def _log_and_secure(response: Response) -> Response:
    duration_ms = round((time.time() - getattr(g, "start_time", time.time())) * 1000, 1)
    logger.info(
        "%s %s -> %s (%sms)",
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    # Baseline security headers. Cheap, uncontroversial, and expected of a
    # production API.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer-when-downgrade")

    if CONFIG.allowed_origins:
        response.headers.setdefault("Access-Control-Allow-Origin", CONFIG.allowed_origins)
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")

    return response


# ---------------------------------------------------------------------------
# 7. ERROR HANDLERS (consistent JSON, no stack traces leaked to clients)
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def _not_found(_exc):
    return error("That endpoint doesn't exist.", status=404, code="NOT_FOUND")


@app.errorhandler(405)
def _method_not_allowed(_exc):
    return error("That HTTP method isn't allowed on this endpoint.", status=405, code="METHOD_NOT_ALLOWED")


@app.errorhandler(413)
def _payload_too_large(_exc):
    return error(
        f"Upload is too large (max {CONFIG.max_upload_bytes // (1024 * 1024)} MB).",
        status=413,
        code="PAYLOAD_TOO_LARGE",
    )


@app.errorhandler(Exception)
def _unhandled_error(exc: Exception):
    # Never leak internals (stack traces, file paths) to the client; log
    # them instead, and return a generic message.
    logger.exception("Unhandled error while processing %s %s", request.method, request.path)
    return error("Something went wrong on our end. Please try again.", status=500, code="INTERNAL_ERROR")


# ---------------------------------------------------------------------------
# 8. ROUTES (the URLs people can call)
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Used by you (and by hosting platforms / load balancers) to check the
    server is alive and the model is actually usable."""
    return jsonify(
        {
            "status": "ok" if model is not None else "degraded",
            "model_loaded": model is not None,
            "model_error": MODEL_LOAD_ERROR,
            "version": APP_VERSION,
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "requests_served": REQUEST_COUNT,
        }
    )


@app.route("/predict", methods=["POST"])
@rate_limited
@require_model
def predict():
    """Score ONE review.

    Send:     {"description": "a rich, velvety wine with a long finish"}
    Receive:  {"label": "HIGH_QUALITY_PROXY", "confidence": 0.91, ...}
    """
    payload = request.get_json(silent=True)  # silent=True -> None on bad JSON instead of crashing
    if payload is None:
        return error("Send a JSON body with Content-Type: application/json.", code="INVALID_JSON")

    text = payload.get("description")
    problem = check_text(text)
    if problem:
        return error(problem, code="INVALID_TEXT")

    return jsonify(score_reviews([text])[0])


@app.route("/predict_batch", methods=["POST"])
@rate_limited
@require_model
def predict_batch():
    """Score MANY reviews at once (much faster than calling /predict in a loop).

    Send:     {"descriptions": ["review one", "review two"]}
    Receive:  {"results": [{...}, {...}]}   (same order as the input)
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return error("Send a JSON body with Content-Type: application/json.", code="INVALID_JSON")

    texts = payload.get("descriptions")
    if not isinstance(texts, list) or not texts:
        return error("'descriptions' must be a non-empty list of strings.", code="INVALID_TEXT")
    if len(texts) > CONFIG.max_batch_size:
        return error(f"Too many reviews (max {CONFIG.max_batch_size} per request).", code="BATCH_TOO_LARGE")

    for i, text in enumerate(texts):
        problem = check_text(text)
        if problem:
            return error(f"Item {i}: {problem}", code="INVALID_TEXT")

    return jsonify({"results": score_reviews(texts)})


@app.route("/predict_file", methods=["POST"])
@rate_limited
@require_model
def predict_file():
    """Score every review inside an UPLOADED FILE (.csv or .txt).

    This is the endpoint behind the "Upload a file" section of the home
    page. Send it as multipart/form-data with the file under the field
    name "file", e.g. with curl:

        curl -F "file=@my_reviews.csv" http://localhost:5000/predict_file

    Receive:
        {
          "results": [{"text": "...", "label": "...", "confidence": ..., ...}, ...],
          "summary": {
             "total": 12,
             "high_quality_count": 7,
             "low_quality_count": 5,
             "high_quality_pct": 58.3,
             "average_confidence": 0.81
          }
        }
    """
    if "file" not in request.files:
        return error("No file was uploaded. Attach it under the field name 'file'.", code="NO_FILE")

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return error("No file was selected.", code="NO_FILE")

    texts, problem = extract_texts_from_upload(uploaded)
    if problem:
        return error(problem, code="INVALID_FILE")

    scored = score_reviews(texts)

    # Build a per-review result that includes the original text (trimmed for
    # display), plus a summary block the front end uses for the big stat
    # cards and the sentiment breakdown chart.
    results = [
        {
            "text": text if len(text) <= 200 else text[:200] + "...",
            **scored_item,
        }
        for text, scored_item in zip(texts, scored)
    ]

    high_count = sum(1 for r in results if r["label"] == LABEL_NAMES[1])
    low_count = len(results) - high_count
    avg_confidence = sum(r["confidence"] for r in results) / len(results)

    summary = {
        "total": len(results),
        "high_quality_count": high_count,
        "low_quality_count": low_count,
        "high_quality_pct": round(100 * high_count / len(results), 1),
        "average_confidence": round(avg_confidence, 3),
    }

    return jsonify({"results": results, "summary": summary})


@app.route("/stats", methods=["GET"])
def stats():
    """Return the last few predictions (most recent last).

    This powers the trend chart on the home page, but it's a normal JSON
    endpoint too -- feel free to call it yourself to see what's been
    predicted recently:
        curl http://localhost:5000/stats
    """
    with _history_lock:
        history = list(PREDICTION_HISTORY)
    return jsonify({"history": history})


# ---------------------------------------------------------------------------
# 9. THE WEB PAGE
# ---------------------------------------------------------------------------
# Everything below is front-end code (HTML + CSS + a little JavaScript),
# served as one string by render_template_string(). It's split into three
# parts for clarity:
#   - single review check (bar chart of this one prediction)
#   - trend line (confidence across recent predictions, from /stats)
#   - drag-and-drop file upload -> batch sentiment analysis
#     (doughnut chart of high vs. low counts + a color-coded results table)
#
# Charts are drawn with Chart.js, loaded from a CDN -- no install needed.
HOME_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wine Review Sentiment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {
    /* ---- Creamy-blue palette ---- */
    --cream:      #f6f1e6;   /* page background */
    --cream-deep: #ece2cd;   /* page background, bottom of gradient */
    --card:       #fffdf8;   /* card surfaces */
    --stone:      #efe6d2;   /* subtle fill (stat tiles, table hover) */
    --line:       #e3d7bd;   /* hairline borders */
    --ink:        #24303d;   /* body text */
    --muted:      #74808f;   /* secondary text */
    --blue:       #2c5b82;   /* primary action color */
    --blue-deep:  #1c3c57;   /* hover / gradient end */
    --blue-soft:  #7fa6c4;   /* secondary chart series */
    --teal:       #3e8a76;   /* "high quality" signal */
    --terracotta: #bf6a4b;   /* "low quality" signal */
  }
  * { box-sizing: border-box; }
  body {
    font-family: "Inter", -apple-system, "Segoe UI", Roboto, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 20px 80px;
    background: linear-gradient(180deg, var(--cream) 0%, var(--cream-deep) 100%);
    color: var(--ink);
    line-height: 1.5;
  }
  header { margin-bottom: 32px; }
  header .eyebrow {
    font-size: 0.82rem;
    color: var(--blue);
    font-weight: 600;
    margin: 0 0 6px;
  }
  header h1 {
    font-family: "Fraunces", Georgia, serif;
    font-weight: 600;
    font-size: 2.3rem;
    margin: 0 0 8px;
    color: var(--blue-deep);
    letter-spacing: -0.01em;
  }
  header p { color: var(--muted); margin: 0; max-width: 60ch; }

  .card {
    background: var(--card);
    border-radius: 14px;
    padding: 26px;
    margin-bottom: 22px;
    box-shadow: 0 6px 20px rgba(28, 60, 87, 0.06);
    border: 1px solid var(--line);
  }
  .card h2 {
    font-family: "Fraunces", Georgia, serif;
    font-weight: 600;
    margin-top: 0;
    font-size: 1.2rem;
    color: var(--blue-deep);
  }
  .card .hint { color: var(--muted); margin-top: -10px; font-size: 0.92rem; }

  textarea {
    width: 100%;
    border: 1.5px solid var(--line);
    border-radius: 10px;
    padding: 12px;
    font-size: 0.95rem;
    resize: vertical;
    font-family: inherit;
    background: var(--cream);
    transition: border-color 0.15s;
  }
  textarea:focus { outline: none; border-color: var(--blue); }

  button {
    background: var(--blue);
    color: white;
    border: none;
    padding: 11px 24px;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    transition: transform 0.1s, background 0.15s;
  }
  button:hover { background: var(--blue-deep); transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  button:disabled { background: #b9c3cc; cursor: not-allowed; transform: none; }

  #result_text {
    font-weight: 600;
    min-height: 1.4em;
    color: var(--blue-deep);
  }

  /* --- Drag-and-drop upload zone --- */
  .dropzone {
    border: 2px dashed #c9b98f;
    border-radius: 12px;
    padding: 36px 20px;
    text-align: center;
    color: var(--muted);
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .dropzone.dragover {
    border-color: var(--blue);
    background: #eef4f8;
    color: var(--blue-deep);
  }
  .dropzone strong { color: var(--blue-deep); }
  .dropzone small { display: block; margin-top: 6px; }
  #file_input { display: none; }
  #file_name {
    margin-top: 10px;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--blue-deep);
  }

  .spinner {
    display: none;
    width: 18px; height: 18px;
    border: 3px solid var(--line);
    border-top-color: var(--blue);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin: 14px auto 0;
  }
  .spinner.active { display: block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* --- Summary stat cards for a batch upload --- */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
    margin: 18px 0;
  }
  .stat {
    background: var(--stone);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }
  .stat .value { font-size: 1.5rem; font-weight: 700; color: var(--blue-deep); }
  .stat .label { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

  .charts-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: center;
  }
  @media (max-width: 640px) { .charts-row { grid-template-columns: 1fr; } }

  table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.88rem; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; font-size: 0.78rem; }
  tbody tr:hover { background: var(--stone); }
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    color: white;
  }
  .badge.high { background: var(--teal); }
  .badge.low { background: var(--terracotta); }

  #upload_error { color: var(--terracotta); font-weight: 600; }

  footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 36px;
  }
  footer span.dot { margin: 0 6px; }
</style>
</head>
<body>

<header>
  <p class="eyebrow">Sentiment scoring tool</p>
  <h1>Wine Review Sentiment</h1>
  <p>Score a single review, or upload a whole file for batch sentiment analysis.</p>
</header>

<!-- ============================== -->
<!-- 1. SINGLE REVIEW CHECK          -->
<!-- ============================== -->
<div class="card">
  <h2>Check one review</h2>
  <textarea id="text" rows="4"
    placeholder="Paste a wine review here, e.g. 'A rich, velvety wine with a long, elegant finish...'"></textarea>
  <br><br>
  <button onclick="send()">Predict</button>
  <p id="result_text"></p>

  <div class="charts-row">
    <div>
      <h3>This prediction</h3>
      <canvas id="probChart" height="140"></canvas>
    </div>
    <div>
      <h3>Confidence trend (last {{ history_maxlen }})</h3>
      <canvas id="trendChart" height="140"></canvas>
    </div>
  </div>
</div>

<!-- ============================== -->
<!-- 2. FILE UPLOAD (BATCH) SECTION  -->
<!-- ============================== -->
<div class="card">
  <h2>Batch sentiment analysis: upload a file</h2>
  <p class="hint">
    Upload a <strong>.csv</strong> (with a "description"/"review"/"text" column) or a
    <strong>.txt</strong> file (one review per line). Up to {{ max_batch_size }} reviews per file.
  </p>

  <div class="dropzone" id="dropzone">
    <strong>Click to choose a file</strong> or drag and drop it here
    <small>.csv or .txt &middot; max {{ max_upload_mb }}&nbsp;MB</small>
    <div id="file_name"></div>
  </div>
  <input type="file" id="file_input" accept=".csv,.txt">
  <div style="text-align:center;">
    <div class="spinner" id="upload_spinner"></div>
  </div>
  <p id="upload_error"></p>

  <div id="upload_results" style="display:none;">
    <div class="stat-grid">
      <div class="stat"><div class="value" id="stat_total">0</div><div class="label">Reviews</div></div>
      <div class="stat"><div class="value" id="stat_high">0</div><div class="label">High quality</div></div>
      <div class="stat"><div class="value" id="stat_low">0</div><div class="label">Low quality</div></div>
      <div class="stat"><div class="value" id="stat_conf">0%</div><div class="label">Avg confidence</div></div>
    </div>

    <canvas id="sentimentChart" height="90"></canvas>

    <table>
      <thead>
        <tr><th>Review</th><th>Sentiment</th><th>Confidence</th></tr>
      </thead>
      <tbody id="results_body"></tbody>
    </table>
  </div>
</div>

<footer>
  Wine Review Sentiment API <span class="dot">&middot;</span> v{{ app_version }}
</footer>

<script>
  // ---------- Charts for the single-review section ----------
  const probChart = new Chart(document.getElementById("probChart"), {
    type: "bar",
    data: {
      labels: ["Low quality", "High quality"],
      datasets: [{ label: "Probability", data: [0, 0], backgroundColor: ["#bf6a4b", "#2c5b82"] }]
    },
    options: {
      scales: { y: { beginAtZero: true, max: 1 } },
      plugins: { legend: { display: false } }
    }
  });

  const trendChart = new Chart(document.getElementById("trendChart"), {
    type: "line",
    data: {
      labels: [],
      datasets: [{ label: "Confidence", data: [], tension: 0.25, borderColor: "#2c5b82", backgroundColor: "rgba(44,91,130,0.08)", fill: true }]
    },
    options: { scales: { y: { beginAtZero: true, max: 1 } } }
  });

  async function refreshTrend() {
    try {
      const res = await fetch("/stats");
      const { history } = await res.json();
      trendChart.data.labels = history.map((_, i) => "#" + (i + 1));
      trendChart.data.datasets[0].data = history.map(h => h.confidence);
      trendChart.update();
    } catch (err) {
      // Trend chart is a nice-to-have; a transient failure shouldn't block the page.
      console.warn("Could not refresh trend chart:", err);
    }
  }

  async function send() {
    const textEl = document.getElementById("text");
    const resultEl = document.getElementById("result_text");
    if (!textEl.value.trim()) {
      resultEl.textContent = "Type or paste a review first.";
      return;
    }
    resultEl.textContent = "Scoring...";
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({description: textEl.value})
      });
      const data = await res.json();

      if (data.error) {
        resultEl.textContent = "Error: " + data.error.message;
        return;
      }

      resultEl.textContent = `${data.label} (confidence: ${data.confidence})`;

      probChart.data.datasets[0].data = [
        1 - data.high_quality_probability,
        data.high_quality_probability
      ];
      probChart.update();

      refreshTrend();
    } catch (err) {
      resultEl.textContent = "Something went wrong reaching the server.";
    }
  }

  refreshTrend(); // draw trend chart with whatever history already exists

  // ---------- File upload (batch) section ----------
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file_input");
  const fileNameEl = document.getElementById("file_name");
  const spinner = document.getElementById("upload_spinner");
  const uploadError = document.getElementById("upload_error");
  const uploadResults = document.getElementById("upload_results");

  let sentimentChart = null; // created lazily on first successful upload

  dropzone.addEventListener("click", () => fileInput.click());

  ["dragover", "dragenter"].forEach(evt =>
    dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("dragover"); })
  );
  ["dragleave", "dragend", "drop"].forEach(evt =>
    dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove("dragover"); })
  );
  dropzone.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  async function handleFile(file) {
    fileNameEl.textContent = file.name;
    uploadError.textContent = "";
    uploadResults.style.display = "none";
    spinner.classList.add("active");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/predict_file", { method: "POST", body: formData });
      const data = await res.json();

      if (data.error) {
        uploadError.textContent = data.error.message;
        return;
      }
      renderUploadResults(data);
      refreshTrend(); // batch predictions also feed the trend chart
    } catch (err) {
      uploadError.textContent = "Something went wrong uploading the file.";
    } finally {
      spinner.classList.remove("active");
    }
  }

  function renderUploadResults(data) {
    const { results, summary } = data;

    document.getElementById("stat_total").textContent = summary.total;
    document.getElementById("stat_high").textContent = summary.high_quality_count;
    document.getElementById("stat_low").textContent = summary.low_quality_count;
    document.getElementById("stat_conf").textContent =
      Math.round(summary.average_confidence * 100) + "%";

    // Doughnut chart of high vs. low counts.
    const ctx = document.getElementById("sentimentChart");
    const chartData = {
      labels: ["High quality", "Low quality"],
      datasets: [{
        data: [summary.high_quality_count, summary.low_quality_count],
        backgroundColor: ["#2c5b82", "#bf6a4b"]
      }]
    };
    if (sentimentChart) {
      sentimentChart.data = chartData;
      sentimentChart.update();
    } else {
      sentimentChart = new Chart(ctx, {
        type: "doughnut",
        data: chartData,
        options: { plugins: { legend: { position: "bottom" } } }
      });
    }

    // Color-coded results table, most confident first.
    const sorted = [...results].sort((a, b) => b.confidence - a.confidence);
    const rows = sorted.map(r => {
      const isHigh = r.label === "HIGH_QUALITY_PROXY";
      return `<tr>
        <td>${escapeHtml(r.text)}</td>
        <td><span class="badge ${isHigh ? "high" : "low"}">${isHigh ? "High" : "Low"}</span></td>
        <td>${Math.round(r.confidence * 100)}%</td>
      </tr>`;
    }).join("");
    document.getElementById("results_body").innerHTML = rows;

    uploadResults.style.display = "block";
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HOME_PAGE,
        history_maxlen=CONFIG.history_maxlen,
        max_batch_size=CONFIG.max_batch_size,
        max_upload_mb=CONFIG.max_upload_bytes // (1024 * 1024),
        app_version=APP_VERSION,
    )


# ---------------------------------------------------------------------------
# 10. START THE SERVER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # This is Flask's built-in development server: fine for testing on your
    # laptop. For real deployment use gunicorn (see README.md), e.g.:
    #   gunicorn -w 4 -b 0.0.0.0:5000 app:app
    logger.info("Starting Wine Review Sentiment API v%s on port %s (debug=%s)",
                APP_VERSION, CONFIG.port, CONFIG.debug)
    app.run(host="0.0.0.0", port=CONFIG.port, debug=CONFIG.debug)
