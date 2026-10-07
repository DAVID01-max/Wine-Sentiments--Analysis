"""Flask API for the wine quality-proxy model (served by gunicorn on Render)."""
import json
import os

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

_here = os.path.dirname(os.path.abspath(__file__))
_art = os.path.join(_here, "artifacts")
model = joblib.load(os.path.join(_art, "best_model.joblib"))

try:
    with open(os.path.join(_art, "model_card.json")) as f:
        model_card = json.load(f)
except FileNotFoundError:
    model_card = {}

PAGE = """<!doctype html><title>Wine review classifier</title>
<body style="font-family:system-ui;max-width:560px;margin:40px auto;padding:0 16px">
<h2>Wine review classifier</h2>
<textarea id="t" rows="5" style="width:100%" placeholder="Paste a wine review..."></textarea>
<p><button onclick="go()">Predict</button></p><pre id="o"></pre>
<script>
async function go(){
  const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({description:document.getElementById('t').value})});
  document.getElementById('o').textContent = JSON.stringify(await r.json(),null,2);
}
</script></body>"""


@app.get("/")
def index():
    return PAGE


@app.get("/health")
def health():
    return jsonify(status="ok", model=model_card.get("model_name", "unknown"))


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("description") or "").strip()
    if not text:
        return jsonify(error="Missing 'description' field"), 400

    label = int(model.predict([text])[0])
    confidence = float(model.predict_proba([text])[0][label])
    return jsonify(
        label="HIGH_QUALITY_PROXY" if label == 1 else "LOW_QUALITY_PROXY",
        confidence=round(confidence, 3),
    )


if __name__ == "__main__":  # local dev only; Render uses gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
