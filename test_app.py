"""
Smoke tests for the Wine Review Sentiment API.

Run with:
    pytest

These don't require the real trained model: they patch `app.model` with a
tiny fake so the tests run fast and don't depend on training data.
"""

import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class FakeModel:
    """Deterministic stand-in for the real sklearn pipeline: any review
    containing the word 'good' scores high quality, everything else low."""

    def predict(self, texts):
        return [1 if "good" in t.lower() else 0 for t in texts]

    def predict_proba(self, texts):
        out = []
        for t in texts:
            if "good" in t.lower():
                out.append([0.1, 0.9])
            else:
                out.append([0.8, 0.2])
        return out


@pytest.fixture()
def client(monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "model", FakeModel())
    monkeypatch.setattr(app_module, "MODEL_LOAD_ERROR", None)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_single(client):
    res = client.post("/predict", json={"description": "a really good wine"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["label"] == "HIGH_QUALITY_PROXY"


def test_predict_rejects_empty_text(client):
    res = client.post("/predict", json={"description": "   "})
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "INVALID_TEXT"


def test_predict_batch(client):
    res = client.post("/predict_batch", json={"descriptions": ["good wine", "bad wine"]})
    assert res.status_code == 200
    results = res.get_json()["results"]
    assert results[0]["label"] == "HIGH_QUALITY_PROXY"
    assert results[1]["label"] == "LOW_QUALITY_PROXY"


def test_predict_file_txt(client):
    data = {"file": (io.BytesIO(b"good wine\nbad wine\n"), "reviews.txt")}
    res = client.post("/predict_file", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    summary = res.get_json()["summary"]
    assert summary["total"] == 2
    assert summary["high_quality_count"] == 1


def test_stats_returns_history(client):
    client.post("/predict", json={"description": "good wine"})
    res = client.get("/stats")
    assert res.status_code == 200
    assert len(res.get_json()["history"]) >= 1


def test_unknown_route_returns_json_404(client):
    res = client.get("/does-not-exist")
    assert res.status_code == 404
    assert res.get_json()["error"]["code"] == "NOT_FOUND"
