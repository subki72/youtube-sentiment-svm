"""API integration tests for FastAPI sentiment inference endpoints and resilience."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from youtube_sentiment.api import create_app
from youtube_sentiment.config import Settings
from youtube_sentiment.model import SentimentClassifier, generate_sample_comments


@pytest.fixture()
def trained_classifier() -> SentimentClassifier:
    """Provide an offline pre-trained SVM classifier for isolated API tests."""
    texts, labels = generate_sample_comments(n_per_class=20, seed=42)
    classifier = SentimentClassifier()
    classifier.train(texts, labels, test_size=0.2, random_state=42)
    return classifier


def test_health_endpoint_returns_ready_status(trained_classifier: SentimentClassifier) -> None:
    """GET /health returns ok status and reports fitted model readiness."""
    app = create_app(classifier=trained_classifier)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["version"] == "1.0.0"


def test_predict_endpoint_classifies_positive_and_negative_comments(
    trained_classifier: SentimentClassifier,
) -> None:
    """POST /predict returns label, confidence, cleaned_text, and probabilities."""
    app = create_app(classifier=trained_classifier)
    client = TestClient(app)

    pos_resp = client.post("/predict", json={"text": "Videonya keren bgt mantul sekali"})
    assert pos_resp.status_code == 200
    pos_data = pos_resp.json()
    assert pos_data["label"] == "positif"
    assert 0.0 < pos_data["confidence"] <= 1.0
    assert "keren" in pos_data["cleaned_text"]

    neg_resp = client.post("/predict", json={"text": "Parah bgt informasinya bohong hoax"})
    assert neg_resp.status_code == 200
    assert neg_resp.json()["label"] == "negatif"


def test_predict_endpoint_rejects_blank_input(trained_classifier: SentimentClassifier) -> None:
    """POST /predict returns HTTP 422 on whitespace-only or empty payload."""
    app = create_app(classifier=trained_classifier)
    client = TestClient(app)

    blank_resp = client.post("/predict", json={"text": "   "})
    assert blank_resp.status_code == 422

    empty_resp = client.post("/predict", json={"text": ""})
    assert empty_resp.status_code == 422


def test_analyze_endpoint_scores_batch_and_computes_summary(
    trained_classifier: SentimentClassifier,
) -> None:
    """POST /analyze returns per-comment results plus aggregate sentiment counts."""
    app = create_app(classifier=trained_classifier)
    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={
            "comments": [
                "Konten bagus bgt sangat bermanfaat",
                "Parah banget isinya bohong semua",
                "Mantap jiwa penjelasannya jelas",
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert data["summary"]["total"] == 3
    assert data["summary"]["positif"] == 2
    assert data["summary"]["negatif"] == 1
    assert data["summary"]["positive_ratio"] == pytest.approx(0.6667, abs=0.001)


def test_rate_limiter_enforces_quota_and_cors_headers(
    trained_classifier: SentimentClassifier,
) -> None:
    """Rate limiter returns 429 after exceeding max requests per minute."""
    settings = Settings(
        rate_limit_per_minute=2,
        cors_origins=("http://localhost:3000",),
    )
    app = create_app(classifier=trained_classifier, settings=settings)
    client = TestClient(app)

    assert client.post("/predict", json={"text": "bagus"}).status_code == 200
    assert client.post("/predict", json={"text": "jelek"}).status_code == 200
    limited = client.post(
        "/predict",
        json={"text": "mantap"},
        headers={"Origin": "http://localhost:3000"},
    )
    assert limited.status_code == 429
    assert limited.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_global_exception_handler_hides_internal_tracebacks() -> None:
    """Unhandled exceptions return safe generic 500 JSON without leaking details."""

    class _BrokenClassifier(SentimentClassifier):
        @property
        def is_fitted(self) -> bool:
            return True

        def predict_one(self, text: str) -> object:  # type: ignore[override]
            raise RuntimeError("sensitive internal path /secret/model.joblib")

    app = create_app(classifier=_BrokenClassifier())
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/predict", json={"text": "bagus banget"})
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_server_error"
    assert "sensitive" not in body["detail"]
