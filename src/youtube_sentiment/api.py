"""FastAPI service exposing Indonesian YouTube sentiment inference endpoints."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from youtube_sentiment.config import Settings, configure_logging, get_settings
from youtube_sentiment.model import (
    PredictionResult,
    SentimentClassifier,
    get_or_train_default_classifier,
)

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"


class HealthResponse(BaseModel):
    """Service liveness and readiness response schema."""

    status: str
    service: str
    version: str
    model_loaded: bool


class PredictRequest(BaseModel):
    """Single YouTube comment classification payload."""

    text: str = Field(..., min_length=1, max_length=2000, description="Raw YouTube comment text.")

    @field_validator("text")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        """Reject whitespace-only comments."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Comment text must not be empty or whitespace-only.")
        return stripped


class PredictResponse(BaseModel):
    """Sentiment classification output for one comment."""

    label: str
    confidence: float
    cleaned_text: str
    probabilities: dict[str, float]


class BatchAnalyzeRequest(BaseModel):
    """Batch sentiment analysis payload for multiple YouTube comments."""

    comments: list[str] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="List of raw YouTube comments.",
    )

    @field_validator("comments")
    @classmethod
    def validate_comments(cls, values: list[str]) -> list[str]:
        """Ensure every comment in the batch is non-empty after stripping."""
        cleaned: list[str] = []
        for idx, comment in enumerate(values):
            stripped = comment.strip()
            if not stripped:
                raise ValueError(f"Comment at index {idx} must not be whitespace-only.")
            cleaned.append(stripped)
        return cleaned


class BatchSummary(BaseModel):
    """Aggregate sentiment counts and positive ratio for a batch."""

    total: int
    positif: int
    negatif: int
    positive_ratio: float


class BatchAnalyzeResponse(BaseModel):
    """Batch sentiment prediction response with summary statistics."""

    results: list[PredictResponse]
    summary: BatchSummary


class ErrorResponse(BaseModel):
    """Safe external error payload without internal stack traces."""

    error: str
    detail: str


class RateLimiter:
    """Thread-safe sliding-window per-client rate limiter."""

    def __init__(self, max_requests: int = 120, window_seconds: float = 60.0) -> None:
        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("Rate limiter max_requests and window_seconds must be positive.")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_key: str) -> None:
        """Record a request hit and raise HTTP 429 if rate limit is exceeded."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._hits[client_key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                logger.warning("Rate limit exceeded for client %s", client_key)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please retry later.",
                )
            bucket.append(now)


def _to_predict_response(result: PredictionResult) -> PredictResponse:
    """Convert internal ``PredictionResult`` dataclass to API response schema."""
    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        cleaned_text=result.cleaned_text,
        probabilities=result.probabilities,
    )


def _register_exception_handlers(application: FastAPI) -> None:
    """Attach global validation and unhandled exception handlers."""

    @application.exception_handler(ValueError)
    async def handle_value_error(_request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("Request validation error: %s", exc)
        payload = ErrorResponse(error="validation_error", detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(),
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled internal server error: %s", exc, exc_info=True)
        payload = ErrorResponse(
            error="internal_server_error",
            detail="An unexpected internal error occurred.",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(),
        )


def _build_batch_response(predictions: list[PredictionResult]) -> BatchAnalyzeResponse:
    """Aggregate batch predictions into a structured ``BatchAnalyzeResponse``."""
    results = [_to_predict_response(item) for item in predictions]
    positif_count = sum(1 for item in results if item.label == "positif")
    negatif_count = len(results) - positif_count
    ratio = round(positif_count / len(results), 4)
    logger.info(
        "Analyzed batch of %d comments (positif=%d, negatif=%d).",
        len(results),
        positif_count,
        negatif_count,
    )
    return BatchAnalyzeResponse(
        results=results,
        summary=BatchSummary(
            total=len(results),
            positif=positif_count,
            negatif=negatif_count,
            positive_ratio=ratio,
        ),
    )


def _register_routes(
    application: FastAPI,
    settings: Settings,
    resolve_classifier: Callable[[], SentimentClassifier],
    rate_limiter: RateLimiter,
) -> None:
    """Register health check and sentiment inference endpoints."""

    def enforce_rate_limit(request: Request) -> None:
        client_host = request.client.host if request.client else "unknown"
        rate_limiter.check(client_host)

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    def health_check() -> HealthResponse:
        """Return service health and model readiness status."""
        model = resolve_classifier()
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            version=API_VERSION,
            model_loaded=model.is_fitted,
        )

    @application.post(
        "/predict",
        response_model=PredictResponse,
        tags=["inference"],
        dependencies=[Depends(enforce_rate_limit)],
    )
    def predict_comment(payload: PredictRequest) -> PredictResponse:
        """Classify sentiment for a single Indonesian YouTube comment."""
        model = resolve_classifier()
        result = model.predict_one(payload.text)
        logger.info("Predicted sentiment=%s confidence=%.4f", result.label, result.confidence)
        return _to_predict_response(result)

    @application.post(
        "/analyze",
        response_model=BatchAnalyzeResponse,
        tags=["inference"],
        dependencies=[Depends(enforce_rate_limit)],
    )
    def analyze_comments(payload: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
        """Classify sentiment for a batch of comments and compute aggregate metrics."""
        model = resolve_classifier()
        return _build_batch_response(model.predict_batch(payload.comments))


def create_app(
    classifier: SentimentClassifier | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create and configure the FastAPI sentiment inference application.

    Args:
        classifier: Optional injected ``SentimentClassifier`` instance.
        settings: Optional injected ``Settings`` instance.

    Returns:
        Configured ``FastAPI`` application instance.
    """
    active_settings = settings or get_settings()
    configure_logging(active_settings.log_level)
    rate_limiter = RateLimiter(max_requests=active_settings.rate_limit_per_minute)

    application = FastAPI(
        title="YouTube Sentiment SVM API",
        version=API_VERSION,
        description="Indonesian YouTube comment sentiment classification via TF-IDF & SVM.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    state: dict[str, SentimentClassifier | None] = {"classifier": classifier}

    def resolve_classifier() -> SentimentClassifier:
        if state["classifier"] is None:
            state["classifier"] = get_or_train_default_classifier(active_settings.model_path)
        return state["classifier"]

    _register_exception_handlers(application)
    _register_routes(application, active_settings, resolve_classifier, rate_limiter)
    return application


app = create_app()
