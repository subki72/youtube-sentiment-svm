"""Unified CLI entrypoint for Indonesian YouTube Sentiment Analysis (SVM)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure src/ is importable when running `python main.py` directly from repo root.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from youtube_sentiment.config import configure_logging, get_settings  # noqa: E402
from youtube_sentiment.lexicon import load_slang_dictionary  # noqa: E402
from youtube_sentiment.model import (  # noqa: E402
    SentimentClassifier,
    generate_sample_comments,
    get_or_train_default_classifier,
)
from youtube_sentiment.preprocessing import preprocess_text  # noqa: E402

logger = logging.getLogger(__name__)


def run_health_check() -> int:
    """Verify offline NLP lexicon, stemmer, and model pipeline readiness."""
    settings = get_settings()
    slang = load_slang_dictionary(allow_remote=False)
    sample_clean = preprocess_text("Kontennya bagus bgt mantul!", slang_dict=slang)
    classifier = get_or_train_default_classifier(settings.model_path)
    payload = {
        "status": "ok",
        "service": settings.app_name,
        "lexicon_entries": len(slang),
        "sample_preprocessed": sample_clean,
        "model_fitted": classifier.is_fitted,
    }
    logger.info("Health check passed (%d lexicon entries).", len(slang))
    print(json.dumps(payload, indent=2))
    return 0


def run_demo(save_model_path: Path | None = None) -> int:
    """Train and evaluate the SVM pipeline on the offline synthetic Indonesian corpus."""
    texts, labels = generate_sample_comments(n_per_class=30, seed=42)
    classifier = SentimentClassifier()
    metrics = classifier.train(texts, labels, test_size=0.2, random_state=42)

    demo_samples = [
        "Videonya keren bgt mantul penjelasannya jelas",
        "Parah bgt informasinya bohong dan bikin kecewa",
    ]
    predictions = [asdict(classifier.predict_one(item)) for item in demo_samples]

    if save_model_path is not None:
        classifier.save(save_model_path)

    output = {
        "accuracy": metrics.accuracy,
        "f1_macro": metrics.f1_macro,
        "train_size": metrics.train_size,
        "test_size": metrics.test_size,
        "confusion_matrix": metrics.confusion_matrix,
        "labels": metrics.labels,
        "sample_predictions": predictions,
    }
    logger.info(
        "Offline demo complete: accuracy=%.4f f1_macro=%.4f",
        metrics.accuracy,
        metrics.f1_macro,
    )
    print(json.dumps(output, indent=2))
    return 0


def run_predict_text(comment_text: str) -> int:
    """Classify a single Indonesian YouTube comment and emit JSON."""
    settings = get_settings()
    classifier = get_or_train_default_classifier(settings.model_path)
    result = classifier.predict_one(comment_text)
    logger.info(
        "CLI prediction completed: label=%s confidence=%.4f",
        result.label,
        result.confidence,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


def run_serve(host: str, port: int) -> int:
    """Start the FastAPI inference server via uvicorn."""
    import uvicorn

    logger.info("Starting FastAPI server on %s:%d", host, port)
    uvicorn.run("youtube_sentiment.api:app", host=host, port=port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Indonesian YouTube Comment Sentiment Analysis (TF-IDF + SVM)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--health", action="store_true", help="Run readiness and health check.")
    group.add_argument("--demo", action="store_true", help="Run offline training/evaluation demo.")
    group.add_argument("--text", type=str, help="Classify sentiment for a single comment string.")
    group.add_argument("--serve", action="store_true", help="Run FastAPI HTTP service.")
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path to save trained model during --demo.",
    )
    parser.add_argument("--host", default=settings.api_host, help="API bind host for --serve.")
    parser.add_argument("--port", type=int, default=settings.api_port, help="API port for --serve.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute CLI command based on parsed arguments."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.health:
            return run_health_check()
        if args.demo:
            return run_demo(save_model_path=args.save_model)
        if args.text is not None:
            return run_predict_text(args.text)
        if args.serve:
            return run_serve(host=args.host, port=args.port)
    except ValueError as exc:
        logger.error("CLI input validation error: %s", exc)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
