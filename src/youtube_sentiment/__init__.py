"""Modular Indonesian YouTube sentiment analysis package."""

from youtube_sentiment.api import create_app
from youtube_sentiment.config import Settings, configure_logging, get_settings
from youtube_sentiment.lexicon import CORE_SLANG_MAP, load_slang_dictionary, normalize_slang
from youtube_sentiment.model import (
    EvaluationMetrics,
    PredictionResult,
    SentimentClassifier,
    build_svm_pipeline,
    generate_sample_comments,
    get_or_train_default_classifier,
)
from youtube_sentiment.preprocessing import clean_raw_text, preprocess_batch, preprocess_text

__version__ = "1.0.0"

__all__ = [
    "CORE_SLANG_MAP",
    "EvaluationMetrics",
    "PredictionResult",
    "SentimentClassifier",
    "Settings",
    "__version__",
    "build_svm_pipeline",
    "clean_raw_text",
    "configure_logging",
    "create_app",
    "generate_sample_comments",
    "get_or_train_default_classifier",
    "get_settings",
    "load_slang_dictionary",
    "normalize_slang",
    "preprocess_batch",
    "preprocess_text",
]
