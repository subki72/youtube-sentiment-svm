"""Unit tests for unified TF-IDF + SVM pipeline, evaluation, and serialization."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from youtube_sentiment.model import (
    SENTIMENT_LABELS,
    SentimentClassifier,
    build_svm_pipeline,
    clear_classifier_cache,
    generate_sample_comments,
    get_or_train_default_classifier,
)


def test_generate_sample_comments_is_balanced_and_deterministic() -> None:
    """Synthetic offline dataset generator balances classes and respects seed."""
    texts_a, labels_a = generate_sample_comments(n_per_class=12, seed=42)
    texts_b, labels_b = generate_sample_comments(n_per_class=12, seed=42)

    assert texts_a == texts_b
    assert labels_a == labels_b
    assert len(texts_a) == 24
    assert labels_a.count("positif") == 12
    assert labels_a.count("negatif") == 12

    with pytest.raises(ValueError, match="n_per_class"):
        generate_sample_comments(n_per_class=1)


def test_build_svm_pipeline_validates_hyperparameters() -> None:
    """Unified pipeline contains TF-IDF bigram vectorizer and balanced SVC."""
    pipeline = build_svm_pipeline(max_features=500, c_value=1.5, kernel="linear")
    assert list(pipeline.named_steps.keys()) == ["tfidf", "svm"]
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)
    assert pipeline.named_steps["svm"].class_weight == "balanced"

    with pytest.raises(ValueError, match="max_features"):
        build_svm_pipeline(max_features=0)
    with pytest.raises(ValueError, match="c_value"):
        build_svm_pipeline(c_value=-1.0)


def test_classifier_train_predict_and_stratified_metrics() -> None:
    """Trained classifier achieves high holdout accuracy and predicts expected polarities."""
    texts, labels = generate_sample_comments(n_per_class=30, seed=42)
    classifier = SentimentClassifier(max_features=1000)
    assert not classifier.is_fitted

    metrics = classifier.train(texts, labels, test_size=0.2, random_state=42)
    assert classifier.is_fitted
    assert metrics.train_size == 48
    assert metrics.test_size == 12
    assert metrics.accuracy >= 0.8
    assert metrics.f1_macro >= 0.8
    assert sum(sum(row) for row in metrics.confusion_matrix) == 12

    pos_pred = classifier.predict_one("Videonya bagus banget dan penjelasannya mantul!")
    neg_pred = classifier.predict_one("Berita bohong parah banget bikin kecewa!")
    assert pos_pred.label == "positif"
    assert neg_pred.label == "negatif"
    assert 0.0 < pos_pred.confidence <= 1.0
    assert set(pos_pred.probabilities.keys()) == set(SENTIMENT_LABELS)


def test_classifier_train_with_hyperparam_tuning() -> None:
    """GridSearchCV hyperparameter tuning integrates cleanly with Pipeline."""
    texts, labels = generate_sample_comments(n_per_class=20, seed=7)
    classifier = SentimentClassifier()
    metrics = classifier.train(texts, labels, test_size=0.2, tune_hyperparams=True)

    assert metrics.best_params is not None
    assert "svm__C" in metrics.best_params
    assert metrics.f1_macro >= 0.7


def test_classifier_batch_prediction_and_edge_cases() -> None:
    """Batch prediction works on valid items and noise-only comment does not crash."""
    texts, labels = generate_sample_comments(n_per_class=30, seed=42)
    classifier = SentimentClassifier()

    with pytest.raises(RuntimeError, match="trained or loaded"):
        classifier.predict_one("konten bagus")

    classifier.train(texts, labels, test_size=0.2)
    results = classifier.predict_batch(["Mantap banget videonya keren", "12345 @user"])
    assert len(results) == 2
    assert results[0].label == "positif"
    assert results[1].cleaned_text == ""
    assert results[1].label in SENTIMENT_LABELS


def test_classifier_validation_errors_on_invalid_datasets_and_inputs() -> None:
    """Guard clauses raise clear ValueError on invalid training or prediction data."""
    classifier = SentimentClassifier()

    with pytest.raises(ValueError, match="identical length"):
        classifier.train(["bagus"], ["positif", "negatif"])
    with pytest.raises(ValueError, match="cannot be empty"):
        classifier.train([], [])
    with pytest.raises(ValueError, match="test_size"):
        classifier.train(["bagus"] * 10, ["positif"] * 10, test_size=0.9)
    with pytest.raises(ValueError, match="at least 10"):
        classifier.train(["bagus", "jelek"], ["positif", "negatif"])

    texts, labels = generate_sample_comments(n_per_class=15, seed=42)
    classifier.train(texts, labels)
    with pytest.raises(ValueError, match="non-empty string"):
        classifier.predict_one("   ")
    with pytest.raises(ValueError, match="cannot be empty"):
        classifier.predict_batch([])
    with pytest.raises(ValueError, match="index 1"):
        classifier.predict_batch(["bagus", "   "])


def test_classifier_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Unified joblib pipeline saves, loads, and produces identical predictions."""
    texts, labels = generate_sample_comments(n_per_class=15, seed=42)
    classifier = SentimentClassifier()
    classifier.train(texts, labels)

    bundle_path = tmp_path / "models" / "sentiment_pipeline.joblib"
    saved = classifier.save(bundle_path)
    assert saved.exists()

    reloaded = SentimentClassifier.load(saved)
    orig = classifier.predict_one("Keren banget analisisnya mantap")
    restored = reloaded.predict_one("Keren banget analisisnya mantap")
    assert restored == orig

    with pytest.raises(FileNotFoundError):
        SentimentClassifier.load(tmp_path / "missing.joblib")

    corrupt_path = tmp_path / "corrupt.joblib"
    joblib.dump({"not_a_pipeline": True}, corrupt_path)
    with pytest.raises(ValueError, match="Invalid sentiment model"):
        SentimentClassifier.load(corrupt_path)


def test_get_or_train_default_classifier_caches_and_loads_artifact(tmp_path: Path) -> None:
    """Default classifier caches in memory and loads persisted artifact if present."""
    clear_classifier_cache()
    texts, labels = generate_sample_comments(n_per_class=12, seed=10)
    trained = SentimentClassifier()
    trained.train(texts, labels)
    artifact = trained.save(tmp_path / "saved.joblib")

    loaded_first = get_or_train_default_classifier(artifact)
    loaded_second = get_or_train_default_classifier(artifact)
    assert loaded_first is loaded_second
    assert loaded_first.is_fitted
    clear_classifier_cache()
