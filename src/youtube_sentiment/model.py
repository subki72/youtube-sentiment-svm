"""Unified TF-IDF + Class-Weighted SVM sentiment modeling and offline dataset tools."""

from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from youtube_sentiment.lexicon import load_slang_dictionary
from youtube_sentiment.preprocessing import preprocess_batch, preprocess_text

logger = logging.getLogger(__name__)

SENTIMENT_LABELS: tuple[str, str] = ("negatif", "positif")
MIN_TRAINING_SAMPLES = 10
DEFAULT_MAX_FEATURES = 2000

_POSITIVE_TEMPLATES: tuple[str, ...] = (
    "Kontennya bagus banget dan penjelasannya sangat jelas mantap",
    "Keren bgt videonya sangat membantu dan menginspirasi sukses terus",
    "Penjelasan ekonomi ini mantul bikin makin paham kondisi sekarang",
    "Alhamdulillah kebijakan baru ini berdampak positif dan hebat sekali",
    "Setuju banget sama analisisnya sangat objektif dan bermutu",
    "Suka banget dgn pembahasan yg mendalam dan mudah dipahami",
    "Mantap jiwa presenternya jujur dan datanya akurat sekali",
    "Kualitas video makin bagus suaranya jernih dan informatif",
    "Luar biasa kerjanya sangat nyata dan membawa harapan baik",
    "Terima kasih ilmunya sangat bermanfaat buat masyarakat luas",
    "Salut dengan langkah berani yg pro rakyat dan transparan",
    "Pembahasan yg cerdas dan solusi ekonominya sangat masuk akal",
    "Videonya top banget bikin semangat belajar hal baru",
    "Bagus sekali penyampaiannya tidak bertele tele dan langsung inti",
    "Dukung penuh program positif ini semoga semakin maju dan sukses",
)

_NEGATIVE_TEMPLATES: tuple[str, ...] = (
    "Parah banget penjelasannya ngawur dan bikin kecewa penonton",
    "Berita bohong dan menyesatkan jangan percaya sama informasi hoax ini",
    "Jelek bgt kualitas argumennya gak masuk akal dan berantakan",
    "Kecewa berat kebijakan ini malah bikin susah rakyat kecil",
    "Kontennya ancur tidak bermutu dan cuma cari sensasi belaka",
    "Penipuan publik semua datanya palsu dan penuh kebohongan",
    "Kacau sekali sistemnya rusak dan merugikan banyak pihak",
    "Gak guna videonya buang waktu dan gak ada solusi nyata",
    "Payah banget analisisnya lemah dan sangat memihak tanpa bukti",
    "Gagal total rencananya malah memperburuk masalah ekonomi sekarang",
    "Menyesal nonton isinya cuma omong kosong dan tipu muslihat",
    "Buruk sekali pelayanannya lambat dan bikin emosi penonton",
    "Zonk banget hasilnya tidak sesuai janji dan sangat mengecewakan",
    "Hoax besar tidak ada fakta yg mendukung pernyataan tersebut",
    "Rusak parah kepercayaannya karena sikap yg tidak jujur",
)

_SLANG_PREFIXES: tuple[str, ...] = (
    "",
    "Yg bener nih ",
    "Klo menurutku ",
    "Emang ",
    "@channel ",
    "https://youtu.be/demo ",
)

_SLANG_SUFFIXES: tuple[str, ...] = (
    "",
    " bgt #indonesia",
    " 2026",
    " deh!",
    " sih menurut gw",
    " trs lanjutkan",
)


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction payload for a single YouTube comment."""

    label: str
    confidence: float
    cleaned_text: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Evaluation summary returned after stratified train/test modeling."""

    accuracy: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    labels: list[str]
    report: dict[str, Any]
    train_size: int
    test_size: int
    best_params: dict[str, Any] | None = None


def generate_sample_comments(
    n_per_class: int = 30,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Generate a deterministic offline dataset of labeled Indonesian YouTube comments.

    Args:
        n_per_class: Number of comments to generate per sentiment class (``>= 2``).
        seed: Random seed for reproducible shuffling.

    Returns:
        Tuple of ``(comments, labels)`` where labels belong to ``("positif", "negatif")``.

    Raises:
        ValueError: If ``n_per_class`` is less than 2.
    """
    if n_per_class < 2:
        raise ValueError("n_per_class must be at least 2.")

    samples: list[tuple[str, str]] = []
    for idx in range(n_per_class):
        prefix = _SLANG_PREFIXES[idx % len(_SLANG_PREFIXES)]
        suffix = _SLANG_SUFFIXES[(idx // len(_SLANG_PREFIXES)) % len(_SLANG_SUFFIXES)]
        pos_base = _POSITIVE_TEMPLATES[idx % len(_POSITIVE_TEMPLATES)]
        neg_base = _NEGATIVE_TEMPLATES[idx % len(_NEGATIVE_TEMPLATES)]
        samples.append((f"{prefix}{pos_base}{suffix}".strip(), "positif"))
        samples.append((f"{prefix}{neg_base}{suffix}".strip(), "negatif"))

    rng = random.Random(seed)
    rng.shuffle(samples)
    texts = [item[0] for item in samples]
    labels = [item[1] for item in samples]
    logger.info("Generated %d synthetic offline comments (%d per class).", len(texts), n_per_class)
    return texts, labels


def build_svm_pipeline(
    max_features: int = DEFAULT_MAX_FEATURES,
    c_value: float = 1.0,
    kernel: str = "linear",
) -> Pipeline:
    """Build a unified TF-IDF bigram vectorizer and balanced SVM pipeline.

    Args:
        max_features: Maximum vocabulary size for TF-IDF.
        c_value: Regularization strength ``C`` for ``SVC``.
        kernel: Kernel function for ``SVC`` (``linear``, ``rbf``, etc.).

    Returns:
        Unfitted ``sklearn.pipeline.Pipeline`` combining ``tfidf`` and ``svm``.

    Raises:
        ValueError: If ``max_features`` or ``c_value`` is non-positive.
    """
    if max_features <= 0:
        raise ValueError("max_features must be strictly positive.")
    if c_value <= 0:
        raise ValueError("c_value must be strictly positive.")

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)),
            (
                "svm",
                SVC(
                    C=c_value,
                    kernel=kernel,
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )


class SentimentClassifier:
    """End-to-end Indonesian sentiment classifier bundling preprocessing and SVM pipeline."""

    def __init__(
        self,
        max_features: int = DEFAULT_MAX_FEATURES,
        c_value: float = 1.0,
        kernel: str = "linear",
        slang_dict: Mapping[str, str] | None = None,
    ) -> None:
        self.slang_dict = dict(slang_dict) if slang_dict is not None else load_slang_dictionary()
        self.pipeline: Pipeline = build_svm_pipeline(
            max_features=max_features,
            c_value=c_value,
            kernel=kernel,
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether the underlying scikit-learn pipeline has been fitted."""
        try:
            check_is_fitted(self.pipeline)
        except NotFittedError:
            return False
        return True

    def _prepare_dataset(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        """Validate, normalize, and preprocess labeled training samples."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have identical length.")
        if not texts:
            raise ValueError("Training dataset cannot be empty.")

        cleaned_inputs = preprocess_batch(texts, slang_dict=self.slang_dict)
        filtered_texts: list[str] = []
        filtered_labels: list[str] = []

        for clean_text, raw_label in zip(cleaned_inputs, labels, strict=True):
            normalized_label = str(raw_label).strip().lower()
            if clean_text and normalized_label in SENTIMENT_LABELS:
                filtered_texts.append(clean_text)
                filtered_labels.append(normalized_label)

        if len(filtered_texts) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_TRAINING_SAMPLES} valid samples after cleaning; "
                f"got {len(filtered_texts)}."
            )

        counts = Counter(filtered_labels)
        if len(counts) < 2 or min(counts.values()) < 2:
            raise ValueError("Each sentiment class requires at least 2 valid samples.")
        return filtered_texts, filtered_labels

    def _fit_estimator(
        self,
        x_train: list[str],
        y_train: list[str],
        tune_hyperparams: bool,
    ) -> dict[str, Any] | None:
        """Fit pipeline directly or tune SVM hyperparameters via stratified CV."""
        if not tune_hyperparams:
            self.pipeline.fit(x_train, y_train)
            return None

        grid = GridSearchCV(
            estimator=self.pipeline,
            param_grid={
                "svm__C": [0.5, 1.0, 5.0],
                "svm__kernel": ["linear", "rbf"],
            },
            cv=3,
            scoring="f1_macro",
            n_jobs=1,
        )
        grid.fit(x_train, y_train)
        self.pipeline = grid.best_estimator_
        return dict(grid.best_params_)

    def _evaluate_split(
        self,
        x_test: list[str],
        y_test: list[str],
        train_size: int,
        best_params: dict[str, Any] | None,
    ) -> EvaluationMetrics:
        """Compute accuracy, F1-macro, confusion matrix, and classification report."""
        y_pred = self.pipeline.predict(x_test)
        acc = round(float(accuracy_score(y_test, y_pred)), 4)
        f1 = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
        matrix = confusion_matrix(y_test, y_pred, labels=list(SENTIMENT_LABELS)).tolist()
        report = classification_report(
            y_test,
            y_pred,
            labels=list(SENTIMENT_LABELS),
            output_dict=True,
            zero_division=0,
        )
        logger.info(
            "Trained SVM pipeline on %d samples (test=%d): accuracy=%.4f, f1_macro=%.4f",
            train_size,
            len(x_test),
            acc,
            f1,
        )
        return EvaluationMetrics(
            accuracy=acc,
            f1_macro=f1,
            confusion_matrix=matrix,
            labels=list(SENTIMENT_LABELS),
            report=report,
            train_size=train_size,
            test_size=len(x_test),
            best_params=best_params,
        )

    def train(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        test_size: float = 0.2,
        random_state: int = 42,
        tune_hyperparams: bool = False,
    ) -> EvaluationMetrics:
        """Train the unified TF-IDF + SVM pipeline using a stratified train/test split.

        Args:
            texts: Raw comment strings.
            labels: Corresponding labels (``"positif"`` or ``"negatif"``).
            test_size: Fraction of samples reserved for holdout evaluation.
            random_state: Deterministic seed for splitting.
            tune_hyperparams: Whether to run ``GridSearchCV`` across SVM hyperparameters.

        Returns:
            ``EvaluationMetrics`` measured strictly on the unseen test split.
        """
        if not 0.05 <= test_size <= 0.5:
            raise ValueError("test_size must be between 0.05 and 0.5.")

        clean_texts, clean_labels = self._prepare_dataset(texts, labels)
        x_train, x_test, y_train, y_test = train_test_split(
            clean_texts,
            clean_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=clean_labels,
        )
        best_params = self._fit_estimator(x_train, y_train, tune_hyperparams=tune_hyperparams)
        return self._evaluate_split(
            x_test, y_test, train_size=len(x_train), best_params=best_params
        )

    def _build_prediction(self, cleaned_text: str, prob_row: Any) -> PredictionResult:
        """Convert probability vector into a structured ``PredictionResult``."""
        classes = [str(label) for label in self.pipeline.classes_]
        probabilities = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(classes, prob_row, strict=True)
        }
        best_label = max(probabilities, key=probabilities.__getitem__)
        return PredictionResult(
            label=best_label,
            confidence=probabilities[best_label],
            cleaned_text=cleaned_text,
            probabilities=probabilities,
        )

    def predict_one(self, text: str) -> PredictionResult:
        """Predict sentiment label and calibrated probability for one comment.

        Args:
            text: Non-empty YouTube comment string.

        Returns:
            ``PredictionResult`` with predicted sentiment, confidence, and cleaned text.
        """
        if not self.is_fitted:
            raise RuntimeError("SentimentClassifier must be trained or loaded before prediction.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input comment text must be a non-empty string.")

        cleaned = preprocess_text(text, slang_dict=self.slang_dict)
        if not cleaned:
            logger.warning("Preprocessed comment had zero tokens; scoring empty feature vector.")
        proba_matrix = self.pipeline.predict_proba([cleaned])
        return self._build_prediction(cleaned, proba_matrix[0])

    def predict_batch(self, texts: Sequence[str]) -> list[PredictionResult]:
        """Predict sentiment for a batch of YouTube comments in O(N) vectorized fashion.

        Args:
            texts: Sequence of non-empty comment strings.

        Returns:
            List of ``PredictionResult`` items aligned with ``texts``.
        """
        if not self.is_fitted:
            raise RuntimeError("SentimentClassifier must be trained or loaded before prediction.")
        if not texts:
            raise ValueError("Batch input comments cannot be empty.")

        validated: list[str] = []
        for idx, item in enumerate(texts):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Comment at index {idx} must be a non-empty string.")
            validated.append(item)

        cleaned_list = preprocess_batch(validated, slang_dict=self.slang_dict)
        proba_matrix = self.pipeline.predict_proba(cleaned_list)
        return [
            self._build_prediction(cleaned, prob_row)
            for cleaned, prob_row in zip(cleaned_list, proba_matrix, strict=True)
        ]

    def save(self, path: Path | str) -> Path:
        """Serialize the unified TF-IDF + SVM pipeline to disk via joblib.

        Args:
            path: Destination file path.

        Returns:
            Resolved destination ``Path``.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted SentimentClassifier.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pipeline": self.pipeline, "version": "1.0.0"}
        joblib.dump(payload, target)
        logger.info("Saved unified sentiment pipeline to %s", target)
        return target

    @classmethod
    def load(
        cls,
        path: Path | str,
        slang_dict: Mapping[str, str] | None = None,
    ) -> SentimentClassifier:
        """Load a serialized unified sentiment pipeline artifact from disk.

        Args:
            path: Path to serialized joblib artifact.
            slang_dict: Optional custom slang dictionary.

        Returns:
            Initialized and fitted ``SentimentClassifier``.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Model artifact not found: {source}")

        payload = joblib.load(source)
        if not isinstance(payload, dict) or "pipeline" not in payload:
            raise ValueError(f"Invalid sentiment model bundle format in {source}")

        instance = cls(slang_dict=slang_dict)
        instance.pipeline = payload["pipeline"]
        check_is_fitted(instance.pipeline)
        logger.info("Loaded serialized sentiment pipeline from %s", source)
        return instance


_DEFAULT_CLASSIFIER_CACHE: dict[str, SentimentClassifier] = {}


def get_or_train_default_classifier(
    model_path: Path | None = None,
) -> SentimentClassifier:
    """Return a cached classifier loaded from disk or trained on offline sample data.

    Args:
        model_path: Optional path to a pre-trained joblib bundle.

    Returns:
        Ready-to-serve ``SentimentClassifier``.
    """
    cache_key = str(model_path.resolve()) if model_path is not None else "__offline_default__"
    if cache_key in _DEFAULT_CLASSIFIER_CACHE:
        return _DEFAULT_CLASSIFIER_CACHE[cache_key]

    if model_path is not None and model_path.exists():
        classifier = SentimentClassifier.load(model_path)
    else:
        texts, labels = generate_sample_comments(n_per_class=30, seed=42)
        classifier = SentimentClassifier()
        classifier.train(texts, labels, test_size=0.2, random_state=42)

    _DEFAULT_CLASSIFIER_CACHE[cache_key] = classifier
    return classifier


def clear_classifier_cache() -> None:
    """Clear the cached default classifier (primarily for test isolation)."""
    _DEFAULT_CLASSIFIER_CACHE.clear()
