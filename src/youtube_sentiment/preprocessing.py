"""Indonesian social-media text cleaning, stopword removal, and stemming."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from youtube_sentiment.lexicon import CUSTOM_STOPWORDS, load_slang_dictionary, normalize_slang

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_HASHTAG_PATTERN = re.compile(r"@\w+|#\w+")
DIGIT_PATTERN = re.compile(r"\d+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


@lru_cache(maxsize=1)
def get_stemmer():
    """Create and cache a Sastrawi Nazief-Adriani stemmer instance."""
    logger.info("Initializing cached Sastrawi stemmer.")
    return StemmerFactory().create_stemmer()


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    """Return deduplicated Indonesian stopwords combining Sastrawi and custom terms."""
    base_stopwords = StopWordRemoverFactory().get_stop_words()
    combined = frozenset(base_stopwords) | CUSTOM_STOPWORDS
    logger.info("Initialized %d deduplicated Indonesian stopwords.", len(combined))
    return combined


@lru_cache(maxsize=4096)
def _stem_token(token: str) -> str:
    """Stem a single Indonesian word with LRU memoization to avoid repeated Sastrawi scans."""
    return str(get_stemmer().stem(token))


def clean_raw_text(text: str) -> str:
    """Lowercase and strip URLs, mentions, hashtags, digits, and punctuation.

    Args:
        text: Raw comment text.

    Returns:
        Normalized lowercase ASCII alphabetic string with single spaces.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    lowered = text.lower()
    without_urls = URL_PATTERN.sub(" ", lowered)
    without_tags = MENTION_HASHTAG_PATTERN.sub(" ", without_urls)
    without_digits = DIGIT_PATTERN.sub(" ", without_tags)
    alpha_only = NON_ALPHA_PATTERN.sub(" ", without_digits)
    return WHITESPACE_PATTERN.sub(" ", alpha_only).strip()


def preprocess_text(
    text: str,
    slang_dict: Mapping[str, str] | None = None,
    apply_stemming: bool = True,
) -> str:
    """Run the full Indonesian NLP preprocessing pipeline on a single comment.

    Args:
        text: Raw YouTube comment string.
        slang_dict: Optional custom slang dictionary (defaults to offline lexicon).
        apply_stemming: Whether to apply Sastrawi Nazief-Adriani stemming.

    Returns:
        Cleaned, normalized, stopword-filtered, and optionally stemmed text.
    """
    cleaned = clean_raw_text(text)
    if not cleaned:
        return ""

    active_slang = slang_dict if slang_dict is not None else load_slang_dictionary()
    tokens = normalize_slang(cleaned.split(), slang_dict=active_slang)
    stopwords = get_stopwords()
    filtered = [token for token in tokens if token not in stopwords]

    if not filtered:
        return ""

    if not apply_stemming:
        return " ".join(filtered)
    return " ".join(_stem_token(token) for token in filtered)


def preprocess_batch(
    texts: Iterable[str],
    slang_dict: Mapping[str, str] | None = None,
    apply_stemming: bool = True,
) -> list[str]:
    """Preprocess an iterable of comments using shared cached NLP resources.

    Args:
        texts: Iterable of raw YouTube comment strings.
        slang_dict: Optional custom slang dictionary.
        apply_stemming: Whether to apply Sastrawi stemming.

    Returns:
        List of preprocessed strings matching the input order.
    """
    active_slang = slang_dict if slang_dict is not None else load_slang_dictionary()
    processed = [
        preprocess_text(item, slang_dict=active_slang, apply_stemming=apply_stemming)
        for item in texts
    ]
    logger.info("Preprocessed batch of %d comments.", len(processed))
    return processed
