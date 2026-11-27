"""Unit tests for Indonesian text cleaning, stopword removal, and stemming."""

from __future__ import annotations

from youtube_sentiment.preprocessing import (
    clean_raw_text,
    get_stopwords,
    preprocess_batch,
    preprocess_text,
)


def test_clean_raw_text_strips_urls_mentions_hashtags_and_digits() -> None:
    """Noise tokens like URLs, mentions, hashtags, digits, and symbols are stripped."""
    raw = "Cek https://youtu.be/abc123 @channel #politik2026 bagus banget!! 99%"
    cleaned = clean_raw_text(raw)

    assert cleaned == "cek bagus banget"


def test_clean_raw_text_handles_empty_and_non_string_inputs() -> None:
    """Blank or non-string inputs return empty strings safely."""
    assert clean_raw_text("") == ""
    assert clean_raw_text("   \n\t  ") == ""
    assert clean_raw_text(None) == ""  # type: ignore[arg-type]


def test_get_stopwords_is_deduplicated_and_non_empty() -> None:
    """Stopword set merges Sastrawi and custom stopwords without duplicates."""
    stopwords = get_stopwords()

    assert isinstance(stopwords, frozenset)
    assert len(stopwords) > 100
    assert {"dan", "yang", "di", "ke", "aja", "kalo"}.issubset(stopwords)


def test_preprocess_text_normalizes_slang_and_stems_indonesian_words() -> None:
    """Full pipeline normalizes slang, filters stopwords, and stems words."""
    raw = "Videonya yg keren bgt membanggakan & mantul #indonesia https://example.com"
    processed = preprocess_text(raw)

    assert "bangga" in processed
    assert "keren" in processed
    assert "mantap" in processed
    assert "yg" not in processed.split()
    assert "https" not in processed


def test_preprocess_text_without_stemming_preserves_uninflected_tokens() -> None:
    """Disabling stemming returns stopword-filtered normalized tokens directly."""
    raw = "Klo menurutku penjelasannya bagus bgt"
    processed = preprocess_text(raw, apply_stemming=False)

    assert processed == "menurutku penjelasannya bagus banget"


def test_preprocess_text_returns_empty_when_only_noise_or_stopwords() -> None:
    """Comments made exclusively of URLs/digits/stopwords become empty string."""
    assert preprocess_text("12345 @user #tag https://youtu.be/demo") == ""
    assert preprocess_text("dan atau di ke yang") == ""


def test_preprocess_batch_preserves_order_and_length() -> None:
    """Batch preprocessing matches input cardinality and handles blank rows."""
    comments = [
        "Kontennya bagus bgt!",
        "12345",
        "Informasinya bohong parah",
    ]
    results = preprocess_batch(comments)

    assert len(results) == 3
    assert "bagus" in results[0]
    assert results[1] == ""
    assert "bohong" in results[2]
