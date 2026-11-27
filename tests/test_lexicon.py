"""Unit tests for Indonesian slang lexicon loading and runtime configuration."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from youtube_sentiment.config import Settings, clear_settings_cache, get_settings
from youtube_sentiment.lexicon import (
    CORE_SLANG_MAP,
    load_slang_dictionary,
    normalize_slang,
)


def test_core_slang_map_contains_required_indonesian_terms() -> None:
    """Verify the built-in offline slang dictionary has >= 40 essential domain terms."""
    required_terms = {
        "yg",
        "gak",
        "ga",
        "nggak",
        "gk",
        "bgt",
        "klo",
        "kalo",
        "tdk",
        "bikin",
        "mantap",
        "parah",
        "jelek",
        "keren",
        "bagus",
        "bohong",
        "rusak",
        "mantul",
    }
    assert len(CORE_SLANG_MAP) >= 40
    assert required_terms.issubset(CORE_SLANG_MAP.keys())


def test_load_slang_dictionary_offline_default_never_calls_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline default must return built-in lexicon without performing HTTP calls."""

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Network request should not occur when allow_remote=False")

    monkeypatch.setattr(requests, "get", _fail_if_called)
    slang = load_slang_dictionary(url="https://example.invalid/slang.csv", allow_remote=False)

    assert isinstance(slang, dict)
    assert len(slang) >= 40
    assert slang["yg"] == "yang"
    assert slang["gak"] == "tidak"


def test_load_slang_dictionary_remote_failure_returns_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote HTTP timeout or failure must return fallback dictionary, never None."""

    def _raise_timeout(*_args: object, **_kwargs: object) -> None:
        raise requests.Timeout("simulated timeout")

    monkeypatch.setattr(requests, "get", _raise_timeout)
    slang = load_slang_dictionary(
        url="https://example.invalid/slang.csv",
        allow_remote=True,
        timeout=2.5,
    )

    assert slang is not None
    assert isinstance(slang, dict)
    assert slang["yg"] == "yang"


def test_load_slang_dictionary_remote_success_merges_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid remote CSV payload merges cleanly with built-in fallback entries."""
    mock_response = Mock()
    mock_response.content = b"slang,formal\nokeey,baik\nmantaps,mantap\n"
    mock_response.raise_for_status = Mock()

    captured_kwargs: dict[str, object] = {}

    def _fake_get(url: str, **kwargs: object) -> Mock:
        captured_kwargs["url"] = url
        captured_kwargs.update(kwargs)
        return mock_response

    monkeypatch.setattr(requests, "get", _fake_get)
    slang = load_slang_dictionary(
        url="https://example.invalid/slang.csv",
        allow_remote=True,
        timeout=4.0,
    )

    assert captured_kwargs["timeout"] == 4.0
    assert slang["okeey"] == "baik"
    assert slang["mantaps"] == "mantap"
    assert slang["yg"] == "yang"


def test_load_slang_dictionary_invalid_csv_columns_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed CSV schema triggers safe fallback instead of crashing."""
    mock_response = Mock()
    mock_response.content = b"col_a,col_b\nfoo,bar\n"
    mock_response.raise_for_status = Mock()
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: mock_response)

    slang = load_slang_dictionary(url="https://example.invalid/bad.csv", allow_remote=True)
    assert slang == CORE_SLANG_MAP


def test_load_slang_dictionary_rejects_non_positive_timeout() -> None:
    """Non-positive timeout must fail upfront with ValueError."""
    with pytest.raises(ValueError, match="greater than zero"):
        load_slang_dictionary(timeout=0)


def test_normalize_slang_handles_multiword_and_empty_tokens() -> None:
    """Multi-word slang expansions split into individual clean tokens."""
    assert normalize_slang([]) == []
    tokens = ["konten", "yg", "mantul", "bgt"]
    assert normalize_slang(tokens) == ["konten", "yang", "mantap", "betul", "banget"]


def test_settings_validation_and_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings validates numeric bounds and parses env overrides cleanly."""
    clear_settings_cache()
    monkeypatch.setenv("APP_NAME", "custom-sentiment")
    monkeypatch.setenv("ALLOW_REMOTE_LEXICON", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "60")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173,https://app.example.com")

    loaded = get_settings()
    assert loaded.app_name == "custom-sentiment"
    assert loaded.allow_remote_lexicon is True
    assert loaded.rate_limit_per_minute == 60
    assert loaded.cors_origins == ("http://localhost:5173", "https://app.example.com")
    clear_settings_cache()

    with pytest.raises(ValueError, match="API_PORT"):
        Settings(api_port=70000)
