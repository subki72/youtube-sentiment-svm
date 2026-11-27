"""End-to-end integration tests for CLI flags in main.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import main as cli_main
from youtube_sentiment.config import clear_settings_cache
from youtube_sentiment.model import clear_classifier_cache


def test_cli_health_check_outputs_valid_json(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI --health runs offline readiness checks and emits JSON payload."""
    clear_settings_cache()
    clear_classifier_cache()
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nonexistent.joblib"))

    exit_code = cli_main.main(["--health"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["lexicon_entries"] >= 40
    assert payload["model_fitted"] is True
    clear_settings_cache()
    clear_classifier_cache()


def test_cli_demo_trains_evaluates_and_persists_model(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """CLI --demo executes stratified evaluation and optionally saves joblib artifact."""
    model_file = tmp_path / "demo_pipeline.joblib"
    exit_code = cli_main.main(["--demo", "--save-model", str(model_file)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert model_file.exists()
    assert payload["accuracy"] >= 0.8
    assert payload["f1_macro"] >= 0.8
    assert len(payload["sample_predictions"]) == 2


def test_cli_text_predicts_single_comment_and_validates_blank(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI --text outputs prediction JSON and returns exit code 2 on blank text."""
    clear_settings_cache()
    clear_classifier_cache()
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nonexistent.joblib"))

    ok_code = cli_main.main(["--text", "Videonya sangat bagus dan bermanfaat!"])
    ok_payload = json.loads(capsys.readouterr().out)
    assert ok_code == 0
    assert ok_payload["label"] == "positif"

    err_code = cli_main.main(["--text", "   "])
    assert err_code == 2
    clear_settings_cache()
    clear_classifier_cache()
