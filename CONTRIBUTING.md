# Contributing Guide

Thank you for contributing to `youtube-sentiment-svm`.

## Local Development Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ruff flake8 pre-commit
pre-commit install
```

## Coding Standards

- Follow Python 3.11+ type annotations (`list[str]`, `str | None`) and Google-style docstrings.
- Use `logging.getLogger(__name__)` rather than bare `print()` inside `src/youtube_sentiment/`.
- Keep functions focused (`< 50` lines) with shallow nesting (`<= 3` levels) and explicit error handling.
- Ensure all tests run 100% offline with zero live network or YouTube scraping dependencies.

## Verification Before Opening a Pull Request

```bash
python -m ruff check src tests main.py
python -m ruff format --check src tests main.py
python -m flake8 --max-line-length=120 src tests main.py
python -m pytest tests/ -v --cov=src --cov-fail-under=75
```

## Commit Conventions

Use descriptive conventional commit messages (`feat(nlp): ...`, `fix(notebook): ...`, `test(api): ...`, `docs(readme): ...`).
