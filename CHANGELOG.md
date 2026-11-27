# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-27

### Added
- Modular Python package `src/youtube_sentiment/` (`config.py`, `lexicon.py`, `preprocessing.py`, `model.py`, `api.py`).
- Curated offline Indonesian colloquial slang lexicon (`CORE_SLANG_MAP`, 71 entries) with safe timeout-bounded remote CSV override.
- Unified `sklearn.pipeline.Pipeline` bundling `TfidfVectorizer(ngram_range=(1, 2), max_features=2000)` and `SVC(class_weight="balanced", probability=True)`.
- Deterministic offline Indonesian synthetic comment generator (`generate_sample_comments`) for reproducible evaluation and testing.
- FastAPI inference service exposing `GET /health`, `POST /predict`, and `POST /analyze` with Pydantic v2 validation, sliding-window rate limiting, CORSMiddleware, and global exception handling.
- Unified CLI entrypoint `main.py` supporting `--health`, `--demo`, `--text`, and `--serve`.
- Comprehensive offline unit and integration test suite under `tests/` (`>= 75%` coverage gate).
- Production operational packaging and automation: multi-stage non-root `Dockerfile`, `docker-compose.yml`, `Makefile`, `pyproject.toml`, `uv.lock`, `.env.example`, `.pre-commit-config.yaml`, `docs/runbook.md`, `docs/adr/001-indonesian-nlp-svm-pipeline.md`, and GitHub Actions CI/Release workflows.

### Fixed
- Patched `notebooks/Sentiment.ipynb` Cell 4 bare `except:` on interactive `input()` to handle `(EOFError, KeyboardInterrupt)`.
- Patched `notebooks/Sentiment.ipynb` Cell 8 `load_slang_dictionary()` to enforce `timeout=10`, `raise_for_status()`, non-`None` fallback dictionary, and deduplicated Sastrawi/custom stopwords.
- Patched `notebooks/Sentiment.ipynb` Cell 10 bare `except:` and bare `raise` statements with explicit `(FileNotFoundError, ValueError, OSError)` error handling.
- Patched `notebooks/Sentiment.ipynb` Cell 12 `train_test_split` to include `stratify=y`.
- Corrected `notebooks/Sentiment.ipynb` Cell 16 confusion matrix plot title from `SMOTE` to `Class-Weighted SVM & Bigram TF-IDF`.
- Guarded `notebooks/Sentiment.ipynb` Cell 18 `plot_wordcloud` against empty text inputs.
