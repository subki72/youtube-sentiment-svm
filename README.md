# YouTube Comment Sentiment Analysis Using Support Vector Machine (SVM)

[![CI Pipeline](https://github.com/subki72/youtube-sentiment-svm/actions/workflows/ci.yml/badge.svg)](https://github.com/subki72/youtube-sentiment-svm/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

`youtube-sentiment-svm` provides an end-to-end Indonesian NLP and Machine Learning pipeline for classifying sentiment (`positif` vs `negatif`) in YouTube comments. It combines colloquial Indonesian slang normalization, Sastrawi Nazief-Adriani stemming, bigram TF-IDF feature extraction, and a class-weighted Support Vector Machine (`SVC(class_weight="balanced")`) delivered in three interoperable modes:

1. **Interactive Research Notebook** (`notebooks/Sentiment.ipynb`) for YouTube scraping, Excel dataset labeling prep, exploratory WordClouds, and model training.
2. **Modular Python Package & CLI** (`src/youtube_sentiment/` + `main.py`) for reproducible offline training, readiness probes, and single-comment classification.
3. **Production FastAPI Inference Service** (`src/youtube_sentiment/api.py`) with `/health`, `/predict`, `/analyze`, Pydantic v2 schema validation, sliding-window rate limiting, and non-root Docker packaging.

## Architecture & Directory Structure

```text
youtube-sentiment-svm/
├── main.py                              # Unified CLI (--health, --demo, --text, --serve)
├── notebooks/
│   └── Sentiment.ipynb                  # Patched end-to-end research & scraping notebook
├── src/
│   └── youtube_sentiment/
│       ├── __init__.py                  # Public package exports
│       ├── config.py                    # Env-backed Settings & structured leveled logging
│       ├── lexicon.py                   # Offline Indonesian slang map (71 terms) & safe remote loader
│       ├── preprocessing.py             # Regex noise cleanup, deduplicated stopwords, LRU stemmer
│       ├── model.py                     # Unified TF-IDF + SVM Pipeline & offline synthetic corpus
│       └── api.py                       # FastAPI service (/health, /predict, /analyze)
├── tests/                               # 100% offline unit & API/CLI integration tests
├── docs/
│   ├── adr/001-indonesian-nlp-svm-pipeline.md
│   └── runbook.md
├── Dockerfile                           # Multi-stage python:3.11-slim non-root container
├── docker-compose.yml                   # Local service orchestration
├── Makefile                             # 1-command install, lint, test, serve, docker targets
├── pyproject.toml                       # Ruff, Flake8, Pytest, Mypy, and optional extras config
├── requirements.txt                     # Pinned Python 3.11 Docker runtime dependencies
└── uv.lock                              # Cross-platform lockfile
```

## Key Technical Features & Hardening

- **Unified Pipeline Serialization**: Bundles `TfidfVectorizer(ngram_range=(1, 2), max_features=2000)` and `SVC(kernel="linear", class_weight="balanced", probability=True)` inside a single `sklearn.pipeline.Pipeline` saved via `joblib` to eliminate train/serve vocabulary skew.
- **Offline-First Slang Lexicon**: Ships a curated 71-term Indonesian colloquial dictionary (`yg`, `gak`, `bgt`, `klo`, `mantul`, `parah`, `hoax`, etc.) with `allow_remote=False` by default and explicit HTTP timeout (`5.0s`) + safe non-`None` fallback when remote lookup is enabled.
- **Patched Notebook Reliability**: Fixed bare `except:` and bare `raise` blocks, unbounded `requests.get()` timeouts, unstratified `train_test_split`, duplicate stopword entries, empty WordCloud crashes, and misleading confusion matrix titles in `notebooks/Sentiment.ipynb`.
- **API Resilience**: Pydantic v2 validation, per-client sliding-window `RateLimiter`, configurable CORS origins, structured leveled logging, and global exception handlers that conceal internal stack traces.

## Installation & Setup

### 1. Clone and Create Virtual Environment

```bash
git clone https://github.com/subki72/youtube-sentiment-svm.git
cd youtube-sentiment-svm
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
cp .env.example .env       # Optional: customize runtime configuration
```

For notebook scraping and visualization extras (`wordcloud`, `youtube-comment-downloader`, `matplotlib`, `seaborn`), install optional extras:

```bash
pip install -e ".[notebook]"
```

## Usage

### Unified CLI (`main.py`)

```bash
# 1. Verify offline lexicon, stemmer, and pipeline readiness
python main.py --health

# 2. Run 100% offline stratified training & evaluation demo on synthetic Indonesian comments
python main.py --demo

# 3. Classify a single Indonesian YouTube comment
python main.py --text "Videonya keren bgt mantul penjelasannya jelas!"

# 4. Start the FastAPI server on port 8000
python main.py --serve
```

### FastAPI Endpoints

Once running (`make serve` or `docker compose up --build`):

- `GET /health` — Service liveness and model readiness status.
- `POST /predict` — Classify a single YouTube comment (`{"text": "..."}`) returning `label`, `confidence`, `cleaned_text`, and `probabilities`.
- `POST /analyze` — Batch classify up to 200 comments (`{"comments": [...]}`) returning per-comment predictions and aggregate sentiment summary.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Kontennya bagus bgt dan sangat bermanfaat"}'
```

### Docker Execution

```bash
make docker-build
make docker-up
```

## Testing & Quality Verification

All unit and integration tests run completely offline without external network calls:

```bash
make lint
make test
```

Equivalent direct commands:

```bash
python -m ruff check src tests main.py
python -m ruff format --check src tests main.py
python -m flake8 --max-line-length=120 src tests main.py
python -m pytest tests/ -v --cov=src --cov-fail-under=75
```

## Documentation & Governance

- **Architecture Decision Record**: [docs/adr/001-indonesian-nlp-svm-pipeline.md](docs/adr/001-indonesian-nlp-svm-pipeline.md)
- **Operational Runbook**: [docs/runbook.md](docs/runbook.md)
- **Security Policy**: [SECURITY.md](SECURITY.md)
- **Contributing Guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **License**: [MIT License](LICENSE)
