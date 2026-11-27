# ADR-001: Unified Indonesian NLP + Class-Weighted SVM Pipeline & Dual-Mode Service

- **Status**: Accepted
- **Date**: 2026-11-27
- **Deciders**: Repository Engineering Team

## Context

The initial repository contained a single exploratory Jupyter Notebook (`notebooks/Sentiment.ipynb`) with several reliability and deployment risks:
1. Unbundled serialization (`svm_sentiment_model.pkl` and `tfidf_vectorizer.pkl` exported separately), risking train/serve vocabulary skew.
2. Unbounded network dependency (`requests.get(url)` without timeout and returning `None` on failure) in `load_slang_dictionary()`.
3. Unstratified train/test splitting, bare `except:` / bare `raise` blocks, and unguarded empty WordCloud plots.
4. No modular Python package, automated tests, CLI runner, or HTTP inference API.

## Decision

1. **Dual-Mode Non-Breaking Architecture**: Keep and repair `notebooks/Sentiment.ipynb` in-place while extracting reusable production modules under `src/youtube_sentiment/` (`config.py`, `lexicon.py`, `preprocessing.py`, `model.py`, `api.py`) and exposing a unified CLI (`main.py`).
2. **Single `Pipeline` Serialization**: Bundle `TfidfVectorizer(ngram_range=(1, 2), max_features=2000)` and `SVC(class_weight="balanced", probability=True)` inside one `sklearn.pipeline.Pipeline` serialized atomically via `joblib`.
3. **Offline-First Lexicon & Memoized Sastrawi Stemming**: Bundle `CORE_SLANG_MAP` (`71` Indonesian social-media terms) with `allow_remote=False` default and explicit timeouts (`5.0s`) when enabled, paired with LRU-cached Sastrawi token stemming (`_stem_token`).
4. **Hardened FastAPI Service**: Expose `/health`, `/predict`, and `/analyze` with Pydantic v2 schema validation, sliding-window per-client rate limiting, CORS controls, and sanitized global exception handling.

## Consequences

- **Positive**: Zero train/serve skew, 100% deterministic offline test execution (`>= 75%` coverage gate), non-root container deployment, and backward-compatible research notebook execution.
- **Trade-offs**: In-memory rate limiting is single-process scoped; multi-replica horizontal scaling should front the API with a shared Redis limiter or API gateway.
