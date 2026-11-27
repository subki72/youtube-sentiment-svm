# Operational Runbook — `youtube-sentiment-svm`

## 1. Service Architecture & Runtime Dependencies

- **Runtime**: Python 3.11+, FastAPI + Uvicorn (`src/youtube_sentiment/api.py`), unified CLI (`main.py`).
- **ML Pipeline**: `TfidfVectorizer(ngram_range=(1, 2), max_features=2000)` + `SVC(kernel="linear", class_weight="balanced", probability=True)`.
- **Offline Lexicon Guarantee**: The service ships with a built-in Indonesian slang dictionary (`CORE_SLANG_MAP`) and defaults to `ALLOW_REMOTE_LEXICON=false` so inference works cleanly in air-gapped containers.

## 2. Health Verification & Smoke Checks

### Local CLI Readiness
```bash
python main.py --health
python main.py --demo
```

### HTTP Service Probe
```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Videonya keren bgt mantul!"}'
```

Expected `/health` payload (`HTTP 200`):
```json
{
  "status": "ok",
  "service": "youtube-sentiment-svm",
  "version": "1.0.0",
  "model_loaded": true
}
```

## 3. Incident Playbook

| Symptom | Likely Cause | Diagnostic Command | Mitigation |
|---|---|---|---|
| `HTTP 429 Too Many Requests` | Client exceeded `RATE_LIMIT_PER_MINUTE` | Check warning logs for client IP | Throttle caller or raise `RATE_LIMIT_PER_MINUTE` in `.env` |
| `HTTP 422 Unprocessable Entity` | Blank/whitespace comment or >200 batch items | Inspect response `detail` field | Validate caller text payload before POST |
| `HTTP 500 Internal Server Error` | Corrupt serialized `.joblib` bundle at `MODEL_PATH` | Check container logs (`logger.error`) | Remove/re-train bundle via `python main.py --demo --save-model artifacts/sentiment_pipeline.joblib` |
| Remote slang fetch warning on startup | GitHub raw lexicon timeout/offline network | `grep "Remote slang fetch failed" app.log` | No downtime: service automatically uses built-in offline `CORE_SLANG_MAP`; keep `ALLOW_REMOTE_LEXICON=false` in production |

## 4. Rollback & Recovery

1. Re-pin or roll back container image tag (`docker pull ...:v1.0.0`).
2. If using a custom serialized model artifact at `MODEL_PATH`, restore previous `.joblib` backup or unset `MODEL_PATH` to fall back to the deterministic baseline model.
3. Confirm `GET /health` returns `"model_loaded": true`.
