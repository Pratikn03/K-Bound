# Production API Runbook

This service is fail-closed by default. A production deployment must provide API keys, an explicit CORS allowlist, and signed model-artifact trust configuration before prediction endpoints serve model outputs.

## Required Environment

```bash
export UAIS_API_KEYS="key-1,key-2"
export UAIS_CORS_ORIGINS="https://console.example.com,https://ops.example.com"
```

Optional bounds and rate limits:

```bash
export UAIS_RATE_LIMIT_REQUESTS=120
export UAIS_RATE_LIMIT_WINDOW_SECONDS=60
export UAIS_MAX_FEATURES=4096
export UAIS_MAX_SCORES=64
export UAIS_MAX_DOMAINS=32
export UAIS_MAX_EMBEDDINGS=8192
export UAIS_MAX_TEXT_CHARS=10000
export UAIS_MAX_IMAGE_BASE64_CHARS=8000000
export UAIS_MAX_IMAGE_PIXELS=25000000
```

## Model Artifact Trust

Model loading is disabled unless artifacts are explicitly trusted. Production should keep checksum enforcement enabled and set a SHA256 value for every artifact that is expected to load.

```bash
export UAIS_TRUSTED_MODEL_ARTIFACTS=true
export UAIS_REQUIRE_MODEL_CHECKSUMS=true
export UAIS_MODEL_SHA256_FRAUD="$(shasum -a 256 models/fraud/supervised/fraud_model.pkl | awk '{print $1}')"
export UAIS_MODEL_SHA256_CYBER="$(shasum -a 256 models/cyber/supervised/cyber_model.pkl | awk '{print $1}')"
export UAIS_MODEL_SHA256_FUSION="$(shasum -a 256 experiments/fusion/models/fusion_meta_model.pkl | awk '{print $1}')"
```

For PyTorch artifacts, use the corresponding variables:

```bash
export UAIS_MODEL_SHA256_NLP="..."
export UAIS_MODEL_SHA256_VISION="..."
export UAIS_MODEL_SHA256_ATTENTION_FUSION="..."
```

If a checksum is missing or mismatched, that model remains unloaded and its endpoint returns `503`.

## Local Compose Smoke

```bash
UAIS_API_KEYS=local-secret \
UAIS_CORS_ORIGINS=http://localhost:8501 \
docker compose config
```

Start the API only after `docker compose config` passes:

```bash
UAIS_API_KEYS=local-secret \
UAIS_CORS_ORIGINS=http://localhost:8501 \
docker compose up --build api
```

Basic health is public:

```bash
curl http://127.0.0.1:8000/health
```

Operational endpoints require `X-API-Key`:

```bash
curl -H "X-API-Key: local-secret" http://127.0.0.1:8000/health/detailed
curl -H "X-API-Key: local-secret" http://127.0.0.1:8000/metrics
curl -H "X-API-Key: local-secret" http://127.0.0.1:8000/system
```
