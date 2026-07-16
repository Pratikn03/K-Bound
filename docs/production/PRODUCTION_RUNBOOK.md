# KGA API production runbook

## Environment

Set `KGA_API_KEYS` and `KGA_CORS_ORIGINS` before starting the API (legacy
`UAIS_API_KEYS` / `UAIS_CORS_ORIGINS` are still accepted by `deploy/api`).

Optional: `KGA_PRODUCTION_MODE=true`, `KGA_REDIS_URL`, `KGA_SCOPE_REFERENCE`.

```bash
export KGA_API_KEYS=...
export KGA_CORS_ORIGINS=https://ops.example
docker compose up --build api
```

## Artifact Checksum Policy

Prefer checksum-pinned model artifacts when `KGA_REQUIRE_MODEL_CHECKSUMS=true`.

## Deployment

```bash
docker compose up --build -d api
curl -s http://127.0.0.1:8000/health
```

## Readiness

`GET /ready` must return 200 before sending traffic to `/decide`.

## Monitoring

Watch `kga_scope_drift` and `kga_out_of_envelope_total` (Prometheus). Alert on
sustained out-of-envelope rates.

## Rollback

Use `POST /models/rollback` with a prior manifest version, then restart the API.

## Incident Response

Fail closed: revoke keys, set production mode, freeze adaptation by serving
freeze/abstain only until envelope restored.

## Research Claim Boundary

Gate E remains a scientific gate. Deployment readiness is not a claim of natural
beats-both or camera publication results.
