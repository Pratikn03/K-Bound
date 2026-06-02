# ELARA Production Runbook

This runbook defines the bounded production release target for the ELARA/UAIS
API service. It covers operational deployment of the authenticated FastAPI
runtime only. Gate E remains a scientific gate, and a deployable API does not
convert bounded research evidence into strict transfer success.

## Environment

Required production variables:

| Variable | Requirement |
| --- | --- |
| `UAIS_API_KEYS` | Comma-separated API keys. Empty values fail closed. |
| `UAIS_CORS_ORIGINS` | Explicit HTTPS origins only. Wildcard origins are forbidden. |
| `UAIS_PRODUCTION_MODE` | Set to `true` for production startup validation. |
| `UAIS_REQUIRED_MODELS` | Core model set for readiness; default is `fraud,cyber,fusion`. |
| `UAIS_TRUSTED_MODEL_ARTIFACTS` | Set to `true` only after artifacts are checksum-pinned. |
| `UAIS_REQUIRE_MODEL_CHECKSUMS` | Keep `true` in production. |

Do not log API keys, bearer tokens, raw request payloads, image payloads, or
model inputs. The API emits request IDs, method, path, status, and duration only.

## Artifact Checksum Policy

Pickle/joblib and PyTorch artifacts are not trusted by default. A model is
production-loadable only when:

1. The artifact exists in the mounted model directory.
2. `UAIS_TRUSTED_MODEL_ARTIFACTS=true`.
3. The matching `UAIS_MODEL_SHA256_<MODEL>` value equals the artifact SHA-256.
4. PyTorch checkpoints load through the `weights_only=True` path.

Optional NLP, vision, and attention routes are mounted but fail closed with
`model_unavailable` until their dependencies, artifacts, and checksums are
productionized.

## Deployment

Default production compose starts the API service only:

```bash
UAIS_API_KEYS='replace-me' \
UAIS_CORS_ORIGINS='https://ops.example' \
docker compose up --build api
```

Streamlit and MLflow are research profile services. Start them only for local
analysis or internal research workflows:

```bash
docker compose --profile research up --build
```

The production container runs as a non-root user, uses a read-only filesystem,
mounts model/experiment artifacts read-only, and binds the published port to
localhost by default. Put it behind a TLS-terminating reverse proxy for remote
operations.

## Readiness

`/health` is unauthenticated liveness. It proves only that the API process can
answer a basic request.

`/ready` is authenticated and returns `503` until:

- production configuration is valid,
- required core models are loaded,
- monitoring initialized successfully.

Use `/ready` for load balancer readiness and deployment promotion.

## Monitoring

Authenticated operational endpoints:

- `/metrics`: Prometheus text metrics.
- `/system`: process and system resource snapshot.
- `/health/detailed`: component health checks.

Alert on repeated `503` readiness responses, elevated 5xx rates, sustained high
latency, missing model-loaded metrics, and rate-limit spikes.

## Rollback

Rollback is artifact and image based:

1. Stop the new API container.
2. Restore the previous image tag or compose digest.
3. Restore the previous model artifact directory and SHA-256 environment values.
4. Start the API and verify `/health`, `/ready`, and one authenticated core
   prediction path.
5. Record the rollback cause and checksum set.

## Incident Response

For suspected credential exposure, rotate `UAIS_API_KEYS` immediately and
restart the API. For suspected artifact tampering, set
`UAIS_TRUSTED_MODEL_ARTIFACTS=false`, restart the API, preserve the mounted
artifact directory for forensic review, and restore from the last known-good
checksum set.

For model-quality incidents, disable traffic at the proxy or remove the model
from `UAIS_REQUIRED_MODELS` only as an emergency operational mitigation. Do not
change research gates or claims as part of an incident response.

## Research Claim Boundary

Production readiness here means the system is secure, monitored, reproducible,
deployable, and fail-closed for unsupported models. It does not claim universal
SOTA, strict clean-transfer success, or final flagship scientific readiness.

Strict Gate E/F remain governed by the research lock and confirmatory statistics.
Bounded real-dataset evidence, Real-IAD D3 natural-degradation evidence, and
synthetic/proxy diagnostics must stay separated in papers, dashboards, PDFs, and
release notes.
