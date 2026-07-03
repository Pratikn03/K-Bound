# Production deployment scorecard

**Date:** 2026-06-30  
**Surface:** ELARA/UAIS scoped SaaS + KGA safety layer  
**Grade:** **9+/10** (Gate P target: all 15 criteria PASS)

---

## Two products, two grades

| Product | Grade | Notes |
|---------|:-----:|-------|
| **Scoped ELARA/UAIS API + KGA `/decide`** | **9+/10** | Auth, scope guard, model governance, load baseline, full cert mode |
| **Full K-Bound TTA SaaS (Tent/EATA + camera)** | **5–6/10** | Edge R2 pending; use research harness for TTA |

This scorecard covers **Surface A**.

---

## Gate P checklist (run: `python src/scripts/audit_gate_p_production.py`)

| ID | Criterion | Fix |
|----|-----------|-----|
| P5 | Distributed rate limiting | `deploy/api/rate_limit.py` + `UAIS_REDIS_URL` + compose `redis` service |
| P11 | Model versioning / rollback | `GET /models/versions`, `POST /models/rollback`, `models/MANIFEST.json` |
| P15 | Load test evidence | `deploy/loadtest/run_baseline.py`, `locustfile.py`, `BASELINE_RESULTS.json` |
| KGA | API tests | `tests/test_kga_api_routes.py` |
| KGA | Full certificate mode | `cert_mode=full` + `benefit_scores` on `POST /decide` |

---

## API modes (`POST /decide`)

| `cert_mode` | Inputs | Use case |
|-------------|--------|----------|
| `proxy` (default) | calib + test scores | Fast score-only deployment gate |
| `full` | + `benefit_scores` or `calib_residuals` | Paper-aligned Theorem 3 certificate |

---

## Verify locally

```bash
# Gate P audit
python src/scripts/audit_gate_p_production.py

# KGA + governance tests
pytest tests/test_kga_api_routes.py tests/test_model_governance.py -q

# Load baseline (P15)
python deploy/loadtest/run_baseline.py

# Multi-replica rate limit (optional)
docker compose --profile distributed up -d redis
UAIS_REDIS_URL=redis://127.0.0.1:6379/0 ...
```

---

## Honest scope (unchanged)

Production-ready **within validated envelope** (`deploy/SCOPE_CONTRACT.md`). Not universal TTA SaaS until edge R2 ships.
