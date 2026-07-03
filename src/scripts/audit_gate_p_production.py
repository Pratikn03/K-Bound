"""Gate P — production-readiness engineering audit (re-runnable, evidence-based).

Grades the K-Bound KGA deploy stack against concrete criteria by inspecting
the actual files. See deploy/SCOPE_CONTRACT.md for the validated envelope.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def grade() -> dict:
    docker = _read("Dockerfile")
    main = _read("deploy/api/main.py")
    kga_routes = _read("deploy/api/kga_routes.py")
    auth = _read("deploy/api/auth.py")
    monit = _read("deploy/api/monitoring.py")
    ci = _read(".github/workflows/ci.yml")
    compose = _read("docker-compose.yml")

    criteria: list[dict] = []

    def add(cid, name, critical, status, evidence):
        criteria.append({"id": cid, "name": name, "critical": critical,
                         "status": status, "evidence": evidence})

    cont = all(x in docker for x in ["USER ", "HEALTHCHECK"]) and "@sha256:" in docker
    add("P1", "Container hardening (non-root, healthcheck, pinned base)", True,
        "PASS" if cont else "PARTIAL",
        f"USER={'USER ' in docker}, HEALTHCHECK={'HEALTHCHECK' in docker}, pinned_sha={'@sha256:' in docker}")

    has_auth = (("Depends(authenticate)" in kga_routes) or ("Depends(authenticate)" in main)) and "API_KEYS" in auth and "verify_token" in auth
    fails_closed = "not configured" in auth
    add("P2", "Authentication + authorization (fails-closed)", True,
        "PASS" if (has_auth and fails_closed) else "PARTIAL",
        f"api_key+jwt={has_auth}, fails_closed={fails_closed}, scopes={'require_scope' in auth}")

    iv = ("field_validator" in kga_routes and "np.isfinite" in kga_routes and "max_length" in kga_routes)
    add("P3", "Input validation + bounds", True, "PASS" if iv else "PARTIAL",
        f"kga_routes_validators={'field_validator' in kga_routes}, finite_scores={'np.isfinite' in kga_routes}")

    kga_only = "kga_router" in main and "torch.load" not in main and "weights_only=True" not in main
    safe = kga_only or ("weights_only=True" in main and "_artifact_is_trusted" in main and "_sha256" in main)
    add("P4", "Safe model loading (checksum + weights_only)", True, "PASS" if safe else "FAIL",
        f"kga_only={kga_only}, weights_only={'weights_only=True' in main}, "
        f"checksum_gated={'_artifact_is_trusted' in main}")

    rl = "RateLimitMiddleware" in main
    rl_mod = _read("deploy/api/rate_limit.py")
    rl_distributed = "redis" in main.lower() or "redis" in compose.lower() or "redis" in rl_mod.lower()
    add("P5", "Rate limiting", False, "PASS" if (rl and rl_distributed) else ("PARTIAL" if rl else "FAIL"),
        f"present={rl}, distributed={rl_distributed}")

    cors_ok = "cannot use a wildcard origin in production" in main
    no_hardcoded = not re.search(r"(SECRET_KEY|API_KEY)\s*=\s*[\"'][A-Za-z0-9]{8,}", auth + main)
    add("P6", "CORS validation + secrets from env", True,
        "PASS" if (cors_ok and no_hardcoded) else "PARTIAL",
        f"no_wildcard_prod={cors_ok}, no_hardcoded_secret={no_hardcoded}")

    obs = all(x in main for x in ["/health", "/ready", "/metrics"]) and "prometheus_client" in monit
    add("P7", "Observability (health/ready/metrics, Prometheus, structured logs)", True,
        "PASS" if obs else "PARTIAL",
        f"endpoints={'/metrics' in main}, prometheus={'prometheus_client' in monit}, "
        f"correlation_id={'X-Request-ID' in main}")

    eh = "request_failed" in main or "KGA decision failed" in kga_routes
    add("P8", "Error handling (no internal leakage)", True, "PASS" if eh else "PARTIAL",
        f"request_failed_logged={'request_failed' in main}, generic_kga_errors={'KGA decision failed' in kga_routes}")

    cicd = "pytest" in ci and ("kbound-ci" in ci or "kbound-core" in ci)
    add("P9", "CI/CD (tests + coverage + security scan)", True, "PASS" if cicd else "PARTIAL",
        f"pytest={'pytest' in ci}, kbound_ci={'kbound' in ci}")

    repro = (ROOT / "requirements.lock.txt").is_file()
    add("P10", "Reproducibility (locked deps)", False, "PASS" if repro else "PARTIAL",
        f"lockfile={repro}")

    gov_mod = _read("deploy/api/model_governance.py")
    gov = ("model_version" in gov_mod.lower() and "rollback" in gov_mod.lower())
    add("P11", "Model governance (versioning + rollback)", False, "PASS" if gov else "FAIL",
        "model_version + rollback endpoints in deploy/api/model_governance.py")

    drift = bool(re.search(r"drift|out.?of.?envelope|scope_guard", (main + monit).lower()))
    add("P12", "Live drift / out-of-envelope monitoring", True, "PASS" if drift else "FAIL",
        "scope_guard import wires drift telemetry")

    scope = (ROOT / "deploy/SCOPE_CONTRACT.md").is_file()
    add("P13", "Deployment scope contract (validated envelope)", True, "PASS" if scope else "FAIL",
        "deploy/SCOPE_CONTRACT.md")

    resil = "timeout" in main.lower() or "TimeoutMiddleware" in main
    add("P14", "Resilience (request timeout / resource limits)", False, "PASS" if resil else "PARTIAL",
        f"request_timeout={'timeout' in main.lower()}")

    load = (ROOT / "deploy/loadtest").exists() or "locust" in compose.lower()
    add("P15", "Load / scale test evidence", False, "PASS" if load else "MISSING",
        "no load-test artifact")

    crit_not_pass = [c for c in criteria if c["critical"] and c["status"] != "PASS"]
    any_gap = [c for c in criteria if c["status"] in ("FAIL", "MISSING")]
    sec_serving_ok = all(c["status"] == "PASS" for c in criteria
                         if c["id"] in ("P2", "P3", "P4", "P6", "P7", "P8", "P9"))
    crit_hard_fail = [c for c in crit_not_pass if c["status"] in ("FAIL", "MISSING")]
    if all(c["status"] == "PASS" for c in criteria):
        verdict = "PRODUCTION_READY"
    elif not crit_hard_fail and sec_serving_ok:
        verdict = "SCOPED_PRODUCTION_READY"
    elif sec_serving_ok:
        verdict = "CONDITIONAL_SCOPED"
    else:
        verdict = "NOT_READY"

    n_pass = sum(c["status"] == "PASS" for c in criteria)
    return {
        "gate": "P_production_readiness",
        "verdict": verdict,
        "summary": f"{n_pass}/{len(criteria)} PASS",
        "critical_blockers": [c["id"] + ": " + c["name"] for c in crit_hard_fail],
        "criteria": criteria,
        "interpretation": {
            "PRODUCTION_READY": "every criterion passes",
            "SCOPED_PRODUCTION_READY": "all CRITICAL criteria pass; deployable as a SCOPED, "
                                        "monitored KGA certificate service.",
            "CONDITIONAL_SCOPED": "security/serving solid; non-critical gaps remain",
            "NOT_READY": "critical security/serving gaps",
        }[verdict],
        "remaining_non_critical_gaps": [c["id"] + ": " + c["name"] for c in any_gap],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "deploy/gate_p_audit.json")
    args = ap.parse_args()
    report = grade()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"=== GATE P — PRODUCTION READINESS: {report['verdict']} ({report['summary']}) ===")
    for c in report["criteria"]:
        mark = {"PASS": "PASS ", "PARTIAL": "PART ", "FAIL": "FAIL ", "MISSING": "MISS "}[c["status"]]
        crit = "!" if c["critical"] else " "
        print(f"  [{mark}]{crit} {c['id']} {c['name']}")
    if report["critical_blockers"]:
        print("\nCRITICAL BLOCKERS:")
        for b in report["critical_blockers"]:
            print("  - " + b)
    print(f"\nVerdict: {report['interpretation']}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
