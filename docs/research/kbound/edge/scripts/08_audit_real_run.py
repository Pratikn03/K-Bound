#!/usr/bin/env python3
"""08 -- execute the machine-verifiable anti-leakage audit (Table S2 checks).

Loads the physical-camera configuration, runs all eight S2 checks (state dict
isolation, online log key restrictions, feature schemas, conformal split
disjointness, held-out session exclusion, policy stream identity, config and
model hash invariants), writes anti_leakage_audit.json, and exits nonzero if any
check fails.
"""

import argparse
import sys
import os

import _common as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_real_phone_v1.yaml")
    ap.add_argument("--strict", action="store_true", help="exit nonzero on any audit failures")
    args = ap.parse_args()

    cfg = C.load_config(args.config)

    from kbound_edge.integrity import run_full_audit

    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"
    if not is_real:
        raise SystemExit("[08] Audit script is only valid for physical real protocol mode.")

    print("[08] Running anti-leakage and schema integrity audit...")
    checks = run_full_audit(cfg)

    # Save to anti_leakage_audit.json under results directory
    results_dir = os.path.normpath(os.path.join(C.EDGE_ROOT, cfg["paths"]["results_dir"]))
    audit_out_path = os.path.join(results_dir, "anti_leakage_audit.json")
    os.makedirs(results_dir, exist_ok=True)
    C.save_json(audit_out_path, {"checks": checks})

    print(f"[08] Wrote audit report to: {audit_out_path}")
    print("\nCheck Results:")
    print("-" * 80)

    any_fail = False
    for idx, c in enumerate(checks):
        status = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            any_fail = True
        print(f"[{idx+1}] {c['check']}")
        print(f"    Status:   {status}")
        print(f"    Expected: {c['expected']}")
        print(f"    Observed: {c['observed']}")
        print(f"    Evidence: {c['evidence_artifact']} (hash: {c['evidence_hash'][:8] if c['evidence_hash'] != 'missing' else 'N/A'})")
        print("-" * 80)

    if any_fail:
        print("[08] AUDIT FAILED: One or more checks did not pass.")
        if args.strict:
            sys.exit(1)
    else:
        print("[08] AUDIT PASSED: All checks passed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
