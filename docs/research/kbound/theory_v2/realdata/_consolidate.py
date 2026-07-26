# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). KB_REPO_ROOT is discovered from this
# --- file's own location; override with $KBOUND_REPO_ROOT.
import os as _kb_os
from pathlib import Path as _KbPath


def _kb_repo_root() -> str:
    override = _kb_os.environ.get("KBOUND_REPO_ROOT", "").strip()
    if override:
        return str(_KbPath(override).expanduser().resolve())
    here = _KbPath(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
    raise RuntimeError(f"repository root not found above {here}; set KBOUND_REPO_ROOT")


KB_REPO_ROOT = _kb_repo_root()

import json, os, hashlib
HERE = KB_REPO_ROOT + "/docs/research/kbound/theory_v2/realdata"
p1 = json.load(open(os.path.join(HERE, "_p1_partial.json")))["P1"]
p2 = json.load(open(os.path.join(HERE, "_p2_partial.json")))["P2"]

# original flagged CIs for side-by-side honesty
flagged = json.load(open(KB_REPO_ROOT + "/experiments/kbound/results/decisive_tta_cis.json"))

out = {
    "_meta": {
        "agent": "K-Bound theory_v2 real-data validation (Agent B)",
        "date": "2026-06-10",
        "python": "3.14.3", "numpy": "2.4.4",
        "compute": "CPU-only, repo venv",
        "integrity": ("All numbers produced by run_p1.py / run_p2.py on real on-disk data. "
                      "Labels in P1 used ONLY to compute ground-truth b_true/sign_true for scoring, "
                      "never to fit the label-free estimators. Failures reported as-is."),
        "data_sources": {
            "P1": KB_REPO_ROOT + "/experiments/elara_u/score_archive/*.npz (123-task anomaly bank, 6 detectors)",
            "P2": KB_REPO_ROOT + "/experiments/kbound/results/cifar10c_65cells.csv (real per-condition CIFAR-10-C grid)",
        },
    },
    "P1_anomaly_bank": p1,
    "P2_tta_stats_hardening": p2,
    "P2_comparison_to_flagged": {
        "flagged_ci_source": flagged.get("ci_source"),
        "flagged_note": ("Original Table-7 CIs were resampled from the 11-point synthetic "
                         "mixing-ratio Pareto curve (each point a bootstrap-mean over a 200-condition "
                         "SYNTHETIC stream with injected harmful fractions). Those reported KGA beating "
                         "always-adapt at p~0.0013-0.0017, d~-1.4."),
        "hardened_finding": ("On the REAL 65-cell CIFAR-10-C grid with a genuine per-condition paired "
                             "bootstrap + Holm: KGA vs always-FREEZE survives Holm for all 3 methods "
                             "(dregret~-0.214, p~2e-20); KGA vs always-ADAPT does NOT survive Holm for "
                             "any method (ties / marginally worse). The always-adapt advantage in the "
                             "flagged table was an artifact of the synthetic harmful-fraction mixing, "
                             "absent from the clean corruption grid where adapting is almost always helpful."),
    },
}
path = os.path.join(HERE, "realdata_audit_results.json")
json.dump(out, open(path, "w"), indent=2)
# checksum + size
b = open(path, "rb").read()
print("wrote", path, len(b), "bytes  sha256", hashlib.sha256(b).hexdigest()[:16])
# quick sanity echo of headline numbers
sm = p1["summary_median"]; sv = p1["summary_val_opt"]
print("\nP1 H-reject rate (median/valopt):", sm["H_reject_rate"], "/", sv["H_reject_rate"])
print("P1 sign-acc H-pass vs H-reject (median):", sm["sign_acc_H_pass"], sm["sign_acc_H_reject"])
print("P1 sign-acc H-pass vs H-reject (valopt):", sv["sign_acc_H_pass"], sv["sign_acc_H_reject"])
print("P2 survivors Holm:", p2["survivors_holm"])
