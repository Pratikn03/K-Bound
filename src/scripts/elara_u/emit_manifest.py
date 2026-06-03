"""Emit experiments/elara_u/manifest.json + sha256sums.txt for the curated ELARA-U
result artifacts (verified evidence). Run as part of release packaging."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/elara_u"
CURATED = [
    "honest_benchmark.json", "learned_router_ablation.json", "shift_stress_ablation.json",
    "heterogeneous_degradation_ablation.json", "synthetic_multimodal_poc.json",
    "natural_shift_results.json", "multimodal_reliability_results.json",
    "calibration_results.json", "statistical_audit.json", "score_archive_manifest.json",
]

def sha256(p): 
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        commit = "unknown"
    files = {}
    sums = []
    for name in CURATED:
        p = EXP / name
        if not p.exists():
            continue
        h = sha256(p)
        files[name] = {"bytes": p.stat().st_size, "sha256": h}
        sums.append(f"{h}  experiments/elara_u/{name}")
    manifest = {"protocol": "ELARA_U_RESULT_MANIFEST_v1", "git_commit": commit,
                "python": sys.version.split()[0], "n_files": len(files), "files": files}
    (EXP / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (EXP / "sha256sums.txt").write_text("\n".join(sums) + "\n")
    print(f"wrote manifest.json + sha256sums.txt ({len(files)} curated artifacts, commit {commit[:8]})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
