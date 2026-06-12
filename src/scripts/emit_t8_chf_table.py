"""Render docs/research/tables/t8_chf.tex from the VERIFIED, surviving T8 validation
artifact (experiments/fusion/t8_chf_validation.json), without re-running the flagship
harness.

The original validate_t8_chf.py both (re)computes the T8 (Certified Heterogeneous
Fusion) result and renders the table. Its recompute path needs a deployment config
(configs/elara_deploy_m2_external_validation_v1.yaml) that was removed in the 2026-06-03
Master-C retirement and is not in the backup tarball. The *result* it produced, however,
survives as the JSON artifact. This script re-emits the identical table from that
artifact so the wiped table is restored and validate_theorem_stack.py goes green --
rendering verified data, not recomputing or fabricating it.
"""
from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "experiments").is_dir() and (parent / "docs").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def main() -> int:
    root = _repo_root()
    art = root / "experiments/fusion/t8_chf_validation.json"
    if not art.exists():
        raise SystemExit(f"missing verified artifact {art}")
    d = json.loads(art.read_text(encoding="utf-8"))
    chf = d.get("elara_chf_v1", {})
    cert = chf.get("certificate", {})
    tex = root / "docs/research/tables/t8_chf.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text(
        "\\begin{tabular}{ll}\n"
        "\\toprule\n"
        "Field & Value \\\\\n"
        "\\midrule\n"
        f"Route & {cert.get('route', '?')} \\\\\n"
        f"Val ROC-AUC & {cert.get('val_roc_auc', 0):.4f} \\\\\n"
        f"Test $\\Delta$ vs SAR & {chf.get('delta_vs_sar', 0):.4f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )
    print(f"Wrote {tex} (from verified {art.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
