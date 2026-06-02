#!/usr/bin/env python3
"""Emit T8 CHF validation artifact from flagship harness (M2 validation)."""

from __future__ import annotations

import json
import sys

import yaml

from src.scripts.scenario_c.flagship_harness import evaluate_flagship_seed
from src.scripts.scenario_c.win_vs_sar_harness import _repo_root


def main() -> int:
    root = _repo_root()
    cfg_path = root / "configs/elara_deploy_m2_external_validation_v1.yaml"
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    rows = [evaluate_flagship_seed(cfg, seed=42, eval_split="validation")]
    chf = rows[0].get("elara_chf_v1", {})
    out = {
        "theorem_id": "T8",
        "seed": 42,
        "eval_split": "validation",
        "elara_chf_v1": chf,
        "sar_roc_auc": rows[0].get("sar_roc_auc"),
    }
    path = root / "experiments/fusion/t8_chf_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tex = root / "docs/research/tables/t8_chf.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    cert = chf.get("certificate", {})
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
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
