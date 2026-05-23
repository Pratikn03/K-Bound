"""Phase 1.E — build the locked metrics manifest.

The manifest is the single source of truth for every manuscript-quoted
result. The claim validator (validate_manuscript_claims.py) and the
generated LaTeX macros (docs/research/generated/elara_verified_metrics_macros.tex)
both read this manifest.

For every audited Family A / Family C cell we emit:
  - the validation-frozen RGA+ head and its test ROC-AUC,
  - the validation-frozen primary comparator and its test ROC-AUC,
  - the single-representative-seed DeLong p-value,
  - the Holm-corrected p-value where Family A K=5 applies,
  - per-method seed mean ± SD ROC-AUC for descriptive instability,
  - allowed and forbidden claim labels per the locked policy,
  - the source artifact path for each value.

Canonical PR-AUC / ECE / Brier values are NOT included in the manifest
(blocked by Phase 1.A audit). ROC-AUC at chance level is the only
canonical metric admitted, and only as descriptive evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CELL_REGISTRY = [
    # cell_id, benchmark, protocol, family, status
    ("A1", "MVTec 3D-AD",    "PatchCore canonical",      "A", "protocol-diagnostic"),
    ("A2", "MVTec 3D-AD",    "PatchCore supervised",     "A", "audited primary reanalysis"),
    ("A3", "MVTec 3D-AD",    "PatchCore held-out",       "A", "audited primary reanalysis"),
    ("A4", "MVTec LOCO-AD",  "PatchCore canonical",      "A", "protocol-diagnostic"),
    ("A5", "MVTec LOCO-AD",  "PatchCore supervised",     "A", "audited primary reanalysis"),
    ("A6", "VisA",           "RGB+edge canonical",       "A", "protocol-diagnostic"),
    ("A7", "VisA",           "RGB+edge supervised",      "A", "audited primary reanalysis"),
    ("A8", "UNSW-NB15",      "flow/conn/context",        "A", "audited primary reanalysis"),
    ("C1", "Real3D-AD",      "PCA shape + depth supervised", "C", "exploratory"),
    ("C2", "VisA",           "RGB+random noise-floor",   "C", "exploratory"),
    ("C3", "UNSW-NB15",      "held-out attack categories", "C", "exploratory"),
]

PAIRING_STRENGTH = {
    "MVTec 3D-AD": "independent_modalities",
    "MVTec LOCO-AD": "independent_modalities",
    "Real3D-AD": "naturally_structured_views",
    "VisA": "derived_view_proxy",
    "UNSW-NB15": "naturally_structured_views",
}


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _idx(rows: list[dict], bm: str, proto: str, filter_fn=None) -> dict | None:
    for r in rows:
        if r.get("benchmark") == bm and r.get("protocol") == proto:
            if filter_fn is None or filter_fn(r):
                return r
    return None


def _float_or_none(v):
    try:
        f = float(v)
        import math
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/metrics_manifest.json"),
    )
    parser.add_argument(
        "--macros-output",
        type=Path,
        default=Path("docs/research/generated/elara_verified_metrics_macros.tex"),
    )
    args = parser.parse_args()

    rga_rows = _load_csv(args.repo_root / "experiments/audit/rga_plus_validation_frozen_selection.csv")
    comp_rows = _load_csv(args.repo_root / "experiments/audit/audited_comparator_selection.csv")
    inf_rows = _load_csv(args.repo_root / "experiments/audit/audited_ensemble_inference_results.csv")
    var_rows = _load_csv(args.repo_root / "experiments/audit/descriptive_seed_variability.csv")

    manifest = {
        "policy_lock": "Phase 1.E — locked audited reanalysis; Family A K=5; Family C K=0; canonical PR/ECE/Brier blocked",
        "claims": [],
        "macros": {},  # populated below for LaTeX expansion
    }

    for cell_id, bm, proto, family, status in CELL_REGISTRY:
        rga = _idx(rga_rows, bm, proto, filter_fn=lambda r: r.get("seed_or_ensemble") == "ensemble")
        comp = _idx(comp_rows, bm, proto)
        inf = _idx(inf_rows, bm, proto)
        pairing = PAIRING_STRENGTH.get(bm, "unknown")

        rga_head = (rga or {}).get("selected_head")
        rga_test = _float_or_none((rga or {}).get("selected_test_auc"))
        rga_val = _float_or_none((rga or {}).get("selected_validation_auc"))
        comp_name = (comp or {}).get("selected_comparator")
        comp_test = _float_or_none((comp or {}).get("selected_comparator_test_auc"))
        comp_val = _float_or_none((comp or {}).get("selected_comparator_validation_auc"))
        p_raw = _float_or_none((inf or {}).get("delong_p_raw_single_rep_seed"))
        p_holm = _float_or_none((inf or {}).get("delong_p_holm_if_applicable"))

        delta = (rga_test - comp_test) if (isinstance(rga_test, float) and isinstance(comp_test, float)) else None

        # Per-method seed variability for this cell.
        method_var = {}
        for v in var_rows:
            if v.get("cell_id") != cell_id:
                continue
            m = v.get("method")
            method_var[m] = {
                "n_seeds": int(v.get("n_seeds") or 0),
                "mean_auc": _float_or_none(v.get("mean_auc")),
                "sd_auc": _float_or_none(v.get("sd_auc")),
                "min_auc": _float_or_none(v.get("min_auc")),
                "max_auc": _float_or_none(v.get("max_auc")),
                "descriptive_only": True,
            }

        allowed: str
        forbidden: str
        if status == "protocol-diagnostic":
            allowed = (
                "Descriptive only: under canonical one-class training every supervised head "
                "collapses near chance ROC-AUC. PR-AUC / ECE / Brier values are degenerate-predictor "
                "artefacts (Phase 1.A audit) and are not promoted."
            )
            forbidden = (
                "No superiority claim. No PR-AUC / ECE / Brier promotion. No 'confirmatory' or "
                "'pre-registered' label."
            )
        elif status == "audited primary reanalysis":
            allowed = (
                "Audited inferential summary: validation-frozen RGA+ vs validation-frozen primary "
                "comparator, single-representative-seed DeLong (seed 42), Holm-corrected within "
                "Family A K=5."
            )
            forbidden = (
                "No 'confirmatory' or 'pre-registered' label. No 'RGA+ beats every baseline'. "
                "No 'best non-router' inferential framing. No SOTA / deployment-grade / "
                "production-ready / universal-superiority claims."
            )
        else:  # exploratory
            allowed = "Descriptive point estimates only. No Holm correction."
            forbidden = (
                "No confirmatory or superiority claim. No inclusion in Family A inferential summary."
            )

        claim = {
            "cell_id": cell_id,
            "benchmark": bm,
            "protocol": proto,
            "analysis_family": family,
            "analysis_status": status,
            "pairing_strength": pairing,
            "rga_plus_head": rga_head,
            "rga_plus_test_roc_auc": rga_test,
            "rga_plus_validation_roc_auc": rga_val,
            "rga_plus_selection_rule": "validation_frozen (router vs boost; tie-break boost)",
            "primary_comparator": comp_name,
            "primary_comparator_test_roc_auc": comp_test,
            "primary_comparator_validation_roc_auc": comp_val,
            "primary_comparator_selection_rule": "validation_frozen (max seed-mean val ROC-AUC; deterministic name tie-break)",
            "delta_auc": delta,
            "delong_p_raw_single_rep_seed": p_raw,
            "delong_p_holm_family_A_K5": p_holm if status == "audited primary reanalysis" else None,
            "descriptive_seed_variability_per_method": method_var,
            "source_artifacts": {
                "rga_plus_selection_csv": "experiments/audit/rga_plus_validation_frozen_selection.csv",
                "comparator_selection_csv": "experiments/audit/audited_comparator_selection.csv",
                "audited_inference_csv": "experiments/audit/audited_ensemble_inference_results.csv",
                "descriptive_variability_csv": "experiments/audit/descriptive_seed_variability.csv",
            },
            "allowed_claim": allowed,
            "forbidden_claim": forbidden,
            "metric_semantics_verified": True if status != "protocol-diagnostic" else False,
            "metric_semantics_verified_note": (
                "ROC-AUC is interpretable at chance level; canonical PR/ECE/Brier are "
                "blocked pending the Phase 1.A audit (METRICS_VALID_BUT_MISINTERPRETED — "
                "PR/ECE/Brier equal test-fold prevalence and are not promoted in the manuscript)."
                if status == "protocol-diagnostic" else
                "Verified by Phase 1.A canonical label/metric semantics audit."
            ),
        }
        manifest["claims"].append(claim)

    # Macros — emit one per cell for the most-cited values.
    macros: dict[str, str] = {}
    for c in manifest["claims"]:
        cid = c["cell_id"]
        def _macro(name: str, value):
            key = f"elara{cid}{name}"  # e.g. elaraA2RgaPlusAuc
            if isinstance(value, float):
                macros[key] = f"{value:.4f}"
            elif value is None:
                macros[key] = "--"
            else:
                macros[key] = str(value)
        _macro("RgaPlusAuc", c.get("rga_plus_test_roc_auc"))
        _macro("ComparatorAuc", c.get("primary_comparator_test_roc_auc"))
        _macro("DeltaAuc", c.get("delta_auc"))
        _macro("DelongPRaw", c.get("delong_p_raw_single_rep_seed"))
        _macro("DelongPHolm", c.get("delong_p_holm_family_A_K5"))
        _macro("PrimaryComparator", c.get("primary_comparator") or "")
        _macro("RgaPlusHead", c.get("rga_plus_head") or "")
    manifest["macros"] = macros

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, default=float))
    print(f"Wrote {args.output} ({len(manifest['claims'])} claims, {len(macros)} macros)")

    # Emit LaTeX macros
    lines = [
        "% Auto-generated by build_metrics_manifest.py — Phase 1 locked audited reanalysis.",
        "% Source: docs/research/metrics_manifest.json",
        "% Do not edit by hand; rerun the build script.",
        "",
    ]
    for k in sorted(macros.keys()):
        v = macros[k]
        # LaTeX-escape underscores in macro names is not possible; use only [A-Za-z]
        # so we sanitise digits→letters here. (Macro names cannot contain digits or
        # hyphens.) Replace digit→letter and remove non-alpha.
        sanitised = (
            k.replace("0", "Zero").replace("1", "One").replace("2", "Two")
             .replace("3", "Three").replace("4", "Four").replace("5", "Five")
             .replace("6", "Six").replace("7", "Seven").replace("8", "Eight")
             .replace("9", "Nine")
        )
        # Replace unsafe chars (none expected after sanitise) with empty.
        sanitised = "".join(ch for ch in sanitised if ch.isalpha())
        # Escape underscores in the value
        v_safe = v.replace("_", r"\_")
        lines.append(rf"\newcommand{{\{sanitised}}}{{{v_safe}}}")
    args.macros_output.parent.mkdir(parents=True, exist_ok=True)
    args.macros_output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.macros_output} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
