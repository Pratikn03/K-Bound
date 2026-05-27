"""Phase 2.A.4 — run the seed-ensemble audited analysis on the
prediction archive for A-POWERED-1 (the in-session pilot cell).

This script loads per-seed test predictions for the validation-frozen
RGA+ head and every comparator method, runs the audited inference
(seed-averaged DeLong + paired test-sample bootstrap), then Holm-
corrects within the family of named comparators.

The remaining A-POWERED-2..5 cells are pending_compute and noted as
such in the report; this script only emits the A-POWERED-1 row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from elara.evaluation.ensemble_inference import audited_analysis  # noqa: E402

CELL_ID = "A-POWERED-1"
BENCHMARK = "MVTec 3D-AD"
PROTOCOL = "PatchCore supervised-paired"
ARCHIVE_DIR = ROOT / "experiments" / "phase2" / "predictions" / "A-POWERED-1__MVTec_3D-AD__PatchCore_supervised-paired"
STATS_DIR = ROOT / "experiments" / "phase2" / "statistics"
SEED_METRICS_CSV = STATS_DIR / "family_a_powered_seed_metrics.csv"

COMPARATORS = [
    "static_attention",
    "craf_attention",
    "early_fusion_mlp",
    "late_fusion_ensemble",
    "confidence_weighted_mean",
    "random_forest",
    "tent_score_adapter",
    "eata_score_adapter",
    "sar_score_adapter",
    "ttt_pseudo_label_adapter",
]


def _load_test_predictions(method: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {seed: (sample_ids, labels, raw_scores)}."""
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    test_dir = ARCHIVE_DIR / method / "test"
    if not test_dir.exists():
        return out
    for p in sorted(test_dir.glob("seed_*.parquet")):
        seed = int(p.stem.replace("seed_", ""))
        df = pd.read_parquet(p)
        out[seed] = (
            df["sample_id"].to_numpy(),
            df["label"].to_numpy().astype(int),
            df["raw_score"].to_numpy().astype(float),
        )
    return out


def _load_rga_plus_validation_frozen() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each seed, return (sample_ids, labels, scores) where scores
    are taken from the validation-frozen chosen head (router or boost)
    as recorded in family_a_powered_seed_metrics.csv."""
    sm = pd.read_csv(SEED_METRICS_CSV).drop_duplicates(subset=["seed"], keep="first")
    chosen_map = dict(zip(sm["seed"].astype(int), sm["chosen_head"]))
    router = _load_test_predictions("rga_meta_router")
    boost = _load_test_predictions("rga_boosted_fusion")
    out = {}
    for s, head in chosen_map.items():
        if head == "router":
            out[s] = router[s]
        else:
            out[s] = boost[s]
    return out


def main() -> int:
    rga_pred = _load_rga_plus_validation_frozen()
    if not rga_pred:
        raise SystemExit("no RGA+ archive found")
    # canonical label / id from first seed
    seed0 = sorted(rga_pred)[0]
    canonical_ids, canonical_labels, _ = rga_pred[seed0]
    # per-seed score dict
    rga_scores: dict[int, np.ndarray] = {}
    for s, (sids, lbls, sc) in rga_pred.items():
        if not (np.array_equal(sids, canonical_ids) and np.array_equal(lbls, canonical_labels)):
            raise SystemExit(f"seed {s}: sample_id / label mismatch vs canonical")
        rga_scores[s] = sc

    # Step 1 — compute per-comparator raw DeLong p-values
    per_comp_raw: dict[str, dict] = {}
    per_comp_inputs: dict[str, dict[int, np.ndarray]] = {}
    for comp in COMPARATORS:
        cdata = _load_test_predictions(comp)
        cscores: dict[int, np.ndarray] = {}
        for s, (sids, lbls, sc) in cdata.items():
            if not (np.array_equal(sids, canonical_ids) and np.array_equal(lbls, canonical_labels)):
                raise SystemExit(f"comparator {comp} seed {s}: sample mismatch")
            cscores[s] = sc
        # First pass: get raw DeLong p (uncorrected) by calling audited_analysis with no holm_input
        res = audited_analysis(
            cell_id=f"{CELL_ID}__{comp}",
            benchmark=BENCHMARK,
            protocol=PROTOCOL,
            rga_method="rga_plus_validation_frozen",
            comparator_method=comp,
            sample_ids=canonical_ids,
            labels=canonical_labels,
            per_seed_rga_scores=rga_scores,
            per_seed_comp_scores=cscores,
        )
        per_comp_raw[comp] = {
            "raw_p": res.delong_p_value,
            "result": res,
        }
        per_comp_inputs[comp] = cscores

    # Step 2 — Holm-Bonferroni across the family of named comparators
    raw_p_map = {comp: per_comp_raw[comp]["raw_p"] for comp in COMPARATORS}
    from elara.evaluation.ensemble_inference import holm_bonferroni

    holm_p_map = holm_bonferroni(raw_p_map, K=len(COMPARATORS))

    # Step 3 — assemble final rows with the Holm-adjusted p stamped in
    rows = []
    for comp in COMPARATORS:
        r = per_comp_raw[comp]["result"]
        d = r.to_row()
        d["delong_p_holm"] = float(holm_p_map[comp])
        # tuple fields → string for CSV
        for k in ("per_seed_rga_aucs", "per_seed_comp_aucs", "per_seed_deltas"):
            d[k] = ";".join(f"{x:.6f}" for x in d[k])
        rows.append(d)

    out_df = pd.DataFrame(rows)
    out_path = STATS_DIR / "family_a_powered_ensemble_inference.csv"
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    # Print a compact table for the report
    print(
        f"\n{'comparator':25s}  {'rga_ens':>8s}  {'comp_ens':>8s}  {'delta':>8s}  {'p_raw':>10s}  {'p_holm':>10s}  {'CI':>22s}  band"
    )
    for r in rows:
        ci = f"[{r['bootstrap_ci_low']:+.4f}, {r['bootstrap_ci_high']:+.4f}]"
        print(
            f"  {r['comparator_method']:25s}  "
            f"{r['ensemble_rga_auc']:8.4f}  {r['ensemble_comparator_auc']:8.4f}  "
            f"{r['ensemble_delta_auc']:+8.4f}  {r['delong_p_value']:10.4g}  "
            f"{r['delong_p_holm']:10.4g}  {ci:>22s}  {r['practical_effect_band']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
