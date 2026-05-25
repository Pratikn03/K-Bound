"""Phase 2.2A — Family-A v2 audited analysis driver.

Computes the PRIMARY_FAMILY_A_CELL_LEVEL_STATIC_REFERENCE_AUDIT
surface from the prediction archive:

    For each completed A-POWERED-N cell:
      - per-seed validation-frozen RGA+ test predictions
      - per-seed static_attention test predictions
      - seed-ensemble paired DeLong + paired sample bootstrap CI
        against static_attention only.

Then, ONLY if all five Family-A cells have completed archives:
      - Holm-Bonferroni across K = 5 cell-level p-values.

Outputs:
  experiments/phase2/statistics/family_a_v2_primary_cell_level_raw.csv
  experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv

Does NOT overwrite:
  experiments/phase2/statistics/family_a_powered_ensemble_inference.csv
  experiments/phase2/statistics/family_a_powered_holm_results.csv

Refuses to:
  - apply K = 5 Holm with fewer than 5 cells (writes a partial CSV
    instead with holm_p = pending_full_family);
  - compute against any comparator other than static_attention;
  - touch any Family B/C/D experiment_id.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from elara.evaluation.ensemble_inference import (  # noqa: E402
    audited_analysis, holm_bonferroni,
)

REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
ARCHIVE_ROOT = ROOT / "experiments" / "phase2" / "predictions"
STATS_DIR = ROOT / "experiments" / "phase2" / "statistics"

FAMILY_A_CELLS = (
    "A-POWERED-1",
    "A-POWERED-2",
    "A-POWERED-3",
    "A-POWERED-4",
    "A-POWERED-5",
)

A1_HISTORICAL_METRICS = STATS_DIR / "family_a_powered_seed_metrics.csv"  # historical


def _registry_lookup() -> dict[str, dict[str, str]]:
    with REGISTRY_V2.open() as f:
        return {r["experiment_id"]: r for r in csv.DictReader(f)}


def _slug(s: str) -> str:
    """Match PredictionArchive's slug: spaces and slashes → underscores."""
    return s.replace(" ", "_").replace("/", "_")


def _cell_dir(experiment_id: str, benchmark: str, protocol: str) -> Path:
    return ARCHIVE_ROOT / f"{experiment_id}__{_slug(benchmark)}__{_slug(protocol)}"


def _v2_seed_metrics_path(eid: str) -> Path:
    return STATS_DIR / f"family_a_v2_{eid}_seed_metrics.csv"


def _seed_metrics_for(eid: str) -> pd.DataFrame:
    """Use the v2 per-cell seed-metrics CSV when it exists; for A-POWERED-1 fall
    back to the historical CSV (which contains the same RGA+ head selection
    log even though it predates Phase 2.2A)."""
    p = _v2_seed_metrics_path(eid)
    if p.exists():
        return pd.read_csv(p).drop_duplicates(subset=["seed"], keep="first")
    if eid == "A-POWERED-1" and A1_HISTORICAL_METRICS.exists():
        return pd.read_csv(A1_HISTORICAL_METRICS).drop_duplicates(subset=["seed"], keep="first")
    raise FileNotFoundError(f"seed metrics not found for {eid}; expected {p}")


def _load_test_predictions(cell_dir: Path, method: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    test_dir = cell_dir / method / "test"
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


def _rga_plus_val_frozen_scores(
    eid: str, cell_dir: Path
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Return (per-seed scores dict, canonical sample_ids, canonical labels)
    for the validation-frozen RGA+ head."""
    sm = _seed_metrics_for(eid)
    chosen = dict(zip(sm["seed"].astype(int), sm["chosen_head"]))
    router = _load_test_predictions(cell_dir, "rga_meta_router")
    boost = _load_test_predictions(cell_dir, "rga_boosted_fusion")
    if not router or not boost:
        raise FileNotFoundError(f"{eid}: missing rga_meta_router or rga_boosted_fusion archive")
    seeds = sorted(chosen)
    sids0, lbl0, _ = router[seeds[0]]
    rga: dict[int, np.ndarray] = {}
    for s in seeds:
        if s not in router or s not in boost:
            raise ValueError(f"{eid} seed {s}: router/boost archive missing")
        sids, lbl, _ = router[s]
        if not (np.array_equal(sids, sids0) and np.array_equal(lbl, lbl0)):
            raise ValueError(f"{eid} seed {s}: sample/label mismatch vs canonical")
        rga[s] = router[s][2] if chosen[s] == "router" else boost[s][2]
    return rga, sids0, lbl0


def _static_scores(eid: str, cell_dir: Path,
                   canonical_ids: np.ndarray, canonical_labels: np.ndarray
                   ) -> dict[int, np.ndarray]:
    static = _load_test_predictions(cell_dir, "static_attention")
    out: dict[int, np.ndarray] = {}
    for s, (sids, lbl, sc) in static.items():
        if not (np.array_equal(sids, canonical_ids) and np.array_equal(lbl, canonical_labels)):
            raise ValueError(f"{eid} seed {s}: static_attention sample/label mismatch")
        out[s] = sc
    if not out:
        raise FileNotFoundError(f"{eid}: missing static_attention archive")
    return out


def _per_cell_audit(eid: str, registry: dict[str, dict[str, str]]):
    row = registry[eid]
    if row["primary_comparator"] != "static_attention":
        raise SystemExit(
            f"{eid}: registry primary_comparator={row['primary_comparator']!r}; "
            "this driver compares only against static_attention"
        )
    cell_dir = _cell_dir(eid, row["benchmark"], row["protocol"])
    if not cell_dir.exists():
        return None

    rga_scores, ids, lbl = _rga_plus_val_frozen_scores(eid, cell_dir)
    static_scores = _static_scores(eid, cell_dir, ids, lbl)
    common = sorted(set(rga_scores) & set(static_scores))
    if not common:
        return None
    rga_scores = {s: rga_scores[s] for s in common}
    static_scores = {s: static_scores[s] for s in common}

    res = audited_analysis(
        cell_id=eid,
        benchmark=row["benchmark"],
        protocol=row["protocol"],
        rga_method="rga_plus_validation_frozen",
        comparator_method="static_attention",
        sample_ids=ids,
        labels=lbl,
        per_seed_rga_scores=rga_scores,
        per_seed_comp_scores=static_scores,
    )
    # Descriptive stats
    deltas = np.array(res.per_seed_deltas)
    mean_delta = float(np.mean(deltas))
    sd_delta = float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0
    sign_consistent = int(res.sign_consistent_seeds)
    return {
        "experiment_id": eid,
        "benchmark": row["benchmark"],
        "protocol": row["protocol"],
        "pairing_strength": row["pairing_strength"],
        "analysis_surface": "PRIMARY_FAMILY_A_CELL_LEVEL_STATIC_REFERENCE_AUDIT",
        "n_seeds": res.n_seeds,
        "n_test_samples": res.n_test_samples,
        "ensemble_rga_auc": res.ensemble_rga_auc,
        "ensemble_static_auc": res.ensemble_comparator_auc,
        "ensemble_delta_auc": res.ensemble_delta_auc,
        "per_seed_mean_delta": mean_delta,
        "per_seed_sd_delta": sd_delta,
        "sign_consistent_seeds": sign_consistent,
        "delong_p_raw": res.delong_p_value,
        "bootstrap_ci_low": res.bootstrap_ci_low,
        "bootstrap_ci_high": res.bootstrap_ci_high,
        "bootstrap_n_iter": res.bootstrap_n_iter,
        "practical_effect_band": res.practical_effect_band,
        "per_seed_rga_aucs": ";".join(f"{x:.6f}" for x in res.per_seed_rga_aucs),
        "per_seed_static_aucs": ";".join(f"{x:.6f}" for x in res.per_seed_comp_aucs),
        "per_seed_deltas": ";".join(f"{x:.6f}" for x in res.per_seed_deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-holm-k5", action="store_true",
                        help="apply Holm K=5 — only valid if all 5 cells are present")
    args = parser.parse_args()
    registry = _registry_lookup()

    rows = []
    completed = []
    missing = []
    for eid in FAMILY_A_CELLS:
        r = _per_cell_audit(eid, registry)
        if r is None:
            missing.append(eid)
        else:
            completed.append(eid)
            rows.append(r)

    # Raw CSV — one row per completed cell
    raw_out = STATS_DIR / "family_a_v2_primary_cell_level_raw.csv"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(rows[0].keys())
        with raw_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {raw_out} ({len(rows)} cells)")
    else:
        print(f"no cells completed; not writing {raw_out}")

    # Holm K = 5 — only if all 5 cells complete
    holm_out = STATS_DIR / "family_a_v2_primary_cell_level_holm_k5.csv"
    holm_fields = [
        "experiment_id", "benchmark", "protocol", "ensemble_delta_auc",
        "delong_p_raw", "delong_p_holm_k5", "bootstrap_ci_low",
        "bootstrap_ci_high", "practical_effect_band", "holm_status",
    ]
    if missing:
        # Partial: write per-cell rows with holm_p = pending_full_family
        with holm_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=holm_fields)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "experiment_id": r["experiment_id"],
                    "benchmark": r["benchmark"],
                    "protocol": r["protocol"],
                    "ensemble_delta_auc": r["ensemble_delta_auc"],
                    "delong_p_raw": r["delong_p_raw"],
                    "delong_p_holm_k5": "pending_full_family",
                    "bootstrap_ci_low": r["bootstrap_ci_low"],
                    "bootstrap_ci_high": r["bootstrap_ci_high"],
                    "practical_effect_band": r["practical_effect_band"],
                    "holm_status": "PARTIAL_FAMILY",
                })
        print(f"wrote {holm_out} — PARTIAL: missing cells = {missing}")
        print("K=5 Holm correction NOT applied (partial family); rerun with all 5 cells.")
    else:
        # Apply K = 5 Holm
        raw_p_map = {r["experiment_id"]: r["delong_p_raw"] for r in rows}
        holm_map = holm_bonferroni(raw_p_map, K=5)
        if args.apply_holm_k5 is False:
            # default is to apply Holm — flag is for explicit confirmation
            pass
        with holm_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=holm_fields)
            w.writeheader()
            for r in rows:
                eid = r["experiment_id"]
                w.writerow({
                    "experiment_id": eid,
                    "benchmark": r["benchmark"],
                    "protocol": r["protocol"],
                    "ensemble_delta_auc": r["ensemble_delta_auc"],
                    "delong_p_raw": r["delong_p_raw"],
                    "delong_p_holm_k5": holm_map[eid],
                    "bootstrap_ci_low": r["bootstrap_ci_low"],
                    "bootstrap_ci_high": r["bootstrap_ci_high"],
                    "practical_effect_band": r["practical_effect_band"],
                    "holm_status": "K5_FULL_FAMILY",
                })
        print(f"wrote {holm_out} — FULL K=5 Holm applied across {len(rows)} cells")

    print(f"completed: {completed}")
    print(f"missing  : {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
