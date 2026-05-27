"""Phase 2.2B.exec — B-MECH-1 seed-ensemble inference + Holm K=2."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from elara.evaluation.ensemble_inference import audited_analysis, holm_bonferroni  # noqa: E402

ARCHIVE = ROOT / "experiments" / "phase2" / "mechanism" / "b_mech_1_prediction_archives"

PHASE1_TARGET = {
    "zero_attack_k4": {"label": "B1", "target_delta": 0.0506, "ci_low": 0.0315, "ci_high": 0.0681},
    "max_attack_k4": {"label": "B2", "target_delta": 0.0319, "ci_low": 0.0050, "ci_high": 0.0617},
}


def _load_per_seed(method_dir: Path) -> dict[int, pd.DataFrame]:
    out = {}
    for p in sorted((method_dir / "test").glob("seed_*.parquet")):
        s = int(p.stem.replace("seed_", ""))
        out[s] = pd.read_parquet(p)
    return out


def main() -> int:
    if not ARCHIVE.exists():
        print("no archive; run B-MECH-1 first")
        return 1
    cell_dirs = [d for d in ARCHIVE.iterdir() if d.is_dir()]
    if not cell_dirs:
        return 1
    cell_dir = cell_dirs[0]
    methods = {d.name: d for d in cell_dir.iterdir() if d.is_dir()}
    # Pair by scenario suffix (e.g. 'static_attention__zero_attack_k4')
    pairs = {}
    for name, d in methods.items():
        if "__" not in name:
            continue
        base, scenario = name.split("__", 1)
        if base in ("static_attention", "rga_mean_gate_tau66"):
            pairs.setdefault(scenario, {})[base] = d

    rows = []
    raw_p_map = {}
    for scenario, mp in sorted(pairs.items()):
        static_per_seed = _load_per_seed(mp["static_attention"])
        rga_per_seed = _load_per_seed(mp["rga_mean_gate_tau66"])
        seeds = sorted(set(static_per_seed) & set(rga_per_seed))
        s0 = seeds[0]
        sids = static_per_seed[s0]["sample_id"].to_numpy()
        labels = static_per_seed[s0]["label"].to_numpy().astype(int)
        rga_scores = {s: rga_per_seed[s]["raw_score"].to_numpy().astype(float) for s in seeds}
        static_scores = {s: static_per_seed[s]["raw_score"].to_numpy().astype(float) for s in seeds}
        res = audited_analysis(
            cell_id=f"B-MECH-1__{scenario}",
            benchmark="ELARA-Bench-LA",
            protocol=f"k-of-D mean-gate tau=0.66 {scenario}",
            rga_method="rga_mean_gate_tau66",
            comparator_method="static_attention",
            sample_ids=sids,
            labels=labels,
            per_seed_rga_scores=rga_scores,
            per_seed_comp_scores=static_scores,
        )
        target = PHASE1_TARGET[scenario]
        rows.append(
            {
                "endpoint": target["label"],
                "scenario": scenario,
                "n_seeds": res.n_seeds,
                "n_test": res.n_test_samples,
                "ensemble_static_auc": res.ensemble_comparator_auc,
                "ensemble_rga_auc": res.ensemble_rga_auc,
                "ensemble_delta_auc": res.ensemble_delta_auc,
                "per_seed_mean_delta": float(np.mean(res.per_seed_deltas)),
                "per_seed_sd_delta": float(np.std(res.per_seed_deltas, ddof=1)),
                "sign_consistent_seeds": res.sign_consistent_seeds,
                "delong_p_raw": res.delong_p_value,
                "bootstrap_ci_low": res.bootstrap_ci_low,
                "bootstrap_ci_high": res.bootstrap_ci_high,
                "practical_effect_band": res.practical_effect_band,
                "phase1_target_delta": target["target_delta"],
                "phase1_target_ci_low": target["ci_low"],
                "phase1_target_ci_high": target["ci_high"],
            }
        )
        raw_p_map[scenario] = res.delong_p_value

    holm = holm_bonferroni(raw_p_map, K=2)
    for r in rows:
        r["delong_p_holm_k2"] = holm[r["scenario"]]
        # Replication decision per spec
        if r["ensemble_delta_auc"] > 0 and r["delong_p_holm_k2"] <= 0.05 and r["bootstrap_ci_low"] > 0:
            r["replication_decision"] = "REPRODUCED"
        elif r["ensemble_delta_auc"] > 0:
            r["replication_decision"] = "DIRECTIONALLY_SUPPORTED"
        elif r["ensemble_delta_auc"] <= 0:
            r["replication_decision"] = "NOT_REPRODUCED"
        else:
            r["replication_decision"] = "INCONCLUSIVE"

    # Write inference + Holm CSVs
    inf_path = ROOT / "experiments" / "phase2" / "mechanism" / "family_b_primary_replication_inference.csv"
    holm_path = ROOT / "experiments" / "phase2" / "mechanism" / "family_b_primary_replication_holm_k2.csv"
    inf_path.parent.mkdir(parents=True, exist_ok=True)
    with inf_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    holm_fields = [
        "endpoint",
        "scenario",
        "ensemble_delta_auc",
        "delong_p_raw",
        "delong_p_holm_k2",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "practical_effect_band",
        "replication_decision",
    ]
    with holm_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=holm_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in holm_fields})
    print(f"wrote {inf_path}")
    print(f"wrote {holm_path}")

    for r in rows:
        print(
            f"{r['endpoint']} ({r['scenario']}): Δ={r['ensemble_delta_auc']:+.4f} "
            f"CI=[{r['bootstrap_ci_low']:+.4f}, {r['bootstrap_ci_high']:+.4f}] "
            f"raw_p={r['delong_p_raw']:.3e}  Holm_K2={r['delong_p_holm_k2']:.3e}  "
            f"band={r['practical_effect_band']}  decision={r['replication_decision']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
