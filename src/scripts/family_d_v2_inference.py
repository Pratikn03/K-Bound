"""Phase 2.2E — Family-D v2 inference: read test labels (one-time, authorised),
compute seed-ensemble DeLong + paired bootstrap CI per primary endpoint,
apply Holm-Bonferroni K=2, and emit the family decision per the frozen
decision rules.

This script reads Eyecandies test metadata.yaml `anomalous` flags for the
first and only time. The flag is binary (0 or 1) and is the official
test label for image-level ROC-AUC.

D-EYE-3 (if predictions exist) is reported as descriptive only — NOT in
the Holm K=2 family.
"""

from __future__ import annotations

import csv
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from elara.evaluation.ensemble_inference import (  # noqa: E402
    audited_analysis,
    holm_bonferroni,
)

ARCHIVE = Path(ROOT, "data", "raw", "eyecandies", "_archives")
PRED_DIR = Path(ROOT, "experiments", "phase2", "family_d", "predictions")
OUT_DIR = Path(ROOT, "experiments", "phase2", "family_d")
PROTOCOL_YAML = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"

PRIMARY_ENDPOINTS = ("D-EYE-1", "D-EYE-2")
SECONDARY_DESCRIPTIVE = ("D-EYE-3",)


def _read_eyecandies_test_labels() -> dict[str, int]:
    """ONE-TIME AUTHORISED LABEL READ. Walk every test_public + test_private
    metadata.yaml in the .tar archives and record the binary `anomalous`
    flag per fully-qualified sample_id (e.g. 'CandyCane__test_public__000').
    """
    labels: dict[str, int] = {}
    for tar_path in sorted(ARCHIVE.glob("*.tar")):
        if tar_path.name.startswith("._"):
            continue
        cat = tar_path.stem
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                if "metadata.yaml" not in m.name:
                    continue
                parts = m.name.split("/")
                split = None
                for p in parts:
                    if p in ("test_public", "test_private"):
                        split = p
                        break
                if split is None:
                    continue
                base = parts[-1]
                try:
                    sample_id = base.split("_")[0]
                except IndexError:
                    continue
                if not sample_id.isdigit():
                    continue
                key = f"{cat}__{split}__{sample_id}"
                content = tf.extractfile(m).read().decode("utf-8")
                meta = yaml.safe_load(content)
                labels[key] = int(meta.get("anomalous", 0))
    return labels


def _load_per_seed_predictions(endpoint: str, method: str) -> dict[int, pd.DataFrame]:
    """Find the prediction-archive cell for this endpoint × method and load
    per-seed parquets."""
    out = {}
    # The archive uses cell_dir slugs like <endpoint>__Eyecandies-1.0.3__<protocol>
    for cell_dir in PRED_DIR.iterdir():
        if not cell_dir.is_dir() or cell_dir.name.startswith("._"):
            continue
        if not cell_dir.name.startswith(f"{endpoint}__"):
            continue
        method_subdirs = [d for d in cell_dir.iterdir() if d.is_dir() and not d.name.startswith("._")]
        target_method_dir = None
        for d in method_subdirs:
            if d.name == method or d.name.startswith(f"{method}_"):
                target_method_dir = d
                break
        if target_method_dir is None:
            continue
        test_dir = target_method_dir / "test"
        if not test_dir.exists():
            continue
        for p in sorted(test_dir.glob("seed_*.parquet")):
            if p.name.startswith("._"):
                continue
            seed = int(p.stem.replace("seed_", ""))
            out[seed] = pd.read_parquet(p)
    return out


def _stack_ensemble(
    per_seed: dict[int, pd.DataFrame], labels: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Return (sample_ids, label_vec, per_seed_score_vec). Sample IDs that
    don't appear in the label dict are dropped."""
    seeds = sorted(per_seed.keys())
    s0 = per_seed[seeds[0]]
    sids_all = s0["sample_id"].to_numpy()
    keep_mask = np.array([sid in labels for sid in sids_all])
    sids = sids_all[keep_mask]
    lbl = np.array([labels[sid] for sid in sids], dtype=int)
    per_seed_scores = {}
    for s in seeds:
        df = per_seed[s]
        arr = df["raw_score"].to_numpy().astype(float)
        per_seed_scores[s] = arr[keep_mask]
    return sids, lbl, per_seed_scores


def _decision(delta, holm_p, ci_low, ci_high, practical_threshold) -> str:
    if delta is None or np.isnan(delta):
        return "INVALID"
    if delta <= 0:
        return "NOT_CONFIRMED"
    holm_pass = holm_p is not None and holm_p <= 0.05
    ci_excludes_zero = ci_low > 0
    practical = delta >= practical_threshold
    if holm_pass and ci_excludes_zero and practical:
        return "CONFIRMED"
    if holm_pass:
        return "DIRECTIONALLY_SUPPORTED"
    return "NOT_CONFIRMED"


def main() -> int:
    protocol = yaml.safe_load(PROTOCOL_YAML.read_text())["protocol"]
    practical_threshold = float(protocol["practical_threshold"]["minimum_delta_for_positive_claim"])

    print("Reading Eyecandies test labels (one-time authorised label read)...", flush=True)
    labels = _read_eyecandies_test_labels()
    print(
        f"  loaded {len(labels)} test sample labels  "
        f"({sum(labels.values())} anomalous, {len(labels) - sum(labels.values())} normal)"
    )

    # Compute per-endpoint primary metrics
    rows = []
    raw_p_map = {}
    for endpoint in PRIMARY_ENDPOINTS:
        static_per_seed = _load_per_seed_predictions(endpoint, "static_attention")
        rga_per_seed = _load_per_seed_predictions(endpoint, "base_RGA")
        if not static_per_seed or not rga_per_seed:
            print(f"[{endpoint}] missing predictions; skipping")
            continue
        sids_s, lbl_s, scores_static = _stack_ensemble(static_per_seed, labels)
        sids_r, lbl_r, scores_rga = _stack_ensemble(rga_per_seed, labels)
        # Sanity check alignment
        if not np.array_equal(sids_s, sids_r) or not np.array_equal(lbl_s, lbl_r):
            raise SystemExit(f"{endpoint}: static/RGA sample alignment mismatch")
        sample_ids = sids_s
        lbl = lbl_s
        res = audited_analysis(
            cell_id=endpoint,
            benchmark="Eyecandies-1.0.3",
            protocol=protocol["name"],
            rga_method="base_RGA",
            comparator_method="static_attention",
            sample_ids=sample_ids,
            labels=lbl,
            per_seed_rga_scores=scores_rga,
            per_seed_comp_scores=scores_static,
        )
        rows.append(
            {
                "endpoint": endpoint,
                "n_seeds": res.n_seeds,
                "n_test_samples": res.n_test_samples,
                "ensemble_static_auc": res.ensemble_comparator_auc,
                "ensemble_rga_auc": res.ensemble_rga_auc,
                "ensemble_delta_auc": res.ensemble_delta_auc,
                "per_seed_mean_delta": float(np.mean(res.per_seed_deltas)),
                "per_seed_sd_delta": float(np.std(res.per_seed_deltas, ddof=1)) if res.n_seeds > 1 else 0.0,
                "sign_consistent_seeds": res.sign_consistent_seeds,
                "delong_p_raw": res.delong_p_value,
                "bootstrap_ci_low": res.bootstrap_ci_low,
                "bootstrap_ci_high": res.bootstrap_ci_high,
                "practical_effect_band": res.practical_effect_band,
            }
        )
        raw_p_map[endpoint] = res.delong_p_value

    if not rows:
        print("no primary predictions found; aborting")
        return 1

    # Holm K=2 across primary endpoints
    holm = holm_bonferroni(raw_p_map, K=2)
    for r in rows:
        r["delong_p_holm_k2"] = holm[r["endpoint"]]
        r["decision"] = _decision(
            r["ensemble_delta_auc"],
            r["delong_p_holm_k2"],
            r["bootstrap_ci_low"],
            r["bootstrap_ci_high"],
            practical_threshold,
        )

    # Family decision per the locked policy
    confirmed = sum(1 for r in rows if r["decision"] == "CONFIRMED")
    if confirmed == len(PRIMARY_ENDPOINTS):
        family_decision = "FAMILY_D_V2_CONFIRMED_BOTH_ENDPOINTS"
    elif confirmed >= 1:
        family_decision = "FAMILY_D_V2_PARTIAL_CONFIRMATION"
    elif any(r["decision"] == "INVALID" for r in rows):
        family_decision = "FAMILY_D_V2_INVALID"
    else:
        family_decision = "FAMILY_D_V2_NOT_CONFIRMED"

    # Emit primary inference CSV
    primary_csv = OUT_DIR / "family_d_v2_primary_inference.csv"
    with primary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {primary_csv}")
    holm_csv = OUT_DIR / "family_d_v2_holm_k2.csv"
    with holm_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "endpoint",
                "ensemble_delta_auc",
                "delong_p_raw",
                "delong_p_holm_k2",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "practical_effect_band",
                "decision",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"wrote {holm_csv}")

    # Secondary descriptive D-EYE-3 if predictions exist
    sec_rows = []
    for endpoint in SECONDARY_DESCRIPTIVE:
        static_per_seed = _load_per_seed_predictions(endpoint, "static_attention")
        rga_per_seed = _load_per_seed_predictions(endpoint, "base_RGA")
        if not static_per_seed or not rga_per_seed:
            continue
        sids_s, lbl_s, scores_static = _stack_ensemble(static_per_seed, labels)
        sids_r, lbl_r, scores_rga = _stack_ensemble(rga_per_seed, labels)
        res = audited_analysis(
            cell_id=endpoint,
            benchmark="Eyecandies-1.0.3",
            protocol=protocol["name"],
            rga_method="base_RGA",
            comparator_method="static_attention",
            sample_ids=sids_s,
            labels=lbl_s,
            per_seed_rga_scores=scores_rga,
            per_seed_comp_scores=scores_static,
        )
        sec_rows.append(
            {
                "endpoint": endpoint,
                "n_seeds": res.n_seeds,
                "n_test_samples": res.n_test_samples,
                "ensemble_static_auc": res.ensemble_comparator_auc,
                "ensemble_rga_auc": res.ensemble_rga_auc,
                "ensemble_delta_auc": res.ensemble_delta_auc,
                "delong_p_raw": res.delong_p_value,
                "bootstrap_ci_low": res.bootstrap_ci_low,
                "bootstrap_ci_high": res.bootstrap_ci_high,
                "practical_effect_band": res.practical_effect_band,
                "note": "descriptive only; NOT in Holm K=2 family",
            }
        )
    if sec_rows:
        sec_csv = OUT_DIR / "family_d_v2_secondary_descriptive.csv"
        with sec_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sec_rows[0].keys()))
            w.writeheader()
            for r in sec_rows:
                w.writerow(r)
        print(f"wrote {sec_csv}")

    print()
    print("=== Family-D v2 primary results ===")
    for r in rows:
        print(
            f"{r['endpoint']}: static={r['ensemble_static_auc']:.4f}  rga={r['ensemble_rga_auc']:.4f}  "
            f"Δ={r['ensemble_delta_auc']:+.4f}  CI=[{r['bootstrap_ci_low']:+.4f},{r['bootstrap_ci_high']:+.4f}]  "
            f"Holm K=2 p={r['delong_p_holm_k2']:.3g}  band={r['practical_effect_band']}  → {r['decision']}"
        )
    print(f"\nFamily decision: {family_decision}")

    # Save family decision
    with (OUT_DIR / "family_d_v2_family_decision.txt").open("w") as f:
        f.write(family_decision + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
