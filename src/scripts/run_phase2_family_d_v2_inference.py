"""Phase 2.2E (Family-D v3) — DeLong + Holm–Bonferroni inference.

Reads per-seed archived predictions for D-EYE-1 and D-EYE-2, computes:
  - Per-seed ROC-AUC (static_attention vs base_RGA)
  - Seed-averaged ensemble prediction vectors
  - DeLong paired test on ensemble vectors
  - Paired test-sample bootstrap CI on Δ(AUC), 10 000 iterations
  - Holm–Bonferroni K=2 correction across {D-EYE-1, D-EYE-2}
  - Family decision per frozen statistical policy

Outputs:
  experiments/phase2/family_d/family_d_v2_primary_inference.csv
  experiments/phase2/family_d/family_d_v2_holm_k2.csv
  docs/research/phase2/FAMILY_D_V3_INFERENCE_REPORT.md

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_family_d_v2_inference.py \\
      --hypotheses docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv \\
      --policy docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "experiments" / "phase2" / "family_d"
DOCS_DIR = ROOT / "docs" / "research" / "phase2"

# Frozen inference parameters (mirrors FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md)
BOOTSTRAP_ITER = 10_000
BOOTSTRAP_SEED = 0
CI_ALPHA = 0.05
MULTIPLICITY_K = 2
MIN_PRACTICAL_DELTA = 0.010
MIN_SEEDS = 15


# ─── DeLong AUC + variance ────────────────────────────────────────────────────


def _kernel(x, y):
    """Kernel for DeLong: P(X > Y) + 0.5 * P(X == Y)."""
    if x > y:
        return 1.0
    elif x == y:
        return 0.5
    return 0.0


def _delong_auc_variance(labels: np.ndarray, scores: np.ndarray):
    """Compute AUC and its variance using DeLong 1988.

    Returns (auc, var_auc).
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    # Placement values
    pv_pos = np.array([np.mean([_kernel(p, n) for n in neg]) for p in pos])
    pv_neg = np.array([np.mean([_kernel(p, n) for p in pos]) for n in neg])
    auc = float(pv_pos.mean())

    # Structural components (DeLong eq. 3).
    # var(AUC) = var(V10) / m + var(V01) / n where m=#positives, n=#negatives.
    # Do not divide twice by class counts.
    var_pos = float(np.var(pv_pos, ddof=1)) if n_pos > 1 else 0.0
    var_neg = float(np.var(pv_neg, ddof=1)) if n_neg > 1 else 0.0
    var = (var_pos / n_pos) + (var_neg / n_neg)
    return auc, var


def _delong_paired_test(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[float, float, float]:
    """DeLong paired test between two classifiers on the same test set.

    Returns (delta_auc, z_stat, p_value).
    delta_auc = AUC(A) - AUC(B).
    """
    auc_a, var_a = _delong_auc_variance(labels, scores_a)
    auc_b, var_b = _delong_auc_variance(labels, scores_b)

    if any(math.isnan(v) for v in [auc_a, auc_b, var_a, var_b]):
        return float("nan"), float("nan"), float("nan")

    # Covariance term via placement values
    pos = labels == 1
    neg = labels == 0
    pv_a_pos = np.array(
        [np.mean([_kernel(scores_a[i], scores_a[j]) for j in np.where(neg)[0]]) for i in np.where(pos)[0]]
    )
    pv_b_pos = np.array(
        [np.mean([_kernel(scores_b[i], scores_b[j]) for j in np.where(neg)[0]]) for i in np.where(pos)[0]]
    )
    pv_a_neg = np.array(
        [np.mean([_kernel(scores_a[j], scores_a[i]) for j in np.where(pos)[0]]) for i in np.where(neg)[0]]
    )
    pv_b_neg = np.array(
        [np.mean([_kernel(scores_b[j], scores_b[i]) for j in np.where(pos)[0]]) for i in np.where(neg)[0]]
    )

    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    cov_pos = float(np.cov(pv_a_pos, pv_b_pos)[0, 1]) / n_pos if n_pos > 1 else 0.0
    cov_neg = float(np.cov(pv_a_neg, pv_b_neg)[0, 1]) / n_neg if n_neg > 1 else 0.0
    var_diff = var_a + var_b - 2 * (cov_pos + cov_neg)
    if var_diff <= 0.0:
        # Degenerate paired variance indicates unstable DeLong covariance in
        # this sample; report as unavailable instead of forcing extreme z.
        return auc_a - auc_b, float("nan"), float("nan")

    delta = auc_a - auc_b
    z = delta / math.sqrt(var_diff)
    # Two-tailed normal p-value
    from scipy.stats import norm  # type: ignore[import]

    p = 2.0 * float(norm.sf(abs(z)))
    return delta, z, p


def _bootstrap_delta_ci(
    labels: np.ndarray,
    scores_rga: np.ndarray,
    scores_static: np.ndarray,
    n_iter: int = BOOTSTRAP_ITER,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = CI_ALPHA,
) -> tuple[float, float, float]:
    """Paired test-sample bootstrap CI on Δ(AUC) = AUC(RGA) - AUC(static).

    Returns (point_estimate, ci_lower, ci_upper).
    """
    from sklearn.metrics import roc_auc_score  # type: ignore[import]

    rng = np.random.RandomState(seed)
    n = len(labels)
    deltas = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        lbl = labels[idx]
        if len(np.unique(lbl)) < 2:
            continue
        auc_r = roc_auc_score(lbl, scores_rga[idx])
        auc_s = roc_auc_score(lbl, scores_static[idx])
        deltas.append(auc_r - auc_s)

    if not deltas:
        return float("nan"), float("nan"), float("nan")

    deltas_arr = np.array(deltas)
    point = float(deltas_arr.mean())
    lo = float(np.percentile(deltas_arr, 100 * alpha / 2))
    hi = float(np.percentile(deltas_arr, 100 * (1 - alpha / 2)))
    return point, lo, hi


def _holm_bonferroni(p_values: list[float], alpha: float = CI_ALPHA) -> list[bool]:
    """Holm–Bonferroni correction. Returns list of booleans (True = reject H0)."""
    k = len(p_values)
    order = sorted(range(k), key=lambda i: p_values[i])
    rejected = [False] * k
    for rank, idx in enumerate(order):
        threshold = alpha / (k - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            # Once we fail, stop (Holm step-down)
            break
    return rejected


def _load_per_seed_csv(cell_id: str) -> list[dict]:
    """Load per-seed evaluation CSV for a cell."""
    fname = OUT_DIR / f"family_d_{cell_id.replace('-','_').lower()}_full_test_evaluation_per_seed.csv"
    if not fname.exists():
        return []
    with fname.open() as f:
        return list(csv.DictReader(f))


def _load_archive_scores(cell_id: str) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Load archived prediction arrays grouped by (method, seed).

    Returns: {condition: {seed: {"scores": arr, "labels": arr}}}
    """

    archive_dir = OUT_DIR / "archives" / f"family_d_{cell_id.replace('-','_').lower()}"
    if not archive_dir.exists():
        return {}

    result: dict[str, dict[int, dict[str, Any]]] = {}
    for parquet_file in sorted(archive_dir.rglob("*.parquet")):
        try:
            import pandas as pd  # type: ignore[import]

            df = pd.read_parquet(parquet_file)
        except Exception:
            continue
        for (method, seed, variant), grp in df.groupby(["method", "seed", "method_variant"]):
            key = str(variant)
            result.setdefault(key, {})
            result[key].setdefault(int(seed), {"static_scores": [], "rga_scores": [], "labels": []})
            labels = grp["label"].values
            scores = grp["raw_score"].values
            if "static" in str(method):
                result[key][int(seed)]["static_scores"] = scores
                result[key][int(seed)]["labels"] = labels
            elif "rga" in str(method):
                result[key][int(seed)]["rga_scores"] = scores
    return result


def _infer_cell(cell_id: str) -> dict[str, Any]:
    """Compute all inference quantities for one primary cell."""

    per_seed_rows = _load_per_seed_csv(cell_id)
    test_rows = [r for r in per_seed_rows if r.get("fold") == "test"]

    if len(test_rows) < MIN_SEEDS:
        return {
            "cell_id": cell_id,
            "status": f"INSUFFICIENT_SEEDS (found {len(test_rows)}, min={MIN_SEEDS})",
            "outcome": "NOT_CONFIRMED",
        }

    # Per-seed delta AUC from CSV
    per_seed_deltas = []
    per_seed_static_aucs = []
    per_seed_rga_aucs = []
    for r in test_rows:
        try:
            delta = float(r["delta_auc"]) if r.get("delta_auc") not in ("", None, "None") else None
            static_auc = float(r["static_auc"]) if r.get("static_auc") not in ("", None, "None") else None
            rga_auc = float(r["rga_auc"]) if r.get("rga_auc") not in ("", None, "None") else None
            if delta is not None:
                per_seed_deltas.append(delta)
            if static_auc is not None:
                per_seed_static_aucs.append(static_auc)
            if rga_auc is not None:
                per_seed_rga_aucs.append(rga_auc)
        except (ValueError, KeyError):
            continue

    mean_delta = float(np.mean(per_seed_deltas)) if per_seed_deltas else float("nan")
    mean_static_auc = float(np.mean(per_seed_static_aucs)) if per_seed_static_aucs else float("nan")
    mean_rga_auc = float(np.mean(per_seed_rga_aucs)) if per_seed_rga_aucs else float("nan")

    # Load archived predictions for DeLong + bootstrap
    archive_scores = _load_archive_scores(cell_id)
    degraded_condition = cell_id  # archive stores condition = cell_id

    if degraded_condition not in archive_scores:
        # Fall back: use per-seed CSV deltas only; skip DeLong
        return {
            "cell_id": cell_id,
            "n_seeds": len(test_rows),
            "mean_delta_auc": mean_delta,
            "mean_static_auc": mean_static_auc,
            "mean_rga_auc": mean_rga_auc,
            "delong_p": float("nan"),
            "bootstrap_delta": mean_delta,
            "bootstrap_ci_lo": float("nan"),
            "bootstrap_ci_hi": float("nan"),
            "status": "ARCHIVE_NOT_FOUND_USING_CSV_DELTAS",
            "outcome": _classify_outcome(mean_delta, float("nan"), float("nan"), float("nan")),
        }

    # Build ensemble vectors (average per-sample scores across seeds)
    all_seeds = sorted(archive_scores[degraded_condition].keys())
    valid_seeds = [
        s
        for s in all_seeds
        if (
            len(archive_scores[degraded_condition][s].get("static_scores", [])) > 0
            and len(archive_scores[degraded_condition][s].get("rga_scores", [])) > 0
        )
    ]

    if len(valid_seeds) < MIN_SEEDS:
        return {
            "cell_id": cell_id,
            "n_seeds": len(valid_seeds),
            "status": f"INSUFFICIENT_ARCHIVE_SEEDS ({len(valid_seeds)} < {MIN_SEEDS})",
            "outcome": "NOT_CONFIRMED",
        }

    # Use labels from first valid seed (all seeds use same test set)
    first_seed = valid_seeds[0]
    labels_arr = np.asarray(archive_scores[degraded_condition][first_seed]["labels"])

    # Ensemble = mean across seeds
    static_ensemble = np.mean([archive_scores[degraded_condition][s]["static_scores"] for s in valid_seeds], axis=0)
    rga_ensemble = np.mean([archive_scores[degraded_condition][s]["rga_scores"] for s in valid_seeds], axis=0)

    # DeLong paired test
    try:
        delta_delong, z_stat, delong_p = _delong_paired_test(labels_arr, rga_ensemble, static_ensemble)
    except Exception as exc:
        print(f"[{cell_id}] DeLong failed: {exc}", flush=True)
        delta_delong, z_stat, delong_p = float("nan"), float("nan"), float("nan")

    # Bootstrap CI
    try:
        bs_point, bs_lo, bs_hi = _bootstrap_delta_ci(labels_arr, rga_ensemble, static_ensemble)
    except Exception as exc:
        print(f"[{cell_id}] Bootstrap failed: {exc}", flush=True)
        bs_point, bs_lo, bs_hi = float("nan"), float("nan"), float("nan")

    outcome = _classify_outcome(delta_delong, delong_p, bs_lo, bs_hi)

    return {
        "cell_id": cell_id,
        "n_seeds": len(valid_seeds),
        "mean_per_seed_delta_auc": mean_delta,
        "mean_static_auc": mean_static_auc,
        "mean_rga_auc": mean_rga_auc,
        "ensemble_delta_auc": delta_delong,
        "delong_z": z_stat,
        "delong_p": delong_p,
        "bootstrap_delta": bs_point,
        "bootstrap_ci_lo": bs_lo,
        "bootstrap_ci_hi": bs_hi,
        "minimum_practical_delta": MIN_PRACTICAL_DELTA,
        "status": "OK",
        "outcome_pre_holm": outcome,
    }


def _classify_outcome(delta: float, p: float, ci_lo: float, ci_hi: float) -> str:
    """Classify per-cell outcome per frozen statistical policy."""
    if any(math.isnan(v) for v in [delta, p]):
        return "INVALID_OR_INSUFFICIENT_DATA"
    if delta <= 0:
        return "NOT_CONFIRMED"
    if p > CI_ALPHA:
        return "NOT_CONFIRMED"
    if not math.isnan(ci_lo) and ci_lo <= 0:
        return "DIRECTIONALLY_SUPPORTED"
    if abs(delta) < MIN_PRACTICAL_DELTA:
        return "DIRECTIONALLY_SUPPORTED"
    return "CONFIRMED"


def _validate_post_run_consistency(cell_results: list[dict], holm_results: list[dict]) -> list[str]:
    """Cross-check DeLong, bootstrap CI, and Holm outcomes.

    Returns a list of human-readable inconsistency messages.
    Empty list means checks passed.
    """
    issues: list[str] = []

    holm_by_cell = {str(r.get("cell_id")): r for r in holm_results}

    for cr in cell_results:
        cell_id = str(cr.get("cell_id", "UNKNOWN"))
        hr = holm_by_cell.get(cell_id, {})

        p = float(cr.get("delong_p", float("nan")))
        delta = float(cr.get("ensemble_delta_auc", float("nan")))
        ci_lo = float(cr.get("bootstrap_ci_lo", float("nan")))
        ci_hi = float(cr.get("bootstrap_ci_hi", float("nan")))
        pre = str(cr.get("outcome_pre_holm", ""))
        holm_out = str(hr.get("holm_outcome", ""))
        rejected = bool(hr.get("h0_rejected", False))

        ci_available = not (math.isnan(ci_lo) or math.isnan(ci_hi))
        ci_excludes_zero = ci_available and (ci_lo > 0 or ci_hi < 0)
        ci_includes_zero = ci_available and (ci_lo <= 0 <= ci_hi)
        p_significant = (not math.isnan(p)) and (p <= CI_ALPHA)
        practically_positive = (not math.isnan(delta)) and (delta > 0) and (abs(delta) >= MIN_PRACTICAL_DELTA)

        # 1) Direct DeLong-vs-bootstrap contradiction.
        if p_significant and ci_includes_zero:
            issues.append(
                f"{cell_id}: DeLong p={p:.6f} is significant but bootstrap CI includes 0 "
                f"([{ci_lo:+.6f}, {ci_hi:+.6f}])."
            )

        # 2) Pre-Holm confirmed claims must satisfy significance + CI + practical + direction.
        if pre == "CONFIRMED":
            if (not p_significant) or (not ci_excludes_zero) or (not practically_positive):
                issues.append(
                    f"{cell_id}: outcome_pre_holm=CONFIRMED without required evidence "
                    f"(p_significant={p_significant}, ci_excludes_zero={ci_excludes_zero}, "
                    f"practically_positive={practically_positive})."
                )

        # 3) Holm outcome must align with reject flag and pre-Holm state.
        if rejected and holm_out != pre:
            issues.append(
                f"{cell_id}: Holm rejected H0 but holm_outcome ({holm_out}) does not match "
                f"outcome_pre_holm ({pre})."
            )
        if (not rejected) and holm_out != "NOT_CONFIRMED":
            issues.append(
                f"{cell_id}: Holm did not reject H0 but holm_outcome is {holm_out} "
                "instead of NOT_CONFIRMED."
            )

        # 4) Holm-confirmed cell cannot carry a CI spanning zero.
        if holm_out == "CONFIRMED" and ci_includes_zero:
            issues.append(
                f"{cell_id}: holm_outcome=CONFIRMED but bootstrap CI includes 0 "
                f"([{ci_lo:+.6f}, {ci_hi:+.6f}])."
            )

    return issues


def _write_markdown_report(cell_results: list[dict], holm_results: list[dict], family_decision: str) -> Path:
    """Write the final inference report markdown."""
    out = DOCS_DIR / "FAMILY_D_V3_INFERENCE_REPORT.md"
    lines = [
        "# Family-D v3 — Inference Report",
        "",
        "**Phase:** 2.2E / Stage 4",
        f"**Family decision:** `{family_decision}`",
        "",
        "## 1. Per-cell primary inference",
        "",
        "| Cell | n_seeds | Δ(AUC) ensemble | DeLong p | Bootstrap CI (95%) | Pre-Holm outcome | Holm-adjusted |",
        "|---|---|---|---|---|---|---|",
    ]
    for cr, hr in zip(cell_results, holm_results):
        delta = cr.get("ensemble_delta_auc", cr.get("mean_delta_auc", float("nan")))
        p = cr.get("delong_p", float("nan"))
        lo = cr.get("bootstrap_ci_lo", float("nan"))
        hi = cr.get("bootstrap_ci_hi", float("nan"))
        pre_holm = cr.get("outcome_pre_holm", cr.get("outcome", "—"))
        holm_out = hr.get("holm_outcome", "—")
        ci_str = f"[{lo:+.4f}, {hi:+.4f}]" if not (math.isnan(lo) or math.isnan(hi)) else "N/A"
        p_str = f"{p:.4f}" if not math.isnan(p) else "N/A"
        delta_str = f"{delta:+.4f}" if not math.isnan(delta) else "N/A"
        lines.append(
            f"| {cr['cell_id']} | {cr.get('n_seeds','—')} | {delta_str} | {p_str} "
            f"| {ci_str} | {pre_holm} | {holm_out} |"
        )

    lines += [
        "",
        "## 2. Holm–Bonferroni correction (K=2)",
        "",
        "| Step | Cell | p-value | Threshold (α/k) | Reject H0 |",
        "|---|---|---|---|---|",
    ]
    for i, hr in enumerate(sorted(holm_results, key=lambda r: r.get("delong_p", 1.0))):
        thresh = CI_ALPHA / (MULTIPLICITY_K - i)
        lines.append(
            f"| {i+1} | {hr['cell_id']} | {hr.get('delong_p', float('nan')):.4f} "
            f"| {thresh:.4f} | {hr.get('h0_rejected', False)} |"
        )

    lines += [
        "",
        "## 3. Family decision",
        "",
        f"> **`{family_decision}`**",
        "",
        "### Claim ceiling (per frozen selection policy §7)",
        "",
    ]
    if "CONFIRMED_BOTH" in family_decision:
        lines.append(
            "> Held-out confirmatory evidence under the frozen Eyecandies RGB+depth "
            "one-class degradation-stress protocol for D-EYE-1 and D-EYE-2."
        )
    elif "PARTIAL" in family_decision:
        lines.append(
            "> Held-out confirmatory evidence for one of the two primary endpoints. "
            "The non-confirmed endpoint is reported as NOT_CONFIRMED."
        )
    else:
        lines.append(
            "> Held-out confirmation was not obtained for the evaluated endpoint(s); " "negative results are retained."
        )

    lines += [
        "",
        "### Forbidden claims (regardless of outcome)",
        "",
        "- ELARA is universal",
        "- ELARA is SOTA",
        "- ELARA is deployment-safe",
        "- Family A becomes confirmatory",
        "- RGA+ beats strongest baselines",
        "- Physical-AI safety validation",
        "- Raw-sensor corruption robustness",
        "",
        "## 4. Reproducibility",
        "",
        f"- Bootstrap: {BOOTSTRAP_ITER:,} iterations, seed={BOOTSTRAP_SEED}",
        f"- CI: {int((1-CI_ALPHA)*100)}% two-sided",
        f"- Multiplicity correction: Holm–Bonferroni K={MULTIPLICITY_K}",
        f"- Minimum practical delta: {MIN_PRACTICAL_DELTA}",
        f"- Minimum seeds for inference: {MIN_SEEDS}",
        "",
        "## 5. selection_used_test_metrics audit",
        "",
        "All per-seed rows must have `selection_used_test_metrics = False`. "
        "Any row with True invalidates the family.",
        "",
    ]

    out.write_text("\n".join(lines))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hypotheses", default="docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv")
    p.add_argument("--policy", default="docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md")
    args = p.parse_args()

    hyp_path = ROOT / args.hypotheses
    if not hyp_path.exists():
        raise SystemExit(f"Hypotheses CSV not found: {hyp_path}")

    # Check all per-seed CSVs for selection_used_test_metrics violations
    for cell in ("D-EYE-1", "D-EYE-2"):
        rows = _load_per_seed_csv(cell)
        violations = [
            r for r in rows if str(r.get("selection_used_test_metrics", "False")).lower() not in ("false", "0", "")
        ]
        if violations:
            raise SystemExit(
                f"INTEGRITY FAIL: {cell} has {len(violations)} rows with "
                f"selection_used_test_metrics != False. Family = INVALID."
            )

    # Compute per-cell inference
    cell_results = []
    for cell in ("D-EYE-1", "D-EYE-2"):
        print(f"[{cell}] computing inference...", flush=True)
        r = _infer_cell(cell)
        cell_results.append(r)
        print(f"  outcome_pre_holm={r.get('outcome_pre_holm', r.get('outcome'))}", flush=True)

    # Holm–Bonferroni
    p_values = [cr.get("delong_p", 1.0) for cr in cell_results]
    rejected = _holm_bonferroni(p_values)

    holm_results = []
    for cr, rej in zip(cell_results, rejected):
        pre = cr.get("outcome_pre_holm", cr.get("outcome", "NOT_CONFIRMED"))
        holm_out = pre if rej else "NOT_CONFIRMED"
        holm_results.append(
            {
                "cell_id": cr["cell_id"],
                "delong_p": cr.get("delong_p", float("nan")),
                "h0_rejected": rej,
                "holm_outcome": holm_out,
            }
        )

    # Family decision
    confirmed_cells = [hr["cell_id"] for hr in holm_results if hr["holm_outcome"] == "CONFIRMED"]
    if len(confirmed_cells) == 2:
        family_decision = "FAMILY_D_V3_CONFIRMED_BOTH_ENDPOINTS"
    elif len(confirmed_cells) == 1:
        family_decision = f"FAMILY_D_V3_PARTIAL_CONFIRMATION ({confirmed_cells[0]} confirmed)"
    else:
        family_decision = "FAMILY_D_V3_NOT_CONFIRMED"

    print(f"\nFamily decision: {family_decision}", flush=True)

    # Automated post-run validator: fail hard on statistical inconsistencies.
    consistency_issues = _validate_post_run_consistency(cell_results, holm_results)
    if consistency_issues:
        msg_lines = [
            "POST-RUN VALIDATOR FAIL: DeLong/Bootstrap/Holm inconsistency detected.",
            "Resolve these issues before treating the run as valid:",
        ]
        msg_lines.extend([f"- {m}" for m in consistency_issues])
        raise SystemExit("\n".join(msg_lines))
    print("[VALIDATOR] Post-run consistency checks passed.", flush=True)

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Primary inference CSV
    inf_csv = OUT_DIR / "family_d_v2_primary_inference.csv"
    all_inf_rows = []
    for cr, hr in zip(cell_results, holm_results):
        row = dict(cr)
        row.update(hr)
        row["family_decision"] = family_decision
        all_inf_rows.append(row)

    if all_inf_rows:
        fields = sorted({k for r in all_inf_rows for k in r.keys()}, key=lambda k: (k != "cell_id", k))
        with inf_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_inf_rows)
        print(f"Wrote {inf_csv}", flush=True)

    # Holm K=2 CSV
    holm_csv = OUT_DIR / "family_d_v2_holm_k2.csv"
    if holm_results:
        fields = sorted({k for r in holm_results for k in r.keys()}, key=lambda k: (k != "cell_id", k))
        fields = ["cell_id"] + [f for f in fields if f != "cell_id"]
        with holm_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(holm_results)
        print(f"Wrote {holm_csv}", flush=True)

    # Markdown report
    report_path = _write_markdown_report(cell_results, holm_results, family_decision)
    print(f"Wrote {report_path}", flush=True)

    # Update manifest
    manifest_path = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_PARTITION_MANIFEST_v3.json"
    existing = {}
    if manifest_path.exists():
        with manifest_path.open() as f:
            existing = json.load(f)
    existing["family_decision"] = family_decision
    existing["inference_complete"] = True
    with manifest_path.open("w") as f:
        json.dump(existing, f, indent=2)
    print(f"Updated manifest → {manifest_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
