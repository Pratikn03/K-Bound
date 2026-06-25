"""kbound_edge.metrics -- OFFLINE evaluation of decisions against true benefit.

These metrics are computed OFFLINE, on held-out windows where ground truth is
available -- they are NOT part of the online path.  They take the decisions a
policy made plus the true per-window benefit ``B`` (= accuracy of the adapted
candidate minus accuracy of the frozen model) and the per-window latency.

Action semantics (realised benefit of each decision)
----------------------------------------------------
    adapt   -> the candidate is used  -> realised benefit = B
    freeze  -> keep frozen            -> realised benefit = 0
    abstain -> keep frozen            -> realised benefit = 0

The oracle would adapt iff B > 0, so oracle benefit = max(B, 0) and

    regret_i = max(B_i, 0) - realised_i  >= 0.

Reported quantities
-------------------
    mean_regret                 average regret over all windows (lower=better)
    false_adapt_uncond          P(decision = adapt AND B < 0)        over ALL windows
    false_adapt_cond            P(B < 0 | decision = adapt)          over ADAPT windows
    adapt_rate / freeze_rate / abstain_rate
    mean_realised_benefit
    latency_ms_mean / latency_ms_p95
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def realised_benefit(decision: str, B: float) -> float:
    """Benefit actually obtained by following ``decision`` given true benefit B."""
    return float(B) if decision == "adapt" else 0.0


def evaluate(
    decisions: Sequence[str],
    true_benefits: Sequence[float],
    latencies_ms: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Compute the offline metric suite for one policy.

    Parameters
    ----------
    decisions : sequence of str
        Per-window 'adapt'/'freeze'/'abstain'.
    true_benefits : sequence of float
        Per-window true benefit B (uses offline labels).
    latencies_ms : sequence of float, optional
        Per-window decision latency in milliseconds.
    """
    decisions = list(decisions)
    B = np.asarray(true_benefits, dtype=float)
    n = len(decisions)
    if n == 0 or len(B) != n:
        raise ValueError("decisions and true_benefits must be equal, non-zero length")

    realised = np.array([realised_benefit(d, b) for d, b in zip(decisions, B)])
    oracle = np.maximum(B, 0.0)
    regret = oracle - realised

    is_adapt = np.array([d == "adapt" for d in decisions])
    is_freeze = np.array([d == "freeze" for d in decisions])
    is_abstain = np.array([d == "abstain" for d in decisions])

    harmful_adapt = is_adapt & (B < 0.0)
    n_adapt = int(is_adapt.sum())

    out = {
        "n_windows": n,
        "mean_regret": float(regret.mean()),
        "total_regret": float(regret.sum()),
        "mean_realised_benefit": float(realised.mean()),
        "mean_true_benefit": float(B.mean()),
        "false_adapt_uncond": float(harmful_adapt.mean()),
        "false_adapt_cond": float(harmful_adapt.sum() / n_adapt) if n_adapt > 0 else 0.0,
        "adapt_rate": float(is_adapt.mean()),
        "freeze_rate": float(is_freeze.mean()),
        "abstain_rate": float(is_abstain.mean()),
    }
    if latencies_ms is not None and len(latencies_ms) == n:
        lat = np.asarray(latencies_ms, dtype=float)
        out["latency_ms_mean"] = float(lat.mean())
        out["latency_ms_p95"] = float(np.percentile(lat, 95))
        out["latency_ms_max"] = float(lat.max())
    return out


def policy_comparison(
    policy_decisions: Dict[str, Sequence[str]],
    true_benefits: Sequence[float],
    latencies_ms: Sequence[float] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate several policies on the SAME windows -> name -> metrics dict."""
    return {
        name: evaluate(decs, true_benefits, latencies_ms)
        for name, decs in policy_decisions.items()
    }


def format_comparison_table(comparison: Dict[str, Dict[str, float]]) -> str:
    """Render a policy-comparison dict as a fixed-width text table."""
    cols = [
        ("policy", "{:<16}"),
        ("mean_regret", "{:>11}"),
        ("false_adapt_uncond", "{:>18}"),
        ("false_adapt_cond", "{:>16}"),
        ("adapt_rate", "{:>10}"),
        ("freeze_rate", "{:>11}"),
        ("abstain_rate", "{:>12}"),
    ]
    header = "  ".join(fmt.format(name) for name, fmt in cols)
    lines = [header, "-" * len(header)]
    for name, m in comparison.items():
        row = [
            "{:<16}".format(name),
            "{:>11.4f}".format(m["mean_regret"]),
            "{:>18.4f}".format(m["false_adapt_uncond"]),
            "{:>16.4f}".format(m["false_adapt_cond"]),
            "{:>10.3f}".format(m["adapt_rate"]),
            "{:>11.3f}".format(m["freeze_rate"]),
            "{:>12.3f}".format(m["abstain_rate"]),
        ]
        lines.append("  ".join(row))
    return "\n".join(lines)


def emitted_predictions(decision: str, p0, pa):
    """Return pa if decision is adapt, else p0."""
    if decision == "adapt":
        return pa
    return p0


def bootstrap_real_metrics(
    outcomes: List[Any],
    true_labels: List[np.ndarray],
    window_metadata: List[Dict[str, str]],
    policy_decisions: Dict[str, List[str]],
    latencies_ms: List[float],
    n_boot: int = 2000,
    seed: int = 20260619,
) -> Dict[str, Dict[str, Any]]:
    """Compute point estimates and 95% bootstrap confidence intervals for all policies."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    n = len(outcomes)
    policies = list(policy_decisions.keys())
    
    window_blocks = []
    for meta in window_metadata:
        window_blocks.append((meta["session_id"], meta["object_id"], meta["shift_id"]))
        
    unique_blocks = list(sorted(set(window_blocks)))
    n_blocks = len(unique_blocks)
    
    block_to_indices = {b: [] for b in unique_blocks}
    for idx, b in enumerate(window_blocks):
        block_to_indices[b].append(idx)
        
    def calc_metrics_for_indices(indices_list: List[int]) -> Dict[str, Dict[str, float]]:
        res = {p: {} for p in policies}
        
        for p in policies:
            preds_all = []
            labels_all = []
            regrets = []
            lats = []
            adapt_cnt = 0
            abstain_cnt = 0
            freeze_cnt = 0
            false_adapt_cnt = 0
            
            for idx in indices_list:
                o = outcomes[idx]
                dec = policy_decisions[p][idx]
                lbls = true_labels[idx]
                
                p_frame = emitted_predictions(dec, o.p0, o.pa).argmax(1)
                preds_all.extend(p_frame.tolist())
                labels_all.extend(lbls.tolist())
                
                froz_acc = float((o.p0.argmax(1) == lbls).mean())
                cand_acc = float((o.pa.argmax(1) == lbls).mean())
                B = cand_acc - froz_acc
                oracle = max(B, 0.0)
                realised = B if dec == "adapt" else 0.0
                regrets.append(oracle - realised)
                
                if dec == "adapt":
                    adapt_cnt += 1
                    if B < 0.0:
                        false_adapt_cnt += 1
                elif dec == "abstain":
                    abstain_cnt += 1
                else:
                    freeze_cnt += 1
                lats.append(latencies_ms[idx])
                
            n_wind = len(indices_list)
            res[p]["balanced_acc"] = float(balanced_accuracy_score(labels_all, preds_all)) if len(labels_all) > 0 else 0.0
            res[p]["macro_f1"] = float(f1_score(labels_all, preds_all, average="macro")) if len(labels_all) > 0 else 0.0
            res[p]["mean_regret"] = float(np.mean(regrets)) if regrets else 0.0
            res[p]["false_adapt_uncond"] = float(false_adapt_cnt / n_wind)
            res[p]["false_adapt_cond"] = float(false_adapt_cnt / adapt_cnt) if adapt_cnt > 0 else 0.0
            res[p]["adapt_rate"] = float(adapt_cnt / n_wind)
            res[p]["abstain_rate"] = float(abstain_cnt / n_wind)
            res[p]["mean_latency"] = float(np.mean(lats)) if lats else 0.0
            
        return res

    point_indices = list(range(n))
    point_est = calc_metrics_for_indices(point_indices)
    
    rng = np.random.default_rng(seed)
    
    bootstrap_results = {p: {m: [] for m in [
        "balanced_acc", "macro_f1", "mean_regret", "false_adapt_uncond", 
        "false_adapt_cond", "adapt_rate", "abstain_rate", "mean_latency"
    ]} for p in policies}
    
    for _ in range(n_boot):
        boot_block_indices = rng.choice(n_blocks, size=n_blocks, replace=True)
        boot_indices = []
        for idx in boot_block_indices:
            b = unique_blocks[idx]
            boot_indices.extend(block_to_indices[b])
            
        if not boot_indices:
            continue
            
        boot_metrics = calc_metrics_for_indices(boot_indices)
        for p in policies:
            for m in bootstrap_results[p].keys():
                bootstrap_results[p][m].append(boot_metrics[p][m])
                
    compiled = {}
    for p in policies:
        compiled[p] = {}
        for m in bootstrap_results[p].keys():
            val = point_est[p][m]
            arr = np.asarray(bootstrap_results[p][m])
            lower = float(np.quantile(arr, 0.025))
            upper = float(np.quantile(arr, 0.975))
            compiled[p][m] = {
                "val": val,
                "ci": [lower, upper]
            }
            
    return compiled


