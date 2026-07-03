"""
per_condition_serialize.py - torch-free per-condition serialization for the K-Bound
WILDS runners (Camelyon17, ImageNet-R, ...), matching the canonical schema produced by
docs/research/kbound/scripts/cifar_tent_mps_v2.py (the stress_grid_multiseed_v1 contract).

WHY THIS EXISTS
  run_camelyon17_kbound.py and run_imagenetr_kbound.py historically wrote a single
  monolithic result_*.json (records[] + conditions[] inline).  The multi-seed paired-CI
  analysis (experiments/kbound/results/stress_grid_multiseed_v1/_locked_analysis_script.py)
  expects, per (method, seed), a file:
      per_condition_<dataset>_<method>_seed<S>.json
  with one object per condition carrying: B, a0, a_adapted, regime, oracle_action, Z,
  Z_names, b_hat, eps_conformal, kga_decision, ...  This module produces exactly that,
  so the GPU multi-seed runs are directly consumable by the locked analysis machinery.

INTEGRITY
  - This module ONLY reshapes already-measured records into the per-condition schema and
    runs the single-candidate KGA certificate over them.  It fabricates nothing.
  - decide_benefit() prefers the REAL certificate analysis.decide_kga (sklearn
    GradientBoostingRegressor + split-conformal eps).  If sklearn is unavailable in the
    runtime (e.g. a torch-free CPU verification sandbox), it falls back to a clearly
    labelled numpy k-NN-in-Z benefit estimator with the SAME split-conformal radius and
    the SAME ADAPT/FREEZE/ABSTAIN decision rule, and stamps `kga_backend` accordingly so
    no consumer can mistake the fallback for the production estimator.

The function is intentionally pure-Python/numpy so it can be unit-tested and smoke-tested
without torch.
"""
from __future__ import annotations
import os
import json
import numpy as np

# Canonical evidence names (mirror tta_methods.EVIDENCE_NAMES; duplicated here so this
# module stays importable without torch).  Asserted equal to tta_methods.EVIDENCE_NAMES
# inside the runners at wiring time.
EVIDENCE_NAMES = [
    "pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf",
    "post_pbal", "pbal_drop", "entropy_drop", "frac_highconf",
    "marginal_KL", "update_norm",
]

ALPHA = 0.10


# --------------------------------------------------------------------------- #
# (a) single-candidate KGA benefit certificate                                #
# --------------------------------------------------------------------------- #
def _decide_kga_sklearn(Z, B, alpha):
    """Production path: identical machinery to analysis.decide_kga."""
    from sklearn.ensemble import GradientBoostingRegressor  # local import: optional dep
    Z = np.asarray(Z, float); B = np.asarray(B, float); N = len(B)
    Bhat = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        m = GradientBoostingRegressor(n_estimators=250, max_depth=2,
                                      learning_rate=0.05, subsample=0.8, random_state=0)
        m.fit(Z[tr], B[tr])
        Bhat[i] = m.predict(Z[i:i + 1])[0]
    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
    dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
    return Bhat, eps, np.asarray(dec)


def _decide_kga_numpy(Z, B, alpha, k=8):
    """sklearn-free fallback: leave-one-out k-NN-in-Z benefit estimate + the SAME
    split-conformal radius and the SAME decision rule as the production certificate.

    This is NOT the production gradient-boosted estimator; it exists so the
    serialization / aggregation / decision-rule plumbing can be exercised in a
    torch+sklearn-free sandbox.  Callers stamp `kga_backend="numpy_knn_fallback"`.
    """
    Z = np.asarray(Z, float); B = np.asarray(B, float); N = len(B)
    # standardize Z columns for a scale-free distance
    mu = Z.mean(0); sd = Z.std(0); sd[sd == 0] = 1.0
    Zs = (Z - mu) / sd
    Bhat = np.zeros(N)
    kk = max(1, min(k, N - 1))
    for i in range(N):
        d = np.sqrt(((Zs - Zs[i]) ** 2).sum(1))
        d[i] = np.inf
        nn = np.argsort(d)[:kk]
        Bhat[i] = float(np.mean(B[nn]))
    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
    dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
    return Bhat, eps, np.asarray(dec)


def decide_benefit(Z, B, alpha=ALPHA, prefer="auto"):
    """Single-candidate KGA benefit certificate over a (method, seed) cell.

    Returns (Bhat: np.ndarray, eps: float, dec: np.ndarray[str], backend: str).
    backend in {"sklearn_gradient_boost", "numpy_knn_fallback"}.

    prefer="auto" uses sklearn if importable, else the numpy fallback.
    prefer="numpy" forces the fallback (used by the CPU verification harness).
    prefer="sklearn" forces the production path (raises if sklearn missing).
    """
    Z = np.asarray(Z, float); B = np.asarray(B, float)
    if len(B) < 2 or len(np.unique(B)) < 2:
        # degenerate cell: no cross-condition variation to certify against.
        Bhat = np.full(len(B), float(np.mean(B)) if len(B) else 0.0)
        return Bhat, 0.0, np.full(len(B), "ABSTAIN"), "degenerate_no_variation"
    if prefer == "numpy":
        bh, eps, dec = _decide_kga_numpy(Z, B, alpha)
        return bh, eps, dec, "numpy_knn_fallback"
    if prefer == "sklearn":
        bh, eps, dec = _decide_kga_sklearn(Z, B, alpha)
        return bh, eps, dec, "sklearn_gradient_boost"
    # auto
    try:
        bh, eps, dec = _decide_kga_sklearn(Z, B, alpha)
        return bh, eps, dec, "sklearn_gradient_boost"
    except Exception:
        bh, eps, dec = _decide_kga_numpy(Z, B, alpha)
        return bh, eps, dec, "numpy_knn_fallback"


# --------------------------------------------------------------------------- #
# per-condition record assembly + file writing                                #
# --------------------------------------------------------------------------- #
def _condition_key_camelyon(r):
    """Stable, order-preserving, SEED-INDEPENDENT condition string for Camelyon17.

    The seed is a file-level field (one per_condition_*_seed<S>.json per seed); the
    multi-seed paired-CI analysis pairs conditions ACROSS seeds, so the condition key
    must be identical for the same cell across seeds and therefore must NOT embed the
    seed (mirrors the stress_grid contract, where `condition` carries severity/regime
    but not the seed).

    `mode` (online/episodic) IS included: the WILDS runners pool {online,episodic}
    under one method, so without it the two would collide to the same key and the
    paired-CI condition arrays would be ambiguous. Each (cell, mode) is a distinct
    condition; this keeps every key unique within a (method, seed) file."""
    mode = r.get("mode", "")
    base = f"{r['domain']}|{r['comp']}|{r['regime']}|{r['aggr']}"
    return f"{base}|{mode}" if mode else base


def _condition_key_imagenetr(r):
    mode = r.get("mode", "")
    base = f"{r.get('domain', 'imagenet_r')}|{r['comp']}|{r['regime']}|{r['aggr']}"
    return f"{base}|{mode}" if mode else base


CONDITION_KEYS = {
    "camelyon17": _condition_key_camelyon,
    "imagenet-r": _condition_key_imagenetr,
    "imagenet_r": _condition_key_imagenetr,
}


def build_per_condition_records(records, method, seed, dataset, alpha=ALPHA,
                                z_names=None, prefer="auto", condition_key_fn=None,
                                method_field="method"):
    """Build the per-condition record list for one (method, seed) cell.

    `records` is the runner's flat records[] list (each is a dict with keys
    seed, method, comp, regime, aggr, a0, aa, B, Z, regime_label, ...).
    `method_field` selects which record field names the "method" axis (default
    "method"; the ImageNet-R diverse-backbone panel passes "candidate" so each
    frozen backbone becomes its own per-condition file).
    Returns (per_cond_list, backend_label).
    """
    if condition_key_fn is None:
        condition_key_fn = CONDITION_KEYS.get(dataset, _condition_key_imagenetr)
    z_names = z_names or EVIDENCE_NAMES
    rs = [r for r in records if r.get(method_field) == method and int(r.get("seed", -1)) == int(seed)]
    # preserve runner order (this defines the cross-seed condition order the locked
    # analysis asserts must match across seeds)
    Z = np.array([r["Z"] for r in rs], float) if rs else np.zeros((0, len(z_names)))
    B = np.array([r["B"] for r in rs], float) if rs else np.zeros((0,))
    Bhat, eps, dec, backend = decide_benefit(Z, B, alpha=alpha, prefer=prefer)
    per_cond = []
    for i, r in enumerate(rs):
        b_hat_i = float(Bhat[i]); eps_i = float(eps)
        lb_i = b_hat_i - eps_i; ub_i = b_hat_i + eps_i
        if lb_i > 0:
            zone_i = "CERTIFIED_ADAPT"
        elif ub_i < 0:
            zone_i = "CERTIFIED_FREEZE"
        else:
            zone_i = "BLIND"
        a0_i = float(r["a0"]); aa_i = float(r["aa"])
        kga_acc_i = aa_i if dec[i] == "ADAPT" else a0_i
        per_cond.append({
            "seed": int(seed), "benchmark": dataset, "method": method,
            "condition": condition_key_fn(r),
            "B": float(r["B"]),
            "a0": a0_i,
            "a_adapted": aa_i,
            "a_kbound": float(kga_acc_i),
            "a_oracle": float(max(a0_i, aa_i)),
            "regime": r.get("regime_label", "marginal"),
            "oracle_action": ("ADAPT" if aa_i > a0_i else "FREEZE"),
            "Z": [float(z) for z in r["Z"]],
            "Z_names": list(z_names),
            # Wave-5 pass-through: the runners now attach the panel agreement
            # matrix + disagreement count (panel_capture.py). None if absent.
            "n_D": r.get("n_D"),
            "c_ij": r.get("c_ij"),
            "Z_ev2": r.get("Z_ev2"),
            "Z_ev2_names": r.get("Z_ev2_names"),
            "tau_hat": 0.0,                  # single-candidate route => degenerate
            "tau_star": 0.0,
            "b_hat": b_hat_i,
            "eps_conformal": eps_i,
            "benefit_ci": [float(lb_i), float(ub_i)],
            "zone": zone_i,
            "gamma_hat": float(0.5 * b_hat_i),
            "gamma_ci": [float(0.5 * lb_i), float(0.5 * ub_i)],
            "kga_decision": str(dec[i]),
        })
    return per_cond, backend


def _match_imagenetr_cell(r, c):
    return (int(r.get("seed", -1)) == int(c["seed"])
            and r.get("comp") == c["comp"]
            and r.get("regime") == c["regime"]
            and r.get("aggr") == c["aggr"])


def build_panel_records(records, conditions, dataset, candidate_order,
                        condition_key_fn=None):
    """One record per panel condition (diverse-backbone / multicandidate grid).

    Each row carries aa_all (anchor + candidates), shared c_ij/n_D, and Z from
    the best-scoring candidate (for the V3 fallback path).
    """
    if condition_key_fn is None:
        condition_key_fn = CONDITION_KEYS.get(dataset, _condition_key_imagenetr)
    rows = []
    for c in conditions:
        rs = [r for r in records if _match_imagenetr_cell(r, c)]
        if not rs:
            continue
        a0 = float(c.get("a0", rs[0]["a0"]))
        aa_all = [float(x) for x in c.get("aa_all", [a0])]
        if len(aa_all) < 2:
            by = {r.get("candidate", r.get("method")): r for r in rs}
            aa_all = [a0] + [float(by[n]["aa"]) for n in candidate_order if n in by]
        best_aa = float(max(aa_all))
        best_r = max(rs, key=lambda r: float(r["aa"]))
        key_r = rs[0]
        rows.append({
            "seed": int(c["seed"]),
            "condition": condition_key_fn(key_r),
            "a0": a0,
            "a_adapted": best_aa,
            "B": float(best_aa - a0),
            "aa_all": aa_all,
            "cand_names": list(c.get("cand_names", ["freeze_f0"] + list(candidate_order))),
            "Z": [float(z) for z in best_r["Z"]],
            "n_D": key_r.get("n_D"),
            "c_ij": key_r.get("c_ij"),
            "Z_ev2": best_r.get("Z_ev2"),
            "Z_ev2_names": best_r.get("Z_ev2_names"),
        })
    return rows


def serialize_panel_run(records, conditions, dataset, out_dir, seeds,
                        candidate_order, alpha=ALPHA):
    """Write per_panel_<dataset>_seed<S>.json for NATURAL_WIN panel scoring."""
    os.makedirs(out_dir, exist_ok=True)
    all_rows = build_panel_records(records, conditions, dataset, candidate_order)
    written = []
    for seed in seeds:
        per = [r for r in all_rows if int(r["seed"]) == int(seed)]
        if not per:
            continue
        fname = f"per_panel_{dataset}_seed{int(seed)}.json"
        path = os.path.join(out_dir, fname)
        payload = {
            "seed": int(seed), "benchmark": dataset, "alpha": float(alpha),
            "n_conditions": len(per), "panel": "diverse_backbones",
            "records": per,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        written.append(path)
    return {"written": written, "n_conditions": len(all_rows)}


def serialize_run(records, dataset, out_dir, seeds=None, methods=None, alpha=ALPHA,
                  z_names=None, prefer="auto", condition_key_fn=None, extra_top=None,
                  method_field="method"):
    """Write per_condition_<dataset>_<method>_seed<S>.json for every (method, seed) cell
    present in `records`, matching the cifar_tent_mps_v2 / stress_grid serialization
    contract.  `method_field` selects the record field used for the method axis
    (default "method"; ImageNet-R diverse-backbone panel passes "candidate").
    Returns a manifest dict {written: [...], cells: {...}, kga_backend: ...}.
    """
    os.makedirs(out_dir, exist_ok=True)
    if methods is None:
        methods = sorted({r[method_field] for r in records})
    if seeds is None:
        seeds = sorted({int(r["seed"]) for r in records})
    written = []
    cells = {}
    backend_seen = set()
    for method in methods:
        for seed in seeds:
            per_cond, backend = build_per_condition_records(
                records, method, seed, dataset, alpha=alpha, z_names=z_names,
                prefer=prefer, condition_key_fn=condition_key_fn, method_field=method_field)
            if not per_cond:
                continue
            backend_seen.add(backend)
            fname = f"per_condition_{dataset}_{method}_seed{seed}.json"
            path = os.path.join(out_dir, fname)
            payload = {
                "seed": int(seed), "benchmark": dataset, "method": method,
                "alpha": float(alpha), "n_conditions": len(per_cond),
                "kga_backend": backend,
                "per_condition_fields_absent": [
                    k for k in ("n_D", "c_ij")
                    if all(pcr.get(k) is None for pcr in per_cond)],
                "per_condition_fields_absent_reason":
                    "n_D/c_ij are attached by the runner via panel_capture.py "
                    "(Wave-5); absent means the runner predates the patch or the "
                    "panel capture was skipped.",
                "per_condition_field_notes": {
                    "tau_hat_tau_star":
                        "single-candidate route (frozen vs adapted), so tau is degenerate "
                        "and set to 0.0.",
                    "zone":
                        "computed from benefit_ci: lb>0 => CERTIFIED_ADAPT, ub<0 => "
                        "CERTIFIED_FREEZE, else BLIND.",
                    "gamma_hat":
                        "single-candidate proxy (b_hat/2) with gamma_ci = benefit_ci/2.",
                    "kga_backend":
                        "sklearn_gradient_boost = production certificate; "
                        "numpy_knn_fallback = torch/sklearn-free verification estimator "
                        "(decision rule + conformal radius identical, estimator differs).",
                },
                "records": per_cond,
            }
            if extra_top:
                payload.update(extra_top)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            written.append(path)
            cells[f"{method}_seed{seed}"] = {"file": fname, "n_conditions": len(per_cond)}
    return {
        "written": written,
        "cells": cells,
        "kga_backend": sorted(backend_seen),
        "dataset": dataset,
        "methods": list(methods),
        "seeds": [int(s) for s in seeds],
        "out_dir": out_dir,
    }
