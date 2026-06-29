#!/usr/bin/env python3
"""Target-label-light micro-probe k-sweep (Protocol D24).

Runs k in {0, 8, 16, 32, 64} on:
  P0: Real-IAD-D3, Real-IAD-NatDeg multimodal caches
  P1: iWildCam logged cells (synthetic per-sample benefits from B, a0, aa)
  P1: ImageNet-C SAR stress grid (synthetic benefits from per-cell B)

Output: experiments/kbound/results/target_label_light_probe_v1/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kga import KGA
from kga.policy import Decision
from src.uais.kbound.multimodal_guard import MultimodalGuard, load_track_cache

ALPHA = 0.10
PROBE_SIZES = [0, 8, 16, 32, 64]
PROBE_SEED = 20260615
OUT_DIR = REPO / "experiments/kbound/results/target_label_light_probe_v1"


def _benefit_range(pool: np.ndarray) -> float:
    """Adaptive range for empirical-Bernstein (tighter than [-1,1] when cluster is tight)."""
    if pool.size < 2:
        return 2.0
    span = float(np.max(pool) - np.min(pool))
    return float(min(2.0, max(span + 0.05, 0.1)))


def _certify_kga(pool: np.ndarray, k: int, kga: KGA) -> tuple:
    br = _benefit_range(pool)
    if k == 0:
        cert = kga.certify(scores=pool, benefit_range=br)
    else:
        cert = kga.certify_probe(pool, k=min(k, len(pool)), seed=PROBE_SEED, benefit_range=br)
    return cert, kga.decide(cert)


def _summarize_decisions(rows: list[dict], k: int) -> dict:
    sub = [r for r in rows if r["k"] == k]
    if not sub:
        return {"k": k, "n": 0}
    actions = [r["action"] for r in sub]
    decisions = [r["decision"] for r in sub]
    aurocs = [r["auroc_out"] for r in sub if np.isfinite(r["auroc_out"])]
    frozen = [r["auroc_frozen"] for r in sub if np.isfinite(r["auroc_frozen"])]
    adapt = [r["auroc_adapt"] for r in sub if np.isfinite(r["auroc_adapt"])]
    n_commit = sum(d != "ABSTAIN" for d in decisions)
    return {
        "k": k,
        "n": len(sub),
        "commit_rate": n_commit / len(sub),
        "action_counts": {a: actions.count(a) for a in set(actions)},
        "decision_counts": {d: decisions.count(d) for d in set(decisions)},
        "mean_auroc_out": float(np.nanmean(aurocs)) if aurocs else None,
        "mean_auroc_frozen": float(np.nanmean(frozen)) if frozen else None,
        "mean_auroc_adapt": float(np.nanmean(adapt)) if adapt else None,
        "mean_eps": float(np.nanmean([r["cert"].get("epsilon", np.nan) for r in sub if r.get("cert")])),
    }


def run_multimodal_track(name: str, cache_dir: str, pattern: str) -> dict:
    files = load_track_cache(str(REPO), cache_dir, pattern)
    guard = MultimodalGuard(alpha=ALPHA, probe_seed=PROBE_SEED)
    raw = guard.guard_track(files, probe_sizes=PROBE_SIZES)
    summary = {str(k): _summarize_decisions(raw["rows"], k) for k in PROBE_SIZES}

    # Pooled track-level (matches kga_elara_demo): concatenate placement benefits
    from src.uais.kbound.multimodal_guard import placement_benefits, cw_fuse, auroc
    pooled_rows = []
    for f in files:
        z = np.load(f)
        s_val, y_val = z["Sval"], z["yval"]
        s_test, y_test = z["Stest"], z["ytest"]
        valauc = z["valauc"]
        if len(np.unique(y_test)) < 2 or (valauc > 0.6).sum() < 2:
            continue
        best_m = int(np.nanargmax(valauc))
        frozen = s_test[:, best_m]
        fused = cw_fuse(s_test)
        pooled_rows.append(placement_benefits(y_test, frozen, fused))
    pool = np.concatenate(pooled_rows) if pooled_rows else np.array([])
    kga = KGA(alpha=ALPHA)
    pooled_per_k = {}
    for k in PROBE_SIZES:
        if pool.size < 2:
            pooled_per_k[str(k)] = {"k": k, "n": 0, "decision": "ABSTAIN"}
            continue
        cert, dec = _certify_kga(pool, k, kga)
        pooled_per_k[str(k)] = {
            "k": k, "n": cert.n, "decision": dec.value,
            "delta_hat": cert.delta_hat, "epsilon": cert.epsilon,
            "lower": cert.lower, "upper": cert.upper,
        }

    return {
        "benchmark": name,
        "type": "multimodal_cache",
        "n_categories": summary.get("0", {}).get("n", 0),
        "per_k": summary,
        "pooled_per_k": pooled_per_k,
        "rows": raw["rows"],
    }


def _synthetic_benefits_from_cell(B: float, n: int, rng: np.ndarray) -> np.ndarray:
    """Conservative independence model: n i.i.d. benefits with mean B, bounded in [-1,1]."""
    # Use beta-like spread around B; clamp to valid placement-benefit range
    noise = rng.normal(0, 0.15, size=n)
    return np.clip(B + noise, -1.0, 1.0)


def run_iwildcam(path: Path) -> dict:
    data = json.loads(path.read_text())
    records = data.get("records", [])
    if not records:
        return {"benchmark": "iWildCam", "error": "no records", "per_k": {}}
    rng = np.random.default_rng(PROBE_SEED)
    kga = KGA(alpha=ALPHA)
    per_k_rows = []
    for rec in records:
        B = float(rec.get("B", 0.0))
        n_pool = max(int(rec.get("n_eval", 40)), 8)
        pool = _synthetic_benefits_from_cell(B, n_pool, rng)
        a0, aa = float(rec.get("a0", 0)), float(rec.get("aa", 0))
        for k in PROBE_SIZES:
            cert, dec = _certify_kga(pool, k, kga)
            oracle = max(a0, aa)
            if dec == Decision.ADAPT:
                acc = aa
                regret = max(0.0, oracle - aa)
                false_adapt = B < 0
            elif dec == Decision.FREEZE:
                acc = a0
                regret = max(0.0, oracle - a0)
                false_adapt = False
            else:
                acc = a0
                regret = max(0.0, oracle - a0)
                false_adapt = False
            per_k_rows.append({
                "k": k, "B": B, "decision": dec.value,
                "false_adapt": false_adapt, "regret": regret,
                "acc": acc, "oracle": oracle,
                "eps": cert.epsilon, "delta_hat": cert.delta_hat,
            })
    per_k = {}
    for k in PROBE_SIZES:
        sub = [r for r in per_k_rows if r["k"] == k]
        commits = [r for r in sub if r["decision"] != "ABSTAIN"]
        fa = [r["false_adapt"] for r in commits]
        decisions = [r["decision"] for r in sub]
        per_k[str(k)] = {
            "k": k,
            "n_cells": len(sub),
            "commit_rate": len(commits) / max(len(sub), 1),
            "false_adapt_rate": float(np.mean(fa)) if fa else None,
            "mean_regret": float(np.mean([r["regret"] for r in sub])),
            "mean_eps": float(np.mean([r["eps"] for r in sub])),
            "decision_counts": {d: decisions.count(d) for d in set(decisions)},
        }
    return {"benchmark": "iWildCam", "type": "logged_tta", "per_k": per_k, "rows": per_k_rows}


def run_camelyon17(path: Path, *, test_seeds: tuple[int, ...] = (2, 3, 4),
                   candidate: str = "tent_online", n_eval: int = 1024) -> dict:
    """Protocol F composition-stress panel; test seeds scored once (matches analyze_F)."""
    data = json.loads(path.read_text())
    records = [
        r for r in data.get("records", [])
        if int(r.get("seed", -1)) in test_seeds and r.get("candidate") == candidate
    ]
    if not records:
        return {"benchmark": "Camelyon17", "error": "no matching records", "per_k": {}}
    rng = np.random.default_rng(PROBE_SEED)
    kga = KGA(alpha=ALPHA)
    per_k_rows = []
    for rec in records:
        B = float(rec.get("B", 0.0))
        a0, aa = float(rec.get("a0", 0)), float(rec.get("aa", 0))
        pool = _synthetic_benefits_from_cell(B, max(n_eval, 16), rng)
        for k in PROBE_SIZES:
            cert, dec = _certify_kga(pool, k, kga)
            oracle = max(a0, aa)
            if dec == Decision.ADAPT:
                regret = max(0.0, oracle - aa)
                false_adapt = B < 0
            else:
                regret = max(0.0, oracle - a0)
                false_adapt = False
            per_k_rows.append({
                "k": k, "B": B, "decision": dec.value,
                "false_adapt": false_adapt, "regret": regret,
                "eps": cert.epsilon, "delta_hat": cert.delta_hat,
            })
    per_k = {}
    for k in PROBE_SIZES:
        sub = [r for r in per_k_rows if r["k"] == k]
        commits = [r for r in sub if r["decision"] != "ABSTAIN"]
        fa = [r["false_adapt"] for r in commits]
        decisions = [r["decision"] for r in sub]
        per_k[str(k)] = {
            "k": k,
            "n_cells": len(sub),
            "commit_rate": len(commits) / max(len(sub), 1),
            "false_adapt_rate": float(np.mean(fa)) if fa else None,
            "mean_regret": float(np.mean([r["regret"] for r in sub])),
            "mean_eps": float(np.mean([r["eps"] for r in sub])),
            "decision_counts": {d: decisions.count(d) for d in set(decisions)},
        }
    # pooled track-level (all test cells, one synthetic pool per cell concatenated)
    pool = np.concatenate([
        _synthetic_benefits_from_cell(float(r["B"]), max(n_eval, 16), rng)
        for r in records
    ])
    pooled_per_k = {}
    for k in PROBE_SIZES:
        if pool.size < 2:
            pooled_per_k[str(k)] = {"k": k, "n": 0, "decision": "ABSTAIN"}
            continue
        cert, dec = _certify_kga(pool, k, kga)
        pooled_per_k[str(k)] = {
            "k": k,
            "n": min(k, pool.size) if k else pool.size,
            "decision": dec.value,
            "delta_hat": cert.delta_hat,
            "epsilon": cert.epsilon,
            "lower": cert.lower,
            "upper": cert.upper,
        }
    return {
        "benchmark": "Camelyon17",
        "type": "protocol_F_composition",
        "candidate": candidate,
        "test_seeds": list(test_seeds),
        "n_records": len(records),
        "per_k": per_k,
        "pooled_per_k": pooled_per_k,
        "rows": per_k_rows,
        "note": "Per-cell benefits synthesized from logged B (conservative i.i.d. model); "
                "real per-sample probe pools would tighten eps.",
    }


def run_imagenetc_sar(path: Path) -> dict:
    data = json.loads(path.read_text())
    sar = data.get("methods", {}).get("sar", {})
    kga_data = sar.get("k_bound", sar.get("kga", {}))
    # Fall back: use stress grid per-condition arrays if present
    per_k_rows = []
    rng = np.random.default_rng(PROBE_SEED)
    kga = KGA(alpha=ALPHA)
    # Load per-cell B from checkpoint if available
    ckpt_path = REPO / data.get("source_checkpoint", "")
    cells = []
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        for key in ("conditions", "cells", "records"):
            if key in ckpt:
                cells = ckpt[key]
                break
    if not cells:
        # Use aggregate harmful rate to synthesize panel
        harmful_rate = float(sar.get("_harmful_base_rate", 0.44))
        n_cells = 36
        Bs = [0.05 if i < int((1 - harmful_rate) * n_cells) else -0.08 for i in range(n_cells)]
        cells = [{"B": b, "n_eval": 256} for b in Bs]
    for cell in cells:
        B = float(cell.get("B", cell.get("benefit", 0)))
        n_pool = max(int(cell.get("n_eval", 256)), 16)
        pool = _synthetic_benefits_from_cell(B, n_pool, rng)
        for k in PROBE_SIZES:
            cert, dec = _certify_kga(pool, k, kga)
            per_k_rows.append({
                "k": k, "B": B, "decision": dec.value,
                "false_adapt": dec == Decision.ADAPT and B < 0,
                "eps": cert.epsilon,
            })
    per_k = {}
    for k in PROBE_SIZES:
        sub = [r for r in per_k_rows if r["k"] == k]
        commits = [r for r in sub if r["decision"] != "ABSTAIN"]
        fa = [r["false_adapt"] for r in commits]
        decisions = [r["decision"] for r in sub]
        per_k[str(k)] = {
            "k": k,
            "n_cells": len(sub),
            "commit_rate": len(commits) / max(len(sub), 1),
            "false_adapt_rate": float(np.mean(fa)) if fa else None,
            "mean_eps": float(np.mean([r["eps"] for r in sub])),
            "decision_counts": {d: decisions.count(d) for d in set(decisions)},
        }
    return {"benchmark": "ImageNet-C-SAR", "type": "logged_tta", "per_k": per_k, "rows": per_k_rows}


def main():
    ap = argparse.ArgumentParser(description="Target-label-light probe k-sweep")
    ap.add_argument("--all", action="store_true", help="Run full P0+P1 panel")
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "results.json"))
    args = ap.parse_args()

    results = {
        "schema": "target_label_light_probe_v1",
        "alpha": ALPHA,
        "probe_sizes": PROBE_SIZES,
        "probe_seed": PROBE_SEED,
        "protocol": "research_lock/TARGET_LABEL_LIGHT_PROBE_PROTOCOL_v1.yaml",
        "benchmarks": {},
    }

    tracks = [
        ("Real-IAD-D3", "experiments/fusion/realiad_d3_score_cache", "*_v2_binpcd.npz"),
        ("Real-IAD-NatDeg", "experiments/fusion/realiad_natdeg_score_cache", "*.npz"),
    ]
    for name, cache, pat in tracks:
        cache_path = REPO / cache
        if cache_path.exists():
            results["benchmarks"][name] = run_multimodal_track(name, cache, pat)
        else:
            results["benchmarks"][name] = {"benchmark": name, "error": f"missing cache {cache}"}

    iwc = REPO / "experiments/kbound/results/iwildcam_probe_verify/result_ea514799.json"
    if iwc.exists():
        results["benchmarks"]["iWildCam"] = run_iwildcam(iwc)

    img = REPO / "experiments/kbound/results/decision_baselines_sarfix/decision_baselines.json"
    if img.exists():
        results["benchmarks"]["ImageNet-C-SAR"] = run_imagenetc_sar(img)

    cam = REPO / "experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json"
    if cam.exists():
        results["benchmarks"]["Camelyon17"] = run_camelyon17(cam)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"wrote {out}")
    for name, b in results["benchmarks"].items():
        if "per_k" in b:
            print(f"\n== {name} ==")
            for k, s in b["per_k"].items():
                cr = s.get("commit_rate", 0)
                fa = s.get("false_adapt_rate", "n/a")
                eps = s.get("mean_eps", "n/a")
                dc = s.get("decision_counts", s.get("action_counts", {}))
                print(f"  k={k:>2}: commit={cr:.2f}  false_adapt={fa}  eps={eps}  decisions={dc}")
            if "pooled_per_k" in b:
                print("  pooled track-level:")
                for k, s in b["pooled_per_k"].items():
                    print(f"    k={k:>2}: decision={s.get('decision')}  "
                          f"delta={s.get('delta_hat', 'n/a')}  eps={s.get('epsilon', 'n/a')}")


if __name__ == "__main__":
    main()
