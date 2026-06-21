"""run_smoke.py — deterministic per-dataset ELARA-Opt smoke test.

For each of the nine runner configs it confirms: the adapter loads, adapts on an
UNLABELED batch, emits label-free telemetry, and KGA consumes the candidate
(certificate + ADAPT/FREEZE/ABSTAIN).  It also checks run-to-run determinism
(identical candidate hash) and exercises the pipeline decide_kga route.

This is integration-mechanics only — a small BN-CNN stand-in + synthetic
covariate shift — NOT a performance result.  STOP-before-eval is respected.

Run:  PYTHONPATH=.:packaging/kbound-tta/src ~/.venv_wilds/bin/python \
        experiments/kbound/elara_opt/run_smoke.py --dataset all --n 16 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import ELARA_OPT_DEFAULTS
from .smoke_models import DATASET_CONFIGS, DATASET_IDS, build_f0, synth_cell
from .modes import load_meta_gate, ELARA_MODES
from .run_elara_candidate import run_elara_candidate, kga_decide_multicell

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_HERE, "..", "results", "elara_opt_smoke")


def smoke_one(dataset: str, n: int, seed: int, modes, outdir: str, cfg=None) -> dict:
    cfg = cfg or ELARA_OPT_DEFAULTS
    dc = DATASET_CONFIGS[dataset]
    nc, hw, in_ch = dc["num_classes"], dc["hw"], dc["in_ch"]
    f0 = build_f0(nc, in_ch, seed=seed)
    stream, eval_x, dev_y = synth_cell(nc, n, in_ch, hw, seed)
    meta_model = load_meta_gate(cfg)

    os.makedirs(outdir, exist_ok=True)
    per_mode, hashes = {}, {}
    for mode in modes:
        mm = meta_model if mode == "elara_meta" else None
        if mode == "elara_meta" and mm is None:
            per_mode[mode] = {"status": "SKIPPED_no_meta_checkpoint"}
            continue
        res = run_elara_candidate(f0, stream, eval_x, dev_y, nc, mode,
                                  steps=cfg["steps"], lr=cfg["lr"], cfg=cfg,
                                  meta_model=mm, seed=seed, alpha=0.1)
        # write telemetry (label-free)
        tpath = os.path.join(outdir, f"{dataset}_{mode}.telemetry.jsonl")
        with open(tpath, "w") as fh:
            fh.write(json.dumps({"_record": "summary", **res["telemetry"]["summary"]}) + "\n")
            for i, s in enumerate(res["telemetry"]["steps"]):
                fh.write(json.dumps({"_record": "step", "_i": i, **s}) + "\n")
        hashes[mode] = res["candidate_hash"]
        per_mode[mode] = {
            "candidate_hash": res["candidate_hash"],
            "update_norm": round(res["update_norm"], 6),
            "B_dev_benefit": round(res["B_dev_benefit"], 6),
            "kga_delta_hat": round(res["kga_delta_hat"], 6),
            "kga_epsilon": round(res["kga_epsilon"], 6),
            "kga_decision": res["kga_decision"],
            "n_telemetry_steps": len(res["telemetry"]["steps"]),
            "telemetry_file": os.path.relpath(tpath, _HERE),
        }

    # determinism check on elara_uniform
    det = None
    if "elara_uniform" in modes:
        r2 = run_elara_candidate(f0, stream, eval_x, dev_y, nc, "elara_uniform",
                                 steps=cfg["steps"], lr=cfg["lr"], cfg=cfg, seed=seed)
        det = (r2["candidate_hash"] == hashes.get("elara_uniform"))

    # pipeline decide_kga route over a tiny multi-cell stack
    multicell = None
    try:
        records = []
        for k in range(5):
            s2, ex2, dy2 = synth_cell(nc, n, in_ch, hw, seed + 17 * (k + 1))
            records.append(run_elara_candidate(f0, s2, ex2, dy2, nc, "elara_rule",
                                               steps=cfg["steps"], lr=cfg["lr"],
                                               cfg=cfg, seed=seed + 17 * (k + 1)))
        mc = kga_decide_multicell(records, alpha=0.1)
        multicell = {"n_cells": mc["n_cells"], "eps": round(mc["eps"], 6),
                     "decisions": mc["decisions"],
                     "beats_both": mc["policy_metrics"]["beats_both"]}
    except Exception as e:  # report straight, never hide
        multicell = {"error": repr(e)[:160]}

    return {
        "dataset": dataset, "num_classes": nc, "smoke_input": f"{in_ch}x{hw}x{hw}",
        "real_arch": dc["arch"], "real_input": dc["real_input"],
        "n": n, "seed": seed, "modes": per_mode,
        "determinism_uniform_same_hash": det,
        "kga_multicell_route": multicell,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default=",".join(ELARA_MODES))
    ap.add_argument("--out", default=_DEFAULT_OUT)
    args = ap.parse_args(argv)

    datasets = DATASET_IDS if args.dataset == "all" else [args.dataset]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results, ok = [], True
    for d in datasets:
        try:
            r = smoke_one(d, args.n, args.seed, modes, args.out)
            results.append(r)
            dec = {m: v.get("kga_decision", v.get("status")) for m, v in r["modes"].items()}
            print(f"[smoke] {d:11s} nc={r['num_classes']:4d} det={r['determinism_uniform_same_hash']} "
                  f"decisions={dec}", flush=True)
        except Exception as e:
            ok = False
            results.append({"dataset": d, "ERROR": repr(e)})
            print(f"[smoke] {d:11s} ERROR: {repr(e)[:200]}", flush=True)

    summary_path = os.path.join(args.out, "smoke_summary.json")
    os.makedirs(args.out, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump({"results": results, "all_ok": ok,
                   "config_version": ELARA_OPT_DEFAULTS["version"]}, fh, indent=2)
    print(f"[smoke] wrote {summary_path}; all_ok={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
