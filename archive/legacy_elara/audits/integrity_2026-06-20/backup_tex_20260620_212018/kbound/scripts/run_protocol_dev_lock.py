#!/usr/bin/env python3
"""Run a Protocol G/H/M-style dev-lock natural-shift evaluation.

Workflow (matches Camelyon17 Protocol G integrity bar):
  1. Dev screen ONLY on calibration-domain records: pick one adapter from a
     pre-registered panel using canonical KGA (GBR + global eps, alpha fixed).
  2. If no candidate passes the dev screen, STOP (no held-out claim).
  3. Score held-out test ONCE with the locked adapter.

This script never sweeps the held-out test over adapters.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import analyze_F as af  # noqa: E402
import score_kbound_holdout as sk  # noqa: E402


def load_all(path: str, candidate: str) -> list[dict]:
    recs, _ = af.load_records(path, candidate=candidate)
    return recs


def filter_seeds(recs: list[dict], seeds: list[int]) -> list[dict]:
    s = set(seeds)
    return [r for r in recs if int(r["seed"]) in s]


def dev_metrics(recs: list[dict], cal_seeds: list[int], eval_seeds: list[int]) -> dict | None:
    m = af.run_split(recs, cal_seeds, eval_seeds, estimator="gbr", conformal="global")
    if not m:
        return None
    m["beats_both"] = bool(
        m["regret_kga"] < m["regret_adapt"] and m["regret_kga"] < m["regret_freeze"]
    )
    m["fa_ok"] = bool(m["false_adapt"] <= af.ALPHA)
    m["margin"] = float(min(m["regret_adapt"], m["regret_freeze"]) - m["regret_kga"])
    m["verdict_win"] = bool(m["beats_both"] and m["fa_ok"])
    return m


def pick_adapter(panel: list[str], dev_path: str, cal_seeds: list[int], eval_seeds: list[int]) -> tuple[str | None, list[dict]]:
    rows = []
    best = None
    for cand in panel:
        recs = load_all(dev_path, cand)
        m = dev_metrics(recs, cal_seeds, eval_seeds)
        row = {"candidate": cand, "dev": m}
        rows.append(row)
        if not m or not m["fa_ok"]:
            continue
        key = (m["margin"], -m["regret_kga"], cand)
        if best is None or key > best[0]:
            best = (key, cand, m)
    return (best[1] if best else None, rows)


def run_holdout_same_file(path: str, candidate: str, cal_seeds: list[int], test_seeds: list[int]) -> dict:
    recs = load_all(path, candidate)
    m = af.run_split(recs, cal_seeds, test_seeds, estimator="gbr", conformal="global")
    if not m:
        raise RuntimeError("empty held-out split")
    cal = filter_seeds(recs, cal_seeds)
    Zc, Bc = af.arrays(cal)[0], af.arrays(cal)[1]
    eps = float(__import__("numpy").quantile(
        __import__("numpy").abs(af.fit_point(Zc, Bc).predict(Zc) - Bc), 1 - af.ALPHA
    ))
    out = dict(m)
    out["eps_global"] = eps
    out["beats_both"] = bool(out["regret_kga"] < out["regret_adapt"] and out["regret_kga"] < out["regret_freeze"])
    out["fa_ok"] = bool(out["false_adapt"] <= af.ALPHA)
    out["verdict_win"] = bool(out["beats_both"] and out["fa_ok"])
    out["margin"] = float(min(out["regret_adapt"], out["regret_freeze"]) - out["regret_kga"])
    return out


def run_holdout_transfer(
    cal_path: str,
    test_path: str,
    candidate: str,
    cal_seeds: list[int],
    test_seeds: list[int],
) -> dict:
    cal = filter_seeds(load_all(cal_path, candidate), cal_seeds)
    test = filter_seeds(load_all(test_path, candidate), test_seeds)
    if len(cal) < 5 or len(test) < 1:
        raise RuntimeError(f"not enough records cal={len(cal)} test={len(test)}")
    return sk.score_transfer(cal, test, "gbr", "global")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol-yaml", required=True)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.protocol_yaml).read_text())
    out_dir = Path(args.output_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = list(cfg["adapter_panel"])
    dev = cfg["dev_screen"]
    locked, dev_rows = pick_adapter(
        panel,
        str(ROOT / dev["records"]),
        list(dev["cal_seeds"]),
        list(dev["eval_seeds"]),
    )

    result = {
        "protocol": cfg["protocol"],
        "alpha": af.ALPHA,
        "estimator": "gbr",
        "conformal": "global",
        "adapter_panel": panel,
        "dev_screen": dev,
        "dev_rows": dev_rows,
        "locked_adapter": locked,
        "dev_screen_pass": locked is not None,
    }

    if locked is None:
        result["verdict_win"] = False
        result["stop_reason"] = "no dev-screen candidate with false_adapt <= alpha"
        print(json.dumps(result, indent=2))
        (out_dir / "protocol_result.json").write_text(json.dumps(result, indent=2))
        print(f"STOP: {result['stop_reason']}")
        return 2

    held = cfg["heldout"]
    mode = held.get("mode", "transfer")
    if mode == "seed_split":
        test_locked = run_holdout_same_file(
            str(ROOT / held["records"]),
            locked,
            list(held["cal_seeds"]),
            list(held["test_seeds"]),
        )
    elif mode == "transfer":
        test_locked = run_holdout_transfer(
            str(ROOT / held["cal_records"]),
            str(ROOT / held["test_records"]),
            locked,
            list(held["cal_seeds"]),
            list(held["test_seeds"]),
        )
    else:
        raise ValueError(mode)

    result["heldout"] = held
    result["test_locked"] = test_locked
    result["verdict_win"] = bool(test_locked.get("verdict_win", False))

    print(json.dumps(result, indent=2))
    (out_dir / "protocol_result.json").write_text(json.dumps(result, indent=2))
    findings = {
        "protocol": cfg["protocol"],
        "locked_adapter": locked,
        "dev_screen_pass": True,
        "heldout_verdict_win": result["verdict_win"],
        "false_adapt": test_locked.get("false_adapt"),
        "regret_kga": test_locked.get("regret_kga"),
        "regret_adapt": test_locked.get("regret_adapt"),
        "regret_freeze": test_locked.get("regret_freeze"),
        "beats_both": test_locked.get("beats_both"),
    }
    (out_dir / "VERIFIED_FINDINGS.json").write_text(json.dumps(findings, indent=2))
    print(f"Saved {out_dir / 'protocol_result.json'}")
    return 0 if result["verdict_win"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
