#!/usr/bin/env python3
"""
Extract Camelyon-style per_condition_*_seed*.json from WILDS/Office-Home result_*.json
runs, then aggregate multi-seed no-harm summaries (same contract as multiseed_natural.py).

The GPU runners write a monolithic result schema (records[] + routing_*). Camelyon's
multiseed path expects per_condition_<ds>_<cand>_seed<S>.json with stored kga_decision.
This script is the missing glue:

  result_*.json
    -> per_condition_serialize.serialize_run  (LOO KGA per seed × candidate)
    -> multiseed_natural-style aggregate JSON + LaTeX row

Usage:
  python3 extract_multiseed_natural.py \\
      --track officehome \\
      --result experiments/kbound/results/multiseed/officehome/**/result_*.json \\
      --out-dir experiments/kbound/results/multiseed/officehome/extracted

  python3 extract_multiseed_natural.py --track iwildcam --result ... --candidates tent_episodic
  python3 extract_multiseed_natural.py --track rxrx1 --result ... --candidates sar_online

Default locked candidates match the short-paper / protocol locks:
  officehome -> sar_online_aggressive  (Protocol M v2)
  iwildcam   -> tent_episodic          (Protocol H v2)
  rxrx1      -> sar_online             (Protocol J)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]  # .../docs/research/kbound/scripts -> repo root
WILDS = REPO / "experiments" / "kbound" / "wilds"
sys.path.insert(0, str(WILDS))

import per_condition_serialize as pcs  # noqa: E402

# Paper / protocol locked adapters (override with --candidates).
LOCKED = {
    "officehome": ["sar_online_aggressive"],
    "iwildcam": ["tent_episodic"],
    "rxrx1": ["sar_online"],
}

# Filename slug used in per_condition_<slug>_<cand>_seed*.json
DATASET_SLUG = {
    "officehome": "officehome",
    "iwildcam": "iwildcam",
    "rxrx1": "rxrx1",
}

# Records use "candidate" for OH and for IW/RX locked names; method alone is coarser.
METHOD_FIELD = {
    "officehome": "candidate",
    "iwildcam": "candidate",
    "rxrx1": "candidate",
}


def _expand(patterns):
    files = []
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        if not hits and os.path.isfile(p):
            hits = [p]
        files.extend(hits)
    # de-dupe, prefer largest file per basename (full-scale over smoke)
    by = {}
    for f in files:
        b = os.path.basename(f)
        sz = os.path.getsize(f)
        if b not in by or sz > by[b][0]:
            by[b] = (sz, f)
    return [by[k][1] for k in sorted(by)]


def _load_records(paths):
    records, evidence, srcs = [], None, []
    for p in paths:
        d = json.load(open(p))
        recs = d.get("records") or []
        if not recs:
            raise SystemExit(f"no records[] in {p}")
        records.extend(recs)
        srcs.append(os.path.basename(p))
        if evidence is None and d.get("evidence_names"):
            evidence = list(d["evidence_names"])
    return records, evidence, srcs


def _per_seed_from_file(path):
    """Mirror multiseed_natural.per_seed (kept local so this script is one-stop)."""
    d = json.load(open(path))
    recs = d["records"]
    a0 = np.array([r["a0"] for r in recs], float)
    aa = np.array([r["a_adapted"] for r in recs], float)
    dec = [str(r.get("kga_decision", "")).lower() for r in recs]
    adapt = np.array(["adapt" in x for x in dec])
    ao = np.array([
        (r["a_oracle"] if r.get("a_oracle") is not None else max(r["a0"], r["a_adapted"]))
        for r in recs
    ], float)
    ak = np.array([
        (r["a_kbound"] if r.get("a_kbound") is not None else (aa[i] if adapt[i] else a0[i]))
        for i, r in enumerate(recs)
    ], float)
    B = np.array([r["B"] for r in recs], float)
    return dict(
        seed=d.get("seed"),
        n=len(recs),
        rk=float((ao - ak).mean()),
        ra=float((ao - aa).mean()),
        rf=float((ao - a0).mean()),
        fau=float(np.mean(adapt & (B <= 0))),
        rk_pc=(ao - ak),
        ra_pc=(ao - aa),
        rf_pc=(ao - a0),
        backend=d.get("kga_backend"),
    )


def _boot(x, nb=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    b = np.empty(nb)
    for i in range(nb):
        b[i] = x[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(b, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def aggregate_candidate(dataset, candidate, per_dir, alpha=0.10):
    pat = f"per_condition_{dataset}_{candidate}_seed*.json"
    files = sorted(glob.glob(os.path.join(per_dir, pat)))
    if not files:
        raise SystemExit(f"no extracted files matching {pat} in {per_dir}")
    S = [_per_seed_from_file(f) for f in files]
    rk = np.array([s["rk"] for s in S])
    ra = np.array([s["ra"] for s in S])
    rf = np.array([s["rf"] for s in S])
    fau = np.array([s["fau"] for s in S])
    better = "freeze" if rf.mean() <= ra.mean() else "adapt"
    rk_pc = np.concatenate([s["rk_pc"] for s in S])
    ra_pc = np.concatenate([s["ra_pc"] for s in S])
    rf_pc = np.concatenate([s["rf_pc"] for s in S])
    gap_better = (rf_pc - rk_pc) if better == "freeze" else (ra_pc - rk_pc)
    gap_worse = (ra_pc - rk_pc) if better == "freeze" else (rf_pc - rk_pc)
    # Forest gaps: regret reduction = fixed - kga (positive favors KGA)
    gap_vs_adapt = ra_pc - rk_pc
    gap_vs_freeze = rf_pc - rk_pc
    ci_b = _boot(gap_better)
    ci_w = _boot(gap_worse)
    ci_a = _boot(gap_vs_adapt)
    ci_f = _boot(gap_vs_freeze)
    ties_better = ci_b[0] <= 0 <= ci_b[1]
    beats_worse = ci_w[0] > 0
    beats_both = ci_b[0] > 0 and beats_worse
    fa_ok = bool(np.all(fau <= alpha))
    verdict = (
        "beats-both (multi-seed)" if beats_both and fa_ok else
        "stable no-harm" if ties_better and beats_worse and fa_ok else
        "unstable/other"
    )
    out = dict(
        dataset=dataset,
        candidate=candidate,
        analysis="loo_within_seed_single_candidate",
        analysis_note=(
            "Per-seed LOO GBR+conformal on that seed's conditions only "
            "(Camelyon multiseed_natural contract). This is a multi-seed "
            "stability check on a monolithic result_*.json grid; it is NOT "
            "a replay of Protocol M/H/J OOF seed-holdout locks."
        ),
        seeds=[s["seed"] for s in S],
        n_seeds=len(S),
        conditions_per_seed=S[0]["n"],
        alpha=alpha,
        kga_backend=sorted({s["backend"] for s in S if s["backend"]}),
        regret_kga=[round(float(rk.mean()), 4), round(float(rk.std()), 4)],
        regret_adapt=[round(float(ra.mean()), 4), round(float(ra.std()), 4)],
        regret_freeze=[round(float(rf.mean()), 4), round(float(rf.std()), 4)],
        FA_u_per_seed=[round(float(x), 4) for x in fau],
        FA_u_max=round(float(fau.max()), 4),
        better_policy=better,
        gap_vs_better_ci95=ci_b,
        gap_vs_worse_ci95=ci_w,
        gap_vs_adapt=dict(mean=round(float(gap_vs_adapt.mean()), 4), ci95=ci_a),
        gap_vs_freeze=dict(mean=round(float(gap_vs_freeze.mean()), 4), ci95=ci_f),
        verdict=verdict,
        files=[os.path.basename(f) for f in files],
        latex_row=(
            f"{dataset} ({candidate}) & {len(S)} & "
            f"{rk.mean():.4f}$\\pm${rk.std():.4f} & "
            f"{ra.mean():.4f} & {rf.mean():.4f} & {fau.max():.3f} & {verdict} \\\\"
        ),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=sorted(LOCKED))
    ap.add_argument("--result", nargs="+", required=True,
                    help="result_*.json path(s) or globs")
    ap.add_argument("--candidates", nargs="+", default=None,
                    help="override locked candidate list")
    ap.add_argument("--out-dir", default="",
                    help="directory for per_condition files + aggregates")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--prefer", default="auto",
                    choices=["auto", "sklearn", "numpy"])
    ap.add_argument("--skip-serialize", action="store_true",
                    help="only re-aggregate existing per_condition files in --out-dir")
    a = ap.parse_args()

    track = a.track
    slug = DATASET_SLUG[track]
    cands = a.candidates or LOCKED[track]
    out_dir = a.out_dir or str(
        REPO / "experiments" / "kbound" / "results" / "multiseed" / track / "extracted"
    )
    os.makedirs(out_dir, exist_ok=True)

    paths = _expand(a.result)
    if not paths and not a.skip_serialize:
        raise SystemExit(f"no result files matched: {a.result}")

    manifest = {"track": track, "sources": [], "serialize": None, "aggregates": []}
    if not a.skip_serialize:
        records, evidence, srcs = _load_records(paths)
        manifest["sources"] = srcs
        # Ensure method_field values exist on every record.
        mf = METHOD_FIELD[track]
        missing = [i for i, r in enumerate(records) if r.get(mf) is None]
        if missing:
            raise SystemExit(
                f"{len(missing)} records lack field {mf!r} (needed for {track})"
            )
        ser = pcs.serialize_run(
            records,
            dataset=slug,
            out_dir=out_dir,
            methods=cands,
            alpha=a.alpha,
            z_names=evidence,
            prefer=a.prefer,
            method_field=mf,
            extra_top={
                "source_result_files": srcs,
                "extract_note": (
                    "LOO single-candidate KGA per (candidate, seed); "
                    "Camelyon-compatible per_condition schema for multiseed_natural."
                ),
            },
        )
        manifest["serialize"] = ser
        print(json.dumps({"serialize": ser}, indent=2))

    aggregates = []
    for c in cands:
        agg = aggregate_candidate(slug, c, out_dir, alpha=a.alpha)
        op = os.path.join(out_dir, f"multiseed_{slug}_{c}.json")
        json.dump(agg, open(op, "w"), indent=2)
        aggregates.append(op)
        print(json.dumps(agg, indent=2))
        print("LaTeX row:\n" + agg["latex_row"])
        print("wrote", op)

    manifest["aggregates"] = aggregates
    mp = os.path.join(out_dir, f"extract_manifest_{slug}.json")
    json.dump(manifest, open(mp, "w"), indent=2)
    print("manifest ->", mp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
