#!/usr/bin/env python3
"""collate_final.py -- consolidate all final-run result JSONs into ONE manifest.

Defensive by design: it globs the known result files, recursively finds any
(regret_kga, regret_adapt, regret_freeze) triple, groups by dataset, and reports
mean +/- std across seeds/records. It NEVER selects the best run -- it reports the
count of beats-both over the total, so cherry-picking is impossible.

This is a convenience aggregator, not the source of truth: always cross-check a
headline number against its own per-dataset JSON.

Usage:  python collate_final.py --results experiments/kbound/results --stamp 20260626_1200
"""
import argparse, glob, json, os
from collections import defaultdict
import statistics as st

ALPHA = 0.10
KGA = ["regret_kga", "regret_K_Bound", "K_Bound"]
ADP = ["regret_adapt", "regret_always_adapt", "always_adapt"]
FRZ = ["regret_freeze", "regret_always_freeze", "always_freeze"]
FA = ["false_adapt", "false_adapt_rate_B<0", "FA_u", "fa_u"]
EXPECTED = ["cifar10c", "imagenetc", "cifar101", "camelyon", "rxrx1",
            "imagenetr", "pacs", "iwildcam", "officehome"]


def _num(d, keys):
    for k in keys:
        if isinstance(d, dict) and isinstance(d.get(k), (int, float)):
            return float(d[k])
    return None


def find_triples(obj):
    """Recursively yield (rk, ra, rf, fa, beats) from any nested dict that has a regret triple."""
    out = []
    if isinstance(obj, dict):
        rk, ra, rf = _num(obj, KGA), _num(obj, ADP), _num(obj, FRZ)
        if rk is not None and ra is not None and rf is not None:
            fa = _num(obj, FA)
            beats = bool(rk < ra and rk < rf and (fa is None or fa <= ALPHA + 1e-9))
            out.append((rk, ra, rf, fa, beats))
        for v in obj.values():
            out += find_triples(v)
    elif isinstance(obj, list):
        for v in obj:
            out += find_triples(v)
    return out


def dataset_of(path):
    s = path.lower()
    for name in ["cifar10c", "imagenetc", "cifar101", "camelyon", "rxrx1",
                 "imagenetr", "pacs", "iwildcam", "officehome", "office_home"]:
        if name in s:
            return "officehome" if name == "office_home" else name
    return os.path.basename(os.path.dirname(path)) or "unknown"


def ms(xs):
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--stamp", default="")
    a = ap.parse_args()

    pats = ["**/decisive_tta_results.json", "**/protocol_result.json", "**/pacs_result.json",
            "**/pacs_vlcs_result.json", "**/holdout_score.json", "**/result_manifest.json"]
    files = []
    for p in pats:
        files += [f for f in glob.glob(os.path.join(a.results, p), recursive=True)
                  if "/._" not in f and "/__pycache__/" not in f]
    files = sorted(set(files))

    by_ds = defaultdict(list)
    for f in files:
        try:
            obj = json.load(open(f))
        except Exception:
            continue
        for t in find_triples(obj):
            by_ds[dataset_of(f)].append(t)

    rows = []
    for ds in sorted(by_ds):
        recs = by_ds[ds]
        if not recs:
            continue
        mk, sk = ms([r[0] for r in recs])
        ma, sa = ms([r[1] for r in recs])
        mf, sf = ms([r[2] for r in recs])
        beats = sum(1 for r in recs if r[4])
        rows.append(dict(dataset=ds, n=len(recs),
                         regret_kga=f"{mk:.4f}+/-{sk:.4f}",
                         regret_adapt=f"{ma:.4f}+/-{sa:.4f}",
                         regret_freeze=f"{mf:.4f}+/-{sf:.4f}",
                         beats_both=f"{beats}/{len(recs)}"))

    base = os.path.join(a.results, f"final_manifest_{a.stamp}" if a.stamp else "final_manifest")
    json.dump({"rows": rows, "files_scanned": files}, open(base + ".json", "w"), indent=2)
    with open(base + ".md", "w") as fh:
        fh.write(f"# Final consolidated manifest ({a.stamp})\n\n")
        fh.write("Mean +/- std across all seeds/records per dataset (all reported, never the best).\n\n")
        fh.write("> **`beats-both (pt-est)` is a point-estimate count, NOT the verdict.** A hairline gap "
                 "(e.g. 0.0157 vs 0.0158) is a **tie / no-harm**, not a win. The CI-robust win/no-harm "
                 "verdict for the natural shifts comes only from the condition-bootstrap "
                 "(`bootstrap_win_cis.py`, `verify_realshift_win.py`).\n\n")
        fh.write("| Dataset | n | regret KGA | regret adapt | regret freeze | beats-both (pt-est) |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['dataset']} | {r['n']} | {r['regret_kga']} | {r['regret_adapt']} "
                     f"| {r['regret_freeze']} | {r['beats_both']} |\n")

    present = {r["dataset"] for r in rows}
    missing = [d for d in EXPECTED if d not in present]
    print("collated datasets :", sorted(present))
    print("MISSING (expected):", missing if missing else "none")
    print("wrote", base + ".md", "and", base + ".json")


if __name__ == "__main__":
    main()
