#!/usr/bin/env python3
"""
Convert OFFICIAL POEM / AETTA output into the --decisions JSON that
official_baselines_headtohead.py consumes:  {condition_string: "adapt"|"freeze"|"abstain"}.

Official code (run these yourself on the same CIFAR-10-C conditions, then point this script at their output):
  POEM  (Bar et al., NeurIPS 2024): https://github.com/yarinbar/poem
  AETTA (Lee et al., CVPR 2024):    https://github.com/taeckyung/AETTA   (estimator: learner/dnn.py::aetta)

Faithful decision mapping (matches how the paper compares gates):
  * POEM protects (keeps the frozen model) unless its betting detector triggers an update.
      per condition -> "adapt" if POEM updated on that condition, else "freeze".
  * AETTA estimates accuracy (it is not itself a gate). Turn it into the same adapt/freeze decision:
      per condition -> "adapt" if est_acc_adapted > est_acc_frozen, else "freeze".

This script RUNS NOTHING and invents nothing: it only reformats an official per-condition table
(CSV or JSON) whose rows align to the K-Bound condition strings
(e.g. "gaussian_noise|s1|large_iid|iid|mild|r0"). Missing conditions are reported, not fabricated.

Usage:
  python3 baseline_decisions_adapter.py --method aetta --input aetta_out.csv  --out aetta_decisions.json
  python3 baseline_decisions_adapter.py --method poem  --input poem_out.json  --out poem_decisions.json
"""
import argparse, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CANON = os.path.join(HERE, "..", "experiments", "kbound", "results", "per_condition_cifar10c_tent_seed0.json")

def canonical_conditions():
    return [r.get("condition","") for r in json.load(open(CANON))["records"]]

def load_rows(path):
    if path.endswith(".json"):
        d = json.load(open(path))
        if isinstance(d, dict):
            d = d.get("records") or d.get("rows") or d.get("results") or list(d.values())
        return [r for r in d if isinstance(r, dict)]
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def get(row, cands, default=None):
    for c in cands:
        if c in row and row[c] not in ("", None): return row[c]
    return default

def to_decision(method, row):
    cond = get(row, ("condition","cond","cell","name","key"))
    if method == "aetta":
        af = get(row, ("est_acc_adapted","acc_adapted_est","adapted_est","est_adapted","yhat_adapted"))
        f0 = get(row, ("est_acc_frozen","acc_frozen_est","frozen_est","est_frozen","yhat_frozen"))
        if af is None or f0 is None: return cond, None
        return cond, ("adapt" if float(af) > float(f0) else "freeze")
    # poem
    act = get(row, ("action","decision","poem_action"))
    if act is not None:
        a = str(act).lower()
        if a in ("adapt","update","spend","1","true"):  return cond, "adapt"
        if a in ("freeze","protect","skip","0","false"): return cond, "freeze"
        if a in ("abstain","none"):                      return cond, "abstain"
    upd = get(row, ("updated","did_update","adapted"))
    if upd is not None: return cond, ("adapt" if str(upd).lower() in ("1","true","yes") else "freeze")
    return cond, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["poem","aetta"])
    ap.add_argument("--input", required=True, help="official per-condition CSV/JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions-from", default=CANON)
    a = ap.parse_args()
    rows = load_rows(a.input)
    dec = {}
    unmapped = 0
    for r in rows:
        c, d = to_decision(a.method, r)
        if c is None or d is None: unmapped += 1; continue
        dec[str(c)] = d
    canon = [r.get("condition","") for r in json.load(open(a.conditions_from))["records"]]
    missing = [c for c in canon if c not in dec]
    extra   = [c for c in dec if c not in set(canon)]
    print(f"[{a.method}] mapped {len(dec)} decisions; {unmapped} rows unmapped; "
          f"{len(missing)} canonical conditions missing; {len(extra)} extra keys")
    if missing[:3]: print("  e.g. missing:", missing[:3])
    if missing:
        print("WARNING: not all 432 conditions covered — the head-to-head will error until every "
              "condition has a decision. Check that the official run used the same conditions/order.", file=sys.stderr)
    json.dump(dec, open(a.out,"w"), indent=2)
    print("wrote", a.out)

if __name__ == "__main__": main()
