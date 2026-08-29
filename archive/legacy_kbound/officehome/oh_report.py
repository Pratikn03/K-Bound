"""
oh_report.py - consolidate Office-Home K-Bound runs into the full honest panel.

Reads the SOURCE (Real_World/val), TARGET-VAL and (optionally) TARGET-TEST manifests,
runs the source-calibrated analyzer (regime scan + route-a/route-b verdict), and writes:
  - panel.csv           : every (domain, candidate) -> n, mean_B, base_rate_harmful,
                          frac_helpful, regime, harm_AUC, detect  (NO row dropped)
  - REGIME_SCAN.json    : the val dev-regime scan (STEP 1 checkpoint)
  - VERDICT_val.json    : source-calibrated router verdict on VAL
  - VERDICT_test.json   : source-calibrated router verdict on HELD-OUT TEST (if given)
  - MANIFEST.json       : consolidated pointer + headline verdict (beats_both T/F)
Every number traces to the input manifests' records[]/conditions[].
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "wilds"))
import oh_analyze as oha


def panel_rows(records, names, split_label):
    doms = sorted(set(r["domain"] for r in records))
    cands = sorted(set(r["candidate"] for r in records))
    rows = []
    for dom in doms:
        for c in cands:
            rs = [r for r in records if r["domain"] == dom and r["candidate"] == c]
            if not rs:
                continue
            B = np.array([r["B"] for r in rs]); Z = np.array([r["Z"] for r in rs])
            harm = (B < 0).astype(int)
            hauc = None
            if harm.sum() not in (0, len(harm)):
                hauc, _ = oha.best_feature_auc(Z, harm, names)
            reg, det, base_h, meanB = oha.regime_of(B, hauc)
            rows.append({"split": split_label, "domain": dom, "candidate": c, "n": len(rs),
                         "mean_B": round(meanB, 4), "base_rate_harmful": round(base_h, 3),
                         "frac_helpful": round(float(np.mean(B > oha.THR)), 3),
                         "regime": reg, "harm_AUC": (round(hauc, 3) if hauc is not None else ""),
                         "detect": det})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target-val", required=True)
    p.add_argument("--target-test", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="officehome_v1")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    so, sr, scd, names = oha.load(args.source)
    vo, vr, vcd, namesv = oha.load(args.target_val)
    names = names or namesv
    src = oha.recompute(sr, scd); tval = oha.recompute(vr, vcd)

    scan = {"schema": "officehome_regime_scan_v1", **oha.regime_scan(src, tval, names)}
    json.dump(scan, open(out / "REGIME_SCAN.json", "w"), indent=2, default=float)
    vval = {"schema": "officehome_verdict_val", **oha.verdict(src, tval, scd, vcd, names)}
    json.dump(vval, open(out / "VERDICT_val.json", "w"), indent=2, default=float)

    rows = panel_rows(src, names, "source(RealWorld/val)") + panel_rows(tval, names, "target/val")
    vtest = None
    if args.target_test:
        to, tr, tcd, namest = oha.load(args.target_test)
        ttest = oha.recompute(tr, tcd)
        vtest = {"schema": "officehome_verdict_test_HELDOUT", **oha.verdict(src, ttest, scd, tcd, names)}
        json.dump(vtest, open(out / "VERDICT_test.json", "w"), indent=2, default=float)
        rows += panel_rows(ttest, names, "target/TEST")

    with open(out / "panel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "domain", "candidate", "n", "mean_B",
                                          "base_rate_harmful", "frac_helpful", "regime", "harm_AUC", "detect"])
        w.writeheader(); w.writerows(rows)

    manifest = {
        "schema": "officehome_kbound_consolidated_v1", "tag": args.tag,
        "inputs": {"source": args.source, "target_val": args.target_val, "target_test": args.target_test or None},
        "deployed_adapter_source_chosen": scan["deployed_adapter_source_chosen"],
        "win_precondition_val_gradTTA_mixed_detectable": scan["win_precondition_met"],
        "gradTTA_detectability_per_domain": {
            d: {"regime": v.get("regime"), "best_harm_AUC": v.get("best_harm_AUC"),
                "certificate_transfer_AUC": v.get("certificate_transfer_AUC"),
                "base_rate_harmful": v.get("base_rate_harmful"), "mean_B": v.get("mean_B")}
            for d, v in scan["gradient_TTA_detectability_DECISIVE"]["per_domain"].items()},
        "gradTTA_detectability_pooled": scan["gradient_TTA_detectability_DECISIVE"]["pooled_all_targets"],
        "any_domain_deployed_mixed_detectable_val": scan["ANY_domain_deployed_mixed_detectable"],
        "val_verdict_regime": vval["regime"], "val_goldilocks": vval["goldilocks"],
        "val_beats_both": vval["beats_both"],
        "test_verdict_regime": (vtest["regime"] if vtest else None),
        "test_goldilocks": (vtest["goldilocks"] if vtest else None),
        "HELD_OUT_TEST_beats_both": (vtest["beats_both"] if vtest else None),
        "n_panel_rows": len(rows),
    }
    json.dump(manifest, open(out / "MANIFEST.json", "w"), indent=2, default=float)
    print("WROTE:", out)
    print(json.dumps(manifest, indent=2, default=float))


if __name__ == "__main__":
    main()
