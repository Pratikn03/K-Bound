#!/usr/bin/env python3
"""Bootstrap 95% CIs on the three dev-locked natural-shift wins.

FAITHFUL bootstrap: we re-run the protocol's OWN scorer (score_kbound_holdout.
score_transfer = GBR benefit estimator fit on DEV + global conformal eps) on each
resample. We resample ONLY the held-out TEST conditions (with replacement); the
DEV calibration that fixes B_hat and eps is held constant. This means the decision
rule is never re-tuned on test -> no p-hacking. The point estimate of every run
reproduces the locked protocol_result.json to 4 decimals (sanity-checked below).

Reported per win:
  delta_freeze = regret_freeze - regret_kga   (>0 => KGA better than always-freeze)
  delta_adapt  = regret_adapt  - regret_kga   (>0 => KGA better than always-adapt)
A 95% CI that excludes 0 => statistically robust beat on that axis.
"beats-both (robust)" requires BOTH CIs to exclude 0.

Usage:  python bootstrap_win_cis.py [--B 2000] [--out research_lock/KBOUND_WIN_BOOTSTRAP_CIS.json]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import analyze_F as af          # noqa: E402
import score_kbound_holdout as sk  # noqa: E402


def filt(recs, seeds):
    s = set(seeds)
    return [r for r in recs if int(r["seed"]) in s]


# (name, records-source / cal+test paths, candidate, dev_seeds, test_seeds, stored-file)
WINS = [
    dict(name="OfficeHome", mode="transfer",
         cal="experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json",
         test="experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json",
         candidate="sar_online_aggressive", dev_seeds=[0, 1], test_seeds=[0, 1],
         stored="experiments/kbound/results/officehome_protocol_M_v2/protocol_result.json"),
    dict(name="iWildCam", mode="seed_split",
         records="experiments/kbound/results/iwildcam_full_test/result_e40faf29.json",
         candidate="tent_episodic", dev_seeds=[0], test_seeds=[1],
         stored="experiments/kbound/results/iwildcam_protocol_H_v2/protocol_result.json"),
    dict(name="Camelyon17", mode="seed_split",
         records="experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json",
         candidate="eata_online", dev_seeds=[0, 1], test_seeds=[2, 3, 4],
         stored="experiments/kbound/results/camelyon17_protocol_G_v1/analyze_F_results.json"),
]


def load_cal_test(w):
    if w["mode"] == "transfer":
        cal = filt(af.load_records(str(ROOT / w["cal"]), candidate=w["candidate"])[0], w["dev_seeds"])
        test = filt(af.load_records(str(ROOT / w["test"]), candidate=w["candidate"])[0], w["test_seeds"])
    else:
        recs = af.load_records(str(ROOT / w["records"]), candidate=w["candidate"])[0]
        cal = filt(recs, w["dev_seeds"]); test = filt(recs, w["test_seeds"])
    return cal, test


def stored_regrets(w):
    d = json.load(open(ROOT / w["stored"]))
    tl = d.get("test_locked", d)
    return tl["regret_kga"], tl["regret_adapt"], tl["regret_freeze"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="research_lock/KBOUND_WIN_BOOTSTRAP_CIS.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {"method": "resample held-out test conditions; GBR+global eps fixed on dev; protocol scorer reused",
           "B": a.B, "alpha": af.ALPHA, "wins": []}
    for w in WINS:
        cal, test = load_cal_test(w)
        m0 = sk.score_transfer(cal, test, "gbr", "global")
        sk_kga, sk_ad, sk_fr = stored_regrets(w)
        n = len(test)
        df = np.empty(a.B); da = np.empty(a.B); fa = np.empty(a.B)
        for b in range(a.B):
            tb = [test[i] for i in rng.integers(0, n, n)]
            mb = sk.score_transfer(cal, tb, "gbr", "global")
            df[b] = mb["regret_freeze"] - mb["regret_kga"]
            da[b] = mb["regret_adapt"] - mb["regret_kga"]
            fa[b] = mb["false_adapt"]
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        rec = dict(
            name=w["name"], n_test=n, candidate=w["candidate"],
            point=dict(regret_kga=m0["regret_kga"], regret_adapt=m0["regret_adapt"],
                       regret_freeze=m0["regret_freeze"], false_adapt=m0["false_adapt"]),
            reproduces_locked=bool(abs(m0["regret_kga"] - sk_kga) < 1e-6
                                   and abs(m0["regret_adapt"] - sk_ad) < 1e-6
                                   and abs(m0["regret_freeze"] - sk_fr) < 1e-6),
            kga_vs_freeze=dict(mean=float(df.mean()), ci95=ci(df), p_better=float((df > 0).mean()),
                               ci_excludes_zero=bool(np.percentile(df, 2.5) > 0)),
            kga_vs_adapt=dict(mean=float(da.mean()), ci95=ci(da), p_better=float((da > 0).mean()),
                              ci_excludes_zero=bool(np.percentile(da, 2.5) > 0)),
            false_adapt_ci95=ci(fa),
        )
        rec["beats_both_robust"] = bool(rec["kga_vs_freeze"]["ci_excludes_zero"]
                                        and rec["kga_vs_adapt"]["ci_excludes_zero"])
        out["wins"].append(rec)
        print(f"{w['name']:11s} n={n:3d} reprod={rec['reproduces_locked']} "
              f"vsFREEZE {rec['kga_vs_freeze']['mean']:+.4f} {rec['kga_vs_freeze']['ci95']} excl0={rec['kga_vs_freeze']['ci_excludes_zero']} | "
              f"vsADAPT {rec['kga_vs_adapt']['mean']:+.4f} {rec['kga_vs_adapt']['ci95']} excl0={rec['kga_vs_adapt']['ci_excludes_zero']} | "
              f"beats_both_robust={rec['beats_both_robust']}")
    Path(ROOT / a.out).write_text(json.dumps(out, indent=2))
    print("saved", ROOT / a.out)


if __name__ == "__main__":
    raise SystemExit(main())
