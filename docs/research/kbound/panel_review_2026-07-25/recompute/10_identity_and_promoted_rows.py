#!/usr/bin/env python3
"""Fix-queue item 5, parts (a) and (b).

(a) The structural-identity check behind F1-1 / the chief reviewer's arbitration:
    when eps is the k-th order statistic of the SAME residual vector it is then
    used to test, the number of miscovered cells is identically N-k, so
    FA_u <= (N-k)/N holds for any data whatsoever.  We verify this on every
    shipped per-condition file and report the two ceilings the chief quotes
    (exact-rank 0.0972 at n=432, 0.0370 at n=27; interpolated 0.1019 / 0.1111).

(b) Decision accounting for the promoted panel rows whose per-cell source is NOT
    in the release (Office-Home, iWildCam, Camelyon17-OOD, RxRx1, CIFAR-10.1,
    three-source, D33).  Counts are taken from the promoted summary artifacts
    (n_test x adapt_rate) and the Clopper-Pearson upper bound is computed from
    those integers, so every panel row gets an honest FA_c bound even where the
    raw cells are missing.

Run: python3 10_identity_and_promoted_rows.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import (ALPHA, REPO, clopper_pearson_upper, eps_exact, eps_interp,
                       read_json, records, wilson)

R = "experiments/kbound/results"


def identity_check():
    files = []
    for pat in [f"{R}/stress_grid_multiseed_v1/seed*/per_condition_cifar10c_*.json",
                f"{R}/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_*.json",
                f"{R}/mixed_headtohead_v1/per_condition_cifar10c_*_kga_seed*.json",
                f"{R}/wilds_kbound/per_condition_camelyon17_*.json",
                f"{R}/cifar101_multiseed_v1/seed*/per_condition_cifar101_*.json"]:
        files += sorted(os.path.relpath(p, REPO)
                        for p in glob.glob(os.path.join(REPO, pat)))
    rows = []
    for f in files:
        try:
            r = records(f)
        except IOError as e:
            rows.append({"path": f, "status": "PLACEHOLDER", "error": str(e)})
            continue
        B = np.array([x["B"] for x in r], float)
        bh = np.array([x["b_hat"] for x in r], float)
        rho = np.abs(bh - B)
        n = len(rho)
        k = min(n, int(math.ceil((n + 1) * (1 - ALPHA))))
        e_ex = eps_exact(rho)
        e_in = eps_interp(rho)
        rows.append({
            "path": f, "status": "OK", "n": n, "k": k,
            "exact_miscovered": int(np.sum(rho > e_ex)),
            "exact_max_possible_miscovered": n - k,
            "exact_hits_ceiling": bool(int(np.sum(rho > e_ex)) == n - k),
            "exact_fa_u_ceiling": (n - k) / n,
            "interp_miscovered": int(np.sum(rho > e_in)),
            "interp_exceedance_fraction": float(np.mean(rho > e_in)),
            "interp_coverage_observed": float(np.mean(rho <= e_in)),
        })
    ok = [r for r in rows if r["status"] == "OK"]
    summary = {
        "n_files": len(rows),
        "n_hitting_exact_ceiling": sum(1 for r in ok if r["exact_hits_ceiling"]),
        "n_ok": len(ok),
        "ceilings_by_n": {},
    }
    for r in ok:
        summary["ceilings_by_n"].setdefault(str(r["n"]), {
            "exact_rank_k": r["k"],
            "exact_fa_u_ceiling": r["exact_fa_u_ceiling"],
            "interp_exceedance_fraction": r["interp_exceedance_fraction"],
            "interp_coverage_observed": r["interp_coverage_observed"],
            "exact_ceiling_below_alpha": r["exact_fa_u_ceiling"] <= ALPHA,
            "interp_exceedance_above_alpha": r["interp_exceedance_fraction"] > ALPHA,
        })
    return {"summary": summary, "files": rows}


def promoted_rows():
    """Every promoted panel row: N, ADAPT/FREEZE/ABSTAIN, FA counts, CP bound."""
    rows = []

    def add(name, n, adapt, freeze, abstain, fa_num, regret, source, note=""):
        rows.append({
            "track": name, "n": n, "adapt": adapt, "freeze": freeze, "abstain": abstain,
            "fa_num": fa_num,
            "fa_u": (fa_num / n if n else None),
            "fa_c": (fa_num / adapt if adapt else None),
            "cp95_upper_fa_c": clopper_pearson_upper(fa_num, adapt) if adapt else None,
            "cp95_upper_fa_u": clopper_pearson_upper(fa_num, n) if n else None,
            "guarantee_untested_lt10_adapts": (adapt is not None and adapt < 10),
            "regret_kga_adapt_freeze": regret, "source": source, "note": note})

    # ---- tracks recomputed cell-by-cell in 06_decision_accounting.py -------------
    here = os.path.dirname(os.path.abspath(__file__))
    da = json.load(open(os.path.join(here, "out_decision_accounting.json")))
    by = {t["track"]: t for t in da}

    def from_da(key, label, note=""):
        t = by[key]
        if t["status"] != "OK":
            add(label, None, None, None, None, None, None, t["artifacts"][0],
                "BLOCKED-NEEDS-DATA")
            return
        a = t["exact_in_pool"]
        add(label, a["n"], a["adapt"], a["freeze"], a["abstain"], a["fa_num"],
            [round(x, 8) for x in a["regret_mean_over_files"]], t["artifacts"][0], note)

    from_da("CIFAR-10-C Tent (promoted panel source, head-to-head KGA arm, 5 seeds)",
            "CIFAR-10-C Tent (5 seeds x 432)", "exact-rank radius, in-pool")
    from_da("CIFAR-10-C EATA (promoted panel source, head-to-head KGA arm, 5 seeds)",
            "CIFAR-10-C EATA (5 seeds x 432)", "exact-rank radius, in-pool")
    from_da("ImageNet-C SAR (promoted, pooled_5seed)",
            "ImageNet-C SAR (5 seeds x 27)", "exact-rank radius, in-pool")

    # ---- tracks whose per-cell source is missing: counts from the summaries -----
    oh = read_json(f"{R}/officehome_protocol_M_v2/protocol_result.json")["test_locked"]
    n = oh["n_test"]; ad = int(round(oh["adapt_rate"] * n))
    cov = int(round(oh["commit_rate"] * n))
    add("Office-Home M v2 (promoted panel row)", n, ad, cov - ad, n - cov,
        int(round(oh["false_adapt"] * ad)),
        [0.0157142857, 0.0468131868, 0.0158241758],
        f"{R}/officehome_protocol_M_v2/protocol_result.json",
        "panel regret triple is the OOF-lock value from research_lock/"
        "KBOUND_WIN_BOOTSTRAP_CIS_oof.json; this artifact's own regret_kga is "
        f"{oh['regret_kga']:.6f} (7.2x smaller). Raw record file absent (F3-6).")

    iw = read_json(f"{R}/iwildcam_protocol_H_v2/protocol_result.json")["test_locked"]
    n = iw["n_test"]; ad = int(round(iw["adapt_rate"] * n))
    cov = int(round(iw["commit_rate"] * n))
    add("iWildCam H v2 (promoted panel row)", n, ad, cov - ad, n - cov,
        int(round(iw["false_adapt"] * ad)),
        [0.0041023691, 0.1028299605, 0.0041023691],
        f"{R}/iwildcam_protocol_H_v2/protocol_result.json",
        "promoted OOF regret_kga == regret_freeze to 18 digits; this artifact's "
        f"regret_kga is {iw['regret_kga']:.10f}. Raw record file absent (F3-6).")

    add("Camelyon17 OOD (promoted panel row, n=18)", 18, None, None, None, 0,
        [0.0, 0.0, 0.1381],
        "docs/research/kbound/audits/integrity_2026-06-20/camelyon_reconciliation/",
        "BLOCKED-NEEDS-DATA: source directory does not exist; the triple "
        "0.0000/0.0000/0.1381 appears in no artifact on disk (F3-6, F4-12). "
        "Live Camelyon artifacts give false_adapt 0.0256 (n=54) and 0.0329 (n=324).")

    rx = read_json(f"{R}/rxrx1_protocol_J_v1/analyze_F_results.json")["test_locked"]
    n = rx["n_test"]; ad = int(round(rx["adapt_rate"] * n))
    cov = int(round(rx["commit_rate"] * n))
    add("RxRx1 J (promoted panel row)", n, ad, cov - ad, n - cov, 0,
        [0.0, 0.2531, 0.0], f"{R}/rxrx1_protocol_J_v1/analyze_F_results.json",
        f"adapt_rate 0.0: KGA == always-freeze by construction. This artifact's "
        f"regret_adapt is {rx['regret_adapt']:.10f}; the 5-seed extracted per-cell "
        "files (seeds 0-4) give 0.258724 instead.")

    # FA_u 0.1667 = 8/48 and FA_c 0.4444 = 8/18  =>  18 ADAPT decisions
    c101 = read_json(f"{R}/cifar101_protocol_K_v1/analyze_F_results.json")["test_locked"]
    n = c101["n_test"]; ad = int(round(c101["adapt_rate"] * n))
    cov = int(round(c101["commit_rate"] * n))
    add("CIFAR-10.1 K (promoted diagnostic-fail row, n=48)", n, ad, cov - ad, n - cov,
        int(round(c101["false_adapt"] * ad)), [0.0021, 0.019, 0.0017],
        f"{R}/cifar101_protocol_K_v1/analyze_F_results.json",
        "FA_u 0.1667 = 8/48 and FA_c 0.4444 = 8/18; commit_rate 0.875 = 42/48. "
        "The manifest entry tracks/cifar10_1_K carries NO `source` field (F4-11); "
        "this file is the source and it exists -- write it in.")

    ts = read_json(f"{R}/mixed_protocol_oof_v2/mixed_protocol_oof_v2_result.json")
    rows.append({"track": "three-source OOF mixture (constructed)",
                 "n": 143, "raw_summary_keys": sorted(ts.keys())[:12],
                 "regret_kga_adapt_freeze": [0.0059117, 0.0632323, 0.0342043],
                 "source": f"{R}/mixed_protocol_oof_v2/mixed_protocol_oof_v2_result.json",
                 "note": "each dataset keeps its own dev-calibrated radius, so the "
                         "router is handed dataset identity (F3-14)."})

    d33 = read_json(f"{R}/controlled_multimodal_d33/results.json")
    rows.append({"track": "controlled multimodal D33",
                 "n": 130, "adapt": 9, "freeze": 119, "abstain": 2, "fa_num": 0,
                 "fa_u": 0.0, "fa_c": 0.0,
                 "cp95_upper_fa_c": clopper_pearson_upper(0, 9),
                 "cp95_upper_fa_u": clopper_pearson_upper(0, 130),
                 "guarantee_untested_lt10_adapts": True,
                 "source": f"{R}/controlled_multimodal_d33/results.json",
                 "note": "counts taken from kbound_result_manifest.json "
                         "tracks/controlled_multimodal_D33.decision_counts"})
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ident = identity_check()
    prom = promoted_rows()
    json.dump({"identity": ident, "promoted_rows": prom},
              open(os.path.join(here, "out_identity_promoted.json"), "w"),
              indent=1, default=float)

    s = ident["summary"]
    print("=" * 104)
    print(f"F1-1 structural identity check: {s['n_hitting_exact_ceiling']} of {s['n_ok']} "
          f"shipped per-condition files hit the exact-rank miscoverage ceiling EXACTLY")
    print(f"{'n':>6s} {'k':>5s} {'exact FA_u ceiling':>19s} {'<= alpha?':>10s} "
          f"{'interp exceedance':>18s} {'> alpha?':>9s} {'interp coverage':>16s}")
    for n, v in sorted(ident["summary"]["ceilings_by_n"].items(), key=lambda kv: -int(kv[0])):
        print(f"{n:>6s} {v['exact_rank_k']:5d} {v['exact_fa_u_ceiling']:19.6f}"
              f" {str(v['exact_ceiling_below_alpha']):>10s}"
              f" {v['interp_exceedance_fraction']:18.6f}"
              f" {str(v['interp_exceedance_above_alpha']):>9s}"
              f" {v['interp_coverage_observed']:16.6f}")

    print("\n" + "=" * 104)
    print("Promoted panel rows -- decision accounting")
    print(f"{'track':46s} {'N':>5s} {'AD':>5s} {'FR':>5s} {'AB':>5s} {'FA':>4s} "
          f"{'FA_u':>8s} {'CP95up(FA_c)':>13s}")
    for r in prom:
        if r.get("n") is None:
            print(f"{r['track'][:46]:46s}  BLOCKED-NEEDS-DATA  {r['source']}")
            continue
        cp = r.get("cp95_upper_fa_c")
        fau = r.get("fa_u")
        fau_s = f"{fau:.5f}" if fau is not None else "n/a"
        cp_s = f"{cp:.5f}" if cp is not None else "undefined"
        flag = "  <-- guarantee untested" if r.get("guarantee_untested_lt10_adapts") else ""
        print(f"{r['track'][:46]:46s} {str(r.get('n')):>5s} {str(r.get('adapt')):>5s} "
              f"{str(r.get('freeze')):>5s} {str(r.get('abstain')):>5s} "
              f"{str(r.get('fa_num')):>4s} {fau_s:>8s} {cp_s:>13s}{flag}")
    print("\nwrote out_identity_promoted.json")


if __name__ == "__main__":
    main()
