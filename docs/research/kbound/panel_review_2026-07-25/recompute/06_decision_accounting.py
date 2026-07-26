#!/usr/bin/env python3
"""Fix-queue item 5 (decision accounting for EVERY track) + the remaining half of
item 4 (leave-one-out-of-pool radius on the natural-shift tracks) + item 23
(per-seed spread behind every multi-seed mean).

For every track with per-cell artifacts on disk this reports:
   N cells, ADAPT / FREEZE / ABSTAIN counts,
   observed false-adapt count (marginal FA_u and conditional FA_c),
   Clopper-Pearson 95% UPPER bound on FA_c (and on FA_u),
   the structural miscoverage ceiling (N-k)/N implied by in-sample rank
     calibration, which is what makes "FA_u <= alpha" an identity (F1-1),
   regret triple under the shipped in-pool radius and under leave-one-out-of-pool,
   how many decisions change.

Tracks whose promoted per-cell source is absent from the release are emitted with
status "BLOCKED-NEEDS-DATA" and the missing path.

Run: python3 06_decision_accounting.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import (ALPHA, REPO, clopper_pearson_upper, decide, radii_in_pool,
                       radii_loo, read_json, records, score, wilson)

R = "experiments/kbound/results"


def ceiling(n, alpha=ALPHA):
    """(n-k)/n with k = min(n, ceil((n+1)(1-alpha))) -- the largest FA_u that an
    in-sample exact-rank radius can produce, for ANY data (F1-1)."""
    k = min(n, int(math.ceil((n + 1) * (1 - alpha))))
    return (n - k) / n, k


def ceiling_interp(residuals):
    """Exceedance fraction of the interpolated np.quantile rule at this n."""
    r = np.asarray(residuals, float)
    q = np.quantile(r, 1 - ALPHA)
    return float(np.mean(r > q))


def load_cells(paths):
    B, bh, a0, aad, cond, dec_file, seedv = [], [], [], [], [], [], []
    for p in paths:
        r = records(p)
        B += [x["B"] for x in r]
        bh += [x.get("b_hat") for x in r]
        a0 += [x["a0"] for x in r]
        aad += [x["a_adapted"] for x in r]
        cond += [x.get("condition") for x in r]
        dec_file += [x.get("kga_decision") for x in r]
        seedv += [x.get("seed", None) for x in r]
    return (np.array(B, float), np.array(bh, float), np.array(a0, float),
            np.array(aad, float), cond, dec_file, seedv)


def per_file_scored(paths, rule, pool):
    """Radius is fitted PER FILE (per seed), as every shipped runner does."""
    dec_all, B_all, a0_all, aad_all, eps_all = [], [], [], [], []
    per_file = []
    for p in paths:
        r = records(p)
        B = np.array([x["B"] for x in r], float)
        bh = np.array([x["b_hat"] for x in r], float)
        a0 = np.array([x["a0"] for x in r], float)
        aad = np.array([x["a_adapted"] for x in r], float)
        eps = radii_in_pool(bh, B, rule) if pool == "in_pool" else radii_loo(bh, B, rule)
        dec = decide(bh, eps)
        sc = score(dec, B, a0, aad)
        c, k = ceiling(len(B))
        per_file.append({"path": p, "n": len(B), "eps_min": float(eps.min()),
                         "eps_max": float(eps.max()),
                         "regret": [sc["regret_kga"], sc["regret_adapt"], sc["regret_freeze"]],
                         "adapt": sc["n_adapt"], "freeze": sc["n_freeze"],
                         "abstain": sc["n_abstain"], "fa_num": sc["fa_num"],
                         "fa_u": sc["fa_u"], "fa_c": sc["fa_c"],
                         "harmful_frac": sc["harmful_frac"],
                         "structural_fa_u_ceiling": c, "exact_rank_k": k,
                         "miscovered_count": int(np.sum(np.abs(bh - B) > eps)),
                         "decisions": list(dec)})
        dec_all += list(dec); B_all += list(B); a0_all += list(a0); aad_all += list(aad)
        eps_all += list(eps)
    sc = score(dec_all, B_all, a0_all, aad_all)
    agg = {
        "n": sc["n"],
        "regret": [sc["regret_kga"], sc["regret_adapt"], sc["regret_freeze"]],
        "regret_mean_over_files": [float(np.mean([f["regret"][i] for f in per_file]))
                                   for i in range(3)],
        "adapt": sc["n_adapt"], "freeze": sc["n_freeze"], "abstain": sc["n_abstain"],
        "fa_num": sc["fa_num"], "fa_u": sc["fa_u"], "fa_c": sc["fa_c"],
        "cp95_upper_fa_c": clopper_pearson_upper(sc["fa_num"], sc["n_adapt"]),
        "cp95_upper_fa_u": clopper_pearson_upper(sc["fa_num"], sc["n"]),
        "wilson95_fa_u": wilson(sc["fa_num"], sc["n"]),
        "harmful_frac": sc["harmful_frac"],
        "eps_range": [float(min(eps_all)), float(max(eps_all))],
        "guarantee_untested_lt10_adapts": bool(sc["n_adapt"] < 10),
        "beats_both_point": bool(sc["regret_kga"] < sc["regret_adapt"] - 1e-12
                                 and sc["regret_kga"] < sc["regret_freeze"] - 1e-12),
    }
    return agg, per_file, dec_all


def track(name, paths, note=""):
    out = {"track": name, "note": note, "artifacts": paths}
    missing = [p for p in paths if not os.path.exists(os.path.join(REPO, p))]
    if missing:
        out["status"] = "BLOCKED-NEEDS-DATA"
        out["missing"] = missing
        return out
    try:
        for rule in ("interp", "exact"):
            for pool in ("in_pool", "loo"):
                agg, pf, dec = per_file_scored(paths, rule, pool)
                out[f"{rule}_{pool}"] = agg
                out[f"{rule}_{pool}_per_file"] = [
                    {k: v for k, v in f.items() if k != "decisions"} for f in pf]
                out[f"_dec_{rule}_{pool}"] = dec
        for rule in ("interp", "exact"):
            a = np.array(out[f"_dec_{rule}_in_pool"], dtype=object)
            b = np.array(out[f"_dec_{rule}_loo"], dtype=object)
            out[f"{rule}_decisions_changed_by_loo"] = int(np.sum(a != b))
        for rule in ("interp", "exact"):
            for pool in ("in_pool", "loo"):
                out.pop(f"_dec_{rule}_{pool}")
        out["status"] = "OK"
    except IOError as e:
        out["status"] = "BLOCKED-NEEDS-DATA"
        out["error"] = str(e)
    return out


def g(pattern):
    return sorted(os.path.relpath(p, REPO)
                  for p in glob.glob(os.path.join(REPO, pattern)))


TRACKS = [
    ("CIFAR-10-C Tent (promoted panel source, head-to-head KGA arm, 5 seeds)",
     [f"{R}/mixed_headtohead_v1/per_condition_cifar10c_tent_primary_kga_seed{s}.json"
      for s in range(5)],
     "manifest cifar10c_tent 0.0015736109/0.0079233799/0.1240979162"),
    ("CIFAR-10-C EATA (promoted panel source, head-to-head KGA arm, 5 seeds)",
     [f"{R}/mixed_headtohead_v1/per_condition_cifar10c_eata_secondary_kga_seed{s}.json"
      for s in range(5)],
     "manifest cifar10c_eata 0.0012675925/0.0032682874/0.1313789343"),
    ("CIFAR-10-C Tent (stress grid, seeds 1-4; seed 0 per-condition dump absent)",
     [f"{R}/stress_grid_multiseed_v1/seed{s}/per_condition_cifar10c_tent_seed{s}.json"
      for s in (1, 2, 3, 4)], "fix-queue item 8: seed0 dump missing"),
    ("CIFAR-10-C EATA (stress grid, seeds 1-4)",
     [f"{R}/stress_grid_multiseed_v1/seed{s}/per_condition_cifar10c_eata_seed{s}.json"
      for s in (1, 2, 3, 4)], ""),
    ("CIFAR-10-C SAR (stress grid, seeds 1-4) [QUARANTINED]",
     [f"{R}/stress_grid_multiseed_v1/seed{s}/per_condition_cifar10c_sar_seed{s}.json"
      for s in (1, 2, 3, 4)], "fix-queue item 6"),
    ("ImageNet-C SAR (promoted, pooled_5seed)",
     [f"{R}/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed{s}.json"
      for s in range(5)], "manifest imagenetc_sar 0.026422222/0.0529333334/0.0318944445"),
    ("ImageNet-C Tent (pooled_5seed)",
     [f"{R}/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_tent_seed{s}.json"
      for s in range(5)], ""),
    ("ImageNet-C EATA (pooled_5seed)",
     [f"{R}/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_eata_seed{s}.json"
      for s in range(5)], ""),
    ("Camelyon17 Tent (Table VIII source, wilds_kbound, 4 seeds x 9)",
     [f"{R}/wilds_kbound/per_condition_camelyon17_tent_seed{s}.json" for s in range(4)],
     "kbound_short.tex:889 0.020+-0.023 / 0.138 / 0.020 / FA 0.00"),
    ("Camelyon17 EATA (Table VIII source)",
     [f"{R}/wilds_kbound/per_condition_camelyon17_eata_seed{s}.json" for s in range(4)],
     "kbound_short.tex:890"),
    ("Camelyon17 SAR (Table VIII source)",
     [f"{R}/wilds_kbound/per_condition_camelyon17_sar_seed{s}.json" for s in range(4)],
     "kbound_short.tex:891 'over-freezes'"),
    ("Camelyon17 fullscale_B_v2 Tent (36 cells x 3 seeds)",
     [f"{R}/camelyon17_fullscale_B_v2/per_condition_camelyon17_tent_seed{s}.json"
      for s in range(3)], ""),
    ("Camelyon17 OOD promoted row (n=18, 0.0000/0.0000/0.1381)",
     [f"{R}/../../docs/research/kbound/audits/integrity_2026-06-20/camelyon_reconciliation/records.json"],
     "fix-queue item 10: source directory does not exist"),
    ("iWildCam tent_episodic (multiseed extracted, 72 cells x 2 seeds)",
     [f"{R}/multiseed/iwildcam/extracted/per_condition_iwildcam_tent_episodic_seed{s}.json"
      for s in range(2)], "promoted iwildcam_H_v2 0.0041023691/0.1028299605/0.0041023691"),
    ("Office-Home sar_online_aggressive (multiseed extracted, 36 cells x 5 seeds)",
     [f"{R}/multiseed/officehome/extracted/per_condition_officehome_sar_online_aggressive_seed{s}.json"
      for s in range(5)], "promoted officehome_M_v2 0.0157142857/0.0468131868/0.0158241758 (n=35)"),
    ("RxRx1 sar_online (multiseed extracted, 12 cells x 5 seeds = 60)",
     [f"{R}/multiseed/rxrx1/extracted/per_condition_rxrx1_sar_online_seed{s}.json"
      for s in range(5)], "promoted rxrx1_J 0.0/0.2531/0.0, adapt_rate 0.0"),
    ("CIFAR-10.1 Tent (cifar101_multiseed_v1, 24 cells x 5 seeds)",
     [f"{R}/cifar101_multiseed_v1/seed{s}/per_condition_cifar101_tent_seed{s}.json"
      for s in range(5)], "promoted cifar10_1_K FA_u 0.1667 / FA_c 0.4444, n=48"),
    ("CIFAR-10.1 SAR (cifar101_multiseed_v1)",
     [f"{R}/cifar101_multiseed_v1/seed{s}/per_condition_cifar101_sar_seed{s}.json"
      for s in range(5)], ""),
    ("CIFAR-10.1 EATA (cifar101_multiseed_v1)",
     [f"{R}/cifar101_multiseed_v1/seed{s}/per_condition_cifar101_eata_seed{s}.json"
      for s in range(5)], ""),
]

# ImageNet-R: one track per backbone (item 23 asks for min/median/max)
IMR = f"{R}/imagenetr_protocol_d_multiseed_v1"
BACKBONES = ["convnext_base", "convnext_tiny", "efficientnet_b0", "efficientnet_b3",
             "resnet101", "resnet152", "resnext101_32x8d", "swin_b", "swin_t", "vit_b_16"]
for bb in BACKBONES:
    TRACKS.append((f"ImageNet-R D / {bb} (12 cells x 4 seeds)",
                   [f"{IMR}/per_condition_imagenet-r_{bb}_seed{s}.json" for s in range(4)],
                   "panel row is the MEAN across these ten backbones"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for name, paths, note in TRACKS:
        out.append(track(name, paths, note))

    json.dump(out, open(os.path.join(here, "out_decision_accounting.json"), "w"),
              indent=1, default=float)

    hdr = (f"{'track':64s} {'N':>5s} {'ADAPT':>6s} {'FREEZE':>6s} {'ABST':>5s} "
           f"{'FAu':>7s} {'FAc':>7s} {'CP95up(FAc)':>12s} {'ceil':>7s} {'chg':>4s}")
    print(hdr); print("-" * len(hdr))
    for o in out:
        if o["status"] != "OK":
            print(f"{o['track'][:64]:64s}  BLOCKED-NEEDS-DATA "
                  f"{o.get('missing', [o.get('error','')])[0] if o.get('missing') else o.get('error','')}")
            continue
        a = o["exact_in_pool"]
        pf = o["exact_in_pool_per_file"][0]
        cp = a["cp95_upper_fa_c"]
        print(f"{o['track'][:64]:64s} {a['n']:5d} {a['adapt']:6d} {a['freeze']:6d} "
              f"{a['abstain']:5d} {a['fa_u']:7.4f} "
              f"{(a['fa_c'] if a['fa_c'] is not None else float('nan')):7.4f} "
              f"{(cp if cp is not None else float('nan')):12.5f} "
              f"{pf['structural_fa_u_ceiling']:7.4f} "
              f"{o['exact_decisions_changed_by_loo']:4d}"
              f"{'   <-- guarantee untested (<10 ADAPT)' if a['guarantee_untested_lt10_adapts'] else ''}")
    print("\nwrote out_decision_accounting.json")


if __name__ == "__main__":
    main()
