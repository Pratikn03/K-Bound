#!/usr/bin/env python3
"""Assemble /home/claude/kb_fixes/NUMBERS_PACK.json from the outputs of scripts 01-10.

Every entry:
  {id, description, value, old_value, artifact_paths, method, changes_decisions, note}

Run: python3 11_build_numbers_pack.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/claude/kb_fixes/NUMBERS_PACK.json"

L = lambda f: json.load(open(os.path.join(HERE, f)))
inc = L("out_imagenetc_perseed.json")
boot = L("out_imagenetc_boot.json")
cifloo = L("out_cifar_loo.json")
clus = L("out_cifar_cluster.json")
loco = L("out_cifar_loco.json")
da = {t["track"]: t for t in L("out_decision_accounting.json")}
ident = L("out_identity_promoted.json")
pv = L("out_panel_variance.json")
gates = L("out_gates.json")
env = L("out_env.json")

E = []


def add(id, description, value, old_value=None, artifact_paths=None, method="",
        changes_decisions=None, note="", item=None):
    E.append({"id": id, "fix_queue_item": item, "description": description,
              "value": value, "old_value": old_value,
              "artifact_paths": artifact_paths or [], "method": method,
              "changes_decisions": changes_decisions, "note": note})


IC = "experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed{0..4}.json"
H2H_T = "experiments/kbound/results/mixed_headtohead_v1/per_condition_cifar10c_tent_primary_kga_seed{0..4}.json"
H2H_E = "experiments/kbound/results/mixed_headtohead_v1/per_condition_cifar10c_eata_secondary_kga_seed{0..4}.json"
SG = "experiments/kbound/results/stress_grid_multiseed_v1/seed{1..4}/per_condition_cifar10c_{tent,eata}_seed{s}.json"

# ============================================================ item 2: one rule
sar = inc["sar"]
add("item2.imagenetc_sar.pooled.exact_rank.regret",
    "ImageNet-C SAR pooled 5-seed regret triple (KGA / always-adapt / always-freeze) "
    "under the PROMOTED exact-rank radius, in-pool (reproduces the manifest)",
    sar["pooled"]["exact_in_pool"]["regret"],
    [0.010748, 0.052933, 0.031894],
    [IC], "eps = rho_(k), k=min(n,ceil((n+1)(1-alpha))), per seed; "
          "ADAPT iff bhat-eps>0, FREEZE iff bhat+eps<0; regret vs oracle=max(a0,a_adapted)",
    False,
    "old_value is the INTERPOLATED-rule pooled triple that kbound_short_appendix.tex:303-310 "
    "currently prints. Manifest tracks/imagenetc_sar reads "
    "[0.026422222, 0.0529333334, 0.0318944445] -- reproduced to 7 decimals.", 2)

add("item2.imagenetc_sar.pooled.exact_rank.actions",
    "ImageNet-C SAR pooled action composition under the exact-rank rule",
    {"n": 135, "ADAPT": sar["pooled"]["exact_in_pool"]["adapt"],
     "FREEZE": sar["pooled"]["exact_in_pool"]["freeze"],
     "ABSTAIN": sar["pooled"]["exact_in_pool"]["abstain"],
     "FA_u": sar["pooled"]["exact_in_pool"]["fa_u"]},
    {"n": 135, "ADAPT": sar["pooled"]["interp_in_pool"]["adapt"],
     "FREEZE": sar["pooled"]["interp_in_pool"]["freeze"],
     "ABSTAIN": sar["pooled"]["interp_in_pool"]["abstain"],
     "FA_u": sar["pooled"]["interp_in_pool"]["fa_u"]},
    [IC], "as above", False,
    "manifest abstain_count 109 reproduced exactly.", 2)

for s in range(5):
    r = sar["per_seed"][str(s)]
    add(f"item2.imagenetc_sar.seed{s}.exact_rank",
        f"ImageNet-C SAR seed {s}: regret triple, FA_u, action counts, exact-rank rule",
        {"regret": r["exact_in_pool"]["regret"], "fa_u": r["exact_in_pool"]["fa_u"],
         "fa_num": r["exact_in_pool"]["fa_num"],
         "ADAPT": r["exact_in_pool"]["adapt"], "FREEZE": r["exact_in_pool"]["freeze"],
         "ABSTAIN": r["exact_in_pool"]["abstain"],
         "beats_both": r["exact_in_pool"]["beats_both"],
         "bit_identical_tie_with_always_freeze": r["exact_in_pool"]["ties_freeze_exactly"],
         "cp95_upper_fa_c": r["exact_in_pool"]["cp95_upper_fa_c"]},
        {"regret": r["interp_in_pool"]["regret"], "fa_u": r["interp_in_pool"]["fa_u"],
         "ADAPT": r["interp_in_pool"]["adapt"], "FREEZE": r["interp_in_pool"]["freeze"],
         "ABSTAIN": r["interp_in_pool"]["abstain"],
         "beats_both": r["interp_in_pool"]["beats_both"]},
        [IC.replace("{0..4}", str(s))], "exact-rank vs interpolated radius, in-pool",
        None, "", 2)

n_bb = sum(1 for s in range(5) if sar["per_seed"][str(s)]["exact_in_pool"]["beats_both"])
n_tie = sum(1 for s in range(5)
            if sar["per_seed"][str(s)]["exact_in_pool"]["ties_freeze_exactly"])
tie_seeds = [s for s in range(5) if sar["per_seed"][str(s)]["exact_in_pool"]["ties_freeze_exactly"]]
bb_seeds = [s for s in range(5) if sar["per_seed"][str(s)]["exact_in_pool"]["beats_both"]]
add("item2.imagenetc_sar.seeds_beating_both",
    "Number of ImageNet-C SAR seeds on which KGA strictly improves BOTH fixed-policy "
    "regrets, under the promoted exact-rank rule",
    {"n_beats_both": n_bb, "seeds_beating_both": bb_seeds,
     "n_bit_identical_ties_with_always_freeze": n_tie, "tie_seeds": tie_seeds},
    {"n_beats_both": 5, "claim_in_paper": "improve both fixed-policy regrets on 5/5 seeds"},
    [IC], "exact-rank radius per seed", None,
    "CONFIRMS the panel: on seeds 0, 1 and 3 KGA never adapts and its regret is "
    "bit-identical to always-freeze (0.031926/0.031926, 0.031241/0.031241, "
    "0.029028/0.029028). The pooled win is driven by seeds 2 and 4.", 2)

# ============================================ item 3: bootstrap at the right unit
b_ex = boot["sar_exact_in_pool"]["designs"]
add("item3.imagenetc_sar.ci.seedavg27.exact_rank",
    "ImageNet-C SAR paired bootstrap at the unit the text describes "
    "(seed-averaged, 27 conditions), promoted exact-rank radius, 20000 replicates, "
    "rng seed 20260720",
    {"adapt_gap_ci95": [b_ex["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["adapt_gap"]["lo"],
                        b_ex["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["adapt_gap"]["hi"]],
     "freeze_gap_ci95": [b_ex["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["freeze_gap"]["lo"],
                         b_ex["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["freeze_gap"]["hi"]],
     "adapt_gap_excludes_zero": b_ex["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["beats_adapt_ci"],
     "freeze_gap_excludes_zero": b_ex["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["beats_freeze_ci"],
     "beats_both_ci": b_ex["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["beats_both_ci"]},
    {"adapt_gap_ci95": [-0.0518416198, -0.0038235196],
     "freeze_gap_ci95": [-0.0086428244, -0.0026000001],
     "beats_both_ci": True,
     "unit": "135 cell-seed rows resampled i.i.d. (g8_exactrank_ci.py:18)"},
    [IC, "docs/research/kbound/scripts/g8_exactrank_ci.py:18"],
    "pooled = mean over 5 seeds per condition, then paired percentile bootstrap over "
    "the 27 conditions (the design _locked_analysis_script.py:54 uses for CIFAR)",
    None,
    "The gap to always-ADAPT is NOT CI-supported at the condition level; the gap to "
    "always-FREEZE survives. Defensible claim: 'beats the better fixed policy "
    "(always-freeze) with a CI excluding zero'.", 3)

for nm, key in [("iid135_as_coded", "iid_135_cellseed_rows_AS_CODED"),
                ("cluster_by_condition", "cluster_by_condition_135_rows"),
                ("cluster_by_seed", "cluster_by_seed_135_rows"),
                ("cluster_by_corruption_family", "cluster_by_corruption_family_27_seedavg")]:
    d = b_ex[key]
    add(f"item3.imagenetc_sar.ci.{nm}.exact_rank",
        f"ImageNet-C SAR gap CIs, resampling unit = {key}",
        {"n_units": d["n_units"],
         "adapt_gap_ci95": [d["adapt_gap"]["lo"], d["adapt_gap"]["hi"]],
         "freeze_gap_ci95": [d["freeze_gap"]["lo"], d["freeze_gap"]["hi"]],
         "beats_both_ci": d["beats_both_ci"]},
        None, [IC], "paired percentile bootstrap, 20000 replicates, rng seed 20260720",
        None,
        "cluster_by_corruption_family has only 3 clusters (the ImageNet-C grid is "
        "gaussian_noise / shot_noise / impulse_noise only) -- a 3-cluster percentile "
        "bootstrap should not be reported as a primary interval.", 3)

b_loo = boot["sar_exact_loo"]["designs"]
add("item3.imagenetc_sar.ci.after_item4_loo_radius",
    "ImageNet-C SAR gap CIs AFTER applying the item-4 leave-one-out-of-pool radius",
    {"seedavg27_adapt_gap_ci95":
        [b_loo["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["adapt_gap"]["lo"],
         b_loo["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["adapt_gap"]["hi"]],
     "seedavg27_freeze_gap_ci95":
        [b_loo["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["freeze_gap"]["lo"],
         b_loo["seedavg_27_conditions_AS_TEXT_DESCRIBES"]["freeze_gap"]["hi"]],
     "iid135_freeze_gap_ci95": [b_loo["iid_135_cellseed_rows_AS_CODED"]["freeze_gap"]["lo"],
                                b_loo["iid_135_cellseed_rows_AS_CODED"]["freeze_gap"]["hi"]],
     "point_regret": boot["sar_exact_loo"]["point"]},
    None, [IC], "exact-rank radius with the scored cell removed from its own pool, "
                "then the same bootstraps", None,
    "*** DISAGREEMENT WITH THE PANEL, FLAGGED LOUDLY *** review_6 says 'beats-both "
    "against freeze survives (0.0289 < 0.0319)'. That is true of the POINT estimate, "
    "but once items 3 and 4 are applied TOGETHER the freeze-gap CI includes zero at "
    "every legitimate unit: seed-averaged 27 conditions [-0.0085,+0.0038], and even "
    "the i.i.d.-135 design [-0.0079,+0.0036]. Only the 3-cluster corruption-family "
    "bootstrap excludes zero. RECOMMENDATION: after the LOO fix, ImageNet-C SAR "
    "supports a POINT-ESTIMATE no-harm claim vs always-freeze, not a CI-supported "
    "beats-both. Do not write 'CI excluding zero' for the freeze gap if the LOO "
    "radius is adopted.", 3)

# ================================================== item 4: leave-one-out radius
for lbl, key, art in [
    ("cifar10c_tent_eata_stressgrid_3456cells", "stress_grid_tent_eata_seed1-4", SG),
    ("cifar10c_tent_headtohead_2160cells", "headtohead_tent_kga_seed0-4", H2H_T),
    ("cifar10c_eata_headtohead_2160cells", "headtohead_eata_kga_seed0-4", H2H_E),
    ("cifar10c_sar_stressgrid_1728cells", "stress_grid_sar_seed1-4", SG)]:
    c = cifloo[key]
    add(f"item4.{lbl}.loo_radius",
        f"CIFAR-10-C leave-one-out-of-pool radius on {c['n_cells_total']} cells: "
        "regret triple and number of decisions that change",
        {"n_cells": c["n_cells_total"],
         "decisions_changed_interp": c["interp_total_decisions_changed"],
         "decisions_changed_exact": c["exact_total_decisions_changed"],
         "regret_interp_in_pool": c["aggregate"]["interp_in_pool"]["regret_mean_over_files"],
         "regret_interp_loo": c["aggregate"]["interp_loo"]["regret_mean_over_files"],
         "regret_exact_in_pool": c["aggregate"]["exact_in_pool"]["regret_mean_over_files"],
         "regret_exact_loo": c["aggregate"]["exact_loo"]["regret_mean_over_files"],
         "fa_u_all_variants": 0.0,
         "adapt_interp_in_pool": c["aggregate"]["interp_in_pool"]["adapt"],
         "adapt_exact_in_pool": c["aggregate"]["exact_in_pool"]["adapt"]},
        None, [art],
        "for each cell i, eps_i = quantile of the residual pool with i removed "
        "(both the interpolated and the exact-rank quantile)",
        0,
        "CONFIRMS the panel: 0 of 3456 CIFAR decisions change. In fact 0 of 9504 change "
        "across every committed CIFAR-10-C file (3456 stress-grid tent+eata, 4320 "
        "head-to-head tent+eata, 1728 stress-grid SAR). FA_u stays 0 everywhere. "
        "The flagship CIFAR-10-C safety result is unaffected and should be DEFENDED.", 4)

add("item4.imagenetc_sar.loo_radius",
    "ImageNet-C SAR under the leave-one-out-of-pool exact-rank radius",
    {"regret": sar["pooled"]["exact_loo"]["regret"],
     "fa_u": sar["pooled"]["exact_loo"]["fa_u"],
     "fa_num": sar["pooled"]["exact_loo"]["fa_num"],
     "ADAPT": sar["pooled"]["exact_loo"]["adapt"],
     "FREEZE": sar["pooled"]["exact_loo"]["freeze"],
     "ABSTAIN": sar["pooled"]["exact_loo"]["abstain"],
     "cp95_upper_fa_c": sar["pooled"]["exact_loo"]["cp95_upper_fa_c"],
     "beats_freeze_point": sar["pooled"]["exact_loo"]["regret"][0]
                           < sar["pooled"]["exact_loo"]["regret"][2],
     "seeds_beating_both": [s for s in range(5)
                            if sar["per_seed"][str(s)]["exact_loo"]["beats_both"]]},
    {"regret": sar["pooled"]["exact_in_pool"]["regret"],
     "fa_u": sar["pooled"]["exact_in_pool"]["fa_u"],
     "ADAPT": sar["pooled"]["exact_in_pool"]["adapt"],
     "ABSTAIN": sar["pooled"]["exact_in_pool"]["abstain"]},
    [IC], "exact-rank radius with the scored index excluded",
    da["ImageNet-C SAR (promoted, pooled_5seed)"]["exact_decisions_changed_by_loo"],
    "CONFIRMS the panel: 0.026422 -> 0.028893, FA_u 0/135 -> 1/135, abstain 109 -> 107. "
    "2 of 135 decisions change. Point estimate still below always-freeze (0.028893 < "
    "0.031894) but see item3.imagenetc_sar.ci.after_item4_loo_radius: the CI does not.", 4)

cam = {}
for c in ("tent", "eata", "sar"):
    k = {"tent": "Camelyon17 Tent (Table VIII source, wilds_kbound, 4 seeds x 9)",
         "eata": "Camelyon17 EATA (Table VIII source)",
         "sar": "Camelyon17 SAR (Table VIII source)"}[c]
    t = da[k]
    cam[c] = {
        "regret_interp_in_pool_AS_SHIPPED": t["interp_in_pool"]["regret"],
        "regret_interp_loo": t["interp_loo"]["regret"],
        "regret_exact_in_pool": t["exact_in_pool"]["regret"],
        "regret_exact_loo": t["exact_loo"]["regret"],
        "fa_u_interp_in_pool": t["interp_in_pool"]["fa_u"],
        "fa_u_interp_loo": t["interp_loo"]["fa_u"],
        "fa_u_exact_loo": t["exact_loo"]["fa_u"],
        "ADAPT_interp_in_pool": t["interp_in_pool"]["adapt"],
        "ADAPT_interp_loo": t["interp_loo"]["adapt"],
        "eps_range_in_pool": t["interp_in_pool"]["eps_range"],
        "eps_range_loo": t["interp_loo"]["eps_range"],
        "decisions_changed_by_loo_interp": t["interp_decisions_changed_by_loo"],
        "per_seed": [{"seed": i, "eps": f["eps_min"], "regret": f["regret"],
                      "ADAPT": f["adapt"], "fa_u": f["fa_u"], "fa_c": f["fa_c"],
                      "harmful_frac": f["harmful_frac"],
                      "structural_fa_u_ceiling": f["structural_fa_u_ceiling"]}
                     for i, f in enumerate(t["interp_in_pool_per_file"])],
    }
add("item4.camelyon17_tableVIII.rescore",
    "Camelyon17 Table VIII (kbound_short.tex:889-891) re-scored: in-pool vs "
    "leave-one-out-of-pool radius, both quantile rules, n=9 per seed x 4 seeds",
    cam,
    {"tent": [0.020, 0.138, 0.020], "eata": [0.039, 0.042, 0.042],
     "sar": [0.041, 0.000, 0.065], "sar_FA": 0.11},
    ["experiments/kbound/results/wilds_kbound/per_condition_camelyon17_{tent,eata,sar}_seed{0..3}.json",
     "docs/research/kbound/scripts/run_wilds_camelyon17.py:56"],
    "run_wilds_camelyon17.py:56 uses np.quantile(|Bhat-B|, 0.9) over all 9 records "
    "including the scored one; the LOO column removes the scored record",
    None,
    "The published Table VIII reproduces EXACTLY from the interpolated in-pool column "
    "(Tent 0.020074/0.138021/0.020074; EATA 0.039280/0.041667/0.042426; SAR "
    "0.041016/0.000217/0.065430 with seed-0 FA_u = 1/9 = 0.1111, which is the '0.11' "
    "the table prints). Realized eps is 0.1527-0.3719 for Tent -- the panel's "
    "'0.153-0.372' -- which is why SAR 'over-freezes'. Under the LOO fix the SAR row "
    "gets WORSE: FA_u 0.0278 -> 0.0556 (2/36) and FA_c 0.143 -> 0.250. "
    "STRUCTURAL NOTE: at n=9, k = min(9, ceil(10*0.9)) = 9, so eps is the MAXIMUM "
    "residual and FA_u is forced to 0 under the exact-rank rule -- the exact-rank "
    "column of this table carries no information.", 4)

# ==================================== item 5: decision accounting + the identity
add("item5.structural_identity",
    "F1-1: with in-sample rank calibration the miscoverage count is identically N-k, "
    "so FA_u <= (N-k)/N holds for any data. Verified on every shipped per-condition file.",
    {"files_checked": ident["identity"]["summary"]["n_ok"],
     "files_hitting_the_ceiling_exactly": ident["identity"]["summary"]["n_hitting_exact_ceiling"],
     "ceilings_by_n": ident["identity"]["summary"]["ceilings_by_n"]},
    None, ["all per_condition_*.json under experiments/kbound/results/"],
    "eps = rho_(k); count rho_i > eps; compare with n-k. Also the interpolated rule's "
    "exceedance fraction and observed coverage, which are functions of n alone.",
    None,
    "CONFIRMS the chief reviewer: exact-rank ceiling 0.097222 at n=432 and 0.037037 at "
    "n=27, both BELOW alpha=0.10, so FA_u <= alpha cannot be violated on the stress "
    "grids. Interpolated exceedance 0.101852 at n=432 and 0.111111 at n=27, both ABOVE "
    "alpha. Observed interval coverage 0.898148 (n=432) and 0.888889 (n=27) are the "
    "numbers decision_metrics.json reports as measurements; they are functions of n "
    "alone. 69/69 files hit the ceiling exactly. At n=9 and n=12 the ceiling is 0 "
    "(k=n), i.e. eps = max residual.", 5)

add("item5.promoted_row_accounting",
    "Decision accounting for every promoted panel row: N, ADAPT/FREEZE/ABSTAIN, "
    "observed false-adapt count, Clopper-Pearson 95% upper bound on FA_c",
    ident["promoted_rows"], None,
    ["experiments/kbound/results/**", "docs/research/kbound/paper/generated/kbound_result_manifest.json"],
    "scipy.stats.beta.ppf(0.95, k+1, n-k) for the one-sided CP upper bound; counts "
    "recomputed cell-by-cell where the per-cell files exist, otherwise taken from the "
    "promoted summary artifact as n_test x adapt_rate",
    None,
    "Tracks with < 10 ADAPT decisions ('guarantee untested'): iWildCam (1), RxRx1 (0), "
    "D33 (9). Office-Home has 22 (CP95 upper 0.1273), ImageNet-C SAR 12 (CP95 upper "
    "0.2209). Only CIFAR-10-C has real power: 1113-1114 adapts, 0 wrong, CP95 upper "
    "0.00269. Camelyon17 OOD is BLOCKED-NEEDS-DATA.", 5)

full = {}
for k, t in da.items():
    if t["status"] != "OK":
        full[k] = {"status": t["status"], "missing": t.get("missing")}
        continue
    full[k] = {
        "n": t["exact_in_pool"]["n"],
        "exact_rank_in_pool": {q: t["exact_in_pool"][q] for q in
                               ("regret", "regret_mean_over_files", "adapt", "freeze",
                                "abstain", "fa_num", "fa_u", "fa_c", "cp95_upper_fa_c",
                                "cp95_upper_fa_u", "harmful_frac", "eps_range",
                                "guarantee_untested_lt10_adapts", "beats_both_point")},
        "interp_in_pool": {q: t["interp_in_pool"][q] for q in
                           ("regret", "regret_mean_over_files", "adapt", "freeze",
                            "abstain", "fa_num", "fa_u", "fa_c", "cp95_upper_fa_c",
                            "eps_range", "beats_both_point")},
        "exact_rank_loo": {q: t["exact_loo"][q] for q in
                           ("regret", "regret_mean_over_files", "adapt", "freeze",
                            "abstain", "fa_num", "fa_u", "fa_c", "cp95_upper_fa_c")},
        "interp_loo": {q: t["interp_loo"][q] for q in
                       ("regret", "regret_mean_over_files", "adapt", "freeze",
                        "abstain", "fa_num", "fa_u", "fa_c", "cp95_upper_fa_c")},
        "decisions_changed_by_loo": {"exact": t["exact_decisions_changed_by_loo"],
                                     "interp": t["interp_decisions_changed_by_loo"]},
        "artifacts": t["artifacts"],
    }
add("item5.full_track_table",
    "Full per-track decision accounting for every track with per-cell artifacts on disk "
    "(30 tracks), under all four radius variants",
    full, None, ["experiments/kbound/results/**"],
    "see 06_decision_accounting.py", None, "", 5)

# ============================================ item 6: CIFAR-10-C SAR quarantine
csar = pv["cifar10c_stress_grid"]["per_candidate"]["sar"]
add("item6.cifar10c_sar.quarantine",
    "CIFAR-10-C SAR quarantined arm: per-seed harmful base rate and the "
    "seeds-1-4-only regret triple",
    {"harmful_base_rate_per_seed": csar["per_seed_harmful_frac"]["values"],
     "seed0_harmful": csar["seed0_only"]["harmful_frac"],
     "seeds1to4_harmful_mean": csar["seeds_1to4_only"]["harmful_frac_mean"],
     "seed0_over_seeds1to4_ratio": csar["seed0_harmful_ratio_vs_seeds1to4_mean"],
     "seeds1to4_regret_kga_adapt_freeze": csar["seeds_1to4_only"]["regret"],
     "seed0_regret_kga_adapt_freeze": csar["seed0_only"]["regret"],
     "five_seed_pooled_regret": csar["pooled_regret"],
     "n_seeds_beating_both": csar["n_seeds_beating_both"],
     "per_seed_beats_both": csar["per_seed_beats_both"],
     "eps_per_seed": csar["eps_per_seed"], "eps_cv": csar["eps_cv"],
     "kga_worse_than_always_adapt_on_seeds_1to4": csar["seeds_1to4_only"]["kga_worse_than_adapt"]},
    {"paper_sentence": "The rebuild yields regret 0.0015/0.0112/0.1286 ... and "
                       "false-adapt 0/2160; paired condition-bootstrap intervals "
                       "exclude zero against both fixed policies. (kbound_short.tex:639-642)"},
    ["experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json",
     "experiments/kbound/results/stress_grid_multiseed_v1/seed{1..4}/per_condition_cifar10c_sar_seed{s}.json"],
    "per-seed rows read from pstar_law.per_seed_cand; the seeds-1-4 triple is the "
    "unweighted mean of those four seeds, and it reproduces EXACTLY from the four "
    "committed per-condition files (0.00159896/0.00030990/0.14049653)",
    None,
    "CONFIRMS review_6's replacement sentence to 5 decimals: seeds 1-4 KGA regret "
    "0.0015990 vs always-adapt 0.0003099. Seed 0's harmful base rate 0.5278 is 5.81x "
    "the mean of seeds 1-4 (0.0909). 1 of 5 seeds beats both, and it is seed 0. "
    "CAVEAT: seed 0's per-condition dump is NOT on disk (stress_grid_multiseed_v1/"
    "seed0/ has only decisive_tta_results.json), so the seed-0 row is readable only "
    "from the stored LOCKED_ANALYSIS_RESULTS.json and cannot be independently "
    "recomputed. The claim 'rebuilt from all five saved per-condition seed files' is "
    "false for the 432-cell grid: only four exist.", 6)

# ======================================= item 17: cluster-robust + LOCO on CIFAR
for lbl, key in [("tent", "cifar10c_tent_headtohead_seed0-4"),
                 ("eata", "cifar10c_eata_headtohead_seed0-4"),
                 ("sar", "cifar10c_sar_stressgrid_seed1-4")]:
    c = clus[key]
    add(f"item17.cifar10c_{lbl}.cluster_robust",
        f"CIFAR-10-C {lbl.upper()}: paired-bootstrap gap CIs at four resampling units, "
        "plus the r0/r1 replicate correlation and the per-corruption breakdown",
        {"point": c["point"],
         "replicate_correlation_r0_r1": c["replicate_correlation_r0_r1"],
         "designs": {k: {"n_units": v["n_units"],
                         "adapt_gap_ci95": [v["adapt_gap"]["lo"], v["adapt_gap"]["hi"]],
                         "freeze_gap_ci95": [v["freeze_gap"]["lo"], v["freeze_gap"]["hi"]],
                         "beats_both_ci": v["beats_both_ci"],
                         "width_ratio_vs_iid_adapt": v["width_ratio_adapt"],
                         "width_ratio_vs_iid_freeze": v["width_ratio_freeze"]}
                     for k, v in c["designs"].items()},
         "per_corruption": c["per_corruption"]},
        None, [H2H_T if lbl == "tent" else (H2H_E if lbl == "eata" else SG)],
        "seed-averaged 432-vector; paired percentile bootstrap, 20000 replicates, "
        "rng seed 20260611 (the stream seed _locked_analysis_script.py uses); "
        "cluster variants resample whole clusters with replacement",
        None,
        ("CONFIRMS the panel for TENT: widths grow 2.32x (adapt) and 3.90x (freeze) at "
         "the 6-corruption unit, all CIs still exclude zero, and gaussian_noise reverses "
         "sign (+0.00189, KGA worse than always-adapt)."
         if lbl == "tent" else
         ("*** DISAGREEMENT WITH THE PANEL *** review_6 item 17 says clustering by "
          "corruption family leaves all CIs excluding zero. That is FALSE for EATA: the "
          "adapt-gap CI is [-0.00436,+0.00035] at 6 corruption clusters and "
          "[-0.00483,+0.00043] at 12 corruption x severity clusters -- both include "
          "zero. EATA also has TWO families where KGA is worse than always-adapt "
          "(gaussian_noise +0.00022, jpeg_compression +0.00292). Write the "
          "cluster-robust row for Tent only, or report EATA's honestly."
          if lbl == "eata" else
          "SAR (quarantined) is worse than always-adapt at EVERY unit: the adapt-gap CI "
          "is entirely POSITIVE."))
        , 17)

loco_rows = {}
for sch in ("leave_one_cell_out_AS_SHIPPED", "leave_one_twin_pair_out",
            "leave_one_corruption_out"):
    for rule in ("interp", "exact"):
        rs = [loco[k][sch] for k in sorted(loco)]
        loco_rows[f"{sch}[{rule}]"] = {
            "n_folds": rs[0]["n_folds"],
            "residual_MAE_mean": float(np.mean([r["residual_MAE"] for r in rs])),
            "R2_mean": float(np.mean([r["R2"] for r in rs])),
            "eps_mean": float(np.mean([r[rule]["eps"] for r in rs])),
            "eps_per_seed": [r[rule]["eps"] for r in rs],
            "adapt_rate_mean": float(np.mean([r[rule]["adapt_rate"] for r in rs])),
            "fa_u_mean": float(np.mean([r[rule]["fa_u"] for r in rs])),
            "regret_kga_mean": float(np.mean([r[rule]["regret"][0] for r in rs])),
            "regret_kga_per_seed": [r[rule]["regret"][0] for r in rs],
            "regret_adapt_mean": float(np.mean([r[rule]["regret"][1] for r in rs])),
            "regret_freeze_mean": float(np.mean([r[rule]["regret"][2] for r in rs])),
        }
add("item17.cifar10c_tent.leave_one_corruption_out",
    "CIFAR-10-C Tent: leave-one-CORRUPTION-out calibration vs leave-one-cell-out "
    "(as shipped) vs leave-one-twin-pair-out, all 5 seeds refitted",
    loco_rows, None,
    [H2H_T, "docs/research/kbound/scripts/cifar_tent_mps_v2.py:151-162"],
    "GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05, "
    "subsample=0.8, random_state=0), out-of-fold predictions per partition; "
    "sklearn 1.8.0, numpy from the container",
    None,
    "CONFIRMS the panel's seed-0 numbers exactly (MAE 0.01021 -> 0.03222, R2 0.991 -> "
    "0.892, eps 0.02144 -> 0.09219, adapt 52.3% -> 41.7%, FA_u stays 0, KGA regret "
    "0.001306 -> 0.005495). 5-seed means: MAE 0.00959 -> 0.03090 (3.22x), eps 0.02106 "
    "-> 0.09717 (4.61x), regret 0.001585 -> 0.005867 (3.70x). FA_u = 0 under every "
    "partition, and beats-both survives (0.005867 < 0.007923 adapt and < 0.124098 "
    "freeze). This ablation STRENGTHENS the paper. The refit reproduces the stored "
    "b_hat at corr 0.999996-1.000000.", 17)

# ==================================== item 18: baseline parity + radius value
gseeds = sorted(gates)
rules = list(gates["seed0"]["rules"].keys())
gm = {}
for nm in rules:
    r = [gates[s]["rules"][nm]["all"] for s in gseeds]
    h = [gates[s]["rules"][nm]["harmful_subset"] for s in gseeds]
    gm[nm] = {
        "calibration": gates["seed0"]["rules"][nm]["calibration"],
        "regret_mean": float(np.mean([x["regret"] for x in r])),
        "FA_u_mean": float(np.mean([x["FA_u"] for x in r])),
        "FA_c_mean": float(np.mean([(x["FA_c"] or 0.0) for x in r])),
        "adapt_rate_mean": float(np.mean([x["adapt_rate"] for x in r])),
        "coverage_mean": float(np.mean([x["coverage"] for x in r])),
        "FA_u_harmful_subset_mean": float(np.mean([x["adapt_rate"] for x in h])),
        "total_adapts_5seeds": int(sum(x["n_adapt"] for x in r)),
        "total_false_adapts_5seeds": int(sum(x["FA_num"] for x in r)),
        "per_seed_regret": [x["regret"] for x in r],
    }
add("item18.tab_gates.regenerated",
    "tab:gates regenerated on real data (CIFAR-10-C Tent, 5 seeds x 432 cells), with "
    "the calibration budget of every rule stated",
    gm,
    {"published_table_kbound_short.tex:707-716":
        {"confidence gate": [0.0084, 0.257, 0.301, 0.85, 1.00, 0.745],
         "entropy gate": [0.0086, 0.255, 0.304, 0.84, 1.00, 0.738],
         "drift/KL gate": [0.1232, 0.000, 0.000, 0.00, 1.00, 0.000],
         "ATC-style gate": [0.0045, 0.116, 0.172, 0.67, 1.00, 0.336],
         "KGA (no radius)": [0.0004, 0.049, 0.071, 0.68, 1.00, 0.141],
         "KGA (certificate)": [0.0017, 0.000, 0.000, 0.51, 0.68, 0.000],
         "n": 432, "n_harmful": 149}},
    [H2H_T, "docs/research/kbound/scripts/gate_baseline_comparison.py"],
    "gate definitions copied verbatim from gate_baseline_comparison.py:45-104; "
    "FA_u uses the strict B<0 convention that script uses",
    None,
    "The published table cannot be regenerated as released (its input "
    "cifar10c_percell.json is absent, F4-7). Regenerated here from the committed "
    "head-to-head per-condition dumps. Shape and magnitudes reproduce; n_harmful is "
    "137-146 per seed (711 over 5 seeds), not the 149 the caption states. "
    "TWO NEW ROWS the paper should carry: (i) 'KGA (certificate, "
    "leave-one-CORRUPTION-out)' -- KGA at the SAME calibration budget the drift and "
    "ATC gates get -- regret 0.0059, FA_u 0.000, coverage 0.42; at that budget the "
    "ATC gate has LOWER regret (0.0041) but breaks the budget (FA_u 0.116 > alpha). "
    "(ii) 'KGA (no radius, leave-one-corruption-out)' regret 0.0005, FA_u 0.046.", 18)

no_r = gm["KGA (no radius)"]
cert = gm["KGA (certificate, interp eps)"]
add("item18.radius_value",
    "What the conformal radius buys, measured (F3-11)",
    {"no_radius_FA_u": no_r["FA_u_mean"], "no_radius_regret": no_r["regret_mean"],
     "no_radius_coverage": no_r["coverage_mean"],
     "no_radius_harmful_cell_adapt_rate": no_r["FA_u_harmful_subset_mean"],
     "certificate_FA_u": cert["FA_u_mean"], "certificate_regret": cert["regret_mean"],
     "certificate_coverage": cert["coverage_mean"],
     "certificate_harmful_cell_adapt_rate": cert["FA_u_harmful_subset_mean"],
     "regret_ratio_certificate_over_no_radius":
        cert["regret_mean"] / no_r["regret_mean"],
     "no_radius_meets_declared_budget": bool(no_r["FA_u_mean"] < 0.10),
     "no_radius_total_false_adapts_5seeds": no_r["total_false_adapts_5seeds"],
     "certificate_total_false_adapts_5seeds": cert["total_false_adapts_5seeds"]},
    {"paper": {"no_radius": [0.0004, 0.049, 0.071, 0.68, 1.00, 0.141],
               "certificate": [0.0017, 0.000, 0.000, 0.51, 0.68, 0.000],
               "claimed_ratio": 4.25}},
    [H2H_T], "5-seed means of the regenerated tab:gates", None,
    "The radius-free variant meets the declared budget (FA_u 0.038 < alpha = 0.10) at "
    "4.0x lower regret and full decision coverage. The argument for the radius is the "
    "harmful-cell column: 11.7% of harmful cells adapted without it (83 false adapts "
    "over 5 seeds) vs 0.0% with it (0 false adapts). Make THAT the argument, not the "
    "aggregate FA_u.", 18)

# ==================================== item 19: environment heterogeneity
sg_env = [m for m in env["manifests"]
          if m["status"] == "OK" and "stress_grid_multiseed_v1" in m["path"]]
ic_env = [m for m in env["manifests"]
          if m["status"] == "OK" and "win_hunt_v5_imagenetc_ms" in m["path"]]
add("item19.env_heterogeneity",
    "Seed-0 environment heterogeneity on the CIFAR-10-C stress grid and on ImageNet-C",
    {"cifar10c_stress_grid": [
        {"seed": m.get("seed"), "git_hash": m.get("git_hash"), "python": m.get("python"),
         "torch": m.get("torch"), "numpy": m.get("numpy"), "finished": m.get("finished")}
        for m in sorted(sg_env, key=lambda x: x.get("seed", 0))],
     "n_distinct_stacks_cifar": 3,
     "imagenetc": [
        {"seed": m.get("seed"), "git_hash": m.get("git_hash"), "python": m.get("python"),
         "torch": m.get("torch"), "numpy": m.get("numpy"), "finished": m.get("finished")}
        for m in sorted(ic_env, key=lambda x: x.get("seed", 0))],
     "imagenetc_seed0_manifest": {
        "path": "experiments/kbound/results/win_hunt_v5/imagenetc_aggr/result_manifest.json",
        "git_hash": "87bf90aaadce8d170a89ef19b9a2459c3ac6c9f6", "python": "3.12.13",
        "torch": "2.5.1", "numpy": "2.4.6", "finished": "2026-07-09 12:06:41",
        "argv_omits": ["--severities 1 3 5", "--max-images 4000"]},
     "pooled_5seed_has_result_manifest": False,
     "pooled_5seed_seed0_is_md5_identical_copy_of":
        "experiments/kbound/results/win_hunt_v5/imagenetc_aggr/per_condition_imagenetc_sar_seed0.json",
     "md5": "8b655a29360a23ca6fa9f5658f91d95a",
     "n_manifests_scanned": env["n_manifests"],
     "n_manifests_pinning_sklearn": len(env["manifests_pinning_sklearn"])},
    None,
    ["experiments/kbound/results/stress_grid_multiseed_v1/seed{0..4}/result_manifest.json",
     "experiments/kbound/results/win_hunt_v5_imagenetc_ms/seed{1..4}/result_manifest.json",
     "experiments/kbound/results/win_hunt_v5/imagenetc_aggr/result_manifest.json"],
    "read every result_manifest*.json under experiments/kbound/", None,
    "CONFIRMS F4-6 and F4-14 exactly. CIFAR-10-C: seed 0 ran 2026-07-02 on commit "
    "4896181799ad under Python 3.12.13 / torch 2.5.1 / numpy 2.4.6; seeds 1-3 ran "
    "2026-06-11/12 on commit 6a237ed489c3 under Python 3.14.3 / torch 2.12.0 / numpy "
    "2.4.4; seed 4 on a third commit 571c89f25989 with the seeds-1-3 stack. "
    "ImageNet-C: seeds 1-2 on commit 27a7e977f033, seeds 3-4 on 1adea4515b8c, both "
    "Python 3.9.23 / torch 2.8.0 / numpy 2.0.2; seed 0 is an md5-identical copy of a "
    "2026-07-09 run under Python 3.12.13 / torch 2.5.1 whose argv omits "
    "'--severities 1 3 5' and '--max-images 4000'. pooled_5seed/ carries no "
    "result_manifest.json. 0 of 43 manifests record a scikit-learn version.", 19)

# ==================================== item 23: panel-row variance
add("item23.pacs", "PACS per-domain per-seed spread behind the panel mean",
    pv["pacs"], {"panel_row": [0.0431, 0.0176, 0.0446], "panel_FA_u": 0.0092592593},
    ["experiments/kbound/results/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json",
     "experiments/kbound/results/per_cell/pacs_*_seed{1,2}_percell.json"],
    "per-seed rates read from PACS_MULTISEED_RESULTS.json; integer counts back-derived "
    "as rate x 18 (exact, since every rate is a multiple of 1/18); Wilson and "
    "Clopper-Pearson on the pooled 2/216",
    None,
    "CONFIRMS F3-15: art_painting seed 1 has FA_u = 0.1111 > alpha (2 of 18 cells), and "
    "art_painting seed 2 abstains on all 18 (coverage 0.0). Pooled FA_u = 2/216 = "
    "0.00926, Wilson95 [0.00254, 0.03313], Clopper-Pearson95 [0.00112, 0.03305]. "
    "KGA regret across the 12 domain-seed cells: min 0.00529, median 0.03616, max "
    "0.15344, sd 0.03895 -- the panel prints only the mean 0.0431. NOTE the per-cell "
    "dumps in results/per_cell/ exist for seeds 1 and 2 only and carry no b_hat, so "
    "PACS decisions cannot be re-scored under a LOO radius from the release.", 23)

add("item23.imagenet_r", "ImageNet-R per-backbone spread behind the panel mean",
    pv["imagenet_r"], {"panel_row": [0.0112, 0.0064, 0.0325],
                       "panel_note": "0/10 CI beats-both, observed false-adapt 1/480"},
    ["experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json",
     "experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/per_condition_imagenet-r_*_seed{0..3}.json"],
    "per-backbone rows read from MULTISEED_ANALYSIS_RESULTS.json and independently "
    "re-scored cell-by-cell in 06_decision_accounting.py", None,
    "STRONGER THAN THE PANEL SAID. KGA is worse than always-adapt on 7 of 10 backbones, "
    "not one: convnext_tiny 14.2x, resnet101 29.1x, vit_b_16 43.9x, swin_b 434x, plus "
    "efficientnet_b3 / resnet152 / resnext101_32x8d where always-adapt has regret "
    "exactly 0 (0% harmful base rate) and KGA has 0.0186 / 0.0088 / 0.0059. FOUR "
    "backbones have a degenerate 0% harmful base rate. Across backbones KGA regret is "
    "min 0.00000, median 0.01195, max 0.02260. Report min/median/max and the per-"
    "backbone harmful base rate, as item 23 asks.", 23)

add("item23.cifar10c_stress_grid", "CIFAR-10-C per-seed spread behind the panel mean",
    pv["cifar10c_stress_grid"], None,
    ["experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json"],
    "per-seed rows from pstar_law.per_seed_cand", None,
    "Tent 5/5 seeds beat both; EATA 5/5; SAR 1/5 (and that one is seed 0). "
    "eps CV: Tent 0.0302, EATA 0.0684, SAR 0.3897.", 23)

add("item23.camelyon17_tableVIII_perseed",
    "Camelyon17 Table VIII per-seed spread (the +- in the table is over 4 seeds of 9 cells)",
    {c: cam[c]["per_seed"] for c in cam}, None,
    ["experiments/kbound/results/wilds_kbound/per_condition_camelyon17_*_seed{0..3}.json"],
    "recomputed cell-by-cell", None,
    "Camelyon17's panel row reports FA_u = 0 while tab:multiseed's SAR row reports "
    "0.11: the 0.11 is seed 0's 1/9. At n = 9 a single false adapt is 0.111, so 1/9 is "
    "well inside binomial noise -- Clopper-Pearson 95% upper bound on FA_u for 1/9 is "
    "0.394. Qualify the row.", 23)

# ============================================================== cross-cutting
add("xx.imagenetc_grid_composition",
    "What the ImageNet-C grid actually is (needed for fix-queue item 7's disclosure)",
    {"n_conditions": 27, "corruptions": ["gaussian_noise", "shot_noise", "impulse_noise"],
     "severities": ["s1", "s3", "s5"], "batch_regimes": ["small"],
     "aggressiveness": ["aggressive"],
     "compositions": ["iid", "imbalanced", "single_class"],
     "adapt_lr": 0.004, "arch": "resnet50", "max_images": 4000},
    None, [IC, "experiments/kbound/results/win_hunt_v5_imagenetc_ms/seed1/result_manifest.json"],
    "parsed from the condition strings and the run manifest argv", None,
    "All three ImageNet-C corruptions are from the NOISE family. There is exactly one "
    "batch regime (small) and one aggressiveness (aggressive), at lr 4e-3. "
    "tab:adapter-hparams' 'mild & aggressive (per cell)' is wrong for this track.", 7)

add("xx.cifar10c_grid_composition",
    "What the CIFAR-10-C stress grid actually is (fix-queue item 7)",
    {"n_conditions": 432,
     "corruptions": ["contrast", "defocus_blur", "fog", "gaussian_noise",
                     "jpeg_compression", "pixelate"],
     "n_corruptions_of_15": 6, "severities": ["s1", "s5"],
     "batch_regimes": ["large_iid", "small", "tiny"],
     "compositions": ["iid", "imbalanced", "single_class"],
     "aggressiveness": ["mild", "aggressive"], "repeats": ["r0", "r1"],
     "quick_flag_in_every_run_manifest": True},
    None, [SG, "docs/research/kbound/scripts/cifar_tent_mps_v2.py:138"],
    "parsed from the condition strings; CIFAR_C_QUICK at cifar_tent_mps_v2.py:138",
    None,
    "6 x 2 x 3 x 3 x 2 x 2 = 432. r0/r1 are exact design-point replicates, so the "
    "effective n is at most 216 (measured replicate correlation 0.948 on the adapt "
    "gap and 0.999 on the freeze gap). kbound.tex:807's 'the official CIFAR-10-C "
    "corruptions' is false: 6 of 15, and 2 of 5 severities.", 7)

ph = L("out_placeholders.json")
add("xx.placeholder_artifacts",
    "iCloud placeholder census over the whole tree (fix-queue item 9)",
    {"n_text_files_scanned": ph["n_text_files_scanned"],
     "n_placeholders_nul_or_zero_byte": ph["n_placeholders"],
     "n_whitespace_only_naive_test": ph["n_whitespace_only"],
     "n_unreadable_oserror": ph["n_unreadable"],
     "by_extension": ph["by_extension"],
     "named_placeholders": [r["path"] for r in ph["placeholders"]
                            if any(k in r["path"] for k in
                                   ("ablation_", "cost_profile", "officehome/",
                                    "checklists/", "checkpoint.json"))][:40]},
    {"panel_reported": 142},
    ["/home/claude/kb_fixes/recompute/out_placeholders.json"],
    "python3 13_placeholder_scan.py -- a file is a placeholder iff it is zero bytes or "
    "contains a NUL byte; the naive whitespace test returns 0",
    None,
    "145 placeholders (78 .json, 45 .py, 10 .csv, 9 .md, 3 .sh) vs 0 by the naive "
    "whitespace test. None of them blocked a number in this pack.", 9)

add("xx.blocked_numbers",
    "Numbers this pack could NOT recompute, and why",
    [{"number": "Camelyon17 OOD promoted row 0.0000 / 0.0000 / 0.1381 (n=18)",
      "missing": "docs/research/kbound/audits/integrity_2026-06-20/camelyon_reconciliation/",
      "status": "BLOCKED-NEEDS-DATA (directory does not exist; the triple appears in no "
                "artifact on disk)"},
     {"number": "Office-Home promoted regret 0.0157142857 (n=35)",
      "missing": "experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json, "
                 "experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json",
      "status": "BLOCKED-NEEDS-DATA for the promoted value. The recomputable "
                "multiseed/officehome/extracted files (36 cells x 5 seeds) give KGA "
                "regret 0.000000 with 114/180 adapts, and officehome_protocol_M_v2/"
                "protocol_result.json gives 0.002198 -- neither is 0.0157."},
     {"number": "iWildCam promoted regret 0.0041023691 (n=72)",
      "missing": "experiments/kbound/results/iwildcam_full_test/result_e40faf29.json",
      "status": "BLOCKED-NEEDS-DATA for the promoted value. The recomputable "
                "multiseed/iwildcam/extracted files (72 cells x 2 seeds) give KGA "
                "regret 0.021174 == always-freeze regret with 0 adapts."},
     {"number": "CIFAR-10-C SAR seed 0 (harmful 0.5278, regret 0.001351/0.054705/0.081192)",
      "missing": "experiments/kbound/results/stress_grid_multiseed_v1/seed0/"
                 "per_condition_cifar10c_sar_seed0.json (and tent/eata seed 0)",
      "status": "PARTIAL: readable only from the stored LOCKED_ANALYSIS_RESULTS.json; "
                "not independently recomputable. Seeds 1-4 recompute exactly."},
     {"number": "tab:gates as published (n=432, 149 harmful)",
      "missing": "cifar10c_percell.json (does not exist anywhere in the tree)",
      "status": "PARTIAL: regenerated from the head-to-head per-condition dumps "
                "(137-146 harmful per seed); magnitudes match, n_harmful does not."},
     {"number": "PACS decisions under a leave-one-out radius",
      "missing": "b_hat / eps_conformal in experiments/kbound/results/per_cell/"
                 "pacs_*_percell.json, and seed 0 per-cell dumps entirely",
      "status": "BLOCKED-NEEDS-DATA: the PACS per-cell files carry Z, a0, aa, B but no "
                "b_hat and no decision, so no radius can be re-derived."}],
    None, [], "", None, "", None)

json.dump({"generated": "recompute agent, K-Bound fix queue",
           "alpha": 0.10,
           "conventions": {
               "interpolated_radius": "np.quantile(|bhat-B|, 1-alpha) over the pool",
               "exact_rank_radius": "rho_(k), k = min(n, ceil((n+1)(1-alpha)))",
               "in_pool": "one radius per file, computed from ALL residuals including "
                          "the scored cell's own (as every shipped runner does)",
               "loo": "leave-one-out-of-pool: cell i's radius excludes cell i's residual",
               "decision": "ADAPT iff bhat-eps>0; FREEZE iff bhat+eps<0; else ABSTAIN",
               "regret": "oracle = max(a0, a_adapted); KGA takes a_adapted iff ADAPT; "
                         "regret = oracle - policy accuracy",
               "FA_u": "mean(is_adapt & (B <= 0))  [the quantity thm:certificate bounds]",
               "FA_c": "mean(B <= 0 | is_adapt)",
               "bootstrap": "paired percentile, 20000 replicates, fixed rng seed",
           },
           "entries": E}, open(OUT, "w"), indent=1, default=float)
print(f"wrote {OUT} with {len(E)} entries")
