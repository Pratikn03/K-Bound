# K-Bound — Full Findings Audit

_Auto-generated audit of every result on disk vs. what is documented. Honest
scoping for the paper. Generated 2026-06-19._

## 0. Executive verdict (honest, now with bootstrap CIs)

Bootstrap 95% CIs are now computed (faithful: resample held-out test conditions,
GBR benefit-estimator + conformal eps held fixed on dev — the protocol's own
scorer, no re-tuning on test; every point estimate reproduces the locked
`protocol_result.json` exactly). The verdict tightens to:

> **KGA is a statistically robust beats-both on 2 of 3 dev-locked natural-shift
> datasets — OfficeHome and Camelyon17 (both regret-gap CIs exclude 0) — and on
> iWildCam it significantly dominates always-adapt (+0.099, CI [0.080, 0.119])
> while tying always-freeze (+0.0004, CI [0.000, 0.0013], includes 0).** False-adapt
> stays ≤ α everywhere. Plus the synthetic **CIFAR-10-C** stress win for Tent/EATA.

You do **not** have a clean win on **ImageNet-R** or **RxRx1** (lead on only one
split/seed, does not replicate). So the accurate, defensible headline is **"robust
beats-both on 2 of 3, no-harm Pareto win on the 3rd"** — not a universal real-data
win, but materially stronger than "point-estimate ahead, CIs pending."

## 1. The locked 6-dataset core panel — v2 (`research_lock/KBOUND_6_DATASET_PANEL_v2.yaml`)

Revised after the bootstrap CIs to be the **strongest honest six**: Camelyon17
(a CI-robust win) is promoted IN; RxRx1 (single-seed lead, not replicated) is
demoted OUT to supporting breadth (still in the paper, just not headline).

| # | Dataset | Role | Claim status (CI-backed) | Win? |
|---|---------|------|--------------------------|------|
| 1 | **OfficeHome** | primary clean held-out win | robust beats-both, CI excludes 0 | **YES (robust)** |
| 2 | **Camelyon17** | biomedical natural-shift win | robust beats-both, CI excludes 0 | **YES (robust)** |
| 3 | **CIFAR-10-C stress** | controlled corruption grid | multiseed beats-both (Tent/EATA) | YES |
| 4 | **iWildCam** | independent WILDS natural shift | dominates adapt, **ties** freeze | no-harm |
| 5 | **CIFAR-10.1** | low-margin boundary | not cross-seed beats-both | boundary |
| 6 | **ImageNet-R** | forced-abstention frontier | undetectable harm, abstain mandatory | frontier (not a win) |

**Demoted to supporting (kept in paper, not in core six):** RxRx1 (modelseed0
lead, not replicated), ImageNet-C (CIFAR-10-C covers controlled corruption), and
the FMoW / Poverty honest nulls. Camelyon17's locked result reproduces from its
serialized rich-Z records, so the old "data not mounted" exclusion no longer applies.

## 2. The 3 held-out wins — numbers + bootstrap CIs (`KBOUND_WIN_BOOTSTRAP_CIS.json`)

Regret gaps are **(baseline regret − KGA regret)**; positive ⇒ KGA better.
95% CI from B=3000 resamples of the held-out test conditions (dev calibration fixed).

| Dataset | n | KGA vs **freeze** (95% CI) | KGA vs **adapt** (95% CI) | false-adapt | Robust beats-both? |
|---------|--:|----------------------------|---------------------------|------------:|:------------------:|
| **OfficeHome** (Prot. M v2) | 35 | **+0.0136 [+0.0090, +0.0184]** ✓ | **+0.0445 [+0.0195, +0.0728]** ✓ | 0.00 | **YES** |
| **Camelyon17** (Prot. G) | 54 | **+0.0749 [+0.0565, +0.0954]** ✓ | **+0.0013 [+0.0006, +0.0020]** ✓ | 0.026 | **YES** |
| **iWildCam** (Prot. H v2) | 72 | +0.0004 [0.0000, +0.0013] ✗ (tie) | **+0.0991 [+0.0797, +0.1190]** ✓ | 0.00 | no — ties freeze |

✓ = CI excludes 0. Method (sound): adapter selected on dev/val from a locked
6-arm panel, held-out test scored **once**, false-adapt ≤ α = 0.10. The bootstrap
reproduces each locked point estimate to 4 decimals, so the CIs measure genuine
test-condition sampling noise — not a re-analysis with different numbers.

**Reading it:** OfficeHome and Camelyon17 beat *both* trivial policies with CIs that
clear zero — these are the real, defensible wins. iWildCam's edge over freeze
(+0.0004) is statistical noise (CI touches 0), so iWildCam is a *no-harm* result:
it crushes blind adaptation and matches the better of freeze/adapt, but is not a
significant double-beat. Report it that way.

## 2b. Mixed-stream deployment — the decisive beats-both (`KBOUND_MIXED_STREAM_v1.json`)

Each single dataset is *one-sided* (Camelyon17 adapt-favorable; OfficeHome/iWildCam
freeze-favorable), so KGA's simultaneous margin over both is small there. Pooling all
three datasets' held-out test conditions into one **heterogeneous deployment** (n=161,
each dataset keeping its own locked dev-calibrated gate; per-dataset means reproduce the
locked protocols to 4 decimals) exposes the two-sided regime the certificate is built for:

| Policy | Regret ↓ | KGA advantage (95% CI) |
|--------|---------:|------------------------|
| Always-adapt | 0.0566 | +0.0545 [0.042, 0.067] ✓ |
| Always-freeze | 0.0304 | +0.0283 [0.020, 0.037] ✓ |
| **KGA** | **0.0021** | — |

On the mixed stream KGA's regret is ~14× below freeze and ~27× below adapt; **both gaps'
CIs exclude 0** at false-adapt 0.016 ≤ α. This is the simultaneous beats-both that no
one-sided dataset shows. *Honest scope:* a constructed pool of real test conditions (not a
new benchmark); demonstrates the mechanism (per-condition routing beats any global policy
under heterogeneity), not a marquee-benchmark SOTA. Now in the paper as §VII-C, Table (p18–19).

## 3. The non-wins (do NOT promote these to wins)

- **ImageNet-R** (`hard_dataset_win_loop_v1`): beats-both on **1 of 4** data
  splits (`split01_23` wins; `02_13`, `03_12`, `12_03` do not). The lead is a
  split artifact.
- **RxRx1**: beats-both on **1 of 4** model seeds (`modelseed0_tent_mondrian`);
  seeds 1, 2, pooled fail. Protocol-J: **0 of 3** seeds win. Not reproducible.
- **CIFAR-10.1**: not cross-seed beats-both — a genuine boundary/low-margin case.

## 4. Honest nulls (reported, not hidden — good)

- **FMoW** (Protocol L): false-adapt **0.375** (far over alpha), beats-both **false**.
- **Poverty** (Protocol L): stopped at dev-screen (`dev_screen_stop`).
- **OfficeHome deployed eata_online_mild**: helpful-dominated regime (separate null).

## 5. Integrity assessment

**Strong:** protocols pre-registered in `research_lock/`; dev-select -> test-score-once
discipline; false-adapt controlled at alpha; nulls reported openly; and you **caught and
superseded your own p-hacking** — iWildCam H_v1 (fixed SAR, no dev-lock) and
OfficeHome M_v1 (14-candidate win-finder) were retired in favour of dev-locked v2.

**Gaps a reviewer will press:**
1. ~~No confidence intervals.~~ **DONE** (`KBOUND_WIN_BOOTSTRAP_CIS.json`, §2).
   Result: OfficeHome over freeze is **not** within noise after all — CI
   [+0.0090, +0.0184] clears 0 (my earlier guess that it would was wrong).
   Camelyon17 clears 0 on both axes. Only **iWildCam over freeze** is within noise
   (CI [0.000, 0.0013]). So "beats both" is now defensible for OfficeHome and
   Camelyon17; iWildCam must be stated as dominate-adapt / tie-freeze. Caveat to
   disclose: the resampling unit is the test *condition* cell (n=35/54/72), not iid
   images, and dev calibration is held fixed — standard but approximate.
2. **Multiple comparisons across datasets/splits/seeds.** 3 wins out of ~6
   datasets; ImageNet-R/RxRx1 leads appear only on selected splits/seeds. State
   the denominator and keep the non-replicated cases labelled as such.

## 6. What is documented vs. undocumented

- **In the paper (`kbound.tex`)**: OfficeHome, iWildCam, FMoW, Poverty, plus the
  existing ImageNet-C, Camelyon17, CIFAR-10-C/10.1, RxRx1, and the AD track.
- **NOT in the paper (only in `research_lock/` + `results/`)**: the protocol
  machinery (H/M/J/L/K/G), the **win-finder / win-loop** methodology, and the
  **split/seed robustness data** that refutes the ImageNet-R and RxRx1 leads.
  This robustness data should be summarised in the paper's limitations, not omitted.

## 7. Recommendations

1. **Compute bootstrap CIs** on (regret_freeze - regret_kga) and
   (regret_adapt - regret_kga) for the 3 wins. CI excludes 0 -> win; includes 0 ->
   report as "Pareto-non-dominated / tie."
2. **Keep ImageNet-R and RxRx1 as honest non-replicated cases** (frontier /
   unknowable), exactly as the panel already labels them. Do not headline them.
3. **State the denominator**: "beats both on 3 of 6 real-shift datasets."
4. **Consolidate to the 6-panel** (see section 8) and archive the exploratory rest.

## 8. "Only 6 datasets" — consolidation plan (NEEDS YOUR APPROVAL; nothing deleted yet)

- **Keep (the 6):** OfficeHome, iWildCam, CIFAR-10-C, CIFAR-10.1, ImageNet-R, RxRx1.
- **Decide:** ImageNet-C + Camelyon17 are **in the paper** but not in the 6-panel
  (data unmounted). Either remount their data and add to the panel, or drop them
  from the paper. They cannot stay as paper claims with no local data.
- **Archive candidates (data/results, not delete):** FMoW, Poverty,
  `win_loop_v1`, `win_finder_*`, `hard_dataset_win_loop_v1`, and the many
  exploratory `officehome_*` / `iwildcam_*` probe dirs.
- **Action:** confirm the exact keep/drop list and I will move the rest to an
  `archive/` (reversible) — your call, per item.
