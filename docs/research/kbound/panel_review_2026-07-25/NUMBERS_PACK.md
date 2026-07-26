# K-Bound NUMBERS PACK — corrected numbers recomputed from raw artifacts

**Produced by:** the RECOMPUTE slice of the fix-queue work.
**Machine-readable twin:** `/home/claude/kb_fixes/NUMBERS_PACK.json` (39 entries, one per number,
with `{id, description, value, old_value, artifact_paths, method, changes_decisions, note}`).
**Scripts:** `/home/claude/kb_fixes/recompute/*.py` — all runnable, all seeded, all reading only from
`/home/claude/kb`. Raw outputs are the `out_*.json` files in the same directory.
**Ready-to-paste LaTeX:** `recompute/latex_item2.tex`, `latex_item3.tex`, `latex_item5.tex`
(also inlined below).

> **Rule for the four editing agents:** every number you write into the paper must appear in this
> pack or in an artifact you read yourself. Where this pack disagrees with `review_6_overall.md`,
> **this pack wins** — the disagreements are flagged in **§0** and each is independently reproducible
> by running the named script.

---

## 0. Where I disagree with the review panel — READ THIS FIRST

Everything else in this pack confirms the panel, usually to the last digit. These four items do not.

### 0.1 **[MAJOR]** Items 3 and 4 applied *together* kill the ImageNet-C freeze-gap CI

`review_6_overall.md` (fix-queue item 4, and "Where the panel disagrees" §3) says:

> "ImageNet-C becomes 0.0289 with FA_u = 1/135 (say so — beats-both against freeze still holds)"
> … "beats-both against freeze survives (0.0289 < 0.0319)"

The **point estimate** claim is correct and I reproduce it exactly. The **interval** claim is not.
Once the leave-one-out-of-pool radius (item 4) is applied *and* the bootstrap is run at the unit the
text describes (item 3), the freeze-side interval includes zero at **every** legitimate unit:

| design | KGA − always-adapt | KGA − always-freeze |
|---|---|---|
| in-pool exact rank, 27 seed-averaged conditions | `[-0.0806, +0.0175]` | `[-0.0092, -0.0023]` ✱ |
| **LOO exact rank, 27 seed-averaged conditions** | `[-0.0755, +0.0181]` | **`[-0.0085, +0.0038]`** |
| **LOO exact rank, 135 rows i.i.d. (the generous design)** | `[-0.0491, -0.0013]` ✱ | **`[-0.0079, +0.0036]`** |
| LOO exact rank, cluster = corruption family | `[-0.0321, -0.0197]` ✱ | `[-0.0036, -0.0020]` ✱ |

(✱ = excludes zero. The corruption-family row has only **3** clusters — the ImageNet-C grid is
gaussian_noise / shot_noise / impulse_noise — so it must not be reported as a primary interval.)

**What to write instead:** if you adopt the LOO radius, ImageNet-C SAR supports a *point-estimate*
no-harm-vs-always-freeze claim (0.0289 vs 0.0319) and **not** a CI-supported beats-both. Do not
write "with a CI excluding zero" for the freeze gap under the LOO radius.
Reproduce: `python3 recompute/03_imagenetc_bootstrap.py`.

### 0.2 **[MAJOR]** EATA's cluster-robust adapt-gap CI does **not** exclude zero

`review_6_overall.md` item 17 says clustering by corruption family "widens CIs 2.4–3.9× (all still
exclude zero)". True for **Tent**. **False for EATA**:

| CIFAR-10-C EATA, adapt gap | interval | excludes zero? |
|---|---|---|
| 432 cells i.i.d. (as run) | `[-0.00280, -0.00123]` | yes |
| 216 twin-pairs | `[-0.00311, -0.00096]` | yes |
| 12 corruption × severity clusters | `[-0.00483, +0.00043]` | **no** |
| 6 corruption-family clusters | `[-0.00436, +0.00035]` | **no** |

EATA also has **two** corruption families where KGA is worse than always-adapt
(`gaussian_noise +0.00022`, `jpeg_compression +0.00292`), not one.
Reproduce: `python3 recompute/04_cifar_cluster.py`.

### 0.3 **[MAJOR]** ImageNet-R: KGA is worse than always-adapt on **7 of 10** backbones, not one

`review_6_overall.md` item 23 says the mean "hides convnext_tiny at 0.0207 vs 0.0015 (14×) and
convnext_base at a degenerate 0% harmful base rate". The real picture is worse: seven backbones,
and **four** have a degenerate 0% harmful base rate. Full table in §7.2.
Reproduce: `python3 recompute/09_panel_variance.py`.

### 0.4 **[MINOR, but it changes item 2's scope]** "One rule everywhere" moves five more tracks

`review_6_overall.md` item 2 says "Declare the rule once, in the config table." If the declared rule
is **exact rank**, then the following published rows all change, because they are currently the
**interpolated** rule:

| track | interpolated (published) | exact rank |
|---|---|---|
| ImageNet-R D, mean across 10 backbones | 0.011203 | **0.015146** |
| CIFAR-10.1 SAR (`cifar101_multiseed_v1`, n=120) | 0.004467, FA_u 0.0250 | **0.005700**, FA_u 0.0083 |
| Camelyon17 Table VIII SAR | 0.041016, FA_u 0.0278 | **0.042535**, FA_u 0.0000 |
| CIFAR-10-C Tent (head-to-head, n=2160) | 0.00157361 | **0.00158518** |
| CIFAR-10-C EATA (head-to-head, n=2160) | 0.00126759 | **0.00127986** |

The CIFAR-10-C shifts are 4th-decimal and harmless. The ImageNet-R shift is 35% and it moves the
row *away* from always-adapt (0.0064). Decide whether the declaration is global or per-track and say
so explicitly. Reproduce: `python3 recompute/06_decision_accounting.py`.

---

## 1. Conventions used everywhere in this pack

Lifted verbatim from the shipped code so that every "old" column reproduces the paper.

| name | definition | shipped at |
|---|---|---|
| interpolated radius | `np.quantile(np.abs(Bhat - B), 1 - alpha)` | `cifar_tent_mps_v2.py:162`, `run_wilds_camelyon17.py:56` |
| exact-rank radius | `rho_(k)`, `k = min(n, ceil((n+1)(1-alpha)))` | `g8_canonical_pooling.py:4` |
| in-pool | one radius per file from **all** residuals, including the scored cell's own | every shipped runner |
| leave-one-out-of-pool (LOO) | cell *i*'s radius excludes cell *i*'s residual | **the fix** (item 4) |
| decision | `ADAPT` iff `bhat-eps>0`; `FREEZE` iff `bhat+eps<0`; else `ABSTAIN` | `g8_canonical_pooling.py:12` |
| regret | `oracle = max(a0, a_adapted)`; KGA takes `a_adapted` iff ADAPT; `regret = oracle − policy` | `_locked_analysis_script.py:37-42` |
| `FA_u` | `mean(is_adapt & (B <= 0))` — the quantity `thm:certificate` bounds | `_locked_analysis_script.py:43` |
| `FA_c` | `mean(B <= 0 | is_adapt)` | `wilds/analysis.py:87` |
| bootstrap | paired percentile, **20 000** replicates, fixed rng seed (20260720 for ImageNet-C, 20260611 for CIFAR — the stream seeds the shipped scripts use) | |
| α | 0.10 everywhere | |

Environment: Python 3.12 container, numpy, scipy 1.17.1, scikit-learn 1.8.0.
**Caveat carried into §8:** the shipped `b_hat` was produced under an unpinned scikit-learn
(0 of 43 run manifests record a version), so refits reproduce it at correlation 0.999996–1.000000
but not bit-for-bit.

---

## 2. Fix-queue item 2 — ImageNet-C per-seed table under one declared rule

**Artifacts:** `experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed{0..4}.json`
**Script:** `recompute/01_imagenetc_perseed.py` → `out_imagenetc_perseed.json`

### 2.1 Reproduction check (both rules, in-pool)

| | pooled KGA | always-adapt | always-freeze | FA_u | ADAPT/FREEZE/ABSTAIN |
|---|---|---|---|---|---|
| **interpolated** (what `kbound_short_appendix.tex:303-310` prints) | 0.010748 | 0.052933 | 0.031894 | 1/135 = 0.0074 | 65 / 16 / 54 |
| **exact rank** (what `kbound_result_manifest.json` promotes) | **0.026422** | 0.052933 | 0.031894 | 0/135 = 0.0000 | 12 / 14 / **109** |

The exact-rank row reproduces the manifest (`[0.026422222, 0.0529333334, 0.0318944445]`,
`abstain_count: 109`) to seven decimals. The interpolated row reproduces the appendix table
(0.0108 / 0.0091 / 0.0128 / 0.0056 / 0.0154, pooled 0.0107, FA_u seed 2 = 0.037, pooled 0.007).
**Both rules are in the same frozen document, 2.5× apart. Confirmed.**

### 2.2 Per seed, exact-rank rule — the sentence at `:287` / `:801` is false

| seed | KGA | always-adapt | always-freeze | FA_u | AD | FR | AB | beats both | tie with freeze |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.031926 | 0.062519 | 0.031926 | 0.000 | 0 | 3 | 24 | no | **bit-identical** |
| 1 | 0.031241 | 0.059481 | 0.031241 | 0.000 | 0 | 3 | 24 | no | **bit-identical** |
| 2 | 0.010222 | 0.042537 | 0.028361 | 0.000 | 9 | 2 | 16 | **yes** | no |
| 3 | 0.029028 | 0.044074 | 0.029028 | 0.000 | 0 | 3 | 24 | no | **bit-identical** |
| 4 | 0.029694 | 0.056056 | 0.038917 | 0.000 | 3 | 3 | 21 | **yes** | no |
| **pooled** | **0.026422** | 0.052933 | 0.031894 | 0.000 | 12 | 14 | 109 | yes | — |

**Seeds strictly beating both: 2/5 (seeds 2 and 4). Bit-identical ties with always-freeze: 3/5
(seeds 0, 1, 3).** The pooled win rests on 12 ADAPT decisions, 9 of them from seed 2.
This **confirms** F1-2 and review_6's item 2 exactly, including the action composition
`seed0 {FREEZE:3, ABSTAIN:24}` … `pooled {FREEZE:14, ABSTAIN:109, ADAPT:12}`.

Per-seed Clopper–Pearson 95% upper bounds on `FA_c` (0 false adapts): seed 2 (9 adapts) **0.2831**,
seed 4 (3 adapts) **0.6316**, seeds 0/1/3 undefined (0 adapts).

### 2.3 Ready-to-paste LaTeX (`recompute/latex_item2.tex`)

```latex
%% fix-queue item 2 -- regenerated under the PROMOTED exact-rank rule
%% source: experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/
%%         per_condition_imagenetc_sar_seed{0..4}.json
%% eps = rho_(k), k = min(n, ceil((n+1)(1-alpha))), fitted per seed, alpha = 0.10
\begin{table}[t]\centering\small
\caption{\textbf{ImageNet-C SAR, per seed, under the exact split-conformal rank rule}
($n{=}27$ conditions per seed, $\alpha{=}0.10$). Regret is against the per-cell oracle
$\max(a_0,a_a)$. ``$\equiv$'' marks a seed on which \textsc{KGA} never adapts, so its
regret is bit-identical to always-freeze. Point estimates improve \emph{both}
fixed-policy regrets on 2/5 seeds (seeds 2, 4); on seeds 0, 1, 3 \textsc{KGA} abstains throughout.
The pooled win is driven by seeds 2, 4.}
\label{tab:imagenetc-perseed}
\begin{tabular}{lcccccccc}
\toprule
seed & \textsc{KGA} & always-adapt & always-freeze & $\mathrm{FA}_{\mathrm u}$
& \textsc{ad} & \textsc{fr} & \textsc{ab} & beats both \\
\midrule
0 & $0.0319$ & $0.0625$ & $0.0319$\;$\equiv$ & $0.000$ & 0 & 3 & 24 & --- \\
1 & $0.0312$ & $0.0595$ & $0.0312$\;$\equiv$ & $0.000$ & 0 & 3 & 24 & --- \\
2 & $0.0102$ & $0.0425$ & $0.0284$ & $0.000$ & 9 & 2 & 16 & \checkmark \\
3 & $0.0290$ & $0.0441$ & $0.0290$\;$\equiv$ & $0.000$ & 0 & 3 & 24 & --- \\
4 & $0.0297$ & $0.0561$ & $0.0389$ & $0.000$ & 3 & 3 & 21 & \checkmark \\
\midrule
pooled & $\mathbf{0.0264}$ & $0.0529$ & $0.0319$ & $0.000$ & 12 & 14 & 109 & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
```

### 2.4 What "one rule everywhere" costs elsewhere

See **§0.4**. Under exact rank: ImageNet-R panel row 0.011203 → **0.015146**; CIFAR-10.1 SAR
0.004467 → **0.005700**; Camelyon Table VIII SAR 0.041016 → **0.042535**; CIFAR-10-C Tent
0.00157361 → **0.00158518**; CIFAR-10-C EATA 0.00126759 → **0.00127986**.

---

## 3. Fix-queue item 3 — ImageNet-C bootstrap at the right unit

**Artifacts:** as §2. **Script:** `recompute/03_imagenetc_bootstrap.py` → `out_imagenetc_boot.json`
**Method:** pooled = mean over the 5 seeds per condition, then a paired percentile bootstrap over the
27 conditions — the design `_locked_analysis_script.py:54` uses for the CIFAR-10-C rows and the one
`kbound_short.tex:797-802` describes. 20 000 replicates, rng seed 20260720.

### 3.1 Promoted (in-pool exact-rank) radius

Point estimates 0.026422 / 0.052933 / 0.031894.

| resampling unit | units | KGA − always-adapt | KGA − always-freeze | beats-both by CI |
|---|---|---|---|---|
| 135 cell-seed rows i.i.d. — **as `g8_exactrank_ci.py:18` codes it** | 135 | `[-0.0519, -0.0034]` ✱ | `[-0.0087, -0.0027]` ✱ | yes |
| **27 conditions, seed-averaged — as the text describes** | 27 | **`[-0.0806, +0.0175]`** | `[-0.0092, -0.0023]` ✱ | **no** |
| cluster = condition (135 rows, 27 clusters) | 27 | `[-0.0808, +0.0175]` | `[-0.0093, -0.0023]` ✱ | no |
| cluster = seed | 5 | `[-0.0308, -0.0204]` ✱ | `[-0.0127, +0.0000]` | no |
| cluster = corruption family | 3 | `[-0.0321, -0.0197]` ✱ | `[-0.0094, -0.0034]` ✱ | yes† |

† only 3 clusters — do not report as primary.

The i.i.d.-135 row reproduces the manifest's `gap_kga_minus_adapt_ci95 = [-0.0518416, -0.0038235]`
and `gap_kga_minus_freeze_ci95 = [-0.0086428, -0.0026000]` (difference is bootstrap noise: the
shipped script uses 5 000 replicates, this pack uses 20 000). **Confirms F3-1 and the chief
reviewer's arbitration.**

**Defensible claim under the promoted (in-pool) radius:** *"beats the better fixed policy
(always-freeze) with a paired-bootstrap CI excluding zero at the condition level; the gap to
always-adapt is not CI-supported at that unit."*

**If item 4's LOO radius is also adopted, see §0.1 — the freeze gap also loses its interval.**

### 3.2 Per-corruption-family breakdown (seed-averaged, in-pool exact rank)

| family | n conds | KGA | always-adapt | always-freeze | gap vs adapt | gap vs freeze |
|---|---|---|---|---|---|---|
| gaussian_noise | 9 | 0.02941 | 0.04908 | 0.03302 | −0.01967 | −0.00362 |
| impulse_noise | 9 | 0.02884 | 0.05662 | 0.03827 | −0.02778 | −0.00943 |
| shot_noise | 9 | 0.02102 | 0.05310 | 0.02439 | −0.03208 | −0.00337 |

### 3.3 Ready-to-paste LaTeX (`recompute/latex_item3.tex`)

```latex
%% fix-queue item 3 -- ImageNet-C SAR gap CIs at the correct unit of analysis
%% 20000 paired percentile-bootstrap replicates, rng seed 20260720
\begin{table}[t]\centering\small
\caption{\textbf{ImageNet-C SAR: the beats-both interval depends on the resampling unit.}
Paired percentile bootstrap, $20{,}000$ replicates, promoted exact-rank radius. The
manuscript describes a seed-averaged condition-level design; the shipped script resampled
the $135$ cell--seed rows i.i.d. At the condition level the gap to always-adapt is
\emph{not} interval-supported. $^{\ast}$ marks an interval excluding zero.}
\label{tab:imagenetc-ci-unit}
\begin{tabular}{lccc}
\toprule
resampling unit & units & \textsc{KGA} $-$ always-adapt & \textsc{KGA} $-$ always-freeze \\
\midrule
$135$ cell--seed rows, i.i.d.\ (as coded) & 135 & $[-0.0519,\,-0.0034]^{\ast}$ & $[-0.0087,\,-0.0027]^{\ast}$ \\
\textbf{$27$ conditions, seed-averaged} (as described) & 27 & $[-0.0806,\,+0.0175]$ & $[-0.0092,\,-0.0023]^{\ast}$ \\
cluster $=$ seed & 5 & $[-0.0308,\,-0.0204]^{\ast}$ & $[-0.0127,\,+0.0000]$ \\
cluster $=$ corruption family & 3 & $[-0.0321,\,-0.0197]^{\ast}$ & $[-0.0094,\,-0.0034]^{\ast}$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 4. Fix-queue item 4 — leave-one-out-of-pool conformal radius

**Scripts:** `recompute/02_cifar_loo_radius.py`, `01_imagenetc_perseed.py`, `06_decision_accounting.py`

### 4.1 CIFAR-10-C: **zero** decisions change — CONFIRMED and extended

| tree | files | cells | decisions changed (interp) | decisions changed (exact) | FA_u |
|---|---|---|---|---|---|
| stress grid tent+eata, seeds 1–4 | 8 | **3 456** | **0** | **0** | 0 everywhere |
| head-to-head tent kga, seeds 0–4 | 5 | 2 160 | **0** | **0** | 0 everywhere |
| head-to-head eata kga, seeds 0–4 | 5 | 2 160 | **0** | **0** | 0 everywhere |
| stress grid SAR, seeds 1–4 | 4 | 1 728 | **0** | **0** | 0 everywhere |
| **total** | 22 | **9 504** | **0** | **0** | |

The panel's "0 of 3456" is confirmed and holds across **all 9 504 committed CIFAR-10-C cells**.
Regret triples are bit-identical in-pool vs LOO on every one of these groups:

| track | regret (KGA / adapt / freeze), interp in-pool **= interp LOO** | exact in-pool **= exact LOO** |
|---|---|---|
| head-to-head Tent, 5 seeds | 0.00157361 / 0.00792338 / 0.12409792 | 0.00158518 / same / same |
| head-to-head EATA, 5 seeds | 0.00126759 / 0.00326829 / 0.13137893 | 0.00127986 / same / same |
| stress-grid Tent, seeds 1–4 | 0.00164062 / 0.00782118 / 0.12412471 | 0.00165509 / same / same |
| stress-grid EATA, seeds 1–4 | 0.00126997 / 0.00317911 / 0.13140596 | 0.00128530 / same / same |
| stress-grid SAR, seeds 1–4 | 0.00159896 / 0.00030990 / 0.14049653 | 0.00161690 / same / same |

The head-to-head Tent/EATA triples reproduce `kbound_result_manifest.json` exactly
(`0.0015736109 / 0.0079233799 / 0.1240979162` and `0.0012675925 / 0.0032682874 / 0.1313789343`),
confirming F4-5: the promoted panel numbers come from `mixed_headtohead_v1/`, **not** from
`stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json` as the manifest's `source` field claims.
`LOCKED_ANALYSIS_RESULTS.json` holds `0.0016259256 / 0.0079756946 / 0.1239368049`.

**Write:** *"we verified that removing the scored cell from its own radius pool changes 0 of 9 504
CIFAR-10-C decisions and leaves FA_u at 0."* This is a strength, not a hedge.

### 4.2 ImageNet-C SAR: 2 of 135 decisions change — CONFIRMED

| | KGA | adapt | freeze | FA_u | AD | FR | AB | CP95 upper on FA_c |
|---|---|---|---|---|---|---|---|---|
| exact rank, in-pool (promoted) | 0.026422 | 0.052933 | 0.031894 | 0/135 = 0.0000 | 12 | 14 | 109 | 0.2209 |
| **exact rank, LOO (the fix)** | **0.028893** | 0.052933 | 0.031894 | **1/135 = 0.0074** | 13 | 15 | 107 | **0.3163** |
| interpolated, in-pool | 0.010748 | 0.052933 | 0.031894 | 1/135 = 0.0074 | 65 | 16 | 54 | 0.0709 |
| interpolated, LOO | 0.012170 | 0.052933 | 0.031894 | 2/135 = 0.0148 | 61 | 17 | 57 | 0.0996 |

Confirms F4-2 to seven decimals (0.0288926, abstain 107, FA_u 1/135). Per-seed under LOO exact rank,
seeds beating both are still {2, 4}; seeds {0, 1, 3} remain bit-identical ties with always-freeze.
**But see §0.1: the CI does not survive.**

Also recomputed: **ImageNet-C EATA** in-pool 0.000924 → LOO 0.000670, FA_u 0/135 → 1/135
(3 decisions change); **ImageNet-C Tent** never adapts under either rule (0/135 ADAPT, KGA ≡ freeze).

### 4.3 Camelyon17 Table VIII (`kbound_short.tex:889-891`) re-scored

**Artifacts:** `experiments/kbound/results/wilds_kbound/per_condition_camelyon17_{tent,eata,sar}_seed{0..3}.json`
(9 cells × 4 seeds each). Published table reproduces **exactly** from the *interpolated in-pool*
column, i.e. from `run_wilds_camelyon17.py:56`.

| candidate | rule / pool | KGA | adapt | freeze | FA_u | FA_c | ADAPT | realized ε range |
|---|---|---|---|---|---|---|---|---|
| Tent | interp in-pool **(published)** | 0.020074 | 0.138021 | 0.020074 | 0/36 | — | 0 | **0.1527 – 0.3719** |
| Tent | interp LOO | 0.020074 | 0.138021 | 0.020074 | 0/36 | — | 0 | 0.1370 – 0.3731 |
| Tent | exact in-pool | 0.020074 | 0.138021 | 0.020074 | 0/36 | — | 0 | 0.1696 – 0.3947 |
| EATA | interp in-pool **(published)** | 0.039280 | 0.041667 | 0.042426 | 0/36 | 0.000 | 1 | 0.0708 – 0.1742 |
| EATA | interp LOO | 0.035373 | 0.041667 | 0.042426 | 0/36 | 0.000 | 2 | 0.0535 – 0.1799 |
| SAR | interp in-pool **(published)** | 0.041016 | 0.000217 | 0.065430 | 1/36 = 0.0278 | 0.143 | 7 | 0.0720 – 0.0915 |
| SAR | **interp LOO (the fix)** | **0.041124** | 0.000217 | 0.065430 | **2/36 = 0.0556** | **0.250** | 8 | 0.0660 – 0.0971 |
| SAR | exact in-pool | 0.042535 | 0.000217 | 0.065430 | 0/36 | 0.000 | 5 | 0.0844 – 0.1481 |
| SAR | exact LOO | 0.042535 | 0.000217 | 0.065430 | 1/36 = 0.0278 | 0.167 | 6 | 0.0689 – 0.1481 |

Per-seed (interp in-pool, as published): Tent ε = 0.1527 / 0.2155 / 0.3305 / 0.3719 — the panel's
"0.153–0.372", **confirmed**; the SAR row's printed `FA 0.11` is **seed 0's 1/9 = 0.1111**.

**Two structural facts the paper must state about this table:**
1. At n = 9, `k = min(9, ceil(10·0.9)) = 9`, so under the exact-rank rule **ε is the maximum
   residual** and `FA_u` is forced to exactly 0. The exact-rank column of Table VIII carries no
   information. The same holds for RxRx1 and ImageNet-R (n = 12: `k = 12`).
2. The "over-freezes" verdict for SAR is a consequence of an ε of 0.07–0.09 on a track whose
   benefits are ~0.001 in magnitude, and the LOO fix makes the SAR row **worse**, not better
   (FA_u 0.0278 → 0.0556, FA_c 0.143 → 0.250).

### 4.4 Every other track's LOO delta

Full table in §5.2. Decisions changed by LOO (exact rank): ImageNet-C SAR 2, ImageNet-C EATA 3,
Camelyon17 SAR 1, CIFAR-10.1 Tent 1, CIFAR-10.1 SAR 1, ImageNet-R efficientnet_b0 1,
ImageNet-R resnet152 1, **everything else 0**.

---

## 5. Fix-queue item 5 — decision accounting and the FA_u identity

**Script:** `recompute/06_decision_accounting.py`, `10_identity_and_promoted_rows.py`

### 5.1 (a) `FA_u ≤ α` is an arithmetic identity — CONFIRMED on 69/69 files

With ε the *k*-th order statistic of the same residual vector it is used to test, the miscoverage
count is identically `N−k`.

| n | k | exact-rank FA_u ceiling (N−k)/N | ≤ α? | interpolated exceedance | > α? | interpolated coverage |
|---|---|---|---|---|---|---|
| 864 | 779 | 0.098380 | yes | 0.100694 | yes | 0.899306 |
| **432** | 390 | **0.097222** | yes | **0.101852** | yes | **0.898148** |
| **27** | 26 | **0.037037** | yes | **0.111111** | yes | **0.888889** |
| 24 | 23 | 0.041667 | yes | 0.125000 | yes | 0.875000 |
| 9 | 9 | **0.000000** | yes | 0.111111 | yes | 0.888889 |

**69 of 69** shipped per-condition files hit the exact-rank ceiling exactly. This confirms the chief
reviewer's arbitration digit for digit (0.0972 / 0.0370 exact rank; 0.1019 / 0.1111 interpolated).
The coverage values 0.898148 and 0.888889 are exactly what
`paper/generated/empirical_audit/decision_metrics.json` reports as `interval_coverage_observed` —
they are functions of *n* alone.

**Sentence to write:** *"Under in-sample rank calibration `FA_u ≤ (N−k)/N` is an identity, so the
informative statistic is `FA_u = 0` against the ceiling — 0.097 at n = 432 and 0.037 at n = 27 —
not `FA_u ≤ α`. At n ≤ 12 the ceiling is 0 and the statistic is vacuous."*
This also justifies **item 5(d)**: delete the Wilson intervals on `interval_coverage_observed`,
which are intervals on a deterministic function of n.

### 5.2 (b) ADAPT/FREEZE/ABSTAIN + Clopper–Pearson for every promoted panel row

CP upper bound = `scipy.stats.beta.ppf(0.95, k+1, n−k)` (one-sided 95%).

| track | N | ADAPT | FREEZE | ABSTAIN | false adapts | FA_u | **CP95 upper on FA_c** | status |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10-C Tent (5×432) | 2160 | 1113 | 358 | 689 | 0 | 0.0000 | **0.00269** | powered |
| CIFAR-10-C EATA (5×432) | 2160 | 1244 | 130 | 786 | 0 | 0.0000 | **0.00241** | powered |
| ImageNet-C SAR (5×27) | 135 | 12 | 14 | 109 | 0 | 0.0000 | 0.2209 | weak |
| Office-Home M v2 | 35 | 22 | 12 | 1 | 0 | 0.0000 | 0.1273 | weak |
| iWildCam H v2 | 72 | **1** | 60 | 11 | 0 | 0.0000 | 0.9500 | **guarantee untested** |
| Camelyon17 OOD (n=18) | 18 | — | — | — | 0 | 0.0000 | undefined | **BLOCKED-NEEDS-DATA** |
| RxRx1 J | 60 | **0** | 60 | 0 | 0 | 0.0000 | undefined | **guarantee untested** |
| CIFAR-10.1 K | 48 | 18 | 24 | 6 | 8 | 0.1667 | 0.6594 | diagnostic fail |
| controlled multimodal D33 | 130 | **9** | 119 | 2 | 0 | 0.0000 | 0.2831 | **guarantee untested** |

Notes on provenance:
- CIFAR-10-C and ImageNet-C rows are recomputed cell-by-cell under the exact-rank rule. Under the
  **interpolated** rule CIFAR-10-C Tent has **1114** adapts (CP95 upper 0.00269 either way — the
  panel's "0/1114 adapts, 0.0027" is confirmed).
- Office-Home / iWildCam / RxRx1 / CIFAR-10.1 counts are `n_test × adapt_rate` from the promoted
  summary artifacts, because the raw record files they name are absent (§8).
  CIFAR-10.1's counts are exact: `FA_u 0.1667 = 8/48`, `FA_c 0.4444 = 8/18`, `commit_rate 0.875 =
  42/48` — source is `experiments/kbound/results/cifar101_protocol_K_v1/analyze_F_results.json`,
  **which exists and should be written into the manifest's currently-absent `source` field.**
- D33 counts from `kbound_result_manifest.json` `tracks/controlled_multimodal_D33.decision_counts`.

**Tracks to mark "guarantee untested" (< 10 ADAPT):** RxRx1 (0), iWildCam (1), D33 (9).
Office-Home (22) and ImageNet-C SAR (12) are above the bar but their CP bounds — 0.127 and 0.221 —
are 2× and 2.2× the declared α, so their observed zeros do **not** certify FA_c ≤ 0.10.

### 5.3 Recomputable per-cell tracks (all four radius variants)

These are the tracks whose per-cell artifacts exist; the numbers may differ from the promoted panel
row because the promoted row was computed on a different split (see §8).

| track (per-cell artifact) | N | AD | FR | AB | FA | FA_u | KGA regret | LOO Δdecisions |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10-C Tent h2h ×5 | 2160 | 1113 | 358 | 689 | 0 | 0.0000 | 0.001585 | 0 |
| CIFAR-10-C EATA h2h ×5 | 2160 | 1244 | 130 | 786 | 0 | 0.0000 | 0.001280 | 0 |
| CIFAR-10-C Tent stress ×4 | 1728 | 887 | 286 | 555 | 0 | 0.0000 | 0.001655 | 0 |
| CIFAR-10-C EATA stress ×4 | 1728 | 999 | 100 | 629 | 0 | 0.0000 | 0.001285 | 0 |
| CIFAR-10-C SAR stress ×4 (quarantined) | 1728 | 1156 | 0 | 572 | 0 | 0.0000 | 0.001617 | 0 |
| ImageNet-C SAR ×5 | 135 | 12 | 14 | 109 | 0 | 0.0000 | 0.026422 | 2 |
| ImageNet-C Tent ×5 | 135 | **0** | 0 | 135 | 0 | 0.0000 | 0.014465 | 0 |
| ImageNet-C EATA ×5 | 135 | 118 | 0 | 17 | 0 | 0.0000 | 0.000924 | 3 |
| Camelyon17 Tent (Table VIII) | 36 | **0** | 2 | 34 | 0 | 0.0000 | 0.020074 | 0 |
| Camelyon17 EATA (Table VIII) | 36 | **1** | 0 | 35 | 0 | 0.0000 | 0.039280 | 0 |
| Camelyon17 SAR (Table VIII) | 36 | **5** | 0 | 31 | 0 | 0.0000 | 0.042535 | 1 |
| Camelyon17 fullscale_B_v2 Tent ×3 | 108 | 40 | 0 | 68 | 1 | 0.0093 | 0.029604 | 0 |
| iWildCam tent_episodic ×2 (extracted) | 144 | **0** | 28 | 116 | 0 | 0.0000 | 0.021174 | 0 |
| Office-Home sar_aggr ×5 (extracted) | 180 | 114 | 66 | 0 | 0 | 0.0000 | 0.000000 | 0 |
| RxRx1 sar_online ×5 (extracted) | 60 | **0** | 60 | 0 | 0 | 0.0000 | 0.000000 | 0 |
| CIFAR-10.1 Tent ×5 | 120 | **0** | 46 | 74 | 0 | 0.0000 | 0.002408 | 1 |
| CIFAR-10.1 SAR ×5 | 120 | 33 | 0 | 87 | 1 | 0.0083 | 0.005700 | 1 |
| CIFAR-10.1 EATA ×5 | 120 | **2** | 10 | 108 | 0 | 0.0000 | 0.003550 | 0 |
| ImageNet-R × 10 backbones × 4 seeds | 48 each | 0–46 | 0–28 | 2–48 | 0 | 0.0000 | see §7.2 | 0–1 |

(exact-rank in-pool columns; bold ADAPT counts are < 10, i.e. guarantee untested.
Full four-variant detail for all 30 tracks is in `out_decision_accounting.json` and in the
`item5.full_track_table` entry of the JSON pack.)

### 5.4 Ready-to-paste LaTeX (`recompute/latex_item5.tex`)

```latex
%% fix-queue item 5 -- action composition and a Clopper-Pearson bound for every panel row
\begin{table}[t]\centering\small
\caption{\textbf{Action composition and the strength of the false-adapt evidence, per track.}
$\mathrm{FA}_{\mathrm u}$ is the marginal false-adapt rate the certificate bounds;
CP$_{95}$ is the one-sided Clopper--Pearson upper bound on the \emph{conditional} rate
$\mathrm{FA}_{\mathrm c}=\Pr[\Delta\le 0\mid\textsc{adapt}]$, i.e.\ how much the observed
zero actually constrains the guarantee. $^{\dagger}$ fewer than $10$ \textsc{adapt}
decisions: guarantee untested. Only CIFAR-10-C exercises the guarantee with real power.}
\label{tab:decision-accounting}
\begin{tabular}{lccccccc}
\toprule
track & $N$ & \textsc{ad} & \textsc{fr} & \textsc{ab} & false & $\mathrm{FA}_{\mathrm u}$ & CP$_{95}$ upper on $\mathrm{FA}_{\mathrm c}$ \\
\midrule
CIFAR-10-C Tent & 2160 & 1113 & 358 & 689 & 0 & $0.0000$ & $0.0027$ \\
CIFAR-10-C EATA & 2160 & 1244 & 130 & 786 & 0 & $0.0000$ & $0.0024$ \\
ImageNet-C SAR & 135 & 12 & 14 & 109 & 0 & $0.0000$ & $0.2209$ \\
Office-Home M v2 & 35 & 22 & 12 & 1 & 0 & $0.0000$ & $0.1273$ \\
iWildCam H v2$^{\dagger}$ & 72 & 1 & 60 & 11 & 0 & $0.0000$ & $0.9500$ \\
Camelyon17 OOD & 18 & --- & --- & --- & 0 & $0.0000$ & undef. \\
RxRx1 J$^{\dagger}$ & 60 & 0 & 60 & 0 & 0 & $0.0000$ & undef. \\
CIFAR-10.1 K & 48 & 18 & 24 & 6 & 8 & $0.1667$ & $0.6594$ \\
controlled multimodal D33$^{\dagger}$ & 130 & 9 & 119 & 2 & 0 & $0.0000$ & $0.2831$ \\
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize With in-sample rank calibration the miscoverage count is identically
$N-k$, so $\mathrm{FA}_{\mathrm u}\le (N-k)/N$ holds for \emph{any} data: the ceiling is
$0.0972$ at $n{=}432$ and $0.0370$ at $n{=}27$, both below $\alpha{=}0.10$. The
informative statistic is therefore $\mathrm{FA}_{\mathrm u}{=}0$ \emph{versus that
ceiling}, not ``$\mathrm{FA}_{\mathrm u}\le\alpha$''.}
\end{table}
```

---

## 6. Fix-queue item 6 — CIFAR-10-C SAR quarantine

**Artifacts:** `experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json`
(per-seed rows) and `seed{1..4}/per_condition_cifar10c_sar_seed{s}.json` (recomputed).
**Script:** `recompute/09_panel_variance.py`, `02_cifar_loo_radius.py`

| seed | harmful base rate | KGA | always-adapt | always-freeze | ε | beats both |
|---|---|---|---|---|---|---|
| **0** | **0.5278** | 0.001351 | **0.054705** | 0.081192 | 0.026788 | **yes** |
| 1 | 0.1019 | 0.001343 | 0.000307 | 0.140056 | 0.012707 | no |
| 2 | 0.0880 | 0.001664 | 0.000310 | 0.140971 | 0.012713 | no |
| 3 | 0.0741 | 0.001928 | 0.000278 | 0.140304 | 0.013698 | no |
| 4 | 0.0995 | 0.001461 | 0.000345 | 0.140655 | 0.013083 | no |
| **5-seed pooled** | 0.1783 | 0.0015493 | 0.0111889 | 0.1286356 | cv **0.3897** | 1/5 |
| **seeds 1–4 only** | 0.0909 | **0.0015990** | **0.0003099** | 0.1404965 | | 0/4 |

- Seed 0's harmful base rate is **5.81×** the mean of seeds 1–4.
- On seeds 1–4, KGA's regret (**0.0015990**) **exceeds** always-adapt's (**0.0003099**) by 5.2×.
- The seeds-1–4 triple reproduces **exactly** from the four committed per-condition files:
  0.00159896 / 0.00030990 / 0.14049653.
- 1 of 5 seeds beats both, and it is seed 0 — the seed the quarantine exists for.

**Confirms review_6's replacement sentence to 5 decimals.** Recommended replacement text is exactly
what the fix queue proposes:

> "SAR's five-seed aggregate is dominated by seed 0, whose harmful base rate (0.53) is 5× the other
> seeds'; on seeds 1–4 KGA's regret (0.00160) exceeds always-adapt's (0.00031). No comparative
> verdict is drawn."

**Two further facts the sentence at `:639-642` gets wrong:**
1. "rebuilt from all five saved per-condition seed files" — **only four exist** for the 432-cell
   grid. `stress_grid_multiseed_v1/seed0/` contains `decisive_tta_results.json`,
   `decisive_tta_table.md`, `result_manifest.json` and nothing else. A fifth complete set exists at
   `win_hunt_v5/cifar10c_aggr/seed0/` but that is a **270-cell** grid.
2. The cluster-robust picture is unambiguous and negative: SAR's adapt-gap CI is **entirely
   positive** at every resampling unit (432 cells `[+0.00101, +0.00158]`; 12 clusters
   `[+0.00003, +0.00298]`; 6 clusters `[+0.00012, +0.00271]`). Drop the "intervals exclude zero"
   clause; the intervals exclude zero **on the wrong side**.

---

## 7. Fix-queue items 17, 18, 19, 23

### 7.1 Item 17 — cluster-robust intervals and leave-one-corruption-out (CIFAR-10-C)

**Script:** `recompute/04_cifar_cluster.py` (fast), `05_cifar_loco.py` (~10 min, 5 seeds).

**r0/r1 replicate correlation** (the reason the effective n is ≤ 216):
Tent adapt-gap **0.9482**, freeze-gap **0.9990**, KGA regret 0.8020, over 216 design-point pairs.
EATA 0.9323 / 0.9994. SAR 0.7537 / 0.9996. Confirms F3-5 (0.948 / 0.999).

**Tent** (head-to-head, 5 seeds; point 0.00157361 / 0.00792338 / 0.12409792):

| resampling unit | units | KGA − always-adapt | width | KGA − always-freeze | width | beats both |
|---|---|---|---|---|---|---|
| 432 cells i.i.d. (as run) | 432 | `[-0.00787, -0.00489]` | 1.00× | `[-0.13738, -0.10810]` | 1.00× | yes |
| 216 twin-pairs | 216 | `[-0.00846, -0.00434]` | 1.38× | `[-0.14351, -0.10235]` | 1.41× | yes |
| 12 corruption × severity | 12 | `[-0.01157, -0.00158]` | 3.35× | `[-0.21426, -0.04230]` | 5.87× | yes |
| **6 corruption families** | 6 | `[-0.00952, -0.00259]` | **2.32×** | `[-0.17790, -0.06364]` | **3.90×** | **yes** |

Per-corruption adapt gaps (Tent): contrast −0.00825, defocus_blur −0.01054, fog −0.01029,
**gaussian_noise +0.00189** (KGA worse than always-adapt), jpeg_compression −0.00347,
pixelate −0.00744. **Confirms F3-5 exactly** (panel: −0.0083 / −0.0105 / −0.0103 / +0.0019 /
−0.0035 / −0.0074, widths 2.4× and 3.9×).

**EATA** — see **§0.2**: the adapt-gap CI does **not** exclude zero at 12 or 6 clusters, and two
families reverse sign (gaussian_noise +0.00022, jpeg_compression +0.00292).

**Leave-one-corruption-out calibration** (Tent, all 5 seeds refitted with the shipped
`GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8,
random_state=0)`; refit reproduces the stored `b_hat` at corr 0.999996–1.000000):

| calibration scheme | folds | residual MAE | R² | ε | adapt rate | FA_u | KGA regret |
|---|---|---|---|---|---|---|---|
| leave-one-**cell**-out (as shipped) | 432 | 0.00959 | 0.9929 | 0.02106 | 51.5% | **0** | 0.001585 |
| leave-one-**twin-pair**-out | 216 | 0.00981 | 0.9924 | 0.02189 | 51.2% | **0** | 0.001663 |
| leave-one-**corruption**-out | 6 | **0.03090** | **0.8954** | **0.09717** | 41.5% | **0** | **0.005867** |

Ratios: MAE **×3.22**, ε **×4.61**, regret **×3.70**. `FA_u = 0` under every partition, and
**beats-both survives** (0.005867 < 0.007923 always-adapt, < 0.124098 always-freeze).
Seed-0 numbers match the panel exactly (MAE 0.01021 → 0.03222, R² 0.991 → 0.892, ε 0.02144 →
0.09219, adapt 52.3% → 41.7%, regret 0.001306 → 0.005495).
**This ablation strengthens the paper — add it.**

Per-seed LOCO ε: 0.09219, 0.09806, 0.09922, 0.09983, 0.09655.
Per-seed LOCO KGA regret: 0.005495, 0.006260, 0.006148, 0.005530, 0.005902.

### 7.2 Item 18 — baseline parity and what the radius buys

`gate_baseline_comparison.py` cannot be run as released (its input `cifar10c_percell.json` exists
nowhere in the tree). **Script `recompute/08_gates_and_radius_value.py` regenerates `tab:gates` from
the committed head-to-head per-condition dumps**, 5 seeds × 432 cells, using the gate definitions
copied verbatim from `gate_baseline_comparison.py:45-104`.

| decision rule | regret | FA_u | FA_c | adapt | cov | FA_u(harm) | **calibration budget actually given** |
|---|---|---|---|---|---|---|---|
| confidence gate | 0.0079 | 0.249 | 0.293 | 0.85 | 1.00 | 0.755 | **none — unfitted sign rule** |
| entropy gate | 0.0081 | 0.247 | 0.294 | 0.84 | 1.00 | 0.750 | **none — unfitted sign rule** |
| drift/KL gate | 0.1241 | 0.000 | 0.000 | 0.00 | 1.00 | 0.000 | leave-one-corruption-out (6 folds) |
| ATC-style gate | 0.0041 | 0.116 | 0.172 | 0.68 | 1.00 | 0.353 | leave-one-corruption-out (6 folds) |
| KGA (no radius) | **0.0004** | 0.038 | 0.056 | 0.68 | 1.00 | 0.117 | **leave-one-CELL-out (431 fits)** |
| **KGA (certificate)** | 0.0016 | **0.000** | **0.000** | 0.52 | 0.68 | **0.000** | **leave-one-CELL-out (431 fits)** |
| *new:* KGA (certificate, leave-one-corruption-out) | 0.0059 | 0.000 | 0.000 | 0.42 | 0.42 | 0.000 | **same budget as gates 3–4** |
| *new:* KGA (no radius, leave-one-corruption-out) | 0.0005 | 0.046 | 0.067 | 0.69 | 1.00 | 0.139 | same budget as gates 3–4 |

Published `tab:gates` for comparison: 0.0084/0.257/0.301/0.85/1.00/0.745; 0.0086/0.255/0.304;
0.1232/0.000; 0.0045/0.116/0.172/0.67; 0.0004/0.049/0.071/0.68; 0.0017/0.000/0.000/0.51/0.68.
Shape and magnitudes reproduce. **The caption's `149` harmful cells does not**: the head-to-head
dumps have 146/145/139/144/137 harmful per seed (711 over 5 seeds).

**The parity fix (F3-10):** gates 1–2 receive *no* calibration at all; gates 3–4 receive
leave-one-corruption-out; KGA receives 431 leave-one-cell-out GBR fits. The docstring's claim that
every gate is leave-one-task-out "exactly like KGA" is false on both sides. Either add the two new
rows above (KGA at the gates' own budget) or say in the caption that gates 1–2 are untuned sign
rules and that KGA's calibration is leave-one-cell-out, not leave-one-task-out.
At equal budget the ATC gate has **lower** regret than the certificate (0.0041 vs 0.0059) but
**breaks the declared budget** (FA_u 0.116 > α = 0.10). That is the honest comparison and it is
still favourable to KGA — on the safety axis, not the regret axis.

**The radius's value (F3-11):**

| | FA_u | regret | coverage | harmful-cell adapt rate | false adapts over 5 seeds |
|---|---|---|---|---|---|
| KGA no radius | **0.038 < α = 0.10** | 0.000359 | 1.00 | **0.117** | **83** |
| KGA certificate | 0.000 | 0.001585 (**4.4×**) | 0.68 | **0.000** | **0** |

The radius-free variant **meets the declared budget** at 4.4× lower regret and full coverage. The
argument for the radius is therefore *not* the aggregate FA_u; it is the harmful-cell column:
11.7% of harmful cells adapted without it (83 false adapts) versus 0.0% (0 false adapts) with it.
**Make that the argument.**

### 7.3 Item 19 — seed-0 environment heterogeneity

**Script:** `recompute/07_env_heterogeneity.py` (43 run manifests scanned).

CIFAR-10-C stress grid — **3 distinct stacks across 5 seeds**:

| seed | git hash | Python | torch | numpy | finished |
|---|---|---|---|---|---|
| **0** | `4896181799ad` | **3.12.13** | **2.5.1** | **2.4.6** | **2026-07-02** |
| 1 | `6a237ed489c3` | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-11 |
| 2 | `6a237ed489c3` | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-12 |
| 3 | `6a237ed489c3` | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-12 |
| **4** | **`571c89f25989`** | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-12 |

ImageNet-C — **2 distinct stacks across seeds 1–4, and seed 0 comes from a third**:

| seed | git hash | Python | torch | numpy | finished |
|---|---|---|---|---|---|
| **0** (via `win_hunt_v5/imagenetc_aggr/`) | `87bf90aaadce` | **3.12.13** | **2.5.1** | **2.4.6** | 2026-07-09 |
| 1, 2 | `27a7e977f033` | 3.9.23 | 2.8.0 | 2.0.2 | 2026-07-15/16 |
| 3, 4 | `1adea4515b8c` | 3.9.23 | 2.8.0 | 2.0.2 | 2026-07-16 |

- `pooled_5seed/per_condition_imagenetc_sar_seed0.json` is **md5-identical**
  (`8b655a29360a23ca6fa9f5658f91d95a`) to `win_hunt_v5/imagenetc_aggr/per_condition_imagenetc_sar_seed0.json`.
- Seed 0's argv **omits `--severities 1 3 5` and `--max-images 4000`**, both present for seeds 1–4.
- **`pooled_5seed/` has no `result_manifest.json` at all.**
- **0 of 43 run manifests record a scikit-learn version.** `b_hat` comes from
  `GradientBoostingRegressor(subsample=0.8)`, so ε and every decision are sklearn-version-dependent.

All of this **confirms F4-6 and F4-14 exactly.**

### 7.4 Item 23 — per-seed spread behind every multi-seed mean

**PACS** (`PACS_MULTISEED_RESULTS.json`, 4 domains × 3 seeds = 12 cells of 18):

- Panel row 0.0431 / 0.0176 / 0.0446 — **reproduces**.
- KGA regret across the 12 domain-seed cells: **min 0.00529, median 0.03616, max 0.15344, sd 0.03895.**
- `art_painting` seed 1: **FA_u = 0.1111 > α** (2 of 18 cells), adapt rate 0.667.
- `art_painting` seed 2: coverage **0.0** — the certificate abstained on all 18 cells.
- Pooled `FA_u` = **2/216 = 0.00926**; Wilson 95% **[0.00254, 0.03313]**;
  Clopper–Pearson 95% **[0.00112, 0.03305]**; CP95 upper **0.03305**.
  (Integer counts back-derived as rate × 18 — exact, since every rate is a multiple of 1/18.
  `false_adapt_count_status: "not_retained"` in the artifact.)
- Adapt rate is 0 on **11 of 12** domain-seed cells; the only non-zero is art_painting seed 1
  (12 of its 18 cells). So PACS's **entire** adapt evidence is 12 ADAPT decisions from one
  domain-seed cell, 2 of them false: `FA_c = 2/12 = 0.1667`, **CP95 upper on FA_c = 0.4381**.
  By the item-5 rule PACS is *not* "guarantee untested" (12 ≥ 10) but the bound is 4.4× α.

**ImageNet-R** (`MULTISEED_ANALYSIS_RESULTS.json`, 10 backbones × 4 seeds × 12 cells):

- Panel row 0.0112 / 0.0064 / 0.0325 — **reproduces** (interpolated rule).
- KGA across backbones: **min 0.00000, median 0.01195, max 0.02260**.
- **KGA is worse than always-adapt on 7 of 10 backbones. 4 of 10 have a 0% harmful base rate.**

| backbone | KGA | always-adapt | always-freeze | ratio | harmful base rate |
|---|---|---|---|---|---|
| convnext_base | 0.00000 | 0.00000 | 0.06932 | tie | **0.0%** |
| convnext_tiny | 0.02073 | 0.00146 | 0.02208 | **14.2×** | 14.6% |
| efficientnet_b0 | 0.00000 | 0.05062 | 0.00000 | better | 100.0% |
| efficientnet_b3 | 0.01859 | **0.00000** | 0.05010 | worse by 0.0186 | **0.0%** |
| resnet101 | 0.01516 | 0.00052 | 0.02375 | **29.1×** | 8.3% |
| resnet152 | 0.00875 | **0.00000** | 0.04208 | worse by 0.0088 | **0.0%** |
| resnext101_32x8d | 0.00589 | **0.00000** | 0.05198 | worse by 0.0059 | **0.0%** |
| swin_b | 0.02260 | 0.00005 | 0.03750 | **434×** | 2.1% |
| swin_t | 0.00432 | 0.01057 | 0.00474 | better | 60.4% |
| vit_b_16 | 0.01599 | 0.00036 | 0.02370 | **43.9×** | 6.2% |

**CIFAR-10-C stress grid** per-seed spread:

| candidate | pooled KGA | per-seed KGA | seeds beating both | ε per seed | ε cv |
|---|---|---|---|---|---|
| Tent | 0.001626 | 0.001567 / 0.001631 / 0.001639 / 0.001440 / 0.001853 | **5/5** | 0.02154 / 0.019934 / 0.020939 / 0.020959 / 0.021424 | 0.0302 |
| EATA | 0.001313 | 0.001484 / 0.001328 / 0.001317 / 0.001472 / 0.000963 | **5/5** | 0.017151 / 0.016791 / 0.018064 / 0.017542 / 0.015024 | 0.0684 |
| SAR | 0.001549 | see §6 | 1/5 | 0.026788 / 0.012707 / 0.012713 / 0.013698 / 0.013083 | **0.3897** |

**Camelyon17 Table VIII** per-seed (interp in-pool, as published):

| candidate | seed | ε | KGA | adapt | freeze | ADAPT | FA_u | harmful frac |
|---|---|---|---|---|---|---|---|---|
| Tent | 0 | 0.1527 | 0.01085 | 0.15061 | 0.01085 | 0 | 0 | 0.778 |
| Tent | 1 | 0.2155 | 0.00781 | 0.10677 | 0.00781 | 0 | 0 | 0.889 |
| Tent | 2 | 0.3305 | 0.00217 | 0.15625 | 0.00217 | 0 | 0 | 0.778 |
| Tent | 3 | 0.3719 | 0.05946 | 0.13845 | 0.05946 | 0 | 0 | 0.444 |
| SAR | 0 | 0.0720 | 0.04731 | 0.00000 | 0.04731 | 1 | **0.1111** | 0.000 |
| SAR | 1 | 0.0898 | 0.06163 | 0.00000 | 0.06771 | 1 | 0 | 0.000 |
| SAR | 2 | 0.0910 | 0.03906 | 0.00043 | 0.03906 | 0 | 0 | 0.111 |
| SAR | 3 | 0.0915 | 0.01606 | 0.00043 | 0.10764 | 5 | 0 | 0.111 |

Camelyon17's panel row reports FA_u = 0 while `tab:multiseed`'s SAR row reports **0.11** — that
0.11 is seed 0's **1/9**. Clopper–Pearson 95% upper bound for 1/9 is **0.4291** (Wilson
[0.0198, 0.4348]), so 1/9 is well inside binomial noise; qualify the row rather than treating
0.11 as a budget violation.

---

## 8. Numbers that could NOT be recomputed, and why

### 8.1 iCloud placeholders (fix-queue item 9 — exact census)

**Script:** `recompute/13_placeholder_scan.py` → `out_placeholders.json`.
Scanned 1 889 text-extension files under `/home/claude/kb`:

| test | count |
|---|---|
| NUL-filled or zero-byte (the reliable test) | **145** |
| whitespace-only (the naive test) | **0** |
| unreadable by `OSError` | 0 |

By extension: **78 `.json`, 45 `.py`, 10 `.csv`, 9 `.md`, 3 `.sh`.**
The panel reported 142; the difference is extension coverage, and the qualitative point stands
exactly: **a whitespace scan returns 0, a NUL scan returns 145.** Any release guard must test for
NUL bytes.

Confirmed placeholders among the named artifacts: all five ablation JSONs
(`docs/research/kbound/experiments/kbound/results/ablation_{alpha,estimator,transfer,dropout,exactrank}.json`),
`cost_profile.json`, the entire Office-Home runner
(`experiments/kbound/officehome/run_officehome_kbound.py`, 17 202 B, zero readable),
`oh_analyze.py`, all ten edge checklists `docs/research/kbound/edge/artifacts_real/checklists/S{01..10}_checklist.csv`,
all four ImageNet-C `win_hunt_v5_imagenetc_ms/seed{1..4}/checkpoint.json`, and the whole
`experiments/kbound/vendored_from_elara/` tree.

**None of these blocked a number in this pack.** What blocked numbers is *absent* files:

### 8.2 Absent artifacts that block promoted numbers

| number | missing artifact | status |
|---|---|---|
| **Camelyon17 OOD promoted row 0.0000 / 0.0000 / 0.1381 (n=18)** | `docs/research/kbound/audits/integrity_2026-06-20/camelyon_reconciliation/` — directory does not exist | **BLOCKED-NEEDS-DATA.** The triple appears in **no** artifact on disk. Live Camelyon artifacts give `false_adapt` 0.0256 (n=54, `camelyon17_protocol_G_v1`) and 0.0329 (n=324, `camelyon17_richZ_F_v1`). Mark the row "not reproducible from release". |
| **Office-Home promoted regret 0.0157142857 (n=35)** | `officehome_full_targetval/result_target_val_361a1e8c.json`, `officehome_full_targettest/result_target_test_6605675d.json` | **BLOCKED-NEEDS-DATA** for the promoted value. Two other scorings *do* exist and disagree: `officehome_protocol_M_v2/protocol_result.json` gives 0.002198 (7.2× smaller, 22/35 adapts), and the recomputable `multiseed/officehome/extracted/` files (36 × 5) give KGA regret **0.000000** with 114/180 adapts and ε ≈ 0.0002–0.0007. |
| **iWildCam promoted regret 0.0041023691 (n=72)** | `iwildcam_full_test/result_e40faf29.json` | **BLOCKED-NEEDS-DATA** for the promoted value. `iwildcam_protocol_H_v2/protocol_result.json` gives 0.0036745149 with 1/72 adapts; the recomputable `multiseed/iwildcam/extracted/` files (72 × 2) give KGA regret **0.021174 ≡ always-freeze** with 0 adapts. |
| **CIFAR-10-C seed 0 per-condition cells (tent, eata, sar)** | `stress_grid_multiseed_v1/seed0/per_condition_cifar10c_*_seed0.json` | **PARTIAL.** Seed-0 rows are readable only from the stored `LOCKED_ANALYSIS_RESULTS.json` and cannot be independently recomputed. Seeds 1–4 recompute exactly. The promoted panel rows are unaffected because they come from `mixed_headtohead_v1/`, which **does** ship all 5 seeds. |
| **`tab:gates` as published (n=432, "149 harmful")** | `cifar10c_percell.json` — exists nowhere in the tree | **PARTIAL.** Regenerated from the head-to-head dumps (§7.2); magnitudes match, harmful count does not (137–146 per seed). |
| **PACS decisions under any radius** | `results/per_cell/pacs_*_percell.json` carry `Z, a0, aa, B` but **no `b_hat`, no `eps_conformal`, no decision**; seed 0 has no per-cell dump at all | **BLOCKED-NEEDS-DATA.** PACS cannot be re-scored under a LOO radius from the release. Only the rates in `PACS_MULTISEED_RESULTS.json` are available. |
| **RxRx1 promoted always-adapt regret 0.2531** | the raw record file behind `rxrx1_protocol_J_v1/analyze_F_results.json` (test seeds 5–9) | **PARTIAL.** `analyze_F_results.json` itself gives 0.2530598958; the recomputable `multiseed/rxrx1/extracted/` files (seeds 0–4) give **0.258724**. Two different seed sets — this is the "0.2531 vs 0.2587" discrepancy F4-15 flags. |

---

## 9. Script index

| script | produces | runtime |
|---|---|---|
| `recompute/kb_common.py` | shared radius/decision/scoring/CP/bootstrap primitives | — |
| `recompute/00_inventory.py` | `inventory.json` — 366 per-cell artifacts, 78 placeholders | 5 s |
| `recompute/01_imagenetc_perseed.py` | `out_imagenetc_perseed.json` — item 2, ImageNet-C half of item 4 | 2 s |
| `recompute/02_cifar_loo_radius.py` | `out_cifar_loo.json` — item 4, CIFAR half | 2 s |
| `recompute/03_imagenetc_bootstrap.py` | `out_imagenetc_boot.json` — item 3 | 4 s |
| `recompute/04_cifar_cluster.py` | `out_cifar_cluster.json` — item 17(a) | 30 s |
| `recompute/05_cifar_loco.py --all-seeds` | `out_cifar_loco.json` — item 17(b) | ~10 min |
| `recompute/06_decision_accounting.py` | `out_decision_accounting.json` — item 5, item 23 | 20 s |
| `recompute/07_env_heterogeneity.py` | `out_env.json` — item 19 | 2 s |
| `recompute/08_gates_and_radius_value.py --all-seeds` | `out_gates.json` — item 18 | ~12 min |
| `recompute/09_panel_variance.py` | `out_panel_variance.json` — items 6, 23 | 2 s |
| `recompute/10_identity_and_promoted_rows.py` | `out_identity_promoted.json` — item 5(a)(b) | 5 s |
| `recompute/11_build_numbers_pack.py` | `/home/claude/kb_fixes/NUMBERS_PACK.json` | 1 s |
| `recompute/12_latex_tables.py` | `latex_item{2,3,5}.tex` | 1 s |
| `recompute/13_placeholder_scan.py` | `out_placeholders.json` — item 9 census | 10 s |

All scripts read only from `/home/claude/kb` and write only under `/home/claude/kb_fixes/`.
