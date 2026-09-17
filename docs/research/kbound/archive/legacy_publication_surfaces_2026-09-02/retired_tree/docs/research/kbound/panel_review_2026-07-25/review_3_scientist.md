# Reviewer 3 — Empirical Scientist / Experimental Methodologist

Target: `docs/research/kbound/kbound_short.tex` (+ `kbound_short_appendix.tex`), the frozen IEEE
short paper, and the evidence tree it indexes (`docs/research/kbound/paper/generated/`,
`experiments/kbound/`, `research_lock/`).

All paths below are relative to `/home/claude/kb` unless absolute. Every number I quote was either
read out of a file on disk or recomputed by me from the raw per-condition JSONs; where I recomputed,
I say so and give the command semantics.

## Bottom line

This is one of the most heavily self-audited empirical projects I have reviewed, and the authors
deserve real credit for the ledger, the pre-registered protocols, the withdrawn-claim machinery, and
for promoting the *less* flattering of two candidate numbers on more than one track. But the
headline inferential claims do not survive contact with the raw artifacts. The ImageNet-C SAR
"beats-both" — one of only two mixed-regime wins in the paper — is significant only because the
paired bootstrap treats 135 cell–seed pairs drawn from **27 distinct conditions** as independent;
under the seed-averaged 27-condition bootstrap that the paper itself twice says it used
(`kbound_short.tex:797`, `kbound_short_appendix.tex:290`), the gap to always-adapt is
`[-0.0811, +0.0176]` and includes zero (my recomputation from the canonical per-seed files). The
abstract's "uniformly no-harm … it matches the better fixed policy" is contradicted by the paper's
own uniform panel on two of the shift families it enumerates (PACS 0.0431 vs always-adapt 0.0176;
ImageNet-R 0.0112 vs 0.0064). The quarantined CIFAR-10-C SAR aggregate is nonetheless reported in
the body with its beats-both CIs, and its significance comes entirely from the one seed the
quarantine says does not reproduce. Three of nine promoted tracks depend on raw record files that
are not in the repository at all. And the "no-harm" tracks where KGA is safest are the ones where it
adapts on 0–1 of 60–72 cells, so FA_u = 0 is definitional rather than evidential. Underneath all of
this sits ~1,387 recorded `beats_both` determinations across the tree, against a Holm family
declared post hoc as *three* comparisons.

**Verdict: reject in present form.** The safety framing is defensible and the CIFAR-10-C stress-grid
track is genuinely good evidence; the mixed-regime "beats-both" claims and the "uniform no-harm"
abstract are not supported at the stated confidence and must be either re-analysed at the correct
unit of analysis or withdrawn.

## What is done well

- **Pre-registration is real, not decorative.** `research_lock/WIN_HUNT_v3_PROTOCOL.yaml` is dated
  before scoring, declares the WIN/DEMO/TIE bar per arm, contains a `report-all-arms rule`, a
  `schema_guard` that exits rather than fabricate, and `prohibited_after_unblinding: any re-tuning`.
  `research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml` similarly fixes the analysis, and
  `_locked_analysis_script.py` carries "No post-hoc metrics" in its docstring and implements exactly
  what the protocol says.
- **The CIFAR-10-C stress grid is a genuine result.** I independently recomputed it from
  `experiments/kbound/results/mixed_headtohead_v1/per_condition_cifar10c_tent_primary_*_seed{0..4}.json`
  and reproduced `0.00157361 / 0.00792338 / 0.12409792` to 8 decimals. Across 2,160 cell–seed pairs
  the certificate makes 1,114 ADAPT decisions with **0** false adapts; the Clopper–Pearson 95% upper
  bound on the conditional false-adapt rate is 0.0027. That is a substantive, well-powered safety
  result and it is the strongest thing in the paper.
- **Honest negative reporting exists.** `kbound_short.tex:826-831` reports the ImageNet-C Tent and
  EATA rows as "no (≈ties freeze)" and "no (ties adapt)"; CIFAR-10.1 is reported as a failure
  (FA_u = 0.167, FA_c = 0.444); `claim_ledger.json` carries 5 `withdrawn` claims out of 30.
- **The exact-rank regeneration went the honest direction.** Moving ImageNet-C SAR from the
  interpolated quantile (KGA regret 0.0108, `paper/generated/uniform_verdicts.json`) to the exact
  split-conformal rank (0.0264) made KGA look *worse*, and they promoted the worse number.
- **Baseline calibration parity is thought about.** `docs/research/kbound/scripts/run_decision_baselines.py`
  gives every gate a leave-one-out-tuned threshold ("the baseline gets the SAME calibration
  information", `:109-111`) and adds a `best_single_hindsight` non-realizable upper envelope. This
  is better baseline hygiene than most TTA papers. (My complaint in F3-10 is that the *short paper*
  does not use it.)
- **β is not tuned.** I grepped the whole shipped library and every protocol YAML: β appears
  nowhere in the decision path (`kga/`, `src/scripts/kbound/`, `research_lock/*.yaml`). The rule
  really is `Δ̂ ± ε`, and the paper says so (`kbound_short.tex:41`). The most obvious fatal flaw —
  a per-dataset "declared" budget silently fitted on test — is **not** present.

---

## Findings

### [BLOCKER] F3-1 — ImageNet-C SAR "beats-both" is an artefact of bootstrapping correlated cells as independent; under the analysis the paper says it ran, the CI includes zero

**Location.** `docs/research/kbound/scripts/g8_exactrank_ci.py:14`;
`docs/research/kbound/kbound_short.tex:797-802`; `docs/research/kbound/kbound_short_appendix.tex:288-291`;
`docs/research/kbound/paper/generated/kbound_result_manifest.json` (`tracks/imagenetc_sar`).

**Evidence.** The manuscript says the pooling is over seed-averaged conditions:

> "Pooled over five seeds (**seed-averaged conditions, the same paired design as the CIFAR-10-C
> rows**) … The pooled 95% paired-bootstrap gap to always-freeze is $[-0.0088,-0.0027]$ and to
> always-adapt $[-0.052,-0.003]$; both intervals exclude zero." (`kbound_short.tex:797-802`)

and repeats it in the appendix: "The pooled seed-averaged bootstrap (the same design as the
CIFAR-10-C rows) excludes zero on both gaps" (`kbound_short_appendix.tex:290-291`).

The code that produced those intervals does not seed-average. `g8_exactrank_ci.py` concatenates all
five seeds into one vector and resamples it i.i.d.:

```python
def boot(g):
    idx=rng.integers(0,n,(5000,n)); ms=g[idx].mean(1); return ...   # n = 135
```

I ran the canonical files
(`experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed{0..4}.json`,
exact-rank ε per seed, `k=⌈(n+1)(1-α)⌉`) and reproduced the promoted point estimate exactly
(`0.0264 / 0.0529 / 0.0319`, FA_u = 0, 135 cells, 109 abstain) and the promoted intervals:

| bootstrap unit | KGA − always-adapt | KGA − always-freeze |
|---|---|---|
| 135 pooled cell–seed pairs (as coded, = manifest `[-0.0518,-0.0038]`) | `[-0.0518, -0.0037]` | `[-0.0086, -0.0026]` |
| **27 conditions, seed-averaged (as the paper describes)** | **`[-0.0811, +0.0176]`** | `[-0.0092, -0.0023]` |
| 5 seeds (cluster = seed) | `[-0.0308, -0.0204]` | `[-0.0127, +0.0000]` |

There are only 27 distinct conditions; each appears five times. The i.i.d.-135 bootstrap is a
bootstrap for an estimand nobody wants: it neither generalises to new conditions (cluster =
condition) nor treats seed randomness as the sampling source (cluster = seed). Under **both**
legitimate readings at least one gap CI touches or crosses zero, so `beats_both` fails.

**Why it matters.** This is one of only two mixed-regime "beats-both" tracks in the paper and the
only one on ImageNet scale. It is listed as `locked paired-bootstrap beats-both` in the manifest, as
"pooled CI-supported utility" in Table XV (`:910`), and in `uniform_verdicts.json` as one of the
three comparisons in the declared Holm family. The claim is unsupported at the stated confidence,
and the paper's *description* of the analysis is not the analysis that was run.

**Fix.** Re-run the ImageNet-C pooling as a seed-averaged 27-condition paired bootstrap (matching
the text and matching `_locked_analysis_script.py`'s `pooled = mean over seeds, then bootstrap
conditions`). Report the resulting interval. If it includes zero — it does for the adapt gap —
demote ImageNet-C SAR from "beats-both" to "beats always-freeze; adapt gap not CI-supported",
update the abstract, Table XV, the primary numeric table, and `uniform_verdicts.json`.

---

### [BLOCKER] F3-2 — The abstract's "uniformly no-harm across every natural distribution shift … it matches the better fixed policy" is contradicted by the paper's own uniform panel on two of the shift families it names

**Location.** `docs/research/kbound/kbound_short.tex:41` (abstract), `:86`, `:114`, `:1221`
(conclusion); contradicted by `:915` (PACS) and `:916` (ImageNet-R) in Table XV; definition at
`:120-122`.

**Evidence.** The paper defines the term: *"'No-harm' means matching the better fixed policy while
avoiding the worse"* (table note, `kbound_short.tex:120-121`). The abstract asserts:

> "Across every natural distribution shift we test---hospital, wildlife-camera, laboratory-batch,
> **domain, and rendition shifts**---KGA is uniformly *no-harm*: **it matches the better fixed
> policy**…"

Table XV, same paper:

| track | KGA | always-adapt | always-freeze |
|---|---|---|---|
| PACS (`:915`) | 0.0431 | **0.0176** | 0.0446 |
| ImageNet-R D (`:916`) | 0.0112 | **0.0064** | 0.0325 |

On PACS, KGA's mean regret is **2.4×** the better fixed policy; on ImageNet-R, **1.75×**. Both are
natural distribution shifts, and "rendition" in the abstract can only mean ImageNet-R. I confirmed
these are the artifact values (`experiments/kbound/results/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json`,
`experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json`) and
that the manifest carries the same numbers.

**Why it matters.** The paper's *primary* claim is the no-harm one ("Our primary evidence is a
safety guarantee, not an accuracy gain", abstract). The body defuses PACS and ImageNet-R by
relabelling them "null diagnostics", but the abstract and conclusion make an unqualified universal
statement over shift families that includes them. A reader who reads only the abstract is misled
about the paper's central claim, and the counter-evidence is in the paper.

**Fix.** Either (a) drop "domain, and rendition shifts" from the abstract's enumeration and state
explicitly "on the tracks we promote as locked; PACS and ImageNet-R are diagnostics on which KGA is
worse than always-adapt", or (b) weaken "matches the better fixed policy" to "never triggers the
catastrophic fixed-policy failure". Option (a) is the honest one.

---

### [BLOCKER] F3-3 — The quarantined CIFAR-10-C SAR beats-both is still reported in the body, and its significance is produced entirely by the one seed the quarantine says does not reproduce

**Location.** `docs/research/kbound/kbound_short.tex:637-644`;
`experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json`;
`docs/research/kbound/CIFAR10C_SAR_QUARANTINE.md:19`.

**Evidence.** The quarantine is unambiguous: *"Until then, **no aggregate, table row, or comparative
wording** from the archived SAR run supports a claim."* The manuscript nonetheless states:

> "The rebuild yields regret $0.0015/0.0112/0.1286$ … and false-adapt $0/2160$; **paired
> condition-bootstrap intervals exclude zero against both fixed policies**." (`:639-642`)

I read the per-seed decomposition out of `LOCKED_ANALYSIS_RESULTS.json` (`pstar_law.per_seed_cand`):

| SAR seed | harmful frac | KGA | always-adapt | always-freeze | beats-both |
|---|---|---|---|---|---|
| **0** | **0.528** | 0.00135 | **0.05470** | 0.08119 | **True** |
| 1 | 0.102 | 0.00134 | 0.00031 | 0.14006 | False |
| 2 | 0.088 | 0.00166 | 0.00031 | 0.14097 | False |
| 3 | 0.074 | 0.00193 | 0.00028 | 0.14030 | False |
| 4 | 0.100 | 0.00146 | 0.00034 | 0.14066 | False |

Seed 0's harmful base rate is 0.528 against 0.074–0.102 for seeds 1–4 — a 5× outlier — and its
conformal radius is 0.0268 vs ~0.0127 (`eps_cv = 0.390`, acknowledged at `:643-644`). On **4 of 5
seeds KGA's regret is 4–6× worse than always-adapt.** The pooled "CI excludes zero" against
always-adapt exists only because seed 0's always-adapt regret (0.0547) is ~180× the other seeds'.

**Why it matters.** The paper simultaneously withholds the SAR row from Table XV ("SAR withheld") and
prints its comparative verdict in prose. The hedge ("retained as a current-tree reconciliation
rather than evidence") does not neutralise a sentence that states both intervals exclude zero. And
because the outlier seed is *the seed the quarantine is about*, the reported reconciliation is not
independent evidence that the archived run was sound — it is the same anomaly re-expressed.

**Fix.** Delete the numeric SAR aggregate and the "intervals exclude zero" clause from `:639-642`.
If the authors want to keep a sentence, it should be: "SAR's five-seed aggregate is dominated by
seed 0, whose harmful base rate (0.53) is 5× the other seeds'; on seeds 1–4 KGA's regret exceeds
always-adapt's. No comparative verdict is drawn." That is also the honest reading of the same file.

---

### [MAJOR] F3-4 — The promoted CIFAR-10-C headline numbers do not come from the source the manifest declares, and they use the interpolated quantile that the ledger's own G8 action item said to drop

**Location.** `docs/research/kbound/paper/generated/kbound_result_manifest.json:20-35`;
`docs/research/kbound/SUBMISSION_LEDGER.md:83-89` vs `:130-131`.

**Evidence.** The manifest declares:

```
"cifar10c_tent": { "regret": [0.0015736109, 0.0079233799, 0.1240979162],
                   "source": "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json" }
```

That file actually contains `tent: kga 0.0016259256, adapt 0.0079756946, freeze 0.1239368049` —
different in the 4th decimal on all three. I traced the promoted values by exhaustive grep
(`grep -rn "0.0015736\|0.0079233\|0.1240979"`): they live in
`research_lock/WIN_HUNT_v3_ARM_F_result.json:81-83` and
`experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json:19-23`,
and I reproduced them exactly from the `mixed_headtohead_v1` per-condition files. So the manifest's
`source` field is wrong for the paper's single most-cited row.

Separately, I confirmed the rule: for seed 0, `np.quantile(|b̂−B|, 0.9) = 0.021128973237763728`
matches the stored `eps_conformal` to the last digit, whereas the exact rank `k=⌈433·0.9⌉=390` gives
`0.021655`. The headline uses the **interpolated** quantile. The ledger's G8 entry says:

> "ACTION: update panel numbers to exact-rank values; state FA_u/eps use the exact rank rule;
> **drop interpolated-quantile from headline path**." (`SUBMISSION_LEDGER.md:88-89`)

and then Phase 7 P2 (`:130-131`) reverts the panel back to the interpolated "canonical" values
`0.0079/0.1241/0.1314`. The ledger contradicts itself, and the action item is recorded as
`[RESOLVED = PASS]`.

**Why it matters.** The manifest is declared "the authoritative index for every promoted number …
and quantile convention" (`kbound_short_appendix.tex:272-274`). A wrong `source` on the flagship row
defeats the entire traceability apparatus, and a reviewer following the pointer lands on numbers
that differ from the paper.

**Fix.** Repoint `cifar10c_tent`/`cifar10c_eata` `source` to `mixed_headtohead_v1` (or regenerate
from `stress_grid_multiseed_v1` and change the paper). State one quantile rule per track and make
the ledger G8 entry consistent with what was actually shipped.

---

### [MAJOR] F3-5 — The 432 CIFAR-10-C "conditions" are not independent; the paired bootstrap over cells understates uncertainty by 2.4–3.9×, and 1 of the 6 corruption families reverses the sign of the adapt gap

**Location.** `experiments/kbound/results/stress_grid_multiseed_v1/_locked_analysis_script.py:60-68`;
`docs/research/kbound/kbound_short.tex:629-637`.

**Evidence.** I decomposed the condition strings
(`gaussian_noise|s1|small|iid|aggressive|r0`): the grid is 6 corruptions × 2 severities × 3 batch
sizes × 3 compositions × 2 aggressiveness × **2 repeats** = 432. The two repeats `r0`/`r1` are the
same design point. Their per-cell regret gaps correlate at **0.948** (KGA−adapt) and **0.999**
(KGA−freeze), so at least 216 of the 432 "observations" are near-duplicates. And there are only
**six** corruption families (`CIFAR_C_QUICK` in `docs/research/kbound/scripts/cifar_tent_mps_v2.py:138`),
not the standard 15.

Recomputing the paired bootstrap on the seed-averaged 432-vector, 20,000 replicates:

| resampling unit | KGA − always-adapt | KGA − always-freeze |
|---|---|---|
| 432 cells i.i.d. (as run) | `[-0.00784, -0.00492]` | `[-0.13718, -0.10820]` |
| 12 corruption × severity clusters | `[-0.01155, -0.00166]` | `[-0.21340, -0.04253]` |
| 6 corruption clusters | `[-0.00952, -0.00263]` | `[-0.17790, -0.06313]` |

CI width grows 2.4× (adapt gap) and 3.9× (freeze gap). Per-corruption adapt gaps:
`contrast −0.0083, defocus_blur −0.0105, fog −0.0103, gaussian_noise **+0.0019**,
jpeg_compression −0.0035, pixelate −0.0074`. **On gaussian_noise, KGA is worse than always-adapt.**

**Why it matters.** The claim being made is generalisation ("the regret gaps to *both* fixed policies
are positive with 95% bootstrap CIs excluding zero", `:629-631`). If the target of inference is new
corruption types, the correct unit is the corruption family and n is 6, not 432. The result still
survives at 6 clusters, but barely, and the 1-in-6 sign reversal is material heterogeneity that the
pooled number hides.

**Fix.** Report the cluster-robust (corruption-level) interval alongside the cell-level one, and
report the per-corruption breakdown. Say plainly that the grid spans 6 of the 15 CIFAR-10-C
corruptions and 2 of the 5 severities, and that `r0/r1` are replicates so the effective n is ≤216.

---

### [MAJOR] F3-6 — Three of the nine promoted tracks depend on raw record files that are not in the repository, and the promoted Camelyon17 numbers appear in no artifact on disk

**Location.** `docs/research/kbound/scripts/bootstrap_win_cis.py:37-49`;
`docs/research/kbound/paper/generated/kbound_result_manifest.json` (`camelyon17_ood.source`);
`docs/research/kbound/kbound_short.tex:911, 951`.

**Evidence.** The script that produces the promoted Office-Home / iWildCam / Camelyon17 numbers
loads four files. All four are absent:

```
MISSING experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json
MISSING experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json
MISSING experiments/kbound/results/iwildcam_full_test/result_e40faf29.json
MISSING experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json
```

These are small per-condition JSONs — exactly the class `EXTERNAL_STORAGE_POLICY.md` says is
*"Tracked in Git"* ("small schema-validated per-condition JSON artifacts"), and exactly the class
that *is* tracked for CIFAR-10-C and ImageNet-C. None of them is listed in `STORAGE_MANIFEST.json`
or `REPRO_INVENTORY.json` as an external dependency.

Worse, the promoted Camelyon17 row `0.0000/0.0000/0.1381 (n=18)` cites
`audits/integrity_2026-06-20/camelyon_reconciliation/` — a directory that does not exist
(`docs/research/kbound/audits/` contains only three `.md` files). `grep -rn "0.1381" --include=*.json .`
finds the value nowhere in any result artifact. The nearest live Camelyon artifacts are
`camelyon17_protocol_G_v1/analyze_F_results.json` (n=54, `false_adapt: 0.0256`) and
`camelyon17_richZ_F_v1/analyze_F_results.json` (n=324, `false_adapt: 0.0329`) — both with **nonzero**
false-adapt, against the promoted `FA_u = 0` on the n=18 slice.

**Why it matters.** Three promoted tracks, plus the constructed three-source mixture that is built
from the same three, cannot be recomputed by a reviewer. The Camelyon row in particular is an n=18
subset of a 324-record run, selected as "OOD test-only", on which FA_u drops from 0.033 to 0.000 —
that is precisely the kind of subsetting that requires the raw file to adjudicate, and the raw file
is gone.

**Fix.** Commit the four record JSONs (they are small) or register them in `STORAGE_MANIFEST.json`
with checksums and an acquisition procedure. Restore or repoint the `camelyon_reconciliation`
artifact. Until then those rows should be marked `not reproducible from release` in the evidence-tier
column, not `locked`.

---

### [MAJOR] F3-7 — The script that generated the promoted Office-Home / iWildCam / Camelyon numbers reports `reproduces_locked: false` for all three, contradicting its own docstring; the two Office-Home artifacts differ by 7×

**Location.** `docs/research/kbound/scripts/bootstrap_win_cis.py:8-9` and `:97-99`;
`research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json`;
`experiments/kbound/results/officehome_protocol_M_v2/protocol_result.json`.

**Evidence.** The docstring asserts:

> "The point estimate of every run **reproduces the locked `protocol_result.json` to 4 decimals**
> (sanity-checked below)."

The sanity check it refers to (`:97`) writes `reproduces_locked`. In the saved output, all three are
`false`:

```json
{"name":"OfficeHome", "reproduces_locked": false, "point": {"regret_kga": 0.015714…}}
{"name":"iWildCam",   "reproduces_locked": false, "point": {"regret_kga": 0.004102…}}
{"name":"Camelyon17", "reproduces_locked": false, "point": {"regret_kga": 0.002875…}}
```

The disagreements are not rounding. For Office-Home, same candidate (`sar_online_aggressive`), same
estimator/conformal (`gbr`/`global`), same n=35:

| artifact | regret_kga | adapt_rate |
|---|---|---|
| `officehome_protocol_M_v2/protocol_result.json` | 0.002198 | 0.629 |
| `officehome_holdout_sar_aggr_gbr_global_single/holdout_score.json` | 0.002198 | 0.629 |
| **promoted** (`KBOUND_WIN_BOOTSTRAP_CIS_oof.json`, → paper `:913`) | **0.015714** | — |

A 7× discrepancy on a promoted headline number. And Camelyon17 in this same file has
`regret_kga 0.002875 > regret_adapt 0.001320` — KGA *worse* than always-adapt — while the promoted
Camelyon row is `0.0000/0.0000` from a different, smaller slice.

The ledger's own note is telling: *"OfficeHome … is an OOF-lock DESIGN value … **Not raw-traceable
BY DESIGN**"* (`SUBMISSION_LEDGER.md:109-110`). A promoted number that is by design not traceable to
per-cell data is not a result; it is a summary of a summary.

**Why it matters.** The Office-Home "no-harm, tiny point edge" row is `0.0157` vs always-freeze
`0.0158` — an edge of `0.0001` with `ci95 = [0.0, 0.00033]` and `p_better = 0.645`, i.e. a coin flip.
The *other* artifact for the same track would have given a 6× larger and CI-excluding edge. Which
number is right is unresolvable from the repository, and the paper does not tell the reader that two
mutually inconsistent scorings exist.

**Fix.** Reconcile the two Office-Home scorings, state which is canonical and why, and fix the
docstring at `:8-9`. If they cannot be reconciled without the missing raw files (F3-6), the track
must be demoted from `locked`.

---

### [MAJOR] F3-8 — On the tracks where "no-harm" is claimed, KGA adapts on ~0 cells, so `FA_u = 0` is definitional; the panel never reports action composition, and the derived audit artifact has it null for every one of those tracks

**Location.** `docs/research/kbound/kbound_short.tex:568` (metric declaration) vs `:908-918`
(Table XV) and `:946-955`; `experiments/kbound/results/rxrx1_protocol_J_v1/analyze_F_results.json`;
`docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json`.

**Evidence.** `:568` declares "adapt rate, action composition, and decision coverage" as **primary**
decision metrics. Neither table reports them for any natural-shift track. From the artifacts:

| track | n_test | ADAPT decisions | FA_u | 95% CP upper bound on FA_c |
|---|---|---|---|---|
| RxRx1 J | 60 | **0** (`adapt_rate: 0.0`) | 0 | undefined |
| iWildCam H v2 (protocol) | 72 | **1** (`adapt_rate: 0.0139`) | 0 | 0.95 |
| iWildCam H v2 (promoted OOF) | 72 | **0** (regret_kga ≡ regret_freeze exactly) | 0 | undefined |
| Office-Home M v2 | 35 | 22 | 0 | 0.127 |
| ImageNet-C SAR (5 seeds) | 135 | 12 (109 ABSTAIN) | 0 | 0.221 |
| CIFAR-10-C Tent (5 seeds) | 2160 | 1114 | 0 | **0.0027** |

On RxRx1 the certificate froze all 60 cells; the manifest and paper report `0.0000/0.2531/0.0000`,
i.e. KGA ≡ always-freeze by construction. On iWildCam the promoted OOF numbers are
`regret_kga = regret_freeze = 0.004102369062102953` to 18 digits, with `kga_vs_freeze.ci95 = [0,0]`.
"FA_u = 0" on a policy that never adapts carries no information.

The appendix (`kbound_short_appendix.tex:274-276`) says `decision_metrics.json` "adds
adapt/freeze/abstain counts and rates, 95% Wilson intervals for action rates". I read the file: for
**21 of its 29 tracks** — every PACS domain, Office-Home, iWildCam, Camelyon17, all three RxRx1 arms,
all ten ImageNet-R backbones, and D33 — `actions.adapt/freeze/abstain` are `null` and
`false_adapt_unconditional.ci95_wilson` is `null`. The counts exist only for CIFAR-10-C, ImageNet-C
seed 0, and CIFAR-10.1.

Also note ImageNet-C SAR seed-0 alone: `FA_u = 0/27`, Wilson upper bound **0.1246 > α = 0.10**. The
observed zero is not statistically incompatible with FA_u exceeding the budget.

**Why it matters.** The paper's primary claim is a safety claim, and safety evidence scales with the
number of commitments made. Only CIFAR-10-C provides that (upper bound 0.0027). Everywhere else the
guarantee is untested because the certificate essentially never commits, and the reader is not told.

**Fix.** Add ADAPT / FREEZE / ABSTAIN counts and the Clopper–Pearson (or Wilson) upper bound on FA_c
to Table XV for every track. Mark tracks with <10 ADAPT decisions as "guarantee untested". Populate
the null fields in `decision_metrics.json` or remove the appendix sentence claiming they are there.

---

### [MAJOR] F3-9 — ~1,387 `beats_both` determinations across the tree, and a Holm family declared post hoc as the three that won

**Location.** `docs/research/kbound/paper/generated/uniform_verdicts.json` (`_meta.wave_holm_family`);
`experiments/kbound/results/stress_grid_multiseed_v1/_locked_analysis_script.py:71-79`;
`research_lock/WIN_HUNT_v{2,3,4,5}_*`.

**Evidence.** Counting over `experiments/kbound/**/*.json` + `research_lock/*.json` (761 files):

- `"beats_both"` appears **1,387** times (**326** `true`, 1,049 `false`);
- `"verdict_win"` **97** times (40 `true`, 57 `false`);
- 70 distinct result units contain a scored KGA-vs-fixed-policy comparison;
- 69 pre-registered protocol YAMLs in `research_lock/`;
- WIN_HUNT campaigns v2 (arms A, C), v3 (D, E, F, G), v4 (A, B, C, D, F × 2–3 datasets), v5.

Against that, the declared multiplicity correction is:

```json
"wave_holm_family": "3 beats-both candidates (CIFAR-10-C tent/eata vs adapt; ImageNet-C sar vs freeze);
                     all survive Holm at 0.05"
```

The family consists of exactly the three comparisons that succeeded. `_locked_analysis_script.py`
does apply Holm honestly over its own 6 comparisons, but the six raw p-values are all at the
bootstrap floor `1/(10^4+1) = 9.999e-5`, so Holm is doing nothing there; and that family excludes
every other arm in the project.

**Why it matters.** With 1,387 determinations recorded and a project-wide 23.5% observed `true` rate,
the family-wise picture for the 3–5 promoted wins cannot be assessed from what is reported. I am not
saying the CIFAR-10-C win is spurious — its effect is large and the FA evidence is strong — but the
*inferential statement attached to it* ("survives Holm at 0.05") is a statement about a family of
three that was chosen after the results were known. Directory names `win_hunt_v*`, `win_finder_v*`,
`win_loop_v1`, `hard_dataset_win_loop_v1` describe the search accurately.

**Fix.** Publish the full arm inventory (protocol → arms → verdicts) as an appendix table, define the
comparison family *prospectively* as all pre-registered beats-both bars across the campaign, and
report Holm or BH over that family. The pre-registration infrastructure to do this already exists;
it is simply not being used for the correction.

---

### [MAJOR] F3-10 — The short paper's decision-baseline table pits a 431-fold LOO-fitted gradient-boosted model against untuned zero-threshold sign rules; the tuned baseline suite in the repo (and in the long manuscript) tells a different story

**Location.** `docs/research/kbound/kbound_short.tex:690-728` (Table `tab:gates`);
`docs/research/kbound/scripts/gate_baseline_comparison.py:49-55`;
`docs/research/kbound/scripts/run_decision_baselines.py:110-130`;
`experiments/kbound/results/decision_baselines/decision_baselines.json`;
`docs/research/kbound/kbound.tex:1948-1958`.

**Evidence.** In the script that produces `tab:gates`, the two headline losers are fixed sign rules
with no fitting at all:

```python
def gate_confidence(...): return np.where(Z[:, POST_CONF] > Z[:, PRE_CONF], "ADAPT", "FREEZE")
def gate_entropy(...):    return np.where(Z[:, ENT_DROP] > 0, "ADAPT", "FREEZE")
```

while KGA gets `_kga_bhat`: 432 leave-one-out fits of a 250-tree, depth-2 `GradientBoostingRegressor`
on all 11 evidence features, trained on the **true** benefits of the other 431 cells, plus a
calibrated radius. The paper then concludes "the confidence/entropy gates false-adapt on ~74% of
harmful cells" (`:698-700`).

The repo contains the fair version. `run_decision_baselines.py` gives each gate an LOO-tuned
threshold — its own comment says "the baseline gets the SAME calibration information" (`:109-111`) —
and adds a hindsight envelope. Its saved output
(`experiments/kbound/results/decision_baselines/decision_baselines.json`, ImageNet-C, 36 cells):

| rule | Tent regret | EATA regret | SAR regret |
|---|---|---|---|
| `atc_conf_loo` | **0.00029** | **0.00015** | 0.01783 |
| `ent_progress_loo` | **0.00029** | **0.00015** | 0.04702 |
| `eata_filter_loo` | **0.00029** | **0.00015** | 0.03704 |
| `gbm_committal` (KGA estimator, ε=0) | **0.00029** | **0.00015** | **0.00000** |
| **KGA (certificate)** | 0.00422 | 0.00025 | 0.00860 |

KGA loses to every LOO-tuned baseline on Tent (14×) and EATA, and loses to its own radius-free
variant on SAR. The long manuscript reports this honestly (`kbound.tex:1948-1958`, rows
"ATC-style confidence (LOO-tuned)", "entropy-progress (LOO-tuned)", "best single statistic
(hindsight)"). I grepped the frozen short paper for `hindsight`, `LOO-tuned`, `decision_baselines`:
**zero hits**.

**Why it matters.** The short paper's claim is explicitly comparative: KGA "is better than *simple
label-free decision rules* under the same locked protocol" (`:686-688`). The version of that
comparison in which the baselines get the same calibration budget exists, was run, and is omitted
from the submitted paper.

**Fix.** Port the LOO-tuned rows and the hindsight envelope from `kbound.tex:1948-1958` into
`tab:gates`, or state in the caption that the gates are untuned sign rules and that tuned variants
are reported in the extended version with different conclusions.

---

### [MAJOR] F3-11 — At the declared operating point the conformal radius has no measured benefit: the radius-free variant already meets the α = 0.10 budget at 4× lower regret

**Location.** `docs/research/kbound/kbound_short.tex:700-728` (Table `tab:gates`), `:986`;
`docs/research/kbound/kbound.tex:1955`.

**Evidence.** From `tab:gates` itself:

| rule | regret | FA_u | FA_c |
|---|---|---|---|
| KGA (no radius) | **0.0004** | **0.049** | 0.071 |
| KGA (certificate) | 0.0017 | 0.000 | 0.000 |

α = 0.10 throughout the paper. `0.049 < 0.10`. The radius-free variant satisfies the declared budget
while achieving **4.25× lower regret** and full decision coverage (1.00 vs 0.68). The long
manuscript's version is starker: `benefit model, committal (ε=0)` gets `0.0000/0.0000/0.0010` with
SAR false-adapt 0.05 — again inside budget — against KGA's `0.0060/0.0047/0.0229`
(`kbound.tex:1955-1957`). The paper's own framing, "the radius is what turns a good benefit
*estimate* into a false-adapt *guarantee*" (`:701-703`), is a statement about guarantees, not
measurements; the parenthetical "(with no such guarantee)" is the only acknowledgement.

**Why it matters.** An operator whose specification is "FA_u ≤ 0.10" gains nothing measurable from
the certificate on the only track where the comparison is well powered, and pays 4× in regret and 32
percentage points of coverage. The empirical case for the paper's central mechanism rests entirely
on the theorem holding under assumptions the paper marks as unverified (`def:risk-align` is flagged
"an ASSUMPTION, not empirically established", `SUBMISSION_LEDGER.md:27`).

**Fix.** Report, at every α in the sweep, the FA_u of the radius-free variant next to KGA's, and
state the operating regime in which the radius pays for itself (presumably α ≪ 0.05). If no such
regime is exhibited on real data, say so in the limitations.

---

### [MAJOR] F3-12 — The calibrated radius does not transport across corruption families: holding out the corruption triples estimator error and quadruples ε

**Location.** `docs/research/kbound/scripts/cifar_tent_mps_v2.py:148-164`;
`docs/research/kbound/kbound_short.tex:307-315`.

**Evidence.** I re-ran `decide_kga` exactly as coded on the seed-0 CIFAR-10-C stress grid
(reproducing the stored `b_hat` to r = 0.999998 and ε to 3 decimals), then re-fitted the same
estimator under two stricter partitions:

| calibration scheme | residual MAE | R² | ε | adapt rate | FA_u | KGA regret |
|---|---|---|---|---|---|---|
| leave-one-**cell**-out (as shipped) | 0.01021 | 0.991 | **0.0214** | 52.3% | 0 | 0.00131 |
| leave-one-**twin-pair**-out | 0.01045 | 0.990 | 0.0217 | 51.9% | 0 | 0.00145 |
| leave-one-**corruption**-out | 0.03222 | 0.892 | **0.0922** | 41.7% | 0 | 0.00550 |

Dropping the `r0/r1` twin changes nothing (good — this rules out the most naive leakage). Holding
out the whole corruption family triples the benefit-estimation error and **quadruples the required
radius**, and quadruples KGA's regret. The deployed ε = 0.021 is calibrated on cells from the same
corruption family as the cell being decided.

**Why it matters.** The deployment story is "calibrate on a dev split, then face a new shift". The
paper is careful to disclaim that "calibration coverage transfers to arbitrary unseen domains"
(`:305`), but it does not quantify the gap, and every reported ε, adapt rate, and coverage number is
the same-family number. An operator sizing an abstention band for a genuinely novel corruption needs
ε ≈ 0.09, not 0.02.

**Fix.** Add a leave-one-corruption-out row to the ablation table. It is cheap (6 model fits, ~1 s in
my run) and it is the single most decision-relevant robustness number in the paper. To the authors'
credit, FA_u stays 0 and beats-both survives under it — this ablation would strengthen the paper.

---

### [MINOR] F3-13 — The Phase-6 leakage audit certifies the CIFAR-10-C radius as cross-fit by citing a line range that stops one line short of the line that is not cross-fit

**Location.** `docs/research/kbound/PHASE6_LEAKAGE_AUDIT.md` §(a), CIFAR-10-C row;
`docs/research/kbound/scripts/cifar_tent_mps_v2.py:162`.

**Evidence.** The audit table says: *"CIFAR-10-C stress | leave-one-cell-out | **the held-out cell
(jackknife)** | q0.9 of LOO residuals | `scripts/cifar_tent_mps_v2.py:143-156` | **PASS (cross-fit;
no cell's estimator saw it)**"*. The cited range `143-156` covers the LOO estimator loop. The radius
is computed at line **162**, outside the cited range:

```python
162:    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
163:    dec = np.where(Bhat - eps > 0, "ADAPT", ...)
```

The quantile is taken over **all N residuals including cell i's own**, and that ε then decides cell
i. There is no held-out cell for ε. The estimator is cross-fit; the radius is not.

To the manuscript's credit, `kbound_short.tex:307-312` and Table `:549-550` describe this accurately
("LOO jackknife $q_{0.9}$") and `:695-697` explicitly disclaims "not the exact distribution-free
validity of clean split conformal or jackknife+". The defect is in the audit document, which is
presented as clearing the pipeline.

**Why it matters.** The audit doc is the artifact a reviewer or a future maintainer will trust. Its
verdict line — "**VERDICT: PASS (clean). No live promoted track computes ε in-sample on the cells it
scores**" — is false as stated for the flagship track. The magnitude of the effect is small (1/432 of
the quantile mass) but the certification is wrong.

**Fix.** Correct the Phase-6 row to "jackknife radius; test residual included in the quantile;
empirical coverage only, matching `kbound_short.tex:695-697`", and amend the VERDICT line.

---

### [MINOR] F3-14 — The three-source "routing" beats-both is structurally guaranteed by construction and by per-dataset calibration

**Location.** `experiments/kbound/results/mixed_protocol_oof_v2/mixed_protocol_oof_v2_result.json`;
`docs/research/kbound/kbound_short.tex:924-928`, `:953`.

**Evidence.** The mixture pools Camelyon17 (36 cells, where always-adapt is optimal: regret
0.0000 vs freeze 0.1381), Office-Home (35) and iWildCam (72) (where always-freeze is optimal). Any
policy that identifies the dataset beats both *global* fixed policies automatically. And the
artifact's own note says: *"each dataset keeps its own dev-calibrated LOO conformal radius"* —
so the decision rule is handed the dataset identity through its calibration.

**Why it matters.** The paper labels the row "constructed" (`:924-926`, `:953`), which is good, but
does not say that per-dataset calibration is the mechanism, nor that the pooled win is arithmetically
implied by mixing datasets with opposite optimal actions. As presented it reads as evidence of
routing capability; it is evidence that the experimenter knew which dataset each condition came from.

**Fix.** State the tautology explicitly, and add the informative version: a single pooled radius
across all 143 conditions (no dataset identity). If that still beats both, it is a real routing
result. `research_lock/WIN_HUNT_v3_PROTOCOL.yaml` arm E already does something close to this
("One universal gate … single eps = 0.096"); promote arm E instead.

---

### [MINOR] F3-15 — PACS: one of twelve domain–seed cells exceeds the α budget, the pooled false-adapt count was not retained, and the reported statistic is a mean over cells with no interval

**Location.** `experiments/kbound/results/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json`
(`per_domain.art_painting`); `docs/research/kbound/kbound_short.tex:915`;
`docs/research/kbound/SUBMISSION_LEDGER.md:105-108`.

**Evidence.** `art_painting` seed 1: `FA_u.per_seed = [0.0, 0.1111, 0.0]` and
`FA_c.per_seed = [0.0, 0.1667, 0.0]`. `0.1111 > α = 0.10`. Seed 2 on the same domain has
`coverage: 0.0` — the certificate abstained on all 18 cells. The panel reports only the mean across
12 domain–seed cells, `0.0093`, and the ledger records `false_adapt_count_status: "not_retained"`,
so no integer count and no Wilson interval can be reconstructed.

**Why it matters.** A mean of 12 rates, one of which is over budget, reported without an interval or
a denominator, is not an adequate summary of a rate that the paper's central theorem bounds.
(Statistically, 1/12 over budget at α = 0.10 is unremarkable — the problem is that the reader cannot
see it and cannot compute a bound.)

**Fix.** Re-derive and retain the pooled action/FA counts (the per-cell files under
`experiments/kbound/results/per_cell/` still exist), report `k/n` with a Wilson interval, and
disclose the one over-budget cell.

---

### [MINOR] F3-16 — ImageNet-R's mean-across-backbone regret hides a 14× reversal on one backbone; 20 comparisons, 0 significant, reported as one row

**Location.** `experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json`;
`docs/research/kbound/kbound_short.tex:916`.

**Evidence.** The panel row is "mean-across-backbone regret 0.0112/0.0064/0.0325 … 0/10 CI
beats-both". Per backbone, e.g. `convnext_tiny`: `kga_mean_regret 0.02073` vs
`adapt_mean_regret 0.00146` — KGA is **14×** worse than always-adapt; `convnext_base`: all three
harmful base rates are 0.0, so the track is degenerate there. Ten backbones × 2 fixed policies = 20
comparisons on this track alone, none significant.

**Why it matters.** A mean over backbones with wildly different harmful base rates (0% to 17%) is not
an interpretable summary, and it is the number that enters the "uniform panel" the abstract
generalises over (see F3-2).

**Fix.** Report the per-backbone min/median/max and the harmful base rate per backbone, or drop the
mean and report the range.

---

### [NIT] F3-17 — Canonical analysis scripts hard-code the author's machine paths, contradicting the release policy

**Location.** `docs/research/kbound/scripts/g8_canonical_pooling.py:2`;
`docs/research/kbound/scripts/g8_exactrank_ci.py:2`;
`docs/research/kbound/EXTERNAL_STORAGE_POLICY.md`.

**Evidence.**
```python
R=os.path.expanduser("~/Documents/AutoML_Flagship_V8/experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed")
```
against the policy statement *"Release guards reject unapproved oversized files, accidental external
data, and **absolute machine-local paths**."* Related: `VERDICT_camelyon_sourcecal.json` records
`"manifest": "/Volumes/T9/uav/AutoML_Flagship_V8/…"`. These are the two scripts that produce the
ImageNet-C headline; neither runs out of the box from a clean checkout.

**Fix.** Parameterise with `KBOUND_RESULTS_ROOT` (the pattern already used by
`_locked_analysis_script.py:10`) and add these files to the release guard's scan.

---

## What I checked and could NOT fault

- **β is never fitted.** I grepped `kga/`, `src/scripts/kbound/`, and all 69 `research_lock/*.yaml`
  for `beta`. The only hits are an unrelated regression coefficient in
  `kbound_full_experiments.py:155` and a power-analysis β in
  `STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml:34`. The deployed rule is `Δ̂ ± ε` everywhere, exactly as
  the abstract says. The obvious fatal flaw — a "declared" per-dataset budget quietly fitted on test
  — is not present.
- **α is fixed at 0.10 everywhere.** `certificate.py`, `analyze_F.py`, `cifar_tent_mps_v2.py:ALPHA`,
  and every protocol JSON. Not selected on test.
- **CIFAR-10-C headline point estimates reproduce exactly.** I rebuilt
  `0.00157361/0.00792338/0.12409792` from the raw per-condition files with an independent script.
  POEM 0.00880463 and AETTA 0.00732986 also reproduce.
- **ImageNet-C SAR point estimates reproduce exactly.** My independent implementation of the exact
  split-conformal rank (`k = min(n, ⌈(n+1)(1−α)⌉)`, per-seed) gives `0.0264/0.0529/0.0319`,
  FA_u = 0, 135 cells, 109 abstain — matching the manifest to 4 decimals. The *point estimates* are
  sound; only the interval is wrong (F3-1).
- **ε is fitted per seed, not once across pooled cells,** on ImageNet-C. `g8_canonical_pooling.py:9-13`
  computes `eps=cexact(rho)` inside the per-file loop. The Phase-6 audit's claim here is correct, and
  I verified that a single pooled ε would give materially different numbers.
- **Twin-replicate leakage is not a real effect.** Despite `r0`/`r1` correlating at 0.95–0.999, my
  leave-one-twin-pair-out refit changes KGA regret from 0.00131 to 0.00145 and ε from 0.0214 to
  0.0217. The estimator is not memorising its twin.
- **The `beats_both` predicate is gated on the false-adapt budget**, not regret alone
  (`cifar_tent_mps_v2.py:200-206`), with an explicit comment recording that the ungated version
  over-counted wins. That is a fix I would otherwise have flagged, already made.
- **Holm is correctly implemented** in `_locked_analysis_script.py:70-79` (step-down, running max,
  clipped at 1). My objection is to the family, not the procedure.
- **The KB-CLAIM-022 Camelyon withdrawal is real.** `claim_ledger.json` carries
  `status="withdrawn"`, `calibration_method="in_sample_radius"`, `test_split="pooled id_val (invalid)"`,
  and `grep -rn "beats both Camelyon"` over the manuscripts returns nothing.
- **The 5-seed / single-run distinction was actually corrected.** Phase 7 P1 changed iWildCam and
  RxRx1 tiers to "single-run" at `kbound_short.tex:912` and `:914`; I verified the current text says
  "single-run", not "5 seeds".
- **Effect-size honesty on Office-Home.** `:913` says "no-harm, **tiny point edge**", which is a fair
  description of a 0.0001 gap. The paper does not oversell it (though it also does not give the
  `p_better = 0.645`).

## Open questions for the author

1. **F3-1 is the decision point.** Will you re-run the ImageNet-C pooling as the seed-averaged
   27-condition bootstrap the paper describes, and report the result whatever it is? If the adapt-gap
   CI includes zero (it does in my recomputation), what is the revised claim for that track?
2. What is the correct unit of analysis you are claiming to generalise over on the stress grids —
   grid cells, corruption families, or seeds? The paper's inferential statements only make sense
   under one of these and the three give materially different intervals (F3-5).
3. Why does `bootstrap_win_cis.py` report `reproduces_locked: false` on all three natural-shift
   tracks, and which of the two Office-Home regret values (0.0022 vs 0.0157) is the protocol's
   intended output (F3-7)?
4. Can the four missing record JSONs be released (F3-6)? Specifically, does an artifact exist
   anywhere that contains the Camelyon17 `0.0000/0.0000/0.1381, n=18` row?
5. Seed 0 of the stress grid is the maximum-harmful seed for *all three* adapters (Tent 0.345,
   EATA 0.271, SAR 0.528) and its per-condition files are the only ones missing from
   `stress_grid_multiseed_v1/seed0/`. If seed 0 is untrustworthy enough to quarantine SAR, why does
   it remain in the promoted Tent and EATA aggregates (F3-3)?
6. What is the operating regime in which the conformal radius measurably beats the radius-free
   variant on real data (F3-11)? An α-sweep showing FA_u of both rules would settle it.
7. Would you accept a leave-one-corruption-out ablation as the headline transportability number
   (F3-12)? It is cheap, it survives, and it would answer the strongest methodological objection to
   the calibration design.
8. How many pre-registered beats-both bars were evaluated in total across WIN_HUNT v2–v5 and the
   protocol registry, and what does Holm over that family look like (F3-9)?
