# Reviewer 4 — Research Integrity & Reproducibility Auditor

## Bottom line

This is one of the most self-audited projects I have reviewed: 67 of 69 evidence-seal SHA-256 hashes
verify byte-for-byte, PACS and ImageNet-R panel rows recompute to the digit from committed raw cells,
the claim ledger carries explicit withdrawals, and the authors' own `EVIDENCE_MATRIX.md` names most of
the problems below before I did. That makes the failures more serious, not less, because the machinery
that was supposed to catch them ran and passed. Working mechanically from the frozen tex to artifacts, I
found three defects that touch headline claims: (i) the quarantined CIFAR-10-C SAR arm is reported in the
paper body with a favourable comparative claim, and its numbers are reproducible **only** by averaging
four raw seeds with the very rounded seed-0 summary the quarantine declares non-reproducing — without
that outlier seed the comparison reverses sign; (ii) the ImageNet-C SAR headline (0.0264/0.0529/0.0319,
FA_u = 0) is produced by an ε computed from `|b_hat − B|` over the **same 27 cells it scores**, i.e. the
radius is a function of the test labels — I reproduced the number to seven decimals under that rule and
got a different number (0.0289, FA_u = 1/135) under a genuine leave-one-out rule, directly contradicting
`PHASE6_LEAKAGE_AUDIT.md`'s "PASS (clean)"; (iii) the Camelyon17 multi-seed table is built from raw
decisions whose ε is an in-pool quantile of the true benefits — the identical `in_sample_radius` defect
for which KB-CLAIM-022 was withdrawn. On top of that the "frozen" tex was edited after the ledger froze
it, 139 text artifacts in the release are dataless placeholders (including every ablation JSON and the
cost table's source), and the two documented one-command reproductions both fail on a clean checkout.

**Verdict: major revision required before submission — the empirical section is not currently
reproducible by an outsider, and at least two headline numbers rest on calibration that uses test labels.**

---

## What is done well

- **The evidence seal actually works.** I recomputed every SHA-256 in
  `experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json`: 67 hashes match exactly, 0 mismatches,
  2 missing files. That is far better than the norm.
- **Two panel rows reproduce to the digit from raw cells.** PACS (`0.0431/0.0176/0.0446`,
  mean FA_u `0.0092593`) and ImageNet-R (`0.0112/0.0064/0.0325`, FA_u `1/480`) both recompute exactly
  from the committed per-condition artifacts. So does CIFAR-10.1 (`0.0021/0.0190/0.0017`, FA_c `0.4444`)
  and the three-source mixture (`0.0059117/0.0632323/0.0342043`, n = 143).
- **Withdrawals are real, not cosmetic.** `claim_ledger.json` KB-CLAIM-022 records
  `calibration_method: "in_sample_radius"`, `test_split: "pooled id_val (invalid)"`, and
  `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml` documents the pooling artifact in full,
  including the exact per-domain harm profile that manufactured the win. Very few papers do this.
- **`EVIDENCE_MATRIX.md` is unusually honest**, including "Office-Home M v2 | promoted 0.0157 NOT FOUND
  in any raw artifact" and "CIFAR-10-C SAR | seed0 non-repro | ... 180x outlier".
- **The manifest declares its own quantile conventions** (`quantile_provenance` in
  `kbound_result_manifest.json`), and the paper's Reproducibility section explicitly scopes what Lean does
  and does not verify. Both are rare and creditable.
- **Deterministic, seeded validators** with persisted artifacts for the theory suite
  (`experiments/kbound/theory_validation/results_thm*.json` all present).

---

## Findings

### [BLOCKER] F4-1 — The quarantined CIFAR-10-C SAR arm is reported in the paper, and its number exists only because of the seed-0 summary the quarantine declared non-reproducible

**Location:** `docs/research/kbound/kbound_short.tex:637-642`; `docs/research/kbound/CIFAR10C_SAR_QUARANTINE.md:19`;
`experiments/kbound/results/stress_grid_multiseed_v1/`

**Evidence.** The paper states:

> `kbound_short.tex:637` "The previously mismatched SAR aggregate was rebuilt from **all five saved
> per-condition seed files** using the locked Protocol~A analysis. The rebuild yields regret
> $0.0015/0.0112/0.1286$ ... **paired condition-bootstrap intervals exclude zero against both fixed
> policies.**"

Three things are wrong.

1. **There are not five per-condition seed files.** `find . -name 'per_condition_cifar10c_sar_seed*'`
   returns seeds 1–4 only inside `stress_grid_multiseed_v1/`; `seed0/` contains
   `decisive_tta_results.json`, `decisive_tta_table.md`, `result_manifest.json` and nothing else.
2. **The quoted triple is reproduced only by mixing raw cells with the rounded seed-0 summary.** I
   recomputed per-seed regrets from the four raw files and appended seed 0 from
   `stress_grid_multiseed_v1/seed0/decisive_tta_results.json`
   (`benchmarks.cifar10c.methods.sar.metrics.regret_vs_oracle`):

   | seed | KGA | always-adapt | always-freeze |
   |---|---|---|---|
   | 0 (aggregate summary) | 0.0013507 | **0.0547049** | **0.0811921** |
   | 1 (raw) | 0.0013426 | 0.0003067 | 0.1400556 |
   | 2 (raw) | 0.0016644 | 0.0003102 | 0.1409711 |
   | 3 (raw) | 0.0019282 | 0.0002778 | 0.1403044 |
   | 4 (raw) | 0.0014606 | 0.0003449 | 0.1406551 |
   | **5-seed mean** | **0.0015493** | **0.0111889** | **0.1286356** |

   `0.0015493/0.0111889/0.1286356` → the paper's `0.0015/0.0112/0.1286`, and it matches
   `LOCKED_ANALYSIS_RESULTS.json` `candidates.sar` exactly. This is precisely what quarantine
   reinstatement gate 4 forbids: *"derive action counts, false-adapt events, intervals, and regret from
   **raw decisions rather than rounded summaries**"* (`CIFAR10C_SAR_QUARANTINE.md:15`).
3. **Drop the outlier seed and the claim reverses.** The 4-seed always-adapt regret is **0.00031**, which
   is *lower* than KGA's 0.00160 — KGA loses to always-adapt. The project's own locked findings say so:
   `stress_grid_multiseed_v1/LOCKED_ANALYSIS_FINDINGS.md` records
   `| sar | always-adapt | 0.00152 | 0.00031 | **+0.00121** | [+0.00094, +0.00148] | ... | **NO (tie)** |`.
   So "paired condition-bootstrap intervals exclude zero against both fixed policies" is true only in the
   sense that one of them excludes zero *in the unfavourable direction* — and only if the disputed seed 0
   is included. Seed 0's SAR `eps_conformal` is 0.026788 against 0.0124–0.0137 for seeds 1–4, i.e. the
   documented non-reproduction.

**Why it matters.** The quarantine promises "no aggregate, table row, or comparative wording from the
archived SAR run supports a claim." The paper prints the aggregate and the comparative wording. A reader
who checks the artifacts finds that the favourable direction of the comparison is an artifact of the
single seed the authors themselves disowned.

**Fix.** Delete `kbound_short.tex:637-642` entirely, or replace it with the four-seed reconciliation and
state plainly that KGA ties/loses to always-adapt on SAR (as `LOCKED_ANALYSIS_FINDINGS.md` already says).
Do not reintroduce SAR until gate 2 ("reproduce seed 0 from clean raw outputs") is met.

---

### [BLOCKER] F4-2 — The ImageNet-C SAR headline ε is computed from the true benefits of the cells it scores; the leakage audit's "PASS (clean)" is wrong

**Location:** `docs/research/kbound/scripts/g8_canonical_pooling.py:11`;
`docs/research/kbound/scripts/g8_exactrank_regen.py:29`;
`docs/research/kbound/PHASE6_LEAKAGE_AUDIT.md:5,11,16,37`; `kbound_short.tex:798-802`

**Evidence.** The canonical pooling script is four lines of arithmetic:

```python
# g8_canonical_pooling.py:11
r=load(f); B=np.array([x['B'] for x in r]); bh=np.array([x.get('b_hat') for x in r])
rho=np.abs(bh-B); eps= cexact(rho) if use_exact else float(np.quantile(rho,1-A))
dec=['ADAPT' if b-eps>0 else ('FREEZE' if b+eps<0 else 'ABSTAIN') for b in bh]
```

`rho` is the residual vector over **all 27 cells of the seed**, and `eps` is its k-th order statistic.
Every cell's decision therefore depends on that cell's own true benefit `B_i` through `eps`. I ran both
variants over `win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed{0..4}.json`:

| ε rule | abstain | KGA regret | FA_u |
|---|---|---|---|
| in-sample (as shipped) | 109 | **0.0264222** | **0** |
| leave-one-cell-out | 107 | 0.0288926 | 1/135 |

The in-sample row matches `kbound_result_manifest.json` `imagenetc_sar`
(`regret [0.026422222, 0.0529333334, 0.0318944445]`, `abstain_count: 109`, `false_adapt: 0.0`) to seven
decimals; the LOO row does not. `PHASE6_LEAKAGE_AUDIT.md:11` claims "the true test benefit `B` **never
enters the ε computation or the decision**, only the post-hoc score", and line 16 tabulates ImageNet-C SAR
as "per-seed leave-one-cell-out ... exact rank ε per seed on that seed's OOF residuals |
`scripts/g8_canonical_pooling.py:6-17` | **PASS**". Line 37's justification — "ε(seed0)=0.084 is
substantial (not ~0), consistent with genuine leave-one-cell-out residuals" — is an inference from
magnitude, not a check of the code. The same pattern is in `g8_exactrank_regen.py:29`
(`rho=np.abs(bh-B); ... ee=cexact(rho)`), the script that produced `G8_EXACTRANK_REGEN.md`'s "G8 = PASS".

Note the estimator `b_hat` *is* out-of-fold (`cifar_tent_mps_v2.py:151-158` refits per held-out cell), so
this is not total leakage — but the conformal radius is not split-calibrated, which is exactly the object
the certificate's FA_u ≤ α guarantee is attached to.

**Why it matters.** The paper's one CI-supported beats-both on a real benchmark
(`kbound_short.tex:799` "beating both trivial policies (regret 0.0264 vs 0.0529/0.0319) at FA_u=0.000,
using the exact split-conformal radius") is produced by a radius that is not split-conformal and that
consumes the labels the method claims never to see. Under the honest LOO rule the reported FA_u = 0.000
becomes 1/135 and the margin to always-freeze shrinks from 0.0055 to 0.0030.

**Fix.** Recompute the ImageNet-C SAR row with a genuine split (or leave-one-cell-out) calibration and
report the resulting numbers, including the non-zero FA_u. Correct
`PHASE6_LEAKAGE_AUDIT.md` §(a) and §(b) — the audit's verdict line is currently false. Add a unit test
that asserts the scored index is excluded from the residual pool.

---

### [BLOCKER] F4-3 — Table VIII (Camelyon17 multi-seed) is built from decisions carrying the exact `in_sample_radius` defect that got KB-CLAIM-022 withdrawn, and its own named artifact says the opposite

**Location:** `kbound_short.tex:868-892`; `docs/research/kbound/scripts/run_wilds_camelyon17.py:57`;
`experiments/kbound/results/wilds_kbound/`; `experiments/kbound/results/multiseed/multiseed_camelyon17_*.json`;
`PHASE6_LEAKAGE_AUDIT.md:57`

**Evidence.** I traced the table row by row. The paper's values

```
889: Tent & $0.020{\pm}0.023$ & $0.138$ & $0.020$ & $0.00$ & stable no-harm
890: EATA & $0.039{\pm}0.025$ & $0.042$ & $0.042$ & $0.00$ & inconclusive
891: SAR  & $0.041{\pm}0.017$ & $0.000$ & $0.065$ & $0.11$ & over-freezes
```

reproduce exactly (0.0201±0.0230 / 0.1380 / 0.0201 / FA 0.0; 0.0393±0.0252 / 0.0417 / 0.0424;
0.0410±0.0165 / 0.0002 / 0.0654 / FA 0.1111) from
`experiments/kbound/results/wilds_kbound/per_condition_camelyon17_*_seed{0..3}.json` (9 cells/seed).
Three problems:

1. **The decisions in those files use an in-pool radius.**
   `run_wilds_camelyon17.py:57`: `eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))` — quantile over
   all N records including the one being decided, using true `B`. This is the calibration method the
   ledger records as the *reason for withdrawal* of KB-CLAIM-022
   (`claim_ledger.json`: `"calibration_method": "in_sample_radius"`). Realized ε per seed here is
   0.153 / 0.215 / 0.330 / 0.372 — enormous, which is exactly why the SAR row "over-freezes".
2. **Phase 6 explicitly warned against promoting this file and the warning was not applied.**
   `PHASE6_LEAKAGE_AUDIT.md:57`: *"`run_wilds_camelyon17.py:45-59` `decide_kga` computes ε in-pool (LOO
   over all seeds) for the **raw** run artifact ...; that raw file is an *input* to `analyze_F`
   re-scoring, **never a promoted number**."* Table VIII is a promoted table computed from precisely
   those raw `kga_decision` fields.
3. **The artifact actually named "multiseed camelyon17" disagrees and even ships a contradictory LaTeX
   row.** `experiments/kbound/results/multiseed/multiseed_camelyon17_tent.json` (36 cells/seed) gives
   `regret_kga [0.0257, 0.0285], regret_adapt 0.0124, regret_freeze 0.0812, "verdict": "unstable/other"`
   and a field
   `"latex_row": "camelyon17 (tent) & 4 & 0.0257$\\pm$0.0285 & 0.0124 & 0.0812 & 0.028 & unstable/other"`.
   The paper's Tent row says *stable no-harm*. The `eata` and `sar` entries in that file are likewise
   labelled *stable no-harm*, the reverse of the paper's *inconclusive* / *over-freezes*.
   `wilds_kbound/` contains no `result_manifest.json`, so the promoted source has no git hash, date,
   or library provenance at all, and appears in no manifest, seal, or ledger entry.

**Why it matters.** Table VIII is the paper's only multi-seed natural-shift stability evidence and is used
to argue candidate-dependence ("with the aggressive helpful-dominated SAR arm KGA over-freezes",
`:872-875`). That conclusion is an artifact of an over-wide in-sample radius, from an unregistered
directory, contradicted by the file that carries the table's own name.

**Fix.** Re-score Camelyon17 multi-seed through `analyze_F.py`'s out-of-fold path, register the resulting
directory in `kbound_result_manifest.json` and `LOCK_SEAL.json`, and reconcile against
`results/multiseed/multiseed_camelyon17_*.json` (or delete the stale copy). Until then Table VIII should
be withdrawn.

---

### [MAJOR] F4-4 — The "frozen" submission kept moving after the freeze, and the pinned PDF hash cannot correspond to the tex on disk

**Location:** `docs/research/kbound/SUBMISSION_LEDGER.md:5-11`; `docs/research/kbound/EDIT_NOTES_2026-07-23.md`

**Evidence.** The ledger pins the freeze:

> `SUBMISSION_LEDGER.md:7-8` — "Git commit (HEAD at freeze): `ff9be6b2a90482394fdb518226d8e0efde2c9c7b`
> ... PDF sha256: `5b01e5e7da41edae5a574c09fb8d5fa6b0cb4cc8d5853ff814441484b755d00a`"

File mtimes: `SUBMISSION_LEDGER.md` 2026-07-22 06:03:20, `kbound_short.tex` **2026-07-23 20:46:51**,
`paper/references_kbound_expanded.tex` 2026-07-23 20:46:52. `EDIT_NOTES_2026-07-23.md` documents 12 edits
to the frozen file, including two that are not purely cosmetic: item 8, *"**K-Bound → KGA row labels** in
Tables ... 5 policy rows renamed"*, and item 9, *"**Two citations added**"* with new bibitems. Both change
the compiled output, so the pinned PDF sha256 is stale by construction. There is no `.git` directory in
the release, so the pinned commit hash is unverifiable by any reader. The same notes concede
"Page-count drift ... 24 pp (your Mac build: 23 pp)", against the ledger's "PDF pages: 23".

**Why it matters.** The ledger's entire authority rests on being a freeze record. A freeze that is edited
the next day, with an unverifiable commit hash and a stale PDF hash, provides no integrity guarantee.

**Fix.** Re-freeze: rebuild, recompute the PDF sha256, record the new commit, and ship a `.git` bundle or a
Software Heritage / Zenodo snapshot so the hash is checkable.

---

### [MAJOR] F4-5 — The manifest's `source` for both CIFAR headline rows names a file that does not contain those numbers

**Location:** `docs/research/kbound/paper/generated/kbound_result_manifest.json` (`cifar10c_tent`,
`cifar10c_eata`); `SUBMISSION_LEDGER.md:130-131`

**Evidence.** The manifest declares

```json
"cifar10c_tent": {"regret": [0.0015736109, 0.0079233799, 0.1240979162],
  "source": "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json"}
```

but that file contains `kga_mean_regret 0.0016259256, adapt_mean_regret 0.0079756946,
freeze_mean_regret 0.1239368049`. The manifest's triple matches a *different* artifact byte-for-byte:
`experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json`
→ `policy_mean_regret {always_adapt: 0.007923379871580337, always_freeze: 0.1240979162355264,
kga: 0.001573610885275735}`. Same for EATA
(`HEADTOHEAD_RESULTS_cifar10c_eata_secondary.json`: `0.0012675925/0.0032682874/0.1313789343`).
I confirmed by recomputing seed-by-seed from both trees.

`SUBMISSION_LEDGER.md:130-131` records the change that created this: *"[P2] Uniform-panel CIFAR-10-C
Tent/EATA 4th-decimals 0.0080/0.1239, 0.1313 → canonical 0.0079/0.1241, 0.1314"* — i.e. the panel was moved
**away from** the values that recompute from `LOCKED_ANALYSIS_RESULTS.json` and the values
`EVIDENCE_MATRIX.md` records as "`[DONE]` Tent/EATA 4dp refreshed to raw", and toward the head-to-head
values, while `G8_EXACTRANK_REGEN.md`'s ACTION line says the opposite: *"drop the interpolated-quantile
from the headline path."*

**Why it matters.** A reader following the manifest's own provenance pointer gets different numbers in the
4th decimal and a different protocol label ("stress grid" vs "mixed head-to-head"). The manifest is
described in `kbound_short_appendix.tex:271` as "the **authoritative index** for every promoted number".

**Fix.** Point `source` at the head-to-head files (or recompute the panel from the stress-grid lock) and
say in one sentence which of the two 5-seed aggregates is canonical and why.

---

### [MAJOR] F4-6 — CIFAR-10-C "seed 0" is a different experiment from seeds 1–4, and two mutually inconsistent seed-0 artifacts are both in the release

**Location:** `experiments/kbound/results/stress_grid_multiseed_v1/seed{0..4}/result_manifest.json`;
`kbound_short.tex:622-624,909`

**Evidence.** Run manifests:

| seed | git_hash | python | torch | numpy | finished |
|---|---|---|---|---|---|
| 0 | `4896181799ad5d…` | **3.12.13** | **2.5.1** | 2.4.6 | **2026-07-02** |
| 1–3 | `6a237ed489c305…` | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-11/12 |
| 4 | `571c89f25989…` | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-12 |

Seed 0 was run three weeks later on a different commit and a different Python/torch/numpy stack.
Consequently two seed-0 results exist and disagree: the stress-grid aggregate has Tent
`eps_conformal 0.02154032`, regret `0.0015671/0.0085938/0.1231852`; the head-to-head per-condition dump
has `eps_conformal 0.02112897`, regret `0.0013056/0.0083322/0.1239907`. Seeds 1–4 are byte-identical
between the two trees (I diffed condition names and ε: identical), so **all** the divergence in F4-5 comes
from seed 0. The paper describes the grid as "pre-registered as Protocol~A; seeds $0$--$4$" (`:624`) and
the panel tier as "locked (Tent/EATA; **five raw seeds**)" (`:909`) with no disclosure.

The same pattern holds for ImageNet-C: `win_hunt_v5_imagenetc_ms/` has run directories for seeds 1–4 only
(python 3.9.23, torch 2.8.0, 2026-07-15/16); the seed-0 file in `pooled_5seed/` is an md5-identical copy of
`win_hunt_v5/imagenetc_aggr/per_condition_imagenetc_sar_seed0.json`, run 2026-07-09 under python 3.12.13 /
torch 2.5.1 on a third commit. `pooled_5seed/` carries no `result_manifest.json`.

**Why it matters.** A five-seed variance claim requires the seeds to differ only in seed. Here seed 0
differs in code version, interpreter and framework, which is the most likely explanation of the SAR
non-reproduction (F4-1) and casts the same doubt on Tent/EATA seed 0, which remain promoted.

**Fix.** Disclose the heterogeneity in the panel footnote, or re-run seed 0 under the seeds-1–4 stack. At
minimum add a `result_manifest.json` to `pooled_5seed/`.

---

### [MAJOR] F4-7 — Table III and all four ablation tables cannot be regenerated: their sole input file is not in the release

**Location:** `kbound_short.tex:702-728` (`tab:gates`), `:988-1062` (`tab:abl-alpha`,
`tab:abl-estimator`, `tab:abl-transfer`); `docs/research/kbound/scripts/ablation_exactrank.py:36`;
`docs/research/kbound/scripts/gate_baseline_comparison.py:213`

**Evidence.** The ablation harness hard-codes its input:

```python
# ablation_exactrank.py:35-36
def load(cand):
    f = os.path.join(RES, f"per_condition_cifar10c_{cand}_seed0.json")
```
with `RES = <scripts>/../experiments/kbound/results`. No such file exists there, nor anywhere for the
stress grid (`stress_grid_multiseed_v1/seed0/` has no per-condition dump — see F4-1). The five output
JSONs that *are* committed (`ablation_alpha.json`, `ablation_estimator.json`, `ablation_transfer.json`,
`ablation_dropout.json`, `ablation_exactrank.json`) are zero-content placeholders — every byte is a space
(F4-8), so the recorded `input_sha12` provenance is unreadable too.

The gate table has the same problem. `REVIEWER_REPRO_PACKET.md:135` instructs
`python scripts/gate_baseline_comparison.py   # reads the committed per-cell dump`; running it verbatim
gives `TypeError: expected str, bytes or os.PathLike object, not NoneType` at line 213, because `--in` is
required and `cifar10c_percell*.json` does not exist anywhere in the tree
(`find / -name 'cifar10c_percell*'` → empty). Only `--selftest` (pure synthetic) runs.

I did confirm that `n=432, 149 harmful` in the `tab:gates` caption corresponds to **seed 0 alone**
(seed-0 harmful base rate 0.3449 × 432 = 149; `B<=0` count in the seed-0 head-to-head dump = 149), which
contradicts `REVIEWER_REPRO_PACKET.md:174`: *"the locked output of the full CIFAR-10-C stress grid ...
432 conditions, **5 seeds**"* — 5 seeds is 2160 cells.

**Why it matters.** Table III is described by the authors as "the central empirical claim" and "the
cheapest high-value thing to reproduce". It cannot be reproduced, and neither can any ablation.

**Fix.** Commit `per_condition_cifar10c_{tent,eata,sar}_seed0.json` for the stress grid (or repoint the
harness at the head-to-head seed-0 dump and re-run), export `cifar10c_percell.json`, and correct the
packet's "5 seeds" to "seed 0, 432 cells".

---

### [MAJOR] F4-8 — 139 committed text artifacts are dataless placeholders, including every ablation result, the cost table's source, and the Office-Home runner source

**Location:** `docs/research/kbound/experiments/kbound/results/*` (13 files);
`experiments/kbound/officehome/*.py` (10 files); `docs/research/kbound/edge/artifacts_real/*`;
`EDIT_NOTES_2026-07-23.md:48-52`

**Evidence.** A scan for files with non-zero size but zero non-whitespace bytes finds 139 `.json`/`.py`/
`.csv`/`.md` artifacts. Examples (size / readable bytes):

```
3113 / 0   docs/research/kbound/experiments/kbound/results/ablation_alpha.json
5587 / 0   docs/research/kbound/experiments/kbound/results/ablation_exactrank.json
 385 / 0   docs/research/kbound/experiments/kbound/results/cost_profile.json
 746 / 0   docs/research/kbound/experiments/kbound/results/multiseed_camelyon17_tent.json
1858 / 0   docs/research/kbound/experiments/kbound/results/official_headtohead.json
17202 / 0  experiments/kbound/officehome/run_officehome_kbound.py
18989 / 0  experiments/kbound/officehome/oh_analyze.py
6888 / 0   docs/research/kbound/edge/artifacts_real/checklists/S01_checklist.csv
 390 / 0   docs/research/kbound/edge/artifacts_real/calibration/kga_edge_meta.json
```

The authors already know the mechanism: `EDIT_NOTES_2026-07-23.md:48-50` — *"**iCloud placeholders.** 24 of
28 figure PNGs in `figures/` and `theory_v2/` are iCloud placeholders on this Mac (0 bytes readable
locally)."* What the note misses is that the same condition affects result JSONs and source code. It also
records that the submission figures were *"reconstructed losslessly from the compiled PDFs"* rather than
regenerated from data — a provenance statement worth putting in the paper if it stands.

Consequences: `tab:cost` ("+0.20 ms/decision, +44.8 MB rollback copy") has no readable source;
`tab:multiseed`'s namesake artifact is unreadable (F4-3); the entire Office-Home experiment code is
unreadable, so the one panel row `EVIDENCE_MATRIX.md` already flags as "NON-TRACEABLE" also cannot be
re-derived from source. `STORAGE_MANIFEST.json` checksums only 3 files, so no release guard catches this.

**Why it matters.** From the outside these files look committed and are cited as evidence; they contain
nothing.

**Fix.** Materialize the files (`Finder → Download Now`, per the authors' own note), then add a release
guard that rejects any tracked text artifact whose content is whitespace-only, and extend
`STORAGE_MANIFEST.json` checksums to every artifact a table depends on.

---

### [MAJOR] F4-9 — `reproduce_submission.sh`, the documented one-command verification, aborts at step 1 on a clean checkout

**Location:** `docs/research/kbound/scripts/reproduce_submission.sh:33-45`;
`docs/research/kbound/tests/test_calibration_split_integrity.py:10-11`

**Evidence.** The script runs `set -euo pipefail` and then pytest over four test files as step `[1]`. I
installed pytest and ran exactly those four files: **15 passed, 2 failed.**

```
FAILED docs/research/kbound/tests/test_calibration_split_integrity.py::test_edge_calibration_sessions_disjoint
FAILED docs/research/kbound/tests/test_calibration_split_integrity.py::test_edge_split_audit_seals_before_heldout
FileNotFoundError: '/root/kb/docs/experiments/kbound/results/edge_real_phone_v1/split_audit.json'
```

Two independent defects: (a) the path is malformed —
`REPO = Path(__file__).resolve().parents[4]` then `REPO / "docs" / "experiments" / ...` yields
`<root>/docs/experiments/...`, which is not a real directory; (b) even at the correct path,
`experiments/kbound/results/edge_real_phone_v1/` contains only `publication_gate.json`, so
`calibration_summary.json` and `split_audit.json` are absent. With `set -e`, step 1 kills the script and
steps 2–9 (theory audit, table regeneration, unified result audit, ledger validation, head-to-head check,
cached-artifact check) never run.

**Why it matters.** This is the single command `REVIEWER_REPRO_PACKET.md:212` gives a reviewer to "verify
cached artifacts". It cannot succeed. The two failing tests are *calibration split integrity* tests —
precisely the property F4-2 and F4-3 show is violated elsewhere.

**Fix.** Fix the path, and either commit the edge split artifacts or mark the tests `skipif` on their
absence so the guard degrades honestly instead of failing.

---

### [MAJOR] F4-10 — The appendix ImageNet-C per-seed table is stale, and its accompanying claim is false under the promoted rule

**Location:** `kbound_short_appendix.tex:286-311`; `kbound_short.tex:798-802`

**Evidence.** The appendix says *"Table~\ref{tab:imagenetc-perseed} reports the SAR row of
Table~\ref{tab:imagenetc-faithful} **per seed**"* and then reports pooled
`regret KGA 0.0107`, `FA_u 0.007` — the *interpolated-quantile* numbers, while the main table and panel
report `0.0264` and `FA_u 0.000` under the exact-rank rule. Recomputing per seed under the promoted
exact-rank rule from the same files:

| seed | KGA (exact) | always-adapt | always-freeze |
|---|---|---|---|
| 0 | 0.0319 | 0.0625 | **0.0319** |
| 1 | 0.0312 | 0.0595 | **0.0312** |
| 2 | 0.0102 | 0.0425 | 0.0284 |
| 3 | 0.0290 | 0.0441 | **0.0290** |
| 4 | 0.0297 | 0.0561 | 0.0389 |

So under the rule the paper says it uses, KGA **exactly ties always-freeze on 3 of 5 seeds** (it abstains
everywhere, and abstain→freeze), and the pooled beats-both margin is carried entirely by seeds 2 and 4.
The appendix prose *"Point estimates improve both fixed-policy regrets on every seed"*
(`kbound_short_appendix.tex:280`) is therefore false under the promoted rule; it was true under the
withdrawn interpolated rule. The per-seed gap-CI statements ("excludes zero on both gaps for seeds 0--1")
likewise describe the old rule.

**Why it matters.** The appendix is the only place a reader can see how concentrated the headline win is.
As printed it silently mixes two calibration rules and overstates per-seed consistency.

**Fix.** Regenerate `tab:imagenetc-perseed` under the exact-rank rule (and, per F4-2, under a genuine
split), and rewrite the accompanying paragraph to say the win is driven by 2 of 5 seeds.

---

### [MAJOR] F4-11 — Evidence seals for two panel rows hash the wrong artifacts

**Location:** `experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json` (`cifar10_1_K`,
`cifar10c_tent_eata`)

**Evidence.** For CIFAR-10.1, the seal hashes
`experiments/kbound/results/cifar101_multiseed_v1/seed{0..4}/result_manifest.json`. But those manifests
belong to a run with `n_conditions_per_seed: 24` and pooled regrets `0.0024/0.0035/0.0045`
(`cifar101_multiseed_v1/pooled_summary.json`), whereas the promoted row is
`0.0021/0.0190/0.0017`, `FA_u 0.167`, `FA_c 0.444`, `n=48` — which I traced to a *different*, unsealed
artifact, `experiments/kbound/results/cifar101_protocol_K_v1/analyze_F_results.json`
(`test_locked: {regret_kga: 0.0020625, regret_adapt: 0.0190208, regret_freeze: 0.0017083,
false_adapt: 0.4444, n_test: 48}`). The manifest entry `cifar10_1_K` has no `source` field at all
(nor does `rxrx1_J`).

For CIFAR-10-C, the seal hashes `LOCKED_ANALYSIS_FINDINGS.md` + `LOCKED_ANALYSIS_RESULTS.json`, neither of
which contains the promoted `0.0079/0.1241/0.1314` (F4-5); indeed `LOCKED_ANALYSIS_FINDINGS.md` reports a
third set again (`tent 0.00139 / 0.00774`, `eata 0.00127`) and an SAR ε range `[0.0124,0.0137]` that
excludes the current seed-0 value 0.0268 — evidence it predates the seed-0 replay.

**Why it matters.** The seal is the release's integrity primitive. Hashing a file that does not contain
the number gives false assurance; I verified 67/69 hashes and still could not use two of them.

**Fix.** Seal the artifacts that actually contain each promoted number; add `source` to every manifest
track; add a CI check that each sealed file contains the manifest's value.

---

### [MAJOR] F4-12 — The Camelyon17 panel row has no artifact in the release at all, and the KB-CLAIM-022 quarantine rationale rests on the same two absent files

**Location:** `kbound_result_manifest.json` `camelyon17_ood.source`; `LOCK_SEAL.json` `camelyon17_ood`;
`claim_ledger.json` KB-CLAIM-022 `supporting_artifacts`; `PHASE6_LEAKAGE_AUDIT.md:43-45`

**Evidence.** Of 69 sealed files, exactly two are missing, and both belong to Camelyon17:

```
MISSING  audits/integrity_2026-06-20/camelyon_reconciliation/VERDICT_phase1.md
MISSING  audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json
```

`docs/research/kbound/audits/` exists but contains only three `MAIN_PAPER_*.md` files; `find . -path
'*camelyon_reconciliation*'` returns nothing. The manifest's `source` for the row is the directory
`audits/integrity_2026-06-20/camelyon_reconciliation/`. `PHASE6_LEAKAGE_AUDIT.md:43-45` derives the whole
quarantine argument (`POOLED_test_val_idval` vs `OOD_test_only`) from `recon_results.json`. Separately,
`claim_ledger.json` points KB-CLAIM-022 at `archive/audit_only/camelyon17_protocol_G_pooled_beats_both`,
which Phase 6 itself notes "is **not materialized** on disk" (fix-list item 1, still open).

**Why it matters.** `0.0000/0.0000/0.1381` is a promoted panel row with zero verifiable backing, and the
withdrawal that protects the paper's integrity story is documented only in prose.

**Fix.** Restore the reconciliation directory (its hashes are already recorded, so restoration is
verifiable) or re-run `camelyon_G_reconciliation.py` and re-seal.

---

### [MAJOR] F4-13 — Data availability is not achievable: `DATA.md` does not exist, and the documented acquisition command for a promoted dataset is not a real command

**Location:** `docs/research/kbound/STORAGE_MANIFEST.json` (`artifacts[3..5]`);
`scripts/download_data.py:156-159`; `docs/research/kbound/scripts/download_all_datasets.sh:52`

**Evidence.** `STORAGE_MANIFEST.json` gives, for ImageNet-R:
`"reproduction_command": "bash scripts/download_data.py --dataset imagenet-r  (see DATA.md)"`.
`scripts/download_data.py` accepts only `--enron`, `--cifar10`, `--all`, `--no-kaggle` (lines 156–159);
there is no `--dataset` flag and no ImageNet-R support — it is the parent monorepo's Enron/CIFAR-10
downloader. `find / -iname 'DATA.md'` returns nothing, yet three manifest entries say "see DATA.md".
PACS says only "see DATA.md (DomainBed PACS download)". CIFAR-10-C says `bash AETTA/download_cifar10c.sh`;
there is no `AETTA/` directory in the release.

Only 2 of 9 tracked datasets have a working acquisition path
(`docs/research/kbound/scripts/download_all_datasets.sh`: Camelyon17 + ImageNet-C, and that script does
pin ImageNet-C to Zenodo 2235448 — good). No dataset version is pinned otherwise: WILDS is installed as
`$PIP install -q wilds` (line 52), unpinned, so a replicator may get different Camelyon17/iWildCam/RxRx1
splits than the paper; there are no dataset checksums; Office-Home split, ImageNet-R revision, and
CIFAR-10.1 version are unspecified. `ImageNet-C` paths in the run manifests are the author's private
`/Users/pratik_n/imagenetc_local`.

**Why it matters.** Seven of nine benchmark tracks cannot be obtained by a third party from the
instructions given.

**Fix.** Write `DATA.md` with, per dataset: canonical URL/DOI, exact version/revision, split definition,
licence, and an archive checksum. Pin `wilds==<version>` in the download script. Fix or delete the
`download_data.py --dataset` line.

---

### [MAJOR] F4-14 — The environment is not reproducible: four different Python versions and three torch versions across promoted runs, none matching the lock file

**Location:** `.python-version`; `requirements.lock.txt`; `requirements.txt`; per-run
`result_manifest.json` files

**Evidence.** Aggregating `"python"` across all committed run manifests gives
`3.12.13` (33 runs), `3.14.3` (13), `3.11.11` (6), `3.9.23` (6). Torch spans 2.5.1 / 2.8.0 / 2.12.0 on
`device: "mps"`. Meanwhile `.python-version` says `3.12.7` (used by none of them),
`requirements.lock.txt` pins `torch==2.12.0`, `numpy==2.4.4`, `scikit-learn==1.8.0` — which matches the
CIFAR seeds 1–4 stack but **not** the ImageNet-C headline runs (`torch 2.8.0`, `numpy 2.0.2`,
python 3.9.23), and `pyproject.toml` says only `requires-python = ">=3.11"`. `requirements.txt` is
inherited from the parent monorepo and pulls `tensorflow-cpu`, `streamlit`, `prefect`, `mlflow`, `fastapi`,
`kaggle`, `shap`, `lime` — none of which the K-Bound pipeline needs, all with `>=` floors rather than pins.

Note also that the benefit estimator is `GradientBoostingRegressor` from scikit-learn; its RNG/subsample
behaviour is version-sensitive, and `sklearn` version is not recorded in any run manifest, so `b_hat` —
and therefore ε and every decision — is not pinned.

**Why it matters.** Even a replicator with the data cannot reconstruct the environment that produced any
given headline number, and the numbers depend on an unpinned estimator.

**Fix.** Record `sklearn` version in `result_manifest.json`; ship one lock file per track (or a Docker
image) matching the stack actually used; delete the inherited monorepo requirements; make
`.python-version` agree with something real.

---

### [MINOR] F4-15 — `claim_ledger.json` has drifted from its own recorded checksum

**Location:** `docs/research/kbound/STORAGE_MANIFEST.json` `artifacts[0]`;
`docs/research/kbound/claim_ledger.json`

**Evidence.** The manifest records `claim_ledger.json` as `size_bytes: 23179`,
`sha256: 81b9d1e0cef4155140551a758e43fc404589ff2983f4d90cb2ef52c09e3adf92`, with
`"reproduction_command": "authoritative source (hand-maintained wording/status authority)"`. On disk it is
25 336 bytes with sha256 `bff76f3c4781f4b863cb362b832c1cbfd16f4c015dc1524b58ef47caa3f9be23`. The other two
tracked artifacts in the manifest verify exactly.

**Why it matters.** The one file declared the "wording/status authority" is the one whose seal is broken,
so a reader cannot tell which version of the withdrawal statuses the paper was checked against.

**Fix.** Regenerate `STORAGE_MANIFEST.json` at freeze time and add its regeneration to
`reproduce_submission.sh`.

---

### [MINOR] F4-16 — Two tables of the same grid at the same operating point disagree, while the prose asserts they agree

**Location:** `kbound_short.tex:718-719` (`tab:gates`), `:993-994`, `:998`, `:1023-1025` (`tab:abl-alpha`)

**Evidence.** The text claims exact agreement:

> `:993-994` — "At the deployed operating point (Tent, $\alpha{=}0.10$) the harness **reproduces the locked
> gate row of Table~\ref{tab:gates}** (regret $0.0017$, $\mathrm{FA}_{\mathrm u}{=}0$, adapt $0.51$,
> coverage $0.69$)."

`tab:gates:719` gives that row as `0.0017 / 0.000 / 0.000 / 0.51 / **0.68** / 0.000`. And the radius-free
row is `FA_u = 0.049` in `tab:gates:718` and in the prose at `:694` and `:698-699`, but `0.051` in
`tab:abl-alpha:1025` and in the prose at `:998`. Same candidate, same n = 432, same α.

**Why it matters.** Small, but it is the exact place the paper asserts that two independently computed
tables agree, and they do not. It also suggests the two tables were produced by different harness
revisions — consistent with F4-7 (the ablation harness input is missing, so the discrepancy cannot be
resolved from the repo).

**Fix.** Regenerate both tables from one harness run and reconcile, or drop the "reproduces" sentence.

---

### [MINOR] F4-17 — The authors' own audit documents contradict each other on three promoted items, and a P1 leakage item the ledger marks RESOLVED is still open

**Location:** `PHASE6_LEAKAGE_AUDIT.md:56`; `PHASE7_INTEGRATION_AUDIT.md` (via `SUBMISSION_LEDGER.md:125-127`);
`EVIDENCE_MATRIX.md`; `KBOUND_REMAINING_TODOS.md`; `REVIEWER_REPRO_PACKET.md:105-108`;
`research_lock/WIN_HUNT_v5_PROTOCOL_SHELL.yaml:97`

**Evidence.**
1. **RxRx1 0.2531 vs 0.2587.** Phase 6 item 2: *"RxRx1 J promoted regret_adapt is `0.2587` ... the promoted
   value **is** the 5-seed real-ckpt rerun"*. Phase 7 P0 (quoted in the ledger): *"RxRx1 always-adapt
   regret 0.2587 → 0.2531 (canonical; **0.2587 was the sar_online sub-candidate**, not the promoted
   protocol-J aggregate)"*. `EVIDENCE_MATRIX.md` says *"RxRx1 fresh 0.0/0.2587/0.0 real ckpt confirmed"*.
   The artifact settles it: `rxrx1_protocol_J_v1/analyze_F_results.json` has
   `"candidate": "sar_online"` and `regret_adapt: 0.2530598958` — so the printed number is right but
   Phase 7's stated reason for it is factually wrong, and three audit docs give three provenance stories.
2. **P1 leakage item still open.** `KBOUND_REMAINING_TODOS.md`: *"**P1** De-register the pooled Camelyon
   `id_val` config ... from `WIN_HUNT_v5_PROTOCOL_SHELL.yaml:97` (re-lists `id_val`)."* That line still
   reads `split_ref: CAMELYON17_PROTOCOL_G_RECONCILED_v2 (default domains test/val/id_val; ...)`, while
   `SUBMISSION_LEDGER.md:111-114` declares G9 `[RESOLVED]` and "Quarantine intact". (The
   `bootstrap_win_cis.py` half of the item *is* done — I grepped and found no `id_val` there.)
3. **Repro packet contradicts the ledger on Office-Home.** `REVIEWER_REPRO_PACKET.md:107`: *"Office-Home
   and the CIFAR stress grid carry the CI-backed beats-both."* The ledger and panel say Office-Home is
   "locked (OOF no-harm only; **LOO BB not promoted**)", and its own artifact
   `research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json` records `"beats_both_robust": false` and
   `kga_vs_freeze.ci_excludes_zero: false`. The packet is what an external reviewer is handed.
4. **`EVIDENCE_MATRIX.md` fix-queue items marked `[TODO-local]`** (regenerate manifest, PACS FA_u, Office-Home
   traceability) are marked `[RESOLVED]` in `SUBMISSION_LEDGER.md §6` without the matrix being updated,
   so the two "single sources of truth" disagree about their own status.

**Why it matters.** The ledger claims to supersede all other audit docs, but the superseded docs are
still shipped and are the ones an external reviewer is pointed at.

**Fix.** Either delete the superseded docs from the release or stamp each with a one-line
"SUPERSEDED BY SUBMISSION_LEDGER §N" header; reconcile the three RxRx1 stories in one sentence; close or
re-open the P1.

---

### [MINOR] F4-18 — Anonymity and archival: the paper cites an "anonymized repository" that the release de-anonymizes, and no DOI exists

**Location:** `kbound_short.tex:1167-1171`; `CITATION.cff:12-20`; `REPRO_INVENTORY.json:6-8`

**Evidence.** `kbound_short.tex:1168` — "link to saved raw cells or seed aggregates in the **anonymized
repository**". `CITATION.cff` gives `family-names: Niroula`, `given-names: Pratik`, the author's email,
and `repository-code: "https://github.com/Pratikn03/AutoML_Flagship_V8"`. `REPRO_INVENTORY.json` gives a
*different* URL, `"remote_origin": "https://github.com/Pratikn03/K-Bound.git"`, plus
`"upstream": "origin/flagship-history"`. `CITATION.cff:26-27` shows the DOI line still commented out
(`# doi: 10.5281/zenodo.XXXXXXX`).

**Why it matters.** For a double-blind venue this breaks anonymity in the artifact; for any venue the two
conflicting repository URLs and absent DOI mean there is no citable, immutable snapshot the paper's
numbers can be pinned to.

**Fix.** Scrub identity from the review artifact, pick one canonical repository, and mint the Zenodo DOI
before submission (the checklist for it already exists at `RELEASE_CHECKLIST.md`).

---

### [NIT] F4-19 — The forbidden-phrase gate is a substring grep and fires on the paper's own disclaimers

**Location:** `docs/research/kbound/claim_ledger.json` `forbidden_wording`;
`EDIT_NOTES_2026-07-23.md:7-9`

**Evidence.** `EDIT_NOTES` claims "the compiled PDFs pass the forbidden-phrase greps from
`claim_ledger.json`". Running those 52 phrases against `kbound_short.tex` + appendix yields 7 hits, all of
which are the paper *denying* the phrase: `:41` "does not claim **universal** improvement", `:402`
"**jackknife+** is not claimed", `:1127` "not an **assumption-free** or conservative default", `:114`
"**beats both** fixed policies only where regimes are mixed and detectable".

**Why it matters.** A gate that fires on correct text will be routinely overridden, so it provides no real
protection against the wording it exists to block (e.g. "beats both Camelyon17").

**Fix.** Make the gate phrase-plus-context (e.g. forbid `beats both` within N tokens of `Camelyon`), or
whitelist the negation forms explicitly.

---

## What I checked and could NOT fault

- **Evidence seal hashes.** 67 of 69 SHA-256 entries in `LOCK_SEAL.json` verify byte-for-byte against
  files on disk; 0 mismatches. Only the two Camelyon reconciliation files are missing (F4-12).
- **PACS row.** Recomputed the 4-domain mean over `PACS_MULTISEED_RESULTS.json`:
  `0.0431 / 0.0176 / 0.0446`, mean FA_u `0.0092593` — matches `kbound_short.tex:915` and the manifest
  exactly. All three declared `source_files` exist. Seeds `[0,1,2]`, 18 cells/seed as claimed. G3 RESOLVED
  is **true**.
- **ImageNet-R row.** 40 per-condition files present = 10 backbones × 4 seeds. Mean across backbones:
  `0.0112 / 0.0064 / 0.0325`, FA_u `1/480` — matches `:916` exactly. G4 RESOLVED is **true**.
- **CIFAR-10.1 row.** `0.0021 / 0.0190 / 0.0017`, `FA_u 0.167`, `FA_c 0.444`, `n=48` all recompute from
  `cifar101_protocol_K_v1/analyze_F_results.json`; the FA_u/FA_c decomposition
  (`adapt_rate 0.375 × 0.4444 = 0.1667`) is correct and correctly labelled. (Only the seal points
  elsewhere — F4-11.)
- **Three-source mixture.** `0.0059117 / 0.0632323 / 0.0342043`, n = 143, composition
  36 + 35 + 72 — exact match between `kbound_short.tex:953`, the manifest, `research_lock/KBOUND_MIXED_STREAM_v2.json`
  and `mixed_protocol_oof_v2_result.json`. The manifest even self-reports the caveat "saved scorer used
  5000 bootstrap replicates while protocol text states 10000" — credit for that.
- **Office-Home `0.0157`.** `EVIDENCE_MATRIX.md` called it "NOT FOUND in any raw artifact"; it *is* in
  `research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json` (`regret_kga 0.015714285714285722`, `n_test 35`,
  `beats_both_robust: false`). The panel's "LOO BB not promoted" tier is consistent with the artifact.
- **Per-seed byte identity.** `win_hunt_v5_imagenetc_ms/seed{1..4}/per_condition_imagenetc_sar_seed*.json`
  are md5-identical to their `pooled_5seed/` copies — no silent divergence between the two trees for
  those seeds.
- **Quarantine coverage in the tables.** `SAR withheld` appears correctly at `kbound_short.tex:523`,
  `:909`, `:1100`, `:1205`; I found no CIFAR-10-C SAR value in the uniform panel, the primary numeric
  table, or any pooled aggregate. The leak is confined to the body paragraph at `:637-642` (F4-1).
- **Theory validator artifacts.** All `results_thm*.json` referenced by `reproduce_submission.sh` step [2]
  are present and non-empty; the `0.028 → 0.0316` correction from `INTEGRITY_FIXES.md` does not appear
  as a stale digit in the short paper (the claim is not in the short paper at all).
- **Cross-file consistency of `kbound_numbers.tex`.** Every macro in
  `paper/generated/kbound_numbers.tex` matches the manifest / head-to-head artifacts it is derived from;
  the `\HeadToHead*` block matches `HEADTOHEAD_RESULTS_cifar10c_tent_primary.json` exactly, including
  `KgaDec 0.6847` and `Poem 0.0088` / `Aetta 0.0073`.

---

## Open questions for the author

1. For ImageNet-C SAR, was leave-one-cell-out ε ever run end-to-end, or has the promoted number always
   come from `cexact(np.abs(bh-B))` over the full 27-cell vector? If the LOO number (0.0289, FA_u 1/135)
   is the honest one, does the beats-both survive the paired bootstrap?
2. Why was stress-grid seed 0 re-run on 2026-07-02 under a different commit and torch version? Was that
   re-run intended to replace seeds 1–4's stack, and if so why were they not re-run too?
3. Which of the three CIFAR-10-C 5-seed aggregates is canonical —
   `LOCKED_ANALYSIS_FINDINGS.md` (0.00139/0.00774), `LOCKED_ANALYSIS_RESULTS.json` (0.0016259/0.0079757),
   or `HEADTOHEAD_RESULTS_*.json` (0.0015736/0.0079234)? What changed between them?
4. Is `experiments/kbound/results/wilds_kbound/` an intentional promoted artifact for Table VIII, or was
   `experiments/kbound/results/multiseed/multiseed_camelyon17_*.json` (which carries a ready-made
   contradictory `latex_row`) the intended source?
5. Can `audits/integrity_2026-06-20/camelyon_reconciliation/` be restored? Its hashes are already sealed,
   so restoration is independently verifiable.
6. What is the `"reproduces_locked": false` flag in `research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json`
   asserting for Office-Home and iWildCam — that the OOF replay diverged from an earlier locked value?
   If so, by how much?
7. Is there a plan to publish `DATA.md` and a Zenodo snapshot, and to pin the `wilds` and `scikit-learn`
   versions the promoted runs used?
