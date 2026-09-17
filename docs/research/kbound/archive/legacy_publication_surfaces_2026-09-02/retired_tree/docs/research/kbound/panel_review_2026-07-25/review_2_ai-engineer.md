# Reviewer 2 - AI / ML Systems Engineer (test-time adaptation)

*Scope of this review: the shipped `kga/` library, the experiment drivers in `src/scripts/kbound/`,
`docs/research/kbound/scripts/`, `experiments/kbound/`, and the archived result JSONs. All paths are
relative to the repository root (the review copy lives at `/root/kb`). Every claim below was checked
by reading the source and, where numeric, by running `python3` against the archived artifacts.*

## Bottom line

This is one of the most carefully self-audited research repos I have reviewed: the manifests
hash-verify, the ledger enumerates its own gaps, and the manuscript volunteers caveats most papers
would bury. But the engineering does not match the ledger's own description of it in three ways that
matter. First, the `kga` package that the paper presents as "KGA" is not the code that produced a
single headline number — every promoted track is scored by one of **seven** copy-pasted
`decide_kga()` functions that use `np.quantile`, i.e. exactly the interpolated rule that
`kga/certificate.py` was written to replace and that G8's own action item said to drop from the
headline path. Second, and worse, in all of those copies the conformal radius ε is the empirical
quantile of the residuals of *the very cells it then decides*; I verified that the in-sample coverage
the paper reports as calibration evidence (`0.898`) is a **deterministic identity of `np.quantile` at
n=432** — it is 0.8981 for all four archived seeds, and 0.8889 for every 27-cell ImageNet-C file —
so it carries zero information about coverage, and the accompanying `FA_u ≤ α` is structurally
forced rather than tested. Third, the flagship CIFAR-10-C locked track cannot be re-derived from the
released tree at all: `stress_grid_multiseed_v1/seed0/` contains no per-condition records, and the
project's own `_locked_analysis_script.py` crashes on its first `load(0, "tent")`. Layered on top are
two protocol-disclosure problems a TTA reviewer will catch immediately: the "pre-registered
432-condition stress grid" is the driver's `--quick` mode (6 of 15 corruptions, severities {1,5};
`result_manifest.json` records `"quick": true`), and the ImageNet-C "mechanism-faithful SAR is
harmful on 15.6% of cells" result was run at `--adapt-lr 0.004`, 16× the official SAR learning rate
the code's own docstring cites, with `layer4` adapted contrary to official SAR.

**Verdict: major revision required.** The theory may well survive; the empirical layer as shipped
cannot support "finite-sample certificate" language, and the shipped library must either become the
scoring path or stop being presented as one.

## What is done well

- **Manifest integrity is real.** All nine `source_artifact` entries in
  `docs/research/kbound/RESULT_MANIFEST.json` hash-verify byte-exactly against the files on disk
  (I recomputed every sha256). Byte counts match too. This is rare and worth saying.
- **Episodic TTA hygiene is correct.** `_clone_for_tta()`
  (`docs/research/kbound/scripts/cifar_tent_mps_v2.py:702`) does `copy.deepcopy(base)` and builds a
  fresh optimizer per cell, and `run_cifar_benchmark` calls `fn(model, stream, ...)` with the frozen
  `model` every time. **There is no optimizer/BN-state leak across corruption cells** — the single
  most common source of fake TTA gains is genuinely absent here. I looked for it specifically.
- **The eval pool is disjoint from the adaptation stream.** `build_stream_and_eval`
  (`cifar_tent_mps_v2.py:826-849`) carves a class-balanced eval pool first and adapts on
  `remain = np.setdiff1d(...)`. Benefit is not measured on the adaptation batch. That is better
  discipline than much of the published TTA literature.
- **The SAM implementation inside `sar_adapt` is genuinely faithful** on the mechanics that usually
  get botched: `old_p` is saved before the ascent step and restored *exactly* before `opt.step()`,
  so there is no residual perturbation drift; the reliable-sample filter, the entropy EMA, and the
  model+optimizer recovery snapshot are all present (`cifar_tent_mps_v2.py:742-795`).
- **The unit tests are not tautological.** `tests/test_kga_package.py` has real boundary tests
  (`test_boundary_lower_zero_abstains`, `test_just_past_lower_boundary_adapts`,
  `test_ebern_lcb_matches_canonical_formula` recomputing the Maurer-Pontil closed form to 1e-12).
- **`kga/certificate.py::split_conformal_rank_radius` gets the conformal rank right** —
  `k = ceil((n+1)(1-alpha))`, returning an observed order statistic, with an explicit docstring
  warning against `numpy.quantile`. The knowledge is in the repo; it just isn't wired to the results.
- **The manuscript volunteers several of these caveats itself** (`kbound_short.tex:330-333` admits
  the archived JSONs use the interpolated quantile; `:340-342` declines to claim jackknife+). The
  problem is the ledger and the audit docs, which overstate what the code does.
- **The fallback-estimator path is stamped and traceable.** `per_condition_serialize.py` records
  `kga_backend`; I confirmed `numpy_knn_fallback` appears only in the smoke run
  (`smoke05_20260701_113034`), never in a promoted track.

---

## Findings

### [BLOCKER] F2-1 — The shipped `kga` package is not the code that produced any headline result; the real scorer is a seven-way-duplicated fork using the rule `certificate.py` exists to replace

**Location.** `kga/certificate.py:239-252`; `docs/research/kbound/scripts/cifar_tent_mps_v2.py:151-165`;
`experiments/kbound/wilds/analysis.py:55-67`; `docs/research/kbound/G8_EXACTRANK_REGEN.md:19-22`.

**Evidence.** `kga/certificate.py:246-249` implements the exact rank rule and explicitly warns:

```python
k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
return float(np.sort(arr)[k - 1])
# "Unlike ``numpy.quantile``'s default interpolation, this is an observed
#  order statistic and matches the finite-sample rank argument."
```

The actual scorer for every promoted track is `decide_kga`, whose docstring claims
"split-conformal radius" but whose body is:

```python
# docs/research/kbound/scripts/cifar_tent_mps_v2.py:164
eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
```

`grep -rn "def decide_kga"` returns **seven** independent copies (`cifar_tent_mps_v2.py:151`,
`wilds/analysis.py:55`, `scripts/kga_breadth.py:83`, `scripts/frontier_validation.py:41`,
`scripts/run_decision_baselines.py:83`, `scripts/run_wilds_camelyon17.py:45`,
`theory_v2/realdata/eps_recal/_probe2.py:14`), plus two more inlined in
`experiments/kbound/wilds/per_condition_serialize.py:60,84`. `grep -rn "np.quantile" ` finds the
same interpolated call in 20+ analysis scripts.

Meanwhile `grep -rn "from kga"` outside `kga/` and `tests/` returns exactly three importers:
`src/scripts/kbound/smoke_trichotomy.py` (a smoke test on synthetic data this repo generates),
`experiments/kbound/wilds/panel_capture.py` (imports `evidence_v2` only), and
`scripts/multicandidate_decide_kga.py` (imports `routing` only). **Nothing imports
`kga.certificate`, `kga.policy`, or `kga.evidence` on any result-producing path.**

I confirmed the archived artifacts still carry the interpolated radius: for
`win_hunt_v5/cifar10c_aggr/seed0/per_condition_cifar10c_tent_seed0.json`,
`eps_conformal = 0.024829934468407443`, which equals `np.quantile(res, 0.9)` to all 17 digits, while
the exact rank `r_(244)` would be `0.025266187151207008`. The stored `kga_decision` field reproduces
exactly from the interpolated eps.

`G8_EXACTRANK_REGEN.md` closes G8 as PASS with the action item *"drop the interpolated-quantile from
the headline path"* and states *"Algorithm 1 eps = exact rank rule"*. `SUBMISSION_LEDGER.md:74-79`
records G8 as `[RESOLVED = PASS]`. The driver was never changed.

**Why it matters.** The paper ships a library whose stated purpose is the finite-sample certificate,
and none of the numbers come from it. A reviewer who `pip install`s `kga` and reruns the pipeline
gets different decisions than the tables. It also means the exact-rank claim in
`kbound_short.tex:800-801` ("using the exact split-conformal radius ε=ρ_(k)") describes a post-hoc
rescoring script (`g8_exactrank_regen.py`) rather than the released scorer, and G8's "RESOLVED"
status in the ledger is not supported by the source.

**Fix.** Delete the seven forks; make every driver call `kga.certificate.split_conformal_rank_radius`
and `kga.policy.decide`. Re-run (or re-score, from the stored `b_hat`/`B`) every promoted track under
that single path and regenerate the tables. If the interpolated numbers are kept for historical
reasons, label every table cell accordingly rather than closing G8.

---

### [BLOCKER] F2-2 — ε is fit on the residuals of the same cells it then decides; the reported "realized coverage 0.898" is an arithmetic identity, and `FA_u ≤ α` is structurally forced

**Location.** `docs/research/kbound/scripts/cifar_tent_mps_v2.py:151-165`;
`experiments/kbound/wilds/analysis.py:55-67`; `kbound_short.tex:316-318` and `:335-337`;
`SUBMISSION_LEDGER.md:104-107`.

**Evidence.** `decide_kga` computes leave-one-out `Bhat[i]` (honest per point), then

```python
eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))   # ALL N residuals, incl. cell i's own
dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
```

and decides all N cells with it. Point i's own residual is one of the N used to build its own radius.
Consequences I verified numerically:

```
$ # over the four archived 432-cell CIFAR-10-C Tent seeds
seed1 n 432 eps 0.019934 in-sample cov 0.8981   exact r_(k) 0.020041
seed2 n 432 eps 0.020939 in-sample cov 0.8981   exact r_(k) 0.021595
seed3 n 432 eps 0.020959 in-sample cov 0.8981   exact r_(k) 0.021011
seed4 n 432 eps 0.021424 in-sample cov 0.8981   exact r_(k) 0.021860
$ # over every 27-cell ImageNet-C file (12 files, 3 methods x 4 seeds)
all files: n 27, in-sample cov 0.8889
```

`0.8981 = 388/432` and `0.8889 = 24/27` are fixed by `n` and `np.quantile`'s linear interpolation
alone. The manuscript reports this as evidence: *"the calibration-set coverage is approximately
nominal (realized 0.898 at nominal 0.90, α=0.10) — empirical calibration evidence"*
(`kbound_short.tex:335-337`). It is not empirical; it is `np.quantile`'s definition.

The same construction forces the safety headline. For seed 0 Tent (n=270) I counted exactly
`27 = α·N` residuals exceeding ε. A false adapt requires `B ≤ 0` while `Bhat − ε > 0`, hence
`|Bhat − B| > ε`; so the number of false adapts is bounded above by `⌈αN⌉` **by construction**,
independent of whether the estimator is any good. `FA_u = 0/2160` is therefore not a test of the
certificate.

Finally, `kbound_short.tex:316-318` states: *"We fit the benefit estimator and its radius out of
fold, so the data used to fit Δ̂ and the data used to calibrate its residual are disjoint for every
point."* That is true of Δ̂ and false of ε. And `SUBMISSION_LEDGER.md:105` (Phase 6 verdict) claims
*"All 7 live tracks fit epsilon on the calibration split only (LOO/cross-fit), score a disjoint test
partition"* — there is no disjoint test partition in `decide_kga`.

**Why it matters.** This is the load-bearing empirical claim of the paper (RQ2, `FA_u ≤ α`). As
implemented it is a tautology, and the only quantitative coverage number offered is an identity. The
Phase-6 audit doc asserts the opposite of what the code does.

**Fix.** Either (a) hold out a genuine calibration partition of cells, fit ε there, and score the
remainder once; or (b) implement jackknife+ (Barber et al. 2021) at level 1−2α, which the manuscript
already names but does not use; or (c) if the in-sample radius is retained, remove the "0.898
realized coverage" sentence, state that FA_u ≤ α is enforced by construction, and correct the ledger's
Phase-6 verdict.

---

### [BLOCKER] F2-3 — The locked CIFAR-10-C headline is not reproducible from the released tree: seed 0's per-condition records are absent and the locked analysis script crashes

**Location.** `experiments/kbound/results/stress_grid_multiseed_v1/seed0/`;
`experiments/kbound/results/stress_grid_multiseed_v1/_locked_analysis_script.py:15-17`;
`kbound_short.tex:637-640`; `RESULT_MANIFEST.json` KB-CLAIM-010.

**Evidence.** `seed0/` contains only `decisive_tta_results.json`, `decisive_tta_table.md`, and
`result_manifest.json`. Seeds 1–4 each contain `per_condition_cifar10c_{tent,eata,sar}_seedN.json`
with 432 records. A repo-wide `find . -name 'per_condition_cifar10c_*seed0*'` returns only the
270-cell `win_hunt_v5` files and an 8-cell rebuild smoke file — **no 432-cell seed-0 per-condition
file exists anywhere.**

The locked analysis script hard-requires it:

```python
def load(seed, cand):
    p = os.path.join(RES, f"seed{seed}", f"per_condition_cifar10c_{cand}_seed{seed}.json")
    return json.load(open(p))
```

Running it as released:

```
$ python3 _locked_analysis_script.py
Traceback (most recent call last):
  File ".../_locked_analysis_script.py", line 87, in <module>
    cond, pooled, betvar, meta, seed_reg = build(cand)
  File ".../_locked_analysis_script.py", line 28, in build
    recs = load(s, cand)["records"]
FileNotFoundError
```

`LOCKED_ANALYSIS_RESULTS.json` reports `"seeds": [0,1,2,3,4]`, `"false_adapt_den": 2160` (= 5×432),
and `eps_conformal_per_seed[0] = 0.02154` (matching seed 0's aggregate) — so the analysis *was* run
with seed 0 present; the raw records were simply not archived. The manuscript then says
(`kbound_short.tex:637`) *"The previously mismatched SAR aggregate was rebuilt from all five saved
per-condition seed files"* — those five files do not exist in the released artifact set.

**Why it matters.** KB-CLAIM-010 is tier "locked", which `kbound_short.tex:571` defines as
*"trace to raw cells/seeds"*. One fifth of the headline track (including the SAR seed that triggered
the G2 quarantine in the first place) traces only to a rounded aggregate. `REVIEWER_REPRO_PACKET.md`
cannot be satisfied.

**Fix.** Archive `seed0/per_condition_cifar10c_{tent,eata,sar}_seed0.json`, or downgrade CIFAR-10-C
to a 4-seed track (`false_adapt_den = 1728`) and restate every 2160-denominator number. Add a CI
check that runs `_locked_analysis_script.py` against the archived tree.

---

### [MAJOR] F2-4 — The measured benefit Δ conflates the TTA update with BatchNorm-statistic replacement, and no BN-stats-only baseline exists anywhere in the repo

**Location.** `docs/research/kbound/scripts/cifar_tent_mps_v2.py:1053` and `:1060`;
`cifar_tent_mps_v2.py:674-700`; `experiments/kbound/wilds/tta_methods.py:293-300`.

**Evidence.** The frozen and adapted models are evaluated in different BN modes:

```python
a0 = acc_on(model,   ex, ey, train_mode=False)   # source running stats
...
aa = acc_on(adapted, ex, ey, train_mode=True)    # batch stats (running stats were nulled)
```

and `_bn_affine_params` does `mod.track_running_stats = False; mod.running_mean = None;
mod.running_var = None` for the adapted clone. So `B = aa − a0` measures *(entropy-minimisation on BN
affine params) + (replacing source BN statistics with target batch statistics)*. The same asymmetry
holds throughout WILDS: every caller uses `train_mode=False` for `f0` and `train_mode=True` for `fa`
(`run_camelyon17_kbound.py:151`, `run_iwildcam_aetta.py:85,90`, `tta_methods.py:349,378`).

`grep -rn "bn_adapt\|bn_stats\|'bn'"` across `experiments/` and `docs/research/kbound/scripts/`
returns nothing: the method sets are `{frozen, tent, eata, sar, shot, kga}`. There is no BN-adapt
control anywhere.

The magnitude is not academic: seed 0 CIFAR-10-C Tent reports `always_freeze = 0.671` vs
`always_adapt = 0.786`, an 11.5-point gap on a benchmark where BN-statistic replacement alone
typically recovers most of Tent's gain.

Secondary issue in the same code: `acc_on` batches the eval pool in chunks of 512
(`cifar_tent_mps_v2.py:857-864`) / 256 (`tta_methods.py:293`) with the model in `train()`, so the
adapted model's predictions depend on the composition of each eval chunk. Accuracy is a function of
an undocumented eval batch size, and the inference procedure is transductive.

**Why it matters.** "Harmful adaptation" and "helpful adaptation" — the regime labels that drive
every downstream claim — are defined against a baseline handicapped by a known, large, well-understood
effect that is not part of the adaptation update the certificate is supposed to gate. The
beats-both margin against `always_freeze` (0.124 regret) is mostly this.

**Fix.** Add a `bn` method (source weights, target BN statistics, no gradient steps) to `TTA_METHODS`
and report Δ against it as well as against the source model; or evaluate `f0` in `train()` mode so
both arms use target BN statistics and Δ isolates the gradient update. State the eval batch size in
the config table.

---

### [MAJOR] F2-5 — The "pre-registered 432-condition stress grid" is the driver's `--quick` smoke mode: 6 of 15 corruptions, severities {1,5} only. The corruption restriction is never disclosed

**Location.** `docs/research/kbound/scripts/cifar_tent_mps_v2.py:134-137`, `:1041-1042`, `:1266`;
`experiments/kbound/results/stress_grid_multiseed_v1/seed1/result_manifest.json`;
`kbound_short.tex:622-625`; `kbound.tex:805-810`.

**Evidence.** The driver's full pre-registered grid is
`15 corruptions × 3 severities × 3 batch regimes × 3 compositions × 2 aggressiveness × 2 repeats =
1620 rows/method` (`cifar_tent_mps_v2.py:1040`, `SEVERITIES=[1,3,5]`, `CIFAR_C_ALL` has 15 entries).
`--quick` restricts to `CIFAR_C_QUICK` (6 corruptions) and severities `{1,5}`:
`6 × 2 × 3 × 3 × 2 × 2 = 432`. The CLI help calls this *"subset of corruptions/severities (fast smoke
run)"* (`:1266`); the code comment at `:134` says *"--quick uses a representative subset"*.

The archived run manifest confirms it:

```json
"quick": true,
"argv": ["docs/.../cifar_tent_mps_v2.py","--benchmarks","cifar10c","--quick","--methods","tent","eata","sar","--seed","1", ...]
```

and decomposing the 432 condition strings gives exactly
`corr ['contrast','defocus_blur','fog','gaussian_noise','jpeg_compression','pixelate']`,
`sev ['s1','s5']`.

`kbound.tex:806-810` describes the grid as *"over the official CIFAR-10-C corruptions, crossing
severity {1,5} × batch {...} × composition {...} × update {...}, 2 repeats (432 conditions/method)"*.
The severity restriction is disclosed; the fact that only 6 of the 15 official corruptions were used
is not, and "over the official CIFAR-10-C corruptions" reads as all of them.

**Why it matters.** The headline safety track is run in the harness's own smoke configuration. The
6 corruptions chosen exclude every one of the noise family except `gaussian_noise` and exclude
`snow`, `frost`, `motion_blur`, `zoom_blur`, `elastic_transform`, `glass_blur`, `impulse_noise`,
`shot_noise`, `brightness`. A reader cannot tell this from either manuscript.

**Fix.** State the six corruption names in the config table, or run the full 15-corruption grid. Also
note (F2-8 below) that because `rng` is a single sequential `default_rng(SEED)` advanced across the
cell loop (`:1026`, `:1052`), the 432-cell run is *not* a subset of a 1620-cell run — the draws differ.

---

### [MAJOR] F2-6 — The ImageNet-C "mechanism-faithful SAR is harmful on 15.6% of cells" result was produced at 16× the official SAR learning rate with `layer4` adapted, and the manuscript does not disclose the operating point

**Location.** `experiments/kbound/results/win_hunt_v5_imagenetc_ms/seed1/result_manifest.json` (argv);
`docs/research/kbound/scripts/cifar_tent_mps_v2.py:96-99`, `:757-759`, `:1035-1036`;
`kbound_short.tex:792-800`.

**Evidence.** The archived argv for the promoted ImageNet-C run is:

```
--corruptions gaussian_noise shot_noise impulse_noise --arch resnet50 --methods tent eata sar
--severities 1 3 5 --imagenetc-composition iid imbalanced single_class
--batch-regimes small --aggressiveness aggressive --adapt-lr 0.004
```

So: batch size 16 (`BATCH_REGIMES["small"]`), `steps=50` (`AGGRESSIVENESS["aggressive"]`), and
`ADAPT_LR = 0.004` overriding the declared aggressive lr of 2.5e-3. `sar_adapt` is called with
`sar_lr=None` (not in argv), so SAR runs SGD(momentum=0.9) at **lr = 4e-3**. The code's own docstring
says:

```python
# cifar_tent_mps_v2.py:758-759
# Official SAR uses its own lr (2.5e-4) + frozen final block; opt in via flags.
```

`SAR_FREEZE_LAYER4 = False` by default and is not set in argv, so the final ResNet stage *is* adapted
— explicitly contrary to Niu et al. Fifty passes over a 4-batch stream is 200 updates on 64 images
at 16× the reference learning rate.

The manuscript says only: *"SAR is harmful on 15.6% of cell–seed pairs under a mechanism-faithful
re-implementation (SAM, recovery reset, EMA) at the learning rate shared across candidates"*
(`kbound_short.tex:794-797`). The learning-rate value, the batch size, the step count, and the
`layer4` deviation are absent.

**Why it matters.** "Beats both" on this track (`0.0264 / 0.0529 / 0.0319`) is the paper's strongest
positive empirical claim, and it exists because SAR collapses. SAR's entire design goal is *not* to
collapse; running it at 16× lr with the block it is designed to freeze unfrozen destroys precisely
the mechanism being credited as "faithful". `SUBMISSION_LEDGER.md:112` lists
"official method != protocol-matched port" as a distinction the manuscript must hold; the wording
"mechanism-faithful" cuts against it.

**Fix.** State the exact operating point (lr 4e-3, bs 16, 50 steps, layer4 adapted) inline. Add a
control run at official SAR settings (`--sar-lr 2.5e-4 --sar-freeze-layer4`) and report whether the
beats-both survives. If it does not, relabel the claim as a stress-regime result.

---

### [MAJOR] F2-7 — `split_conformal_rank_radius` silently under-covers when `n < 1/α − 1`; the project's own validator handles this case correctly, and `route_panel`'s Bonferroni level makes it reachable

**Location.** `kga/certificate.py:249`; `kga/routing.py:57-60`, `:126`;
`docs/research/kbound/theory_v2/val_multicandidate.py:93-98`.

**Evidence.** The shipped implementation clamps:

```python
k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
return float(np.sort(arr)[k - 1])
```

When `ceil((n+1)(1−α)) > n` the clamp returns the **maximum** residual and coverage silently drops to
`n/(n+1)`. Measured (20 000 Monte-Carlo draws, |N(0,1)| residuals):

```
n_cal=5, alpha=0.1  ->  empirical coverage 0.8309   (nominal 0.90)
n_cal=9, alpha=0.1  ->  empirical coverage 0.9003
```

The repo's own validator does the right thing (`val_multicandidate.py:93-98`): it computes
`k = ceil((1-level)(n+1))` and returns `inf` when `k > n_cal`, with the comment
*"HONEST behaviour: the certificate abstains when the corrected level alpha/K"* cannot be met.

`kga/routing.py:126` sets `bonf = alpha / k` and passes it straight to
`candidate_lcb_from_calibration`. At α=0.1 with K=5 candidates the effective level is 0.02, requiring
`n_cal ≥ 49`; below that the FWER guarantee silently evaporates and `route_panel` will still return a
`committed=True` decision. `tests/test_kga_routing.py:42` only ever exercises `n_cal = 60`.

**Why it matters.** The shipped certificate library — the thing a practitioner would deploy — is
strictly less correct than the validator that supposedly justifies it, and fails silently in exactly
the small-calibration regime where a deployment gate is most likely to be used.

**Fix.** Return `float("inf")` (→ ABSTAIN) when `ceil((n+1)(1−α)) > n`, matching
`val_multicandidate.py`. Add a regression test at `n_cal ∈ {3,5,8}` asserting `epsilon == inf` and
`decide() == ABSTAIN`. Add the same guard inside `route_panel` for the α/K level.

---

### [MAJOR] F2-8 — `cifar10c_suite.py` seeds every cell's RNG with Python's salted `hash()`, so the grid is non-reproducible across processes

**Location.** `src/scripts/kbound/cifar10c_suite.py:70`.

**Evidence.**

```python
def load_cell(corr,sev):
    r=np.random.default_rng(sev*131+hash(corr)%9973); idx=r.choice(POOL,N_PER,replace=False)
```

`hash()` on `str` is salted per interpreter process unless `PYTHONHASHSEED` is set.
`grep -rn "PYTHONHASHSEED"` across the whole repo (`.py`, `.sh`, `.md`) returns **nothing**. Verified:

```
$ for i in 1 2 3; do python3 -c 'print(hash("gaussian_noise")%9973)'; done
9788
6852
9119
```

Every invocation draws a different 800-image subsample for every cell, despite the file's
`torch.manual_seed(0); np.random.seed(0)` at line 25.

**Why it matters.** This is a plausible mechanical explanation for the G2 quarantine
(`CIFAR10C_SAR_QUARANTINE.md`: *"not reproducible from the current seed-0 replay"*), and it means
`experiments/kbound/results/cifar10c_suite_results.json` cannot be regenerated. The quarantine doc's
reinstatement gate #2 ("reproduce seed 0 from clean raw outputs") is unachievable while this line
stands.

**Fix.** Replace with a stable digest, e.g.
`int(hashlib.blake2b(corr.encode(), digest_size=4).hexdigest(), 16)`, and set `PYTHONHASHSEED=0` in
the runbooks. Then re-check whether seed 0 reproduces.

---

### [MAJOR] F2-9 — The false-adapt event is defined inconsistently across code paths (`B < 0` vs `B ≤ 0`); 102 archived ADAPT decisions sit on `B == 0.0` exactly

**Location.** `experiments/kbound/wilds/analysis.py:85` and `:101-103`;
`docs/research/kbound/scripts/cifar_tent_mps_v2.py:167+` (`policy_metrics`);
`experiments/kbound/results/stress_grid_multiseed_v1/_locked_analysis_script.py:43`;
`docs/research/kbound/scripts/g8_exactrank_regen.py:27`; `kbound_short.tex:305`.

**Evidence.** The theorem statement controls `Pr(adapt, Δ ≤ 0)` (`kbound_short.tex:305`:
*"controls the marginal probabilities Pr(adapt, Δ≤0)"*). The locked script and the G8 regen both use
that event:

```python
fa_num = int(np.sum(is_adapt & (B <= 0)))          # _locked_analysis_script.py:43
poolFAe += sum((d=='ADAPT') and (b<=0) ...)        # g8_exactrank_regen.py:27
```

But `policy_metrics` — which produces the `false_adapt_rate_B<0` field in every
`decisive_tta_results.json`, and which gates the `beats_both` flag — uses strict inequality **and**
conditions on ADAPT:

```python
"false_adapt_rate_B<0": float(np.mean(B[adapt] < 0)) if adapt.any() else None,
...
"beats_both": bool(... and float(np.mean(B[adapt] < 0)) <= alpha),
```

That is `FA_c` (conditional), not the `FA_u` the ledger defines
(`SUBMISSION_LEDGER.md:24`: *"FA_u (unconditional false-adapt): Pr(commit to wrong sign) over the
mixture"*). Scanning all 65 606 archived per-condition records I found **500 cells with `B` exactly
`0.0`, of which 102 were ADAPT decisions** — violations of `Pr(adapt, Δ≤0)` that
`false_adapt_rate_B<0` scores as clean. (Concentrated in `mixed_headtohead_v1` baselines, but also
1 in `win_hunt_v5` CIFAR-10-C SAR and 1 in `natural_win_v1_camelyon` EATA.)

`SUBMISSION_LEDGER.md:79` even flags this as outstanding — *"Still fix FA_u marginal code label"* —
inside a gap marked `[RESOLVED = PASS]`.

**Why it matters.** Two different quantities are reported under the same name in different tables,
and the one used to gate `beats_both` is the more permissive of the two on both axes (strict `<`,
and conditioned on ADAPT). Any cross-table comparison of "FA_u" is apples-to-oranges.

**Fix.** Define one function, `false_adapt_unconditional(dec, B) = mean(is_adapt & (B <= 0))`, in
`kga/policy.py`; call it everywhere; emit `fa_u` and `fa_c` as separate, explicitly named fields; and
regenerate the summaries.

---

### [MINOR] F2-10 — The shipped `kga` CLI can only ever print `ABSTAIN`, and mixes incompatible units to do it

**Location.** `kga/cli.py:66-70`.

**Evidence.**

```python
calib_1d = calib.ravel()
residual_proxy = np.abs(calib_1d - float(np.median(calib_1d)))
cert = conformal_split(0.0, residual_proxy, alpha=args.alpha)
dec = decide(cert, alpha=args.alpha)
```

`delta_hat` is hard-coded to `0.0` and `epsilon ≥ 0` always, so `decide` can never satisfy
`delta_hat − epsilon > 0` or `delta_hat + epsilon < 0`: the `decide` subcommand is a constant
`ABSTAIN` generator. Separately, `residual_proxy` is a MAD-like spread of raw *detector scores*,
whereas `delta_hat` is a *risk difference* — the radius and the estimate are in different units, so
even the printed `epsilon` is meaningless.

**Why it matters.** `python -m kga decide` is the package's only entry point and the first thing a
reader will run. It looks like a working gate and is a no-op.

**Fix.** Either require `--benefits` / `--calib-residuals` and run a real certificate, or make the
subcommand `kga evidence` (report Z only) and remove the fake decision.

---

### [MINOR] F2-11 — `_importance_ess` computes the reciprocal of the weight its docstring specifies; on a variance-shrink shift the two directions disagree by 27× and give opposite conclusions

**Location.** `kga/evidence.py:192-215`.

**Evidence.** The docstring says the weights are
`w_i = N(test_i; mu_c, s_c) / N(test_i; mu_t, s_t)` — *"the weight that re-expresses a test
expectation as a calibration-distributed one"*. The code computes the inverse:

```python
log_w = (-0.5 * ((t - mu_t) / s_t) ** 2 - math.log(s_t)) - (-0.5 * ((t - mu_c) / s_c) ** 2 - math.log(s_c))
```

i.e. `log N(t; mu_t, s_t) − log N(t; mu_c, s_c) = log(p_test/p_calib)`. ESS is not invariant under
`w → 1/w`. Measured on calib ~ N(0,3), test ~ N(0,1):

```
code           ess_frac = 0.8848    ("good overlap")
docstring form ess_frac = 0.0322    ("severe overlap failure")
```

For a pure mean shift the two agree to within 10%, so the impact is regime-dependent — but the
`ess_frac` feature is documented as flagging poor source→target support overlap, and on
variance-shrinking shifts the implemented direction reports the opposite of the documented one.

**Fix.** Pick a direction, fix the docstring or the sign, and add a unit test asserting
`ess_frac < 0.1` on a variance-shrink pair.

---

### [MINOR] F2-12 — `empirical_bernstein` estimates the range from the data (invalidating Maurer–Pontil) and the module presents `[lower, upper]` as a simultaneous 1−α interval when it is 1−2α

**Location.** `kga/certificate.py:29-33` (module docstring), `:82-90` (`Certificate.lower/upper`),
`:178-181`; `kga/kga.py` `explain()`; `kbound_short.tex:41` (abstract).

**Evidence.** (a) When `benefit_range is None`:

```python
rng = float(arr.max() - arr.min())
```

Maurer–Pontil requires an *a priori* bounded interval `[a,b]`; substituting the sample range makes
`R` data-dependent and the stated LCB no longer holds at 1−α. The docstring flags this
("For `|p - y|` paired losses the exact range is 2.0 and should be passed explicitly") but it is the
default, and `KGA.certify(scores=...)` passes `benefit_range=None` unless the caller supplies it.

(b) Both `empirical_bernstein` (`ln(2/alpha)`) and `hoeffding` (`ln(1/alpha)`) produce *one-sided*
radii at level α. The module docstring at `:29-33` claims *"and, for the two-sided estimators, also
Delta <= Delta_hat + epsilon"*, and `Certificate.upper` is documented as *"Certified upper bound"*.
Two-sided simultaneous coverage from a one-sided bound requires a union bound: `[lower, upper]` is a
1−2α interval, not 1−α. `explain()` emits both, and the abstract speaks of *"a calibrated interval
Δ̂±ε"*.

**Why it matters.** `decide()` only uses one side per branch, so the per-decision guarantee survives;
but any statement about the interval (abstract, `explain()`, plots with a "conformal band ±ε") is off
by a factor of 2 in α.

**Fix.** Make `benefit_range` a required argument (or raise when it is `None`); relabel `lower`/
`upper` as one-sided at α each, and state the interval level as 1−2α wherever both are shown.

---

### [MINOR] F2-13 — Three mutually incompatible "evidence" implementations coexist; the one in the shipped package is used by nothing

**Location.** `kga/evidence.py` (338 lines), `kga/evidence_v2.py` (118 lines),
`docs/research/kbound/scripts/cifar_tent_mps_v2.py:651-674` (`evidence_vector`),
`experiments/kbound/wilds/tta_methods.py:53-74` (a verbatim copy of the latter).

**Evidence.** `kga/evidence.py` produces `[ks_mean, ks_max, disagree, entropy_shift, conf_shift,
calib_entropy, test_entropy, calib_conf, test_conf, ess_frac]` from *detector score* arrays.
`evidence_vector` produces the 11-dim `[pre_entropy, pre_conf, pre_pbal, post_entropy, post_conf,
post_pbal, pbal_drop, entropy_drop, frac_highconf, marginal_KL, update_norm]` — this is the `Z` in
every archived per-condition JSON. `evidence_v2.py` produces a third, disjoint 13-dim panel
(MaNo / nuclear / GdScore / per-sample quantiles) used only by `wilds/panel_capture.py`.

`kga/evidence.py` has no importer outside `kga/`, `tests/`, and `smoke_trichotomy.py`. Its docstring
nonetheless claims it computes *"exactly the observable quantities used throughout the K-Bound
experiments"* — it does not; none of its ten features appear in any result artifact.

Minor correctness note inside `evidence_v2.py:38`: `softrun`'s regime switch is a hard-coded
`if u < 1.0` on the mean negative log max-probability, and `X = X / (np.abs(X).max() + 1e-12)` uses a
**global** matrix max rather than per-row, making `mano_score` batch-composition dependent in a way
the MaNo paper's per-sample normalisation is not.

**Fix.** Promote `evidence_vector` into `kga/evidence.py` as the canonical schema, delete or clearly
mark the score-based version as an unrelated anomaly-detection utility, and stop claiming in the
docstring that it is what the experiments use.

---

### [MINOR] F2-14 — EATA and SAR ports deviate from the reference implementations in ways that change behaviour in exactly the collapse regime being studied

**Location.** `docs/research/kbound/scripts/cifar_tent_mps_v2.py:720-740` (EATA), `:781-783` (SAR);
`src/scripts/kbound/cifar10c_suite.py:80-84`.

**Evidence.**

1. **EATA has no redundancy filter.** Official EATA filters both high-entropy *and* samples whose
   softmax is too similar to a running average (`cosine similarity > ε_d`). Only the entropy filter
   `ent < 0.4·ln(K)` is implemented (`:727`). The docstring says "faithful-ish", but the ledger's
   Table XV row is labelled "EATA".
2. **EATA's Fisher is computed from the deployment stream, not source data.** `:723-726` takes
   `x0 = next(iter(stream))` — the first *target* batch — and uses squared entropy gradients.
   Official EATA estimates the Fisher on held-out source samples with a pseudo-label CE loss. Using
   the target batch makes the anti-forgetting regulariser anchor to a shifted distribution.
3. **SAR's second-filter fallback inverts the reliability principle.**
   `loss2 = ent2[keep2].mean() if keep2.any() else ent2.mean()` (`:783`). Official SAR produces `nan`
   and skips the update when the second filter is empty. Here, when *every* sample becomes unreliable
   — i.e. precisely at collapse onset — the code falls back to the **unfiltered** mean and takes the
   step anyway.
4. **`cifar10c_suite.py` implements "SAR" as running the EATA step twice.**
   `if method=="sar": step_tent(m,opt,bx,filt)` (`:83`) — no SAM, no EMA, no recovery. The docstring
   correctly calls it "SAR-style", but this file writes `experiments/kbound/results/
   cifar10c_suite_results.json` with a method key literally named `"sar"`, alongside the faithful
   implementation elsewhere in the tree.

**Fix.** Rename the ports consistently (`eata_entropy_only`, `sar_sam`) in code, results keys, and
tables; port EATA's redundancy filter and source-side Fisher; make SAR skip the step when
`keep2` is empty; delete or clearly quarantine `cifar10c_suite.py`'s `"sar"` key.

---

### [MINOR] F2-15 — `AnytimeMulticandidatePanel.update` early-returns, so candidates after the first rejecter never ingest that step

**Location.** `kga/routing.py:196-204`.

**Evidence.**

```python
for i, x in enumerate(benefits):
    self._procs[i].update(float(x))
    if self._procs[i].rejected_null(self.alpha, self.k):
        return i
```

Once candidate `i` crosses `log(K/α)`, processes `i+1 … K−1` are never updated with this step's
benefit. If the caller keeps streaming (the natural anytime use), those processes have a gap in their
observation sequence and their wealth is no longer a function of the full stream — and the returned
index is biased toward low candidate indices independent of evidence strength.

Related dead code in the same file: `bonferroni_multicandidate_route(..., alpha=alpha, ...)`
(`:98-111`) never reads `alpha`; `_BettingEProcess.__init__` stores `self.alpha = alpha` which is
never used (`rejected_null` takes `global_alpha` instead).

**Fix.** Update all K processes first, then scan for rejections and return `argmax` of the crossed
set (or the full crossed set). Remove the unused parameters.

---

### [NIT] F2-16 — `test_sar_faithful.py`'s headline "SAM restore must be exact" assertion is tautological, and `g8_exactrank_regen.py` hard-codes a personal absolute path

**Location.** `docs/research/kbound/scripts/test_sar_faithful.py:56-60`;
`docs/research/kbound/scripts/g8_exactrank_regen.py:2`.

**Evidence.**

```python
res_old = ((w + rho * g1 / g1.norm()) - rho * g2 / g2.norm() - w).norm().item()
res_new = (w.clone() - w).norm().item()
assert res_new < 1e-9 < res_old, "SAM restore must be exact"
```

`res_new` is `‖w − w‖ = 0` by construction and touches neither `sar_adapt` nor any project code. The
assertion is an identity presented as validation of the fix. (A3 immediately below it *is* a real
test — it calls `H.sar_adapt(..., reset_constant_em=1e9)` and checks the weights return to init.)

`g8_exactrank_regen.py:2` is `R = os.path.expanduser("~/Documents/AutoML_Flagship_V8/experiments/
kbound/results")`. The script that closed gap G8 cannot be run by anyone else without editing it.
The same absolute path appears in the archived ImageNet-C `result_manifest.json` argv.

**Fix.** Replace A2 with an assertion on the actual parameter tensors before/after
`sar_adapt`'s second step. Make `g8_exactrank_regen.py` resolve the results root relative to
`__file__` (as `_canonical.py` already does).

---

## What I checked and could NOT fault

These are things I went looking for specifically and did not find a problem with. Listing them so the
findings above can be read as targeted rather than shotgun.

1. **No TTA state leak across cells.** `_clone_for_tta` deep-copies the frozen model and a fresh
   `Adam`/`SGD` is constructed per cell in `tent_adapt`/`eata_adapt`/`sar_adapt`/`shot_adapt`. The
   frozen `model` object is never mutated by an adapter. I traced `run_cifar_benchmark`,
   `run_cifar101_benchmark`, and `wilds/tta_methods.py::_adapt`. This is the classic fake-gain bug and
   it is not present.
2. **Adaptation stream and evaluation pool are disjoint** (`build_stream_and_eval:838-840`,
   `remain = np.setdiff1d(np.arange(len(sev_y)), ev_idx)`); the eval pool is class-balanced, so
   collapse shows up as real accuracy loss rather than a label-prior artifact.
3. **SAM's weight restore is exact.** `old_p = [p.data.clone() for p in ps]` before the ascent and
   `for p, q in zip(ps, old_p): p.data = q` before `opt.step()` — no perturbation residue accumulates,
   and `load_state_dict` in the recovery branch copies in place so the optimizer's parameter
   references stay valid.
4. **`RESULT_MANIFEST.json` hashes all verify.** All nine `artifact_sha256`/`artifact_bytes` pairs
   match the on-disk files exactly (recomputed).
5. **Stored decisions reproduce from stored `b_hat`/`eps_conformal`.** For
   `win_hunt_v5/.../per_condition_cifar10c_tent_seed0.json`, recomputing
   `np.where(bh-e>0,'ADAPT',np.where(bh+e<0,'FREEZE','ABSTAIN'))` reproduces all 270 stored
   `kga_decision` values. No post-hoc decision editing.
6. **The `numpy_knn_fallback` estimator never contaminates a promoted track.** Grouping every
   per-condition file by `kga_backend`: 18 fallback records, all in `smoke05_20260701_113034`; every
   promoted track is `sklearn_gradient_boost`.
7. **The pooled/per-seed ImageNet-C duplicates are byte-identical** (`cmp` over all 12 pairs), so
   `g8_exactrank_regen.py::pick_seeds`'s reliance on unsorted `glob` order is harmless in practice.
8. **`regret` definitions agree across code paths.** `policy_metrics`'s
   `mean(max(a0,aa) − policy_acc)` is algebraically identical to `g8_exactrank_regen.py`'s
   `mean(|B|·1{action ≠ oracle})`, which is the ledger's definition. I checked this rather than
   assuming it.
9. **Numerical hygiene in `kga/certificate.py`:** unbiased variance (`arr.var(ddof=1)`), `n<2`
   returns `epsilon=inf` → ABSTAIN, `_check_alpha` bounds, `_as_1d` rejects empty/non-finite, no
   float `==` in the decision path, log-sum-exp stabilisation in `_importance_ess` and
   `evidence_v2._softmax`. `evalue_anytime`'s bets are genuinely predictable (statistics updated
   *after* the wager) and `bet_cap_frac=0.5` keeps `1+λx > 0`.
10. **`kga/policy.py` tie-breaking matches the ledger.** Strict inequalities mean `Δ̂ = ε` ABSTAINs,
    which is the closed-band convention of `def:strict-sound` ("abstain on |M| ≤ β"). `math.isinf`
    is handled before the comparisons. Tested at the boundary in `test_kga_package.py:189-204`.
11. **`evidence_v2` energy and margin computations are correct.** `energy = −log Σexp(L−max) − max`
    is `−logsumexp(L)`; `margin = top2[:,1] − top2[:,0]` on an ascending sort is
    (largest − second-largest) ≥ 0. I checked both by hand because sign errors here are common.
12. **`atc_threshold_acc` (`wilds/tta_methods.py:105`) implements ATC correctly** — threshold at the
    source error quantile of max-softmax, evaluated on target.
13. **The 27-cell ImageNet-C count matches the driver.** 3 corruptions × 3 severities × 3
    compositions × 1 batch regime × 1 aggressiveness = 27, consistent with the argv and the condition
    strings. (The 432 count does *not* match the driver's declared grid — see F2-5.)
14. **`torch.load` is called with `map_location` everywhere** (16 call sites), so no CUDA-pinned
    checkpoint will fail to load on MPS/CPU. Missing `weights_only=True` in the older CIFAR scripts is
    a supply-chain nit, not a correctness bug, and the WILDS scripts set it explicitly to `False` for
    a documented reason (pickled `algorithm` objects).

---

## Open questions for the author

1. **Is `kga/` intended to be the released artifact or a rewrite?** If the former, please make it the
   scoring path and regenerate the tables; if the latter, please say so explicitly, because the
   package docstring ("Theorem 3 of the K-Bound paper") and the ledger's G8 closure both imply the
   former.
2. **Where are `stress_grid_multiseed_v1/seed0/per_condition_cifar10c_*_seed0.json`?** They evidently
   existed when `LOCKED_ANALYSIS_RESULTS.json` was produced (`eps_conformal_per_seed[0] = 0.02154`).
   Can they be restored, or should the CIFAR-10-C track be restated at 4 seeds / n=1728?
3. **What is the beats-both margin against a BN-stats-only baseline?** You have the machinery; a
   `bn` method is ~5 lines. If `always_freeze` regret drops from 0.124 to something near
   `always_adapt`'s 0.008, the CIFAR-10-C beats-both claim needs rewording.
4. **Does the ImageNet-C SAR result survive at official SAR settings** (`--sar-lr 2.5e-4
   --sar-freeze-layer4`)? The flags exist and default to off; a single 27-cell run per seed would
   settle whether "SAR is harmful on 15.6% of cells" is a property of SAR or of the operating point.
5. **Was the `--quick` flag on the CIFAR-10-C stress runs deliberate?** If the intent was the full
   15-corruption grid, the locked track needs rerunning; if 6 corruptions was the pre-registration,
   the protocol YAML should name them and both manuscripts should say "six corruptions".
6. **How should `Δ = 0` cells be scored?** 102 archived ADAPT decisions sit on exactly `B = 0.0`.
   Theorem 3 controls `Pr(adapt, Δ ≤ 0)`; `policy_metrics` scores `B < 0`. Which is the promoted FA_u?
7. **Has the `hash(corr)` non-determinism in `cifar10c_suite.py` been ruled out as the cause of the
   SAR seed-0 non-reproduction?** If that file (or an ancestor of it) produced any archived SAR
   aggregate, the quarantine may have a one-line mechanical explanation.
8. **Do you intend `Certificate.lower`/`upper` to be used together?** If yes, the estimators need
   `alpha/2`; if no, `explain()` and the abstract's "calibrated interval" phrasing should be adjusted.
