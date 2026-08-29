# Reproducing K-Bound

> **CURRENT ENTRY POINT (2026-08-29):** run
> `bash docs/research/kbound/runbooks/release_candidate.sh all` under Python 3.12. Sections 0a--5
> below are retained as a forensic account of historical heterogeneous runs and legacy helper
> scripts; several named headline artifacts and commands are intentionally superseded. Current
> claims come from `KBOUND_SHORT_RESULT_AUDIT.md`, `claim_ledger.json`, the canonical reconciled
> panel, and the separate receipt-linked CCT-20/So2Sat authorities. A green legacy helper is not a
> current release PASS.

The current release gate reconstructs or validates every maintained headline surface from committed
artifacts. Historical GPU provenance remains heterogeneous as documented below. The So2Sat target
was never accessed, and the CCT-20 target result uses its own receipt-linked release manifest.

## 0. Environment
```
python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements-research.txt
KBOUND_PYTHON=.venv/bin/python \
  bash docs/research/kbound/runbooks/release_candidate.sh all
```
The full release profile includes Torch, torchvision, WILDS, and the Word-export dependencies because
repository-wide test collection imports those surfaces even though the release gate never trains a
model. Linux CI uses the hashed `requirements-research-ci.lock.txt`; the smaller
`requirements-paper.lock.txt` is sufficient only for a presentation rebuild from already validated
authorities. A `Dockerfile` at the repo root pins the full environment for GPU re-runs.

### 0a. Disclosure: the committed multi-seed runs were NOT produced under one environment

Read this before treating any across-seed spread in this project as seed variance. It is not:
some of it is toolchain variance. Scanned from the 43 committed `result_manifest.json` files
(fix-queue item 19; F4-6, F4-14; `NUMBERS_PACK.md §7.3`).

**CIFAR-10-C stress grid — three distinct stacks across five seeds:**

| seed | git hash | Python | torch | numpy | finished |
|---|---|---|---|---|---|
| **0** | `4896181799ad` | **3.12.13** | **2.5.1** | **2.4.6** | 2026-07-02 |
| 1 | `6a237ed489c3` | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-11 |
| 2 | `6a237ed489c3` | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-12 |
| 3 | `6a237ed489c3` | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-12 |
| **4** | **`571c89f25989`** | 3.14.3 | 2.12.0 | 2.4.4 | 2026-06-12 |

**ImageNet-C — two stacks across seeds 1-4, and seed 0 comes from a third:**

| seed | git hash | Python | torch | numpy | finished |
|---|---|---|---|---|---|
| **0** (via `win_hunt_v5/imagenetc_aggr/`) | `87bf90aaadce` | **3.12.13** | **2.5.1** | **2.4.6** | 2026-07-09 |
| 1, 2 | `27a7e977f033` | 3.9.23 | 2.8.0 | 2.0.2 | 2026-07-15/16 |
| 3, 4 | `1adea4515b8c` | 3.9.23 | 2.8.0 | 2.0.2 | 2026-07-16 |

Four further facts, all verified:

1. **ImageNet-C seed 0's `argv` omits `--severities 1 3 5` and `--max-images 4000`**, both present
   for seeds 1-4. Seed 0 is not the same experiment as seeds 1-4.
2. `pooled_5seed/per_condition_imagenetc_sar_seed0.json` is **md5-identical**
   (`8b655a29360a23ca6fa9f5658f91d95a`) to
   `win_hunt_v5/imagenetc_aggr/per_condition_imagenetc_sar_seed0.json`, so the pooled tree's seed 0
   is a copy of the older run, not a re-run under the seeds-1-4 stack.
3. **`pooled_5seed/` has no `result_manifest.json` at all** — the aggregate that carries the
   ImageNet-C headline records no environment.
4. **0 of 43 run manifests record a scikit-learn version.** `b_hat` comes from
   `GradientBoostingRegressor(subsample=0.8)`, so ε and therefore *every decision* is
   scikit-learn-version-dependent. An independent recompute reproduced the shipped `b_hat` at
   correlation 0.999996-1.000000 but **not bit-for-bit**, which is exactly the signature of an
   unpinned estimator.

**What this permits and forbids.**
*Permitted:* reporting the five seeds as five runs, and reporting their spread as an upper bound on
seed variance. *Forbidden:* calling it a five-seed variance estimate, or attributing any seed-0
outlier to the seed. The CIFAR-10-C SAR quarantine (`CIFAR10C_SAR_QUARANTINE.md`) is the concrete
instance: seed 0's harmful base rate is 0.53 against ~0.10 on seeds 1-4, and seed 0 is also the one
seed on a different Python, torch and commit. Those two facts cannot be separated from the release.

**Historical closeout requirement:** the original draft required a seed-0 rerun under the
seeds-1-4 stack and `argv`, a recorded `scikit_learn` version, a pooled-run manifest, and a paper
footnote for every multi-seed claim. The maintained paper instead takes the bounded route: these
heterogeneous runs are retrospective/descriptive only and support neither a seed-variance estimate
nor a current significance claim. A future paper claim that attributes dispersion specifically to
random seed would still require the homogeneous rerun and complete environment manifests above;
the historical blanket footnote prescription is not a condition imposed on the current manuscript.

## 1. One-command verification (CPU, seconds)
```
python3 docs/research/kbound/scripts/reproduce_headlines.py
```
Exits 0 iff all checks PASS. It:
- **rebuilds** the CIFAR-10-C Tent and EATA beats-both verdicts from the raw per-condition logs
  (`experiments/kbound/results/per_condition_cifar10c_{tent,eata}_seed0.json`) using the exact-rank
  KGA certificate — the verdict is recomputed, not read back;
- checks the decision-gate certificate has FA_u = 0 (`gate_comparison.json`);
- confirms the ImageNet-C SAR, three-source, and Camelyon17 headline numbers are present in the
  locked artifacts (`results_source.json`, `research_lock/KBOUND_MIXED_STREAM_v2.json`,
  `recon_results.json`, `.../imagenetc_aggr/decisive_tta_results.json`).

> **Known failures of this script as of 2026-07-26 — do not report a green run as a clean bill.**
> - `recon_results.json` **does not exist** in this release. The Camelyon17 check either skips or
>   must be repointed at `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29`, which carries
>   the regret triple but **not** the promoted `FA_u = 0`. See `SUBMISSION_LEDGER.md §8`.
> - `.../imagenetc_aggr/decisive_tta_results.json`'s sibling `checkpoint.json` is a NUL-filled
>   iCloud placeholder (`PLACEHOLDER_INVENTORY.md`, group D).
> - The ImageNet-C SAR number this script confirms is the **in-pool** value 0.0264. Under the
>   2026-07-26 leave-one-out-of-pool radius fix it is 0.0289 with FA_u 1/135
>   (`SUBMISSION_LEDGER.md §9`). A check that passes against the old constant is checking the wrong
>   constant.

## 2. Locked artifacts (CI-confirmed headlines)
| Result | Artifact |
|--------|----------|
| CIFAR-10-C Tent/EATA stress beats-both | `results_source.json` (locked_analysis) + raw `per_condition_cifar10c_*` |
| ImageNet-C SAR beats-both | `experiments/kbound/results/win_hunt_v5/imagenetc_aggr/decisive_tta_results.json` |
| Decision-gate comparison | `gate_comparison.json` |
| Three-source mixture (constructed) | `research_lock/KBOUND_MIXED_STREAM_v2.json` |
| Anytime streaming | label-informed offline diagnostic only; not promoted as label-free evidence |
| Exact-rank ablations | `experiments/kbound/results/ablation_exactrank.json` (input SHAs inside) |
| Controller cost profile | `experiments/kbound/results/cost_profile.json` — **NUL-filled placeholder, unreadable** |
| Camelyon17 OOD | **sealed but not recomputable**; `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29` only |

> `cost_profile.json` and `ablation_{alpha,dropout,estimator,transfer}.json` are iCloud
> placeholders with zero readable bytes. The published cost and ablation tables therefore cannot be
> re-derived from this release. Full census and recovery command:
> `PLACEHOLDER_INVENTORY.md`.

## 3. Re-run from logged data (no raw images, seconds each)
```
python3 docs/research/kbound/scripts/ablation_exactrank.py        # alpha/estimator/dropout/transfer
python3 docs/research/kbound/scripts/cost_profile.py              # controller overhead
# The historical anytime script uses target-label window accuracies and is
# intentionally excluded from the label-free reproduction path.
```

## 4. Strengthening harnesses (for camera-ready)
- **Official baselines.** Run the authors' released code on the same conditions —
  POEM <https://github.com/yarinbar/poem> (Bar et al., NeurIPS 2024) and
  AETTA <https://github.com/taeckyung/AETTA> (Lee et al., CVPR 2024; estimator `learner/dnn.py::aetta`) —
  convert their per-condition output to the decisions format, then score:
  ```
  python3 docs/research/kbound/scripts/baseline_decisions_adapter.py --method aetta --input aetta_out.csv --out aetta_decisions.json
  python3 docs/research/kbound/scripts/baseline_decisions_adapter.py --method poem  --input poem_out.json  --out poem_decisions.json
  python3 docs/research/kbound/scripts/official_baselines_headtohead.py \
      --decisions poem=poem_decisions.json aetta=aetta_decisions.json
  ```
  (Without `--decisions` it uses clearly-labelled protocol-matched ports.) Produces the head-to-head
  table with a paired-bootstrap CI on the KGA gap and Holm correction.
- **Multi-seed no-harm.** One command runs seeds 0–4 and aggregates (Camelyon17 fully wired; other
  tracks are templated in the script):
  ```
  bash docs/research/kbound/scripts/run_multiseed.sh camelyon
  ```
  or aggregate existing per-seed runs manually:
  ```
  python3 docs/research/kbound/scripts/multiseed_aggregate.py --track iWildCam \
      --glob "experiments/kbound/results/iwildcam_*seed*/*.json"
  ```
  Emits across-seed mean±std, a seed-level bootstrap CI, a stable-no-harm verdict, and a LaTeX row.
  (`--demo` shows a labelled synthetic example.)

## 5. Full GPU re-run (raw data)
The WILDS/CIFAR/ImageNet runners live in `docs/research/kbound/scripts/` (e.g.
`cifar_tent_mps_v2.py`, `run_wilds_camelyon17.py`, `pacs_vlcs_runner.py`) and regenerate the
per-condition logs consumed above. Protocols are pre-registered in `research_lock/*.yaml`.
