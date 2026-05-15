# Full Research Audit - 2026-05-14

Scope: research-facing repository surface for the ELARA/RGA paper and thesis:
manuscripts, benchmark builders, experiment runner, reliability/fusion modules,
asset generation, generated JSON/CSV/LaTeX/PNG artifacts, configs, reports, and
tests. I did not manually inspect every raw image/CSV row byte-by-byte, but I
did inspect the generated metadata and run structured checks over the active
research input artifacts.

Verdict: the paper is not ready for symposium or conference submission as a
defensible empirical paper. The current story is interesting, especially the
negative cross-benchmark result, but the benchmark construction has split
validity problems that make the headline metrics weaker than the manuscript
currently implies.

Remediation update from the 2026-05-14 gap-closing pass:

- C1 is addressed in code and regenerated local artifacts: RealFusion-LA now
  builds source-row-disjoint train/validation/test source pools, emits
  `fusion_split`, and the runner consumes that predefined split.
- C2 is addressed for score generation: MVTec 3D-AD scores and embedding
  normalization are fit from original train-good/train observations only, and
  metadata records the score protocol. The attention-fusion run remains framed
  as a supervised fusion stress test, not the canonical MVTec anomaly-detection
  protocol.
- C3 is addressed for the current working tree: paper assets and PDFs rebuild
  with `./scripts/rebuild_paper.sh`; generated metadata tables now expose split
  safety and score-fit protocol rows.
- H4 is addressed: the DeLong test now filters finite rows pairwise and uses
  paired covariance for correlated ROC curves.
- H3 is quarantined: the README now labels `reports/metrics_*.csv` as legacy
  dashboard placeholders rather than research metrics.
- M1 is resolved for RealFusion-LA and explicitly scoped for MVTec 3D-AD:
  missing-domain CDA impacts no longer poison the Spearman calculation, and
  two-domain benchmarks now report the statistic as not applicable.
- M2 is marked: `run_fusion_experiment.py` is explicitly deprecated for
  research evidence and points to the split-safe benchmark/experiment path.
- M3 is addressed for the default cyber loader path: when official UNSW train
  and test files are present, discovery uses those files instead of recursively
  sweeping unrelated CSV exports.
- Verification after remediation: full pytest suite passed, RealFusion overlap
  audit reported zero cross-split source keys, MVTec metadata reported
  train/good score fitting, and paper/thesis PDFs rebuilt.

## Executive Decision

Do not submit the current PDF as-is.

The safest current claim is:

> ELARA/RGA is a research prototype for reliability-gated score-level anomaly
> fusion. Current experiments show a useful negative result: a validation-derived
> KS drift gate can help under label-aligned coherent score collapse while
> hurting on naturally paired MVTec 3D-AD. However, the current benchmarks need
> split-safe regeneration before the numeric results can be treated as strong
> held-out evidence.

Best near-term submission path:

1. Fix benchmark leakage and artifact rebuild issues.
2. Re-run both RealFusion-LA and MVTec with split-safe score generation.
3. Submit a workshop or symposium paper framed around the negative result and
   reliability-gate failure modes.

## Critical Findings

### C1. RealFusion-LA reuses source rows across fusion train/test splits

Impact: the label-aligned benchmark is not clean held-out evidence at the
source-observation level. This is likely one reason the clean table saturates at
about 0.999 ROC-AUC for many methods.

Evidence:

- `src/scripts/prepare_real_fusion_benchmark.py:328-347` builds all composite
  samples before the fusion split and samples source rows by label.
- `src/scripts/run_breakthrough_experiment.py:1173` performs the train/val/test
  split after the composite benchmark has already been created.
- The builder samples one source row per domain for each composite sample with
  replacement, but the later split is by composite `sample_id`, not by
  `domain/source_row`.

Structured check on `experiments/fusion/real_domain_fusion_inputs.csv`:

| Domain | Rows | Unique source+label rows | Repeated rows | Test source rows also in train | Test source rows also in val |
|---|---:|---:|---:|---:|---:|
| behavior | 7,077 | 5,234 | 1,843 | 454 | 66 |
| cyber | 7,083 | 5,830 | 1,253 | 369 | 37 |
| fraud | 7,061 | 4,817 | 2,244 | 437 | 103 |
| nlp | 7,016 | 5,783 | 1,233 | 318 | 43 |

Required fix:

- Build source-row-disjoint splits first, then construct train/val/test
  composites only from the corresponding source split.
- Add a test that asserts no `(domain, source_row, label)` key appears in more
  than one fusion split.

### C2. MVTec 3D score generation is transductive relative to the fusion split

Impact: the MVTec numbers should not be described as strict held-out benchmark
performance. The domain scores and normalization statistics are computed before
the train/test split over all discovered observations.

Evidence:

- `src/scripts/prepare_mvtec3d_fusion_benchmark.py:180-188` computes image
  features and normal-reference scores for the full dataset before any fusion
  split exists.
- `src/scripts/prepare_mvtec3d_fusion_benchmark.py:157-167` fits the normal
  reference from all normal observations in the prepared artifact.
- `src/scripts/run_breakthrough_experiment.py:1173` then applies a random
  supervised split over those precomputed samples.

Structured check on `experiments/fusion/mvtec3d_fusion_inputs.csv`:

- 3,226 paired samples, 6,452 rows, positive fraction 0.224.
- Fusion test fold contains 646 samples.
- 463 of those 646 fusion test samples come from original MVTec `train` or
  `validation` splits.
- Original split composition inside the fusion test fold: 183 `test`, 417
  `train`, 46 `validation`.

Required fix:

- Respect MVTec's original protocol: fit normal-reference scorers on original
  train-good samples only.
- Evaluate on original test samples, or explicitly call the current experiment
  a supervised random-split fusion stress test rather than an MVTec benchmark.
- Add a metadata field recording scorer-fit split and evaluation split.

### C3. Build reproducibility depends on untracked regenerated assets

Impact: the current working tree can rebuild the paper, but the submission
source package is not yet reproducible unless the regenerated assets and rebuild
script are versioned or recreated in CI.

Evidence:

- The current manuscript inputs `elara_*` and `mvtec3d_*` assets, and
  `./scripts/rebuild_paper.sh` regenerates those assets successfully in this
  working tree.
- `./scripts/rebuild_paper.sh` completed and produced
  `output/pdf/PAPER_DRAFT_v1.pdf` with 16 pages.
- Many required generated assets and the rebuild script itself are currently
  untracked in git, including `docs/research/figures/elara_*.png`,
  `docs/research/tables/elara_*.tex`, `docs/research/tables/mvtec3d_*.tex`,
  `src/scripts/emit_mvtec3d_assets.py`, and `scripts/rebuild_paper.sh`.
- A clean checkout or submission source archive that excludes those files may
  fail to rebuild or may rebuild a different paper from the same JSON results.

Required fix:

- Pick one asset policy: commit the generated paper assets, or make CI generate
  them from JSON before compiling.
- Track the rebuild script and any asset-generation scripts needed for a clean
  source rebuild.
- Add a CI check that runs the asset generator in a clean tree and fails if any
  manuscript `\input{}` or `\includegraphics{}` target is missing.

## High Findings

### H1. The paper formalism and current result mode differ on gating granularity

Impact: the manuscript describes per-sample reliability notation, while the
current configs preserve batch-level gating by default.

Evidence:

- `src/scripts/run_breakthrough_experiment.py:337-348` documents two modes and
  states batch-level gating preserves existing paper numbers.
- `configs/attention_real_fusion.yaml:81-83` leaves `per_sample_gating` disabled
  and comments that setting it true matches the paper formalism.
- `configs/attention_mvtec3d_fusion.yaml:81-83` has the same setting.

Required fix:

- Either regenerate all paper numbers with `per_sample_gating: true`, or state
  clearly in the method and captions that reported numbers use batch-level
  gating.

### H2. Current MVTec config comments are stale and contradict the active artifact

Impact: reproducibility instructions mislead readers and reviewers.

Evidence:

- `configs/attention_mvtec3d_fusion.yaml:1-8` says "Bagel category smoke-run",
  ResNet-18 penultimate features, 512 dimensions, about 78 test samples, and
  "not headline result."
- Active metadata says eight categories, 3,226 paired samples, 8-dimensional
  lightweight image-statistic embeddings, and the manuscript treats MVTec as the
  primary benchmark.
- `data/README.md` still emphasizes bagel as the listed MVTec asset, although
  the current generated artifact spans eight categories.

Required fix:

- Update config comments and data README to match the actual artifact.
- If stronger features are not implemented, remove ResNet-18 feature claims from
  the config comments.

### H3. Legacy report CSVs are placeholders and must not be cited

Impact: top-level `reports/metrics_*.csv` look like final metrics but contain
round numbers and empty standard deviations.

Evidence:

- `reports/metrics_fraud.csv`: `roc_auc,0.96,` with empty std.
- `reports/metrics_cyber.csv`: `roc_auc,0.93,` with empty std.
- `reports/metrics_behavior.csv`, `reports/metrics_vision.csv`, and
  `reports/metrics_fusion.csv` show the same pattern.

Required fix:

- Move these to `reports/archive/placeholder_metrics/` or delete them.
- Ensure README/dashboard text does not present them as research evidence.

### H4. The DeLong implementation is not a paired DeLong comparison

Impact: any claim that the paper uses DeLong tests should be softened or the
implementation should be replaced. The current function ignores covariance
between paired predictions.

Evidence:

- `src/uais/utils/stats.py:48-62` computes a p-value from `var_a + var_b`.
- A paired DeLong test for two models evaluated on the same labels requires the
  covariance of the two AUC estimates.

Required fix:

- Replace with a validated paired DeLong implementation, or use the existing
  paired bootstrap delta intervals as the primary inferential statement.

### H5. Protocol docs still promise split-safe alignment that the active
benchmark does not implement

Impact: reviewers can find a direct doc-code contradiction.

Evidence:

- `docs/research/EXPERIMENTAL_PROTOCOL.md` says "Alignment is performed within
  split boundaries only."
- RealFusion-LA aligns and samples source rows before fusion splitting.
- The same protocol doc says default split is 70/15/15, while the active runner
  uses 20 percent test and 10 percent of the remaining training set as val,
  which is 72/8/20.

Required fix:

- Update the protocol after C1/C2 are fixed.

## Medium Findings

### M1. CDA/Spearman evidence is not available despite table support

Remediation update: partly addressed. RealFusion-LA now computes finite
CDA/ECE Spearman after the missing-domain NaN handling fix
(`spearman_cda_vs_ece_reliability = 0.0516`, status `computed`). MVTec 3D-AD
now records an explicit status of `undefined: fewer than three finite domains`,
which is expected because Spearman correlation is not meaningful for the
two-domain RGB/depth benchmark.

Evidence:

- `experiments/fusion/craf_real_results.json` has `cda_validation.n_samples =
  400` but `spearman_cda_vs_ece_reliability = null`.
- `experiments/fusion/mvtec3d_results.json` is the same.
- The generated calibration/CDA tables show `CDA/ECE Spearman --`.

Required fix:

- If CDA correlation is a claim, compute and report it.
- Otherwise state CDA is qualitative/domain-attribution only in the current
  evidence package.

### M2. Legacy fusion pipeline remains research-invalid if used

Evidence:

- `src/scripts/run_fusion_experiment.py:44-61` trains fraud on a split, then
  scores the full fraud dataset with that model.
- `src/scripts/run_fusion_experiment.py:64-87` does the same for cyber.
- `src/scripts/run_fusion_experiment.py:115-149` aligns unrelated domains by
  index and truncation.

The current paper mostly moved to `prepare_real_fusion_benchmark.py` and
`run_breakthrough_experiment.py`, but this legacy script is still in the repo
and can regenerate invalid fusion inputs.

Required fix:

- Mark this script deprecated or update it to call the split-safe benchmark
  path.

### M3. Cyber loader still concatenates every CSV in the raw directory when
called as a directory loader

Evidence:

- `src/uais/data/load_cyber_data.py:97-115` recursively loads and concatenates
  all CSVs under `data/raw/cyber`.
- The raw directory contains official train/test files plus raw UNSW-NB15
  shard files.

This does not directly affect the current RealFusion-LA builder, which uses
the official train/test files explicitly, but it remains a trap for legacy
domain experiments.

Required fix:

- Default to official `UNSW_NB15_training-set.csv` and
  `UNSW_NB15_testing-set.csv`.
- Require an explicit flag to concatenate arbitrary raw CSV shards.

## What Is Defensible

- The codebase now has a real, scriptable multi-seed experiment runner.
- Both main result JSON files contain seeds `[42, 43, 44, 45, 46]`.
- Generated tables match the JSON content for the checked-in `rga_*` tables
  apart from the generator prefix comment.
- The MVTec artifact is naturally paired at the observation level: every sample
  has both `rgb` and `depth_or_xyz`.
- The manuscript is honest that RealFusion-LA is label-aligned and that MVTec
  uses lightweight image-statistic scorers.
- The negative result is real enough to be worth preserving after split-safe
  reruns: the gate misfires when the KS component treats legitimate category
  variation as drift.

## Verification Run

Commands and results:

- `PYTHONPATH=src ./.venv/bin/python -m pytest -q`
  - Result: passed, with 3 skipped tests and warnings.
- Fresh paper rebuild:
  - Command: `./scripts/rebuild_paper.sh`
  - Result: passed; regenerated ELARA and MVTec3D assets and wrote a 16-page
    `output/pdf/PAPER_DRAFT_v1.pdf`.
- Existing PDFs:
  - `output/pdf/PAPER_DRAFT_v1.pdf`: 16 pages.
  - `output/pdf/THESIS_CHAPTER_v1.pdf`: 23 pages.
- PDF visual rendering was not verified because `pdftoppm` is not installed in
  this environment.

## Venue Implication

Current state:

- Main conference: no.
- Symposium/workshop: not yet, unless submitted explicitly as an early
  prototype/negative-result abstract with limitations.
- Thesis chapter: salvageable, but it must clearly label current metrics as
  preliminary until C1/C2 are fixed.

After fixes C1-C2, clean build reproducibility, and reruns:

- IEEE CARS or a workshop/symposium is realistic.
- IEEE BigData main or similar mid-tier venues need the split-safe reruns plus a
  stronger MVTec feature baseline or a third naturally paired dataset.

## Required Fix Order

1. Fix artifact prefix mismatch and make a clean rebuild pass.
2. Rebuild RealFusion-LA with source-row-disjoint splits.
3. Rebuild MVTec with scorer-fit statistics learned only from allowed training
   normals and evaluation on an explicit held-out split.
4. Decide batch-level vs per-sample gating and regenerate results accordingly.
5. Replace or remove DeLong claims.
6. Delete/archive placeholder report CSVs.
7. Update configs, data README, experimental protocol, paper, and thesis to
   match the regenerated artifacts.
