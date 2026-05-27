# Family-D v2 — Execution Commands (NOT RUN)

> **STATUS: NOT RUN**
> **FAMILY-D LOCKED-TEST EVALUATION HAS NOT BEEN EXECUTED.**
> **RUN ONLY AFTER INDEPENDENT REVIEW OF THE FROZEN v2 CONTRACT.**

These commands are recorded for the future Phase 2.2D (or later) task that may execute Family-D after independent review. **Do not run any command in this file inside Phase 2.2C.**

## Stage 0 — Pre-flight (must run before any other stage)

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
source .venv/bin/activate

# 0.1 Verify the freeze SHA256 anchors match what is recorded in
# FAMILY_D_PARTITION_MANIFEST_v2.json. If any anchor drifts, STOP.
shasum -a 256 \
  configs/phase2/family_d_v2_eyecandies_protocol.yaml \
  docs/research/phase2/FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md \
  docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv \
  docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md

# 0.2 Confirm Family-D state is still pre-test
grep "test_evaluation_executed" docs/research/phase2/FAMILY_D_PARTITION_MANIFEST_v2.json
# Must show: "test_evaluation_executed": false

# 0.3 Confirm independent review sign-off exists.
ls docs/research/phase2/FAMILY_D_V2_INDEPENDENT_REVIEW_SIGNOFF.md
# If file absent — STOP. Execution is not authorised.
```

## Stage 1 — Hash-only download (one-time, no model evaluation)

```bash
# Download every Eyecandies category to data/raw/eyecandies/.
# This step is hash-only. It downloads archives, computes SHA256,
# verifies modality alignment, and counts samples. It does NOT
# inspect anomaly labels and does NOT compute any model output.

.venv/bin/eyec ec-get +o data/raw/eyecandies
shasum -a 256 data/raw/eyecandies/*.zip \
    > experiments/phase2/family_d/eyecandies_archive_sha256.txt

# Update FAMILY_D_PARTITION_MANIFEST_v2.json with the recorded hashes;
# this update happens in a SEPARATE commit before any model run.
```

## Stage 2 — Modality + schema verification (no labels read)

```bash
# Verify: for every (category, split) the per-sample directory contains
# both rgb_*.png and depth.png; train/val carry NO anomaly mask files;
# sample-ID uniqueness within split.
PYTHONPATH=src .venv/bin/python src/scripts/run_family_d_v2_schema_verify.py
# Output: docs/research/phase2/FAMILY_D_V2_SCHEMA_VERIFICATION_REPORT.md (updated)
```

## Stage 3 — Family-D v2 execution (one-time, after independent review)

```bash
# This is the one-time held-out evaluation. It must run AFTER:
#   - Stage 0 pre-flight passes;
#   - independent review sign-off exists;
#   - hash recording is committed;
#   - protocol YAML SHA256 matches manifest.

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_d_v2_cell.py \
    --cell D-EYE-1 --seeds 30 --seed-start 42 \
    --protocol configs/phase2/family_d_v2_eyecandies_protocol.yaml

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_d_v2_cell.py \
    --cell D-EYE-2 --seeds 30 --seed-start 42 \
    --protocol configs/phase2/family_d_v2_eyecandies_protocol.yaml

# Optional secondary descriptive:
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_d_v2_cell.py \
    --cell D-EYE-3 --seeds 15 --seed-start 42 \
    --protocol configs/phase2/family_d_v2_eyecandies_protocol.yaml
```

## Stage 4 — Inference + Holm K=2

```bash
# Compute seed-averaged DeLong + paired bootstrap CI per primary
# endpoint, then apply Holm K=2 across {D-EYE-1, D-EYE-2}.
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_d_v2_inference.py \
    --hypotheses docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv \
    --policy docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md

# Outputs:
#   experiments/phase2/family_d/family_d_v2_primary_inference.csv
#   experiments/phase2/family_d/family_d_v2_holm_k2.csv
```

## Stage 5 — Hostile review + final decision

```bash
# Independent hostile-review report (separate reviewer; NOT the
# original Phase 2 agent).
# Output: docs/research/phase2/FAMILY_D_V2_POST_TEST_HOSTILE_REVIEW_REPORT.md
# Final family decision: FAMILY_D_V2_CONFIRMED_BOTH_ENDPOINTS /
#                         FAMILY_D_V2_PARTIAL_CONFIRMATION /
#                         FAMILY_D_V2_NOT_CONFIRMED /
#                         FAMILY_D_V2_INVALID.
```

## Stage 6 — Commit

```bash
git add experiments/phase2/family_d/ \
        docs/research/phase2/FAMILY_D_V2_POST_TEST_HOSTILE_REVIEW_REPORT.md \
        docs/research/phase2/FAMILY_D_V2_FINAL_DECISION.md
git commit -m "Execute ELARA Family D v2 held-out evaluation under frozen contract"
```

## Forbidden operations (never run in any stage)

- Read Eyecandies test anomaly masks before Stage 3.
- Compute any test ROC-AUC before Stage 4 inference (and only via the inference script).
- Change the protocol YAML, operator spec, hypotheses CSV, or selection policy after the freeze SHA256 anchors are recorded.
- Re-tune τ in response to a test outcome.
- Run any Stage 3+ command before independent review sign-off.
- Edit paper or thesis based on Family-D v2 outcome inside this execution task.
