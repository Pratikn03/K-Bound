# Family-D v2 — Pre-Test Hostile Review Report

**Phase:** 2.2C / Step 7
**Reviewer posture:** held-out evaluation guardian + confirmatory-study protocol auditor.
**Reviewer identity:** Phase 2.2C audit pass (NOT the independent reviewer required before execution).

## 15 reviewer questions

### Q1. Is Eyecandies genuinely untouched by prior ELARA outcome inspection?

**YES.** `grep -rli "eyecandies"` finds only design / related-work references; no experiment CSV, prediction archive, inference table, or model run output references Eyecandies. No `data/raw/eyecandies/` directory exists. Verified in [FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md](./FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md) §2.

### Q2. Are RGB and depth officially documented and aligned?

**YES** at the documentary level. Per the official project page + `eyecandies==1.0.3` source code, every (sample_id, category, split) carries paired RGB + depth files. Byte-level on-disk alignment verification is **DEFERRED** to the future hash-only download pass — see [FAMILY_D_V2_SCHEMA_VERIFICATION_REPORT.md](./FAMILY_D_V2_SCHEMA_VERIFICATION_REPORT.md) §2.

### Q3. Are normal maps excluded from primary endpoints unless pre-registered?

**YES.** Protocol YAML §A names `[normal]` under `documented_but_excluded_from_primary`. Primary endpoints D-EYE-1 and D-EYE-2 use RGB+depth only. Any future inclusion of normal maps in a primary endpoint requires re-versioning to v3.

### Q4. Does the official split permit the selected one-class protocol?

**YES.** Official Eyecandies splits are anomaly-free train + anomaly-free validation + anomalous test. The selected `validation_only_degradation_calibrated_one_class_multimodal` protocol uses only the normal-only train and validation splits for fitting and calibration. The test split is held out and is read only at the one-time future execution. See protocol YAML §B.

### Q5. Does any method selection require true anomalous validation labels?

**NO.** Primary method is **base RGA** (reliability-aware gating under normal-only calibration). Supervised RGA+ head selection (router vs boost) is explicitly **disabled** (`rga_plus_supervised_head: DISABLED` in protocol YAML §C). Comparator is the fixed `static_attention` reference — no per-cell selection.

### Q6. Are all degradation operators frozen before test evaluation?

**YES.** Three operators (D-EYE-1 depth-score collapse; D-EYE-2 RGB-score collapse; D-EYE-3 single-modality missingness) are fully specified in [FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md](./FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md) §2–§3 with target modality, transformation level, operator function, parameter set, and PRNG seed policy. SHA256 of the spec is `e18bc05d12fb717b6b5dac738e41c022480374cefaa5556b5cfb48ce1e667f5d`.

### Q7. Is the clean false-fire budget fixed?

**YES.** `clean_false_fire_budget.value = 0.010` (≤ 1.0%); `measurement_split: validation`; `overrideable: false`. Aligned with the Phase-2 RGA-v2 false-fire budget convention. Recorded in protocol YAML §D.

### Q8. Is `static_attention` fixed as comparator?

**YES.** No per-cell selection. Documented in protocol YAML §C and hypotheses CSV (`comparator = static_attention` for every row).

### Q9. Is RGA, not supervised RGA+, the primary method?

**YES.** `primary: base_RGA`; `rga_plus_supervised_head: DISABLED`. No anomalous validation labels are admissible under the official Eyecandies one-class structure.

### Q10. Is the hypothesis family size fixed?

**YES.** `multiplicity.family = D-EYE-PRIMARY-K2`; `family_size_K = 2`. Holm-Bonferroni applies across exactly two primary cells (D-EYE-1 + D-EYE-2). The secondary D-EYE-3 is descriptive-only and is **not** in the Holm family.

### Q11. Are every required hash and release identifier recorded?

**PARTIAL.** Recorded:

- Eyecandies release tag `1.0.3` (per `eyecandies==1.0.3` package and GitHub Releases).
- Per-category official Drive file IDs (per `eyecandies/commands/download.py` package source).
- SHA256 of the four freeze design files:
  - `family_d_v2_eyecandies_protocol.yaml`: `104d90c6bab38671bb4dba15414a05ccebc890679cd681a5d46e06e7c8be4f15`
  - `FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md`: `e18bc05d12fb717b6b5dac738e41c022480374cefaa5556b5cfb48ce1e667f5d`
  - `FAMILY_D_HYPOTHESES_v2.csv`: `0361a960217f0b32f9a96eef9c261d47af2877a895cbb5e10a0115e8303ad8e2`
  - `FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md`: `65f81a240b41e54fd7dafdbdf045f65d5e2d5c06909f0e05d9a56e286712e60b`

NOT recorded (the explicit blocker):

- Per-category Eyecandies archive SHA256 — **RECORDED in Phase 2.2D** (10/10 categories hashed via `gdown` against the official `eyecandies==1.0.3` Drive file IDs). See `FAMILY_D_V2_DATA_PROVENANCE_AND_HASH_REPORT.md` §4 and `experiments/phase2/family_d/eyecandies_archive_sha256.txt`.

### Q12. Are all fields complete with no placeholders?

**YES (Phase 2.2D update).** `FAMILY_D_PARTITION_MANIFEST_v2.json` is now written with:
- Per-category real SHA256 + size_bytes (10/10 entries).
- Per-archive on-disk schema (rgb / depth / normal sample counts per split; mask file counts recorded but never opened).
- Protocol YAML SHA256, hypotheses CSV SHA256, selection policy SHA256, operator spec SHA256.
- `freeze_commit_hash`, `frozen_utc`.
- `test_evaluation_executed: false`.

No `TBD`, `TO_BE_FILLED`, `TO_BE_RECORDED`, `placeholder`, `unknown hash`, or `planned later` fields. Verified by [tests/test_family_d_v2_manifest_no_placeholders.py](../../../tests/test_family_d_v2_manifest_no_placeholders.py).

### Q13. Has any test outcome been accessed?

**NO.** Per [FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md](./FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md):
- No download performed.
- No anomaly mask inspected.
- No test ROC-AUC / PR-AUC / F1 / ECE / Brier / delta / p-value computed.
- No method-comparison ranking on test outcomes.

`test_evaluation_executed = false` is asserted in the protocol YAML and every row of the hypotheses CSV.

### Q14. Are allowed and forbidden claims conservative?

**YES.** Allowed positive claim ceiling is precisely "Held-out confirmatory evidence under the frozen Eyecandies RGB+depth one-class degradation-stress protocol." Forbidden list explicitly includes universality, SOTA, deployment-safety, Family-A retroactive confirmation, raw-sensor robustness, RGA+ supervised-baseline superiority. See protocol YAML §claim_ceiling and selection policy §7.

### Q15. Can an independent reviewer safely authorise one future execution?

**YES — pending independent external review.** Phase 2.2D closed items 1–3:

1. Complete partition manifest with real per-archive SHA256 — **DONE** (`FAMILY_D_PARTITION_MANIFEST_v2.json`).
2. Hash-only download pass — **DONE** (`src/scripts/family_d_v2_download_eyecandies.py`; 10/10 archives at `data/raw/eyecandies/_archives/`).
3. On-disk schema verification — **DONE** (`src/scripts/family_d_v2_schema_verify.py`; per-(category, split) RGB / depth / normal sample counts recorded; mask file presence COUNTED but NEVER opened; invariants pass: every train/val sample has paired RGB+depth + 6 RGB views; test_private has no public masks).
4. Independent external review (NOT the original Phase-2 agent) — **REQUIRED next**.

## Final pre-test verdict

> **`FAMILY_D_V2_VALID_FOR_PRE_TEST_FREEZE`**

**Phase 2.2D rationale:** All 15 reviewer questions now answer favourably. The partition manifest is real, no placeholders. The clean false-fire budget, the primary method, the comparator, the protocol, the operator spec, the hypothesis family, and the claim ceiling are all locked and recorded by SHA256. No test outcomes have been accessed.

The freeze is therefore **VALID for independent external review**. Family-D execution remains forbidden until an independent reviewer (NOT the original Phase-2 agent) signs off on the frozen artifacts; the sign-off file path is `docs/research/phase2/FAMILY_D_V2_INDEPENDENT_REVIEW_SIGNOFF.md` (per Stage 0.3 of `FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md`).
