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

- Per-category Eyecandies archive SHA256 — Eyecandies maintainers do not publish official hashes; recording requires a local download pass. Not performed in Phase 2.2C (see [FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md](./FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md)).

### Q12. Are all fields complete with no placeholders?

**NO** — and this is the freeze blocker. The Phase 2.2C spec at Step 5 forbids creating `FAMILY_D_PARTITION_MANIFEST_v2.json` with any of `TBD`, `TO_BE_FILLED`, `TO_BE_RECORDED`, `placeholder`, `unknown hash`, or `planned later`. The per-category archive SHA256 cannot be recorded without the download pass; therefore the partition manifest **cannot be written** in this task without violating that rule.

Therefore the partition manifest is **deliberately NOT created** in this task. Every other freeze-required artifact is complete with no placeholders.

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

**NOT YET.** Authorisation requires:

1. A complete partition manifest with real per-archive SHA256.
2. Hash-only download pass (Stage 1 of [FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md](./FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md)).
3. On-disk schema verification.
4. Independent external review (NOT the original Phase-2 agent).

Until items 1–3 land in a follow-up task, item 4 cannot meaningfully proceed.

## Final pre-test verdict

> **`FAMILY_D_V2_FREEZE_BLOCKED`**

**Reason:** the `FAMILY_D_PARTITION_MANIFEST_v2.json` cannot be written without placeholders because the Eyecandies maintainers do not publish per-archive SHA256 hashes and a local download pass was not performed in Phase 2.2C. Every other freeze artifact is complete and no-placeholder.

This is a clean, narrow blocker — not a contract validity issue. A follow-up Phase 2.2D task that performs only the hash-only download pass (Stage 1 + Stage 2 of [FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md](./FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md)) will unblock the freeze.
