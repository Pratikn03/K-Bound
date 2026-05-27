# Family-D v2 — Independent Pre-Execution Review Sign-Off

**Reviewer role:** Independent pre-execution freeze reviewer.
**Review date (UTC):** 2026-05-25
**Review mode:** read-only audit; no model execution; no metric computation; no anomaly-mask inspection.
**Reviewer self-attestation:** acting under the explicit reviewer constraint that the protocol must not be modified, no Family-D model may be run, no test outcome may be inspected, and no manuscript may be edited during this review.

## 1. Files inspected

- `docs/research/phase2/FAMILY_D_PARTITION_MANIFEST_v2.json`
- `docs/research/phase2/FAMILY_D_V2_DATA_PROVENANCE_AND_HASH_REPORT.md`
- `docs/research/phase2/FAMILY_D_V2_SCHEMA_VERIFICATION_REPORT.md`
- `docs/research/phase2/FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md`
- `docs/research/phase2/FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md`
- `docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv`
- `docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md`
- `docs/research/phase2/FAMILY_D_V2_PRE_TEST_HOSTILE_REVIEW_REPORT.md`
- `docs/research/phase2/PHASE_2_READY_FOR_INDEPENDENT_FAMILY_D_REVIEW.md`
- `configs/phase2/family_d_v2_eyecandies_protocol.yaml`
- `experiments/phase2/family_d/eyecandies_schema_verification.json` (verified existence; not modified)
- `experiments/phase2/family_d/eyecandies_archive_sha256.txt` (verified existence; not modified)

## 2. Commit hash reviewed

**`09153cc`** — *Phase 2.2D: freeze ELARA Family D v2 Eyecandies contract before test evaluation*

## 3. Item-by-item verification (20 mandatory checks)

| # | Check | Status | Evidence |
|---:|---|:---:|---|
| 1 | Freeze commit exists and contains the frozen files | ✅ | `git log 09153cc` returns the freeze commit; all 10 freeze files present on `HEAD` |
| 2 | `FAMILY_D_PARTITION_MANIFEST_v2.json` exists | ✅ | direct file inspection |
| 3 | Manifest contains no placeholder values | ✅ | grepped for `TBD`, `TO_BE_FILLED`, `TO_BE_RECORDED`, `placeholder`, `unknown hash`, `planned later` — **zero hits** |
| 4 | All 10 Eyecandies category archive SHA256 values recorded | ✅ | 10/10 categories carry a valid 64-char hex SHA256 in the manifest's `archives` map; **live recomputation matches the manifest values for all 10 archives** |
| 5 | Schema report confirms local RGB / depth alignment + sample counts | ✅ | per-(category, split) sample counts read from `eyecandies_schema_verification.json`: train 1000 / val 100 / test_public 50 / test_private 400 per category; every sample has paired RGB + depth + 6 RGB views + 1 normal map across all 10 categories |
| 6 | Test outcome access remains false before execution | ✅ | manifest `prohibited_test_access_before_execution: true`; no Family-D inference / metric / ranking artifact exists under `experiments/phase2/family_d/` (only hash + schema files) |
| 7 | `test_evaluation_executed=false` in manifest and hypotheses | ✅ | manifest `test_evaluation_executed: false`; protocol YAML `invariants.test_evaluation_executed: false` AND `provenance.test_evaluation_executed: false`; every row of `FAMILY_D_HYPOTHESES_v2.csv` has `test_evaluation_executed = false` |
| 8 | Dataset is Eyecandies 1.0.3 | ✅ | manifest `dataset.name = "Eyecandies"`, `dataset.release_version = "1.0.3"` |
| 9 | Primary modalities are RGB + depth only; normal maps excluded from primary | ✅ | manifest `modalities.primary = ["rgb", "depth"]`, `excluded_from_primary = ["normal"]`; YAML same |
| 10 | Primary method is base RGA and comparator is fixed `static_attention` | ✅ | YAML `method.primary = "base_RGA"`, `method.comparator = "static_attention"` |
| 11 | Supervised RGA+ head selection is disabled | ✅ | YAML `method.rga_plus_supervised_head = "DISABLED"` |
| 12 | Operators frozen: D-EYE-1 depth (primary), D-EYE-2 RGB (primary), D-EYE-3 missingness (secondary descriptive) | ✅ | YAML `degradation_operators.primary_endpoints` = [D-EYE-1 depth, D-EYE-2 rgb]; `secondary_descriptive` = [D-EYE-3 missingness]; operator spec MD matches |
| 13 | Corruption applies before reliability computation and before gate decision | ✅ | Operator spec §1 and YAML §F define `transformation_level: modality_score_level` ("post anomaly-expert", before fusion / reliability / gate). This places corruption upstream of reliability weights and the gate. Verified consistent across all three operators. |
| 14 | Clean false-fire budget is fixed | ✅ | YAML `clean_false_fire_budget.value = 0.010`, `overrideable: false`, `measurement_split: validation` |
| 15 | Threshold selection is validation-only | ✅ | YAML `threshold_selection.rule = "validation_only"`; permitted inputs limited to normal-only validation + frozen degradation; forbidden inputs include official anomalous test labels and test-fold metrics |
| 16 | Official test labels/masks forbidden for selection; accessible only during authorised final held-out metric computation | ✅ | Selection policy §2 explicitly lists "Official anomalous test labels", "Test-fold metrics", and "Anomaly masks (under any split)" as forbidden; execution-commands file gates test-label access behind Stage 3 (one-time, post-sign-off) |
| 17 | Seed count, bootstrap rule, DeLong test, Holm K=2, practical-effect threshold frozen | ✅ | YAML: seeds target 30 / min_for_inference 15; bootstrap 10 000 iter seed 0; DeLong paired test enabled; multiplicity family `D-EYE-PRIMARY`, K=2, Holm–Bonferroni; practical Δ ≥ 0.010 |
| 18 | Primary and secondary endpoints not mixed | ✅ | Hypotheses CSV: D-EYE-1 and D-EYE-2 carry `multiplicity_family = D-EYE-PRIMARY-K2`; D-EYE-3 carries `DESCRIPTIVE_ONLY_NO_HOLM` and `primary_or_secondary = secondary`. Holm is applied across primary cells only. |
| 19 | Claim ceiling excludes universality, SOTA, deployment safety, retroactive Family-A confirmation, strongest-baseline superiority, raw-sensor robustness | ✅ | YAML `claim_ceiling.forbidden` enumerates all six items verbatim |
| 20 | No Family-D performance output already exists | ✅ | `experiments/phase2/family_d/` contains only the hash + schema verification + archive inventory files; no inference / Holm / prediction-archive artifacts present |

## 4. SHA256-anchor verification status

**MATCH on all 14 anchored hashes:**

- `protocol_yaml_sha256`: `104d90c6bab38671bb4dba15414a05ccebc890679cd681a5d46e06e7c8be4f15` — live recomputation matches manifest.
- `hypotheses_csv_sha256`: `0361a960217f0b32f9a96eef9c261d47af2877a895cbb5e10a0115e8303ad8e2` — live recomputation matches.
- `selection_policy_sha256`: `65f81a240b41e54fd7dafdbdf045f65d5e2d5c06909f0e05d9a56e286712e60b` — live recomputation matches.
- `operator_spec_sha256`: `e18bc05d12fb717b6b5dac738e41c022480374cefaa5556b5cfb48ce1e667f5d` — live recomputation matches.
- Per-archive SHA256 for all 10 Eyecandies category zips — live recomputation of the on-disk archives matches the manifest values **for every one of CandyCane, ChocolateCookie, ChocolatePraline, Confetto, GummyBear, HazelnutTruffle, LicoriceSandwich, Lollipop, Marshmallow, PeppermintCandy**.

No drift between recorded freeze hashes and live file contents.

## 5. No-placeholder verification status

**CLEAN.** Grep of the partition manifest for `TBD`, `TO_BE_FILLED`, `TO_BE_RECORDED`, `placeholder`, `unknown hash`, `planned later` returns zero matches. All required fields populated with concrete values.

## 6. No-prior-test-execution status

**CLEAN.**

- Manifest, protocol YAML (twice), and all hypothesis rows declare `test_evaluation_executed = false`.
- `experiments/phase2/family_d/` contains only the hash file, the schema verification JSON, and the archive inventory CSV. No inference / Holm / prediction-archive / metric file present.
- No file under `experiments/phase2/` or `docs/research/phase2/` carries any Family-D performance number, ROC-AUC, PR-AUC, F1, ECE, Brier, delta, p-value, or method ranking.

## 7. Protocol-integrity decision

**INTACT.**

- All 20 mandatory items verified against actual file contents (not summaries).
- All 14 SHA256 anchors match live file contents.
- Validation-only selection is enforced at the protocol level and the implementation pointer in the operator spec.
- The held-out invariant (test fold never read pre-execution) is enforced by manifest, YAML, hypothesis CSV, selection policy, and execution-commands NOT_RUN file in a mutually consistent way.

## 8. Claim-boundary decision

**CONSERVATIVE AND ACCEPTABLE.**

- Allowed positive claim ceiling: *"Held-out confirmatory evidence under the frozen Eyecandies RGB+depth one-class degradation-stress protocol for the evaluated endpoint(s)."* — appropriately narrow.
- Allowed negative claim ceiling: *"Held-out confirmation was not obtained for the evaluated endpoint(s); negative results are retained."* — appropriately bounded.
- Six forbidden claim categories enumerated verbatim in the protocol YAML and propagated to the hypothesis CSV's `allowed_positive_claim` / `allowed_negative_claim` columns.

## 9. Final authorisation decision

> **`FAMILY_D_V2_EXECUTION_AUTHORISED_UNDER_FROZEN_CONTRACT`**

**Authorised for one-time execution of D-EYE-1 and D-EYE-2 under the unchanged frozen contract. D-EYE-3 may be executed only as secondary descriptive evidence. No protocol change is authorised.**

## 10. Reviewer constraints

This sign-off authorises execution **only** under the following conditions:

1. The execution task **must not** modify any of the freeze artifacts (any modification re-invalidates this sign-off and requires a fresh independent review).
2. Selection (RGA reliability calibration, τ tuning) **must** use only the normal-only validation split and the three frozen degradation operators; no test-fold inspection during selection.
3. The official anomalous test split is read **exactly once** during the final held-out metric computation per cell; no iterative inspection.
4. The execution task **must** emit the per-seed prediction archive, the Holm K=2 inference, and the practical-effect band per cell; if any of these fails, the cell becomes `FAMILY_D_V2_INVALID`.
5. The execution task **must** verify all 14 SHA256 anchors via Stage 0.1 of `FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md` before any model training; if any anchor drifts, abort.
6. D-EYE-3 results may **not** be cited in the Holm K=2 family or in any confirmatory statement.
7. The post-execution hostile review (`FAMILY_D_V2_POST_TEST_HOSTILE_REVIEW_REPORT.md`) must be written by a reviewer (which **may** be the execution agent — independence is required pre-test, not post-test) but the family decision must follow the locked decision rules in `FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md` §5–§6 with no post-hoc redefinition.
8. The paper and thesis must **not** be edited during the execution task.

## 11. What this sign-off does not authorise

- Phase 3 / ELARA-Universal / ORIUS work.
- Manuscript editing (paper / thesis).
- Retroactive conversion of Family-A into confirmatory evidence regardless of Family-D outcome.
- Re-execution under a modified protocol (would require v3 freeze + fresh independent review).
- Execution of a different held-out dataset under this same freeze (this sign-off is Eyecandies-specific).

---

**Sign-off complete. Family-D v2 execution may proceed under the constraints above.**
