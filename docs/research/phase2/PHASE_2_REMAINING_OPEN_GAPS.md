# Phase 2 — Remaining Open Gaps

Every gap here is **known, named, and bounded**. Each line records what
is missing, why it is missing, and what closing it would entitle the
manuscript to say.

## G1 — A-POWERED-2..5 not executed

- **State:** 4 of 5 A-POWERED cells are `pending_compute`.
- **Why:** session scope = 1 pilot cell.
- **Closing this gap entitles:** the same Family-A audited statement for each cell. Does not entitle confirmatory language.
- **Cost estimate:** 4 cells × ~pilot duration each.

## G2 — Family B mechanism replication not executed

- **State:** B-MECH-1 (B1/B2 primary endpoints), B-MECH-2 (RGA-v2 partial-failure surface), B-MECH-3 (KS reference / mixture-shift control), B-MECH-4 (KS true-degradation power) are scaffolds with `pending_compute` rows.
- **Why:** session scope = scaffold only.
- **Closing this gap entitles:** an explicit `Reproduced` / `Directionally supported` / `Not reproduced` / `Inconclusive` label on the inherited B1/B2 endpoints, plus a documented RGA-v2 gate decision under [configs/phase2/rga_v2_gate_contract.yaml](../../../configs/phase2/rga_v2_gate_contract.yaml).
- **Compute window:** a single 30-seed run per protocol cell on the existing prediction archive; reuses the audited-inference path.

## G3 — Family D (held-out confirmatory) frozen but not executed

- **State:** Family-D contract is frozen at the SHA256 anchors recorded in [PHASE_2_ARTIFACT_MANIFEST.md](./PHASE_2_ARTIFACT_MANIFEST.md). No Family-D test split has been read.
- **Why:** explicit stop boundary — confirmatory weight requires that the freeze commit predates any test read.
- **Closing this gap entitles:** the phrase "confirmatory evidence on a held-out benchmark" in the manuscript abstract **iff** the CONFIRMED criteria in [FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md](./FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md) §4 are met.
- **What it does NOT entitle:** any of the forbidden claims preserved verbatim (universality, SOTA, deployment-readiness, clinical validation, broad cross-domain superiority, Real3D generalization).

## G4 — RGA-v2 promotion decision not made

- **State:** 5 candidate gates locked in [configs/phase2/rga_v2_gate_contract.yaml](../../../configs/phase2/rga_v2_gate_contract.yaml); zero executed.
- **Why:** session scope = contract only.
- **Closing this gap entitles:** one of the four pre-registered decisions — `PROMOTED_CANDIDATE`, `MECHANISM_IMPROVEMENT_PARTIAL`, `NOT_IMPROVED`, `INVALID_SELECTION` — to be recorded in the partial-failure report.
- **Hard rule:** no post-hoc gate tuning. The contract YAML is the only source of valid gate candidates and promotion criteria.

## G5 — Risk-dominance + switching certificate not run on archived predictions

- **State:** B-CERT-1 code + 4 tests passing; no execution against the A-POWERED-1 archive.
- **Why:** scope = code + tests only.
- **Closing this gap entitles:** a per-gate `(q0, q1, Δ0, Δ1, π*)` row and a CERTIFIED / NOT CERTIFIED label on the fired subset for the deployment window. **Does not** entitle a production-safety claim — the certificate is a retrospective evaluation surface on a defined stress protocol.

## G6 — Paper / thesis prose not yet under forbidden-claim integrity check

- **State:** forbidden claims are preserved verbatim in 4 markdown locations but no automated grep test guards the LaTeX sources.
- **Why:** scope ended before the test was written.
- **Closing this gap entitles:** a CI-level guarantee that no future commit can introduce verbatim forbidden text into [docs/research/PAPER_DRAFT_v1.tex](../../PAPER_DRAFT_v1.tex) or [docs/research/THESIS_CHAPTER_v1.tex](../../THESIS_CHAPTER_v1.tex).
- **Estimated effort:** ~20 lines of pytest plus a regex list.

## G7 — Single-trained-model variance band not surfaced in the report

- **State:** the per-seed RGA+ AUC vector (`per_seed_rga_aucs`) is archived in [experiments/phase2/statistics/family_a_powered_ensemble_inference.csv](../../../experiments/phase2/statistics/family_a_powered_ensemble_inference.csv) but the Family-A report only quotes the ensemble-predictor AUC.
- **Why:** scope = ensemble-predictor inferential statement only; the single-model band is descriptive.
- **Closing this gap entitles:** a "typical single trained model on this cell has AUC in [low, high] across seeds" sentence in the manuscript. Pure prose addition; no new computation.

## G8 — Phase 3 / ELARA-Universal / ORIUS

- **State:** untouched. Out of scope.
- **Why:** explicit stop boundary in the Phase-2 contract.
- **Closing this gap entitles:** nothing in the Phase-2 manuscript window. These are deliberately deferred until Family D returns.
