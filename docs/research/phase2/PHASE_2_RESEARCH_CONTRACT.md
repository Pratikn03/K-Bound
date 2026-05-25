# ELARA Phase 2 Research Contract

**Status:** locked before any new Phase-2 evaluation is run.
**Branch:** `exp/elara-phase2-mechanism-and-replication`
**Predecessor commit:** `6d381a9` (Phase 1.1.1 closure).
**Phase 1.1.1 verified PDF SHA256s:**
- Paper: `fb06129f618a9d8bd32a7865dbab0fdc7c838fd93536cb6080f71497ce04f5c5`
- Thesis: `81bc79c42b26844c88a10f6ba7d162fa18ccf6f4a379d3a584ea4fc06d2e36ab`

This contract supersedes any earlier ad-hoc Phase-2 sketches. Every Phase-2 result must trace back to a row in `PHASE_2_EXPERIMENT_REGISTRY.csv` and a claim in `PHASE_2_CLAIM_MATRIX.csv`.

---

## 1. Primary Phase-2 questions

- **Q1.** Can raw-prediction-archived reruns reproduce or revise the current audited public-benchmark story under stronger inference (ensemble DeLong + paired sample bootstrap)?
- **Q2.** Can an RGA-v2 gate improve isolated and partial failure behaviour (k=1, k=2, k=3) while staying within a locked clean false-fire budget?
- **Q3.** Does a category / cohort-aware reliability reference reduce false firing under legitimate mixture shift?
- **Q4.** When do risk-dominance and finite-sample switching certificates support using the reliability-aware path?
- **Q5.** What future unseen Family D evaluation is sufficiently clean to support confirmatory language?

## 2. Non-goals (explicitly out of scope for Phase 2)

- Universal ELARA claims.
- ORIUS integration.
- Real-world / live deployment validation.
- Top-paper headline framing.
- Changing the Phase-1 PRIMARY mechanism numbers (+0.0506 / +0.0319) to make them larger.
- Re-classifying any previously inspected cell as Family D.
- Any Family D test-evaluation execution in this task.

## 3. Mandatory terminology

| Forbidden | Permitted |
|---|---|
| "Family A confirmatory" | "Family A audited reproduction" / "powered audited reproduction" |
| "Family A pre-registered" | "audited reanalysis under locked Phase-2 policy" |
| "RGA+ beats every baseline" | "audited Δ vs validation-frozen primary comparator" |
| "ELARA is SOTA" / "production-ready" / "clinically validated" | "evaluated under defined stress / non-canonical / retrospective replay protocol" |
| "Solves distribution shift" | "reduces false firing under the evaluated pure mixture-shift controls" |
| "Confirmed cross-domain generalization" | "audited reanalysis across the inspected benchmark family" |

The words **"confirmatory"** and **"pre-registered"** are reserved exclusively for Family D — and only after Family D is independently reviewed and unfrozen for execution. In this Phase-2 task no Family D outcome is reported.

## 4. Locked Phase-2 statistical contract (summary)

Full text in `PHASE_2_STATISTICAL_POLICY.md`. Headline rules:

1. All method-head / comparator / threshold selection is validation-only.
2. No Fisher combination of seed-level p-values.
3. Every new Phase-2 run produces a raw per-seed test-prediction archive (Phase 2.B contract).
4. Primary inferential statistic per audited cell: paired DeLong on seed-averaged ensemble predictions plus paired bootstrap over test samples (10 000 iterations, fixed seed) for a 95% AUROC-Δ CI.
5. Practical-effect-size band is reported alongside every p-value.
6. Holm-Bonferroni applies inside the locked family registry; no implicit cross-family correction.

## 5. Compute / scope deviation declared up front

This contract acknowledges that running every Phase-2.C/D/E/F cell at the policy default of 30 seeds is **not feasible in a single interactive session**. The Phase 2 work in this branch therefore proceeds in two layers:

- **Layer 1 (this task):** all contracts + prediction-archive code + inference pipeline + Family-D contract freeze + one 30-seed pilot cell to validate the end-to-end pipeline.
- **Layer 2 (future compute-budgeted sessions):** the remaining Family A / Family B / Family E experimental runs, the mixture-shift / KS power sweeps, and the risk-dominance certificate audits across all registered cells.

This deviation is recorded in `PHASE_2_COMPUTE_PLAN.md`. Any Phase-2 cell whose Layer-2 run has not yet executed carries a `status = pending_compute` flag in the experiment registry and may not be cited as evidence in the manuscript until the run completes and its prediction archive is validated.

## 6. Commit cadence

1. Lock this contract + the four sibling files. (commit before any experiment runs.)
2. Land the prediction-archive infrastructure + tests.
3. Land the runner patch that calls the archiver.
4. Run the pilot, validate predictions, commit.
5. Lock the Family-D contract.
6. Land all reports + final commit.

No commit may mix manuscript claims with Family-D outcomes (Family-D execution is out of scope for this task).
