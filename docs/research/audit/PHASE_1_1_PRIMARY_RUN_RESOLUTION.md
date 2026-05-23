# Phase 1.1 — Primary ELARA-Bench-LA Run Resolution

**Decision:** the **k-of-D corruption sweep at k=4 with the mean / hybrid gate** is the **PRIMARY** Family B mechanism endpoint result for B1 (zero-attack all-domain at τ=0.66) and B2 (max-attack all-domain at τ=0.66). The standard adversarial table (`table_3_adversarial` from `craf_real_results.json`) is the **SECONDARY** descriptive surface.

This decision is locked by Phase 1.1 and supersedes any prior implicit mixing of the two regimes.

---

## 1. Evidence table

| field | RUN A (PRIMARY) | RUN B (SECONDARY) |
|---|---|---|
| run_id | `elara_bench_la_k_of_d_k4_mean` | `elara_bench_la_table3_default_gate` |
| artifact_path | `experiments/fusion/craf_real_k_domain_results.json` (table_10_k_domain_corruption rows where `failed_domain_count = 4`) | `experiments/fusion/craf_real_results.json` (table_3_adversarial rows where `target_domain = "all"`) |
| generated_table | `docs/research/tables/elara_k_domain_corruption_results.tex` | `docs/research/tables/elara_adversarial_results.tex`, `docs/research/tables/elara_tau_sweep_results.tex` |
| config_path | `configs/attention_real_fusion.yaml` (with k_domain_corruption_values: [0,1,2,3,4]) | `configs/attention_real_fusion.yaml` |
| scorer_train_fraction | 0.05 (per `experiments/fusion/real_domain_fusion_metadata.json`) | 0.05 (same) |
| domains | fraud, cyber, behavior, text (D=4) | same |
| sample_count | 8 000 | same |
| seeds | 5 ([42..46]) | same |
| gate_threshold | τ=0.66 (mean / hybrid gate explicitly evaluated; also minimum gate evaluated) | τ=0.66 default RGA gate (mean gate inside `_predict_craf`) |
| zero_attack_all_static | 0.7212 | 0.8270 |
| zero_attack_all_rga | 0.7718 | 0.8637 |
| **zero_attack_all_delta** | **+0.0506** [0.0315, 0.0681] | +0.0367 [0.0029, 0.0607] |
| max_attack_all_static | 0.7401 | 0.7809 |
| max_attack_all_rga | 0.7720 | 0.8348 |
| **max_attack_all_delta** | **+0.0319** [0.0050, 0.0617] | +0.0538 [−0.0232, 0.1123] |
| analysis_family | Family B audited mechanism endpoint (B1, B2) | Family B4 / B5 descriptive surface |
| primary_or_secondary | **PRIMARY** | SECONDARY |
| permitted_claim | "Audited mechanism endpoint at locked τ=0.66 on ELARA-Bench-LA under k-of-D coherent collapse at k=4 produces a ROC-AUC delta of +0.0506 (zero-attack all) and +0.0319 (max-attack all)" | Descriptive surface: at the standard adversarial-table protocol (not the k-of-D sweep) the RGA gate produces a +0.0367 / +0.0538 ROC-AUC delta; the difference vs PRIMARY is attributable to the different test protocol (gate-mode sweep vs default gate) |

## 2. Why these two runs report different numbers

Both runs use the same ELARA-Bench-LA dataset (`real_domain_fusion_inputs.csv`), the same scorer-train fraction (0.05), the same seeds, and the same τ=0.66 gate threshold. They differ in **which evaluation harness produces the all-domain coherent-collapse delta**:

- **Run A (PRIMARY)** evaluates the gate **per gate-mode** in `_evaluate_k_domain_corruption` (`run_breakthrough_experiment.py:1198+`). At k=4 it corrupts every domain in turn and runs the gate with each of `mean`, `minimum`, `hybrid` gate-modes explicitly. The mean / hybrid gate rows at k=4 are the audited mechanism endpoints B1/B2.
- **Run B (SECONDARY)** evaluates the gate via the default `_predict_craf` path on the standard adversarial sweep (`table_3_adversarial`). It uses the runtime-default gate decoder.

The two harnesses produce different static-AUC baselines on the same test fold (0.7212 vs 0.8270 for zero-attack-all; 0.7401 vs 0.7809 for max-attack-all), confirming they use distinct test-fold prediction paths. Both are valid descriptive views; **only Run A is the audited mechanism endpoint**.

## 3. Why Run A is PRIMARY (under the existing research contract)

1. The audited B1/B2 endpoints in `STATISTICAL_ANALYSIS_POLICY.md` §1 are defined as "all-domain coherent zero-attack at locked τ=0.66" and "all-domain coherent max-attack at locked τ=0.66". The k-of-D sweep at k=4 directly enumerates this case under controlled gate modes.
2. The abstract has cited +0.0506 / +0.0319 as the headline B1/B2 numbers since Tier-D closure; these are the values registered with the audited mechanism claim.
3. The k-of-D sweep is the same harness referenced by Theorem T3 (mean-gate dilution) in the thesis appendix; using its k=4 row as B1/B2 keeps the mechanism story and the theorem stack internally consistent.

Per Step 2 of Phase 1.1: *"Preserve the original registered ELARA-Bench-LA mechanism benchmark as the primary mechanism result if it was the pre-existing benchmark tied to the original B1/B2 claim."* Run A satisfies this preservation rule.

## 4. Permitted claim for each run

- **PRIMARY (Run A) — B1/B2:** "On ELARA-Bench-LA at locked τ=0.66, the audited mechanism endpoints are +0.0506 ROC-AUC (zero-attack all-domain) and +0.0319 ROC-AUC (max-attack all-domain). Both are derived from the k-of-D corruption sweep at k=4 under the mean / hybrid gate (`experiments/fusion/craf_real_k_domain_results.json`). 95 % CIs from the per-seed bootstrap are [0.0315, 0.0681] and [0.0050, 0.0617] respectively."
- **SECONDARY (Run B):** "A descriptive subsidiary table (`tables/elara_adversarial_results.tex`) reports the standard-protocol adversarial sweep with the default gate; under that protocol the all-domain deltas are +0.0367 and +0.0538. These numbers are NOT the audited B1/B2 endpoints and are reported as a Family B descriptive surface only."

The two pairs must NEVER be combined in a single headline sentence or table cell.

## 5. Manuscript locations requiring correction

| Location | Required action |
|---|---|
| Paper abstract + intro mentions of +0.0506 / +0.0319 | KEEP — already correct; cite source `tables/elara_k_domain_corruption_results.tex` (PRIMARY) |
| Paper master comparison table caption | Add: "Family B mechanism endpoints (+0.0506 / +0.0319) are the k-of-D k=4 mean-gate audited values; descriptive tau-sweep / adversarial subsidiary tables show different numbers under different protocols" |
| Paper §sec:elara-bench-la-mechanism subsection prose | Add explicit "PRIMARY" + "SECONDARY" framing |
| `tables/elara_tau_sweep_results.tex` table caption | Add "descriptive secondary table; NOT the audited B1/B2 endpoint" |
| `tables/elara_adversarial_results.tex` table caption | Add "descriptive secondary table" |
| Thesis abstract + mechanism section | Same as paper |
| Thesis theorem-stack appendix § T3 / T4 references | Keep the k-of-D sweep numbers (already consistent) |

## 6. Locked metrics-manifest entries

The Phase 1.1 metrics manifest will include:

```json
{
  "elara_bench_la_b1_zero_attack_all": {
    "delta_auc_mean": 0.0506,
    "delta_auc_ci_95": [0.0315, 0.0681],
    "source_artifact": "experiments/fusion/craf_real_k_domain_results.json",
    "source_artifact_row": "table_10_k_domain_corruption where attack=zero_attack AND failed_domain_count=4 AND gate_mode=mean",
    "analysis_family": "B",
    "analysis_status": "audited mechanism endpoint",
    "primary_or_secondary": "PRIMARY"
  },
  "elara_bench_la_b2_max_attack_all": {
    "delta_auc_mean": 0.0319,
    "delta_auc_ci_95": [0.0050, 0.0617],
    "source_artifact": "experiments/fusion/craf_real_k_domain_results.json",
    "source_artifact_row": "table_10_k_domain_corruption where attack=max_attack AND failed_domain_count=4 AND gate_mode=mean",
    "analysis_family": "B",
    "analysis_status": "audited mechanism endpoint",
    "primary_or_secondary": "PRIMARY"
  }
}
```

## 7. Test gate

A new test `tests/test_one_primary_elara_bench_la_story.py` must fail if any manuscript artifact uses Run B's +0.0367 / +0.0538 deltas as the headline B1/B2 endpoint (the manifest's locked-primary entry must always reference Run A).
