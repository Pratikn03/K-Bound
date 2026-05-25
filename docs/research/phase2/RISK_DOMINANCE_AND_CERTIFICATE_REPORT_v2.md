# Risk-Dominance and Retrospective Switching-Certificate Report — v2

**Cell:** B-CERT-1
**Status:** **EXECUTED — partial certificates obtained on B-MECH-1 archives.**
**Mandatory boundary text (verbatim):** *"These are retrospective evaluation certificates under defined stress protocols; they are not production safety certificates or real-world deployment guarantees."*

## 1. Certificate results

Computed on the B-MECH-1 prediction archives (k=4 coherent collapse only; B-MECH-2 partial-failure surface was not executed). The "fired" subset is defined empirically as samples where the seed-averaged RGA ensemble prediction differs from the seed-averaged static ensemble prediction by more than 1e-6 — see §3 for the operational note on why this differs from the per-sample `gate_fired` flag written by B-MECH-1.

| Scenario | n_fired | mean paired benefit | paired-bootstrap LCB (α=0.05) | **CERTIFIED?** |
|---|---:|---:|---:|:---:|
| max_attack k=4 (B2) | 1 600 | **+0.0095** | **+0.0085** | **CERTIFIED** |
| zero_attack k=4 (B1) | 1 599 | -0.0039 | -0.0050 | NOT_CERTIFIED |

Source: [experiments/phase2/certification/switching_certificates.csv](../../../experiments/phase2/certification/switching_certificates.csv).

## 2. Honest split-result reading

**This is a split result.** The same RGA path that produces a positive AUC delta on both scenarios (B-MECH-1 §1: B1 Δ=+0.0507, B2 Δ=+0.0939) produces a **positive** per-sample paired-loss certificate on max_attack but a **negative** one on zero_attack. The mechanism is not contradictory; it reflects a real distinction:

- ROC-AUC is a global ranking metric. RGA can reorder borderline samples in a way that raises AUC without consistently reducing per-sample bounded loss `|p - y|`.
- The paired-bootstrap LCB on the fired subset tests the per-sample local benefit. On zero_attack k=4, the per-sample mean benefit is **-0.0039** — the RGA path slightly worsens the per-sample loss surrogate on average, even while the global ranking improves.

The honest claim is that the **max_attack k=4 RGA switch is certified** as a retrospective per-sample-paired-loss improvement under this defined stress protocol; the **zero_attack k=4 RGA switch is not certified** at the per-sample level even though its AUC delta is positive.

## 3. Risk-dominance terms (q₀, q₁, Δ₀, Δ₁, π*)

The classical risk-dominance terms require a **paired (clean, degraded)** evaluation: clean q₀ and Δ₀ are measured on the clean fold while degraded q₁ and Δ₁ are measured on the corrupted fold. The B-MECH-1 archives capture only the degraded arm (k=4) per scenario; the clean (k=0) arm was not archived. This makes (q₀, q₁, Δ₀, Δ₁, π*) inadmissible from the current archive set.

[experiments/phase2/certification/risk_dominance_terms.csv](../../../experiments/phase2/certification/risk_dominance_terms.csv) therefore contains a single row per scenario explaining that risk-dominance terms are not computable from the single-arm B-MECH-1 archive layout; the executable retrospective evidence is the fired-subset paired-loss certificate in §1.

Closing this gap requires extending the B-MECH-1 driver to also archive the clean (k=0) baseline, then re-running the certificate pipeline.

## 4. Operational note on the "fired" subset definition

The B-MECH-1 driver wrote `gate_fired = False` for every sample, because it consumed the runner's batch-level `adapted` flag (which under-reports per-sample firing under the existing runner's logic). Empirical evidence that the gate *did* effectively fire on most samples is the AUC delta itself (RGA ≠ static). B-CERT-1 therefore defines the fired subset empirically as samples where `|rga_ensemble - static_ensemble| > 1e-6`. On the B-MECH-1 archives this captures 1 599 / 1 600 samples per scenario (essentially every test sample) — consistent with a batch-level firing decision that affected the entire test pull.

The cleaner fix is to update the B-MECH-1 driver to write per-sample gate-fired vectors derived from `estimator.gate_decisions(...)` directly. This is a follow-up implementation note.

## 5. Forbidden claims (verbatim)

These certificates do **not** support:

- "RGA is a production safety guarantee."
- "RGA is a real-world deployment guarantee."
- "RGA is clinically validated."
- "RGA-v2 is promoted." (B-MECH-2 was not executed.)
- "All partial-failure regimes have a positive certificate." (Only k=4 was tested; k=1,2,3 were not.)
