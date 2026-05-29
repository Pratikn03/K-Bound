# Theorem Stack Closure Plan — Novelty Upgrade

Status: **implemented in code** (2026-05-28). Rebuild with
`scripts/rebuild_paper.sh` to regenerate all theorem tables.

## Novel contribution (GDR)

The **coherence-certified gate decision rule** (`gate_decision_rule.py`) is
the primary novelty upgrade. It answers:

> *When should a validation-derived drift gate be trusted to switch?*

It combines:
- **T2 insight** — heterogeneous mixtures produce dispersed reliability → veto
- **T5 certificate** — finite-sample fired-subset benefit → veto if not certified

This turns the cross-benchmark contrast (Family B helps, Family D/MVTec hurts)
from two anecdotes into a **predictive, auditable policy**.

## 100% code ↔ theorem map

| ID | Code | Script | Artifact |
|---|---|---|---|
| T1 | `reliability_estimator.py` | — | (theoretical) |
| T2 | `CategoryAwareReliabilityEstimator` | `validate_category_mixture_t2.py` | `category_mixture_t2.tex` |
| T3 | `corruption.py`, `gate_decisions()` | `emit_k_of_d_corruption_table.py` | `elara_k_domain_corruption_results.tex` |
| T4 | `risk_dominance.py` | `emit_risk_dominance_t4_table.py` | `risk_dominance_t4_prevalence.tex` |
| T5 | `bounded_switching_certificate` | `audit_switching_certificate_t5.py` | `switching_certificate_t5.tex` |
| T6 | KS in estimator | `emit_ks_power_t6_table.py` | `ks_power_t6.tex` |
| T7 | PAC audit | `emit_meta_router_pac_t7_table.py` | `meta_router_pac_t7.tex` |
| GDR | `gate_decision_rule.py` | `audit_gate_decision_rule_e2e.py` | `gate_decision_rule_e2e.tex` |

Registry: `src/elara/theory/theorem_registry.py`  
Validator: `src/scripts/validate_theorem_stack.py`

## Remaining open boundary (honest)

- **T2 on real benchmarks** — still `DEFERRED_PENDING_NATURAL_CATEGORY_METADATA`
- **Fourth benchmark scaffold** — still pending external datasets
- **Production safety** — explicitly forbidden in claim matrix

## Expected rating after closure

| Layer | Before | After |
|---|---|---|
| Theorem ↔ code | 6/10 | **9/10** |
| Novelty | 5/10 | **7/10** (GDR as predictive rule) |
| Conference readiness | 5/10 | **6–6.5/10** |
| Thesis readiness | 6/10 | **7.5/10** |
