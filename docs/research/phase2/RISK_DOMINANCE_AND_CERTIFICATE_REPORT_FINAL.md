# Risk-Dominance and Retrospective Switching-Certificate Report — FINAL

**Phase:** 2.2B.2 / Step 5
**Status:** EXECUTED. Risk-dominance terms (q₀, q₁, Δ₀, Δ₁, π*) populated using paired clean (k=0) and degraded (k=4) arms from B-MECH-1.

> **Mandatory boundary text (verbatim):** "These are retrospective evaluation certificates under defined stress protocols; they are not production safety certificates or real-world deployment guarantees."

## 1. Source archives

| Arm | Method dirs |
|---|---|
| Clean (k=0) | `static_attention__clean_k0`, `rga_mean_gate_tau66__clean_k0` (each 30 seeds) — produced by `run_phase2_b_mech_1_clean_arm.py` |
| Degraded zero_attack k=4 (B1) | `static_attention__zero_attack_k4`, `rga_mean_gate_tau66__zero_attack_k4` (each 30 seeds) — B-MECH-1 original |
| Degraded max_attack k=4 (B2) | `static_attention__max_attack_k4`, `rga_mean_gate_tau66__max_attack_k4` (each 30 seeds) — B-MECH-1 original |

Seed-ensemble predictions: 30-seed pooled mean per arm per method. Fired-subset definition: `|rga_ensemble − static_ensemble| > 1e-6` (empirical-firing).

## 2. Risk-dominance terms

Source: `experiments/phase2/certification/risk_dominance_terms_v2.csv`.

| Endpoint | Scenario | q₀ (clean fire) | q₁ (degraded fire) | Δ₀ (clean cost of switching) | Δ₁ (degraded benefit of switching) | π* (indifference prevalence) | n_clean | n_deg |
|---|---|---:|---:|---:|---:|---|---:|---:|
| B1 | zero_attack k=4 | 0.0000 | 0.9994 | +0.0000 | **−0.0039** | undefined (Δ₁·q₁ < 0; numerator/denominator yields no admissible π*) | 1600 | 1600 |
| B2 | max_attack k=4 | 0.0000 | 1.0000 | +0.0000 | **+0.0095** | **0.0000** (gate dominates at any positive degradation prevalence) | 1600 | 1600 |

### Honest reading

- For **both** scenarios, q₀ = 0 (the empirical-firing definition gives zero clean firing because static and RGA predictions are identical to within 1e-6 on uncorrupted ELARA-Bench-LA features), and Δ₀ = 0 (no clean cost of switching when nobody switches).
- For **B1**, Δ₁ = −0.0039: under zero_attack k=4, the RGA path's per-sample bounded loss `|p − y|` is actually slightly **worse** than static on average. This is a per-sample local-cost statement; it is consistent with the global AUC delta being +0.0507 (because AUC is a global ranking metric, not a sum of per-sample losses).
- For **B2**, Δ₁ = +0.0095: under max_attack k=4, the RGA path's per-sample bounded loss is **better** than static.
- π* = (Δ₀·q₀) / (Δ₀·q₀ + Δ₁·q₁). With q₀ = 0 and Δ₀ = 0, the numerator is 0:
  - **B2**: denominator > 0 → π* = 0 → "at any positive degradation prevalence the gated policy dominates static in the modelled mixture."
  - **B1**: denominator < 0 → π* is **undefined** (the indifference prevalence does not exist on a positive-mixture interval; the gated policy is **inferior** under any positive degradation prevalence for B1's per-sample loss).

## 3. Certificate decisions

Source: `experiments/phase2/certification/switching_certificates_v2.csv`.

| Endpoint | Scenario | n_fired | Mean paired benefit | 95% paired-bootstrap LCB | Certified? |
|---|---|---:|---:|---:|:---:|
| B2 | max_attack k=4 | 1600 | +0.0095 | **+0.0085** | **CERTIFIED** |
| B1 | zero_attack k=4 | 1599 | −0.0039 | **−0.0050** | **NOT_CERTIFIED** |

Both rows carry the verbatim boundary notice in their `boundary_notice` column.

## 4. Bounded interpretation

- "Under max_attack k=4 on ELARA-Bench-LA, the G0 mean-gate at τ=0.66 yields a positive retrospective fired-subset paired-loss certificate (LCB = +0.0085) and dominates the static reference at any positive degradation prevalence in the modelled mixture (π* = 0)."
- "Under zero_attack k=4 on ELARA-Bench-LA, the same gate yields a negative retrospective fired-subset paired-loss certificate (LCB = −0.0050); the per-sample local-loss perspective contradicts the positive global AUC delta (+0.0507). The risk-dominance terms imply the gated policy is per-sample-loss-inferior under any positive degradation prevalence for this scenario."
- The two scenarios diverge: the gate is per-sample beneficial under max_attack but per-sample harmful under zero_attack. This is a real measurement; it is not averaged or hidden.

## 5. Forbidden claims

- "RGA gates yield production safety guarantees."
- "RGA gates are deployment-safe."
- "Certificates demonstrate real-world reliability."
- "Positive max_attack certificate implies positive zero_attack certificate."
- "Negative zero_attack certificate implies B1 should not be cited as reproduced." (AUC reproduction and per-sample certificate are two different surfaces; B1 AUC reproduction stands at +0.0507.)
- "Phase-2 closes Family B." (Family-B closure is the responsibility of `PHASE_2_FAMILY_B_FINAL_DECISION.md`; this report contributes evidence, not closure.)

## 6. Test guards

- [tests/test_phase2_risk_dominance_clean_arm.py](../../../tests/test_phase2_risk_dominance_clean_arm.py) — clean arm present + v2 CSVs populated with required columns + boundary notice.
- [tests/test_phase2_rga_v2_certificate_extension_boundary.py](../../../tests/test_phase2_rga_v2_certificate_extension_boundary.py) — only G0 rows admissible in v2 cert CSV until a candidate passes C1.

## 7. Reproduction

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_b_mech_1_clean_arm.py --seeds 30 --seed-start 42
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_b_cert_1_v2.py
```
