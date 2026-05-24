# Risk-Dominance + Switching-Certificate Report (B-CERT-1)

**Status:** code complete; execution **pending_compute** in this task.

---

## 1. Definitions

For a (gate, evaluation-scenario) pair we estimate the five terms in
the standard linear-mixture-of-costs analysis:

| symbol | meaning |
|---|---|
| q₀ | P(gate fires \| clean / non-degraded evaluation) |
| q₁ | P(gate fires \| specified degraded evaluation) |
| Δ₀ | expected cost of switching when the static path would have been right |
| Δ₁ | expected benefit of switching under the specified degraded scenario |
| π* | indifference prevalence: π* = (Δ₀·q₀) / (Δ₀·q₀ + Δ₁·q₁) |

For operating prevalence π > π*, the gated policy is preferred under the modelled mixture.

The finite-sample switching certificate works on the fired subset only:
- per-sample paired benefit `Xᵢ = L_static(i) - L_gated(i)` with the bounded loss surrogate `L(p, y) = |p - y|`;
- mean fired-case benefit `μ_F = mean(X)`;
- paired bootstrap (10 000 iterations, fixed seed) gives the lower confidence bound `LCB_α = quantile(boot, α)`;
- CERTIFIED iff `LCB_α > 0`.

## 2. Scope

Compute risk-dominance terms + the switching certificate for:

- **B1**: ELARA-Bench-LA k=4 mean-gate zero-attack at locked τ=0.66.
- **B2**: ELARA-Bench-LA k=4 mean-gate max-attack at locked τ=0.66.
- Each RGA-v2 partial-failure cell that meets the preliminary clean-budget criterion (Phase 2.E).
- Selected public-benchmark stress / reliability cells **only if** the protocol makes a per-sample fired-vs-not-fired certificate interpretation meaningful.

## 3. Code

- `src/elara/certification/risk_dominance.py` (implementation of the five-term estimator).
- `src/elara/certification/switching_certificate.py` (paired-bootstrap LCB on the fired subset).

Both modules are tested via the `tests/test_phase2_certification.py` (added in this stage).

## 4. Output schemas

| Path | Columns |
|---|---|
| `experiments/phase2/certification/risk_dominance_terms.csv` | gate_id, scenario_id, q0, q1, delta_0, delta_1, pi_star, n_clean_samples, n_degraded_samples, notes, status |
| `experiments/phase2/certification/switching_certificates.csv` | gate_id, scenario_id, n_fired_samples, mean_paired_benefit, bootstrap_lcb, alpha, n_iter, certified, notes, status |

## 5. Boundary

These certificates are **retrospective evaluation certificates under the defined stress protocols**. They are NOT:
- production safety certificates;
- clinical / physical deployment guarantees;
- evidence that ELARA is safe in any real-world operating mixture.

The boundary must be reiterated wherever a certificate is cited in prose.

## 6. Status

**pending_compute** until the B-MECH-1 (Phase 2.D) and B-MECH-2 (Phase 2.E) runs land and produce the per-sample fired-vs-not-fired prediction archives the certificate consumes.
