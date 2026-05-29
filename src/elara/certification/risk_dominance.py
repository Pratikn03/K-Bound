"""Phase 2.G — risk-dominance term estimation.

For a given (gate, evaluation scenario) pair, estimate:

  q0       = P(gate fires | clean / non-degraded evaluation)
  q1       = P(gate fires | specified degraded evaluation)
  Delta_0  = expected cost of switching when static would have been right
  Delta_1  = expected benefit of switching under the specified degraded scenario
  pi_star  = degradation prevalence at which the gated policy is estimated
             to dominate static under the modelled operating mixture

The derivation follows the standard linear-mixture-of-costs analysis
(see thesis appendix T4). pi_star is the indifference threshold:

  E[L_static | mixture]  ==  E[L_gated | mixture]
  ⇒  pi * Delta_1 * q1  ==  (1 - pi) * Delta_0 * q0
  ⇒  pi_star  ==  (Delta_0 * q0) / (Delta_0 * q0 + Delta_1 * q1)

For pi > pi_star, the gated policy is preferred under the modelled mixture.

This is a retrospective-evaluation estimate. It is NOT a production safety
guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskDominanceTerms:
    gate_id: str
    scenario_id: str
    q0: float
    q1: float
    delta_0: float
    delta_1: float
    pi_star: float
    n_clean_samples: int
    n_degraded_samples: int
    notes: str


def _loss_proxy(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Bounded per-sample 0--1 loss surrogate `|p - y|`. Sufficient for the
    paired-difference statistic the certificate uses; not the deployed cost
    function in any real system."""
    return np.abs(y_prob.astype(np.float64) - y_true.astype(np.float64))


def estimate_risk_dominance(
    *,
    gate_id: str,
    scenario_id: str,
    clean_static_scores: np.ndarray,
    clean_gated_scores: np.ndarray,
    clean_gate_fired: np.ndarray,
    clean_labels: np.ndarray,
    degraded_static_scores: np.ndarray,
    degraded_gated_scores: np.ndarray,
    degraded_gate_fired: np.ndarray,
    degraded_labels: np.ndarray,
    notes: str = "",
) -> RiskDominanceTerms:
    """Estimate the five risk-dominance terms from paired clean and
    degraded prediction archives."""
    clean_gate_fired = clean_gate_fired.astype(bool)
    degraded_gate_fired = degraded_gate_fired.astype(bool)

    q0 = float(clean_gate_fired.mean()) if clean_gate_fired.size else float("nan")
    q1 = float(degraded_gate_fired.mean()) if degraded_gate_fired.size else float("nan")

    # Delta_0 = E[L_gated - L_static | clean fold, gate fired]
    if clean_gate_fired.any():
        l_static_c = _loss_proxy(clean_labels[clean_gate_fired], clean_static_scores[clean_gate_fired])
        l_gated_c = _loss_proxy(clean_labels[clean_gate_fired], clean_gated_scores[clean_gate_fired])
        delta_0 = float((l_gated_c - l_static_c).mean())
    else:
        delta_0 = 0.0

    # Delta_1 = E[L_static - L_gated | degraded fold, gate fired] (positive = benefit)
    if degraded_gate_fired.any():
        l_static_d = _loss_proxy(degraded_labels[degraded_gate_fired], degraded_static_scores[degraded_gate_fired])
        l_gated_d = _loss_proxy(degraded_labels[degraded_gate_fired], degraded_gated_scores[degraded_gate_fired])
        delta_1 = float((l_static_d - l_gated_d).mean())
    else:
        delta_1 = 0.0

    # Indifference prevalence pi*
    numer = delta_0 * q0
    denom = numer + delta_1 * q1
    if denom > 0:
        pi_star = float(numer / denom)
    else:
        pi_star = float("nan")

    return RiskDominanceTerms(
        gate_id=gate_id,
        scenario_id=scenario_id,
        q0=q0,
        q1=q1,
        delta_0=delta_0,
        delta_1=delta_1,
        pi_star=pi_star,
        n_clean_samples=int(clean_static_scores.shape[0]),
        n_degraded_samples=int(degraded_static_scores.shape[0]),
        notes=notes,
    )


def risk_dominance_margin(
    *,
    pi: float,
    q0: float,
    q1: float,
    delta_0: float,
    delta_1: float,
) -> float:
    """Signed margin for thesis T4: positive means gated policy dominates static.

    Margin = pi * q1 * Delta_1 - (1 - pi) * q0 * Delta_0.
    """
    return float(pi * q1 * delta_1 - (1.0 - pi) * q0 * delta_0)


def dominates_at_prevalence(
    *,
    pi: float,
    q0: float,
    q1: float,
    delta_0: float,
    delta_1: float,
) -> bool:
    """Return True when the T4 inequality holds at deployment prevalence pi."""
    margin = risk_dominance_margin(pi=pi, q0=q0, q1=q1, delta_0=delta_0, delta_1=delta_1)
    return bool(margin > 0.0)


def prevalence_sensitivity_rows(
    terms: RiskDominanceTerms,
    *,
    pi_values: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5),
) -> list[dict[str, float | str | bool]]:
    """Build deployment-prevalence sensitivity rows for one scenario."""
    rows: list[dict[str, float | str | bool]] = []
    for pi in pi_values:
        margin = risk_dominance_margin(
            pi=pi,
            q0=terms.q0,
            q1=terms.q1,
            delta_0=terms.delta_0,
            delta_1=terms.delta_1,
        )
        rows.append(
            {
                "gate_id": terms.gate_id,
                "scenario_id": terms.scenario_id,
                "pi": float(pi),
                "q0": terms.q0,
                "q1": terms.q1,
                "delta_0": terms.delta_0,
                "delta_1": terms.delta_1,
                "pi_star": terms.pi_star,
                "margin": margin,
                "dominates": dominates_at_prevalence(
                    pi=pi,
                    q0=terms.q0,
                    q1=terms.q1,
                    delta_0=terms.delta_0,
                    delta_1=terms.delta_1,
                ),
            }
        )
    return rows
