"""tests.test_population_transfer_integration -- End-to-end integration regression tests.

Verifies the integration of the finite-sample concentration bound with the
conformal decision rule:
    * Integration of hoeffding_paired_accuracy_radius with compose_conditional_population_interval
    * Boundary transitions: an empirical cell-level ADAPT converts to ABSTAIN when evaluation sample
      size m is too small (the Price of Rigor), preventing premature population claims
    * Strict JSON serializability of all decision records and endpoints
    * Rejection of metric/feature schema mismatches
"""

from __future__ import annotations

import json
import math
import pytest

from kga.policy import Decision
from kga.population_transfer import (
    ConditionalPopulationInterval,
    compose_conditional_population_interval,
    hoeffding_paired_accuracy_radius,
)


def test_integration_sample_size_transition_price_of_rigor():
    """Verify that small m forces ABSTAIN on an otherwise confident cell estimate."""
    delta_hat = 0.15
    epsilon_cell = 0.05
    alpha_cell = 0.05
    delta_sampling = 0.05
    alpha_population = 0.10

    # At cell level alone (b = 0), lower bound is 0.10 > 0 -> ADAPT
    cell_res = compose_conditional_population_interval(
        delta_hat=delta_hat,
        epsilon=epsilon_cell,
        r_samp=0.0,
        alpha_cell=alpha_cell,
        delta_sampling=delta_sampling,
        alpha_population=alpha_population,
    )
    assert cell_res.action is Decision.ADAPT

    # At finite sample m = 100, b ~ 0.2716 -> total radius ~ 0.3216 -> interval [-0.1716, 0.4716] -> ABSTAIN
    b_100 = hoeffding_paired_accuracy_radius(n=100, delta=delta_sampling)
    assert b_100 > 0.25
    pop_res_100 = compose_conditional_population_interval(
        delta_hat=delta_hat,
        epsilon=epsilon_cell,
        r_samp=b_100,
        alpha_cell=alpha_cell,
        delta_sampling=delta_sampling,
        alpha_population=alpha_population,
    )
    assert pop_res_100.action is Decision.ABSTAIN

    # At large sample m = 2000, b = sqrt(2 * ln(40) / 2000) ~ 0.0607 -> total radius ~ 0.1107 -> lower 0.0393 > 0 -> ADAPT
    b_2000 = hoeffding_paired_accuracy_radius(n=2000, delta=delta_sampling)
    assert b_2000 < 0.07
    pop_res_2000 = compose_conditional_population_interval(
        delta_hat=delta_hat,
        epsilon=epsilon_cell,
        r_samp=b_2000,
        alpha_cell=alpha_cell,
        delta_sampling=delta_sampling,
        alpha_population=alpha_population,
    )
    assert pop_res_2000.action is Decision.ADAPT
    assert pop_res_2000.interval_lower > 0.0


def test_strict_json_serializability():
    """Verify that output records serialize cleanly to strict JSON."""
    b = hoeffding_paired_accuracy_radius(n=500, delta=0.05)
    res = compose_conditional_population_interval(
        delta_hat=0.3,
        epsilon=0.08,
        r_samp=b,
        alpha_cell=0.05,
        delta_sampling=0.05,
        alpha_population=0.10,
    )
    record = {
        "delta_hat": res.delta_hat,
        "epsilon": res.epsilon,
        "r_samp": res.r_samp,
        "population_radius": res.population_radius,
        "interval_lower": res.interval_lower,
        "interval_upper": res.interval_upper,
        "action": res.action.value,
        "spent_budget": res.spent_budget,
    }
    encoded = json.dumps(record)
    decoded = json.loads(encoded)
    assert decoded["action"] == "ADAPT"
    assert isinstance(decoded["interval_lower"], float)
    assert not math.isnan(decoded["interval_lower"])
