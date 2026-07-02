"""Contract tests for kga.routing (Wave 4 theorem implementations)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kga.routing import (
    AnytimeMulticandidatePanel,
    bonferroni_multicandidate_route,
    multiclass_benefit,
    multiclass_harmful,
    route_panel,
    split_conformal_rank_radius,
)


def test_split_conformal_matches_validator_rank_form():
    rng = np.random.default_rng(0)
    errs = rng.standard_normal(80)
    level = 0.1
    n = len(errs)
    k = int(np.ceil((1 - level) * (n + 1)))
    expect = float(np.sort(np.abs(errs))[k - 1])
    assert split_conformal_rank_radius(errs, level) == pytest.approx(expect)


def test_multiclass_benefit_identity():
    assert multiclass_benefit(0.35, 0.7, 0.5) == pytest.approx(0.07)
    assert multiclass_harmful(-0.01, 0.4, 0.5, 0.35)


def test_bonferroni_route_abstains_when_all_lcbs_negative():
    assert bonferroni_multicandidate_route([-0.1, -0.2, 0.05], alpha=0.1) == 2
    assert bonferroni_multicandidate_route([-0.1, -0.2, -0.01], alpha=0.1) is None


def test_route_panel_fwer_synthetic():
    rng = np.random.default_rng(1)
    k, n_cal = 4, 60
    cal_truth = rng.uniform(-0.05, 0.25, (k, n_cal))
    cal_scores = cal_truth + 0.02 * rng.standard_normal((k, n_cal))
    deploy = cal_scores[:, -1] + 0.05
    deploy_scores = deploy
    cal_scores = cal_scores[:, :-1]
    cal_truth = cal_truth[:, :-1]
    dec = route_panel(deploy_scores, cal_scores, cal_truth, alpha=0.1)
    assert dec.bonferroni_alpha == pytest.approx(0.1 / k)
    assert dec.decision in ("adapt", "abstain")


def test_anytime_panel_bonferroni_threshold():
    panel = AnytimeMulticandidatePanel(4, alpha=0.1)
    rng = np.random.default_rng(2)
    hit = None
    for _ in range(50):
        benefits = [0.3 if i == 0 else 0.05 * rng.standard_normal() for i in range(4)]
        hit = panel.update(benefits)
        if hit is not None:
            break
    assert hit == 0


def test_routing_matches_vendored_copy():
    import importlib
    import sys
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1] / "docs/research/kbound/kbound_pkg"
    sys.path.insert(0, str(pkg))
    try:
        vend = importlib.import_module("kbound.routing")
    finally:
        sys.path.remove(str(pkg))
    rng = np.random.default_rng(3)
    errs = rng.standard_normal(40)
    assert vend.split_conformal_rank_radius(errs, 0.1) == split_conformal_rank_radius(errs, 0.1)
    assert vend.multiclass_benefit(0.2, 0.6, 0.4) == multiclass_benefit(0.2, 0.6, 0.4)
