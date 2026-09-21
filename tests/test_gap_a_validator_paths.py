"""The monolithic and chunked Gap A routes share the recorded acceptance rule."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def validator():
    path = Path(__file__).resolve().parents[1] / "docs/research/kbound/gapclose_wave5/val_gapA_radius.py"
    spec = importlib.util.spec_from_file_location("gap_a_validator_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def results():
    values = {"V0_baseZ": {"ratio80": 2.1}, "V1_latent_control": {"cov_lo": 0.9, "cov_hi": 0.9}}
    for variant in ("V1", "V2", "V3"):
        values[variant + "_augZ"] = dict(ratio80=1.2, fa_emp=0.01, fa_mc_se=0.01, cov_lo=0.9, cov_hi=0.9)
    for level, unweighted, weighted in [("moderate", 0.90, 0.92), ("mid", 0.80, 0.90), ("severe", 0.70, 0.80)]:
        values["_gmax_" + level] = "3"
        for variant, coverage in [("V1", unweighted), ("V4_oracle", weighted), ("V4_estim", weighted)]:
            values[f"{variant}_drift_{level}" if variant == "V1" else f"{variant}_{level}"] = {
                "cov_lo_by_group": {"3": coverage}
            }
    return values


def test_restoration_at_predeclared_mid_scale_counts(validator):
    outcome = validator.summarize(results())
    assert outcome["PASS"] is True
    assert outcome["checks"]["A3_weighted_restores"] is True
    assert outcome["drift_profile"]["moderate"]["V1_cov"] == 0.9
    assert outcome["drift_profile"]["mid"]["V1_cov"] == 0.8


def test_no_restoration_remains_negative(validator):
    values = results()
    values["V4_oracle_mid"]["cov_lo_by_group"]["3"] = 0.87
    outcome = validator.summarize(values)
    assert outcome["PASS"] is False
    assert outcome["checks"]["A3_weighted_restores"] is False
