"""Adversarial input tests for the maintained K-Bound decision shim."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

DRIVER_DIR = Path(__file__).resolve().parents[1] / "docs" / "research" / "kbound" / "scripts"
sys.path.insert(0, str(DRIVER_DIR))
import kbound_decide as decision  # noqa: E402


def test_nonfinite_estimates_and_invalid_radii_abstain_per_cell() -> None:
    """Malformed scalar evidence must never fall through to a commitment."""

    observed = decision.decide(
        [float("inf"), float("-inf"), float("nan"), 0.2, -0.2],
        [0.1, 0.1, 0.1, -0.1, -0.1],
    )

    assert observed.tolist() == ["ABSTAIN"] * 5


def test_malformed_record_batch_abstains_including_excluded_cell() -> None:
    """A NaN benefit cannot certify its own cell merely because LOO excludes it."""

    b_hat = np.full(20, 0.5)
    benefit = np.full(20, 0.5)
    benefit[0] = np.nan

    epsilon, observed = decision.decide_from_records(b_hat, benefit)

    assert np.all(np.isinf(epsilon))
    assert observed.tolist() == ["ABSTAIN"] * 20


def test_nonfinite_calibration_pool_has_no_finite_radius() -> None:
    """NaN residuals are missing evidence, not sortable zero-width evidence."""

    radius = decision.conformal_radius([0.0] * 18 + [float("nan")])

    assert np.isinf(radius)


def test_loo_pool_with_any_missing_residual_abstains_globally() -> None:
    residuals = np.zeros(20)
    residuals[0] = np.nan

    radii = decision.radii_loo(residuals)

    assert np.all(np.isinf(radii))


def test_insufficient_model_fit_batch_abstains_without_fitting() -> None:
    """A one-cell wrapper call retains the frozen model instead of crashing."""

    b_hat, epsilon, observed = decision.decide_kga(
        np.asarray([[1.0, 2.0]]),
        np.asarray([0.25]),
    )

    assert np.isnan(b_hat).all()
    assert np.all(np.isinf(epsilon))
    assert observed.tolist() == ["ABSTAIN"]


@pytest.mark.parametrize("alpha", [0.0, 1.0, float("nan")])
def test_invalid_alpha_is_rejected_before_decision(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        decision.decide([0.2], [0.1], alpha=alpha)


def test_available_library_radius_failure_cannot_fall_through_to_local_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken imported backend is unavailable evidence, not permission to route."""

    def broken_radius(*_args: object, **_kwargs: object) -> float:
        raise RuntimeError("backend failure")

    monkeypatch.setattr(decision, "_kga_radius", broken_radius)

    assert np.isinf(decision.conformal_radius(np.arange(20, dtype=float)))


def test_available_library_decision_failure_abstains_instead_of_using_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An integration exception must never become a locally certified action."""

    class CertificateStub:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def broken_decision(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("backend failure")

    monkeypatch.setattr(decision, "_Certificate", CertificateStub)
    monkeypatch.setattr(decision, "_kga_decide", broken_decision)

    observed = decision.decide([1.0, -1.0], [0.1, 0.1])

    assert observed.tolist() == ["ABSTAIN", "ABSTAIN"]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"records": [], "records": [{"decision": "ADAPT"}]}',
        b'{"benefit": NaN}',
        b'{"benefit": Infinity}',
    ],
)
def test_json_boundary_rejects_duplicate_keys_and_nonfinite_constants(tmp_path: Path, payload: bytes) -> None:
    artifact = tmp_path / "unsafe.json"
    artifact.write_bytes(payload)

    with pytest.raises(ValueError):
        decision.read_json(artifact)


def test_records_boundary_requires_an_explicit_list(tmp_path: Path) -> None:
    artifact = tmp_path / "wrong-shape.json"
    artifact.write_text('{"status": "PASS"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="records"):
        decision.records(artifact)


def test_false_adapt_rejects_missing_or_misaligned_outcomes() -> None:
    with pytest.raises(ValueError, match="finite"):
        decision.false_adapt(["ADAPT"], [float("nan")])
    with pytest.raises(ValueError, match="matching"):
        decision.false_adapt(["ADAPT", "FREEZE"], [0.1])
    with pytest.raises(ValueError, match="decision"):
        decision.false_adapt(["UNKNOWN"], [0.1])
