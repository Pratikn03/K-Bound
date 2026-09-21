"""Data-free policy identities, not a natural-shift benchmark or calibration proof."""

import math

import numpy as np
import pytest

from kga.policy import decide_batch


@pytest.mark.parametrize("radius", [0.0, 0.01, 0.2, 1.0, math.inf])
def test_scalar_kga_is_exact_symmetric_fixed_margin(radius):
    predictions = [-1.0, -0.1, 0.0, 0.1, 1.0]
    if math.isfinite(radius):
        for boundary in (-radius, radius):
            predictions.extend([
                np.nextafter(boundary, -math.inf), boundary,
                np.nextafter(boundary, math.inf),
            ])
    expected = [
        "ADAPT" if h > radius else "FREEZE" if h < -radius else "ABSTAIN"
        for h in predictions
    ]
    assert decide_batch(predictions, radius).tolist() == expected


def test_zero_margin_matches_point_serving_but_not_commitment_count():
    predictions = np.array([-0.2, 0.0, 0.3])
    kga = decide_batch(predictions, 0.0)
    point = np.where(predictions > 0.0, "ADAPT", "FREEZE")
    assert kga.tolist() == ["FREEZE", "ABSTAIN", "ADAPT"]
    assert point.tolist() == ["FREEZE", "FREEZE", "ADAPT"]
    frozen = np.array([0.7, 0.8, 0.6])
    candidate = np.array([0.6, 0.5, 0.9])
    np.testing.assert_array_equal(
        np.where(kga == "ADAPT", candidate, frozen),
        np.where(point == "ADAPT", candidate, frozen),
    )
    assert np.count_nonzero(kga == "ADAPT") == np.count_nonzero(point == "ADAPT") == 1
    assert np.count_nonzero(kga != "ABSTAIN") == 2
    assert np.count_nonzero(point != "ABSTAIN") == 3
