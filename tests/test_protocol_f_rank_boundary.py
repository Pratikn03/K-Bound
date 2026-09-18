import math

import numpy as np
import pytest

from docs.research.kbound.scripts.analyze_F import conformal_rank_radius, decide_global


@pytest.mark.parametrize("n", [1, 8])
def test_protocol_f_infeasible_rank_abstains(n):
    radius = conformal_rank_radius(np.zeros(n), alpha=0.1)
    assert math.isinf(radius) and radius > 0
    assert decide_global(np.array([-1.0, 1.0]), radius).tolist() == ["ABSTAIN", "ABSTAIN"]


def test_protocol_f_preserves_signed_cqr_order_statistic():
    # Signed CQR scores are not absolute residuals: preserve their order statistic.
    assert conformal_rank_radius(np.arange(-9.0, 0.0), alpha=0.1) == -1.0
