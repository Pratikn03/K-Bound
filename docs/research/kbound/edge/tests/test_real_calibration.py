import pytest
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from kbound_edge.conformal import fit_real_certificate

@pytest.fixture
def calibration_bundle():
    # Z has shape (n, 14), B has shape (n,)
    Z_fit = np.random.normal(0.0, 1.0, (10, 14))
    B_fit = np.random.normal(0.1, 0.2, (10,))
    Z_conf = np.random.normal(0.0, 1.0, (8, 14))
    B_conf = np.random.normal(0.1, 0.2, (8,))
    
    return {
        "fit": {
            "Z": Z_fit,
            "B": B_fit,
            "sessions": ["S03", "S04"],
            "source_hashes": ["hash1", "hash2"]
        },
        "conformal": {
            "Z": Z_conf,
            "B": B_conf,
            "sessions": ["S05", "S06"],
            "source_hashes": ["hash3", "hash4"]
        }
    }

def test_estimator_and_radius_use_disjoint_sessions(calibration_bundle):
    result = fit_real_certificate(calibration_bundle)
    assert set(result.fit_sessions) == {"S03", "S04"}
    assert set(result.conformal_sessions) == {"S05", "S06"}
    assert set(result.fit_source_hashes).isdisjoint(result.conformal_source_hashes)
