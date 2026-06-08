"""Theorem 1: identical Z-law + opposite benefit -> KGA must abstain (cannot certify)."""
import numpy as np
import cifar_tent_mps_v2 as K

def test_identical_evidence_forces_abstain():
    rng = np.random.default_rng(0); N = 120
    Z = rng.normal(size=(N, 5))                  # evidence independent of benefit sign
    B = np.where(np.arange(N) % 2 == 0, 0.4, -0.4)  # opposite, |B| equal
    Bhat, eps, dec = K.decide_kga(Z, B, alpha=0.10)
    abstain_frac = float(np.mean(dec == "ABSTAIN"))
    assert abstain_frac >= 0.9                    # uninformative Z -> band straddles 0
