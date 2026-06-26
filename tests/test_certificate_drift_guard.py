"""Drift guard: the paper's vendored certificate copy must stay numerically
identical to the canonical ``kga.certificate`` implementation.

The K-Bound certificate math is intentionally vendored in three places (the
maintained ``kga/`` core, the ELARA companion, and the frozen reproduction
package shipped with the paper at ``docs/research/kbound/kbound_pkg/``). This
test makes *drift* a build failure: if anyone edits the math in one copy without
the other, CI fails. It pins the single-source-of-truth invariant rather than
relying on a comment.

Pairing checked here: ``kga`` (canonical) vs ``kbound_pkg`` (paper artifact).
Note ``kga.empirical_bernstein`` carries an explicit ``benefit_range`` factor in
its range term; the frozen copy fixes that range to 1.0, so we compare with
``benefit_range=1.0`` to assert exact agreement.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KBOUND_PKG = REPO_ROOT / "docs" / "research" / "kbound" / "kbound_pkg"


@pytest.fixture(scope="module")
def vendored():
    """Import the frozen paper-reproduction certificate copy by path."""
    if not (KBOUND_PKG / "kbound" / "certificate.py").exists():
        pytest.skip("kbound_pkg reproduction copy not present")
    sys.path.insert(0, str(KBOUND_PKG))
    try:
        mod = importlib.import_module("kbound.certificate")
    finally:
        sys.path.remove(str(KBOUND_PKG))
    return mod


def test_empirical_bernstein_matches_vendored(vendored):
    from kga import certificate as kga_cert

    rng = np.random.default_rng(0)
    for n in (5, 25, 500):
        for alpha in (0.05, 0.1, 0.2):
            x = rng.uniform(-0.4, 0.6, size=n)
            kga_lower = kga_cert.empirical_bernstein(x, alpha=alpha, benefit_range=1.0).lower
            vend_lcb = vendored.empirical_bernstein_lcb(x, alpha=alpha)
            assert abs(kga_lower - vend_lcb) < 1e-9, (n, alpha, kga_lower, vend_lcb)


def test_conformal_radius_matches_vendored(vendored):
    from kga import certificate as kga_cert

    rng = np.random.default_rng(1)
    for alpha in (0.05, 0.1, 0.2):
        r = np.abs(rng.standard_normal(200))
        kga_eps = kga_cert.conformal_split(0.0, r, alpha=alpha).epsilon
        vend_eps = vendored.conformal_radius(r, alpha=alpha)
        assert abs(kga_eps - vend_eps) < 1e-12, (alpha, kga_eps, vend_eps)
