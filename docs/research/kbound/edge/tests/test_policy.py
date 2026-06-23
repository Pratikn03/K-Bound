"""test_policy -- the kga_decide gate and the 6-policy family.

Core guarantee: lower>0 -> adapt; upper<0 -> freeze; interval crosses 0 -> abstain,
and kga_decide reuses kbound.certificate.decide verbatim.
"""

import pytest

from kbound_edge.policy import kga_decide, PolicyContext, apply_policy, POLICIES, Decision
from kbound_edge._bridge import decide as certificate_decide


class TestKgaDecide:
    def test_lower_positive_adapts(self):
        d = kga_decide(0.5, 0.1)
        assert isinstance(d, Decision)
        assert d.decision == "adapt"
        assert d.lower > 0

    def test_upper_negative_freezes(self):
        d = kga_decide(-0.5, 0.1)
        assert d.decision == "freeze"
        assert d.upper < 0

    def test_interval_crossing_zero_abstains(self):
        d = kga_decide(0.05, 0.2)
        assert d.decision == "abstain"
        assert d.lower < 0 < d.upper

    def test_boundary_is_abstain(self):
        # lower == 0 is NOT > 0 -> abstain (strict inequality in the certificate)
        assert kga_decide(0.10, 0.10).decision == "abstain"
        assert kga_decide(-0.10, 0.10).decision == "abstain"

    def test_interval_endpoints(self):
        d = kga_decide(0.2, 0.05)
        assert d.lower == pytest.approx(0.15)
        assert d.upper == pytest.approx(0.25)

    def test_matches_certificate_decide(self):
        for bhat, eps in [(0.3, 0.1), (-0.3, 0.1), (0.01, 0.5), (0.1, 0.1), (-0.1, 0.1)]:
            assert kga_decide(bhat, eps).decision == certificate_decide(bhat, eps)

    def test_reason_is_populated(self):
        for bhat, eps in [(0.5, 0.1), (-0.5, 0.1), (0.0, 0.2)]:
            assert isinstance(kga_decide(bhat, eps).reason, str) and kga_decide(bhat, eps).reason


class TestPolicyFamily:
    def test_exactly_six_policies(self):
        assert set(POLICIES) == {
            "always_freeze", "always_adapt", "confidence_gate",
            "entropy_gate", "kga_no_radius", "kga_full",
        }

    def test_always_policies(self):
        ctx = PolicyContext(bhat=0.0, eps=0.1, evidence={})
        assert apply_policy("always_freeze", ctx) == "freeze"
        assert apply_policy("always_adapt", ctx) == "adapt"

    def test_kga_full_matches_kga_decide(self):
        ctx = PolicyContext(bhat=0.5, eps=0.1, evidence={})
        assert apply_policy("kga_full", ctx) == kga_decide(0.5, 0.1).decision == "adapt"

    def test_kga_no_radius_is_sign_of_bhat(self):
        # eps ignored -> decision is the sign of bhat (no abstain band)
        assert apply_policy("kga_no_radius", PolicyContext(0.01, 5.0, {})) == "adapt"
        assert apply_policy("kga_no_radius", PolicyContext(-0.01, 5.0, {})) == "freeze"

    def test_confidence_gate_threshold(self):
        assert apply_policy("confidence_gate", PolicyContext(0.0, 0.1, {"post_conf": 0.9}, conf_tau=0.5)) == "adapt"
        assert apply_policy("confidence_gate", PolicyContext(0.0, 0.1, {"post_conf": 0.2}, conf_tau=0.5)) == "freeze"

    def test_entropy_gate_threshold(self):
        assert apply_policy("entropy_gate", PolicyContext(0.0, 0.1, {"entropy_drop": 0.2}, entropy_tau=0.05)) == "adapt"
        assert apply_policy("entropy_gate", PolicyContext(0.0, 0.1, {"entropy_drop": 0.0}, entropy_tau=0.05)) == "freeze"

    def test_unknown_policy_raises(self):
        with pytest.raises(KeyError):
            apply_policy("nope", PolicyContext(0.0, 0.1, {}))
