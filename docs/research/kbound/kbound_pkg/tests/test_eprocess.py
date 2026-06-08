"""Tests for kbound.eprocess (EProcess).

Anytime Thm (K-Bound paper / Ville's inequality):
    P(exists t >= 1: E_t^+ >= 1/alpha | Delta <= 0) <= alpha.

These tests validate:
  1. False-adapt rate <= alpha under H0 (Delta <= 0) on simulated streams.
  2. The e-process is a supermartingale under H0 (E[E_t^+] <= 1).
  3. Detection power > 0 under H1 (Delta > 0).
  4. API correctness (decision strings, wealth, reset).
"""

import math
import numpy as np
import pytest

from kbound.eprocess import EProcess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simulate_false_adapt_rate(
    delta: float,
    alpha: float = 0.10,
    horizon: int = 500,
    n_runs: int = 1000,
    seed: int = 0,
    stream: str = "twopoint",
) -> float:
    """Run n_runs independent EProcess instances for `horizon` steps.

    Returns fraction of runs in which ADAPT is ever decided (false-adapt rate
    when delta <= 0).
    """
    a, b = -1.0, 1.0
    rng = np.random.default_rng(seed)

    if stream == "twopoint":
        p_b = min(max((delta - a) / (b - a), 0.0), 1.0)
        X = np.where(rng.random((n_runs, horizon)) < p_b, b, a)
    elif stream == "uniform":
        # Shift uniform to have mean = delta
        lo = max(a, delta - 0.3)
        hi = min(b, delta + 0.3)
        X = rng.uniform(lo, hi, (n_runs, horizon))
    else:
        raise ValueError(f"Unknown stream: {stream}")

    crossed = np.zeros(n_runs, dtype=bool)
    for run_idx in range(n_runs):
        ep = EProcess(alpha=alpha, a=a, b=b)
        for t in range(horizon):
            ep.update(float(X[run_idx, t]))
            if ep.decision() == "adapt":
                crossed[run_idx] = True
                break   # first crossing counts

    return float(crossed.mean())


def simulate_mean_wealth(
    delta: float,
    alpha: float = 0.10,
    horizon: int = 300,
    n_runs: int = 500,
    seed: int = 0,
) -> dict:
    """Simulate E[E_t^+] at several time points. Returns {t: mean_wealth}."""
    rng = np.random.default_rng(seed)
    a, b = -1.0, 1.0
    p_b = min(max((delta - a) / (b - a), 0.0), 1.0)
    X = np.where(rng.random((n_runs, horizon)) < p_b, b, a)

    record_ts = [1, 10, 50, 100, 200, horizon]
    wealth_records = {t: [] for t in record_ts}

    for run_idx in range(n_runs):
        ep = EProcess(alpha=alpha, a=a, b=b)
        for t in range(1, horizon + 1):
            ep.update(float(X[run_idx, t - 1]))
            if t in wealth_records:
                wealth_records[t].append(ep.wealth_plus)

    return {t: float(np.mean(v)) for t, v in wealth_records.items()}


# ---------------------------------------------------------------------------
# Test: False-adapt rate under H0 (Anytime Thm / Ville's inequality)
# ---------------------------------------------------------------------------

class TestFalseAdaptRateH0:
    """EProcess controls false-adapt <= alpha under H0: Delta <= 0."""

    ALPHA = 0.10

    @pytest.mark.parametrize("delta,stream", [
        (0.0, "twopoint"),    # hardest H0 boundary
        (-0.10, "twopoint"),
        (-0.20, "twopoint"),
        (0.0, "uniform"),
    ])
    def test_false_adapt_le_alpha(self, delta, stream):
        """Anytime Thm: P(ever ADAPT | Delta <= 0) <= alpha.

        We allow a slack of 1.5 * 1/sqrt(n_runs) for Monte-Carlo noise.
        """
        n_runs = 800
        far = simulate_false_adapt_rate(
            delta=delta, alpha=self.ALPHA, horizon=400,
            n_runs=n_runs, seed=abs(int(delta * 100)), stream=stream
        )
        mc_slack = 1.5 / math.sqrt(n_runs)
        assert far <= self.ALPHA + mc_slack, (
            f"delta={delta}, stream={stream}: false-adapt rate {far:.4f} "
            f"> alpha + slack = {self.ALPHA + mc_slack:.4f} -- "
            "Anytime Thm violated"
        )


# ---------------------------------------------------------------------------
# Test: Supermartingale property (E[E_t^+] <= 1 under H0)
# ---------------------------------------------------------------------------

class TestSupermartingale:
    """Under H0 (Delta = 0), E[E_t^+] should stay <= 1 (within MC noise)."""

    def test_mean_wealth_le_1_under_h0(self):
        wealth_means = simulate_mean_wealth(delta=0.0, horizon=250, n_runs=600, seed=1)
        for t, mw in wealth_means.items():
            # Allow 3 * SE slack: SE ~ sqrt(Var[E_t^+] / n_runs), conservatively ~0.05
            assert mw <= 1.0 + 3 * 0.1, (
                f"E[E_t^+] at t={t} is {mw:.4f} > 1.3 -- "
                "supermartingale property violated"
            )


# ---------------------------------------------------------------------------
# Test: Detection power under H1
# ---------------------------------------------------------------------------

class TestDetectionPower:
    """EProcess should detect Delta > 0 with reasonable power."""

    @pytest.mark.parametrize("delta", [0.20, 0.40])
    def test_power_positive(self, delta):
        """With clear benefit, EProcess should detect it in many runs."""
        far = simulate_false_adapt_rate(
            delta=delta, alpha=0.10, horizon=300, n_runs=400, seed=42
        )
        # Power should be > 0 for Delta > 0
        assert far > 0.0, f"EProcess has zero power at delta={delta}"

    def test_higher_delta_higher_power(self):
        far_low = simulate_false_adapt_rate(delta=0.10, horizon=300, n_runs=400, seed=5)
        far_high = simulate_false_adapt_rate(delta=0.30, horizon=300, n_runs=400, seed=5)
        assert far_high >= far_low, "Higher delta should give at least as much power"


# ---------------------------------------------------------------------------
# Test: API correctness
# ---------------------------------------------------------------------------

class TestEProcessAPI:
    def test_initial_decision_abstain(self):
        ep = EProcess(alpha=0.1)
        assert ep.decision() == "abstain"
        assert ep.t == 0

    def test_initial_wealth_one(self):
        ep = EProcess(alpha=0.1)
        assert ep.wealth_plus == pytest.approx(1.0)
        assert ep.wealth_minus == pytest.approx(1.0)
        assert ep.wealth == pytest.approx(1.0)

    def test_update_increments_t(self):
        ep = EProcess(alpha=0.1)
        ep.update(0.1)
        assert ep.t == 1
        ep.update(0.2)
        assert ep.t == 2

    def test_update_batch_shape(self):
        ep = EProcess(alpha=0.1)
        ep.update_batch(np.array([0.1, 0.2, -0.05, 0.3]))
        assert ep.t == 4

    def test_decision_strings(self):
        ep = EProcess(alpha=0.1)
        d = ep.decision()
        assert d in ("adapt", "freeze", "abstain")

    def test_reset(self):
        ep = EProcess(alpha=0.1)
        for x in np.linspace(-0.5, 0.5, 20):
            ep.update(x)
        ep.reset()
        assert ep.t == 0
        assert ep.decision() == "abstain"
        assert ep.wealth_plus == pytest.approx(1.0)

    def test_freeze_decision_under_strong_h0prime(self):
        """Under Delta << 0, the E^- process should eventually cross."""
        ep = EProcess(alpha=0.05)
        rng = np.random.default_rng(77)
        # Delta = -0.5: strongly negative benefit -> E^- should cross
        for _ in range(500):
            x = rng.uniform(-0.9, -0.3)  # mean ~ -0.6
            ep.update(x)
            if ep.decision() == "freeze":
                break
        assert ep.decision() == "freeze", (
            "EProcess should have decided FREEZE on strongly negative benefit stream"
        )

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            EProcess(alpha=0.0)
        with pytest.raises(ValueError):
            EProcess(alpha=1.5)

    def test_invalid_a_nonneg(self):
        with pytest.raises(ValueError):
            EProcess(a=0.0, b=1.0)

    def test_invalid_b_nonpos(self):
        with pytest.raises(ValueError):
            EProcess(a=-1.0, b=0.0)

    def test_adapt_decision_under_strong_h1(self):
        """Under Delta > 0, the E^+ process should cross 1/alpha."""
        ep = EProcess(alpha=0.10)
        rng = np.random.default_rng(88)
        for _ in range(500):
            x = rng.uniform(0.3, 0.7)   # mean ~ 0.5 >> 0
            ep.update(x)
            if ep.decision() == "adapt":
                break
        assert ep.decision() == "adapt", (
            "EProcess should have decided ADAPT on strongly positive benefit stream"
        )
