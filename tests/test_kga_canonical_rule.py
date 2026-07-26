"""Regression guards for the ONE canonical K-Bound decision rule.

These are the tests that would have caught the defects the review panel found:

* **item 4** -- the scored cell was in its own radius calibration pool, so
  ``epsilon`` was a function of the test label the ``FA_u <= alpha`` guarantee
  attaches to.  :func:`test_scored_index_is_excluded_from_its_own_pool` fails
  against the in-pool rule.
* **item 15** -- seven copy-pasted ``decide_kga`` forks produced every published
  number while the shipped ``kga`` package produced none, and the forks used a
  *different* radius rule from the library.
  :class:`TestLibraryAndDriverAgree` pins them together on a fixed synthetic
  input.
* **item 25** -- ``split_conformal_rank_radius`` clamped ``k`` to ``n`` and
  silently under-covered at small ``n``.  :class:`TestSmallNInfeasible` asserts
  ``+inf`` -> ABSTAIN at ``n in {3, 5, 8}`` and a finite radius at ``n = 9``.
* **item 26** -- the estimators defaulted to a data-estimated ``benefit_range``,
  voiding Maurer-Pontil / Hoeffding.  :class:`TestBenefitRangeRequired`.
* **item 29** -- the CLI could only ever print ABSTAIN.
  :class:`TestCliIsNotAConstantAbstainGenerator`.

Everything here is deterministic: fixed seeds, closed-form expectations, no
filesystem artifacts, no network, no torch.
"""

from __future__ import annotations

import io
import json
import math
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from kga import cli
from kga.certificate import (
    InsufficientCalibrationError,
    conformal_attained_level,
    conformal_radii_loo,
    conformal_split,
    empirical_bernstein,
    hoeffding,
    legacy_clamped_radius,
    min_calibration_size,
    split_conformal_rank_radius,
)
from kga.evidence import compute_evidence
from kga.policy import Decision, decide, decide_batch, decide_kga
from kga.routing import AnytimeMulticandidatePanel, route_panel

ALPHA = 0.1
REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# item 25 / F1-5 / F2-7 -- exact rank, and honest infeasibility at small n
# ---------------------------------------------------------------------------
class TestExactRankNotInterpolatedQuantile:
    """The radius is an observed order statistic, never ``np.quantile``.

    Mixing the two rules inside one claim is panel finding F1-2 / F4-10 / F5-2:
    the ImageNet-C SAR aggregate came from the exact rank while the per-seed
    table came from the interpolated quantile.
    """

    def test_radius_equals_the_exact_order_statistic(self):
        rng = np.random.default_rng(7)
        r = np.abs(rng.standard_normal(244))
        n = r.size
        k = int(math.ceil((n + 1) * (1.0 - ALPHA)))
        assert k <= n
        assert split_conformal_rank_radius(r, ALPHA) == float(np.sort(r)[k - 1])

    def test_radius_differs_from_np_quantile(self):
        """This assertion FAILS against ``np.quantile(r, 1 - alpha)``.

        Interpolation puts the radius strictly between two order statistics, so
        it is not attained by any residual.  On the archived CIFAR-10-C tent
        seed-0 pool the two rules differ in the 4th significant figure
        (0.024829934 interpolated vs 0.025266187 exact rank), which is the
        discrepancy the panel used to prove the drivers were not running the
        library.
        """
        rng = np.random.default_rng(11)
        r = np.abs(rng.standard_normal(244))
        exact = split_conformal_rank_radius(r, ALPHA)
        interpolated = float(np.quantile(r, 1.0 - ALPHA))
        assert exact != interpolated
        # The exact rank is the conservative one: it is the smallest order
        # statistic at or above the interpolated value.
        assert exact > interpolated
        assert exact in set(r.tolist())
        assert interpolated not in set(r.tolist())

    def test_rank_index_is_ceil_n_plus_1(self):
        """k = ceil((n+1)(1-alpha)), not ceil(n(1-alpha)) and not floor."""
        for n in (10, 19, 20, 27, 100, 432):
            r = np.arange(1.0, n + 1.0)  # order statistic i has value i
            eps = split_conformal_rank_radius(r, ALPHA)
            assert eps == float(math.ceil((n + 1) * (1.0 - ALPHA)))


class TestSmallNInfeasible:
    """``n < 1/alpha - 1`` admits no finite radius at level ``1 - alpha``."""

    def test_min_calibration_size(self):
        assert min_calibration_size(0.1) == 9
        assert min_calibration_size(0.05) == 19
        assert min_calibration_size(0.02) == 49

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_infeasible_n_returns_inf_and_abstains(self, n):
        r = np.abs(np.random.default_rng(n).standard_normal(n))
        with pytest.warns(UserWarning, match="needs n >="):
            eps = split_conformal_rank_radius(r, ALPHA)
        assert math.isinf(eps)
        with pytest.warns(UserWarning):
            cert = conformal_split(0.99, r, alpha=ALPHA)
        assert math.isinf(cert.epsilon)
        # A huge positive point estimate must still not commit.
        assert decide(cert, alpha=ALPHA) == Decision.ABSTAIN

    def test_n_nine_is_the_first_feasible_size(self):
        r = np.arange(1.0, 10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # must NOT warn at n = 9
            eps = split_conformal_rank_radius(r, ALPHA)
        assert eps == 9.0
        assert decide(conformal_split(20.0, r, alpha=ALPHA), alpha=ALPHA) == Decision.ADAPT

    def test_raise_mode(self):
        with pytest.raises(InsufficientCalibrationError):
            split_conformal_rank_radius(np.arange(1.0, 6.0), ALPHA, on_infeasible="raise")

    def test_clamp_is_not_a_mode_of_the_canonical_radius(self):
        """Defect D9: the clamp must not be selectable from the shipped rule."""
        with pytest.raises(ValueError, match="'inf' or 'raise'"):
            split_conformal_rank_radius(np.arange(1.0, 9.0), ALPHA, on_infeasible="clamp")

    def test_legacy_clamped_radius_is_separately_named_and_undercovering(self):
        """The superseded value is still computable -- but only under its own name."""
        r = np.arange(1.0, 9.0)  # n = 8
        assert legacy_clamped_radius(r, ALPHA) == 8.0
        assert conformal_attained_level(8, ALPHA) == pytest.approx(8 / 9)
        assert conformal_attained_level(8, ALPHA) < 1.0 - ALPHA

    def test_undercoverage_is_measurable(self):
        """Monte-Carlo: the clamped rule really does miss its nominal level.

        This is the measurement in F2-7 (empirical coverage 0.83 at n = 5,
        nominal 0.90), reproduced with a small deterministic replication count.
        """
        rng = np.random.default_rng(2024)
        n, trials = 5, 4000
        hits = 0
        for _ in range(trials):
            draws = np.abs(rng.standard_normal(n + 1))
            eps = legacy_clamped_radius(draws[:n], ALPHA)
            hits += int(draws[n] <= eps)
        coverage = hits / trials
        assert coverage < 1.0 - ALPHA, f"clamped rule covered {coverage:.4f}, expected < 0.90"
        # ... and n/(n+1) is where it lands.
        assert coverage == pytest.approx(n / (n + 1), abs=0.02)


class TestRoutePanelFeasibility:
    """``route_panel``'s Bonferroni level makes the small-n bug reachable."""

    def _panel(self, k, n_cal, seed=1):
        rng = np.random.default_rng(seed)
        cal_truth = rng.uniform(-0.05, 0.25, (k, n_cal))
        cal_scores = cal_truth + 0.02 * rng.standard_normal((k, n_cal))
        deploy = cal_scores[:, -1] + 0.05
        return deploy, cal_scores, cal_truth

    def test_k5_ncal10_does_not_commit(self):
        """F2-7 measured ``committed=True`` here at an unattainable level."""
        deploy, cs, ct = self._panel(5, 10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dec = route_panel(deploy, cs, ct, alpha=ALPHA)
        assert dec.bonferroni_alpha == pytest.approx(ALPHA / 5)
        assert dec.min_n_cal == 49
        assert dec.feasible is False
        assert dec.committed is False
        assert dec.selected is None
        assert dec.decision == "abstain"

    def test_k5_ncal60_is_feasible(self):
        deploy, cs, ct = self._panel(5, 60)
        dec = route_panel(deploy, cs, ct, alpha=ALPHA)
        assert dec.feasible is True
        assert all(math.isfinite(c.epsilon) for c in dec.certificates)


# ---------------------------------------------------------------------------
# item 4 -- the scored index is not in its own calibration pool
# ---------------------------------------------------------------------------
class TestLeaveOneOutOfPool:
    def test_scored_index_is_excluded_from_its_own_pool(self):
        """Each LOO radius equals the rank radius of the OTHER residuals.

        With ``n = 21`` distinct residuals ``1..21`` the in-pool rank is
        ``k = ceil(22 * 0.9) = 20``, so the shared radius is ``20.0``.  Dropping
        the worst cell leaves ``n = 20`` and ``k = 19``, so that cell is scored at
        ``19.0`` -- strictly tighter, and computed without ever looking at its own
        residual.
        """
        residuals = np.arange(1.0, 22.0)
        eps = conformal_radii_loo(residuals, ALPHA)
        assert eps.shape == residuals.shape
        for i in range(residuals.size):
            expected = split_conformal_rank_radius(np.delete(residuals, i), ALPHA)
            assert eps[i] == expected
        in_pool = split_conformal_rank_radius(residuals, ALPHA)
        assert in_pool == 20.0
        # The worst cell's own radius must not be inflated by itself.
        assert eps[-1] == 19.0
        assert eps[-1] < in_pool

    def test_loo_radius_never_sees_the_scored_residual(self):
        """Perturbing residual i changes eps[j] for j != i but never eps[i]."""
        base = np.abs(np.random.default_rng(3).standard_normal(30))
        eps_before = conformal_radii_loo(base, ALPHA)
        bumped = base.copy()
        bumped[0] = 1e3  # make cell 0's own residual enormous
        eps_after = conformal_radii_loo(bumped, ALPHA)
        assert eps_after[0] == eps_before[0], "cell 0's radius must ignore cell 0's residual"
        assert not np.allclose(eps_after[1:], eps_before[1:]), "other cells must see the change"

    def test_decide_kga_uses_loo_by_default(self):
        rng = np.random.default_rng(5)
        b_true = rng.normal(0.0, 0.1, 40)
        b_hat = b_true + rng.normal(0.0, 0.01, 40)
        eps_loo, _ = decide_kga(b_hat, b_true, alpha=ALPHA)
        eps_pool, _ = decide_kga(b_hat, b_true, alpha=ALPHA, calibration="in_pool")
        assert np.allclose(eps_loo, conformal_radii_loo(np.abs(b_hat - b_true), ALPHA))
        assert len(np.unique(eps_pool)) == 1, "the legacy rule is one shared radius"
        assert not np.allclose(eps_loo, eps_pool)

    def test_in_pool_radius_is_reachable_only_on_request(self):
        with pytest.raises(ValueError, match="calibration must be one of"):
            decide_kga([0.1, 0.2], [0.1, 0.2], calibration="whatever")


# ---------------------------------------------------------------------------
# item 15 -- one rule, one implementation
# ---------------------------------------------------------------------------
class TestTrichotomyTieBreaking:
    """Strict inequalities, matching the ``|M| > beta`` frontier convention."""

    def test_exact_zero_lower_bound_abstains(self):
        assert decide_batch([0.05], 0.05).tolist() == ["ABSTAIN"]
        assert decide_batch([-0.05], 0.05).tolist() == ["ABSTAIN"]
        assert decide_batch([0.0], 0.0).tolist() == ["ABSTAIN"]

    def test_strictly_past_the_boundary_commits(self):
        assert decide_batch([0.05 + 1e-9], 0.05).tolist() == ["ADAPT"]
        assert decide_batch([-0.05 - 1e-9], 0.05).tolist() == ["FREEZE"]

    def test_infinite_radius_always_abstains(self):
        assert decide_batch([1e6, -1e6], float("inf")).tolist() == ["ABSTAIN", "ABSTAIN"]

    def test_per_cell_radius_is_applied_elementwise(self):
        dec = decide_batch([0.2, 0.2, 0.2], [0.1, 0.3, float("inf")])
        assert dec.tolist() == ["ADAPT", "ABSTAIN", "ABSTAIN"]


class TestLibraryAndDriverAgree:
    """The guard that would have caught the seven ``decide_kga`` forks.

    ``docs/research/kbound/scripts/kbound_decide.py`` is the single driver-side
    entry point; it must produce bit-identical radii and decisions to
    :func:`kga.policy.decide_kga` on a fixed synthetic input.  If either side
    drifts -- a forked ``np.quantile``, a different clamp, a flipped tie-break --
    this fails.
    """

    @staticmethod
    def _driver():
        scripts = REPO / "docs" / "research" / "kbound" / "scripts"
        if not (scripts / "kbound_decide.py").exists():
            pytest.skip("driver module kbound_decide.py is not present in this release")
        sys.path.insert(0, str(scripts))
        try:
            import kbound_decide  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - reported, never swallowed
            pytest.skip(f"driver module kbound_decide.py is not importable: {exc!r}")
        finally:
            if str(scripts) in sys.path:
                sys.path.remove(str(scripts))
        return kbound_decide

    @staticmethod
    def _fixture(n=60, seed=0):
        rng = np.random.default_rng(seed)
        b_true = rng.normal(0.0, 0.08, n)
        b_hat = b_true + rng.normal(0.0, 0.02, n)
        return b_hat, b_true

    def test_driver_runs_on_the_library_backend(self):
        kb = self._driver()
        assert kb.backend() == "kga-library", (
            f"driver fell back to its local copy of the rule: {kb.backend()}"
        )

    def test_radii_and_decisions_match_on_fixed_input(self):
        kb = self._driver()
        b_hat, b_true = self._fixture()
        eps_lib, dec_lib = decide_kga(b_hat, b_true, alpha=kb.ALPHA)
        eps_drv, dec_drv = kb.decide_from_records(b_hat, b_true, alpha=kb.ALPHA, calibration="loo")
        assert np.array_equal(np.asarray(eps_lib), np.asarray(eps_drv, dtype=float))
        assert [str(d) for d in np.asarray(dec_lib).ravel()] == [str(d) for d in np.asarray(dec_drv).ravel()]

    def test_in_pool_replay_also_matches(self):
        kb = self._driver()
        b_hat, b_true = self._fixture(seed=1)
        eps_lib, dec_lib = decide_kga(b_hat, b_true, alpha=kb.ALPHA, calibration="in_pool")
        eps_drv, dec_drv = kb.decide_from_records(b_hat, b_true, alpha=kb.ALPHA, calibration="in_pool")
        assert np.allclose(np.asarray(eps_lib), np.asarray(eps_drv, dtype=float))
        assert [str(d) for d in np.asarray(dec_lib).ravel()] == [str(d) for d in np.asarray(dec_drv).ravel()]

    def test_driver_radius_helper_matches_the_library(self):
        kb = self._driver()
        rng = np.random.default_rng(9)
        r = np.abs(rng.standard_normal(27))
        assert kb.conformal_radius(r, kb.ALPHA) == split_conformal_rank_radius(r, kb.ALPHA)


# ---------------------------------------------------------------------------
# item 26 -- estimator defaults
# ---------------------------------------------------------------------------
class TestBenefitRangeRequired:
    """Maurer-Pontil and Hoeffding need an a-priori range, so there is no default."""

    @pytest.mark.parametrize("estimator", [empirical_bernstein, hoeffding])
    def test_omitting_benefit_range_raises(self, estimator):
        x = np.linspace(-0.5, 0.5, 50)
        with pytest.raises(TypeError):
            estimator(x, alpha=ALPHA)

    @pytest.mark.parametrize("estimator", [empirical_bernstein, hoeffding])
    def test_explicit_none_raises_with_an_explanation(self, estimator):
        x = np.linspace(-0.5, 0.5, 50)
        with pytest.raises(ValueError, match="benefit_range is required"):
            estimator(x, alpha=ALPHA, benefit_range=None)

    @pytest.mark.parametrize("estimator", [empirical_bernstein, hoeffding])
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_non_positive_or_non_finite_range_raises(self, estimator, bad):
        x = np.linspace(-0.5, 0.5, 50)
        with pytest.raises(ValueError, match="finite and > 0"):
            estimator(x, alpha=ALPHA, benefit_range=bad)

    def test_data_estimated_range_would_have_been_anti_conservative(self):
        """The removed default under-estimated R, hence under-sized epsilon.

        Concentrated benefits have an observed spread far below the true 0/1
        support width of 2.0, so the old default produced a strictly smaller
        radius than the honest one -- anti-conservative in exactly the direction
        that makes ADAPT easier to certify.
        """
        rng = np.random.default_rng(13)
        x = rng.normal(0.05, 0.01, 200)
        observed_range = float(x.max() - x.min())
        assert observed_range < 2.0
        eps_true = empirical_bernstein(x, alpha=ALPHA, benefit_range=2.0).epsilon
        eps_old = empirical_bernstein(x, alpha=ALPHA, benefit_range=observed_range).epsilon
        assert eps_old < eps_true


class TestSidednessIsLabelled:
    """``[lower, upper]`` is a ``1 - 2 alpha`` interval for the LCB estimators."""

    def test_ebern_and_hoeffding_report_one_sided_alpha_each(self):
        x = np.linspace(-0.4, 0.4, 100)
        for cert in (
            empirical_bernstein(x, alpha=ALPHA, benefit_range=2.0),
            hoeffding(x, alpha=ALPHA, benefit_range=2.0),
        ):
            assert cert.interval_level == pytest.approx(1.0 - 2.0 * ALPHA)

    def test_conformal_radius_is_genuinely_two_sided(self):
        r = np.abs(np.random.default_rng(17).standard_normal(100))
        cert = conformal_split(0.1, r, alpha=ALPHA)
        assert cert.interval_level == pytest.approx(1.0 - ALPHA)


# ---------------------------------------------------------------------------
# item 29 -- shipped-library defects with no paper impact
# ---------------------------------------------------------------------------
class TestCliIsNotAConstantAbstainGenerator:
    """``python -m kga decide`` used to be unable to print anything but ABSTAIN."""

    @staticmethod
    def _run(argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(argv)
        assert rc == 0
        return json.loads(buf.getvalue())

    def _benefits(self, tmp_path, p_positive, seed=0, n=400):
        rng = np.random.default_rng(seed)
        x = np.where(rng.random(n) < p_positive, 1.0, -1.0)
        path = tmp_path / f"benefits_{p_positive}_{seed}.npy"
        np.save(path, x)
        return str(path)

    def test_decide_can_adapt(self, tmp_path):
        out = self._run(["decide", "--benefits", self._benefits(tmp_path, 0.8), "--benefit-range", "2.0"])
        assert out["decision"] == "ADAPT"
        assert out["lower"] > 0

    def test_decide_can_freeze(self, tmp_path):
        out = self._run(["decide", "--benefits", self._benefits(tmp_path, 0.2), "--benefit-range", "2.0"])
        assert out["decision"] == "FREEZE"
        assert out["upper"] < 0

    def test_decide_can_abstain(self, tmp_path):
        out = self._run(["decide", "--benefits", self._benefits(tmp_path, 0.5), "--benefit-range", "2.0"])
        assert out["decision"] == "ABSTAIN"

    def test_decide_refuses_scores_only(self, tmp_path):
        rng = np.random.default_rng(1)
        calib = tmp_path / "c.npy"
        test = tmp_path / "t.npy"
        np.save(calib, rng.normal(0, 1, (200, 2)))
        np.save(test, rng.normal(3, 1, (200, 2)))
        with pytest.raises(SystemExit, match="exactly one of"):
            cli.main(["decide", "--calib", str(calib), "--test", str(test)])

    def test_decide_requires_benefit_range(self, tmp_path):
        with pytest.raises(SystemExit, match="--benefit-range is required"):
            cli.main(["decide", "--benefits", self._benefits(tmp_path, 0.8)])

    def test_evidence_subcommand_reports_z_only(self, tmp_path):
        rng = np.random.default_rng(2)
        calib = tmp_path / "c.npy"
        test = tmp_path / "t.npy"
        np.save(calib, rng.normal(0, 1, (300, 2)))
        np.save(test, rng.normal(2, 1, (300, 2)))
        out = self._run(["evidence", "--calib", str(calib), "--test", str(test)])
        assert "decision" not in out
        assert out["evidence"]["ks_mean"] > 0.3


class TestImportanceWeightDirection:
    """``ess_frac`` must match its documented weight (F2-11)."""

    def test_variance_shrink_reports_poor_overlap(self):
        rng = np.random.default_rng(21)
        calib = rng.normal(0.0, 3.0, 4000)
        test = rng.normal(0.0, 1.0, 4000)
        ess_frac = compute_evidence(calib, test).ess_frac
        assert ess_frac < 0.5, (
            f"variance-shrink shift reported ess_frac={ess_frac:.4f}; the reciprocal "
            "weight direction reports ~0.88 here and calls it 'good overlap'"
        )

    def test_no_shift_reports_full_ess(self):
        rng = np.random.default_rng(22)
        assert compute_evidence(rng.normal(0, 1, 4000), rng.normal(0, 1, 4000)).ess_frac > 0.9

    def test_mean_shift_reports_poor_overlap(self):
        rng = np.random.default_rng(23)
        assert compute_evidence(rng.normal(0, 1, 2000), rng.normal(4, 1, 2000)).ess_frac < 0.5


class TestAnytimePanelIngestsEveryStep:
    """All K e-processes see every step before any rejection is reported (F2-15)."""

    def test_all_processes_advance_on_every_update(self):
        panel = AnytimeMulticandidatePanel(4, alpha=ALPHA)
        for _ in range(40):
            panel.update([0.9, 0.9, 0.9, 0.9])
        counts = {p.cnt for p in panel._procs}
        assert counts == {40.0}, f"processes have unequal observation counts: {counts}"

    def test_returned_index_is_not_biased_to_the_lowest(self):
        """With candidate 3 the only strong one, the panel must return 3.

        Under the old early-return the loop still reached index 3, but any step
        on which an earlier candidate crossed first would have starved it; here
        we assert the wealth-ranked answer directly.
        """
        panel = AnytimeMulticandidatePanel(4, alpha=ALPHA)
        hit = None
        for _ in range(60):
            hit = panel.update([0.02, 0.02, 0.02, 0.6])
            if hit is not None:
                break
        assert hit == 3
        assert panel.crossed() == (3,)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            AnytimeMulticandidatePanel(3, alpha=ALPHA).update([0.1, 0.2])


# ---------------------------------------------------------------------------
# end-to-end: the canonical rule keeps its guarantee on synthetic data
# ---------------------------------------------------------------------------
def test_decide_kga_false_adapt_stays_at_or_below_alpha_under_h0():
    """No cell has a positive true benefit, so every ADAPT is a false adapt."""
    rng = np.random.default_rng(31)
    n_trials, n = 300, 60
    false_adapts = 0
    total = 0
    for _ in range(n_trials):
        b_true = rng.normal(-0.05, 0.05, n)  # Delta <= 0 in expectation
        b_hat = b_true + rng.normal(0.0, 0.05, n)
        _, dec = decide_kga(b_hat, b_true, alpha=ALPHA)
        adapt = np.asarray(dec) == "ADAPT"
        false_adapts += int(np.sum(adapt & (b_true <= 0.0)))
        total += n
    fa_u = false_adapts / total
    assert fa_u <= ALPHA, f"FA_u = {fa_u:.4f} exceeds alpha = {ALPHA}"


def test_decide_kga_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        decide_kga([0.1, 0.2, 0.3], [0.1, 0.2])
