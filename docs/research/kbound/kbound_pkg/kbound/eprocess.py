"""kbound.eprocess -- Anytime-valid betting e-process for adaptation gating.

Implements the anytime-valid complement to the batch empirical-Bernstein
certificate (Theorem thm:cert / Anytime Thm, K-Bound paper).

Theory (from val_thm3_evalue.py):
    For paired benefits X_i in [a, b] with E[X_i] = Delta, form

        E_t^+ = prod_{i=1}^t (1 + lambda_i * X_i),   E_0^+ = 1.

    With predictable (F_{i-1}-measurable) nonneg bets lambda_i,
    (E_t^+) is a supermartingale under H0: Delta <= 0.
    By Ville's inequality: P(exists t: E_t^+ >= 1/alpha) <= alpha.

    The symmetric E_t^- tests H0': Delta >= 0 using -X_i.

    Decision (anytime-valid):
        ADAPT   when E_t^+ >= 1/alpha  (reject Delta <= 0)
        FREEZE  when E_t^- >= 1/alpha  (reject Delta >= 0)
        ABSTAIN otherwise

Betting rule (truncated aGRAPA, Waudby-Smith & Ramdas 2024):
    lambda_t = clip(mu_hat_{t-1} / sigma2_hat_{t-1}, 0, bet_cap_frac/|a|)

where mu_hat and sigma2_hat are the running mean and variance of past X.
"""

from __future__ import annotations

import math

import numpy as np


class EProcess:
    """Online anytime-valid betting e-process for adaptation gating.

    Runs two one-sided e-processes simultaneously:
        E_t^+  tests H0:  Delta <= 0  (reject -> ADAPT)
        E_t^-  tests H0': Delta >= 0  (reject -> FREEZE)

    Implements the Anytime Thm (K-Bound paper) / Ville's inequality.

    Parameters
    ----------
    alpha : float, default=0.1
        Anytime type-I error budget.  P(ever ADAPT under H0) <= alpha.
    a : float, default=-1.0
        Lower bound of the benefit range [a, b].
    b : float, default=1.0
        Upper bound of the benefit range [a, b].
    bet_cap_frac : float, default=0.5
        Lambda bound: lambda in [0, bet_cap_frac/|a|].  Must be < 1 for
        strict positivity of wealth factors.
    prior_var : float, default=0.25
        Initial prior variance for the running second-moment estimate.
    prior_weight : float, default=1.0
        Prior pseudo-count for the variance estimate.

    Attributes
    ----------
    t : int
        Number of observations seen so far.
    wealth_plus : float
        Current value of E_t^+.
    wealth_minus : float
        Current value of E_t^-.
    wealth : float
        max(wealth_plus, wealth_minus) -- the dominant e-process.

    Examples
    --------
    >>> ep = EProcess(alpha=0.10)
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> for x in rng.uniform(0.0, 0.3, 200):
    ...     ep.update(x)
    ...     if ep.decision() != 'abstain':
    ...         break
    >>> ep.decision() in ('adapt', 'freeze', 'abstain')
    True
    """

    def __init__(
        self,
        alpha: float = 0.1,
        a: float = -1.0,
        b: float = 1.0,
        bet_cap_frac: float = 0.5,
        prior_var: float = 0.25,
        prior_weight: float = 1.0,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if a >= 0.0:
            raise ValueError(f"a must be < 0 for the one-sided test on H0: Delta<=0; got {a}")
        if b <= 0.0:
            raise ValueError(f"b must be > 0 for the one-sided test on H0': Delta>=0; got {b}")

        self.alpha = alpha
        self.a = a
        self.b = b
        self.bet_cap_frac = bet_cap_frac
        self._log_thr = math.log(1.0 / alpha)

        self._lam_max_plus = bet_cap_frac / (-a)    # for E^+ (bets on H0: Delta<=0)
        self._lam_max_minus = bet_cap_frac / b       # for E^- (bets on H0': Delta>=0)

        # Running stats (predictable -- updated AFTER each observation)
        self._s1 = 0.0          # sum X_j
        self._s1m = 0.0         # sum (-X_j)
        self._s2 = prior_weight * prior_var   # sum X_j^2 + prior (E^+)
        self._s2m = prior_weight * prior_var  # sum X_j^2 + prior (E^-)
        self._cnt = 0.0
        self._cnt_var = prior_weight

        self._log_w_plus = 0.0
        self._log_w_minus = 0.0
        self.t = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x: float) -> "EProcess":
        """Ingest one new benefit observation X_t.

        Parameters
        ----------
        x : float
            New benefit sample X_t in [a, b].  Values outside [a, b] are
            clamped with a warning (does not affect validity, only tightness).

        Returns
        -------
        self  (for chaining)
        """
        x = float(x)
        x_clamp = max(self.a, min(self.b, x))

        # Compute predictable bets from PAST statistics
        mu_hat = self._s1 / self._cnt if self._cnt > 0 else 0.0
        mu_hat_m = self._s1m / self._cnt if self._cnt > 0 else 0.0
        sig2 = self._s2 / self._cnt_var
        sig2m = self._s2m / self._cnt_var

        lam_plus = float(np.clip(mu_hat / sig2 if sig2 > 0 else 0.0, 0.0, self._lam_max_plus))
        lam_minus = float(np.clip(mu_hat_m / sig2m if sig2m > 0 else 0.0, 0.0, self._lam_max_minus))

        # Update wealth (log scale for numerical stability)
        factor_plus = max(1.0 + lam_plus * x_clamp, 1e-300)
        factor_minus = max(1.0 + lam_minus * (-x_clamp), 1e-300)
        self._log_w_plus += math.log(factor_plus)
        self._log_w_minus += math.log(factor_minus)

        # Update running stats (predictable: include current sample for NEXT step)
        self._s1 += x_clamp
        self._s1m += -x_clamp
        self._s2 += x_clamp ** 2
        self._s2m += x_clamp ** 2
        self._cnt += 1.0
        self._cnt_var += 1.0
        self.t += 1
        return self

    def update_batch(self, xs: np.ndarray) -> "EProcess":
        """Update with a sequence of observations (calls update in order).

        Parameters
        ----------
        xs : array-like of shape (n,)
        Returns
        -------
        self
        """
        for x in np.asarray(xs, dtype=float):
            self.update(float(x))
        return self

    def decision(self) -> str:
        """Current anytime-valid decision.

        Returns
        -------
        str : ``'adapt'``, ``'freeze'``, or ``'abstain'``.
            ADAPT  when E_t^+ >= 1/alpha (reject H0: Delta <= 0).
            FREEZE when E_t^- >= 1/alpha (reject H0': Delta >= 0).
            ABSTAIN while neither threshold is crossed.
        """
        if self._log_w_plus >= self._log_thr:
            return "adapt"
        if self._log_w_minus >= self._log_thr:
            return "freeze"
        return "abstain"

    @property
    def wealth_plus(self) -> float:
        """Current value of E_t^+ (one-sided process against Delta <= 0)."""
        return math.exp(self._log_w_plus)

    @property
    def wealth_minus(self) -> float:
        """Current value of E_t^- (one-sided process against Delta >= 0)."""
        return math.exp(self._log_w_minus)

    @property
    def wealth(self) -> float:
        """max(wealth_plus, wealth_minus) -- dominant e-process."""
        return max(self.wealth_plus, self.wealth_minus)

    def reset(self) -> "EProcess":
        """Reset to the initial state (t=0, wealth=1)."""
        self._s1 = 0.0
        self._s1m = 0.0
        self._s2 = self._cnt_var * 0.25  # restore prior
        self._s2m = self._cnt_var * 0.25
        self._cnt = 0.0
        self._cnt_var = 1.0  # reset to default prior_weight
        self._log_w_plus = 0.0
        self._log_w_minus = 0.0
        self.t = 0
        return self
