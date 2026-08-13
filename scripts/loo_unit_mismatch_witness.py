"""Witness: leave-one-CELL-out calibration undercovers on a new condition.

This is the failure mode the K-Bound stress grids are exposed to. It is an (A5)
failure -- right level alpha, wrong resampling unit -- not a defect of conformal
prediction. Nothing here is adversarial: the noise is Gaussian, the estimator is the
natural one, and the nominal level is honoured exactly at the unit the radius was
calibrated on.

Model
-----
    K conditions, m cells each.  For condition k, cell j:
        Delta_kj = mu_k + eps_kj,   mu_k ~ N(0, tau^2),  eps_kj ~ N(0, sigma^2)

    tau >> sigma: cells inside a condition are strongly correlated. That is the
    corruption x severity structure of CIFAR-10-C and the domain structure of PACS.

Estimator
---------
    The label-free evidence Z identifies the condition -- which it does in practice,
    since corruption type and severity are what entropy/confidence features track. So
    the fitted estimator predicts a cell by the mean of same-condition calibration
    cells, falling back to the global mean on an unseen condition. This is what a GBM
    on condition-identifying features learns.

Result (alpha = 0.10, K = 20, m = 18, tau/sigma = 10, 4000 replicates)
---------------------------------------------------------------------
    new-condition coverage, cell-out radius : 0.1378   (nominal 0.90)
    new-condition coverage, cond-out radius : 0.8918
    same-condition new cell, cell-out radius: 0.9005   <- valid at its own unit
    mean radius ratio cond/cell             : 10.1x    (= tau/sigma)

    At tau = 0, where cells are genuinely exchangeable, BOTH radii attain nominal
    coverage. The witness therefore isolates the dependence structure and nothing else.
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.10


def conformal_radius(residuals: np.ndarray, alpha: float) -> float:
    """Exact-rank radius: k = ceil((n+1)(1-alpha)); inf if unattainable at this n."""
    r = np.sort(np.abs(residuals))
    n = r.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(r[k - 1]) if k <= n else float("inf")


def experiment(K=20, m=18, tau=1.0, sigma=0.1, n_rep=4000, alpha=ALPHA, seed=0):
    rng = np.random.default_rng(seed)
    hit_cell, hit_cond, rad_cell, rad_cond = [], [], [], []

    for _ in range(n_rep):
        mu = rng.normal(0.0, tau, size=K)
        delta = mu[:, None] + rng.normal(0.0, sigma, size=(K, m))
        cond_mean = delta.mean(axis=1)
        grand = delta.mean()

        # leave-one-CELL-out: the m-1 siblings of the same condition remain in pool
        pred_cell = (cond_mean[:, None] * m - delta) / (m - 1)
        res_cell = np.abs(pred_cell - delta).ravel()

        # leave-one-CONDITION-out: the estimator has never seen condition k
        res_cond = []
        for k in range(K):
            keep = np.ones(K, dtype=bool); keep[k] = False
            res_cond.append(np.abs(delta[keep].mean() - delta[k]))
        res_cond = np.concatenate(res_cond)

        eps_cell = conformal_radius(res_cell, alpha)
        eps_cond = conformal_radius(res_cond, alpha)
        rad_cell.append(eps_cell); rad_cond.append(eps_cond)

        # deployment on a genuinely NEW condition
        mu_new = rng.normal(0.0, tau)
        delta_new = mu_new + rng.normal(0.0, sigma)
        err = abs(grand - delta_new)
        hit_cell.append(err <= eps_cell)
        hit_cond.append(err <= eps_cond)

    return {
        "alpha": alpha, "nominal": 1 - alpha, "K": K, "m": m,
        "tau": tau, "sigma": sigma, "n_rep": n_rep,
        "coverage_leave_one_CELL_out": float(np.mean(hit_cell)),
        "coverage_leave_one_CONDITION_out": float(np.mean(hit_cond)),
        "mean_radius_cell": float(np.mean(rad_cell)),
        "mean_radius_cond": float(np.mean(rad_cond)),
    }


def in_pool_check(K=20, m=18, tau=1.0, sigma=0.1, n_rep=2000, alpha=ALPHA, seed=7):
    """Control: the cell-out radius IS valid for a new cell of a SEEN condition.

    Included so the witness cannot be misread as "LOO is broken". It is exactly valid
    at the unit it was calibrated on; the failure is entirely in the transfer.
    """
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(n_rep):
        mu = rng.normal(0.0, tau, size=K)
        delta = mu[:, None] + rng.normal(0.0, sigma, size=(K, m))
        cond_mean = delta.mean(axis=1)
        pred_cell = (cond_mean[:, None] * m - delta) / (m - 1)
        eps = conformal_radius(np.abs(pred_cell - delta).ravel(), alpha)
        k = rng.integers(K)
        new_cell = mu[k] + rng.normal(0.0, sigma)
        hits.append(abs(delta[k].mean() - new_cell) <= eps)
    return float(np.mean(hits))


if __name__ == "__main__":
    base = experiment()
    print("LOO undercoverage witness -- right level alpha, wrong resampling unit")
    for k, v in base.items():
        print(f"  {k:36s} {v}")
    print(f"\n  new-condition cov (cell-out) : {base['coverage_leave_one_CELL_out']:.4f}"
          f"   nominal {base['nominal']}")
    print(f"  new-condition cov (cond-out) : {base['coverage_leave_one_CONDITION_out']:.4f}")
    print(f"  same-condition new cell      : {in_pool_check():.4f}  <- valid at its unit")
    print(f"  radius ratio cond/cell       : "
          f"{base['mean_radius_cond']/base['mean_radius_cell']:.1f}x")

    print("\n  tau/sigma   new-cond cov (cell-out)   (cond-out)")
    for tau in (0.0, 0.1, 0.3, 1.0, 3.0):
        r = experiment(tau=tau, sigma=0.1, n_rep=2000, seed=1)
        print(f"  {tau/0.1:9.1f} {r['coverage_leave_one_CELL_out']:23.4f} "
              f"{r['coverage_leave_one_CONDITION_out']:12.4f}")
