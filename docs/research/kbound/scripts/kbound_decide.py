#!/usr/bin/env python3
"""kbound_decide.py -- THE single K-Bound decision path used by every driver.

This module exists to kill two defects at once:

  * fix-queue item 4 -- "remove the scored cell from its own radius pool".
    Every shipped runner computed ``eps = np.quantile(|Bhat - B|, 1-alpha)`` over
    *all* N residuals, including cell i's own, and then used that eps to decide
    cell i.  eps was therefore a function of the test labels it was being used to
    protect.  The default here is a **leave-one-out-of-pool** radius: cell i's
    radius is the exact-rank conformal quantile of the OTHER N-1 residuals.

  * fix-queue item 15 -- "route every driver through the shipped library".
    Seven copy-pasted ``decide_kga`` forks produced every reported number while
    ``kga/certificate.py`` + ``kga/policy.py`` -- the artifact the paper ships --
    produced none.  This module calls the library when it is importable and
    falls back to a byte-identical local implementation when it is not, so the
    scripts still run in a bare checkout.  ``BACKEND`` records which path ran.

Radius rule
-----------
The exact split-conformal *rank* quantile only::

    r_(1) <= ... <= r_(n),  k = ceil((n+1)(1-alpha)),  eps = r_(k)

``np.quantile``'s linear interpolation is never used, and **there is no clamp
convention any more** (fix-queue item 25 / defect D9).  When ``k > n`` -- i.e.
``n <= 8`` at ``alpha = 0.10``, which under leave-one-out-of-pool calibration
means ``n <= 9`` cells -- no finite radius attains ``1-alpha``, so::

    eps = +inf   =>   every cell ABSTAINs, and a UserWarning is emitted.

This module used to default to ``clamp="min_n"`` (``k <- min(n, k)``, i.e. the
maximum residual) "so that re-running the fixed code reproduces
``NUMBERS_PACK.md``".  That made the shim implement a *different* rule from the
shipped ``kga`` library, which stopped clamping in item 25 -- two rules, one
declaration.  The clamp is gone from both.  The superseded value is still
computable, but only from the explicitly named
``kga.certificate.legacy_clamped_radius``, which nothing here calls.

**Consequence, stated rather than hidden.**  Any table whose calibration pool
has ``n <= 8`` cannot be produced under the declared rule; it ABSTAINs
everywhere.  In this repository that is the ``n = 9``-cells-per-seed Camelyon17
Table VIII panel under LOO (pools of 8) and the iWildCam source-CV certificate
in ``experiments/kbound/wilds/analyze_iwildcam_kbound.py`` (source ``n < 9``).
Those rows were computed under the clamp; they are not reproducible under the
rule the paper declares, and must be labelled accordingly rather than silently
re-emitted.  Every other promoted track has a pool of at least 17
(``NUMBERS_PACK.md`` §5.2: Camelyon17 pooled 18, Office-Home 35, CIFAR-10.1 48,
RxRx1 60, iWildCam 72, ImageNet-C 27/seed, D33 130, CIFAR-10-C 432/seed), so
the clamp never fired for them and their numbers are unchanged.

False-adapt (fix-queue item 28)
-------------------------------
"False adapt" has exactly one definition here: an ADAPT decision on a cell with
``B <= 0``.  ``fa_u`` is the marginal rate (what ``thm:certificate`` bounds),
``fa_c`` the conditional rate among ADAPT decisions.  Both are emitted; neither
is silently substituted for the other.  The legacy field ``false_adapt_rate_B<0``
(conditional, strict ``<``) is preserved by callers for artifact comparability
and marked deprecated.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ALPHA = 0.10

# --------------------------------------------------------------------------- #
# repo root, so no script needs a machine-local absolute path (fix-queue item 30)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[4]


def repo_path(*parts) -> str:
    """Resolve a path relative to the repository root.

    Honours ``KBOUND_REPO_ROOT`` / ``KBOUND_RESULTS_ROOT`` so an external results
    volume can be pointed at from the environment instead of being hard-coded to
    one author's laptop -- ``EXTERNAL_STORAGE_POLICY.md:18`` bans machine-local
    paths in tracked code.
    """
    root = Path(os.environ.get("KBOUND_REPO_ROOT", REPO_ROOT))
    return str(root.joinpath(*parts))


def results_root() -> str:
    """Root of the experiment results tree (override: ``KBOUND_RESULTS_ROOT``)."""
    env = os.environ.get("KBOUND_RESULTS_ROOT")
    if env:
        return env
    return repo_path("experiments", "kbound", "results")


# --------------------------------------------------------------------------- #
# defensive import of the shipped library (fix-queue item 15)
# --------------------------------------------------------------------------- #
_ROOT_STR = str(REPO_ROOT)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

try:  # pragma: no cover - exercised implicitly by every driver
    from kga.certificate import Certificate as _Certificate
    from kga.certificate import split_conformal_rank_radius as _kga_radius
    from kga.policy import decide as _kga_decide

    BACKEND = "kga-library"
except Exception as _e:  # noqa: BLE001 - a bare checkout must still run
    _Certificate = None
    _kga_radius = None
    _kga_decide = None
    BACKEND = f"local-fallback ({type(_e).__name__}: {_e})"


def backend() -> str:
    """Which implementation actually ran -- stamp this into run manifests."""
    return BACKEND


# --------------------------------------------------------------------------- #
# (1) conformal radius -- exact rank only, never np.quantile
# --------------------------------------------------------------------------- #
def _rank_radius_local(residuals, alpha=ALPHA) -> float:
    """Byte-identical fallback for ``kga.certificate.split_conformal_rank_radius``.

    Used only when the library is not importable (bare checkout).  It must stay
    behaviourally identical, including the infeasible branch: no clamp.
    """
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = r.size
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        warnings.warn(
            f"split-conformal at alpha={alpha} needs n >= {int(math.ceil(1.0 / alpha)) - 1} "
            f"calibration residuals but got n={n}: exact rank k={k} exceeds n, so no finite "
            f"radius attains 1-alpha. Returning +inf => ABSTAIN.",
            UserWarning,
            stacklevel=2,
        )
        return float("inf")
    return float(r[k - 1])


def conformal_radius(residuals, alpha=ALPHA) -> float:
    """Exact split-conformal rank radius ``eps = r_(k)``, ``k = ceil((n+1)(1-a))``.

    ONE rule, no options.  Routes through
    ``kga.certificate.split_conformal_rank_radius`` whenever the library is
    importable and falls back to the behaviourally identical local
    implementation otherwise.  ``k > n`` yields ``+inf`` (forced ABSTAIN) in both
    paths -- the clamp that used to be the default here was removed in defect D9
    because it made this shim implement a second, under-covering rule.
    """
    arr = np.abs(np.asarray(residuals, dtype=float))
    if arr.size == 0:
        return float("inf")
    if _kga_radius is not None:
        try:
            return float(_kga_radius(arr, alpha))
        except Exception:  # noqa: BLE001 - a bare checkout must still run
            pass
    return _rank_radius_local(arr, alpha=alpha)


def radii_in_pool(residuals, alpha=ALPHA) -> np.ndarray:
    """LEGACY, LEAKY *pool*: one radius from all N residuals, reused for every cell.

    Kept only so a driver can reproduce a pre-fix archived number on demand
    (``--calibration in_pool``).  Do not make this the default anywhere: it makes
    eps a function of the very test labels the FA_u guarantee attaches to.  Note
    that it is leaky in the *pool*, not in the *rank rule*: the radius it returns
    is still the exact-rank one, with no clamp.
    """
    arr = np.abs(np.asarray(residuals, dtype=float))
    return np.full(arr.size, conformal_radius(arr, alpha), dtype=float)


def radii_loo(residuals, alpha=ALPHA) -> np.ndarray:
    """Leave-one-out-of-pool radii: cell i's radius excludes cell i's residual.

    This is fix-queue item 4.  Measured effect (NUMBERS_PACK.md sec. 4):
    0 of 9,504 CIFAR-10-C decisions change; ImageNet-C SAR moves 2 of 135
    (regret 0.026422 -> 0.028893, FA_u 0/135 -> 1/135); Camelyon17 SAR moves 1.
    """
    arr = np.abs(np.asarray(residuals, dtype=float))
    n = arr.size
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = conformal_radius(np.delete(arr, i), alpha)
    return out


def radii_holdout(residuals, calib_mask, alpha=ALPHA) -> np.ndarray:
    """Genuine held-out calibration split: one radius from the calibration cells,
    applied to every cell.  Cells inside the calibration split get the
    leave-one-out radius so that they are not scored by their own residual."""
    arr = np.abs(np.asarray(residuals, dtype=float))
    mask = np.asarray(calib_mask, dtype=bool)
    if mask.sum() == 0:
        raise ValueError("calibration split is empty")
    eps_out = conformal_radius(arr[mask], alpha)
    out = np.full(arr.size, eps_out, dtype=float)
    idx = np.flatnonzero(mask)
    for i in idx:
        keep = mask.copy()
        keep[i] = False
        out[i] = conformal_radius(arr[keep], alpha)
    return out


CALIBRATIONS = {"loo": radii_loo, "in_pool": radii_in_pool}


# --------------------------------------------------------------------------- #
# (2) decision rule -- routed through kga.policy
# --------------------------------------------------------------------------- #
def decide(bhat, eps, alpha=ALPHA) -> np.ndarray:
    """ADAPT / FREEZE / ABSTAIN for each cell, via ``kga.policy.decide``.

    ``eps`` may be a scalar or a per-cell array.  A non-finite radius (the
    infeasible small-n case) always yields ABSTAIN.
    """
    bh = np.asarray(bhat, dtype=float)
    ep = np.broadcast_to(np.asarray(eps, dtype=float), bh.shape)
    out = np.full(bh.shape, "ABSTAIN", dtype=object)
    for i in range(bh.size):
        e = float(ep.flat[i])
        b = float(bh.flat[i])
        if not math.isfinite(e):
            continue  # no finite radius => nothing is certifiable => ABSTAIN
        if _kga_decide is not None and _Certificate is not None:
            try:
                cert = _Certificate(delta_hat=b, epsilon=e, method="conformal",
                                    alpha=alpha, n=bh.size)
                out.flat[i] = str(_kga_decide(cert, alpha=alpha))
                continue
            except Exception:  # noqa: BLE001 - fall through to the local rule
                pass
        out.flat[i] = "ADAPT" if b - e > 0 else ("FREEZE" if b + e < 0 else "ABSTAIN")
    # Decision is a plain str in every artifact schema in this repo.
    return np.asarray([str(x).replace("Decision.", "") for x in out.ravel()],
                      dtype=object).reshape(bh.shape)


# --------------------------------------------------------------------------- #
# (3) the benefit estimator (unchanged machinery, one copy)
# --------------------------------------------------------------------------- #
def loo_bhat(Z, B, n_estimators=250, max_depth=2, lr=0.05, seed=0) -> np.ndarray:
    """Leave-one-cell-out gradient-boosted benefit estimate Bhat(Z).

    NOTE (fix-queue item 18): this is leave-one-*cell*-out, not
    leave-one-*task*-out.  Several docstrings in the tree claim the latter.
    Under leave-one-corruption-out calibration the residual MAE triples and eps
    quadruples (NUMBERS_PACK.md sec. 7.1) -- report that ablation, do not
    describe this function as task-level.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    Z = np.asarray(Z, dtype=float)
    B = np.asarray(B, dtype=float)
    N = len(B)
    bh = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=lr, subsample=0.8, random_state=seed)
        m.fit(Z[tr], B[tr])
        bh[i] = m.predict(Z[i:i + 1])[0]
    return bh


def minimum_crossfit_size(alpha=ALPHA, min_train=2) -> int:
    """Minimum cells for label-disjoint score/train/calibration partitions."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if int(min_train) < 2:
        raise ValueError("min_train must be at least two")
    min_calibration = int(math.ceil(1.0 / float(alpha)) - 1)
    return min_calibration + int(min_train) + 1


def _stable_crossfit_order(sample_ids, salt, seed):
    return sorted(
        range(len(sample_ids)),
        key=lambda index: hashlib.sha256(
            f"{seed}|{salt}|{sample_ids[index]}".encode("utf-8")
        ).hexdigest(),
    )


def decide_kga_crossfit(
    Z,
    B,
    alpha=ALPHA,
    n_estimators=250,
    max_depth=2,
    lr=0.05,
    seed=0,
    sample_ids=None,
    n_folds=5,
):
    """Label-disjoint cross-fitted K-Bound diagnostic.

    For every score fold, the remaining cells are split—using only stable sample
    identifiers—into an estimator-fit set and an exact-rank calibration set. The
    score-fold labels enter neither fit. Therefore changing ``B[i]`` cannot change
    cell ``i``'s prediction, radius, or decision.

    This removes the subtler leakage in the historical two-stage LOO procedure:
    although residual ``i`` was deleted from radius ``i``, every other residual's
    estimator had been trained with ``B[i]``. Cross-fitting is suitable for
    development diagnostics; confirmation still requires a predeclared, disjoint
    validation lock and an unopened test partition.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    Z = np.asarray(Z, dtype=float)
    B = np.asarray(B, dtype=float)
    if Z.ndim != 2 or B.ndim != 1 or len(Z) != len(B):
        raise ValueError("Z must be two-dimensional and match one-dimensional B")
    if not (np.isfinite(Z).all() and np.isfinite(B).all()):
        raise ValueError("Z and B must contain only finite values")
    n = len(B)
    if sample_ids is None:
        sample_ids = [f"index:{index}" for index in range(n)]
    else:
        sample_ids = [str(value) for value in sample_ids]
    if len(sample_ids) != n or len(set(sample_ids)) != n:
        raise ValueError("sample_ids must be unique and match Z/B length")

    minimum = minimum_crossfit_size(alpha)
    if n < minimum or len(np.unique(B)) < 2:
        bhat = np.full(n, float(np.mean(B)) if n else 0.0)
        epsilon = np.full(n, float("inf"))
        return bhat, epsilon, decide(bhat, epsilon, alpha=alpha)

    min_calibration = int(math.ceil(1.0 / float(alpha)) - 1)
    max_score_fold = n - min_calibration - 2
    folds_required = int(math.ceil(n / max_score_fold))
    fold_count = min(n, max(2, int(n_folds), folds_required))
    order = _stable_crossfit_order(sample_ids, "score-fold", seed)
    score_folds = [order[offset::fold_count] for offset in range(fold_count)]

    bhat = np.full(n, np.nan, dtype=float)
    epsilon = np.full(n, np.inf, dtype=float)
    all_indices = set(range(n))
    for fold_index, score_indices in enumerate(score_folds):
        if not score_indices:
            continue
        remaining = sorted(all_indices - set(score_indices))
        calibration_order_local = _stable_crossfit_order(
            [sample_ids[index] for index in remaining],
            f"calibration-fold:{fold_index}",
            seed,
        )
        ordered_remaining = [remaining[index] for index in calibration_order_local]
        desired_calibration = max(
            min_calibration,
            int(math.ceil(0.30 * len(ordered_remaining))),
        )
        calibration_n = min(desired_calibration, len(ordered_remaining) - 2)
        if calibration_n < min_calibration:
            continue
        calibration_indices = ordered_remaining[:calibration_n]
        train_indices = ordered_remaining[calibration_n:]
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.8,
            random_state=seed,
        )
        model.fit(Z[train_indices], B[train_indices])
        calibration_prediction = model.predict(Z[calibration_indices])
        fold_radius = conformal_radius(
            np.abs(calibration_prediction - B[calibration_indices]),
            alpha=alpha,
        )
        bhat[score_indices] = model.predict(Z[score_indices])
        epsilon[score_indices] = fold_radius

    missing = np.isnan(bhat)
    if missing.any():
        # A future split-policy regression must fail closed, never partially route.
        bhat[missing] = float(np.mean(B))
        epsilon[missing] = float("inf")
    return bhat, epsilon, decide(bhat, epsilon, alpha=alpha)


def decide_kga(Z, B, alpha=ALPHA, n_estimators=250, max_depth=2, lr=0.05, seed=0,
               calibration="loo"):
    """The one K-Bound decision path.  Returns ``(Bhat, eps, dec)``.

    ``eps`` is an ndarray of length N -- **one radius per cell** -- because the
    scored cell is excluded from its own calibration pool.  Callers that used to
    receive a scalar must serialise ``eps[i]`` per cell; a scalar summary is
    available as ``float(np.mean(eps))`` but must be labelled as a mean.

    calibration
        ``"loo"``      leave-one-out-of-pool (default; the fix)
        ``"in_pool"``  the archived leaky rule, for reproducing old artifacts
    """
    if calibration not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {sorted(CALIBRATIONS)}, got {calibration!r}")
    Z = np.asarray(Z, dtype=float)
    B = np.asarray(B, dtype=float)
    bh = loo_bhat(Z, B, n_estimators=n_estimators, max_depth=max_depth, lr=lr, seed=seed)
    resid = np.abs(bh - B)
    eps = CALIBRATIONS[calibration](resid, alpha=alpha)
    dec = decide(bh, eps, alpha=alpha)
    return bh, eps, dec


def decide_from_records(bhat, B, alpha=ALPHA, calibration="loo"):
    """Re-score stored ``b_hat``/``B`` per-condition dumps.  Returns ``(eps, dec)``."""
    bh = np.asarray(bhat, dtype=float)
    Bv = np.asarray(B, dtype=float)
    eps = CALIBRATIONS[calibration](np.abs(bh - Bv), alpha=alpha)
    return eps, decide(bh, eps, alpha=alpha)


# --------------------------------------------------------------------------- #
# (4) ONE definition of false-adapt (fix-queue item 28)
# --------------------------------------------------------------------------- #
def false_adapt(dec, B):
    """Return ``{"fa_u", "fa_c", "n_false_adapt", "n_adapt", "n"}``.

    A *false adapt* is an ADAPT decision on a cell with ``B <= 0``.  The weak
    inequality is deliberate: 500 archived cells have ``B`` exactly 0.0 and 102
    of them ADAPT, so the strict ``B < 0`` used by ``wilds/analysis.py:87`` and
    ``cifar_tent_mps_v2.py:182`` silently exempts them.

      fa_u = Pr[ADAPT and B <= 0]        <- the quantity thm:certificate bounds
      fa_c = Pr[B <= 0 | ADAPT]          <- the conditional rate
    """
    dec = np.asarray(dec, dtype=object)
    Bv = np.asarray(B, dtype=float)
    is_adapt = dec == "ADAPT"
    n = int(Bv.size)
    n_adapt = int(is_adapt.sum())
    k = int(np.sum(is_adapt & (Bv <= 0)))
    return {
        "n": n,
        "n_adapt": n_adapt,
        "n_false_adapt": k,
        "fa_u": (k / n) if n else None,
        "fa_c": (k / n_adapt) if n_adapt else None,
    }


def fa_ceiling(n, alpha=ALPHA):
    """``(n-k)/n``, the value FA_u cannot exceed under IN-POOL rank calibration.

    With eps the k-th order statistic of the same residual vector it is used to
    test, the miscoverage count is identically ``n-k``, so ``FA_u <= (n-k)/n``
    holds for *any* data: 0.0972 at n=432, 0.0370 at n=27, and exactly 0 at
    n <= 9.  Report ``FA_u = 0`` against this ceiling, not against alpha.
    """
    n = int(n)
    if n <= 0:
        return None
    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    return (n - k) / n


# --------------------------------------------------------------------------- #
# (5) artifact IO that names the file when it is an iCloud placeholder
# --------------------------------------------------------------------------- #
def read_json(path):
    """Read a JSON artifact, failing with an actionable message.

    145 tracked text artifacts in this tree are NUL-filled iCloud placeholders
    (fix-queue item 9); a whitespace-only test does not catch them.  A bare
    ``FileNotFoundError`` / ``JSONDecodeError`` tells a reviewer nothing, so name
    the file and say what to do.
    """
    import json

    p = str(path)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Required artifact is absent: {p}\n"
            f"  -> it is not in this release. See docs/research/kbound/STORAGE_MANIFEST.json\n"
            f"     and DATA.md for how to obtain or regenerate it."
        )
    raw = open(p, "rb").read()
    if len(raw) == 0 or b"\x00" in raw:
        raise IOError(
            f"Required artifact is an unmaterialised placeholder: {p}\n"
            f"  -> {len(raw)} bytes, NUL-filled (iCloud 'Optimise Mac Storage').\n"
            f"     Run 'Download Now' on the source machine, or regenerate it."
        )
    return json.loads(raw)


def records(path):
    d = read_json(path)
    if isinstance(d, dict) and "records" in d:
        return d["records"]
    return d


__all__ = [
    "ALPHA", "BACKEND", "REPO_ROOT", "backend", "repo_path", "results_root",
    "conformal_radius", "radii_in_pool", "radii_loo", "radii_holdout",
    "CALIBRATIONS", "decide", "loo_bhat", "minimum_crossfit_size",
    "decide_kga_crossfit", "decide_kga", "decide_from_records",
    "false_adapt", "fa_ceiling", "read_json", "records",
    "selftest_radius_excludes_scored_cell",
]


def selftest_radius_excludes_scored_cell(n=40, alpha=ALPHA, seed=0):
    """FIX-QUEUE ITEM 4 regression check: cell i's radius must not see cell i.

    Two properties are asserted, both of which the pre-fix in-pool rule violates:

    1. *Independence of the scored residual.*  Perturb residual ``i`` to a value
       far larger than every other residual.  Under LOO, ``eps[i]`` must be
       bit-identical to what it was before the perturbation (cell i is not in its
       own pool), while every ``eps[j != i]`` may move.  Under the in-pool rule
       ``eps[i]`` moves, which is exactly what makes eps a function of the label
       the FA_u guarantee attaches to.
    2. *Explicit construction.*  ``eps[i]`` must equal the exact-rank radius
       recomputed from ``np.delete(residuals, i)``.

    Raises AssertionError on failure; returns the max deviation on success.
    """
    rng = np.random.default_rng(seed)
    resid = np.abs(rng.normal(0, 0.05, n))
    eps0 = radii_loo(resid, alpha)

    i = int(np.argmin(resid))                 # a cell whose residual is NOT the max
    bumped = resid.copy()
    bumped[i] = resid.max() * 100.0 + 1.0     # make cell i the extreme outlier
    eps1 = radii_loo(bumped, alpha)
    assert eps0[i] == eps1[i], (
        f"LEAKAGE: cell {i}'s radius changed ({eps0[i]!r} -> {eps1[i]!r}) when only "
        f"cell {i}'s OWN residual changed. The scored cell is still in its own pool.")

    dev = 0.0
    for j in range(n):
        expect = conformal_radius(np.delete(resid, j), alpha)
        dev = max(dev, abs(float(eps0[j]) - float(expect)))
    assert dev < 1e-12, f"eps[j] != rank radius of the other n-1 residuals (max dev {dev:g})"

    # And the in-pool rule must FAIL property 1 -- otherwise the test is vacuous.
    ip0 = radii_in_pool(resid, alpha)
    ip1 = radii_in_pool(bumped, alpha)
    assert ip0[i] != ip1[i], (
        "the in-pool control did not move when the scored residual moved; "
        "this regression test would pass trivially and proves nothing.")
    return dev


if __name__ == "__main__":  # tiny self-check
    rng = np.random.default_rng(0)
    Bt = rng.normal(0, 0.1, 60)
    Zt = np.column_stack([Bt + rng.normal(0, 0.01, 60) for _ in range(3)])
    bh, eps, dec = decide_kga(Zt, Bt)
    print(f"backend            : {backend()}")
    print(f"eps per-cell (LOO) : min {eps.min():.5f}  max {eps.max():.5f}")
    print(f"decisions          : "
          f"{ {d: int((dec == d).sum()) for d in ('ADAPT', 'FREEZE', 'ABSTAIN')} }")
    print(f"false-adapt        : {false_adapt(dec, Bt)}")
    print(f"in-pool FA_u ceiling at n=60: {fa_ceiling(60):.4f}")
    _dev = selftest_radius_excludes_scored_cell()
    print(f"item-4 regression  : PASS (scored cell excluded from its own pool; "
          f"max dev vs explicit np.delete recomputation {_dev:.1e})")
