"""Deterministic synthetic K-Bound score-archive generator (hermetic, torch-free).

This module fabricates a *tiny* score archive that is byte-for-byte schema-
compatible with the real 123-task ELARA-U archive consumed by
``src/scripts/kbound/knowability_experiment.py`` and
``src/scripts/kbound/mixed_regime_experiment.py`` -- but it contains **no real
data**: every score is drawn from a fixed-seed numpy generator.  Its sole purpose
is to drive the hermetic CPU smoke path (``smoke_trichotomy.py``) that exercises
the real :mod:`kga` package without any external download.

Schema (one ``.npz`` per task, identical to the real archive)
-------------------------------------------------------------
``Sval``      : ``(n_val, n_det)``  float64 -- validation detector scores.
``yval``      : ``(n_val,)``        int64   -- validation labels (0=normal, 1=anomaly).
``Stest``     : ``(n_test, n_det)`` float64 -- (unlabelled-at-decision-time) test scores.
``ytest``     : ``(n_test,)``       int64   -- test labels (used ONLY for oracle eval).
``det_names`` : ``(n_det,)``        str     -- detector names.
``val_auc``   : ``(n_det,)``        float64 -- per-detector validation AUC.
``domain``    : scalar              str     -- dataset/domain tag.

The three tasks deliberately span the K-Bound trichotomy so the downstream gate
must return one of each:

* ``synth_helpful``  -- on the *test* split the validation-selected detector
  ``f0`` has degraded while the rank-normalised ensemble ``fa`` is strongly more
  correct per sample.  The paired benefit ``X_i = loss(f0_i) - loss(fa_i)`` is
  clearly and consistently positive  ->  **ADAPT** is certifiable.
* ``synth_harmful``  -- ``f0`` remains an excellent detector on test while the
  ensemble ``fa`` is corrupted, so the paired benefit is clearly and
  consistently negative  ->  **FREEZE** is certifiable.
* ``synth_unknowable`` -- ``f0`` and ``fa`` are statistically tied per sample
  (benefit centred on zero with spread bracketing it), the non-identifiable
  regime  ->  **ABSTAIN**.

The separations are large relative to the empirical-Bernstein radius at the
default operating level, so the decisions are robust, not lucky (see
``smoke_trichotomy.py`` for the assertions).

All randomness flows from a single seed; re-running produces identical files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Default detector roster mirrors the real archive's six classic anomaly
# detectors (ECOD/COPOD/IForest/LOF/KNN/OCSVM); the smoke only needs the count
# and names to match the schema, not the algorithms.
DEFAULT_DET_NAMES: list[str] = ["ECOD", "COPOD", "IForest", "LOF", "KNN", "OCSVM"]

#: Default output directory for the synthetic archive (repo-relative).
DEFAULT_OUT_SUBDIR = os.path.join("experiments", "kbound", "_smoke", "score_archive")


def _repo_root() -> str:
    """Return the repository root, four levels above this file.

    ``src/scripts/kbound/make_synth_archive.py`` -> repo root is
    ``parents[3]`` (matches the other K-Bound scripts).
    """
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))


def _logit(p: np.ndarray) -> np.ndarray:
    """Numerically-stable logit used to spread probabilities into score space."""
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class TaskSpec:
    """Specification of one synthetic regime task.

    Attributes
    ----------
    name : str
        Output file stem (``<name>.npz``).
    regime : str
        One of ``'helpful'``, ``'harmful'``, ``'unknowable'``; selects how the
        test split is constructed relative to the validation-selected detector.
    domain : str
        Domain tag written into the archive (purely cosmetic for the smoke).
    """

    name: str
    regime: str
    domain: str


#: The three tasks spanning the trichotomy, in a fixed order.
DEFAULT_TASKS: list[TaskSpec] = [
    TaskSpec(name="synth_helpful", regime="helpful", domain="synthetic"),
    TaskSpec(name="synth_harmful", regime="harmful", domain="synthetic"),
    TaskSpec(name="synth_unknowable", regime="unknowable", domain="synthetic"),
]


def _make_split(
    rng: np.random.Generator,
    n: int,
    n_det: int,
    signal: float,
    anomaly_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a labelled detector-score block whose columns separate the classes.

    Each detector column is a noisy, monotone function of the latent label, so a
    higher ``signal`` yields a more discriminative (higher-AUC) detector.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator (the only randomness source).
    n : int
        Number of samples.
    n_det : int
        Number of detector columns.
    signal : float
        Class-separation strength in score space (larger == easier).
    anomaly_rate : float, default=0.5
        Fraction of positive (anomaly) labels.

    Returns
    -------
    (scores, labels) : tuple of numpy.ndarray
        ``scores`` of shape ``(n, n_det)`` float64 and ``labels`` of shape
        ``(n,)`` int64.
    """
    y = (rng.random(n) < anomaly_rate).astype(np.int64)
    base = signal * (y.astype(float) - 0.5)  # +/- signal/2 around 0 by class
    scores = np.empty((n, n_det), dtype=np.float64)
    for j in range(n_det):
        # DECREASING per-detector gain so detector 0 is the strongest (and is the
        # one selected as f0 on validation); independent noise keeps the columns
        # correlated but not identical.
        gain = 1.0 - 0.1 * j
        noise = rng.normal(0.0, 1.0, size=n)
        scores[:, j] = gain * base + noise
    return scores, y


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC (no sklearn dependency) for a 1-D score vector.

    Returns ``0.5`` for a degenerate (single-class) label vector.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Mann-Whitney-U estimate of P(score_pos > score_neg) using tie-averaged
    # ranks over the pooled scores (positives are first in ``combined``).
    combined = np.concatenate([pos, neg])
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg_rank_by_value = (start + csum + 1.0) / 2.0  # 1-indexed mean rank per value
    ranks = avg_rank_by_value[inv]
    rank_sum_pos = ranks[: pos.size].sum()
    auc = (rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def _build_task(
    spec: TaskSpec,
    rng: np.random.Generator,
    n_val: int,
    n_test: int,
    n_det: int,
) -> dict[str, np.ndarray]:
    """Construct one task's archive arrays for the requested regime.

    The validation split makes detector ``0`` the clear best-val detector (so the
    downstream ``f0 = argmax(val_auc)`` selects it).  The *test* split is then
    shaped per regime to make the paired benefit of the ensemble ``fa`` over
    ``f0`` clearly positive (helpful), clearly negative (harmful), or
    statistically zero (unknowable).

    Returns
    -------
    dict
        Mapping of the archive keys (``Sval``/``yval``/``Stest``/``ytest``/
        ``det_names``/``val_auc``/``domain``) to numpy arrays.
    """
    # --- Validation split: detector 0 is the strongest (best-val selection). ---
    # The decreasing gain in _make_split already makes detector 0 the highest-AUC
    # column; an extra boost makes argmax(val_auc) == 0 robust to the noise draw.
    Sval, yval = _make_split(rng, n_val, n_det, signal=3.0)
    Sval[:, 0] += 2.0 * (yval.astype(float) - 0.5)
    val_auc = np.array([_auc(yval, Sval[:, j]) for j in range(n_det)], dtype=np.float64)

    # --- Test split: shared latent labels; per-regime score construction. ---
    ytest = (rng.random(n_test) < 0.5).astype(np.int64)
    Stest = np.empty((n_test, n_det), dtype=np.float64)

    if spec.regime == "helpful":
        # f0 (detector 0) DEGRADES on test (near-random), while detectors 1..k-1
        # stay STRONGLY discriminative (low noise), so the rank-normalised
        # ensemble fa is far more correct per sample than f0 -> large positive
        # paired benefit -> ADAPT, with a wide margin over the certificate radius.
        for j in range(n_det):
            if j == 0:
                Stest[:, j] = rng.normal(0.0, 1.0, size=n_test)  # signal destroyed
            else:
                Stest[:, j] = 4.0 * (ytest.astype(float) - 0.5) + rng.normal(0.0, 0.35, size=n_test)
    elif spec.regime == "harmful":
        # f0 stays an EXCELLENT detector on test; the other detectors (hence the
        # ensemble fa) are corrupted, so adapting away from f0 clearly hurts.
        for j in range(n_det):
            if j == 0:
                Stest[:, j] = 4.5 * (ytest.astype(float) - 0.5) + rng.normal(0.0, 0.4, size=n_test)
            else:
                Stest[:, j] = rng.normal(0.0, 1.0, size=n_test)  # signal destroyed
    elif spec.regime == "unknowable":
        # Non-identifiable regime: all detectors are NEAR-IDENTICAL (one shared
        # moderately-informative signal plus tiny independent jitter). Because the
        # ensemble equals each member up to that jitter, f0 (a single column) and
        # fa (the ensemble) make essentially the same per-sample prediction, so
        # the paired benefit is centred on zero with spread bracketing it
        # -> ABSTAIN. (Distinct, independently-noisy columns would instead let the
        # ensemble average out noise and beat f0, which is why we tie them here.)
        shared = 2.0 * (ytest.astype(float) - 0.5) + rng.normal(0.0, 1.0, size=n_test)
        for j in range(n_det):
            Stest[:, j] = shared + rng.normal(0.0, 0.02, size=n_test)
    else:  # pragma: no cover - guarded by TaskSpec construction
        raise ValueError(f"unknown regime {spec.regime!r}")

    return {
        "Sval": Sval,
        "yval": yval,
        "Stest": Stest,
        "ytest": ytest,
        # Native unicode dtype (matches the archive's `str` fields and avoids
        # requiring pickle to load).
        "det_names": np.asarray(DEFAULT_DET_NAMES[:n_det], dtype="U16"),
        "val_auc": val_auc,
        "domain": np.asarray(spec.domain, dtype="U32"),
    }


def make_synth_archive(
    out_dir: str | os.PathLike[str] | None = None,
    *,
    seed: int = 0,
    n_val: int = 600,
    n_test: int = 600,
    n_det: int = 6,
    tasks: list[TaskSpec] | None = None,
) -> list[str]:
    """Write the deterministic synthetic score archive to ``out_dir``.

    Parameters
    ----------
    out_dir : path-like, optional
        Target directory; created if missing.  Defaults to
        ``<repo>/experiments/kbound/_smoke/score_archive``.
    seed : int, default=0
        Master seed for the single numpy generator (full determinism).
    n_val, n_test : int, default=600
        Per-task validation/test sample counts.  600 keeps the run well under a
        second while giving the empirical-Bernstein certificate a tight radius.
    n_det : int, default=6
        Number of detector columns (matches the real archive's six detectors).
    tasks : list of TaskSpec, optional
        Tasks to emit (defaults to the three trichotomy regimes).

    Returns
    -------
    list of str
        Absolute paths of the written ``.npz`` files, in task order.

    Examples
    --------
    >>> import tempfile, numpy as np
    >>> d = tempfile.mkdtemp()
    >>> paths = make_synth_archive(d)
    >>> len(paths)
    3
    >>> z = np.load(paths[0], allow_pickle=True)
    >>> sorted(z.files)
    ['Stest', 'Sval', 'det_names', 'domain', 'val_auc', 'ytest', 'yval']
    """
    if out_dir is None:
        out_dir = os.path.join(_repo_root(), DEFAULT_OUT_SUBDIR)
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if tasks is None:
        tasks = DEFAULT_TASKS

    rng = np.random.default_rng(seed)
    written: list[str] = []
    for spec in tasks:
        arrays = _build_task(spec, rng, n_val=n_val, n_test=n_test, n_det=n_det)
        path = os.path.join(out_dir, f"{spec.name}.npz")
        np.savez(path, **arrays)
        written.append(os.path.abspath(path))
    return written


def main() -> int:
    """CLI entry point: write the synthetic archive to the default location.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    paths = make_synth_archive()
    print(f"Wrote {len(paths)} synthetic task(s) to:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
