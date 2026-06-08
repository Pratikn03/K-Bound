"""Hermetic K-Bound trichotomy smoke test -- exercises the REAL ``kga`` package.

This is the zero-dependency, CPU-only, sub-minute reproduction smoke path.  It

1. generates a tiny deterministic synthetic score archive
   (:func:`make_synth_archive`), which needs **no external data**;
2. runs the **real** :mod:`kga` package over every task -- computing label-free
   evidence ``Z`` (:func:`kga.evidence.compute_evidence`), a finite-sample
   certificate ``Delta_hat +/- epsilon`` (:meth:`kga.KGA.certify`), and the
   ADAPT/FREEZE/ABSTAIN decision (:meth:`kga.KGA.decide`);
3. asserts the three regimes map to the three decisions, writes a JSON summary,
   and prints ``SMOKE PASS`` (exiting non-zero on any failure).

The decision pipeline is identical in spirit to
``src/scripts/kbound/knowability_experiment.py``: ``f0`` is the
validation-selected detector and ``fa`` is the rank-normalised ensemble, the
benefit is ``B = loss(f0) - loss(fa)`` (positive == adapting helps), and the
trichotomy is the certified-sign rule.  Here the per-sample paired benefits are
fed to the real empirical-Bernstein certificate so the run genuinely executes the
KGA math rather than re-implementing it.

Run directly::

    python src/scripts/kbound/smoke_trichotomy.py

or via the wrapper ``scripts/smoke_kbound.sh``.  Pure ``numpy`` + ``kga``; no
torch, no downloads, no real datasets.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# Allow running both as a module (``python -m ...``) and as a script
# (``python src/scripts/kbound/smoke_trichotomy.py``) by ensuring the repo root
# and ``src`` are importable, mirroring tests/conftest.py.
_HERE = os.path.abspath(__file__)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kga import KGA, Decision  # noqa: E402  (after sys.path shim)

# Import the synthetic generator by absolute module path; fall back to a direct
# file load so the smoke works regardless of how it was launched.
try:  # pragma: no cover - import-path robustness
    from scripts.kbound.make_synth_archive import (  # type: ignore
        DEFAULT_TASKS,
        make_synth_archive,
    )
except Exception:  # pragma: no cover
    import importlib.util

    _mod_path = os.path.join(os.path.dirname(_HERE), "make_synth_archive.py")
    _spec = importlib.util.spec_from_file_location("_kbound_make_synth_archive", _mod_path)
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DEFAULT_TASKS = _mod.DEFAULT_TASKS
    make_synth_archive = _mod.make_synth_archive


#: Operating miscoverage level for the smoke certificate.
ALPHA = 0.10

#: Maps each synthetic regime to the decision the gate MUST return.
EXPECTED_DECISION: dict[str, Decision] = {
    "helpful": Decision.ADAPT,
    "harmful": Decision.FREEZE,
    "unknowable": Decision.ABSTAIN,
}

#: Default output paths (repo-relative), created under the _smoke sandbox.
DEFAULT_SMOKE_DIR = os.path.join(_REPO, "experiments", "kbound", "_smoke")
DEFAULT_RESULT_JSON = os.path.join(DEFAULT_SMOKE_DIR, "smoke_result.json")


def _rank_norm_col(x: np.ndarray) -> np.ndarray:
    """Rank-normalise a 1-D score vector to ``[0, 1]`` (matches the K-Bound code).

    ``(rank - 1) / (n - 1)``; a length-1 vector maps to ``0.5``.
    """
    n = x.size
    if n <= 1:
        return np.full(n, 0.5, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return ranks / (n - 1.0)


def paired_benefits_from_task(Stest: np.ndarray, ytest: np.ndarray, val_auc: np.ndarray) -> np.ndarray:
    """Per-sample paired benefits ``X_i = loss(f0_i) - loss(fa_i)`` for one task.

    ``f0`` is the validation-selected detector (``argmax(val_auc)``) and ``fa`` is
    the rank-normalised ensemble (mean of the per-detector rank-normalised test
    scores).  Both are mapped to ``[0, 1]`` "probabilities" via rank
    normalisation, and the loss is the bounded absolute error ``|p - y|`` in
    ``[0, 1]``.  The benefit is therefore in ``[-1, 1]`` (range ``2.0``) and is
    positive exactly when adapting to the ensemble reduces per-sample loss.

    Parameters
    ----------
    Stest : numpy.ndarray, shape (n_test, n_det)
        Test detector scores.
    ytest : numpy.ndarray, shape (n_test,)
        Test labels (used here only to form the oracle per-sample loss, exactly
        as the reference experiment uses labels for evaluation).
    val_auc : numpy.ndarray, shape (n_det,)
        Per-detector validation AUC; selects ``f0``.

    Returns
    -------
    numpy.ndarray, shape (n_test,)
        The paired benefits ``X_i``.
    """
    Stest = np.asarray(Stest, dtype=float)
    ytest = np.asarray(ytest, dtype=float)
    n_det = Stest.shape[1]

    j0 = int(np.argmax(np.asarray(val_auc, dtype=float)))
    p_f0 = _rank_norm_col(Stest[:, j0])

    rn = np.column_stack([_rank_norm_col(Stest[:, j]) for j in range(n_det)])
    p_fa = rn.mean(axis=1)

    loss_f0 = np.abs(p_f0 - ytest)
    loss_fa = np.abs(p_fa - ytest)
    return loss_f0 - loss_fa


def run_task(kga: KGA, path: str, regime: str) -> dict[str, object]:
    """Run the full KGA gate on a single synthetic archive file.

    Parameters
    ----------
    kga : KGA
        A configured gate (carries ``alpha`` and the batch certificate method).
    path : str
        Path to the task ``.npz``.
    regime : str
        The task's regime tag (for the expected-decision check / reporting).

    Returns
    -------
    dict
        JSON-serialisable per-task record with the decision, certificate,
        evidence summary, and the oracle mean benefit.
    """
    d = np.load(path, allow_pickle=True)
    Sval, Stest = d["Sval"], d["Stest"]
    ytest, val_auc = d["ytest"], d["val_auc"]

    # Stage 1: REAL label-free evidence Z (no test labels used here).
    ev = kga.evidence(Sval, Stest)

    # Stage 2 + 3: REAL certificate + trichotomy from per-sample paired benefits.
    benefits = paired_benefits_from_task(Stest, ytest, val_auc)
    cert = kga.certify(scores=benefits, benefit_range=2.0)
    decision = kga.decide(cert)

    return {
        "task": os.path.splitext(os.path.basename(path))[0],
        "regime": regime,
        "decision": decision.value,
        "expected": EXPECTED_DECISION[regime].value,
        "delta_hat": cert.delta_hat,
        "epsilon": cert.epsilon,
        "lower": cert.lower,
        "upper": cert.upper,
        "method": cert.method,
        "n": cert.n,
        "oracle_mean_benefit": float(np.mean(benefits)),
        "evidence": {
            "ks_mean": ev.ks_mean,
            "disagree": ev.disagree,
            "ess_frac": ev.ess_frac,
            "n_test": ev.n_test,
            "n_detectors": ev.n_detectors,
        },
    }


def run_smoke(
    out_dir: str | None = None,
    *,
    alpha: float = ALPHA,
    seed: int = 0,
) -> dict[str, object]:
    """Generate the synthetic archive and run the KGA gate over every task.

    Parameters
    ----------
    out_dir : str, optional
        Directory for the synthetic archive and the result JSON.  Defaults to
        ``<repo>/experiments/kbound/_smoke``.
    alpha : float, default=0.10
        Operating miscoverage level for the certificate.
    seed : int, default=0
        Seed forwarded to the synthetic generator.

    Returns
    -------
    dict
        The full result summary with a ``status`` of ``"PASS"`` iff every regime
        mapped to its expected decision; any mismatch is recorded in
        ``summary["failures"]`` (the CLI :func:`main` turns that into a non-zero
        exit code, and the pytest suite asserts ``status == "PASS"``).
    """
    if out_dir is None:
        out_dir = DEFAULT_SMOKE_DIR
    archive_dir = os.path.join(out_dir, "score_archive")
    paths = make_synth_archive(archive_dir, seed=seed, tasks=list(DEFAULT_TASKS))

    kga = KGA(alpha=alpha, method="ebern")
    by_name = {spec.name: spec for spec in DEFAULT_TASKS}

    records: list[dict[str, object]] = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        regime = by_name[name].regime
        records.append(run_task(kga, path, regime))

    # Robust assertions: each regime must yield exactly its expected decision.
    failures = [
        f"{r['task']}: regime={r['regime']} -> got {r['decision']}, "
        f"expected {r['expected']} (delta_hat={r['delta_hat']:.4f}, "
        f"epsilon={r['epsilon']:.4f}, lower={r['lower']:.4f}, upper={r['upper']:.4f})"
        for r in records
        if r["decision"] != r["expected"]
    ]

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "alpha": alpha,
        "seed": seed,
        "n_tasks": len(records),
        "archive_dir": os.path.abspath(archive_dir),
        "tasks": records,
        "failures": failures,
    }
    return summary


def main() -> int:
    """Run the smoke, write ``smoke_result.json``, print a one-line summary.

    Returns
    -------
    int
        ``0`` if every regime mapped to its expected decision, else ``1``.
    """
    summary = run_smoke()
    os.makedirs(DEFAULT_SMOKE_DIR, exist_ok=True)
    with open(DEFAULT_RESULT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if summary["status"] != "PASS":
        sys.stderr.write("SMOKE FAIL\n")
        for line in summary["failures"]:  # type: ignore[union-attr]
            sys.stderr.write(f"  {line}\n")
        sys.stderr.write(f"Wrote {DEFAULT_RESULT_JSON}\n")
        return 1

    metrics = "  ".join(
        f"{r['task']}={r['decision']}(Delta_hat={float(r['delta_hat']):+.3f},"  # type: ignore[arg-type]
        f"eps={float(r['epsilon']):.3f})"
        for r in summary["tasks"]  # type: ignore[union-attr]
    )
    print("SMOKE PASS")
    print(f"  {summary['n_tasks']} tasks @ alpha={summary['alpha']}  ->  {metrics}")
    print(f"  wrote {DEFAULT_RESULT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
