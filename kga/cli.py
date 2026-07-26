"""kga.cli -- command-line entry point for the KGA gate.

Two subcommands, and neither of them fakes anything.

``kga evidence``
    Report the label-free evidence ``Z`` for a calibration/test score pair::

        python -m kga evidence --calib calib.npy --test test.npy

    This is the only thing that can be computed from unlabelled scores alone, so
    it is the only thing this subcommand claims.

``kga decide``
    Run a real certificate and the ADAPT/FREEZE/ABSTAIN trichotomy.  It needs a
    benefit signal, which unlabelled scores do not contain, so exactly one of the
    two conventions must be supplied::

        # (1) paired benefits X_i = loss(f0_i) - loss(fa_i)
        python -m kga decide --benefits benefits.npy --benefit-range 2.0

        # (2) a benefit point estimate + held-out calibration residuals
        python -m kga decide --delta-hat 0.031 --calib-residuals resid.npy

    ``--calib``/``--test`` may be added to either form to attach the evidence
    block to the output; they never influence the decision.

Why this file was rewritten (panel finding F2-10)
-------------------------------------------------
The previous ``decide`` subcommand took only ``--calib``/``--test``, hard-coded
``delta_hat = 0.0``, and built ``epsilon`` from the spread of the raw detector
scores.  Since ``epsilon >= 0`` always, neither ``delta_hat - epsilon > 0`` nor
``delta_hat + epsilon < 0`` could ever fire: it was a constant-ABSTAIN generator,
and its ``epsilon`` was in score units while its ``delta_hat`` was in risk units.
It is the first thing a reader runs, and it looked like a working gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np

from kga.certificate import (
    Certificate,
    conformal_split,
    empirical_bernstein,
    evalue_anytime,
    hoeffding,
)
from kga.evidence import compute_evidence
from kga.policy import decide

_BATCH_ESTIMATORS = {"ebern": empirical_bernstein, "hoeffding": hoeffding, "evalue": evalue_anytime}


def _load_array(path: str, name: str) -> np.ndarray:
    """Load a 1-D or 2-D float array from a ``.npy`` file."""
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{path}: expected a 1-D or 2-D array for {name}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path}: {name} contains non-finite values")
    return arr


def _load_scores(path: str) -> np.ndarray:
    """Backwards-compatible alias of :func:`_load_array` for score files."""
    return _load_array(path, "scores")


def _evidence_block(calib_path: str | None, test_path: str | None) -> dict | None:
    """Compute the reported evidence block, or ``None`` if no scores were given."""
    if not calib_path or not test_path:
        return None
    ev = compute_evidence(_load_array(calib_path, "calib"), _load_array(test_path, "test"))
    return {
        "ks_mean": ev.ks_mean,
        "ks_max": ev.ks_max,
        "disagree": ev.disagree,
        "entropy_shift": ev.entropy_shift,
        "conf_shift": ev.conf_shift,
        "ess_frac": ev.ess_frac,
        "n_calib": ev.n_calib,
        "n_test": ev.n_test,
        "n_detectors": ev.n_detectors,
    }


def _evidence_command(args: argparse.Namespace) -> int:
    """Run the ``evidence`` subcommand: report Z, decide nothing."""
    out = {
        "evidence": _evidence_block(args.calib, args.test),
        "note": (
            "Label-free evidence only. Z alone does not identify the sign of the "
            "benefit Delta (Theorem 1); use `kga decide` with paired benefits or "
            "calibration residuals to obtain a decision."
        ),
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _certificate_from_args(args: argparse.Namespace) -> Certificate:
    """Build a real certificate from whichever convention the user supplied."""
    has_benefits = args.benefits is not None
    has_point = args.delta_hat is not None or args.calib_residuals is not None
    if has_benefits == has_point:
        raise SystemExit(
            "kga decide: supply exactly one of\n"
            "  --benefits BENEFITS.npy [--benefit-range R] [--method ebern|hoeffding|evalue]\n"
            "  --delta-hat D --calib-residuals RESID.npy\n"
            "Unlabelled --calib/--test scores alone cannot yield a decision "
            "(that is Theorem 1); they may be added to either form for reporting."
        )

    if has_benefits:
        benefits = _load_array(args.benefits, "benefits").ravel()
        if args.method == "evalue":
            return evalue_anytime(benefits, alpha=args.alpha)
        if args.benefit_range is None:
            raise SystemExit(
                f"kga decide --method {args.method}: --benefit-range is required. "
                "It is the a-priori support width b - a of the paired benefits; "
                "estimating it from the sample voids the finite-sample guarantee. "
                "For |p - y| paired 0/1 losses pass --benefit-range 2.0."
            )
        estimator = _BATCH_ESTIMATORS[args.method]
        return estimator(benefits, alpha=args.alpha, benefit_range=args.benefit_range)

    if args.delta_hat is None or args.calib_residuals is None:
        raise SystemExit("kga decide: --delta-hat and --calib-residuals must be given together.")
    residuals = _load_array(args.calib_residuals, "calib_residuals").ravel()
    return conformal_split(float(args.delta_hat), residuals, alpha=args.alpha)


def _decide_command(args: argparse.Namespace) -> int:
    """Run the ``decide`` subcommand; returns a process exit code."""
    cert = _certificate_from_args(args)
    dec = decide(cert, alpha=args.alpha)
    out = {
        "decision": dec.value,
        "delta_hat": cert.delta_hat,
        "epsilon": cert.epsilon,
        "lower": cert.lower,
        "upper": cert.upper,
        "method": cert.method,
        "alpha": cert.alpha,
        "n": cert.n,
        "interval_level": cert.interval_level,
        "evidence": _evidence_block(args.calib, args.test),
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``kga`` command-line tool."""
    parser = argparse.ArgumentParser(
        prog="kga",
        description="Knowability-Guided Adaptation: label-free evidence Z, and the ADAPT/FREEZE/ABSTAIN gate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ev = sub.add_parser("evidence", help="Report the label-free evidence Z (no decision).")
    p_ev.add_argument("--calib", required=True, help="Path to calibration scores (.npy).")
    p_ev.add_argument("--test", required=True, help="Path to unlabelled test scores (.npy).")
    p_ev.set_defaults(func=_evidence_command)

    p_decide = sub.add_parser(
        "decide",
        help="Decide ADAPT/FREEZE/ABSTAIN from paired benefits or a point estimate + residuals.",
    )
    p_decide.add_argument("--benefits", help="Paired benefits X_i = loss(f0_i) - loss(fa_i) (.npy).")
    p_decide.add_argument(
        "--benefit-range",
        type=float,
        default=None,
        help=(
            "A-priori benefit support width b - a. Required with --benefits for "
            "ebern/hoeffding (2.0 for paired 0/1 losses)."
        ),
    )
    p_decide.add_argument(
        "--method",
        choices=sorted(_BATCH_ESTIMATORS),
        default="ebern",
        help="Batch estimator for --benefits. Default ebern.",
    )
    p_decide.add_argument("--delta-hat", type=float, default=None, help="Benefit point estimate Delta_hat.")
    p_decide.add_argument(
        "--calib-residuals",
        default=None,
        help="Held-out |Delta_hat_i - Delta_i| residuals (.npy) for the split-conformal radius.",
    )
    p_decide.add_argument("--calib", default=None, help="Optional calibration scores (.npy), reported as evidence only.")
    p_decide.add_argument("--test", default=None, help="Optional test scores (.npy), reported as evidence only.")
    p_decide.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level in (0, 1). Default 0.1.")
    p_decide.set_defaults(func=_decide_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch the chosen subcommand.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
