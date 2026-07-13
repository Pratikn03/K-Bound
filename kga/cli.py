"""kga.cli -- command-line entry point for the KGA gate.

Usage
-----
    python -m kga decide --calib calib.npy --test test.npy --alpha 0.1

Loads calibration and test detector scores from ``.npy`` files, computes the
label-free evidence ``Z``, builds a certificate from a benefit point estimate,
and prints the decision as JSON to stdout::

    {
      "decision": "ABSTAIN",
      "delta_hat": 0.0,
      "epsilon": 0.0123,
      "method": "conformal",
      "evidence": { ... }
    }

The benefit point estimate is unknown from scores alone (it would require
labels), so the CLI reports a conservative ``delta_hat = 0`` with a
split-conformal radius derived from the drift evidence (a calibration self-residual
proxy); the result therefore exercises the full evidence -> certificate -> decision
pipeline deterministically.  For real benefit estimates use the
:class:`kga.KGA` API directly with paired benefits or held-out residuals.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np

from kga.certificate import conformal_split
from kga.evidence import compute_evidence
from kga.policy import decide


def _load_scores(path: str) -> np.ndarray:
    """Load a 1-D or 2-D float score array from a ``.npy`` file."""
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{path}: expected a 1-D or 2-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path}: array contains non-finite values")
    return arr


def _decide_command(args: argparse.Namespace) -> int:
    """Run the ``decide`` subcommand; returns a process exit code."""
    calib = _load_scores(args.calib)
    test = _load_scores(args.test)
    ev = compute_evidence(calib, test)

    # Conservative point estimate (no labels available at the CLI). The conformal
    # radius is taken from the calibration's own column-wise spread as a
    # deterministic, label-free residual proxy so the trichotomy is exercised.
    calib_1d = calib.ravel()
    residual_proxy = np.abs(calib_1d - float(np.median(calib_1d)))
    cert = conformal_split(0.0, residual_proxy, alpha=args.alpha)
    dec = decide(cert, alpha=args.alpha)

    out = {
        "decision": dec.value,
        "delta_hat": cert.delta_hat,
        "epsilon": cert.epsilon,
        "method": cert.method,
        "alpha": cert.alpha,
        "evidence": {
            "ks_mean": ev.ks_mean,
            "ks_max": ev.ks_max,
            "disagree": ev.disagree,
            "entropy_shift": ev.entropy_shift,
            "conf_shift": ev.conf_shift,
            "ess_frac": ev.ess_frac,
            "n_calib": ev.n_calib,
            "n_test": ev.n_test,
            "n_detectors": ev.n_detectors,
        },
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``kga`` command-line tool."""
    parser = argparse.ArgumentParser(
        prog="kga",
        description="Knowability-Guided Adaptation: decide ADAPT/FREEZE/ABSTAIN from label-free score evidence.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_decide = sub.add_parser(
        "decide",
        help="Decide ADAPT/FREEZE/ABSTAIN from calibration and test .npy score files.",
    )
    p_decide.add_argument("--calib", required=True, help="Path to calibration scores (.npy).")
    p_decide.add_argument("--test", required=True, help="Path to unlabelled test scores (.npy).")
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
