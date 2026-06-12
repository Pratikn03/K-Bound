"""Command-line interface for the unified UAIS training harness.

Run a single training config end-to-end::

    python -m uais.training.cli --config configs/training/isolation_forest.yaml

The CLI loads a :class:`~uais.training.trainer.TrainConfig` from YAML, resolves
the matching trainer from the registry, and executes the full lifecycle. Use
``--list`` to enumerate registered trainers, ``--dry-run`` to validate a config
without training, and ``--seed`` / ``--tracker`` to override config values.
"""

from __future__ import annotations

import argparse
import json
import sys

from uais.training.registry import available_trainers, get_trainer
from uais.training.trainer import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the training CLI.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="uais.training.cli",
        description="Run a UAIS training job from a YAML config via the unified harness.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to a training config YAML (see configs/training/).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered trainer names and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from the config.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["json", "mlflow"],
        default=None,
        help="Override the tracking backend from the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the config (and resolve the trainer) without training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m uais.training.cli``.

    Parameters
    ----------
    argv:
        Optional argument vector (excluding the program name). Defaults to
        :data:`sys.argv`.

    Returns
    -------
    int
        Process exit code: ``0`` on success, non-zero on error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        names = available_trainers()
        print("Registered trainers:")
        for name in names:
            print(f"  - {name}")
        return 0

    if not args.config:
        parser.error("--config is required (or use --list).")

    try:
        config = TrainConfig.from_yaml(args.config)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"[error] Failed to load config: {exc}", file=sys.stderr)
        return 2

    if args.seed is not None:
        config.seed = args.seed
        config.__post_init__()  # re-validate after override
    if args.tracker is not None:
        config.tracker = args.tracker
        config.__post_init__()

    try:
        trainer = get_trainer(config.trainer_key, config)
    except KeyError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 3

    if args.dry_run:
        print(f"[ok] Config valid. Resolved trainer: {type(trainer).__name__} (key={config.trainer_key}).")
        print(json.dumps(config.to_dict(), indent=2, default=str))
        return 0

    summary = trainer.run()
    print("[ok] Training complete.")
    print(json.dumps({k: v for k, v in summary.items() if k != "model_card"}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry guard
    raise SystemExit(main())
