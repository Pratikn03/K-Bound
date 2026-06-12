"""Command-line entry point for the orchestration flows.

Usage
-----
Run a flow by name, passing flow parameters as ``--key value`` pairs::

    python -m src.orchestration.runner fraud
    python -m src.orchestration.runner nlp --max_samples 2000 --max_features 4000
    python -m src.orchestration.runner vision --epochs 2 --image_size 224
    python -m src.orchestration.runner fusion --run_attention_validation true

List the available flows::

    python -m src.orchestration.runner list

The registry maps short flow names to their callables. To keep this module
import-light (the vision flow pulls in TensorFlow), flows are resolved lazily:
``list`` never imports a flow module, and running a flow imports only that one.

Parameter values are coerced from strings with light heuristics: booleans
(``true``/``false``), ``none``/``null``, integers, and floats are converted;
everything else is left as a string. This keeps the CLI ergonomic while
forwarding to the typed flow signatures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib import import_module
from typing import Any, Callable

# Map of flow name -> (module name within this package, callable attribute).
# Resolved lazily via :func:`get_flow` so importing this module is cheap and
# does not require optional heavy dependencies (e.g. TensorFlow for vision).
_FLOW_SPECS: dict[str, tuple[str, str]] = {
    "fraud": ("fraud_flow", "fraud_pipeline"),
    "cyber": ("cyber_flow", "cyber_pipeline"),
    "behavior": ("behavior_flow", "behavior_pipeline"),
    "fusion": ("fusion_flow", "fusion_pipeline"),
    "nlp": ("nlp_flow", "nlp_pipeline"),
    "vision": ("vision_flow", "vision_pipeline"),
}

#: Public list of registered flow names (stable ordering for the CLI).
FLOW_NAMES: list[str] = list(_FLOW_SPECS)


def get_flow(name: str) -> Callable[..., dict[str, Any]]:
    """Resolve and import a flow callable by its registry name.

    Parameters
    ----------
    name:
        Registered flow name (see :data:`FLOW_NAMES`).

    Returns
    -------
    Callable
        The flow callable (a Prefect flow when Prefect is installed, otherwise
        the underlying plain function via the compatibility shim).

    Raises
    ------
    KeyError
        If ``name`` is not a registered flow.
    """

    if name not in _FLOW_SPECS:
        raise KeyError(name)
    module_name, attr_name = _FLOW_SPECS[name]
    module = import_module(f"{__package__}.{module_name}")
    return getattr(module, attr_name)


class FlowsRegistry:
    """Lazy, dict-like registry mapping flow names to their callables.

    Behaves like ``{name: flow_callable}`` but imports a flow module only when
    that flow is actually accessed, so merely referencing the registry (e.g. for
    ``list``) never triggers heavy optional imports.
    """

    def __iter__(self):
        return iter(FLOW_NAMES)

    def __contains__(self, name: object) -> bool:
        return name in _FLOW_SPECS

    def __len__(self) -> int:
        return len(_FLOW_SPECS)

    def keys(self) -> list[str]:
        """Return the registered flow names."""

        return list(FLOW_NAMES)

    def __getitem__(self, name: str) -> Callable[..., dict[str, Any]]:
        try:
            return get_flow(name)
        except KeyError as exc:  # pragma: no cover - mirror dict semantics
            raise KeyError(name) from exc

    def get(
        self, name: str, default: Callable[..., dict[str, Any]] | None = None
    ) -> Callable[..., dict[str, Any]] | None:
        """Return the flow callable for ``name`` or ``default`` if absent."""

        if name in _FLOW_SPECS:
            return get_flow(name)
        return default


#: Importable registry instance: ``FLOWS["fraud"]`` -> fraud flow callable.
FLOWS = FlowsRegistry()


def _coerce(value: str) -> Any:
    """Coerce a CLI string value into a bool/None/int/float when sensible.

    Parameters
    ----------
    value:
        Raw string from the command line.

    Returns
    -------
    Any
        ``True``/``False`` for ``true``/``false`` (case-insensitive), ``None``
        for ``none``/``null``, an ``int`` or ``float`` when parseable, else the
        original string.
    """

    lowered = value.lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_flow_params(tokens: Sequence[str]) -> dict[str, Any]:
    """Parse ``--key value`` (or ``--flag``) tokens into a kwargs dict.

    A lone ``--flag`` not followed by a value is treated as boolean ``True``.

    Parameters
    ----------
    tokens:
        The remaining CLI tokens after the flow name.

    Returns
    -------
    dict
        Mapping of parameter name to coerced value.

    Raises
    ------
    ValueError
        If a token where a ``--key`` is expected does not start with ``--``.
    """

    params: dict[str, Any] = {}
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Expected --key before value, got: {token!r}")
        key = token[2:].replace("-", "_")
        if "=" in key:
            key, _, inline_value = key.partition("=")
            params[key] = _coerce(inline_value)
            i += 1
            continue
        if i + 1 < n and not tokens[i + 1].startswith("--"):
            params[key] = _coerce(tokens[i + 1])
            i += 2
        else:
            # Bare flag, e.g. "--run_attention_validation" -> True.
            params[key] = True
            i += 1
    return params


def main(argv: Sequence[str] | None = None) -> int:
    """Run the orchestration CLI.

    Parameters
    ----------
    argv:
        Optional argument vector (excluding the program name). Defaults to
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: ``0`` on success, non-zero on error.
    """

    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="python -m src.orchestration.runner",
        description="Run a UAIS orchestration flow or list available flows.",
    )
    parser.add_argument(
        "flow",
        help="Flow name to run, or 'list' to print the registry.",
    )
    # Flow-specific parameters are parsed manually so arbitrary --key value
    # pairs can be forwarded to each flow's typed signature.
    known, extra = parser.parse_known_args(argv)

    if known.flow == "list":
        for name in FLOW_NAMES:
            module_name, attr_name = _FLOW_SPECS[name]
            print(f"{name}\t-> src.orchestration.{module_name}.{attr_name}")
        return 0

    if known.flow not in FLOWS:
        available = ", ".join(FLOW_NAMES)
        print(
            f"Unknown flow {known.flow!r}. Available flows: {available} (or 'list').",
            file=sys.stderr,
        )
        return 2

    try:
        params = _parse_flow_params(extra)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    flow_callable = get_flow(known.flow)
    result = flow_callable(**params)

    try:
        print(json.dumps(result, indent=2, default=str))
    except TypeError:  # pragma: no cover - defensive; result should be JSON-able
        print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
