"""Prefect compatibility shim for the orchestration package.

This module lets every ``*_flow.py`` use the *real* Prefect ``@flow`` / ``@task``
decorators and ``get_run_logger`` when Prefect is installed, while remaining
fully importable and runnable when it is **not**.

When Prefect is present, the genuine objects are re-exported unchanged, so flows
gain real retries, caching, structured run logs, and (optionally) deployment to
work pools and schedules.

When Prefect is absent, lightweight no-op fallbacks are provided. The fallback
``task`` / ``flow`` factories accept the *same* keyword arguments as Prefect
(``name``, ``retries``, ``retry_delay_seconds``, ``cache_key_fn``,
``log_prints``, ``persist_result``, ``timeout_seconds``, ...) and return the
decorated function unchanged. Both forms are supported::

    @task
    def step(...): ...

    @task(retries=2, retry_delay_seconds=5)
    def step(...): ...

The fallback ``get_run_logger`` returns a standard library logger named
``"orchestration"`` so structured logging calls keep working.

Downstream code should always import from this module rather than from
``prefect`` directly::

    from ._compat import flow, task, get_run_logger, PREFECT

Attributes
----------
PREFECT : bool
    ``True`` when the real Prefect package was imported successfully,
    ``False`` when the no-op fallbacks are in use.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

__all__ = ["flow", "task", "get_run_logger", "PREFECT"]

#: Type variable for the wrapped callable so the decorators are type-preserving.
F = TypeVar("F", bound=Callable[..., Any])

try:  # pragma: no cover - exercised only when Prefect is installed
    from prefect import flow, get_run_logger, task

    PREFECT: bool = True
except Exception:  # noqa: BLE001 - any import failure must fall back gracefully
    PREFECT = False

    _FALLBACK_LOGGER_NAME = "orchestration"

    def _identity_decorator(func: F) -> F:
        """Return ``func`` unchanged (the no-op decoration step)."""

        return func

    def _make_passthrough(kind: str) -> Callable[..., Any]:
        """Build a Prefect-compatible no-op ``task``/``flow`` decorator factory.

        The returned object supports both bare and parametrized decoration::

            @task
            def f(...): ...

            @task(retries=2, retry_delay_seconds=5, log_prints=True)
            def f(...): ...

        Parameters
        ----------
        kind:
            Either ``"task"`` or ``"flow"``. Used only for documentation and
            the ``__name__`` of the produced decorator.

        Returns
        -------
        Callable
            A decorator/decorator-factory mirroring the Prefect signature.
        """

        def decorator(*args: Any, **kwargs: Any) -> Any:
            # Bare usage: ``@task`` -> called with the function as the sole arg.
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return _identity_decorator(args[0])

            # Parametrized usage: ``@task(...)`` -> swallow kwargs (name,
            # retries, retry_delay_seconds, cache_key_fn, log_prints, etc.)
            # and return a decorator that leaves the function untouched.
            def wrap(func: F) -> F:
                return _identity_decorator(func)

            return wrap

        decorator.__name__ = kind
        decorator.__qualname__ = kind
        decorator.__doc__ = f"No-op Prefect-compatible {kind} decorator (Prefect not installed)."
        return decorator

    #: No-op ``task`` decorator factory mirroring :func:`prefect.task`.
    task = _make_passthrough("task")
    #: No-op ``flow`` decorator factory mirroring :func:`prefect.flow`.
    flow = _make_passthrough("flow")

    def get_run_logger(*_args: Any, **_kwargs: Any) -> logging.Logger:
        """Return a standard library logger as a Prefect ``get_run_logger`` stand-in.

        Returns
        -------
        logging.Logger
            A logger named ``"orchestration"``. A basic stream handler is
            attached on first use so messages are visible even when no logging
            configuration has been applied by the host application.
        """

        logger = logging.getLogger(_FALLBACK_LOGGER_NAME)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        return logger
