"""Environment helpers: prefer ``KGA_*`` names, accept legacy ``UAIS_*``."""

from __future__ import annotations

import os


def env_first(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment value among ``names``."""
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip() != "":
            return raw
    return default


def csv_env(*names: str) -> list[str]:
    raw = env_first(*names, default="") or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def int_env(*names: str, default: int, minimum: int = 1) -> int:
    raw = env_first(*names)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


def bool_env(*names: str, default: bool = False) -> bool:
    raw = env_first(*names)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
