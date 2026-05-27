"""Placeholder NLP preprocessing pipeline."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def clean_emails(emails: Iterable[str]) -> list[str]:  # pragma: no cover - placeholder
    return [e.strip() for e in emails]


def load_enron_emails(path: Path) -> pd.DataFrame:  # pragma: no cover - placeholder
    return pd.read_csv(path)


__all__ = ["clean_emails", "load_enron_emails"]
