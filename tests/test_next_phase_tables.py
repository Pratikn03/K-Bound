"""Presentation must retain the exact completed-study authorities."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/make_next_phase_tables.py"
spec = importlib.util.spec_from_file_location("next_phase_tables", SCRIPT)
assert spec and spec.loader
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


def test_current_tables_reproduce_from_pinned_authorities(tmp_path: Path) -> None:
    tables.build(tables.SOURCE_DIR, tmp_path)
    for name in tables.TABLE_NAMES:
        assert (tmp_path / name).read_bytes() == (tables.OUTPUT_DIR / name).read_bytes()


def test_changed_authority_cannot_overwrite_a_table(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in tables.AUTHORITIES:
        data = (tables.SOURCE_DIR / name).read_bytes()
        (sources / name).write_bytes(data)
    changed = sources / "calibration_value_summary.json"
    changed.write_bytes(changed.read_bytes().replace(b'"n": 2160', b'"n": 2161', 1))
    outputs = tmp_path / "tables"
    outputs.mkdir()
    target = outputs / tables.TABLE_NAMES[0]
    target.write_text("previous verified table\n", encoding="utf-8")
    with pytest.raises(ValueError, match="authority hash mismatch"):
        tables.build(sources, outputs)
    assert target.read_text(encoding="utf-8") == "previous verified table\n"
