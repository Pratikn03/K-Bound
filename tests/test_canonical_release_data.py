from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "docs/research/kbound/scripts/validate_canonical_release_data.py"
STORAGE_REFRESH = ROOT / "docs/research/kbound/scripts/refresh_storage_manifest.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_release_data_is_consistent() -> None:
    validator = load_module(VALIDATOR, "canonical_release_validator")
    assert validator.validate() == []


def test_storage_refresh_is_bounded_to_declared_authorities() -> None:
    refresh = load_module(STORAGE_REFRESH, "storage_manifest_refresh")
    manifest = refresh.load_manifest()
    before = copy.deepcopy(manifest)
    refresh.refresh(manifest)

    before_rows = refresh.direct_rows(before)
    after_rows = refresh.direct_rows(manifest)
    changed = {
        location
        for location in before_rows
        if before_rows[location] != after_rows[location]
    }
    assert changed <= refresh.REFRESHABLE_AUTHORITIES
    for location in refresh.REFRESHABLE_AUTHORITIES:
        path = ROOT / location
        assert after_rows[location]["size_bytes"] == path.stat().st_size
        assert after_rows[location]["sha256"] == refresh.sha256(path)


def test_storage_refresh_preserves_all_status_fields() -> None:
    refresh = load_module(STORAGE_REFRESH, "storage_manifest_status_refresh")
    before = refresh.load_manifest()
    after = copy.deepcopy(before)
    refresh.refresh(after)

    def statuses(value):
        if isinstance(value, dict):
            yield value.get("status", "__missing__")
            for child in value.values():
                yield from statuses(child)
        elif isinstance(value, list):
            for child in value:
                yield from statuses(child)

    assert list(statuses(after)) == list(statuses(before))
