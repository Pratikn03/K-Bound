"""Relocating provenance must not regenerate outcomes or read target data."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from docs.research.kbound.scripts import seal_nine_track_lock as historical

spec = importlib.util.spec_from_file_location(
    "sync_historical_routes", Path(__file__).resolve().parents[1] / "scripts/sync_reconciled_panels.py"
)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_relocation_preserves_all_numerical_and_nonhistorical_fields():
    table = {"tracks": {"example": {"regret": [0.1, 0.2], "n": 3}},
             "nine_track_lock_seal": {"path": "old", "research_lock": "old",
                                      "note": "keep this qualification"}}
    before = copy.deepcopy(table)
    sync.relocate_historical_seal(table)
    assert table["tracks"] == before["tracks"]
    route = table["nine_track_lock_seal"]
    assert route["path"] == historical.SEAL_JSON.relative_to(historical.ROOT).as_posix()
    assert route["research_lock"] == historical.LOCK_YAML.relative_to(historical.ROOT).as_posix()
    assert route["note"] == "keep this qualification"
    assert route["current_policy_authority"] is False
    assert route["status"] == "historical_policy_only"


def test_failed_authentication_never_mutates_manifest(monkeypatch):
    table = {"nine_track_lock_seal": {"path": "old"}}
    before = copy.deepcopy(table)
    monkeypatch.setattr(historical, "verify_historical_seal", lambda: ({}, ["bad baseline"]))
    with pytest.raises(ValueError, match="bad baseline"):
        sync.relocate_historical_seal(table)
    assert table == before


def test_metadata_cli_never_loads_panel_or_other_authorities(tmp_path, monkeypatch):
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"tracks": {"held": 42}, "nine_track_lock_seal": {}}))
    monkeypatch.setattr(sync, "TABLE_PATH", table)
    monkeypatch.setattr(sync, "ROOT", tmp_path)
    load = sync._load

    def only_table(path):
        assert path == table, "relocation accessed a separate result authority"
        return load(path)

    monkeypatch.setattr(sync, "_load", only_table)
    monkeypatch.setattr(sys, "argv", ["sync", "--relocate-historical-only"])
    sync.main()
    updated = json.loads(table.read_text())
    assert updated["tracks"] == {"held": 42}
    assert updated["nine_track_lock_seal"]["current_policy_authority"] is False
