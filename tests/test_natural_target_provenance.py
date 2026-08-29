from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/research/kbound/scripts/audit_natural_target_provenance.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("audit_natural_target_provenance", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_result_marks_track_opened(tmp_path: Path) -> None:
    module = load_module()
    results = tmp_path / "experiments/kbound/results"
    artifact = results / "officehome_run/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"metric": 0.5}\n', encoding="utf-8")
    payload = module.audit(results)
    assert payload["tracks"]["officehome"]["status"] == "OPENED_BEFORE_PROSPECTIVE_CLOSURE"
    assert payload["tracks"]["officehome"]["artifact_count"] == 1
    assert payload["prospective_natural_track_available"] is False


def test_absence_is_unknown_not_unopened(tmp_path: Path) -> None:
    module = load_module()
    results = tmp_path / "experiments/kbound/results"
    results.mkdir(parents=True)
    payload = module.audit(results)
    assert all(
        row["status"] == "UNKNOWN_REQUIRES_EXTERNAL_VERIFICATION"
        for row in payload["tracks"].values()
    )
    assert payload["verified_unopened_tracks"] == []
