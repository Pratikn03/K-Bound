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


def test_raw_rxrx1_backup_csv_is_not_release_evidence(tmp_path: Path) -> None:
    module = load_module()
    results = tmp_path / "experiments/kbound/results"
    raw_prediction = results / "rxrx1_internal_backup/predictions.csv"
    raw_prediction.parent.mkdir(parents=True)
    raw_prediction.write_text("label,prediction\n0,0\n", encoding="utf-8")

    payload = module.audit(results)

    assert payload["tracks"]["rxrx1"]["status"] == "UNKNOWN_REQUIRES_EXTERNAL_VERIFICATION"
    assert payload["tracks"]["rxrx1"]["artifact_count"] == 0


def test_unavailable_historical_result_still_marks_track_opened(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    results = tmp_path / "experiments/kbound/results"
    artifact = results / "officehome_run/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"metric": 0.5}\n', encoding="utf-8")

    def unavailable(_path: Path) -> str:
        raise OSError(89, "operation canceled")

    monkeypatch.setattr(module, "sha256_file", unavailable)
    payload = module.audit(results)

    track = payload["tracks"]["officehome"]
    assert track["status"] == "OPENED_BEFORE_PROSPECTIVE_CLOSURE"
    assert track["evidence"][0]["sha256"] is None
    assert track["evidence"][0]["hash_status"] == "unavailable"
    assert track["evidence"][0]["os_error_errno"] == 89
