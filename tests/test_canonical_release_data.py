from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "docs/research/kbound/scripts/validate_canonical_release_data.py"
STORAGE_REFRESH = ROOT / "docs/research/kbound/scripts/refresh_storage_manifest.py"
RESULT_BUILDER = ROOT / "docs/research/kbound/scripts/build_result_manifest.py"
PANEL_SYNC = ROOT / "scripts/sync_reconciled_panels.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_release_data_is_consistent() -> None:
    validator = load_module(VALIDATOR, "canonical_release_validator")
    assert validator.validate() == []


def test_separate_natural_authorities_survive_result_manifest_regeneration() -> None:
    builder = load_module(RESULT_BUILDER, "result_manifest_builder")
    ledger = builder.json.loads(builder.LEDGER.read_text())
    authorities = builder.validated_separate_authorities(ledger)
    assert authorities["cct20"]["claim_id"] == "KB-CLAIM-051"
    assert authorities["so2sat_development"]["claim_id"] == "KB-CLAIM-052"
    historical = builder.validated_historical_diagnostic_authorities(ledger)
    assert historical["fmow_protocol_l"]["claim_id"] == "KB-CLAIM-054"
    assert historical["poverty_protocol_l_development"]["claim_id"] == "KB-CLAIM-055"

    cct20 = builder.special_metrics("KB-CLAIM-051")
    assert cct20["decision_counts"] == {"ADAPT": 0, "FREEZE": 44, "ABSTAIN": 1}
    assert cct20["point_beats_both"] is False
    assert cct20["ci_robust_beats_both"] is False

    so2sat = builder.special_metrics("KB-CLAIM-052")
    assert so2sat["verdict"] == "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
    assert so2sat["selected_candidate_id"] is None
    assert so2sat["target_score"] is None
    assert so2sat["target_access"] == {
        "target_inputs": [],
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "gate_cal_rows_read_before_selection": 0,
    }


def test_panel_sync_preserves_later_studies_and_is_idempotent() -> None:
    sync = load_module(PANEL_SYNC, "reconciled_panel_sync")
    builder = load_module(RESULT_BUILDER, "result_manifest_builder_after_sync")
    ledger = sync._load(sync.LEDGER_PATH)
    current_cluster = sync._load(sync.CURRENT_CLUSTER_PATH)

    sync._sync_ledger(ledger, current_cluster)
    first_sync = copy.deepcopy(ledger)
    sync._sync_ledger(ledger, current_cluster)
    assert ledger == first_sync
    assert ledger["generated_at"] == "2026-08-29"
    assert len(ledger["claims"]) == 42

    by_id = {row["claim_id"]: row for row in ledger["claims"]}
    assert by_id["KB-CLAIM-044"]["status"] == "diagnostic"
    assert by_id["KB-CLAIM-045"]["status"] == "diagnostic"
    assert by_id["KB-CLAIM-051"]["verdict"] == "SAFE_UTILITY_ONLY"
    assert by_id["KB-CLAIM-052"]["target_access"]["target_labels_read"] == 0
    assert by_id["KB-CLAIM-053"]["status"] == "diagnostic"
    assert by_id["KB-CLAIM-054"]["status"] == "diagnostic"
    assert by_id["KB-CLAIM-055"]["status"] == "diagnostic"
    builder.validated_separate_authorities(ledger)
    builder.validated_historical_diagnostic_authorities(ledger)

    promoted = [
        row
        for row in ledger["claims"]
        if row.get("claim_type") == "empirical"
        and row.get("status") in {"supported", "no-harm", "descriptive", "diagnostic"}
        and any((ROOT / rel).is_file() for rel in row.get("supporting_artifacts", []))
    ]
    assert len(promoted) == 16
    assert {
        "KB-CLAIM-051",
        "KB-CLAIM-052",
        "KB-CLAIM-053",
        "KB-CLAIM-054",
        "KB-CLAIM-055",
    } <= {
        row["claim_id"] for row in promoted
    }


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
