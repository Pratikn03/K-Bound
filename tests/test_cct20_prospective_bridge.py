from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from docs.research.kbound.scripts import verify_cct20_prospective_evidence as bridge


def external_bundle() -> Path:
    location = os.environ.get('KBOUND_CCT20_ROOT', '').strip()
    if not location:
        pytest.skip('external sealed CCT-20 bundle not configured; set KBOUND_CCT20_ROOT')
    root = Path(location).expanduser()
    assert root.is_dir(), f'configured CCT-20 bundle is missing: {root}'
    return root


def test_verified_cct20_bundle_reports_protocol_scoped_exchangeability() -> None:
    root = external_bundle()

    summary = bridge.verify_cct20_prospective_bundle(
        root,
        local_release_manifest=Path(
            "docs/research/kbound/paper/generated/cct20_release_manifest.json"
        ),
        verify_large_files=False,
    )

    assert summary["verification_outcome"] == "PASS"
    assert summary["population_unit"] == "camera_location"
    assert summary["target_location_count"] == 9
    assert summary["target_outcomes_unopened_before_execution"] is True
    assert summary["literal_label_unopened"] is False
    assert summary["exchangeability_status"] == (
        "ASSUMED_AT_PROTOCOL_SCOPE_NOT_EMPIRICALLY_PROVEN"
    )
    assert summary["result_status"] == "SAFE_UTILITY_ONLY"


def test_missing_seal_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(bridge.BridgeIntegrityError, match="execution seal"):
        bridge.verify_cct20_prospective_bundle(tmp_path, verify_large_files=False)


def test_tampered_firewall_fails_closed(tmp_path: Path) -> None:
    root = external_bundle()
    # Copying only the seal is sufficient to exercise the fail-closed firewall
    # check before any large dependency is read.
    seal = json.loads((root / "cct20_execution_seal_v1.json").read_text())
    seal["firewall"]["target_runner_imports_scorer"] = True
    unsigned = dict(seal)
    unsigned.pop("seal_payload_sha256", None)
    seal["seal_payload_sha256"] = bridge.stable_sha256(unsigned)
    (tmp_path / "cct20_execution_seal_v1.json").write_text(json.dumps(seal))
    with pytest.raises(bridge.BridgeIntegrityError, match="firewall"):
        bridge._validate_execution_seal_document(seal, tmp_path / "cct20_execution_seal_v1.json")
