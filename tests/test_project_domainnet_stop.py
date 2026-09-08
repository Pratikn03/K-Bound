"""Path-only STOP projection contracts; no outcome or linked artifact is read."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/project_domainnet_stop.py"
COMPANION = ROOT / "protocols/confirmatory_v2/DOMAINNET_DEV_PILOT_v1_STOP_PORTABLE.json"


def module():
    assert SCRIPT.is_file(), "hash-bound DomainNet STOP projector is missing"
    spec = importlib.util.spec_from_file_location("project_domainnet_stop", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture():
    return {
        "status": "STOP_SOURCE_OVERLAP",
        "scientific_outcome": "NOT_TESTED",
        "rows": 31502,
        "cross_source_duplicate_groups": 20,
        "minimum_geometry_passes": True,
        "candidate_or_gate_outcomes_scored": False,
        "unavailable_evidence": None,
        "preparation": {
            "directory": "/Volumes/synthetic-private/prepared",
            "summary_sha256": "a" * 64,
        },
        "independent_verification": {
            "receipt": "/Volumes/synthetic-private/independent.json",
            "status": "PASS_CONFIRMED_STOP",
            "receipt_sha256": "b" * 64,
        },
    }


def test_projection_changes_only_two_declared_locations_and_preserves_evidence():
    # Dropped/retyped counts, booleans, nulls, hashes or STOP conclusions must fail.
    source = fixture()
    before = copy.deepcopy(source)
    payload, redactions = module().redact_payload(source)
    expected = copy.deepcopy(before)
    expected["preparation"]["directory"] = "private-workspace/painting-v1-preparation"
    expected["independent_verification"]["receipt"] = "private-artifact/painting-v1-independent-verification"
    assert payload == expected
    assert source == before
    assert payload["minimum_geometry_passes"] is True
    assert payload["candidate_or_gate_outcomes_scored"] is False
    assert payload["unavailable_evidence"] is None
    assert [(r["json_pointer"], r["replacement_value"]) for r in redactions] == [
        ("/preparation/directory", "private-workspace/painting-v1-preparation"),
        ("/independent_verification/receipt", "private-artifact/painting-v1-independent-verification"),
    ]
    assert [r["redacted_utf8_sha256"] for r in redactions] == [
        hashlib.sha256(b"/Volumes/synthetic-private/prepared").hexdigest(),
        hashlib.sha256(b"/Volumes/synthetic-private/independent.json").hexdigest(),
    ]


@pytest.mark.parametrize("fault", ["undeclared_value", "undeclared_key", "missing", "nonpath"])
def test_unmapped_or_missing_locations_fail_closed(fault):
    # Silent broad redaction or an incomplete declared locator inventory must fail.
    document = fixture()
    if fault == "undeclared_value":
        document["extra"] = "/Users/private/unmapped.json"
    elif fault == "undeclared_key":
        document["extra"] = {"/Volumes/private/unmapped.json": "c" * 64}
    elif fault == "missing":
        del document["preparation"]["directory"]
    else:
        document["preparation"]["directory"] = None
    with pytest.raises(ValueError):
        module().redact_payload(document)


@pytest.mark.parametrize("raw", [b"{}", b"[]", b"", b"{\"status\": \"PASS\"}"])
def test_arbitrary_bytes_cannot_replace_the_historical_stop(raw):
    # Removing the immutable original-byte pin would accept these substitutions.
    with pytest.raises(ValueError, match="approved historical STOP"):
        module().project_bytes(raw)


def test_cli_rejects_unapproved_bytes_without_creating_output(tmp_path):
    # Computing an output before identity validation would publish unbound evidence.
    module()
    source, output = tmp_path / "source.json", tmp_path / "output.json"
    source.write_bytes(b"{}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(source), "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "approved historical STOP" in result.stderr
    assert not output.exists()
    assert source.read_bytes() == b"{}"


def test_distributed_companion_keeps_original_identity_and_stopped_evidence():
    # Promoting the historical STOP or silently replacing its input breaks this contract.
    assert COMPANION.is_file(), "portable historical STOP companion is missing"
    raw = COMPANION.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "ab0ab3c9b2513a7d4db5472a9d7fdda7edfa7523aa97b6e630225efce2cfe3b9"
    companion = json.loads(raw)
    assert companion["original_stop"] == {
        "logical_path": "protocols/confirmatory_v2/DOMAINNET_DEV_PILOT_v1_STOP.json",
        "sha256": "164c276ab5db4c720964d54c4ea0f44455c8a32e3b36c1b88c2a5c815ee1db61",
        "bytes": 4244,
        "original_bytes_included": False,
    }
    payload = companion["historical_stop"]
    assert payload["status"] == "STOP_SOURCE_OVERLAP"
    assert payload["stage"] == "PRE_MODEL_DEVELOPMENT_INPUT_VALIDATION"
    assert payload["scientific_outcome"] == "NOT_TESTED"
    assert [payload[k] for k in ("rows", "decoded_content_groups", "cross_source_duplicate_groups",
                                "overlapping_painting_rows", "overlapping_clipart_rows")] == [31502, 30369, 20, 21, 20]
    assert payload["all_twenty_groups_also_share_identical_encoded_bytes"] is True
    assert payload["candidate_or_gate_outcomes_scored"] is False
    assert payload["historical_target_nonaccess_proved"] is False
    assert payload["software_scope"]["full_repository_or_release_pass"] is False
    assert companion["redaction_count"] == len(companion["redactions"]) == 2
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == companion["projected_payload_sha256"]
    assert b"/Volumes/" not in COMPANION.read_bytes()
