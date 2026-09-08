"""Path-only projection tests require no original private receipt or outcomes."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/project_historical_diagnostic_receipt.py"


def module():
    assert SCRIPT.is_file(), "historical diagnostic receipt projector is missing"
    spec = importlib.util.spec_from_file_location("diagnostic_projection_under_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sample():
    return {
        "schema_version": 1,
        "classification": "recovered_historical_assertions",
        "scope": "non-K-Bound MVTec/VisA diagnostics",
        "source_checkout": "/private-location/historical-checkout",
        "metric_recomputation_performed": False,
        "kbound_confirmation": False,
        "uncertain_observation": None,
        "artifacts": [{"destination_path": "experiments/audit/example.csv", "bytes": 1744,
                       "sha256": "a" * 64, "source_sha256": "a" * 64, "blank_fields_preserved": True}],
    }


def test_projection_changes_only_source_checkout_and_preserves_original():
    original = sample()
    before = copy.deepcopy(original)
    projected, redactions = module().redact_payload(original)
    assert original == before
    expected = copy.deepcopy(before)
    expected["source_checkout"] = "private-workspace/recovered-historical-checkout"
    assert projected == expected
    assert projected["uncertain_observation"] is None
    assert projected["metric_recomputation_performed"] is False
    assert projected["artifacts"][0]["blank_fields_preserved"] is True
    assert len(redactions) == 1
    assert redactions[0]["json_pointer"] == "/source_checkout"
    assert redactions[0]["redacted_utf8_sha256"] == hashlib.sha256(b"/private-location/historical-checkout").hexdigest()
    assert "/private-location/" not in json.dumps((projected, redactions))


def test_projection_is_deterministic():
    projector = module()
    assert projector.redact_payload(sample()) == projector.redact_payload(sample())


@pytest.mark.parametrize("change", [
    lambda d: d.pop("source_checkout"),
    lambda d: d.update(source_checkout=123),
    lambda d: d.update(source_checkout="already-public"),
    lambda d: d.update(unexpected_private_path="/private-location/unreviewed"),
    lambda d: d.update(embedded_private_path="See /Users/example/unreviewed for details"),
    lambda d: d.update({"/private-location/key": "unchanged"}),
])
def test_unreviewed_private_locations_or_wrong_projection_shape_are_rejected(change):
    original = sample()
    change(original)
    with pytest.raises(ValueError):
        module().redact_payload(original)


@pytest.mark.parametrize("raw", [b"{}", b'{"schema": 1, "schema": 2}', b'{"number": NaN}'])
def test_projector_rejects_bytes_outside_the_exact_approved_original(raw):
    with pytest.raises(ValueError):
        module().project_bytes(raw)


def test_cli_wrong_original_keeps_source_and_does_not_create_output(tmp_path):
    module()
    original = tmp_path / "original.json"
    original.write_bytes(b"{}")
    output = tmp_path / "portable.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--source", str(original),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.exists()
    assert original.read_bytes() == b"{}"
