from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = (
    ROOT
    / "docs/research/kbound/scripts/audit_empirical_data_quality_2026_08_27.py"
)
AUDIT_DIR = ROOT / "docs/research/kbound/audits/empirical_data_quality_2026_08_27"
VALIDATOR_SCRIPT = ROOT / "src/scripts/validate_manuscript_claims.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("kbound_data_quality_audit", AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("kbound_manuscript_validator", VALIDATOR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_initial_finding_has_one_release_disposition() -> None:
    audit = _load_audit_module()
    combined, remediation = audit.findings_with_remediation(
        {"status_counts": {"match": 14}},
        {"status": "quarantined_invalid_derived_artifacts"},
    )

    assert [row["rank"] for row in combined] == list(range(1, 16))
    assert [row["rank"] for row in remediation] == list(range(1, 16))
    assert all(row["finding_stage"] == "initial_audit_pre_remediation" for row in combined)
    assert all(row["remediation_status"] for row in combined)
    assert all(row["release_disposition"] for row in combined)
    assert all(row["remaining_requirement"] for row in combined)
    assert combined[11]["finding"].startswith("Six of fourteen outer release checksum")
    assert combined[11]["remediation_status"] == "RELEASE_SEAL_VERIFIED"
    assert combined[12]["release_disposition"] == "NO_CONFIRMATORY_NATURAL_WIN_CLAIM_ALLOWED"


def test_csv_writer_rejects_duplicate_columns(tmp_path: Path) -> None:
    audit = _load_audit_module()
    with pytest.raises(ValueError, match="duplicate CSV field names"):
        audit.write_csv(tmp_path / "bad.csv", [{"harmful": 1}], ["harmful", "harmful"])


def test_generated_audit_separates_initial_and_post_remediation_state() -> None:
    summary = json.loads((AUDIT_DIR / "audit_summary.json").read_text())
    with (AUDIT_DIR / "remediation_status.csv").open(newline="") as handle:
        remediation = list(csv.DictReader(handle))
    with (AUDIT_DIR / "natural_opportunity.csv").open(newline="") as handle:
        opportunity = list(csv.DictReader(handle))

    assert summary["schema"] == "kbound_empirical_data_quality_audit_v2"
    assert summary["audit_stages"]["initial_diagnosis"]["empirical_readiness_score_out_of_10"] == 5.8
    post = summary["audit_stages"]["post_remediation"]
    assert post["natural_shift_routing_evidence_score_out_of_10"] == 4.0
    assert post["overall_empirical_readiness_score_out_of_10"] is None
    assert post["historical_results_revalidated_by_code_fixes"] is False
    assert post["invalid_derived_artifacts_quarantined"] == 14
    assert len(remediation) == 15
    iwild = next(row for row in opportunity if row["panel"] == "iWildCam H-v2")
    assert iwild["numeric_release_eligible"] == "False"
    assert iwild["claim_scope"] == "withheld_invalid_archived_metric_contract_historical_values"
    assert summary["bottom_line"]["defensible_natural_beats_both_win"] is False


def test_report_artifact_and_notebook_are_complete() -> None:
    artifact = json.loads((AUDIT_DIR / "artifact.json").read_text())
    table_ids = {table["id"] for table in artifact["manifest"]["tables"]}
    block_ids = {block["id"] for block in artifact["manifest"]["blocks"]}
    datasets = artifact["snapshot"]["datasets"]

    assert artifact["manifest"]["title"].endswith("Remediation Record")
    assert "remediation_table" in table_ids
    assert "remediation_block" in block_ids
    assert len(datasets["remediation"]) == 15
    assert datasets["headline_metrics"][0]["natural_evidence_score"] == 4.0

    notebook = json.loads(
        (
            ROOT
            / "docs/research/kbound/notebooks/kbound_empirical_data_quality_audit_2026_08_27.ipynb"
        ).read_text()
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    serialized = json.dumps(notebook, sort_keys=True)
    assert "/Users/" not in serialized
    assert "/Volumes/" not in serialized
    assert "/tmp/" not in serialized


def test_empirical_audit_artifacts_use_portable_paths() -> None:
    summary = json.loads((AUDIT_DIR / "audit_summary.json").read_text())
    assert summary["repository_path_binding"] == {
        "schema": "git-repository-relative-posix-v1",
        "root": ".",
        "root_role": "git_repository_root",
    }
    assert set(summary["runtime"]) == {
        "python",
        "numpy",
        "python_executable_basename",
    }
    serialized = json.dumps(summary, sort_keys=True)
    assert "/Users/" not in serialized
    assert "/Volumes/" not in serialized
    assert "/tmp/" not in serialized


def test_storage_manifest_internal_hashes_and_summary_match_disk() -> None:
    manifest = json.loads(
        (ROOT / "docs/research/kbound/STORAGE_MANIFEST.json").read_text()
    )

    direct_records = {}
    for row in manifest["artifacts"]:
        location = row.get("expected_location")
        if not location or location.startswith("$") or row.get("sha256") is None:
            continue
        path = ROOT / location
        assert path.is_file(), location
        assert path.stat().st_size == row["size_bytes"], location
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], location
        assert location not in direct_records
        direct_records[location] = (row["size_bytes"], row["sha256"])

    # The direct set may grow as Phase-1 adds independently verifiable release
    # authorities.  Pin the required authorities instead of a brittle row count.
    assert {
        "docs/research/kbound/claim_ledger.json",
        "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
        "docs/research/kbound/audits/phase1_provenance_2026_08_27/provenance_seal.json",
    } <= set(direct_records)

    counts = {"present": 0, "absent": 0}
    for location, row in manifest["sealed_evidence_checksums"].items():
        path = ROOT / location
        status = row["status"].lower()
        assert status in counts, (location, status)
        counts[status] += 1
        if status == "present":
            assert path.is_file(), location
            assert path.stat().st_size == row["size_bytes"], location
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], location
        else:
            assert not path.exists(), location

    summary = manifest["sealed_evidence_summary"]
    assert len(manifest["sealed_evidence_checksums"]) == summary["files"]
    assert counts["present"] == summary["present"]
    assert counts["absent"] == summary["absent"]
    assert summary["files"] == summary["present"] == 71
    assert summary["absent"] == 0

    for row in manifest["unsealed_present_artifacts"]:
        assert row["status"] == "present_unsealed"
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        assert path.stat().st_size == row["current_bytes"], row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["current_sha256"], row["path"]

    lock = json.loads(
        (ROOT / "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json").read_text()
    )
    locked_records = {}
    for track in lock["tracks"].values():
        for location, row in track["files"].items():
            record = (row["bytes"], row["sha256"])
            assert location not in locked_records or locked_records[location] == record
            locked_records[location] = record
    sealed_records = {
        location: (row["size_bytes"], row["sha256"])
        for location, row in manifest["sealed_evidence_checksums"].items()
        if row["status"] == "present"
    }
    assert locked_records.items() <= sealed_records.items()

    generated = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text()
    )

    def sources(value):
        if isinstance(value, dict):
            if isinstance(value.get("source"), str):
                yield value["source"]
            for child in value.values():
                yield from sources(child)
        elif isinstance(value, list):
            for child in value:
                yield from sources(child)

    covered = set(direct_records) | set(sealed_records)
    assert set(sources(generated)) <= covered


@pytest.mark.parametrize(
    ("mutation", "expected_problem"),
    [
        ("tracked_hash", "storage artifact row"),
        ("sealed_hash", "sealed evidence SHA-256 mismatch"),
        ("sealed_status", "sealed evidence has invalid status"),
        ("summary_count", "storage manifest sealed summary files mismatch"),
    ],
)
def test_storage_validator_fails_closed_on_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, expected_problem: str
) -> None:
    validator = _load_validator_module()
    manifest = json.loads(
        (ROOT / "docs/research/kbound/STORAGE_MANIFEST.json").read_text()
    )
    if mutation == "tracked_hash":
        tracked = next(row for row in manifest["artifacts"] if row.get("tracked") is True)
        tracked["sha256"] = "0" * 64
    elif mutation == "sealed_hash":
        sealed = next(iter(manifest["sealed_evidence_checksums"].values()))
        sealed["sha256"] = "0" * 64
    elif mutation == "sealed_status":
        sealed = next(iter(manifest["sealed_evidence_checksums"].values()))
        sealed["status"] = "unknown"
    elif mutation == "summary_count":
        manifest["sealed_evidence_summary"]["files"] += 1
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    tampered = tmp_path / "STORAGE_MANIFEST.json"
    tampered.write_text(json.dumps(manifest))
    monkeypatch.setattr(validator, "STORAGE_MANIFEST", tampered)
    generated = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text()
    )
    problems = []
    validator.validate_storage_manifest(problems, generated)
    assert any(expected_problem in problem for problem in problems), problems
