"""Live-code drift must fail even when release-artifact hashes still agree.

Only an explicit temporary copy of materialized public metadata and source is
changed. No raw images, labels, model execution, or original evidence writes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from pathlib import Path

import pytest

from docs.research.kbound.scripts import validate_canonical_release_data as validator

LIVE_SOURCES = {
    "crossfit": "kga/crossfit.py",
    "policy": "kga/policy.py",
    "certificate": "kga/certificate.py",
    "numeric_validation": "kga/_validation.py",
    "preregistered_protocol": "research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml",
}


@pytest.fixture
def release_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    paths = {
        *validator.SURFACES.values(),
        validator.CURRENT_POLICY_REL,
        validator.CANONICAL_REL,
        validator.SOURCE_MANIFEST_REL,
        validator.CCT20_REL,
        validator.CCT20_RECEIPT_REL,
        validator.SO2SAT_REL,
        validator.SO2SAT_RECEIPT_REL,
        validator.SO2SAT_NUMBERS_REL,
        validator.SO2SAT_NUMBERS_BUILDER_REL,
        validator.FMOW_REL,
        validator.POVERTY_REL,
        "docs/research/kbound/scripts/build_result_manifest.py",
        "docs/research/kbound/scripts/analyze_current_policy_cluster_inference.py",
        "docs/research/kbound/kbound_submission.tex",
        "docs/research/kbound/kbound_tmlr.tex",
        *LIVE_SOURCES.values(),
    }
    for relative in sorted(paths):
        source = source_root / relative
        metadata = source.lstat()
        assert stat.S_ISREG(metadata.st_mode), f"not a regular materialized fixture input: {relative}"
        assert not getattr(metadata, "st_flags", 0) & 0x40000000, f"dataless fixture input: {relative}"
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    # Isolate this regression from the separately tracked audit-refresh task.
    # Only the test copy is made internally consistent; no scientific value is
    # changed and this fixture is not a regenerated release-evidence artifact.
    authority_hash = hashlib.sha256((tmp_path / validator.CURRENT_POLICY_REL).read_bytes()).hexdigest()
    audit_path = tmp_path / validator.SURFACES["audit_summary"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for row in audit["release_checksums"]["rows"]:
        if row["path"] == validator.CURRENT_POLICY_REL:
            row["actual_sha256"] = authority_hash
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    return tmp_path


def test_canonical_release_accepts_matching_live_bindings(release_copy: Path) -> None:
    assert validator.validate() == []


@pytest.mark.parametrize("dependency", tuple(LIVE_SOURCES))
def test_canonical_release_rejects_live_drift_despite_consistent_artifact_hashes(
    release_copy: Path, dependency: str
) -> None:
    assert validator.validate() == []
    authority = release_copy / validator.CURRENT_POLICY_REL
    authority_before = authority.read_bytes()
    changed_source = release_copy / LIVE_SOURCES[dependency]
    changed_source.write_bytes(changed_source.read_bytes() + b"\n# changed only in isolated regression fixture\n")

    problems = validator.validate()

    assert authority.read_bytes() == authority_before
    assert problems == [f"current-policy family sensitivity {dependency} binding is stale"]


def test_canonical_release_rejects_generator_drift_despite_consistent_artifact_hashes(release_copy: Path) -> None:
    assert validator.validate() == []
    authority = release_copy / validator.CURRENT_POLICY_REL
    authority_before = authority.read_bytes()
    generator = release_copy / "docs/research/kbound/scripts/analyze_current_policy_cluster_inference.py"
    generator.write_bytes(generator.read_bytes() + b"\n# changed only in isolated regression fixture\n")

    problems = validator.validate()

    assert authority.read_bytes() == authority_before
    assert problems == ["current-policy family sensitivity analysis-script binding is stale"]
