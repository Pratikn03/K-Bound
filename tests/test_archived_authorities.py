from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "660d893caede49c3b7daa8c18e43bb6cbbce5480"
ARCHIVE = Path("docs/research/kbound/archive/superseded_empirical_authorities_2026-09-02")

MOVED_PATHS = (
    "experiments/kbound/results/camelyon17_protocol_G_v1/VERIFIED_FINDINGS.md",
    "experiments/kbound/results/camelyon17_richZ_F_v1/VERIFIED_FINDINGS.md",
    "experiments/kbound/results/iwildcam_protocol_H_v1/VERIFIED_FINDINGS.md",
    "experiments/kbound/results/iwildcam_protocol_H_v2/VERIFIED_FINDINGS.json",
    "experiments/kbound/results/officehome_protocol_M_v2/VERIFIED_FINDINGS.json",
    "experiments/kbound/results/officehome_protocol_m_repl_holdout/VERIFIED_FINDINGS.md",
    "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_FINDINGS.md",
    "experiments/kbound/results/win_hunt_v5/imagenetc_aggr_1pct/per_condition_imagenetc_tent_seed0.json",
    "research_lock/CAMELYON17_FULLSCALE_PROTOCOL_B_v1.yaml",
    "research_lock/CLEAN_COMPLEMENTARY_TRANSFER_PROTOCOL_v1.yaml",
    "research_lock/GATE_U_CROSS_DOMAIN_PROTOCOL_v1.yaml",
    "research_lock/INDEPENDENT_EXTERNAL_PROTOCOL_v1.yaml",
    "research_lock/MULTIMODAL_RELIABILITY_PROTOCOL_v1.yaml",
    "research_lock/NATURAL_SHIFT_SEALED_PROTOCOL_v1.yaml",
    "research_lock/NATURAL_WIN_PROTOCOL_v1.yaml",
    "research_lock/TARGET_LABEL_LIGHT_PPI_PROTOCOL_D25_v1.yaml",
    "research_lock/TARGET_LABEL_LIGHT_PROBE_PROTOCOL_v1.yaml",
    "research_lock/WIN_HUNT_v2_PROTOCOL.yaml",
    "research_lock/WIN_HUNT_v3_PROTOCOL.yaml",
    "research_lock/WIN_HUNT_v4_PROTOCOL.yaml",
    "research_lock/WIN_HUNT_v5_PROTOCOL_SHELL.yaml",
)

SUPERSEDED_PROTOCOLS = frozenset(path for path in MOVED_PATHS if path.startswith("research_lock/"))

ACTIVE_IMMUTABLE = {
    "research_lock/RICH_EVIDENCE_CAMELYON_PROTOCOL_F_v1.yaml": (
        4368,
        "b1a3d74ad022a163d63455f0172c7e3b557f5619eaf4dc21b7596c67343e5aaf",
    ),
    "research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml": (
        3146,
        "ee6caac547c32c0fb23bc21c2a319cbf166961f50ddc2d8c4bd2cca0626167ca",
    ),
}


def _git_bytes(relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_active_malformed_protocols_remain_exact_immutable_authorities() -> None:
    for relative_path, (expected_size, expected_sha256) in ACTIVE_IMMUTABLE.items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        payload = path.read_bytes()
        assert payload == _git_bytes(relative_path)
        assert len(payload) == expected_size
        assert _sha256(payload) == expected_sha256


def test_superseded_authorities_are_archived_byte_for_byte() -> None:
    manifest_path = ROOT / ARCHIVE / "MANIFEST.json"
    readme_path = ROOT / ARCHIVE / "README.md"
    assert manifest_path.is_file()
    assert readme_path.is_file()
    assert "not current authority" in readme_path.read_text(encoding="utf-8").lower()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["archive_id"] == ARCHIVE.name
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["policy"] == "historical_bytes_only_not_current_authority"
    records = manifest["records"]
    assert [record["original_path"] for record in records] == list(MOVED_PATHS)

    for record in records:
        original_path = record["original_path"]
        expected_archive_path = (ARCHIVE / "tree" / original_path).as_posix()
        assert record["archive_path"] == expected_archive_path
        assert record["category"] in {
            "invalid_empty_output",
            "superseded_claim_narrative",
            "superseded_protocol",
        }
        assert record["reason"].strip()
        if original_path in SUPERSEDED_PROTOCOLS:
            assert record["replacement_authority"] is None
            assert record["replacement_status"] == "historical_only_no_current_successor"
        else:
            replacement = record["replacement_authority"]
            assert isinstance(replacement, str) and replacement.strip()
            assert (ROOT / replacement).is_file(), replacement

        assert not (ROOT / original_path).exists(), original_path
        archived_path = ROOT / record["archive_path"]
        assert archived_path.is_file(), record["archive_path"]
        archived_bytes = archived_path.read_bytes()
        assert archived_bytes == _git_bytes(original_path)
        assert record["bytes"] == len(archived_bytes)
        assert record["sha256"] == _sha256(archived_bytes)

    archived_files = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / ARCHIVE / "tree").rglob("*") if path.is_file()
    }
    assert archived_files == {record["archive_path"] for record in records}


def test_archive_manifest_is_canonical_json() -> None:
    manifest_path = ROOT / ARCHIVE / "MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    assert manifest_path.read_text(encoding="utf-8") == expected


def test_current_research_index_routes_superseded_authorities_to_archive() -> None:
    archive_id = ARCHIVE.name
    docs_index = (ROOT / "docs/research/kbound/DOCS_INDEX.md").read_text(encoding="utf-8")
    audit_index = (ROOT / "docs/research/kbound/audits/README.md").read_text(encoding="utf-8")
    assert archive_id in docs_index
    assert archive_id in audit_index
    assert "not current authority" in docs_index.lower()
