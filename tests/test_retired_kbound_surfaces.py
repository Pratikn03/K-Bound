from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from docs.research.kbound.scripts import run_repository_verification as repository_inventory

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "660d893caede49c3b7daa8c18e43bb6cbbce5480"
ARCHIVE = Path("docs/research/kbound/archive/superseded_empirical_authorities_2026-09-02")
RETIRED_MANIFEST = ARCHIVE / "RETIRED_SURFACES_MANIFEST.json"
VERIFIER = ROOT / "docs/research/kbound/scripts/seal_nine_track_lock.py"
MANUSCRIPT_VALIDATOR = ROOT / "src/scripts/validate_manuscript_claims.py"

ARCHIVED_ONLY = (
    "docs/research/kbound/gapclose_wave5/estimator_v2.py",
    "docs/research/kbound/gapclose_wave5/natural_win_analysis.py",
    "docs/research/kbound/gapclose_wave5/radius_jackknife_plus.py",
    "docs/research/kbound/gapclose_wave5/rerun_A_jkplus_logged.py",
    "docs/research/kbound/gapclose_wave5/rerun_BC_logged.py",
    "docs/research/kbound/gapclose_wave5/rerun_F_composite_logged.py",
    "docs/research/kbound/gapclose_wave5/tau_adaptive.py",
    "docs/research/kbound/gapclose_wave5/val_estimator_v2.py",
    "docs/research/kbound/gapclose_wave5/val_jackknife_plus.py",
    "docs/research/kbound/gapclose_wave5/val_tau_adaptive.py",
    "docs/research/kbound/gapclose_wave5/win_hunt_A_universal_gate.py",
    "docs/research/kbound/gapclose_wave5/win_hunt_C_panel_select.py",
    "docs/research/kbound/gapclose_wave5/win_hunt_D_anytime_stream.py",
    "docs/research/kbound/gapclose_wave5/win_hunt_E_universal7.py",
    "docs/research/kbound/gapclose_wave5/win_hunt_F_headtohead.py",
    "docs/research/kbound/gapclose_wave5/win_hunt_G_lambda_real.py",
    "experiments/kbound/poem_aetta/score_official_headtohead.py",
    "experiments/kbound/ppi_micro_probe.py",
    "experiments/kbound/results/camelyon17_fullscale_B_v1/_locked_B_analysis.py",
    "scripts/run_natural_win_v1.sh",
    "experiments/kbound/results/KGA_ELARA_ANALYSIS.md",
    "research_lock/KBOUND_6_DATASET_PANEL_v1.yaml",
    "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json",
    "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.sha256",
    "research_lock/NINE_TRACK_LOCK_SEAL_v1.yaml",
)

REPLACED_ACTIVE = (
    "docs/research/kbound/runbooks/finish_empirical_training.sh",
    "docs/research/kbound/scripts/run_final_showcase.sh",
    "scripts/rebuild_kbound.sh",
)

ADDITIONAL_ARCHIVED = (
    "docs/research/kbound/KBound_Reproduction.ipynb",
    "docs/research/kbound/notebooks/00_KBound_Reproduction.ipynb",
    "docs/research/kbound/notebooks/09_Conclusions_and_Reproducibility.ipynb",
    "docs/research/kbound/REVIEWER_REPRO_PACKET.md",
    "research_lock/KBOUND_FINDINGS_ANALYSIS.md",
)

ADDITIONAL_REROUTED = (
    "docs/research/kbound/RUN_FINAL_SHOWCASE.md",
    "docs/research/kbound/RUNSHEET_WAVE7.md",
    "docs/research/kbound/gapclose_wave5/RUNSHEET_WAVE6.md",
    "docs/research/kbound/g5_finalize/G5_STATUS_AND_FINALIZE.md",
    "docs/research/kbound/g5_finalize/run_g5_finalize.sh",
    "docs/research/kbound/notebooks/05_TTA_CIFAR_and_Online.ipynb",
    "docs/research/kbound/reports/READINESS_85PLUS.md",
    "docs/research/kbound/scripts/run_85plus_readiness.sh",
    "docs/research/kbound/scripts/run_full_panel.sh",
    "research_lock/KBOUND_HEADLINE_FINDINGS_v3.json",
)

ALL_ARCHIVED = frozenset(ARCHIVED_ONLY + ADDITIONAL_ARCHIVED)
ALL_REROUTED = frozenset(REPLACED_ACTIVE + ADDITIONAL_REROUTED)

REPLACEMENT_AUTHORITIES = {
    "docs/research/kbound/runbooks/finish_empirical_training.sh": ("docs/research/kbound/scripts/build_pdfs.sh"),
    "docs/research/kbound/scripts/run_final_showcase.sh": ("docs/research/kbound/runbooks/release_candidate.sh"),
    "experiments/kbound/results/KGA_ELARA_ANALYSIS.md": "docs/research/kbound/claim_ledger.json",
    "research_lock/KBOUND_6_DATASET_PANEL_v1.yaml": "research_lock/KBOUND_6_DATASET_PANEL_v2.yaml",
    "scripts/rebuild_kbound.sh": "docs/research/kbound/scripts/build_pdfs.sh",
    "docs/research/kbound/KBound_Reproduction.ipynb": ("docs/research/kbound/notebooks/00_KBound_Master_Guide.ipynb"),
    "docs/research/kbound/notebooks/00_KBound_Reproduction.ipynb": (
        "docs/research/kbound/notebooks/00_KBound_Master_Guide.ipynb"
    ),
    "docs/research/kbound/notebooks/09_Conclusions_and_Reproducibility.ipynb": (
        "docs/research/kbound/notebooks/00_KBound_Master_Guide.ipynb"
    ),
    "docs/research/kbound/REVIEWER_REPRO_PACKET.md": ("docs/research/kbound/runbooks/CAMERA_READY_RUNBOOK.md"),
    "docs/research/kbound/RUN_FINAL_SHOWCASE.md": ("docs/research/kbound/runbooks/release_candidate.sh"),
    "docs/research/kbound/RUNSHEET_WAVE7.md": "docs/research/kbound/claim_ledger.json",
    "docs/research/kbound/gapclose_wave5/RUNSHEET_WAVE6.md": ("docs/research/kbound/claim_ledger.json"),
    "docs/research/kbound/g5_finalize/G5_STATUS_AND_FINALIZE.md": ("docs/research/kbound/claim_ledger.json"),
    "docs/research/kbound/g5_finalize/run_g5_finalize.sh": ("docs/research/kbound/claim_ledger.json"),
    "docs/research/kbound/notebooks/05_TTA_CIFAR_and_Online.ipynb": (
        "docs/research/kbound/notebooks/00_KBound_Master_Guide.ipynb"
    ),
    "docs/research/kbound/reports/READINESS_85PLUS.md": ("docs/research/kbound/runbooks/release_candidate.sh"),
    "docs/research/kbound/scripts/run_85plus_readiness.sh": ("docs/research/kbound/runbooks/release_candidate.sh"),
    "docs/research/kbound/scripts/run_full_panel.sh": ("docs/research/kbound/runbooks/release_candidate.sh"),
    "research_lock/KBOUND_HEADLINE_FINDINGS_v3.json": ("docs/research/kbound/claim_ledger.json"),
    "research_lock/KBOUND_FINDINGS_ANALYSIS.md": ("docs/research/kbound/claim_ledger.json"),
}

ARCHIVED_PROTOCOL_NAMES = (
    "CAMELYON17_FULLSCALE_PROTOCOL_B_v1",
    "CLEAN_COMPLEMENTARY_TRANSFER_PROTOCOL_v1",
    "GATE_U_CROSS_DOMAIN_PROTOCOL_v1",
    "INDEPENDENT_EXTERNAL_PROTOCOL_v1",
    "MULTIMODAL_RELIABILITY_PROTOCOL_v1",
    "NATURAL_SHIFT_SEALED_PROTOCOL_v1",
    "NATURAL_WIN_PROTOCOL_v1",
    "TARGET_LABEL_LIGHT_PPI_PROTOCOL_D25_v1",
    "TARGET_LABEL_LIGHT_PROBE_PROTOCOL_v1",
    "WIN_HUNT_v2_PROTOCOL",
    "WIN_HUNT_v3_PROTOCOL",
    "WIN_HUNT_v4_PROTOCOL",
    "WIN_HUNT_v5_PROTOCOL_SHELL",
)


def _git_bytes(relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _load_verifier_module():
    spec = importlib.util.spec_from_file_location("kbound_historical_seal_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _authored_executable_sources():
    """Yield maintained Python/shell sources without tree traversal."""

    for relative in repository_inventory.validated_authored_source_paths(
        repo=ROOT,
        revision="HEAD",
        suffixes=(".py", ".sh"),
    ):
        if "tests" not in Path(relative).parts:
            yield relative, ROOT / relative


def test_retired_surfaces_are_byte_preserved_and_manifested() -> None:
    manifest_path = ROOT / RETIRED_MANIFEST
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["schema_version"] == 1
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["policy"] == "retired_historical_surfaces_not_current_authority"

    records = manifest["records"]
    expected_paths = list(ARCHIVED_ONLY + REPLACED_ACTIVE + ADDITIONAL_ARCHIVED + ADDITIONAL_REROUTED)
    assert [record["original_path"] for record in records] == expected_paths
    for record in records:
        original_path = record["original_path"]
        archive_path = (ARCHIVE / "retired_tree" / original_path).as_posix()
        assert record["archive_path"] == archive_path
        archived_bytes = (ROOT / archive_path).read_bytes()
        assert archived_bytes == _git_bytes(original_path)
        assert record["bytes"] == len(archived_bytes)
        assert record["sha256"] == hashlib.sha256(archived_bytes).hexdigest()
        replacement = REPLACEMENT_AUTHORITIES.get(original_path)
        if replacement is None:
            assert record["replacement_authority"] is None
            assert record["replacement_status"] == "historical_only_no_current_successor"
        else:
            assert record["replacement_authority"] == replacement
            assert (ROOT / replacement).is_file(), replacement
        if original_path in ALL_ARCHIVED:
            assert record["disposition"] == "archived_only"
            assert not (ROOT / original_path).exists(), original_path
        else:
            assert original_path in ALL_REROUTED
            assert record["disposition"] == "retired_or_rerouted_active_stub"
            assert (ROOT / original_path).is_file(), original_path

    archived_files = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / ARCHIVE / "retired_tree").rglob("*") if path.is_file()
    }
    assert archived_files == {record["archive_path"] for record in records}
    canonical = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    assert manifest_text == canonical


def test_historical_nine_track_verifier_is_read_only_and_passes() -> None:
    script = VERIFIER
    script_text = script.read_text(encoding="utf-8")
    assert "build_seal" not in script_text
    assert "write_yaml_sidecar" not in script_text
    assert "LOCKED_ANALYSIS_FINDINGS.md" not in script_text
    assert ".write_text(" not in script_text
    verify = subprocess.run(
        [sys.executable, str(script), "--verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "historical" in verify.stdout.lower()

    archived_findings_original = (
        ROOT / "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_FINDINGS.md"
    )
    assert not archived_findings_original.exists()
    no_mode = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert no_mode.returncode == 2
    assert "read-only historical verifier" in no_mode.stderr.lower()
    assert not archived_findings_original.exists()


def test_maintained_manuscript_validator_consumes_the_archived_historical_seal() -> None:
    result = subprocess.run(
        [sys.executable, str(MANUSCRIPT_VALIDATOR)],
        cwd=ROOT,
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (ROOT / "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json").exists()


def test_historical_verifier_rejects_tampered_yaml_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    verifier = _load_verifier_module()
    tampered = tmp_path / "NINE_TRACK_LOCK_SEAL_v1.yaml"
    tampered.write_bytes(verifier.LOCK_YAML.read_bytes() + b"\n# forged metadata\n")
    monkeypatch.setattr(verifier, "LOCK_YAML", tampered)

    _, errors = verifier.verify_historical_seal()

    assert errors
    assert any("YAML" in error or "receipt" in error for error in errors)


def test_historical_verifier_rejects_tandem_seal_and_sidecar_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    verifier = _load_verifier_module()
    seal = json.loads(verifier.SEAL_JSON.read_text(encoding="utf-8"))
    seal["policy"] = "forged historical policy"
    seal_bytes = (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode("utf-8")
    seal_path = tmp_path / "LOCK_SEAL.json"
    seal_path.write_bytes(seal_bytes)
    sidecar_path = tmp_path / "LOCK_SEAL.sha256"
    sidecar_path.write_text(
        f"{hashlib.sha256(seal_bytes).hexdigest()}  LOCK_SEAL.json\n",
        encoding="ascii",
    )
    monkeypatch.setattr(verifier, "SEAL_JSON", seal_path)
    monkeypatch.setattr(verifier, "SEAL_SHA", sidecar_path)

    _, errors = verifier.verify_historical_seal()

    assert errors
    assert any("baseline" in error or "retired" in error for error in errors)


def test_historical_verifier_rejects_malformed_checksum_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    verifier = _load_verifier_module()
    sidecar = tmp_path / "LOCK_SEAL.sha256"
    sidecar.write_text(
        f"{hashlib.sha256(verifier.SEAL_JSON.read_bytes()).hexdigest()} garbage\n",
        encoding="ascii",
    )
    monkeypatch.setattr(verifier, "SEAL_SHA", sidecar)

    _, errors = verifier.verify_historical_seal()

    assert errors
    assert any("checksum" in error or "sidecar" in error for error in errors)


def test_historical_verifier_rejects_tampered_retired_receipt_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    verifier = _load_verifier_module()
    retired_manifest = ARCHIVE / "RETIRED_SURFACES_MANIFEST.json"
    manifest = json.loads((ROOT / retired_manifest).read_text(encoding="utf-8"))
    record = next(
        row
        for row in manifest["records"]
        if row["original_path"] == "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json"
    )
    record["sha256"] = "0" * 64
    tampered = tmp_path / "RETIRED_SURFACES_MANIFEST.json"
    tampered.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "RETIRED_MANIFEST", tampered, raising=False)

    _, errors = verifier.verify_historical_seal()

    assert errors
    assert any("retired" in error or "baseline" in error for error in errors)


@pytest.mark.parametrize("path_form", ["absolute", "traversal"])
def test_historical_verifier_rejects_unsafe_authority_archive_paths(
    tmp_path: Path,
    monkeypatch,
    path_form: str,
) -> None:
    verifier = _load_verifier_module()
    manifest = json.loads(verifier.ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
    record = next(
        row
        for row in manifest["records"]
        if row["original_path"] == "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_FINDINGS.md"
    )
    valid_archive_path = record["archive_path"]
    if path_form == "absolute":
        record["archive_path"] = str((ROOT / valid_archive_path).resolve())
    else:
        marker = "/tree/"
        prefix, suffix = valid_archive_path.split(marker, 1)
        record["archive_path"] = f"{prefix}/tree/../tree/{suffix}"
    tampered = tmp_path / "MANIFEST.json"
    tampered.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "ARCHIVE_MANIFEST", tampered)

    _, errors = verifier.verify_historical_seal()

    assert errors
    assert any("archive_path" in error or "canonical" in error for error in errors)


@pytest.mark.parametrize("malformation", ["duplicate", "nonfinite"])
def test_historical_verifier_rejects_non_strict_authority_manifest_json(
    tmp_path: Path,
    monkeypatch,
    malformation: str,
) -> None:
    verifier = _load_verifier_module()
    manifest_text = verifier.ARCHIVE_MANIFEST.read_text(encoding="utf-8")
    if malformation == "duplicate":
        manifest_text = manifest_text.replace("{\n", '{\n  "records": [],\n', 1)
    else:
        manifest_text = manifest_text.replace("{\n", '{\n  "nonfinite": NaN,\n', 1)
    tampered = tmp_path / "MANIFEST.json"
    tampered.write_text(manifest_text, encoding="utf-8")
    monkeypatch.setattr(verifier, "ARCHIVE_MANIFEST", tampered)

    _, errors = verifier.verify_historical_seal()

    assert errors
    assert any("JSON" in error or "manifest" in error for error in errors)


def test_current_result_manifest_routes_the_historical_seal_to_the_archive() -> None:
    manifest = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text(encoding="utf-8")
    )
    seal = manifest["nine_track_lock_seal"]
    assert seal["status"] == "historical_policy_only"
    assert seal["current_policy_authority"] is False
    assert seal["path"].startswith(f"{ARCHIVE.as_posix()}/retired_tree/")
    assert seal["research_lock"].startswith(f"{ARCHIVE.as_posix()}/retired_tree/")
    assert seal["verify"] == "python3 docs/research/kbound/scripts/seal_nine_track_lock.py --verify"


def test_stale_headline_findings_is_a_non_authoritative_canonical_route() -> None:
    headline = json.loads((ROOT / "research_lock/KBOUND_HEADLINE_FINDINGS_v3.json").read_text())

    assert headline == {
        "manifest": "KBOUND_HEADLINE_FINDINGS_v3",
        "status": "historical_policy_only",
        "authority": False,
        "current_policy_authority": False,
        "headline_promotion_eligible": False,
        "numeric_release_eligible": False,
        "archived_original": (
            "docs/research/kbound/archive/superseded_empirical_authorities_2026-09-02/"
            "retired_tree/research_lock/KBOUND_HEADLINE_FINDINGS_v3.json"
        ),
        "superseded_by": "docs/research/kbound/claim_ledger.json",
        "note": (
            "Historical headline findings are preserved only for provenance; canonical claim wording "
            "and promotion status come from the current claim ledger."
        ),
    }


def test_current_data_and_storage_indexes_do_not_route_to_retired_originals() -> None:
    data_guide = (ROOT / "DATA.md").read_text(encoding="utf-8")
    # An authenticated archive link is valid provenance; a bare active v1
    # execution path is not. Reject the stale route, not the historical name.
    assert "`research_lock/KBOUND_6_DATASET_PANEL_v1.yaml" not in data_guide
    assert f"{ARCHIVE.as_posix()}/retired_tree/research_lock/KBOUND_6_DATASET_PANEL_v1.yaml" in data_guide
    assert "research_lock/KBOUND_6_DATASET_PANEL_v2.yaml" in data_guide

    storage_manifest = (ROOT / "docs/research/kbound/STORAGE_MANIFEST.json").read_text(encoding="utf-8")
    old_findings = "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_FINDINGS.md"
    archived_findings = "docs/research/kbound/archive/superseded_empirical_authorities_2026-09-02/tree/" + old_findings
    assert f'"{old_findings}":' not in storage_manifest
    assert archived_findings in storage_manifest


def test_no_runnable_source_names_an_archived_protocol() -> None:
    offenders: dict[str, list[str]] = {}
    for relative, path in _authored_executable_sources():
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [name for name in ARCHIVED_PROTOCOL_NAMES if name in text]
        if hits:
            offenders[relative] = hits
    assert not offenders, offenders


def test_publication_entry_points_only_name_canonical_outputs() -> None:
    forbidden = (
        "kbound_short.pdf",
        "kbound.pdf",
        "K-Bound_paper.pdf",
        "kbound_full_ieee_diagnostic.pdf",
    )
    surfaces = (
        "docs/research/kbound/runbooks/finish_empirical_training.sh",
        "docs/research/kbound/scripts/build_pdfs.sh",
        "docs/research/kbound/scripts/run_final_showcase.sh",
        "docs/research/kbound/notebooks/00_KBound_Master_Guide.ipynb",
        "scripts/rebuild_kbound.sh",
        "scripts/shrink_git_history.sh",
    )
    for relative_path in surfaces:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert not [token for token in forbidden if token in text], relative_path
    combined = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in surfaces)
    assert "kbound_short_final_draft.pdf" in combined
    assert "kbound_tmlr.pdf" in combined
    assert "kbound_short_final_draft.docx" in combined
    finish = (ROOT / "docs/research/kbound/runbooks/finish_empirical_training.sh").read_text(encoding="utf-8")
    # The refusal-only compatibility launcher cannot fix the paper's page count.
    # Its no-execution behavior is tested in test_retired_launcher_boundary.py.
    assert "retired" in finish.lower()


def test_stale_g5_and_showcase_launchers_are_fail_closed() -> None:
    for relative_path in (
        "docs/research/kbound/g5_finalize/run_g5_finalize.sh",
        "docs/research/kbound/scripts/run_85plus_readiness.sh",
        "docs/research/kbound/scripts/run_full_panel.sh",
    ):
        result = subprocess.run(
            ["bash", relative_path],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, relative_path
        assert "retired" in result.stderr.lower(), relative_path
        assert "release_candidate.sh all" in result.stderr, relative_path


def test_no_active_executable_dispatches_a_retired_route() -> None:
    forbidden_routes = (
        "experiments/kbound/poem_aetta/score_official_headtohead.py",
        "docs/research/kbound/scripts/run_final_showcase.sh",
        "scripts/rebuild_kbound.sh",
    )
    compatibility_stubs = {
        "docs/research/kbound/scripts/run_final_showcase.sh",
        "scripts/rebuild_kbound.sh",
    }
    offenders: dict[str, list[str]] = {}
    for relative_path, path in _authored_executable_sources():
        if relative_path in compatibility_stubs:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [route for route in forbidden_routes if route in text]
        if hits:
            offenders[relative_path] = hits
    assert not offenders, offenders


def test_current_reproduction_surfaces_route_to_the_canonical_release() -> None:
    for relative_path in ADDITIONAL_ARCHIVED:
        assert not (ROOT / relative_path).exists(), relative_path

    current_surfaces = (
        "docs/research/kbound/DOCS_INDEX.md",
        "docs/research/kbound/README.md",
        "docs/research/kbound/REPRODUCE.md",
        "docs/research/kbound/RELEASE_CHECKLIST.md",
        "docs/research/kbound/RUN_FINAL_SHOWCASE.md",
        "docs/research/kbound/RUNSHEET_WAVE7.md",
        "docs/research/kbound/gapclose_wave5/RUNSHEET_WAVE6.md",
        "docs/research/kbound/g5_finalize/G5_STATUS_AND_FINALIZE.md",
        "docs/research/kbound/notebooks/README.md",
        "docs/research/kbound/notebooks/00_KBound_Master_Guide.ipynb",
        "docs/research/kbound/notebooks/05_TTA_CIFAR_and_Online.ipynb",
        "docs/research/kbound/reports/READINESS_85PLUS.md",
    )
    forbidden = (
        "bash scripts/rebuild_kbound.sh",
        "bash docs/research/kbound/scripts/run_final_showcase.sh",
        "experiments/kbound/poem_aetta/score_official_headtohead.py",
        "K-Bound_paper.pdf",
    )
    for relative_path in current_surfaces:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert not [token for token in forbidden if token in text], relative_path

    for relative_path in (
        "docs/research/kbound/RUN_FINAL_SHOWCASE.md",
        "docs/research/kbound/RUNSHEET_WAVE7.md",
        "docs/research/kbound/gapclose_wave5/RUNSHEET_WAVE6.md",
        "docs/research/kbound/g5_finalize/G5_STATUS_AND_FINALIZE.md",
        "docs/research/kbound/reports/READINESS_85PLUS.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "retired" in text.lower(), relative_path
        assert "release_candidate.sh all" in text, relative_path


def test_smoke_report_uses_factual_labels_and_the_canonical_release_route() -> None:
    text = (ROOT / "docs/research/kbound/scripts/smoke_pipeline_report.py").read_text(encoding="utf-8")
    assert "locked Holm WIN" not in text
    assert "beats-both(pt)" not in text
    assert "run_final_showcase.sh" not in text
    assert "release_candidate.sh all" in text
    assert "lower-regret point flag" in text
