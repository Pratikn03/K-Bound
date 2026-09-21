from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "src/scripts/validate_manuscript_claims.py"
PUBLIC_STORAGE_LOCATIONS = (
    "docs/research/kbound/claim_ledger.json",
    "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json",
    "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
    "docs/research/kbound/audits/phase1_provenance_2026_08_27/provenance_seal.json",
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("kbound_manuscript_storage_boundary", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _public_projection(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, location in enumerate(PUBLIC_STORAGE_LOCATIONS):
        path = tmp_path / location
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"public authority {index}\n".encode()
        path.write_bytes(payload)
        rows.append(
            {
                "expected_location": location,
                "tracked": True,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def test_default_storage_validation_never_touches_protected_so2sat_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    public_rows = _public_projection(tmp_path)
    protected_relative = "experiments/kbound/results/so2sat_synthetic/target-result.json"
    protected = tmp_path / protected_relative
    nonpublic_relative = "private/protected-canary.bin"
    nonpublic = tmp_path / nonpublic_relative
    absolute_canary = tmp_path / "absolute-protected-canary.bin"
    traversal_canary = tmp_path.parent / "traversal-protected-canary.bin"
    manifest = {
        "artifacts": public_rows
        + [
            {
                "expected_location": nonpublic_relative,
                "tracked": True,
                "size_bytes": 999,
                "sha256": "a" * 64,
            },
            {
                "expected_location": protected_relative,
                "tracked": True,
                "size_bytes": 999,
                "sha256": "a" * 64,
            },
            {
                "expected_location": str(absolute_canary),
                "tracked": True,
                "size_bytes": 999,
                "sha256": "a" * 64,
            },
            {
                "expected_location": "../traversal-protected-canary.bin",
                "tracked": True,
                "size_bytes": 999,
                "sha256": "a" * 64,
            },
        ],
        "sealed_evidence_checksums": {
            protected_relative: {
                "status": "present",
                "size_bytes": 999,
                "sha256": "a" * 64,
            }
        },
        "sealed_evidence_summary": {"files": 1, "present": 1, "absent": 0},
        "unsealed_present_artifacts": [
            {
                "path": protected_relative,
                "status": "present_unsealed",
                "current_bytes": 999,
                "current_sha256": "a" * 64,
            }
        ],
    }
    storage = tmp_path / "STORAGE_MANIFEST.json"
    storage.write_text(json.dumps(manifest), encoding="utf-8")
    lock = tmp_path / "LOCK_SEAL.json"
    lock.write_text('{"tracks": {}}\n', encoding="utf-8")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "STORAGE_MANIFEST", storage)
    monkeypatch.setattr(validator, "LOCK_SEAL", lock)

    forbidden = {protected, nonpublic, absolute_canary, traversal_canary}
    real_stat = Path.stat

    def guarded_stat(path: Path, *args: Any, **kwargs: Any):
        if path in forbidden:
            raise AssertionError(f"nonpublic manifest path was statted: {path}")
        return real_stat(path, *args, **kwargs)

    real_hash = validator.file_sha256

    def guarded_hash(path: Path) -> str:
        if path in forbidden:
            raise AssertionError(f"nonpublic manifest path was opened: {path}")
        return cast(str, real_hash(path))

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(validator, "file_sha256", guarded_hash)
    problems: list[str] = []

    counts = validator.validate_storage_manifest(
        problems,
        {
            "public": {"source": PUBLIC_STORAGE_LOCATIONS[0]},
            "protected": {"source": protected_relative},
        },
    )

    assert problems == []
    assert counts == (4, 0)


def test_default_storage_validation_rejects_public_authority_symlink_without_opening_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    public_rows = _public_projection(tmp_path)
    redirected_location = PUBLIC_STORAGE_LOCATIONS[0]
    redirected = tmp_path / redirected_location
    canary = tmp_path / "protected-canary.bin"
    canary.write_bytes(b"must not be opened\n")
    redirected.unlink()
    redirected.symlink_to(canary)

    storage = tmp_path / "STORAGE_MANIFEST.json"
    storage.write_text(json.dumps({"artifacts": public_rows}), encoding="utf-8")
    lock = tmp_path / "LOCK_SEAL.json"
    lock.write_text('{"tracks": {}}\n', encoding="utf-8")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "STORAGE_MANIFEST", storage)
    monkeypatch.setattr(validator, "LOCK_SEAL", lock)

    real_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any):
        if path == redirected or path == canary:
            raise AssertionError("redirected public authority was opened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    problems: list[str] = []

    counts = validator.validate_storage_manifest(problems, {})

    assert counts == (4, 0)
    assert any("symlink" in problem for problem in problems), problems


def test_explicit_storage_authorization_crosses_the_synthetic_access_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    protected_relative = "experiments/kbound/results/so2sat_synthetic/target-result.json"
    protected = tmp_path / protected_relative
    storage = tmp_path / "STORAGE_MANIFEST.json"
    storage.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "expected_location": protected_relative,
                        "tracked": True,
                        "size_bytes": 1,
                        "sha256": "a" * 64,
                    }
                ],
                "sealed_evidence_checksums": {},
                "sealed_evidence_summary": {"files": 0, "present": 0, "absent": 0},
                "unsealed_present_artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    lock = tmp_path / "LOCK_SEAL.json"
    lock.write_text('{"tracks": {}}\n', encoding="utf-8")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "STORAGE_MANIFEST", storage)
    monkeypatch.setattr(validator, "LOCK_SEAL", lock)
    real_stat = Path.stat

    def guarded_stat(path: Path, *args: Any, **kwargs: Any):
        if path == protected:
            raise AssertionError("authorized synthetic protected access")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)

    with pytest.raises(AssertionError, match="authorized synthetic protected access"):
        validator.validate_storage_manifest(
            [],
            {},
            authorize_protected_so2sat=True,
        )


@pytest.mark.parametrize("tampered_archive", [False, True])
def test_authorized_storage_compares_relocated_historical_receipt_without_opening_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_archive: bool,
) -> None:
    """Exercise the complete opt-in comparison with synthetic files only."""
    validator = _load_validator()
    original, relocated = next(iter(validator.HISTORICAL_LOCK_PATH_REMAP.items()))
    payload = b"synthetic historical receipt\n"
    recorded_sha = hashlib.sha256(payload).hexdigest()
    archived_payload = b"changed historical receipt\n" if tampered_archive else payload
    archived = tmp_path / relocated
    archived.parent.mkdir(parents=True)
    archived.write_bytes(archived_payload)
    storage = tmp_path / "STORAGE_MANIFEST.json"
    storage.write_text(json.dumps({
        "artifacts": [],
        "sealed_evidence_checksums": {relocated: {
            "status": "present", "size_bytes": len(archived_payload),
            "sha256": hashlib.sha256(archived_payload).hexdigest(),
        }},
        "sealed_evidence_summary": {"files": 1, "present": 1, "absent": 0},
    }))
    lock = tmp_path / "LOCK_SEAL.json"
    lock.write_text(json.dumps({"tracks": {"synthetic": {"files": {original: {
        "bytes": len(payload), "sha256": recorded_sha,
    }}}}}))
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "STORAGE_MANIFEST", storage)
    monkeypatch.setattr(validator, "LOCK_SEAL", lock)
    real_stat, real_open = Path.stat, Path.open

    def guard_stat(path, *args, **kwargs):
        assert path != tmp_path / original, "retired original was inspected"
        return real_stat(path, *args, **kwargs)

    def guard_open(path, *args, **kwargs):
        assert path != tmp_path / original, "retired original was opened"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guard_stat)
    monkeypatch.setattr(Path, "open", guard_open)
    problems: list[str] = []
    assert validator.validate_storage_manifest(
        problems, {}, authorize_protected_so2sat=True,
    ) == (0, 1)
    if tampered_archive:
        assert any("disagrees with nine-track lock metadata" in item for item in problems), problems
    else:
        assert problems == []


def test_active_claim_corpus_skips_binary_graphics_without_decoding_them(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    root = tmp_path / "repo"
    tex = root / "docs/research/kbound/paper.tex"
    style = root / "docs/research/kbound/vendor/style.sty"
    graphic = root / "docs/research/kbound/figures/plot.png"
    tex.parent.mkdir(parents=True)
    style.parent.mkdir(parents=True)
    graphic.parent.mkdir(parents=True)
    tex.write_text("Text claim from TeX.\n", encoding="utf-8")
    style.write_text("Style claim from STY.\n", encoding="utf-8")
    graphic.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfeprivate-binary-canary")

    text_sources, corpus = validator._active_manuscript_corpus(
        root,
        (tex, graphic, style),
    )

    assert text_sources == (tex, style)
    assert "Text claim from TeX." in corpus
    assert "Style claim from STY." in corpus
    assert "private-binary-canary" not in corpus


def test_active_claim_corpus_rejects_unknown_dependency_type_before_text_io(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "outside.tex"
    external.write_text("external canary must not be read\n", encoding="utf-8")
    linked_text = root / "paper.tex"
    linked_text.symlink_to(external)
    unknown = root / "opaque.bin"
    unknown.write_bytes(b"unclassified dependency")

    with pytest.raises(RuntimeError, match="unsupported active manuscript dependency type"):
        validator._active_manuscript_corpus(root, (linked_text, unknown))


def test_validator_main_does_not_read_binary_members_of_the_active_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_validator()
    graphic = validator.ROOT / "docs/research/kbound/figures/fig_decision_flow.png"
    assert graphic.is_file()
    monkeypatch.setattr(
        validator.manuscript_sources,
        "active_source_paths",
        lambda _root: (graphic,),
    )
    real_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == graphic:
            raise AssertionError("binary active dependency reached text decoding")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    assert validator.main([]) in {0, 1}
