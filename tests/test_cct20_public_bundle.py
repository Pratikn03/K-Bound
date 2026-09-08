from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_cct20_public_bundle as bundle
from docs.research.kbound.scripts.release_privacy import PrivacyError, build_deterministic_zip


def test_release_archive_clis_are_directly_executable_from_outside_repository(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = root / "docs/research/kbound/scripts"

    for name in ("build_cct20_public_bundle.py", "build_anonymous_supplement.py"):
        completed = subprocess.run(
            [sys.executable, str(scripts / name), "--help"],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("environment_ok", [True, False])
def test_public_bundle_cli_verifies_exact_python_content_before_bundle_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment_ok: bool
) -> None:
    calls: list[str] = []

    def verify_selected_profile(lock: Path, profile: Path) -> None:
        assert lock.name == "requirements-release-macos-arm64.lock.txt"
        assert profile.name == "release_python_environment_macos_arm64_v2.json"
        calls.append("environment")
        if not environment_ok:
            raise ValueError("synthetic Python content rejection")

    monkeypatch.setattr(bundle.verify_python_environment, "verify_exact_content_profile", verify_selected_profile)
    monkeypatch.setattr(
        bundle,
        "verify_public_bundle",
        lambda _path, **_kwargs: calls.append("bundle") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_cct20_public_bundle.py", "--check", "--output", str(tmp_path / "bundle.zip")],
    )

    if environment_ok:
        assert bundle.main() == 0
        assert calls == ["environment", "bundle"]
    else:
        with pytest.raises(ValueError, match="synthetic Python content rejection"):
            bundle.main()
        assert calls == ["environment"]


def test_public_bundle_portable_check_runs_full_bundle_semantics_without_release_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    release = tmp_path / "cct20_release_manifest.json"
    monkeypatch.setattr(
        bundle,
        "verify_release_python_content",
        lambda: (_ for _ in ()).throw(AssertionError("portable verification must not inspect the build runtime")),
    )
    monkeypatch.setattr(
        bundle,
        "verify_public_bundle",
        lambda _path, **kwargs: calls.append(kwargs) or {"status": "PASS"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_cct20_public_bundle.py",
            "--portable-check",
            "--release-manifest",
            str(release),
            "--output",
            str(tmp_path / "bundle.zip"),
        ],
    )

    assert bundle.main() == 0
    assert calls == [{"source_release_path": release}]


def test_portable_verifier_rejects_an_internally_valid_bundle_for_a_stale_source_release(tmp_path: Path) -> None:
    release = tmp_path / "cct20_release_manifest.json"
    release.write_bytes(b'{"schema":"first"}\n')
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)
    assert bundle.verify_public_bundle(output, source_release_path=release)["status"] == "PASS"

    release.write_bytes(b'{"schema":"second"}\n')
    with pytest.raises(PrivacyError, match="source release"):
        bundle.verify_public_bundle(output, source_release_path=release)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _rewrite_bundle(source: Path, output: Path, mutate: object) -> None:
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    assert callable(mutate)
    mutate(manifest, entries)
    entries["manifest.json"] = _canonical(manifest)
    build_deterministic_zip(output, entries)


def test_public_bundle_is_portable_content_addressed_and_offline_verifiable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    private_path = "/Us" + "ers/private/raw"
    result = json.dumps({"metric": 0.75, "source_path": private_path}).encode() + b"\n"
    result_path = source / "result.json"
    result_path.write_bytes(result)
    checkpoint = b"private checkpoint bytes"
    checkpoint_path = source / "model.ckpt"
    checkpoint_path.write_bytes(checkpoint)
    release = {
        "schema": "fixture",
        "generated_artifacts": {
            "result": {
                "path": str(result_path),
                "bytes": len(result),
                "sha256": _sha(result),
            }
        },
        "upstream_artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "bytes": len(checkpoint),
                "sha256": _sha(checkpoint),
            }
        },
    }
    release_path = source / "cct20_release_manifest.json"
    release_bytes = json.dumps(release, indent=2).encode() + b"\n"
    release_path.write_bytes(release_bytes)
    output = tmp_path / "cct20-public.zip"

    receipt = bundle.build_public_bundle(release_path, output)

    assert receipt["archive_sha256"] == _sha(output.read_bytes())
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        assert names == sorted(names)
        assert archive.read("manifest.json") == _canonical(manifest)
        assert all(not name.startswith(("/", "../")) for name in names)
        assert all(str(tmp_path).encode() not in archive.read(name) for name in names)
        assert checkpoint not in (archive.read(name) for name in names)
        descriptor = manifest["nonredistributable_objects"][0]
        assert descriptor == {
            "bytes": len(checkpoint),
            "content_role": "upstream_artifacts.checkpoint",
            "descriptor_only": True,
            "sha256": _sha(checkpoint),
        }
        copied = manifest["objects"][0]
        assert copied["bundle_path"].startswith("objects/")
        assert copied["source_sha256"] == _sha(result)
        assert copied["published_sha256"] != copied["source_sha256"]
        assert copied["normalization"] == "canonical-json-with-portable-path-descriptors"

    relocated = tmp_path / "fresh" / "renamed.zip"
    relocated.parent.mkdir()
    shutil.copyfile(output, relocated)
    assert bundle.verify_public_bundle(relocated)["status"] == "PASS"
    shutil.rmtree(source)
    assert bundle.verify_public_bundle(relocated)["status"] == "PASS"


def test_public_bundle_rejects_duplicate_nonfinite_and_unbound_descriptors(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out.zip"
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema":"x","schema":"y"}\n')
    with pytest.raises(PrivacyError, match="duplicate JSON key"):
        bundle.build_public_bundle(duplicate, output)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"score":NaN}\n')
    with pytest.raises(PrivacyError, match="non-finite"):
        bundle.build_public_bundle(nonfinite, output)

    unbound = tmp_path / "unbound.json"
    unbound.write_bytes(json.dumps({"checkpoint": {"path": str(tmp_path / "missing.ckpt")}}).encode())
    with pytest.raises(PrivacyError, match="hash-bound"):
        bundle.build_public_bundle(unbound, output)

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    false_descriptor = tmp_path / "false-descriptor.json"
    false_descriptor.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": "0" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PrivacyError, match="commitment mismatch"):
        bundle.build_public_bundle(false_descriptor, output)

    for label, claimed_bytes, claimed_sha in (
        ("boolean bytes", True, _sha(checkpoint.read_bytes())),
        ("uppercase hash", checkpoint.stat().st_size, _sha(checkpoint.read_bytes()).upper()),
    ):
        malformed = tmp_path / f"{label.replace(' ', '-')}.json"
        malformed.write_text(
            json.dumps(
                {
                    "checkpoint": {
                        "path": str(checkpoint),
                        "bytes": claimed_bytes,
                        "sha256": claimed_sha,
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(PrivacyError, match="non-redistributable"):
            bundle.build_public_bundle(malformed, tmp_path / f"{malformed.stem}.zip")


def test_nonredistributable_sources_are_verified_without_loading_them_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "large-model.ckpt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": _sha(checkpoint.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == checkpoint:
            raise AssertionError("descriptor verification must stream large evidence")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    receipt = bundle.build_public_bundle(release, tmp_path / "bundle.zip")
    assert receipt["status"] == "PASS"


def test_nonredistributable_source_is_hashed_once_and_may_be_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "dataset.tar.gz"
    raw.write_bytes(b"raw dataset bytes")
    digest = _sha(raw.read_bytes())
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "first_raw_archive": {
                    "path": str(raw),
                    "bytes": raw.stat().st_size,
                    "sha256": digest,
                },
                "second_raw_archive": {
                    "path": str(raw),
                    "bytes": raw.stat().st_size,
                    "sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )
    calls = 0
    real_file_sha256 = bundle.file_sha256

    def counted_file_sha256(path: Path) -> str:
        nonlocal calls
        calls += 1
        return real_file_sha256(path)

    monkeypatch.setattr(bundle, "file_sha256", counted_file_sha256)
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)
    assert calls == 1

    raw.unlink()
    with pytest.raises(PrivacyError, match="missing"):
        bundle.build_public_bundle(release, tmp_path / "strict.zip")
    bundle.build_public_bundle(
        release,
        tmp_path / "offline.zip",
        descriptor_mode=bundle.DESCRIPTOR_MODE_OFFLINE,
    )


def test_offline_descriptor_mode_is_explicit_and_never_passes_as_strict(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "unavailable.ckpt"
    payload = b"checkpoint known from the sealed source machine"
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "path": str(missing),
                    "bytes": len(payload),
                    "sha256": _sha(payload),
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "offline.zip"

    receipt = bundle.build_public_bundle(
        release,
        output,
        descriptor_mode=bundle.DESCRIPTOR_MODE_OFFLINE,
    )
    assert receipt["descriptor_source_verification"] == bundle.DESCRIPTOR_MODE_OFFLINE
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["descriptor_source_verification"] == bundle.DESCRIPTOR_MODE_OFFLINE
    with pytest.raises(PrivacyError, match="offline descriptor"):
        bundle.verify_public_bundle(output)
    assert (
        bundle.verify_public_bundle(output, allow_offline_descriptors=True)["descriptor_source_verification"]
        == bundle.DESCRIPTOR_MODE_OFFLINE
    )

    with pytest.raises(PrivacyError, match="descriptor mode"):
        bundle.build_public_bundle(
            release,
            tmp_path / "invalid.zip",
            descriptor_mode="sometimes-check",
        )


def test_verified_replacement_is_explicit_atomic_and_idempotent(tmp_path: Path) -> None:
    release = tmp_path / "release.json"
    release.write_bytes(b'{"schema":"first"}\n')
    output = tmp_path / "bundle.zip"
    first = bundle.build_public_bundle(release, output)
    first_inode = output.stat().st_ino
    first_mtime = output.stat().st_mtime_ns

    with pytest.raises(PrivacyError, match="already exists"):
        bundle.build_public_bundle(release, output)

    repeated = bundle.build_public_bundle(release, output, replace_verified=True)
    assert repeated["archive_sha256"] == first["archive_sha256"]
    assert output.stat().st_ino == first_inode
    assert output.stat().st_mtime_ns == first_mtime

    release.write_bytes(b'{"schema":"second"}\n')
    replaced = bundle.build_public_bundle(release, output, replace_verified=True)
    assert replaced["archive_sha256"] != first["archive_sha256"]
    assert bundle.verify_public_bundle(output)["archive_sha256"] == replaced["archive_sha256"]

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")
    with pytest.raises(PrivacyError):
        bundle.build_public_bundle(release, corrupt, replace_verified=True)
    assert corrupt.read_bytes() == b"not a zip"


def test_verified_replacement_rejects_an_output_swap_after_candidate_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release.json"
    release.write_bytes(b'{"schema":"first"}\n')
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)
    original_verify = bundle.verify_public_bundle

    def swap_after_candidate(path: Path, **kwargs: object) -> dict[str, object]:
        receipt = original_verify(path, **kwargs)
        if path != output:
            output.write_bytes(b"attacker swap")
        return receipt

    monkeypatch.setattr(bundle, "verify_public_bundle", swap_after_candidate)
    with pytest.raises(PrivacyError, match="changed.*replacement"):
        bundle.build_public_bundle(release, output, replace_verified=True)
    assert output.read_bytes() == b"attacker swap"


def test_verified_replacement_rejects_a_swap_at_the_atomic_displacement_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release.json"
    release.write_bytes(b'{"schema":"first"}\n')
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)
    release.write_bytes(b'{"schema":"second"}\n')
    original_replace = bundle.os.replace

    def swap_before_displacement(source: Path, destination: Path) -> None:
        if Path(source) == output:
            output.write_bytes(b"attacker at displacement boundary")
        original_replace(source, destination)

    monkeypatch.setattr(bundle.os, "replace", swap_before_displacement)
    with pytest.raises(PrivacyError, match="changed.*replacement"):
        bundle.build_public_bundle(release, output, replace_verified=True)
    assert output.read_bytes() == b"attacker at displacement boundary"


def test_publication_rejects_a_candidate_swap_after_relocated_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release.json"
    release.write_bytes(b'{"schema":"first"}\n')
    output = tmp_path / "bundle.zip"
    original_verify = bundle.verify_public_bundle

    def swap_candidate_after_verify(path: Path, **kwargs: object) -> dict[str, object]:
        receipt = original_verify(path, **kwargs)
        if path != output:
            candidates = list(tmp_path.glob(f".{output.name}.*.candidate"))
            assert len(candidates) == 1
            candidates[0].write_bytes(b"attacker candidate swap")
        return receipt

    monkeypatch.setattr(bundle, "verify_public_bundle", swap_candidate_after_verify)
    with pytest.raises(PrivacyError, match="candidate.*verified"):
        bundle.build_public_bundle(release, output)
    assert not output.exists()


def test_source_paths_allow_trusted_alias_and_in_root_symlink_only(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    evidence = real / "evidence.txt"
    evidence.write_bytes(b"portable\n")
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "evidence": {
                    "path": str(linked / "evidence.txt"),
                    "bytes": evidence.stat().st_size,
                    "sha256": _sha(evidence.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    assert bundle.build_public_bundle(release, tmp_path / "linked.zip")["status"] == "PASS"

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    try:
        escaped = tmp_path / "escaped"
        escaped.symlink_to(outside, target_is_directory=True)
        outside_evidence = outside / "evidence.txt"
        outside_evidence.write_bytes(b"outside\n")
        release.write_text(
            json.dumps(
                {
                    "evidence": {
                        "path": str(escaped / "evidence.txt"),
                        "bytes": outside_evidence.stat().st_size,
                        "sha256": _sha(outside_evidence.read_bytes()),
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(PrivacyError, match="trusted root"):
            bundle.build_public_bundle(release, tmp_path / "escaped.zip")
    finally:
        shutil.rmtree(outside)

    release.write_text(json.dumps({"schema": "alias-fixture"}), encoding="utf-8")
    resolved_release = release.resolve()
    if resolved_release.is_relative_to(Path("/private/var")):
        alias_release = Path("/var") / resolved_release.relative_to("/private/var")
        assert alias_release != resolved_release
        assert bundle.build_public_bundle(alias_release, tmp_path / "var-alias.zip")["status"] == "PASS"


def test_builder_and_verifier_enforce_aggregate_member_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"a" * 32)
    second.write_bytes(b"b" * 32)
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                name: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path.read_bytes()),
                }
                for name, path in (("first", first), ("second", second))
            }
        ),
        encoding="utf-8",
    )
    good = tmp_path / "good.zip"
    bundle.build_public_bundle(release, good)
    with zipfile.ZipFile(good) as archive:
        total = sum(info.file_size for info in archive.infolist())

    monkeypatch.setattr(bundle, "MAX_PUBLIC_UNCOMPRESSED_BYTES", total - 1)
    with pytest.raises(PrivacyError, match="aggregate uncompressed"):
        bundle.verify_public_bundle(good)
    with pytest.raises(PrivacyError, match="aggregate uncompressed"):
        bundle.build_public_bundle(release, tmp_path / "limited.zip")


def test_public_bundle_verification_streams_the_archive_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"schema":"fixture"}\n')
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(source, output)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == output:
            raise AssertionError("the complete archive must not be loaded into memory")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    assert bundle.verify_public_bundle(output)["status"] == "PASS"


@pytest.mark.parametrize(
    "suffix",
    [
        ".7z",
        ".arrow",
        ".bz2",
        ".db",
        ".docx",
        ".feather",
        ".gz",
        ".joblib",
        ".npy",
        ".npz",
        ".parquet",
        ".pdf",
        ".pickle",
        ".png",
        ".rar",
        ".sqlite",
        ".xz",
    ],
)
def test_broad_opaque_formats_are_descriptor_only(tmp_path: Path, suffix: str) -> None:
    opaque = tmp_path / f"artifact{suffix}"
    opaque.write_bytes(b"opaque payload")
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "artifact": {
                    "path": str(opaque),
                    "bytes": opaque.stat().st_size,
                    "sha256": _sha(opaque.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["objects"] == []
    assert len(manifest["nonredistributable_objects"]) == 1


def test_binary_payload_disguised_as_text_is_descriptor_only(tmp_path: Path) -> None:
    disguised = tmp_path / "not-really-text.txt"
    disguised.write_bytes(b"PK\x03\x04\x00binary archive payload")
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "artifact": {
                    "path": str(disguised),
                    "bytes": disguised.stat().st_size,
                    "sha256": _sha(disguised.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["objects"] == []
    assert manifest["nonredistributable_objects"][0]["content_role"] == "artifact"


def test_public_verifier_rejects_forged_schema_provenance_and_non_content_addresses(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"schema":"fixture"}\n')
    good = tmp_path / "good.zip"
    bundle.build_public_bundle(source, good)

    def forge_source(manifest: dict[str, object], _entries: dict[str, bytes]) -> None:
        manifest["source_release"] = {"bytes": -1, "sha256": "bogus"}

    forged_source = tmp_path / "forged-source.zip"
    _rewrite_bundle(good, forged_source, forge_source)
    with pytest.raises(PrivacyError, match="source_release"):
        bundle.verify_public_bundle(forged_source)

    def add_unknown(manifest: dict[str, object], _entries: dict[str, bytes]) -> None:
        manifest["unexpected"] = True

    unknown = tmp_path / "unknown.zip"
    _rewrite_bundle(good, unknown, add_unknown)
    with pytest.raises(PrivacyError, match="schema fields"):
        bundle.verify_public_bundle(unknown)

    def rename_portable(manifest: dict[str, object], entries: dict[str, bytes]) -> None:
        portable = manifest["portable_release"]
        assert isinstance(portable, dict)
        old = portable["bundle_path"]
        assert isinstance(old, str)
        entries["objects/not-a-content-hash"] = entries.pop(old)
        portable["bundle_path"] = "objects/not-a-content-hash"

    renamed = tmp_path / "renamed.zip"
    _rewrite_bundle(good, renamed, rename_portable)
    with pytest.raises(PrivacyError, match="content-address"):
        bundle.verify_public_bundle(renamed)


def test_public_verifier_rejects_tampered_readme_and_malformed_object_rows(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_bytes(b"evidence\n")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "evidence": {
                    "path": str(evidence),
                    "bytes": evidence.stat().st_size,
                    "sha256": _sha(evidence.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    good = tmp_path / "good.zip"
    bundle.build_public_bundle(source, good)

    def tamper_readme(_manifest: dict[str, object], entries: dict[str, bytes]) -> None:
        entries["README.md"] = b"different\n"

    tampered = tmp_path / "tampered.zip"
    _rewrite_bundle(good, tampered, tamper_readme)
    with pytest.raises(PrivacyError, match="README"):
        bundle.verify_public_bundle(tampered)

    def forge_row(manifest: dict[str, object], _entries: dict[str, bytes]) -> None:
        rows = manifest["objects"]
        assert isinstance(rows, list) and rows
        assert isinstance(rows[0], dict)
        rows[0]["source_bytes"] = -9
        rows[0]["normalization"] = "made-up"

    forged = tmp_path / "forged-row.zip"
    _rewrite_bundle(good, forged, forge_row)
    with pytest.raises(PrivacyError, match="object ledger"):
        bundle.verify_public_bundle(forged)


def test_checkpoint_audit_json_is_redistributable_evidence(tmp_path: Path) -> None:
    audit = tmp_path / "independent_checkpoint_audit.json"
    audit.write_bytes(b'{"independent":true}\n')
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "upstream_artifacts": {
                    "checkpoint_audit": {
                        "path": str(audit),
                        "bytes": audit.stat().st_size,
                        "sha256": _sha(audit.read_bytes()),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["nonredistributable_objects"] == []
    assert [row["content_role"] for row in manifest["objects"]] == ["upstream_artifacts.checkpoint_audit"]


@pytest.mark.parametrize("suffix", [".bin", ".onnx", ".h5", ".pkl", ".zip"])
def test_opaque_binary_formats_are_descriptor_only(tmp_path: Path, suffix: str) -> None:
    opaque = tmp_path / f"artifact{suffix}"
    opaque.write_bytes(b"opaque private model or archive bytes")
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "artifact": {
                    "path": str(opaque),
                    "bytes": opaque.stat().st_size,
                    "sha256": _sha(opaque.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["objects"] == []
        assert manifest["nonredistributable_objects"][0]["sha256"] == _sha(opaque.read_bytes())
        assert opaque.read_bytes() not in [archive.read(name) for name in archive.namelist()]


def test_recursive_evidence_reference_is_bound_without_infinite_expansion(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"next_path": second.name}), encoding="utf-8")
    second.write_text(json.dumps({"next_path": first.name}), encoding="utf-8")
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"start_path": first.name}), encoding="utf-8")
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)
    assert bundle.verify_public_bundle(output)["status"] == "PASS"
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        normalized = [
            json.loads(archive.read(row["bundle_path"]))
            for row in manifest["objects"]
            if row["normalization"].startswith("canonical-json")
        ]
    back_reference = next(
        item["next_path_source_reference"] for item in normalized if "next_path_source_reference" in item
    )
    assert back_reference["normalization"] == "recursive-source-reference"
    assert back_reference["source_sha256"] == _sha(first.read_bytes())
    assert str(tmp_path) not in json.dumps(back_reference)


def test_yaml_evidence_is_normalized_recursively_with_relative_object_closure(
    tmp_path: Path,
) -> None:
    private_root = "/Vol" + "umes/T9"
    evidence = tmp_path / "evidence.txt"
    evidence.write_bytes(b"portable evidence\n")
    raw_archive = tmp_path / "dataset.tar.gz"
    raw_archive.write_bytes(b"nonredistributable raw bytes")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "\n".join(
            [
                "evidence:",
                "  path: evidence.txt",
                f"  file_sha256: {_sha(evidence.read_bytes())}",
                "raw_archive:",
                "  path: dataset.tar.gz",
                f"  bytes: {raw_archive.stat().st_size}",
                f"  sha256: {_sha(raw_archive.read_bytes())}",
                f"private_output_path: {private_root}/private/output",
                f"audit_note: mounted {private_root} filename search",
                "repository: https://github.com/example/project",
                "",
            ]
        ),
        encoding="utf-8",
    )
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "protocol": {
                    "path": str(protocol),
                    "bytes": protocol.stat().st_size,
                    "sha256": _sha(protocol.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bundle.zip"
    bundle.build_public_bundle(release, output)

    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        protocol_row = next(row for row in manifest["objects"] if row["content_role"] == "protocol")
        assert protocol_row["normalization"] == ("canonical-json-from-yaml-with-portable-path-descriptors")
        portable_protocol = json.loads(archive.read(protocol_row["bundle_path"]))
        assert portable_protocol["private_output_path"] == "private-source-location-redacted"
        assert portable_protocol["evidence"]["path"].startswith("objects/")
        assert "path_descriptor" in portable_protocol["raw_archive"]
        all_payload = b"".join(archive.read(name) for name in archive.namelist())
        assert private_root.encode() not in all_payload
        assert raw_archive.read_bytes() not in all_payload

    assert bundle.verify_public_bundle(output)["status"] == "PASS"


def test_public_bundle_verifier_rejects_noncanonical_manifest_and_symlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"schema":"fixture"}\n')
    good = tmp_path / "good.zip"
    bundle.build_public_bundle(source, good)

    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries["manifest.json"] = json.dumps(json.loads(entries["manifest.json"]), indent=2).encode()
    bad = tmp_path / "bad.zip"
    build_deterministic_zip(bad, entries)
    with pytest.raises(PrivacyError, match="canonical JSON"):
        bundle.verify_public_bundle(bad)

    linked = tmp_path / "linked.zip"
    linked.symlink_to(good)
    with pytest.raises(PrivacyError, match="symlink"):
        bundle.verify_public_bundle(linked)


def test_public_bundle_builder_rejects_symlinked_source_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"schema":"fixture"}\n')
    linked = tmp_path / "linked.json"
    linked.symlink_to(source)
    with pytest.raises(PrivacyError, match="symlink"):
        bundle.build_public_bundle(linked, tmp_path / "bundle.zip")


def test_public_bundle_verifier_rejects_unbound_portable_path_reference(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_bytes(b"portable evidence\n")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "evidence": {
                    "path": str(evidence),
                    "bytes": evidence.stat().st_size,
                    "sha256": _sha(evidence.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    good = tmp_path / "good.zip"
    bundle.build_public_bundle(source, good)
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    portable_path = manifest["portable_release"]["bundle_path"]
    portable = json.loads(entries.pop(portable_path))
    portable["evidence"]["path"] = "objects/" + "f" * 64
    portable_payload = _canonical(portable)
    replacement = "objects/" + _sha(portable_payload)
    entries[replacement] = portable_payload
    manifest["portable_release"].update(
        {
            "bundle_path": replacement,
            "bytes": len(portable_payload),
            "published_sha256": _sha(portable_payload),
        }
    )
    entries["manifest.json"] = _canonical(manifest)
    bad = tmp_path / "bad.zip"
    build_deterministic_zip(bad, entries)

    with pytest.raises(PrivacyError, match="unbound portable path"):
        bundle.verify_public_bundle(bad)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bytes", 1, "byte-identical"),
        ("source_sha256", "f" * 64, "byte-identical"),
    ],
)
def test_verifier_enforces_byte_identical_ledger_semantics(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_bytes(b"evidence\n")
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "evidence": {
                    "path": str(evidence),
                    "bytes": evidence.stat().st_size,
                    "sha256": _sha(evidence.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    good = tmp_path / "good.zip"
    bundle.build_public_bundle(release, good)

    def forge(manifest: dict[str, object], _entries: dict[str, bytes]) -> None:
        rows = manifest["objects"]
        assert isinstance(rows, list) and isinstance(rows[0], dict)
        rows[0][field] = value

    bad = tmp_path / f"bad-{field}.zip"
    _rewrite_bundle(good, bad, forge)
    with pytest.raises(PrivacyError, match=message):
        bundle.verify_public_bundle(bad)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ghost_path_published_sha256", "f" * 64, "orphan portable path commitment"),
        ("path_published_sha256", "f" * 64, "orphan portable path commitment"),
        (
            "path_normalization",
            "location-only; no content claim",
            "orphan portable path normalization",
        ),
    ],
)
def test_verifier_rejects_orphan_path_commitment_fields(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"schema":"fixture"}\n')
    good = tmp_path / "good.zip"
    bundle.build_public_bundle(source, good)
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    portable_path = manifest["portable_release"]["bundle_path"]
    portable = json.loads(entries.pop(portable_path))
    portable[field] = value
    portable_payload = _canonical(portable)
    replacement = "objects/" + _sha(portable_payload)
    entries[replacement] = portable_payload
    manifest["portable_release"].update(
        {
            "bundle_path": replacement,
            "bytes": len(portable_payload),
            "published_sha256": _sha(portable_payload),
        }
    )
    entries["manifest.json"] = _canonical(manifest)
    bad = tmp_path / "bad.zip"
    build_deterministic_zip(bad, entries)

    with pytest.raises(PrivacyError, match=message):
        bundle.verify_public_bundle(bad)
