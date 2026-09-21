from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_next_phase_evidence as evidence
from docs.research.kbound.scripts import build_next_phase_review_package as package
from docs.research.kbound.scripts import run_repository_verification as verification


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    study = "output/next_phase/study"
    (repo / study).mkdir(parents=True)
    (repo / study / "result.json").write_text('{"result":"negative"}\n')
    manifest_path = repo / "evidence.json"
    evidence.write_manifest(manifest_path, evidence.build_manifest(repo, [study], [study + "/result.json"]))
    evidence.build_archive(repo, manifest_path, tmp_path / "evidence.zip")
    (repo / "source.py").write_text("# source\n")
    rows = []
    for relative in ["evidence.json", "source.py"]:
        data = (repo / relative).read_bytes()
        rows.append(
            {
                "path": relative,
                "category": "source",
                "bytes": len(data),
                "sha256": package._sha(data),
                "git_blob": hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest(),
            }
        )
    seal = {
        "schema_version": "kbound-release-source-seal-v1",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "artifacts": rows,
        "sealed_artifact_count": len(rows),
        "exclusions": [],
        "artifacts_sha256": package._sha(
            json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ),
    }
    (repo / "seal.json").write_bytes(package._canonical(seal))
    gates = [gate.name for gate in verification.release_gate_plan(python="python")]
    receipt = verification.build_inventory_payload(tracked_paths=[], source_commit="a" * 40, source_tree="b" * 40)
    receipt.update(
        execution_status="PASS",
        executed_gates=gates,
        pytest_execution={
            "status": "COMPLETE_EXACTLY_ONCE",
            "executed_module_count": 0,
            "executed_group_count": 0,
            "executed_group_ids": [],
        },
        immutable_hook_configuration={"status": "PASS"},
        private_path_scan={"status": "PASS"},
        validator_execution={"status": "PASS", "source_commit": "a" * 40, "results": []},
    )
    receipt["inventory_sha256"] = verification.inventory_digest(receipt)
    (repo / "verification.json").write_bytes(package._canonical(receipt))
    (repo / "paper.pdf").write_bytes(b"%PDF-fixture")
    (repo / "paper.docx").write_bytes(b"docx fixture")
    docs = {"main.pdf": repo / "paper.pdf", "main.docx": repo / "paper.docx"}
    qa = {
        "status": "PASS",
        "documents": [
            {
                "path": p.name,
                "bytes": p.stat().st_size,
                "sha256": package._sha(p.read_bytes()),
                "pages": 2,
                "inspected_pages": [1, 2],
            }
            for p in docs.values()
        ],
    }
    (repo / "qa.json").write_bytes(package._canonical(qa))
    required = ("seal.json", "verification.json", "paper.pdf", "paper.docx")
    monkeypatch.setattr(package.checksums, "REQUIRED_RELEASE_PATHS", required)
    package.checksums.write_checksum_file(repo / "checksums.txt", root=repo, required_paths=required)
    return {
        "repo": repo,
        "seal": repo / "seal.json",
        "checksums_path": repo / "checksums.txt",
        "evidence_manifest": manifest_path,
        "evidence_archive": tmp_path / "evidence.zip",
        "verification_path": repo / "verification.json",
        "qa": repo / "qa.json",
        "documents": docs,
        "reports": {},
        "output_directory": tmp_path / "release",
    }


def test_exact_scoped_package_is_deterministic_create_only_and_self_contained(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    package.build_package(**args)
    package.verify_package(args["output_directory"])
    with zipfile.ZipFile(args["output_directory"] / "RELEASE_SOURCE.zip") as archive:
        assert set(archive.namelist()) == {
            "source.py",
            "evidence.json",
            "seal.json",
            "verification.json",
            "paper.pdf",
            "paper.docx",
            "checksums.txt",
        }
    original = (tmp_path / "release.zip").read_bytes()
    with pytest.raises(FileExistsError):
        package.build_package(**args)
    args["output_directory"] = tmp_path / "release_second"
    package.build_package(**args)
    assert original == (tmp_path / "release_second.zip").read_bytes()


@pytest.mark.parametrize("mutation", ["source", "checksum_inventory", "verification_commit", "qa_page", "symlink"])
def test_fail_closed_inputs(tmp_path, monkeypatch, mutation):
    args = _fixture(tmp_path, monkeypatch)
    repo = args["repo"]
    if mutation == "source":
        (repo / "source.py").write_text("changed")
    elif mutation == "checksum_inventory":
        lines = (repo / "checksums.txt").read_text().splitlines()
        (repo / "checksums.txt").write_text("\n".join(lines[:-1]) + "\n")
    elif mutation == "verification_commit":
        receipt = json.loads((repo / "verification.json").read_text())
        receipt["source_commit"] = "c" * 40
        receipt["inventory_sha256"] = verification.inventory_digest(receipt)
        (repo / "verification.json").write_bytes(package._canonical(receipt))
        package.checksums.write_checksum_file(
            repo / "checksums.txt", root=repo, required_paths=package.checksums.REQUIRED_RELEASE_PATHS
        )
    elif mutation == "qa_page":
        qa = json.loads((repo / "qa.json").read_text())
        qa["documents"][0]["inspected_pages"] = [1]
        (repo / "qa.json").write_bytes(package._canonical(qa))
    else:
        (repo / "source.py").unlink()
        (repo / "source.py").symlink_to(repo / "paper.pdf")
    with pytest.raises((ValueError, FileNotFoundError)):
        package.build_package(**args)
    assert not args["output_directory"].exists()


def _rezip(data: bytes, change) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    change(members)
    return package._zip_bytes(members)


@pytest.mark.parametrize("mutation", ["evidence", "source", "unexpected_source", "unsealed_manifest"])
def test_nested_archive_crossbindings_survive_outer_rehash(tmp_path, monkeypatch, mutation):
    args = _fixture(tmp_path, monkeypatch)
    package.build_package(**args)
    with zipfile.ZipFile(tmp_path / "release.zip") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if mutation == "evidence":
        members["NEXT_PHASE_EVIDENCE.zip"] = _rezip(
            members["NEXT_PHASE_EVIDENCE.zip"],
            lambda x: x.__setitem__("output/next_phase/study/result.json", b"tampered"),
        )
    elif mutation == "source":
        members["RELEASE_SOURCE.zip"] = _rezip(
            members["RELEASE_SOURCE.zip"], lambda x: x.__setitem__("source.py", b"tampered")
        )
    elif mutation == "unexpected_source":
        members["RELEASE_SOURCE.zip"] = _rezip(
            members["RELEASE_SOURCE.zip"], lambda x: x.__setitem__("undeclared.txt", b"extra")
        )
    else:
        members["EVIDENCE_MANIFEST.json"] = members["EVIDENCE_MANIFEST.json"].replace(b"negative", b"positive") + b" "
    manifest = json.loads(members[package.MANIFEST_MEMBER])
    manifest["artifacts"] = package._rows({k: v for k, v in members.items() if k != package.MANIFEST_MEMBER})
    members[package.MANIFEST_MEMBER] = package._canonical(manifest)
    with pytest.raises(ValueError):
        package.verify_archive_bytes(package._zip_bytes(members))


def test_verifier_rejects_unsafe_and_duplicate_archive_names(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    package.build_package(**args)
    original = (tmp_path / "release.zip").read_bytes()
    buffer = io.BytesIO(original)
    with zipfile.ZipFile(buffer, "a") as archive:
        archive.writestr(evidence._zip_info("../outside"), b"bad")
    unsafe = buffer.getvalue()
    with pytest.raises(ValueError, match="unsafe"):
        package.verify_archive_bytes(unsafe)
    buffer = io.BytesIO(original)
    with pytest.warns(UserWarning, match="Duplicate"):
        with zipfile.ZipFile(buffer, "a") as archive:
            archive.writestr(package.MANIFEST_MEMBER, b"{}")
    with pytest.raises(ValueError, match="duplicate"):
        package.verify_archive_bytes(buffer.getvalue())


@pytest.mark.parametrize(
    "field", ["execution_status", "executed_gates", "validator_execution", "immutable_hook_configuration"]
)
def test_rejects_rehashed_incomplete_verification(tmp_path, monkeypatch, field):
    args = _fixture(tmp_path, monkeypatch)
    path = args["verification_path"]
    payload = json.loads(path.read_bytes())
    payload[field] = (
        []
        if field == "executed_gates"
        else {"status": "NOT_RUN"}
        if field.endswith(("execution", "configuration"))
        else "PYTEST_ONLY_PASS"
    )
    payload["inventory_sha256"] = verification.inventory_digest(payload)
    path.write_bytes(package._canonical(payload))
    package.checksums.write_checksum_file(
        args["checksums_path"], root=args["repo"], required_paths=package.checksums.REQUIRED_RELEASE_PATHS
    )
    with pytest.raises(ValueError, match="verification"):
        package.build_package(**args)


def test_optional_post_checksum_payload_and_external_receipt(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    repo = args["repo"]
    (repo / "post.zip").write_bytes(b"post checksum archive")
    monkeypatch.setattr(package.checksums, "POST_CHECKSUM_REQUIRED_PATHS", ("post.zip",))
    package.checksums.write_checksum_file(repo / "post_checksums.txt", root=repo, required_paths=("post.zip",))
    args["post_checksums"] = repo / "post_checksums.txt"
    package.build_package(**args)
    with zipfile.ZipFile(args["output_directory"] / "RELEASE_SOURCE.zip") as archive:
        assert archive.read("post.zip") == b"post checksum archive"
        assert "post_checksums.txt" in archive.namelist()
    receipt_path = tmp_path / "release.zip.receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["sha256"] = "0" * 64
    receipt_path.write_bytes(package._canonical(receipt))
    with pytest.raises(ValueError, match="external ZIP"):
        package.verify_package(args["output_directory"])


def test_check_rejects_changed_directory_payload(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    package.build_package(**args)
    (args["output_directory"] / "documents/main.pdf").write_bytes(b"later copy")
    with pytest.raises(ValueError, match="directory and final ZIP"):
        package.verify_package(args["output_directory"])
