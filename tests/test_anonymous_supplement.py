from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_anonymous_supplement as supplement
from docs.research.kbound.scripts import build_cct20_public_bundle
from docs.research.kbound.scripts import build_release_source_seal as source_seal_module
from docs.research.kbound.scripts.release_privacy import PrivacyError, build_deterministic_zip
from docs.research.kbound.scripts.verify_release_checksums import REQUIRED_RELEASE_PATHS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pdf_inspection_tools(
    root: Path,
    *,
    metadata: str,
    xmp: str = "",
) -> None:
    root.mkdir()
    escaped_metadata = metadata.replace("'", "'\\''")
    escaped_xmp = xmp.replace("'", "'\\''")
    (root / "pdfinfo").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-meta" ]; then\n'
        f"  printf '%s' '{escaped_xmp}'\n"
        "else\n"
        f"  printf '%s' '{escaped_metadata}'\n"
        "fi\n",
        encoding="utf-8",
    )
    (root / "pdftotext").write_text("#!/bin/sh\nprintf 'Anonymous paper\\n'\n", encoding="utf-8")
    (root / "pdfdetach").write_text("#!/bin/sh\nprintf '0 embedded files\\n'\n", encoding="utf-8")
    for tool in root.iterdir():
        tool.chmod(0o755)


def _docx_with_metadata() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(
            "docProps/core.xml",
            '<cp:coreProperties xmlns:cp="x" xmlns:dc="y"><dc:creator>Pratik Niroula</dc:creator><cp:lastModifiedBy>pratik_n</cp:lastModifiedBy></cp:coreProperties>',
        )
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="x"><w:body><w:p><w:r><w:t>Anonymous manuscript</w:t></w:r></w:p></w:body></w:document>',
        )
        archive.writestr("word/comments.xml", '<w:comments xmlns:w="x"/>')
    return output.getvalue()


def _docx_with_unknown_identity_properties() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types><Override PartName="/docProps/custom.xml" ContentType="application/vnd.openxmlformats-officedocument.custom-properties+xml"/></Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties" Target="docProps/custom.xml"/></Relationships>',
        )
        archive.writestr(
            "docProps/core.xml",
            '<cp:coreProperties xmlns:cp="x" xmlns:dc="y"><dc:creator>Alice Smith</dc:creator><dc:subject>Alice Smith private review</dc:subject></cp:coreProperties>',
        )
        archive.writestr(
            "docProps/app.xml",
            "<Properties><Company>Alice Smith Laboratory</Company><Manager>Alice Smith</Manager></Properties>",
        )
        archive.writestr(
            "docProps/custom.xml",
            '<Properties><property name="Reviewer identity"><value>Alice Smith</value></property></Properties>',
        )
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="x"><w:body><w:p><w:r><w:t>Anonymous manuscript</w:t></w:r></w:p></w:body></w:document>',
        )
    return output.getvalue()


def _public_bundle(root: Path) -> Path:
    release = root / "release.json"
    release.write_bytes(b'{"schema":"fixture"}\n')
    public = root / "public.zip"
    build_cct20_public_bundle.build_public_bundle(release, public)
    return public


def _write_seal(path: Path, marker: str) -> None:
    rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "schema_version": source_seal_module.SCHEMA,
        "source_commit": marker * 40,
        "source_tree": marker * 40,
        "working_tree_gate": "fixture",
        "sealed_artifact_count": 0,
        "artifacts_sha256": hashlib.sha256(b"[]").hexdigest(),
        "artifacts": rows,
        "exclusions": [],
    }
    path.write_bytes(source_seal_module._seal_bytes(payload))


def _complete_release_controls(
    root: Path,
    marker: str,
    *,
    extra_paths: tuple[str, ...] = (),
) -> tuple[Path, Path, Path]:
    """Create a complete checksum fixture, with real seal/public-bundle controls."""

    seal = root / supplement.DEFAULT_SOURCE_SEAL
    seal.parent.mkdir(parents=True, exist_ok=True)
    _write_seal(seal, marker)
    public = root / supplement.DEFAULT_PUBLIC_BUNDLE
    public.parent.mkdir(parents=True, exist_ok=True)
    fixture_public = _public_bundle(root)
    fixture_public.replace(public)
    for relative in REQUIRED_RELEASE_PATHS:
        path = root / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{relative}\n".encode())
    checksum = root / supplement.DEFAULT_CHECKSUMS
    checksum.parent.mkdir(parents=True, exist_ok=True)
    inventory = list(REQUIRED_RELEASE_PATHS)
    inventory.extend(relative for relative in extra_paths if relative not in inventory)
    checksum.write_text(
        "".join(f"{_sha(root / relative)}  {relative}\n" for relative in inventory),
        encoding="utf-8",
    )
    return seal, checksum, public


def test_anonymous_package_is_deterministic_scrubbed_and_commitment_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    manuscript = root / "paper.docx"
    manuscript.write_bytes(_docx_with_metadata())
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplementary material.\n", encoding="utf-8")
    source_seal, checksums, public_bundle = _complete_release_controls(
        root, "a", extra_paths=("notes.tex", "paper.docx")
    )

    first = tmp_path / "anonymous-1.zip"
    second = tmp_path / "anonymous-2.zip"
    kwargs = {
        "root": root,
        "include_paths": ["notes.tex", "paper.docx"],
        "source_seal_path": source_seal,
        "checksum_path": checksums,
        "public_bundle_path": public_bundle,
        "require_git_checkout": False,
    }
    supplement.build_anonymous_supplement(output_path=first, **kwargs)
    supplement.build_anonymous_supplement(output_path=second, **kwargs)

    assert first.read_bytes() == second.read_bytes()
    verified = supplement.verify_anonymous_supplement(first)
    assert verified["status"] == "PASS"
    with zipfile.ZipFile(first) as archive:
        assert "artifacts/paper.docx" in archive.namelist()
        assert "receipts/KBOUND_RELEASE_SHA256SUMS.txt" in archive.namelist()
        assert "receipts/release_source_seal.json" in archive.namelist()
        assert "evidence/cct20_public_evidence_bundle.zip" not in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        row = next(item for item in manifest["artifacts"] if item["published_path"].endswith("paper.docx"))
        assert row["source_sha256"] == _sha(manuscript)
        assert row["published_sha256"] != row["source_sha256"]
        with zipfile.ZipFile(io.BytesIO(archive.read("artifacts/paper.docx"))) as docx:
            assert "word/comments.xml" not in docx.namelist()
            core = docx.read("docProps/core.xml")
            assert b"Pratik" not in core and b"pratik_n" not in core
            assert b"Anonymous" in core


def test_anonymous_verified_replacement_is_explicit_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement version one.\n", encoding="utf-8")
    seal = root / supplement.DEFAULT_SOURCE_SEAL
    seal.parent.mkdir(parents=True)
    _write_seal(seal, "a")
    public = root / supplement.DEFAULT_PUBLIC_BUNDLE
    public.parent.mkdir(parents=True)
    _public_bundle(root).replace(public)
    required = (
        str(supplement.DEFAULT_SOURCE_SEAL),
        str(supplement.DEFAULT_PUBLIC_BUNDLE),
        "notes.tex",
    )
    monkeypatch.setattr(supplement, "REQUIRED_RELEASE_PATHS", required)
    sums = root / supplement.DEFAULT_CHECKSUMS

    def write_checksums() -> None:
        sums.write_text(
            "".join(f"{_sha(root / relative)}  {relative}\n" for relative in required),
            encoding="utf-8",
        )

    write_checksums()
    output = tmp_path / "anonymous.zip"
    kwargs = {
        "root": root,
        "include_paths": ["notes.tex"],
        "source_seal_path": seal,
        "checksum_path": sums,
        "public_bundle_path": public,
        "require_git_checkout": False,
    }
    first = supplement.build_anonymous_supplement(output_path=output, **kwargs)
    first_inode = output.stat().st_ino
    first_mtime = output.stat().st_mtime_ns
    assert (
        supplement.verify_anonymous_release_bindings(
            output,
            source_seal_path=seal,
            checksum_path=sums,
            public_bundle_path=public,
        )["status"]
        == "PASS"
    )
    original_checksums = sums.read_bytes()
    sums.write_bytes(original_checksums + b"# stale external receipt\n")
    with pytest.raises(PrivacyError, match="external release checksum"):
        supplement.verify_anonymous_release_bindings(
            output,
            source_seal_path=seal,
            checksum_path=sums,
            public_bundle_path=public,
        )
    sums.write_bytes(original_checksums)

    with pytest.raises(PrivacyError, match="already exists"):
        supplement.build_anonymous_supplement(output_path=output, **kwargs)

    repeated = supplement.build_anonymous_supplement(
        output_path=output,
        replace_verified=True,
        **kwargs,
    )
    assert repeated["archive_sha256"] == first["archive_sha256"]
    assert output.stat().st_ino == first_inode
    assert output.stat().st_mtime_ns == first_mtime

    notes.write_text("Anonymous supplement version two.\n", encoding="utf-8")
    _write_seal(seal, "b")
    write_checksums()
    changed = supplement.build_anonymous_supplement(
        output_path=output,
        replace_verified=True,
        **kwargs,
    )
    assert changed["archive_sha256"] != first["archive_sha256"]
    assert supplement.verify_anonymous_supplement(output)["archive_sha256"] == changed["archive_sha256"]

    output.write_bytes(b"corrupt existing archive")
    with pytest.raises(PrivacyError):
        supplement.build_anonymous_supplement(
            output_path=output,
            replace_verified=True,
            **kwargs,
        )
    assert output.read_bytes() == b"corrupt existing archive"


def test_anonymous_publication_rejects_a_candidate_swap_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, checksums, public = _complete_release_controls(root, "a", extra_paths=("notes.tex",))
    output = tmp_path / "anonymous.zip"
    original_verify = supplement.verify_anonymous_supplement

    def verify_then_swap(path: Path) -> dict[str, object]:
        receipt = original_verify(path)
        path.write_bytes(b"attacker candidate swap")
        return receipt

    monkeypatch.setattr(supplement, "verify_anonymous_supplement", verify_then_swap)
    with pytest.raises(PrivacyError, match="candidate.*verified"):
        supplement.build_anonymous_supplement(
            root=root,
            include_paths=["notes.tex"],
            source_seal_path=seal,
            checksum_path=checksums,
            public_bundle_path=public,
            output_path=output,
            require_git_checkout=False,
        )
    assert not output.exists()


def test_portable_binding_rejects_anonymous_archive_swap_after_semantic_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal = root / supplement.DEFAULT_SOURCE_SEAL
    seal.parent.mkdir(parents=True)
    _write_seal(seal, "a")
    public = root / supplement.DEFAULT_PUBLIC_BUNDLE
    public.parent.mkdir(parents=True)
    _public_bundle(root).replace(public)
    required = (str(supplement.DEFAULT_SOURCE_SEAL), str(supplement.DEFAULT_PUBLIC_BUNDLE), "notes.tex")
    monkeypatch.setattr(supplement, "REQUIRED_RELEASE_PATHS", required)
    checksums = root / supplement.DEFAULT_CHECKSUMS
    checksums.write_text(
        "".join(f"{_sha(root / relative)}  {relative}\n" for relative in required),
        encoding="utf-8",
    )
    output = tmp_path / "anonymous.zip"
    supplement.build_anonymous_supplement(
        root=root,
        include_paths=["notes.tex"],
        source_seal_path=seal,
        checksum_path=checksums,
        public_bundle_path=public,
        output_path=output,
        require_git_checkout=False,
    )
    original_verify = supplement.verify_anonymous_supplement

    def verify_then_swap(path: Path) -> dict[str, object]:
        receipt = original_verify(path)
        path.write_bytes(b"attacker swap")
        return receipt

    monkeypatch.setattr(supplement, "verify_anonymous_supplement", verify_then_swap)
    with pytest.raises(PrivacyError, match="changed.*verified"):
        supplement.verify_anonymous_release_bindings(
            output,
            source_seal_path=seal,
            checksum_path=checksums,
            public_bundle_path=public,
        )
    assert output.read_bytes() == b"attacker swap"


def test_ooxml_scrubber_removes_unknown_custom_and_application_identity() -> None:
    published, normalization = supplement.prepare_anonymous_payload(
        "paper.docx", _docx_with_unknown_identity_properties()
    )

    assert normalization == "deterministic-ooxml-metadata-scrub"
    assert b"Alice Smith" not in published
    with zipfile.ZipFile(io.BytesIO(published)) as archive:
        assert "docProps/custom.xml" not in archive.namelist()
        assert b"custom-properties" not in archive.read("_rels/.rels")
        assert b"custom.xml" not in archive.read("[Content_Types].xml")


@pytest.mark.parametrize(
    ("relative", "payload", "needle"),
    [
        (".git/config", b"safe", "forbidden"),
        (".DS_Store", b"safe", "forbidden"),
        ("._paper", b"safe", "forbidden"),
        ("~$paper.docx", b"safe", "unsafe"),
        ("identity.txt", b"Pratik Niroula", "identity"),
        ("mail.txt", b"author@example.edu", "email"),
        ("path.txt", b"/Us" + b"ers/private/private", "private POSIX"),
        ("uri.txt", b"file:///private/tmp/result", "file URI"),
        ("unknown.bin", b"opaque", "unclassified"),
    ],
)
def test_anonymous_builder_rejects_every_unpublishable_input_without_omission(
    tmp_path: Path, relative: str, payload: bytes, needle: str
) -> None:
    root = tmp_path / "repo"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    seal, sums, public = _complete_release_controls(root, "b", extra_paths=(relative,))

    output = tmp_path / "anonymous.zip"
    with pytest.raises(PrivacyError, match=needle):
        supplement.build_anonymous_supplement(
            root=root,
            include_paths=[relative],
            source_seal_path=seal,
            checksum_path=sums,
            public_bundle_path=public,
            output_path=output,
            require_git_checkout=False,
        )
    assert not output.exists()


def test_anonymous_builder_rejects_tracked_changes_and_nested_archive_leaks(
    tmp_path: Path,
) -> None:
    with pytest.raises(PrivacyError, match="tracked-change"):
        supplement.sanitize_ooxml(
            "paper.docx",
            supplement._replace_docx_document(_docx_with_metadata(), '<w:ins w:author="A"/>'),
        )

    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("secret.txt", "Pratikn03")
    with pytest.raises(PrivacyError, match="identity"):
        supplement.prepare_anonymous_payload("nested.zip", nested.getvalue())


@pytest.mark.parametrize("revision_tag", ["moveFrom", "moveTo", "rPrChange", "numberingChange"])
def test_ooxml_rejects_every_revision_element_that_can_carry_an_author(
    revision_tag: str,
) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="urn:word"><w:{revision_tag} w:author="Alice Smith"/></w:document>',
        )

    with pytest.raises(PrivacyError, match="revision metadata"):
        supplement.sanitize_ooxml("paper.docx", archive_bytes.getvalue())


def test_ooxml_size_limit_is_checked_before_reading_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"12345")
    monkeypatch.setattr(supplement, "MAX_MEMBER_BYTES", 4, raising=False)

    with pytest.raises(PrivacyError, match="oversized OOXML member"):
        supplement.sanitize_ooxml("paper.docx", archive_bytes.getvalue())


def test_release_checksum_and_runbook_order_avoid_anonymous_hash_cycle() -> None:
    from docs.research.kbound.scripts.verify_release_checksums import REQUIRED_RELEASE_PATHS

    public = "docs/research/kbound/release/cct20_public_evidence_bundle.zip"
    anonymous = "docs/research/kbound/release/kbound_anonymous_supplement.zip"
    assert public in REQUIRED_RELEASE_PATHS
    assert anonymous not in REQUIRED_RELEASE_PATHS
    post = "docs/research/kbound/KBOUND_POST_CHECKSUM_SHA256SUMS.txt"
    runbook = (Path(supplement.__file__).resolve().parents[1] / "runbooks" / "release_candidate.sh").read_text(
        encoding="utf-8"
    )
    public_call = '"$KB/scripts/build_cct20_public_bundle.py"'
    anonymous_call = '"$KB/scripts/build_anonymous_supplement.py"'
    assert public_call in runbook and anonymous_call in runbook
    all_mode = runbook.split("  all)", 1)[1]
    assert all_mode.index("step_verify_public_bundle") < all_mode.index("emit_checksums")
    assert all_mode.index("emit_checksums") < all_mode.index("step_anonymous_supplement")
    assert all_mode.index("step_anonymous_supplement") < all_mode.index("emit_post_checksums")
    assert "$KB/" + Path(post).name in runbook
    assert '"$KB/scripts/run_repository_verification.py"' in runbook


@pytest.mark.parametrize("environment_ok", [True, False])
def test_anonymous_cli_verifies_exact_python_content_before_archive_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment_ok: bool
) -> None:
    calls: list[str] = []

    def verify_selected_profile(lock: Path, profile: Path) -> None:
        assert lock.name == "requirements-release-macos-arm64.lock.txt"
        assert profile.name == "release_python_environment_macos_arm64_v2.json"
        calls.append("environment")
        if not environment_ok:
            raise ValueError("synthetic Python content rejection")

    monkeypatch.setattr(supplement.verify_python_environment, "verify_exact_content_profile", verify_selected_profile)
    monkeypatch.setattr(
        supplement,
        "verify_anonymous_supplement",
        lambda _path: calls.append("anonymous") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_anonymous_supplement.py", "--check", "--output", str(tmp_path / "anonymous.zip")],
    )

    if environment_ok:
        assert supplement.main() == 0
        assert calls == ["environment", "anonymous"]
    else:
        with pytest.raises(ValueError, match="synthetic Python content rejection"):
            supplement.main()
        assert calls == ["environment"]


def test_anonymous_portable_check_runs_full_archive_semantics_without_release_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        supplement,
        "verify_release_python_content",
        lambda: (_ for _ in ()).throw(AssertionError("portable verification must not inspect the build runtime")),
    )
    monkeypatch.setattr(
        supplement,
        "verify_anonymous_release_bindings",
        lambda _path, **_kwargs: calls.append("anonymous-release-bindings") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        supplement,
        "verify_portable_pdf_toolchain",
        lambda: calls.append("portable-poppler"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_anonymous_supplement.py", "--portable-check", "--output", str(tmp_path / "anonymous.zip")],
    )

    assert supplement.main() == 0
    assert calls == ["portable-poppler", "anonymous-release-bindings"]


def test_portable_pdf_toolchain_rejects_mutable_or_unbound_tool_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected: dict[str, str] = {}
    for name in ("pdfinfo", "pdftotext", "pdfdetach"):
        path = tmp_path / name
        path.write_bytes(f"{name} pinned bytes\n".encode())
        path.chmod(0o755)
        monkeypatch.setenv(f"KBOUND_TOOL_{name.upper()}", str(path))
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    library = tmp_path / "libpoppler.so"
    library.write_bytes(b"pinned library bytes\n")
    monkeypatch.setenv("KBOUND_PORTABLE_POPPLER_LIBRARY", str(library))
    monkeypatch.setattr(supplement, "PORTABLE_PDF_TOOL_SHA256", expected)
    monkeypatch.setattr(
        supplement,
        "PORTABLE_POPPLER_LIBRARY_SHA256",
        hashlib.sha256(library.read_bytes()).hexdigest(),
    )

    supplement.verify_portable_pdf_toolchain()
    (tmp_path / "pdfinfo").write_bytes(b"mutable replacement\n")
    with pytest.raises(PrivacyError, match="digest"):
        supplement.verify_portable_pdf_toolchain()


def test_standalone_post_checksums_uses_portable_external_verification_before_hashing(tmp_path: Path) -> None:
    """The post receipt must rely on standalone portable archive semantics."""

    repo = tmp_path / "repo"
    runbook = repo / "docs/research/kbound/runbooks/release_candidate.sh"
    runbook.parent.mkdir(parents=True)
    runbook.write_text(
        (Path(supplement.__file__).resolve().parents[1] / "runbooks/release_candidate.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text("[project]\nname = 'release-fixture'\n", encoding="utf-8")
    toolchain_receipt = repo / "docs/research/kbound/audits/release_toolchain_2026_09_05_v2.json"
    toolchain_receipt.parent.mkdir(parents=True)
    toolchain_receipt.write_text('{"status": "verified"}\n', encoding="utf-8")
    release = repo / "docs/research/kbound/release"
    release.mkdir(parents=True)
    (release / "kbound_anonymous_supplement.zip").write_bytes(b"not a semantically valid ZIP")
    post_receipt = repo / "docs/research/kbound/KBOUND_POST_CHECKSUM_SHA256SUMS.txt"
    post_receipt.write_text("previous verified receipt\n", encoding="utf-8")
    event_log = tmp_path / "events.jsonl"
    fake_python = tmp_path / "verified-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"event_log = pathlib.Path({str(event_log)!r})\n"
        "args = sys.argv[1:]\n"
        "with event_log.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "if args and args[0].endswith('verify_python_environment.py'):\n"
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text('{\"status\": \"verified\"}\\n', encoding='utf-8')\n"
        "    raise SystemExit(0)\n"
        "if args and args[0].endswith('verify_release_toolchain.py'):\n"
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text('{\"status\": \"verified\"}\\n', encoding='utf-8')\n"
        "    resolved = pathlib.Path(args[args.index('--resolved-tools-output') + 1])\n"
        "    names = ('LATEXMK', 'LATEXPAND', 'PANDOC', 'PDFDETACH', 'PDFINFO', 'PDFLATEX', 'PDFTOPPM', 'PDFTOTEXT', 'PERL', 'SOFFICE')\n"
        "    executable = pathlib.Path(sys.argv[0]).resolve()\n"
        "    resolved.write_text(''.join(f'KBOUND_TOOL_{name}\\t{executable}\\n' for name in names), encoding='utf-8')\n"
        "    raise SystemExit(0)\n"
        "if args and args[0].endswith('build_anonymous_supplement.py') and '--portable-check' in args:\n"
        "    raise SystemExit(71)\n"
        "if args and args[0].endswith('verify_release_checksums.py'):\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('unexpected release command: ' + repr(args))\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(runbook), "post-checksums"],
        cwd=repo,
        env={**os.environ, "KBOUND_PYTHON": str(fake_python)},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 71
    events = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    assert events[0][0].endswith("verify_python_environment.py")
    assert "--content-profile" in events[0]
    assert events[0][events[0].index("--content-profile") + 1] == (
        "docs/research/kbound/release_python_environment_macos_arm64_v2.json"
    )
    assert events[1][0].endswith("verify_release_toolchain.py")
    assert events[2][0].endswith("build_anonymous_supplement.py")
    assert "--portable-check" in events[2]
    assert "--check" not in events[2]
    assert not any(event[0].endswith("verify_release_checksums.py") for event in events)
    assert post_receipt.read_text(encoding="utf-8") == "previous verified receipt\n"


@pytest.mark.parametrize("linked_input", ["source_seal", "checksums"])
def test_anonymous_builder_rejects_symlinked_control_inputs(tmp_path: Path, linked_input: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    readme = root / "notes.tex"
    readme.write_text("Anonymous supplement.\n", encoding="utf-8")
    real_seal = root / "real-seal.json"
    _write_seal(real_seal, "c")
    real_sums = root / "real-sums.txt"
    real_sums.write_text(_sha(readme) + "  notes.tex\n", encoding="utf-8")
    seal = real_seal
    sums = real_sums
    if linked_input == "source_seal":
        seal = root / "seal.json"
        seal.symlink_to(real_seal)
    else:
        sums = root / "sums.txt"
        sums.symlink_to(real_sums)
    with pytest.raises(PrivacyError, match="symlink"):
        supplement.build_anonymous_supplement(
            root=root,
            include_paths=["notes.tex"],
            source_seal_path=seal,
            checksum_path=sums,
            public_bundle_path=_public_bundle(root),
            output_path=tmp_path / "anonymous.zip",
            require_git_checkout=False,
        )


def test_default_anonymous_inventory_is_exact_tmlr_dependency_closure() -> None:
    root = Path(supplement.__file__).resolve().parents[4]
    inventory = supplement.anonymous_tmlr_inventory(root)
    assert {
        "docs/research/kbound/kbound_tmlr.pdf",
        "docs/research/kbound/kbound_tmlr.tex",
        "docs/research/kbound/kbound_submission_body.tex",
        "docs/research/kbound/kbound_submission_supplement.tex",
        "docs/research/kbound/kbound_abstract.tex",
        "docs/research/kbound/kbound_abstract_core.tex",
        "docs/research/kbound/paper/vendor/tmlr/tmlr.sty",
        "docs/research/kbound/paper/vendor/tmlr/LICENSE",
        "docs/research/kbound/paper/references_kbound_expanded.tex",
        "docs/research/kbound/paper/references/refs.bib",
        "docs/research/kbound/figures/fig_frontier_schematic.png",
        "docs/research/kbound/paper/figures/decision_flow.tex",
    } <= set(inventory)
    assert {
        "docs/research/kbound/kbound_short_final_draft.pdf",
        "docs/research/kbound/kbound_short_final_draft.docx",
        "docs/research/kbound/kbound_submission.tex",
        "CITATION.cff",
        "README.md",
        "docs/research/kbound/paper/generated/cct20_release_manifest.json",
        "docs/research/kbound/figures/fig_decision_flow.png",
        "docs/research/kbound/figures/fig_certificate.png",
    }.isdisjoint(inventory)
    assert all((root / relative).is_file() for relative in inventory)


def test_default_inventory_rejects_a_missing_literal_graphic(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paper = root / "docs/research/kbound"
    paper.mkdir(parents=True)
    (paper / "kbound_tmlr.tex").write_text(
        "\\includegraphics{figures/missing.png}\n",
        encoding="utf-8",
    )

    with pytest.raises(PrivacyError, match="live TMLR graphic is missing"):
        supplement.anonymous_tmlr_inventory(root)


def test_default_inventory_resolves_an_extensionless_graphic(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paper = root / "docs/research/kbound"
    figures = paper / "figures"
    figures.mkdir(parents=True)
    (paper / "kbound_tmlr.tex").write_text(
        "\\includegraphics{figures/plot}\n",
        encoding="utf-8",
    )
    (figures / "plot.png").write_bytes(b"fixture")

    inventory = supplement.anonymous_tmlr_inventory(root)

    assert "docs/research/kbound/figures/plot.png" in inventory


def test_default_inventory_ignores_commented_tex_dependencies(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paper = root / "docs/research/kbound"
    references = paper / "paper/references"
    vendor = paper / "paper/vendor/tmlr"
    references.mkdir(parents=True)
    vendor.mkdir(parents=True)
    (paper / "kbound_tmlr.tex").write_text(
        "% \\input{missing_private_draft}\nAnonymous manuscript.\n",
        encoding="utf-8",
    )
    (references / "refs.bib").write_text("", encoding="utf-8")
    (vendor / "LICENSE").write_text("license", encoding="utf-8")
    (paper / "kbound_tmlr.pdf").write_bytes(b"%PDF-1.7\nfixture")

    inventory = supplement.anonymous_tmlr_inventory(root)

    assert "docs/research/kbound/kbound_tmlr.tex" in inventory


def test_every_default_non_pdf_input_is_classified_and_privacy_scannable() -> None:
    """A package inventory is useful only when its real members can be prepared."""

    root = Path(supplement.__file__).resolve().parents[4]
    for relative in supplement.anonymous_tmlr_inventory(root):
        if Path(relative).suffix.lower() == ".pdf":
            continue
        published, normalization = supplement.prepare_anonymous_payload(relative, (root / relative).read_bytes())
        assert published
        assert normalization


@pytest.mark.parametrize(
    "relative",
    [
        "docs/research/kbound/kbound_short_final_draft.pdf",
        "docs/research/kbound/kbound_short_final_draft.docx",
        "docs/research/kbound/kbound_submission.tex",
        "CITATION.cff",
        "README.md",
    ],
)
def test_anonymous_builder_rejects_named_release_inputs(relative: str) -> None:
    with pytest.raises(PrivacyError, match="forbidden named-release"):
        supplement.validate_anonymous_inventory([relative])


def test_pdf_anonymity_uses_metadata_and_extracted_text_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    pdfinfo = tools / "pdfinfo"
    pdftotext = tools / "pdftotext"
    pdfdetach = tools / "pdfdetach"
    pdfinfo.write_text(
        '#!/bin/sh\n[ "$1" = "-meta" ] && exit 0\nprintf \'Author: Pratik Niroula\\nEncrypted: no\\nJavaScript: no\\nForm: none\\n\'\n',
        encoding="utf-8",
    )
    pdftotext.write_text("#!/bin/sh\nprintf 'Anonymous paper\\n'\n", encoding="utf-8")
    pdfdetach.write_text("#!/bin/sh\nprintf '0 embedded files\\n'\n", encoding="utf-8")
    pdfinfo.chmod(0o755)
    pdftotext.chmod(0o755)
    pdfdetach.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ.get("PATH", ""))
    with pytest.raises(PrivacyError, match="identity"):
        supplement.verify_pdf_anonymity("paper.pdf", b"%PDF-1.7\ncompressed-metadata-fixture")


def test_pdf_anonymity_exact_overrides_defeat_cross_directory_path_shadows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_tools = tmp_path / "verified-source"
    _write_pdf_inspection_tools(
        source_tools,
        metadata=(
            "Title: K-Bound: When Is Label-Free Adaptation Knowable?\n"
            "Author: \nCreator: LaTeX with hyperref\nProducer: pdfTeX-1.40.27\n"
            "Subject: \nKeywords: \nEncrypted: no\nJavaScript: no\nForm: none\n"
        ),
    )
    exact_tools: dict[str, Path] = {}
    for name in ("pdfinfo", "pdftotext", "pdfdetach"):
        destination = tmp_path / f"verified-{name}" / f"pinned-{name}"
        destination.parent.mkdir()
        (source_tools / name).replace(destination)
        exact_tools[name] = destination

    shadows = tmp_path / "path-shadows"
    shadows.mkdir()
    for name in exact_tools:
        shadow = shadows / name
        shadow.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
        shadow.chmod(0o755)
    monkeypatch.setenv("PATH", str(shadows))
    for name, path in exact_tools.items():
        monkeypatch.setenv(f"KBOUND_TOOL_{name.upper()}", str(path))

    supplement.verify_pdf_anonymity("paper.pdf", b"%PDF-1.7\nfixture")


@pytest.mark.parametrize("field", ["Creator", "Producer", "Subject", "Keywords"])
def test_pdf_anonymity_rejects_arbitrary_identity_in_info_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    tools = tmp_path / "tools"
    _write_pdf_inspection_tools(
        tools,
        metadata=(
            "Title: K-Bound: When Is Label-Free Adaptation Knowable?\n"
            "Author: \n"
            f"{field}: Alice Smith Laboratory\n"
            "Encrypted: no\nJavaScript: no\nForm: none\n"
        ),
    )
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ.get("PATH", ""))

    with pytest.raises(PrivacyError, match=field):
        supplement.verify_pdf_anonymity("paper.pdf", b"%PDF-1.7\nfixture")


def test_pdf_anonymity_rejects_nonpublication_title_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = tmp_path / "tools"
    _write_pdf_inspection_tools(
        tools,
        metadata=("Title: Alice Smith private draft\nAuthor: \nEncrypted: no\nJavaScript: no\nForm: none\n"),
    )
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ.get("PATH", ""))

    with pytest.raises(PrivacyError, match="title"):
        supplement.verify_pdf_anonymity("paper.pdf", b"%PDF-1.7\nfixture")


def test_pdf_anonymity_rejects_unknown_namespaced_xmp_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = tmp_path / "tools"
    _write_pdf_inspection_tools(
        tools,
        metadata=(
            "Title: K-Bound: When Is Label-Free Adaptation Knowable?\nAuthor: \n"
            "Encrypted: no\nJavaScript: no\nForm: none\n"
        ),
        xmp=(
            '<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:private="urn:private">'
            "<private:reviewer>Alice Smith</private:reviewer></x:xmpmeta>"
        ),
    )
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ.get("PATH", ""))

    with pytest.raises(PrivacyError, match="XMP"):
        supplement.verify_pdf_anonymity("paper.pdf", b"%PDF-1.7\nfixture")


def test_tex_anonymity_rejects_arbitrary_author_macro() -> None:
    with pytest.raises(PrivacyError, match="author metadata"):
        supplement.prepare_anonymous_payload("paper.tex", b"\\author{Alice Smith\\\\Private University}\n")


def test_pdf_anonymity_uses_each_exact_override_across_hostile_tool_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "verified first"
    second = tmp_path / "verified second"
    first.mkdir()
    second.mkdir()
    pinned = {
        "pdfinfo": first / "pdfinfo-real",
        "pdftotext": second / "pdftotext-real",
        "pdfdetach": first / "pdfdetach-real",
    }
    pinned["pdfinfo"].write_text(
        '#!/bin/sh\n[ "$1" = "-meta" ] && exit 0\nprintf \'Author: Anonymous Authors\\nEncrypted: no\\nJavaScript: no\\nForm: none\\n\'\n',
        encoding="utf-8",
    )
    pinned["pdftotext"].write_text("#!/bin/sh\nprintf 'Anonymous paper\\n'\n", encoding="utf-8")
    pinned["pdfdetach"].write_text("#!/bin/sh\nprintf '0 embedded files\\n'\n", encoding="utf-8")
    (first / "pdftotext").write_text("#!/bin/sh\nprintf 'author@example.edu\\n'\n", encoding="utf-8")
    (second / "pdfinfo").write_text("#!/bin/sh\nprintf 'Author: Private Reviewer\\n'\n", encoding="utf-8")
    (second / "pdfdetach").write_text("#!/bin/sh\nprintf '1 embedded files\\n'\n", encoding="utf-8")
    for tool in (*pinned.values(), first / "pdftotext", second / "pdfinfo", second / "pdfdetach"):
        tool.chmod(0o755)
    for name, path in pinned.items():
        monkeypatch.setenv(f"KBOUND_TOOL_{name.upper()}", str(path))
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(first), str(second), os.environ.get("PATH", ""))),
    )

    supplement.verify_pdf_anonymity("paper.pdf", b"%PDF-1.7\nfixture")


def test_safe_pdf_is_not_scanned_as_raw_compressed_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    pdfinfo = tools / "pdfinfo"
    pdftotext = tools / "pdftotext"
    pdfdetach = tools / "pdfdetach"
    pdfinfo.write_text(
        '#!/bin/sh\n[ "$1" = "-meta" ] && exit 0\nprintf \'Author: Anonymous Authors\\nEncrypted: no\\nJavaScript: no\\nForm: none\\n\'\n',
        encoding="utf-8",
    )
    pdftotext.write_text("#!/bin/sh\nprintf 'Anonymous paper\\n'\n", encoding="utf-8")
    pdfdetach.write_text("#!/bin/sh\nprintf '0 embedded files\\n'\n", encoding="utf-8")
    pdfinfo.chmod(0o755)
    pdftotext.chmod(0o755)
    pdfdetach.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ.get("PATH", ""))

    payload = b"%PDF-1.7\ncompressed-stream P@DhoYC.wv /7o/WCCCe\n"
    published, normalization = supplement.prepare_anonymous_payload("paper.pdf", payload)

    assert published == payload
    assert normalization == "verified-pdf-byte-identical-copy"


def test_verified_pdf_can_cross_the_generic_zip_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "pdfinfo").write_text(
        '#!/bin/sh\n[ "$1" = "-meta" ] && exit 0\nprintf \'Author: Anonymous Authors\\nEncrypted: no\\nJavaScript: no\\nForm: none\\n\'\n',
        encoding="utf-8",
    )
    (tools / "pdftotext").write_text("#!/bin/sh\nprintf 'Anonymous paper\\n'\n", encoding="utf-8")
    (tools / "pdfdetach").write_text("#!/bin/sh\nprintf '0 embedded files\\n'\n", encoding="utf-8")
    for tool in tools.iterdir():
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ.get("PATH", ""))
    root = tmp_path / "repo"
    root.mkdir()
    paper = root / "paper.pdf"
    paper.write_bytes(b"%PDF-1.7\nfixture")
    seal, sums, public = _complete_release_controls(root, "9", extra_paths=("paper.pdf",))

    output = tmp_path / "anonymous.zip"
    receipt = supplement.build_anonymous_supplement(
        root=root,
        include_paths=["paper.pdf"],
        source_seal_path=seal,
        checksum_path=sums,
        public_bundle_path=public,
        output_path=output,
        require_git_checkout=False,
    )

    assert receipt["status"] == "PASS"


@pytest.mark.parametrize(
    ("unsafe_line", "needle"),
    [
        ("Encrypted: yes", "encrypted"),
        ("JavaScript: yes", "JavaScript"),
        ("Form: AcroForm", "interactive form"),
    ],
)
def test_pdf_privacy_rejects_active_or_noninspectable_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_line: str,
    needle: str,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    defaults = {"Encrypted": "no", "JavaScript": "no", "Form": "none"}
    key, value = unsafe_line.split(": ", 1)
    defaults[key] = value
    (tools / "pdfinfo").write_text(
        '#!/bin/sh\n[ "$1" = "-meta" ] && exit 0\nprintf \'Author: Anonymous Authors\\n'
        + "".join(f"{name}: {setting}\\n" for name, setting in defaults.items())
        + "'\n",
        encoding="utf-8",
    )
    (tools / "pdftotext").write_text("#!/bin/sh\nprintf 'Anonymous paper\\n'\n", encoding="utf-8")
    (tools / "pdfdetach").write_text("#!/bin/sh\nprintf '0 embedded files\\n'\n", encoding="utf-8")
    for tool in tools.iterdir():
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ.get("PATH", ""))

    with pytest.raises(PrivacyError, match=needle):
        supplement.verify_pdf_anonymity("paper.pdf", b"%PDF-1.7\nfixture")


def test_pdf_privacy_rejects_embedded_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "pdfinfo").write_text(
        '#!/bin/sh\n[ "$1" = "-meta" ] && exit 0\nprintf \'Author: Anonymous Authors\\nEncrypted: no\\nJavaScript: no\\nForm: none\\n\'\n',
        encoding="utf-8",
    )
    (tools / "pdftotext").write_text("#!/bin/sh\nprintf 'Anonymous paper\\n'\n", encoding="utf-8")
    (tools / "pdfdetach").write_text("#!/bin/sh\nprintf '1 embedded files\\n1: secret.txt\\n'\n", encoding="utf-8")
    for tool in tools.iterdir():
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ.get("PATH", ""))

    with pytest.raises(PrivacyError, match="embedded file"):
        supplement.verify_pdf_anonymity("paper.pdf", b"%PDF-1.7\nfixture")


def test_standalone_anonymous_verifier_rejects_forged_embedded_source_seal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "d", extra_paths=("notes.tex",))
    good = tmp_path / "good.zip"
    supplement.build_anonymous_supplement(
        root=root,
        include_paths=["notes.tex"],
        source_seal_path=seal,
        checksum_path=sums,
        public_bundle_path=public,
        output_path=good,
        require_git_checkout=False,
    )
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    forged = json.loads(entries["receipts/release_source_seal.json"])
    forged["sealed_artifact_count"] = 1
    forged_payload = source_seal_module._seal_bytes(forged)
    entries["receipts/release_source_seal.json"] = forged_payload
    manifest = json.loads(entries["manifest.json"])
    commitment = manifest["commitments"]["release_source_seal"]
    commitment.update(
        {
            "published_bytes": len(forged_payload),
            "published_sha256": hashlib.sha256(forged_payload).hexdigest(),
            "source_bytes": len(forged_payload),
            "source_sha256": hashlib.sha256(forged_payload).hexdigest(),
        }
    )
    entries["manifest.json"] = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    bad = tmp_path / "bad.zip"
    from docs.research.kbound.scripts.release_privacy import build_deterministic_zip

    build_deterministic_zip(bad, entries)
    with pytest.raises(PrivacyError, match="invalid source seal"):
        supplement.verify_anonymous_supplement(bad)


def test_standalone_verifier_rejects_an_empty_artifact_ledger(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "e", extra_paths=("notes.tex",))
    good = tmp_path / "good.zip"
    supplement.build_anonymous_supplement(
        root=root,
        include_paths=["notes.tex"],
        source_seal_path=seal,
        checksum_path=sums,
        public_bundle_path=public,
        output_path=good,
        require_git_checkout=False,
    )
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["artifacts"] = []
    entries = {name: payload for name, payload in entries.items() if not name.startswith("artifacts/")}
    entries["manifest.json"] = supplement._canonical_json(manifest)
    bad = tmp_path / "empty.zip"
    build_deterministic_zip(bad, entries)

    with pytest.raises(PrivacyError, match="non-empty"):
        supplement.verify_anonymous_supplement(bad)


def test_standalone_verifier_rejects_forged_artifact_provenance(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "f", extra_paths=("notes.tex",))
    good = tmp_path / "good.zip"
    supplement.build_anonymous_supplement(
        root=root,
        include_paths=["notes.tex"],
        source_seal_path=seal,
        checksum_path=sums,
        public_bundle_path=public,
        output_path=good,
        require_git_checkout=False,
    )
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["artifacts"][0].update({"source_bytes": -99, "source_sha256": "not-a-hash", "normalization": "made-up"})
    entries["manifest.json"] = supplement._canonical_json(manifest)
    bad = tmp_path / "forged.zip"
    build_deterministic_zip(bad, entries)

    with pytest.raises(PrivacyError, match="artifact provenance"):
        supplement.verify_anonymous_supplement(bad)


def test_standalone_verifier_rejects_pdf_normalization_on_a_non_pdf_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "0", extra_paths=("notes.tex",))
    good = tmp_path / "good.zip"
    supplement.build_anonymous_supplement(
        root=root,
        include_paths=["notes.tex"],
        source_seal_path=seal,
        checksum_path=sums,
        public_bundle_path=public,
        output_path=good,
        require_git_checkout=False,
    )
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["artifacts"][0]["normalization"] = "verified-pdf-byte-identical-copy"
    entries["manifest.json"] = supplement._canonical_json(manifest)
    bad = tmp_path / "bad.zip"
    build_deterministic_zip(bad, entries)

    with pytest.raises(PrivacyError, match="non-PDF artifact"):
        supplement.verify_anonymous_supplement(bad)


def test_standalone_verifier_rejects_duplicate_artifact_rows(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "1", extra_paths=("notes.tex",))
    good = tmp_path / "good.zip"
    supplement.build_anonymous_supplement(
        root=root,
        include_paths=["notes.tex"],
        source_seal_path=seal,
        checksum_path=sums,
        public_bundle_path=public,
        output_path=good,
        require_git_checkout=False,
    )
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["artifacts"].append(dict(manifest["artifacts"][0]))
    entries["manifest.json"] = supplement._canonical_json(manifest)
    bad = tmp_path / "duplicate.zip"
    build_deterministic_zip(bad, entries)

    with pytest.raises(PrivacyError, match="duplicated or non-canonical"):
        supplement.verify_anonymous_supplement(bad)


def test_builder_rejects_an_incomplete_release_checksum_receipt(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal = root / "seal.json"
    _write_seal(seal, "2")
    sums = root / "sums.txt"
    sums.write_text(_sha(notes) + "  notes.tex\n", encoding="utf-8")

    with pytest.raises(PrivacyError, match="required checksum entries are missing"):
        supplement.build_anonymous_supplement(
            root=root,
            include_paths=["notes.tex"],
            source_seal_path=seal,
            checksum_path=sums,
            public_bundle_path=_public_bundle(root),
            output_path=tmp_path / "anonymous.zip",
            require_git_checkout=False,
        )


def test_standalone_verifier_rejects_valid_but_unbound_source_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    manuscript = root / "paper.docx"
    manuscript.write_bytes(_docx_with_metadata())
    seal, sums, public = _complete_release_controls(root, "3", extra_paths=("paper.docx",))
    good = tmp_path / "good.zip"
    supplement.build_anonymous_supplement(
        root=root,
        include_paths=["paper.docx"],
        source_seal_path=seal,
        checksum_path=sums,
        public_bundle_path=public,
        output_path=good,
        require_git_checkout=False,
    )
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["artifacts"][0]["source_sha256"] = "a" * 64
    entries["manifest.json"] = supplement._canonical_json(manifest)
    bad = tmp_path / "bad.zip"
    build_deterministic_zip(bad, entries)

    with pytest.raises(PrivacyError, match="source authority mismatch"):
        supplement.verify_anonymous_supplement(bad)


def test_standalone_verifier_cross_binds_public_bundle_to_checksums(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "4", extra_paths=("notes.tex",))
    good = tmp_path / "good.zip"
    supplement.build_anonymous_supplement(
        root=root,
        include_paths=["notes.tex"],
        source_seal_path=seal,
        checksum_path=sums,
        public_bundle_path=public,
        output_path=good,
        require_git_checkout=False,
    )
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["commitments"]["cct20_public_evidence_bundle"]["sha256"] = "b" * 64
    entries["manifest.json"] = supplement._canonical_json(manifest)
    bad = tmp_path / "bad.zip"
    build_deterministic_zip(bad, entries)

    with pytest.raises(PrivacyError, match="CCT-20.*checksum"):
        supplement.verify_anonymous_supplement(bad)


def test_standalone_verifier_rejects_a_truncated_embedded_checksum_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "8", extra_paths=("notes.tex",))
    good = tmp_path / "good.zip"
    supplement.build_anonymous_supplement(
        root=root,
        include_paths=["notes.tex"],
        source_seal_path=seal,
        checksum_path=sums,
        public_bundle_path=public,
        output_path=good,
        require_git_checkout=False,
    )
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    receipt_path = "receipts/KBOUND_RELEASE_SHA256SUMS.txt"
    truncated = b"\n".join(entries[receipt_path].splitlines()[1:]) + b"\n"
    entries[receipt_path] = truncated
    manifest = json.loads(entries["manifest.json"])
    commitment = manifest["commitments"]["release_checksums"]
    commitment["bytes"] = len(truncated)
    commitment["sha256"] = hashlib.sha256(truncated).hexdigest()
    entries["manifest.json"] = supplement._canonical_json(manifest)
    bad = tmp_path / "bad.zip"
    build_deterministic_zip(bad, entries)

    with pytest.raises(PrivacyError, match="required checksum entries are missing"):
        supplement.verify_anonymous_supplement(bad)


def test_builder_validates_source_seal_from_a_git_worktree_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "5", extra_paths=("notes.tex",))
    (root / ".git").write_text("gitdir: /safe/fixture\n", encoding="utf-8")

    def reject_worktree(repo: Path, path: Path) -> dict[str, object]:
        raise ValueError("worktree sentinel")

    monkeypatch.setattr(source_seal_module, "validate_seal", reject_worktree)
    with pytest.raises(PrivacyError, match="worktree sentinel"):
        supplement.build_anonymous_supplement(
            root=root,
            include_paths=["notes.tex"],
            source_seal_path=seal,
            checksum_path=sums,
            public_bundle_path=public,
            output_path=tmp_path / "anonymous.zip",
        )


def test_builder_requires_a_git_checkout_by_default(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "7", extra_paths=("notes.tex",))

    with pytest.raises(PrivacyError, match="Git checkout"):
        supplement.build_anonymous_supplement(
            root=root,
            include_paths=["notes.tex"],
            source_seal_path=seal,
            checksum_path=sums,
            public_bundle_path=public,
            output_path=tmp_path / "anonymous.zip",
        )


def test_invalid_git_worktree_metadata_fails_with_a_privacy_error(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    notes = root / "notes.tex"
    notes.write_text("Anonymous supplement.\n", encoding="utf-8")
    seal, sums, public = _complete_release_controls(root, "6", extra_paths=("notes.tex",))
    (root / ".git").write_text("gitdir: /missing/repository\n", encoding="utf-8")

    with pytest.raises(PrivacyError, match="does not validate against repository"):
        supplement.build_anonymous_supplement(
            root=root,
            include_paths=["notes.tex"],
            source_seal_path=seal,
            checksum_path=sums,
            public_bundle_path=public,
            output_path=tmp_path / "anonymous.zip",
        )
