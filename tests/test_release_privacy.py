from __future__ import annotations

import io
import stat
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from docs.research.kbound.scripts import release_privacy as privacy


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


@pytest.mark.parametrize(
    "value",
    [
        "/Us" + "ers/reviewer/secret.json",
        "/Volumes" + "/T9/evidence.json",
        "file:///private/result.json",
        "FILE:///private/result.json",
        "~/result.json",
        "$HOME/result.json",
        "${HOME}/result.json",
        "../result.json",
        "safe/../../result.json",
        r"C:\\Users\\reviewer\\result.json",
        r"\\server\\share\\result.json",
        "safe\\result.json",
        "",
        ".",
    ],
)
def test_public_member_path_rejects_absolute_traversal_and_non_posix(value: str) -> None:
    with pytest.raises(privacy.PrivacyError):
        privacy.validate_public_member_path(value)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(privacy.PrivacyError, match="duplicate JSON key"):
        privacy.strict_json_loads(b'{"x": 1, "x": 2}')
    with pytest.raises(privacy.PrivacyError, match="non-finite"):
        privacy.strict_json_loads(b'{"x": NaN}')


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":Infinity}'])
def test_json_members_are_parsed_strictly_during_recursive_scans(payload: bytes) -> None:
    with pytest.raises(privacy.PrivacyError):
        privacy.scan_member("evidence.json", payload)


def test_privacy_scanner_decodes_utf16_and_case_insensitive_file_uris() -> None:
    with pytest.raises(privacy.PrivacyError, match="identity"):
        privacy.scan_member("metadata.txt", "Pratik Niroula".encode("utf-16"))
    with pytest.raises(privacy.PrivacyError, match="file URI"):
        privacy.scan_member("metadata.txt", b"FILE:///private/result")


def test_privacy_scanner_distinguishes_file_label_from_file_uri() -> None:
    privacy.scan_member(
        "runner.py",
        b'raise RuntimeError(f"sealed image is not a regular file: {path}")\n',
    )
    with pytest.raises(privacy.PrivacyError, match="file URI"):
        privacy.scan_member("metadata.txt", b"file:/private/result")


@pytest.mark.parametrize(
    "payload",
    [b"/private/tmp/result.json", b"/data/private/result.json", b"D:\\work\\result.json"],
)
def test_privacy_scanner_rejects_general_absolute_machine_paths(payload: bytes) -> None:
    with pytest.raises(privacy.PrivacyError, match="path disclosure"):
        privacy.scan_member("metadata.txt", payload)


def test_privacy_scanner_allows_a_standard_interpreter_shebang_only() -> None:
    privacy.scan_member("runner.py", b"#!/usr/bin/env python3\nprint('safe')\n")
    privacy.scan_member("runner.sh", b"#!/bin/bash\nset -euo pipefail\n")
    privacy.scan_member("references.txt", b"https://github.com/example/project\n")
    with pytest.raises(privacy.PrivacyError, match="path disclosure"):
        privacy.scan_member("runner.py", b"#!/usr/bin/env python3\nDATA=/private/tmp/input\n")


@pytest.mark.parametrize("payload", [b"$HOME/private.json", b"${HOME}/private.json"])
def test_privacy_scanner_rejects_home_alias_disclosures(payload: bytes) -> None:
    with pytest.raises(privacy.PrivacyError, match="home alias"):
        privacy.scan_member("metadata.txt", payload)


@pytest.mark.parametrize(
    ("member", "payload", "needle"),
    [
        ("paper.txt", b"Pratik Niroula", "identity"),
        ("paper.txt", b"author@example.edu", "email"),
        ("paper.txt", b"/Us" + b"ers/reviewer/private", "private POSIX"),
        ("paper.txt", b"file:///tmp/private", "file URI"),
        (".git/config", b"safe", "forbidden member"),
        ("._paper.tex", b"safe", "forbidden member"),
        ("~$paper.docx", b"safe", "unsafe public member"),
        ("word/comments.xml", b"<comments/>", "forbidden OOXML"),
        ("word/document.xml", b'<w:ins w:author="A">', "tracked-change"),
    ],
)
def test_privacy_scanner_fails_closed(member: str, payload: bytes, needle: str) -> None:
    with pytest.raises(privacy.PrivacyError, match=needle):
        privacy.scan_member(member, payload)


def test_nested_archive_is_scanned_recursively() -> None:
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("hidden.txt", "contact author@example.edu")

    with pytest.raises(privacy.PrivacyError, match="email"):
        privacy.scan_member("nested.zip", nested.getvalue())


def test_deterministic_archive_has_stable_bytes_and_verifies(tmp_path: Path) -> None:
    entries = {
        "source/main.tex": b"Anonymous authors\n",
        "README.txt": b"Self-contained supplementary package.\n",
    }
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    privacy.build_deterministic_zip(first, entries)
    privacy.build_deterministic_zip(second, dict(reversed(list(entries.items()))))

    assert first.read_bytes() == second.read_bytes()
    receipt = privacy.verify_anonymous_zip(first)
    assert receipt["member_count"] == 2
    assert receipt["archive_sha256"] == privacy.file_sha256(first)


def test_archive_builder_rejects_unclassified_leak_instead_of_omitting_it(
    tmp_path: Path,
) -> None:
    output = tmp_path / "anonymous.zip"
    with pytest.raises(privacy.PrivacyError, match="identity"):
        privacy.build_deterministic_zip(
            output,
            {
                "safe.txt": b"safe\n",
                "unsafe.txt": b"Pratik Niroula\n",
            },
        )
    assert not output.exists()


def test_generic_archive_api_cannot_silently_accept_an_unverified_pdf(
    tmp_path: Path,
) -> None:
    output = tmp_path / "anonymous.zip"
    with pytest.raises(privacy.PrivacyError, match="type-specific"):
        privacy.build_deterministic_zip(
            output,
            {"paper.pdf": b"%PDF-1.7\nfixture"},
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "allowlist",
    [
        ("notes.txt",),
        ("missing.pdf",),
        (1,),
    ],
)
def test_verified_pdf_allowlist_fails_closed_on_caller_misuse(
    tmp_path: Path,
    allowlist: tuple[object, ...],
) -> None:
    with pytest.raises(privacy.PrivacyError):
        privacy.build_deterministic_zip(
            tmp_path / "anonymous.zip",
            {"paper.pdf": b"%PDF-1.7\nfixture"},
            verified_pdf_members=allowlist,  # type: ignore[arg-type]
        )


def test_archive_builder_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(privacy.PrivacyError, match="symlink"):
        privacy.build_deterministic_zip(linked / "anonymous.zip", {"safe.txt": b"safe"})


def test_zip_verifier_rejects_archive_comment_metadata(tmp_path: Path) -> None:
    archive_path = tmp_path / "commented.zip"
    privacy.build_deterministic_zip(archive_path, {"safe.txt": b"safe"})
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.comment = b"Pratik Niroula /Users/private/secret"

    with pytest.raises(privacy.PrivacyError, match="archive comment"):
        privacy.verify_anonymous_zip(archive_path)


def test_zip_verifier_rejects_member_comment_and_extra_metadata(tmp_path: Path) -> None:
    archive_path = tmp_path / "member-metadata.zip"
    info = zipfile.ZipInfo("safe.txt", date_time=privacy.FIXED_ZIP_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.comment = b"private comment"
    info.extra = b"\xff\xff\x04\x00name"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, b"safe")

    with pytest.raises(privacy.PrivacyError, match="member metadata"):
        privacy.verify_anonymous_zip(archive_path)


def test_zip_verifier_rejects_noncanonical_regular_file_mode(tmp_path: Path) -> None:
    archive_path = tmp_path / "executable.zip"
    info = zipfile.ZipInfo("safe.txt", date_time=privacy.FIXED_ZIP_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o755) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, b"safe")

    with pytest.raises(privacy.PrivacyError, match="noncanonical ZIP mode"):
        privacy.verify_anonymous_zip(archive_path)


def test_prefixed_nested_zip_cannot_hide_compressed_private_content() -> None:
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("secret.txt", "Pratik Niroula" * 30)
    payload = b"MZ-STUB" + nested.getvalue()
    if len(payload) % 2:
        payload += b"X"
    assert zipfile.is_zipfile(io.BytesIO(payload))

    with pytest.raises(privacy.PrivacyError, match="identity"):
        privacy.scan_member("nested.zip", payload)


def test_top_level_zip_total_size_is_checked_before_member_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "oversized.zip"
    privacy.build_deterministic_zip(
        archive_path,
        {"a.txt": b"a" * 1_000, "b.txt": b"b" * 1_000},
    )
    assert archive_path.stat().st_size < 1_500
    monkeypatch.setattr(privacy, "MAX_TOTAL_BYTES", 1_500)

    with pytest.raises(privacy.PrivacyError, match="expands beyond safety limit"):
        privacy.verify_anonymous_zip(archive_path)


@pytest.mark.parametrize("placement", ["prefix", "trailing"])
def test_zip_verifier_rejects_bytes_outside_the_canonical_archive(tmp_path: Path, placement: str) -> None:
    canonical = tmp_path / "canonical.zip"
    privacy.build_deterministic_zip(canonical, {"safe.txt": b"safe"})
    raw = canonical.read_bytes()
    malformed = tmp_path / f"{placement}.zip"
    malformed.write_bytes(b"hidden" + raw if placement == "prefix" else raw + b"hidden")

    with pytest.raises(privacy.PrivacyError, match="canonical ZIP structure"):
        privacy.verify_anonymous_zip(malformed)


def test_png_sanitizer_removes_unknown_ancillary_private_chunks() -> None:
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"vpAg", b"Pratik Niroula /Users/private")
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + _png_chunk(b"IEND", b"")
    )

    scrubbed = privacy.sanitize_png_metadata(payload)

    assert b"vpAg" not in scrubbed
    assert b"Pratik" not in scrubbed
    privacy.scan_member("figure.png", scrubbed)


def test_zip_verifier_rejects_local_header_only_extra_metadata(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.zip"
    privacy.build_deterministic_zip(canonical, {"safe.txt": b"safe"})
    raw = bytearray(canonical.read_bytes())
    assert raw[:4] == b"PK\x03\x04"
    name_size = struct.unpack_from("<H", raw, 26)[0]
    assert struct.unpack_from("<H", raw, 28)[0] == 0
    hidden = b"identity-metadata"
    insertion = 30 + name_size
    raw[insertion:insertion] = hidden
    struct.pack_into("<H", raw, 28, len(hidden))
    end_offset = len(raw) - 22
    central_offset = struct.unpack_from("<I", raw, end_offset + 16)[0]
    struct.pack_into("<I", raw, end_offset + 16, central_offset + len(hidden))
    malformed = tmp_path / "local-extra.zip"
    malformed.write_bytes(raw)
    with zipfile.ZipFile(malformed) as archive:
        assert archive.infolist()[0].extra == b""
        assert archive.read("safe.txt") == b"safe"

    with pytest.raises(privacy.PrivacyError, match="local-header metadata"):
        privacy.verify_anonymous_zip(malformed)


def test_recursive_zip_scan_rejects_local_header_only_extra_metadata(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.zip"
    privacy.build_deterministic_zip(canonical, {"safe.txt": b"safe"})
    raw = bytearray(canonical.read_bytes())
    name_size = struct.unpack_from("<H", raw, 26)[0]
    hidden = b"Pratik Niroula"
    insertion = 30 + name_size
    raw[insertion:insertion] = hidden
    struct.pack_into("<H", raw, 28, len(hidden))
    end_offset = len(raw) - 22
    central_offset = struct.unpack_from("<I", raw, end_offset + 16)[0]
    struct.pack_into("<I", raw, end_offset + 16, central_offset + len(hidden))

    with pytest.raises(privacy.PrivacyError, match="local-header metadata"):
        privacy.scan_member("nested.zip", bytes(raw))


def test_recursive_zip_scan_rejects_an_empty_archive() -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w"):
        pass

    with pytest.raises(privacy.PrivacyError, match="empty nested archive"):
        privacy.scan_member("nested.zip", payload.getvalue())


def test_recursive_zip_scan_checks_directory_entry_metadata() -> None:
    payload = io.BytesIO()
    directory = zipfile.ZipInfo("folder/", date_time=privacy.FIXED_ZIP_TIME)
    directory.comment = b"Pratik Niroula"
    safe = zipfile.ZipInfo("folder/safe.txt", date_time=privacy.FIXED_ZIP_TIME)
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(directory, b"")
        archive.writestr(safe, b"safe")

    with pytest.raises(privacy.PrivacyError, match="ZIP member metadata"):
        privacy.scan_member("nested.zip", payload.getvalue())
