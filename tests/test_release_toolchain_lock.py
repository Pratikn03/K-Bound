from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from docs.research.kbound.scripts import render_pdf_pages

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "docs/research/kbound/scripts/verify_release_toolchain.py"
PROFILE_PATH = ROOT / "docs/research/kbound/release_toolchain_macos_arm64.json"
EXPECTED_TOOLS = {
    "latexmk",
    "latexpand",
    "pandoc",
    "perl",
    "pdfdetach",
    "pdfinfo",
    "pdflatex",
    "pdftoppm",
    "pdftotext",
    "soffice",
}


def test_release_runner_exposes_verified_tool_overrides_to_all_child_steps() -> None:
    """The verifier and the document builders must resolve the same binaries.

    The bundled Poppler/LibreOffice programs are not necessarily on the login
    ``PATH``.  A release that verifies ``KBOUND_TOOL_*`` overrides but then
    lets later child processes rediscover unrelated/default programs is not a
    pinned release.
    """

    runbook = (ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text(encoding="utf-8")
    configure = runbook.index("configure_release_tool_path()")
    verify = runbook.index("verify_release_toolchain_for_phase()")
    invoke = runbook.index('configure_release_tool_path "$resolved_file"', verify)
    dispatch = runbook.index('case "$MODE" in')
    assert configure < verify < invoke < dispatch
    assert "--resolved-tools-output" in runbook
    for name in EXPECTED_TOOLS:
        assert f"KBOUND_TOOL_{name.upper()}" in runbook
    assert 'export "$override_name=$override"' in runbook
    assert 'export PATH="$tool_directory:$PATH"' not in runbook
    for phase in (
        "step_preflight",
        "step_validate_results",
        "step_generate",
        "step_test",
        "step_pdf",
        "step_anonymous_supplement",
    ):
        body = runbook.split(f"{phase}() {{", 1)[1].split("\n}", 1)[0]
        assert "verify_release_toolchain_for_phase" in body


def _verifier():
    assert VERIFIER_PATH.is_file(), VERIFIER_PATH
    spec = importlib.util.spec_from_file_location("verify_release_toolchain", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_executable(path: Path, version_line: str) -> str:
    path.write_text(
        f"#!/bin/sh\nprintf '%s\\n' '{version_line}'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return _verifier().sha256_file(path)


def _profile(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "probe"
    digest = _write_executable(executable, "probe version 1.2.3")
    document: dict[str, object] = {
        "schema": "kbound-release-toolchain-lock-v1",
        "platform": "macos-arm64",
        "python_zlib": {"compile": "1.2.12", "runtime": "1.2.12"},
        "tools": {
            "probe": {
                "command": "probe",
                "sha256": digest,
                "version_args": ["--version"],
                "version_line": "probe version 1.2.3",
                "version_returncode": 0,
            }
        },
    }
    profile = tmp_path / "toolchain.json"
    profile.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return profile, document


def test_repository_release_toolchain_profile_is_canonical_and_complete() -> None:
    verifier = _verifier()
    profile = verifier.load_profile(PROFILE_PATH)

    assert profile["platform"] == "macos-arm64"
    assert set(profile["tools"]) == EXPECTED_TOOLS
    assert profile["python_zlib"] == {
        "compile": "1.2.12",
        "runtime": "1.2.12",
    }
    assert PROFILE_PATH.read_bytes() == verifier.canonical_json_bytes(profile)


def test_poppler_v2_changes_only_four_reviewed_executable_identities() -> None:
    verifier = _verifier()
    historical = verifier.load_profile(PROFILE_PATH)
    new_path = PROFILE_PATH.with_name("release_toolchain_macos_arm64_v2.json")
    assert new_path.is_file(), "separate v2 authority is required"
    current = verifier.load_profile(new_path)
    expected = {
        "pdfdetach": "d4374729e7ab492697e1ef61b8a567da21b972ef02522c36295f975b979918ab",
        "pdfinfo": "23cc8f9bbe0a10109967b5811cb9a94d194b11380b43a40eac7089e676a98d83",
        "pdftoppm": "ab4c415cb29a229b039221de9af5a4ba539acc1aebd3a3a0a6dd677d11093c8e",
        "pdftotext": "e8fb86921c589674ee6ba880594e9dd3dba4d471d2b7c7fdb00eac55975db25a",
    }
    assert {k: v for k, v in current.items() if k != "tools"} == {
        k: v for k, v in historical.items() if k != "tools"
    }
    assert set(current["tools"]) == EXPECTED_TOOLS
    for name, old_record in historical["tools"].items():
        new_record = current["tools"][name]
        if name in expected:
            assert new_record == {**old_record, "sha256": expected[name], "version_line": f"{name} version 26.08.0"}
        else:
            assert new_record == old_record
    assert hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() == "c38388b86f5591bc101b4eb36b615774dfdabbb4a56d5ab13ca72acce5a04035"
    receipt = PROFILE_PATH.parent / "audits/release_toolchain_2026_09_02.json"
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == "aa28c196e4888f9d66180403b062c909e52b7826e99575b1e574eb68d8de185b"


def test_receipt_binds_explicit_v2_profile_name_and_bytes(tmp_path, monkeypatch):
    verifier = _verifier()
    old_path, profile = _profile(tmp_path)
    new_path = tmp_path / "release_toolchain_macos_arm64_v2.json"
    new_path.write_bytes(old_path.read_bytes())
    monkeypatch.setattr(verifier.shutil, "which", lambda command: str(tmp_path / "tools/probe"))
    receipt = verifier.build_receipt(profile, profile_path=new_path, runtime_platform="macos-arm64",
                                     zlib_compile="1.2.12", zlib_runtime="1.2.12")
    assert receipt["profile"] == {"path": new_path.name, "sha256": hashlib.sha256(new_path.read_bytes()).hexdigest()}


def test_toolchain_verification_rejects_a_one_tool_profile(tmp_path: Path) -> None:
    verifier = _verifier()
    _, profile = _profile(tmp_path)

    with pytest.raises(ValueError, match="exactly the canonical ten tools"):
        verifier.require_canonical_tool_profile(profile)


def test_partial_tool_profile_cannot_leave_a_verified_public_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A one-tool fixture must never be publishable as a release receipt.

    Removing the canonical-profile gate in the CLI would make this test leave a
    public ``status=verified`` receipt behind.
    """

    verifier = _verifier()
    profile_path, _ = _profile(tmp_path)
    output = tmp_path / "receipt.json"
    output.write_text('{"status":"verified"}\n', encoding="ascii")

    assert verifier.main(["--profile", str(profile_path), "--output", str(output)]) == 1

    assert "exactly the canonical ten tools" in capsys.readouterr().err
    assert not output.exists()


def test_perl_script_version_probe_uses_the_profile_bound_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _verifier()
    tools = tmp_path / "tools"
    tools.mkdir()
    latexpand = tools / "latexpand"
    perl = tools / "perl"
    latexpand.write_text(
        "#!/bin/sh\nprintf 'untrusted shebang interpreter output\\n'\n",
        encoding="utf-8",
    )
    perl.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '-v' ]; then printf 'pinned perl 1.0\\n'; "
        "else printf 'pinned latexpand 2.0\\n'; fi\n",
        encoding="utf-8",
    )
    latexpand.chmod(0o755)
    perl.chmod(0o755)
    profile_path = tmp_path / "toolchain.json"
    profile = {
        "schema": "kbound-release-toolchain-lock-v1",
        "platform": "macos-arm64",
        "python_zlib": {"compile": "1.2.12", "runtime": "1.2.12"},
        "tools": {
            "latexpand": {
                "command": "latexpand",
                "sha256": verifier.sha256_file(latexpand),
                "version_args": ["--version"],
                "version_line": "pinned latexpand 2.0",
                "version_returncode": 0,
            },
            "perl": {
                "command": "perl",
                "sha256": verifier.sha256_file(perl),
                "version_args": ["-v"],
                "version_line": "pinned perl 1.0",
                "version_returncode": 0,
            },
        },
    }
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        verifier.shutil,
        "which",
        lambda command: str({"latexpand": latexpand, "perl": perl}[command]),
    )

    receipt = verifier.build_receipt(
        profile,
        profile_path=profile_path,
        runtime_platform="macos-arm64",
        zlib_compile="1.2.12",
        zlib_runtime="1.2.12",
    )

    assert receipt["tools"]["latexpand"]["version_line"] == "pinned latexpand 2.0"
    assert receipt["tools"]["latexpand"]["sha256"] == verifier.sha256_file(latexpand)
    assert receipt["tools"]["perl"]["sha256"] == verifier.sha256_file(perl)


def test_verified_realpaths_are_exported_only_to_private_ephemeral_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _verifier()
    profile_path, profile = _profile(tmp_path)
    executable = (tmp_path / "tools/probe").resolve()
    monkeypatch.setattr(verifier.shutil, "which", lambda command: str(executable))
    resolved: dict[str, str] = {}

    receipt = verifier.build_receipt(
        profile,
        profile_path=profile_path,
        runtime_platform="macos-arm64",
        zlib_compile="1.2.12",
        zlib_runtime="1.2.12",
        resolved_tools=resolved,
    )
    output = tmp_path / "resolved-tools.tsv"
    verifier.write_resolved_tools_atomic(output, resolved)

    assert resolved == {"KBOUND_TOOL_PROBE": str(executable)}
    assert output.read_text(encoding="utf-8") == f"KBOUND_TOOL_PROBE\t{executable}\n"
    assert output.stat().st_mode & 0o777 == 0o600
    assert str(tmp_path).encode() not in verifier.canonical_json_bytes(receipt)


def test_explicit_tool_override_must_already_be_an_absolute_realpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _verifier()
    profile_path, profile = _profile(tmp_path)
    executable = tmp_path / "tools/probe"
    alias = tmp_path / "probe-alias"
    alias.symlink_to(executable)

    monkeypatch.setenv("KBOUND_TOOL_PROBE", str(alias))
    with pytest.raises(ValueError, match="absolute non-symlink realpath"):
        verifier.build_receipt(
            profile,
            profile_path=profile_path,
            runtime_platform="macos-arm64",
            zlib_compile="1.2.12",
            zlib_runtime="1.2.12",
        )

    monkeypatch.setenv("KBOUND_TOOL_PROBE", "tools/probe")
    with pytest.raises(ValueError, match="absolute non-symlink realpath"):
        verifier.build_receipt(
            profile,
            profile_path=profile_path,
            runtime_platform="macos-arm64",
            zlib_compile="1.2.12",
            zlib_runtime="1.2.12",
        )


def test_pdf_renderer_executes_exact_pdfinfo_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = tmp_path / "pinned/pdfinfo-real"
    shadow = tmp_path / "shadow/pdfinfo"
    pinned.parent.mkdir()
    shadow.parent.mkdir()
    pinned.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    shadow.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
    pinned.chmod(0o755)
    shadow.chmod(0o755)
    monkeypatch.setenv("KBOUND_TOOL_PDFINFO", str(pinned))
    monkeypatch.setenv("PATH", str(shadow.parent))
    observed: list[list[str]] = []

    def check_output(command: list[str], **kwargs: object) -> str:
        observed.append(command)
        return "Pages: 7\n"

    monkeypatch.setattr(render_pdf_pages.subprocess, "check_output", check_output)
    assert render_pdf_pages.page_count(tmp_path / "paper.pdf") == 7
    assert observed == [[str(pinned), str(tmp_path / "paper.pdf")]]


def test_pdf_renderer_executes_exact_pdftoppm_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = tmp_path / "pinned/pdftoppm-real"
    shadow = tmp_path / "shadow/pdftoppm"
    pinned.parent.mkdir()
    shadow.parent.mkdir()
    pinned.write_text(
        "#!/bin/sh\nprintf 'rendered' > \"${5}-1.png\"\n",
        encoding="utf-8",
    )
    shadow.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
    pinned.chmod(0o755)
    shadow.chmod(0o755)
    monkeypatch.setenv("KBOUND_TOOL_PDFTOPPM", str(pinned))
    monkeypatch.setenv("PATH", str(shadow.parent))
    monkeypatch.setattr(render_pdf_pages, "page_count", lambda _pdf: 1)
    output = tmp_path / "rendered"

    render_pdf_pages.render(tmp_path / "paper.pdf", output)

    assert (output / "page-1.png").read_text(encoding="utf-8") == "rendered"


def test_toolchain_verification_rejects_missing_wrong_or_changed_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _verifier()
    profile_path, profile = _profile(tmp_path)

    monkeypatch.setattr(verifier.shutil, "which", lambda command: None)
    with pytest.raises(ValueError, match="required tool.*probe"):
        verifier.build_receipt(
            profile,
            profile_path=profile_path,
            runtime_platform="macos-arm64",
            zlib_compile="1.2.12",
            zlib_runtime="1.2.12",
        )


def test_toolchain_verification_rejects_executable_replaced_during_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid pre-probe hash must not authorize bytes swapped in during probing."""

    verifier = _verifier()
    profile_path, profile = _profile(tmp_path)
    executable = tmp_path / "tools/probe"
    monkeypatch.setattr(verifier.shutil, "which", lambda _command: str(executable))

    def replacing_probe(_command: list[str], _args: list[str], *, name: str) -> tuple[str, int]:
        assert name == "probe"
        executable.write_text("#!/bin/sh\nprintf 'probe version 1.2.3\\n'\n# replacement\n", encoding="utf-8")
        executable.chmod(0o755)
        return "probe version 1.2.3", 0

    monkeypatch.setattr(verifier, "_version_probe", replacing_probe)

    with pytest.raises(ValueError, match="changed during verification"):
        verifier.build_receipt(
            profile,
            profile_path=profile_path,
            runtime_platform="macos-arm64",
            zlib_compile="1.2.12",
            zlib_runtime="1.2.12",
        )

    monkeypatch.setattr(verifier.shutil, "which", lambda command: str(executable))
    executable.write_text("#!/bin/sh\nprintf 'probe version 9\\n'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verifier.build_receipt(
            profile,
            profile_path=profile_path,
            runtime_platform="macos-arm64",
            zlib_compile="1.2.12",
            zlib_runtime="1.2.12",
        )


@pytest.mark.parametrize(
    ("platform", "compile_version", "runtime_version", "message"),
    [
        ("linux-amd64", "1.2.12", "1.2.12", "platform"),
        ("macos-arm64", "1.3.1", "1.2.12", "zlib compile"),
        ("macos-arm64", "1.2.12", "1.3.1", "zlib runtime"),
    ],
)
def test_toolchain_verification_rejects_runtime_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    compile_version: str,
    runtime_version: str,
    message: str,
) -> None:
    verifier = _verifier()
    profile_path, profile = _profile(tmp_path)
    monkeypatch.setattr(
        verifier.shutil,
        "which",
        lambda command: str(tmp_path / "tools/probe"),
    )
    with pytest.raises(ValueError, match=message):
        verifier.build_receipt(
            profile,
            profile_path=profile_path,
            runtime_platform=platform,
            zlib_compile=compile_version,
            zlib_runtime=runtime_version,
        )


def test_toolchain_receipt_writer_is_atomic_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    verifier = _verifier()
    output = tmp_path / "receipt.json"
    verifier.write_receipt_atomic(output, {"status": "verified"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "verified"}

    outside = tmp_path / "outside.json"
    outside.write_text('{"sentinel":true}\n', encoding="ascii")
    output.unlink()
    output.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        verifier.write_receipt_atomic(output, {"status": "forged"})
    assert outside.read_text(encoding="ascii") == '{"sentinel":true}\n'
