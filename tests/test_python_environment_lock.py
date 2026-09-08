from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "docs" / "research" / "kbound" / "scripts" / "verify_python_environment.py"
LOCKS = {
    "requirements-api-py311-linux.lock.txt": ("3.11.15", "linux-amd64"),
    "requirements-ci-py312-linux.lock.txt": ("3.12.13", "linux-amd64"),
    "requirements-release-macos-arm64.lock.txt": ("3.12.13", "macos-arm64"),
}
CONTENT_PROFILE = ROOT / "docs/research/kbound/release_python_environment_macos_arm64.json"


def _verifier():
    assert VERIFIER_PATH.is_file(), f"missing environment verifier: {VERIFIER_PATH.relative_to(ROOT)}"
    spec = importlib.util.spec_from_file_location("verify_python_environment", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lock_text(*requirements: str, python: str = "3.12.13", platform: str = "macos-arm64") -> str:
    header = [
        "# K-Bound reproducibility lock",
        f"# Python: {python}",
        f"# Platform: {platform}",
        "# Exclude-Newer: 2026-09-01T00:00:00Z",
        "# Resolver: uv 0.11.19",
        "# Only-Binary: :all:",
        "",
    ]
    return "\n".join([*header, *requirements, ""])


def _record_hash(payload: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")


def _fake_distribution(root: Path, *, module_payload: bytes = b"VALUE = 1\n"):
    module = root / "demo.py"
    metadata_dir = root / "demo-1.0.dist-info"
    metadata = metadata_dir / "METADATA"
    wheel = metadata_dir / "WHEEL"
    record = metadata_dir / "RECORD"
    metadata_dir.mkdir(parents=True)
    module.write_bytes(module_payload)
    metadata.write_text("Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n", encoding="utf-8")
    wheel.write_text("Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n", encoding="utf-8")
    rows = []
    for path in (module, metadata, wheel):
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        rows.append(f"{relative},sha256={_record_hash(payload)},{len(payload)}")
    rows.append("demo-1.0.dist-info/RECORD,,")
    record.write_text("\n".join(rows) + "\n", encoding="utf-8")
    distributions = list(importlib.metadata.distributions(path=[str(root)]))
    assert len(distributions) == 1
    return distributions[0], module, record


def test_repository_locks_are_hash_complete_and_match_their_declared_targets() -> None:
    verifier = _verifier()
    for name, (python, platform) in LOCKS.items():
        lock = verifier.load_lock(ROOT / name)
        assert lock.python == python
        assert lock.platform == platform
        assert lock.exclude_newer == "2026-09-01T00:00:00Z"
        assert lock.only_binary == ":all:"
        assert lock.packages


def test_release_lock_fixes_the_canonical_numpy_version() -> None:
    lock = _verifier().load_lock(ROOT / "requirements-release-macos-arm64.lock.txt")
    assert lock.packages["numpy"] == "2.4.4"


def test_release_content_profile_is_sealed_and_required_by_preflight() -> None:
    profile = json.loads(CONTENT_PROFILE.read_text(encoding="utf-8"))
    assert profile["schema"] == "kbound-python-environment-content-v1"
    assert profile["runtime"] == {"python": "3.12.13", "platform": "macos-arm64"}
    assert profile["distributions"]["numpy"]["version"] == "2.4.4"
    assert sum(row["file_count"] for row in profile["distributions"].values()) == 27816
    assert len(profile["aggregate_sha256"]) == 64

    runbook = (ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text(encoding="utf-8")
    assert '--content-profile "$KB/release_python_environment_macos_arm64_v2.json"' in runbook
    source_seal = (ROOT / "docs/research/kbound/scripts/build_release_source_seal.py").read_text(encoding="utf-8")
    assert "docs/research/kbound/release_python_environment_macos_arm64.json" in source_seal
    checksum_verifier = (ROOT / "docs/research/kbound/scripts/verify_release_checksums.py").read_text(encoding="utf-8")
    assert "docs/research/kbound/release_python_environment_macos_arm64.json" in checksum_verifier


def test_active_v2_profile_is_the_independently_authenticated_candidate() -> None:
    active = CONTENT_PROFILE.with_name("release_python_environment_macos_arm64_v2.json")
    assert active.is_file(), "active portable v2 profile is missing"
    assert hashlib.sha256(active.read_bytes()).hexdigest() == (
        "5dd49432b24c24f4a9408fc398876eb43f742bd1c07f2d7a566a4d3c64acd145"
    )
    assert hashlib.sha256(CONTENT_PROFILE.read_bytes()).hexdigest() == (
        "3eab8a12592247254b0126ad4202e9a71591cd57ef7dd504b9568641c8d5964d"
    )
    profile = json.loads(active.read_bytes())
    assert profile["schema"] == "kbound-python-environment-content-v2"
    assert profile["aggregate_sha256"] == "eaa1aa6b460ae93e120407c844232cc33ba517b53dc16c68fa8ffb4855f2e5fb"


def test_release_profile_and_lock_include_every_python_release_gate_tool() -> None:
    """The verified release interpreter must be able to execute every Python gate."""

    expected = {
        "bandit": "1.9.4",
        "build": "1.3.0",
        "mypy": "2.1.0",
        "pre-commit": "4.6.0",
        "ruff": "0.15.13",
        "twine": "6.2.0",
    }
    profile = (ROOT / "requirements-release.txt").read_text(encoding="utf-8")
    lock = _verifier().load_lock(ROOT / "requirements-release-macos-arm64.lock.txt")

    for name, version in expected.items():
        profile_name = "bandit[toml]" if name == "bandit" else name
        assert f"{profile_name}=={version}" in profile
        assert lock.packages[name] == version


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            _lock_text(
                "numpy==2.4.4 --hash=sha256:" + "1" * 64,
                "NumPy==2.4.4 --hash=sha256:" + "2" * 64,
            ),
            "duplicate",
        ),
        (_lock_text("numpy>=2.4 --hash=sha256:" + "1" * 64), "exactly pinned"),
        (
            _lock_text("numpy @ https://example.invalid/numpy.whl#sha256=" + "1" * 64),
            "direct URL",
        ),
        (_lock_text("numpy==2.4.4"), "SHA-256"),
        (_lock_text("--index-url https://example.invalid/simple"), "option"),
    ],
)
def test_lock_parser_rejects_ambiguous_or_mutable_requirements(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.lock.txt"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _verifier().load_lock(path)


def test_lock_parser_rejects_header_target_disagreement(tmp_path: Path) -> None:
    path = tmp_path / "requirements-release-macos-arm64.lock.txt"
    path.write_text(
        _lock_text(
            "numpy==2.4.4 --hash=sha256:" + "1" * 64,
            python="3.11",
            platform="linux-amd64",
        ),
        encoding="utf-8",
    )
    lock = _verifier().load_lock(path)
    with pytest.raises(ValueError, match="Python"):
        _verifier().verify_runtime(lock, python_version="3.12.13", platform="macos-arm64")


def test_installed_distribution_agreement_rejects_missing_wrong_and_extra_packages() -> None:
    verifier = _verifier()
    expected = {"numpy": "2.4.4", "scipy": "1.17.1"}
    with pytest.raises(ValueError, match="missing.*scipy"):
        verifier.verify_installed(expected, {"numpy": "2.4.4"})
    with pytest.raises(ValueError, match="version.*numpy"):
        verifier.verify_installed(expected, {"numpy": "2.4.3", "scipy": "1.17.1"})
    with pytest.raises(ValueError, match="unexpected.*pandas"):
        verifier.verify_installed(
            expected,
            {"numpy": "2.4.4", "scipy": "1.17.1", "pandas": "3.0.3"},
        )


def test_receipt_binds_lock_runtime_and_installed_set(tmp_path: Path) -> None:
    path = tmp_path / "release.lock.txt"
    path.write_text(
        _lock_text(
            "numpy==2.4.4 --hash=sha256:" + "1" * 64,
            "scipy==1.17.1 --hash=sha256:" + "2" * 64,
        ),
        encoding="utf-8",
    )
    verifier = _verifier()
    lock = verifier.load_lock(path)
    receipt = verifier.build_receipt(
        lock,
        python_version="3.12.13",
        platform="macos-arm64",
        installed={"numpy": "2.4.4", "scipy": "1.17.1"},
        content_profile={
            "path": "release_python_environment_macos_arm64.json",
            "sha256": "3" * 64,
            "aggregate_sha256": "4" * 64,
        },
    )
    assert receipt["schema_version"] == 2
    assert receipt["lock"]["sha256"] == verifier.sha256_file(path)
    assert receipt["runtime"] == {"python": "3.12.13", "platform": "macos-arm64"}
    assert receipt["distributions"] == {"numpy": "2.4.4", "scipy": "1.17.1"}
    assert receipt["content_profile"]["aggregate_sha256"] == "4" * 64
    assert json.loads(verifier.canonical_json(receipt)) == receipt


def test_installed_content_profile_detects_mutated_package_bytes(tmp_path: Path) -> None:
    verifier = _verifier()
    distribution, module, _record = _fake_distribution(tmp_path)
    evidence = verifier.verify_distribution_contents([distribution], expected={"demo": "1.0"}, prefix=tmp_path)
    profile = verifier.build_content_profile(
        lock_sha256="1" * 64,
        python_version="3.12.13",
        platform="macos-arm64",
        evidence=evidence,
    )

    assert profile["schema"] == "kbound-python-environment-content-v1"
    assert profile["distributions"]["demo"]["file_count"] == 4
    assert len(profile["aggregate_sha256"]) == 64

    module.write_bytes(b"VALUE = 9\n")
    with pytest.raises(ValueError, match="RECORD hash mismatch"):
        verifier.verify_distribution_contents([distribution], expected={"demo": "1.0"}, prefix=tmp_path)


def test_installed_content_profile_rejects_unhashed_non_record_file(tmp_path: Path) -> None:
    verifier = _verifier()
    distribution, _module, record = _fake_distribution(tmp_path)
    record.write_text(
        record.read_text(encoding="utf-8").replace(
            next(line for line in record.read_text(encoding="utf-8").splitlines() if line.startswith("demo.py,")),
            "demo.py,,",
        ),
        encoding="utf-8",
    )
    distribution = list(importlib.metadata.distributions(path=[str(tmp_path)]))[0]
    with pytest.raises(ValueError, match="missing a sha256 RECORD hash"):
        verifier.verify_distribution_contents([distribution], expected={"demo": "1.0"}, prefix=tmp_path)


def test_installed_content_profile_rejects_duplicate_distribution_identity(tmp_path: Path) -> None:
    verifier = _verifier()
    distribution, _module, _record = _fake_distribution(tmp_path)
    with pytest.raises(ValueError, match="duplicate installed distribution"):
        verifier.verify_distribution_contents([distribution, distribution], expected={"demo": "1.0"}, prefix=tmp_path)


def test_receipt_writer_is_atomic_and_refuses_symlink_destination(
    tmp_path: Path,
) -> None:
    verifier = _verifier()
    receipt = {"schema_version": 1, "status": "verified"}
    destination = tmp_path / "receipt.json"

    verifier.write_receipt_atomic(destination, receipt)

    assert json.loads(destination.read_text(encoding="utf-8")) == receipt
    outside = tmp_path / "outside.json"
    outside.write_text('{"sentinel": true}\n', encoding="utf-8")
    destination.unlink()
    destination.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        verifier.write_receipt_atomic(destination, receipt)
    assert outside.read_text(encoding="utf-8") == '{"sentinel": true}\n'


def test_failed_environment_rerun_removes_a_stale_verified_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    verifier = _verifier()
    output = tmp_path / "environment-receipt.json"
    output.write_text('{"status":"verified"}\n', encoding="utf-8")

    def fail_before_verification(_lock: Path):
        raise ValueError("simulated verification failure")

    monkeypatch.setattr(verifier, "load_lock", fail_before_verification)

    assert (
        verifier.main(
            [
                "--lock",
                str(tmp_path / "release.lock.txt"),
                "--content-profile",
                str(tmp_path / "content.json"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "simulated verification failure" in capsys.readouterr().err
    assert not output.exists()


def test_installed_structure_mode_is_truthful_and_does_not_require_a_content_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "requirements-ci-py312-linux.lock.txt"
    lock_path.write_text(
        _lock_text(
            "demo==1.0 --hash=sha256:" + "1" * 64,
            python="3.12.13",
            platform="linux-amd64",
        ),
        encoding="utf-8",
    )
    output = tmp_path / "receipt.json"
    verifier = _verifier()
    monkeypatch.setattr(verifier, "runtime_python_version", lambda: "3.12.13")
    monkeypatch.setattr(verifier, "runtime_platform", lambda: "linux-amd64")
    monkeypatch.setattr(verifier, "installed_distributions", lambda: {"demo": "1.0"})
    monkeypatch.setattr(
        verifier,
        "verify_distribution_contents",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("installed-structure verification must not claim distribution-byte verification")
        ),
    )

    assert (
        verifier.main(
            [
                "--lock",
                str(lock_path),
                "--installed-structure-only",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["verification_scope"] == "runtime-and-installed-distribution-versions"
    assert "content_profile" not in receipt
    assert receipt["distributions"] == {"demo": "1.0"}


V2_SCHEMA = "kbound-python-environment-content-v2"
LAUNCHERS = {
    "demo": {
        "bin/demo": {"group": "console_scripts", "name": "demo", "target": "demo:main"},
    }
}


@pytest.fixture
def portable_tmp_path() -> Iterator[Path]:
    # Supported POSIX macOS/Linux hosts have a short physical /tmp. Avoid pytest
    # node-name nesting and caller TMPDIR/--basetemp lengths while retaining the
    # production 127-byte launcher limit. TemporaryDirectory creates a unique
    # 0700 directory and cleans up only its own tiny synthetic payloads. These
    # fake distributions are not an installed or fully verified base runtime.
    with tempfile.TemporaryDirectory(prefix="kbv2-", dir=Path("/tmp").resolve(strict=True)) as directory:
        yield Path(directory)


def _portable_fixture(prefix: Path):
    site = prefix / "lib/python3.12/site-packages"
    distribution, module, record = _fake_distribution(site)
    bin_dir = prefix / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(Path(sys.executable).resolve())
    launcher = bin_dir / "demo"
    launcher.write_bytes(b"#!" + os.fsencode(prefix) + b"/bin/python\nfrom demo import main\nmain()\n")
    entry_points = record.parent / "entry_points.txt"
    entry_points.write_bytes(b"[console_scripts]\ndemo = demo:main\n")
    _rewrite_record(record, site, [module, record.parent / "METADATA", record.parent / "WHEEL", entry_points, launcher])
    return distribution, module, record, launcher


def _rewrite_record(record: Path, site: Path, paths: list[Path]) -> None:
    rows = []
    for path in paths:
        payload = path.read_bytes()
        rows.append(f"{os.path.relpath(path, site)},sha256={_record_hash(payload)},{len(payload)}\n")
    rows.append(f"{os.path.relpath(record, site)},,\n")
    record.write_bytes("".join(rows).encode())


def _portable_profile(verifier, distribution, prefix: Path, manifest=None):
    evidence = verifier.verify_distribution_contents_v2(
        [distribution],
        expected={"demo": "1.0"},
        prefix=prefix,
        launcher_manifest=LAUNCHERS if manifest is None else manifest,
    )
    return verifier.build_content_profile_v2(
        lock_sha256="1" * 64,
        python_version="3.12.13",
        platform="macos-arm64",
        evidence=evidence,
        launcher_manifest=LAUNCHERS if manifest is None else manifest,
    )


def test_v2_relocation_preserves_identity_while_v1_changes(portable_tmp_path: Path) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    prefixes = [tmp_path / "a", tmp_path / "bbbb"]
    fixtures = [_portable_fixture(prefix) for prefix in prefixes]
    assert hasattr(verifier, "verify_distribution_contents_v2"), "portable v2 verification is missing"
    profiles = [_portable_profile(verifier, fixture[0], prefix) for fixture, prefix in zip(fixtures, prefixes)]
    assert profiles[0] == profiles[1]
    assert profiles[0]["schema"] == V2_SCHEMA
    assert profiles[0]["distributions"]["demo"]["file_count"] == 6
    raw = [
        verifier.verify_distribution_contents([fixture[0]], expected={"demo": "1.0"}, prefix=prefix)
        for fixture, prefix in zip(fixtures, prefixes)
    ]
    assert raw[0] != raw[1]


@pytest.mark.parametrize("target", ["module", "launcher", "metadata", "native"])
@pytest.mark.parametrize("update_record", [False, True])
def test_v2_tampering_is_raw_rejected_or_sealed_rejected(
    portable_tmp_path: Path, target: str, update_record: bool
) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    distribution, module, record, launcher = _portable_fixture(tmp_path / "env")
    native = launcher.parent / "native"
    native.write_bytes(b"\xcf\xfa\xed\xfeNATIVE\n")
    paths = [
        module,
        record.parent / "METADATA",
        record.parent / "WHEEL",
        record.parent / "entry_points.txt",
        launcher,
        native,
    ]
    _rewrite_record(record, module.parent, paths)
    profile = _portable_profile(verifier, distribution, launcher.parent.parent)
    sealed = tmp_path / "sealed.json"
    sealed.write_text(json.dumps(profile))
    affected = {"module": module, "launcher": launcher, "metadata": record.parent / "WHEEL", "native": native}[target]
    affected.write_bytes(affected.read_bytes() + b"changed\n")
    if update_record:
        _rewrite_record(record, module.parent, paths)
        computed = _portable_profile(verifier, distribution, launcher.parent.parent)
        with pytest.raises(ValueError, match="sealed profile"):
            verifier.verify_content_profile(sealed, computed)
    else:
        with pytest.raises(ValueError, match="RECORD (size|hash) mismatch"):
            _portable_profile(verifier, distribution, launcher.parent.parent)


@pytest.mark.parametrize(
    "first",
    [
        b"#!/usr/bin/env python\n",
        b"#!/other/bin/python\n",
        b"{prefix}3\n",
        b"{prefix} -I\n",
        b"{prefix}\r\n",
        b"{prefix}",
        b"#!/bin/sh\n",
        b"{prefix}3.12\n",
        b"{prefix}suffix\n",
    ],
)
def test_v2_rejects_unknown_launchers_even_with_consistent_record(portable_tmp_path: Path, first: bytes) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    distribution, module, record, launcher = _portable_fixture(tmp_path / "env")
    first = first.replace(b"{prefix}", b"#!" + os.fsencode(launcher.parent.parent) + b"/bin/python")
    launcher.write_bytes(first + b"main()\n")
    paths = [module, record.parent / "METADATA", record.parent / "WHEEL", record.parent / "entry_points.txt", launcher]
    _rewrite_record(record, module.parent, paths)
    with pytest.raises(ValueError, match="launcher"):
        _portable_profile(verifier, distribution, launcher.parent.parent)


@pytest.mark.parametrize(
    "mutation",
    [
        "columns",
        "duplicate",
        "alias",
        "absolute",
        "backslash",
        "size",
        "hash",
        "foreign-record",
        "crlf",
        "quote",
        "traversal",
    ],
)
def test_v2_strict_record_grammar_and_paths(portable_tmp_path: Path, mutation: str) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    distribution, module, record, launcher = _portable_fixture(tmp_path / "env")
    raw = record.read_text()
    replacements = {
        "columns": raw.replace("demo.py,", "demo.py,extra,", 1),
        "duplicate": raw + raw.splitlines(keepends=True)[0],
        "alias": raw.replace("demo.py,", "./demo.py,", 1),
        "absolute": raw.replace("demo.py,", str(module) + ",", 1),
        "backslash": raw.replace("demo.py,", "x\\demo.py,", 1),
        "size": raw.replace(",10\n", ",010\n", 1),
        "hash": raw.replace("sha256=", "sha512=", 1),
        "foreign-record": raw + "foreign-1.0.dist-info/RECORD,,\n",
        "crlf": raw.replace("\n", "\r\n"),
        "quote": raw.replace("demo.py,", '"demo.py",', 1),
        "traversal": raw.replace("../../../bin/demo,", "../../../bin/../bin/demo,", 1),
    }
    assert replacements[mutation] != raw
    record.write_bytes(replacements[mutation].encode())
    with pytest.raises(ValueError):
        _portable_profile(verifier, distribution, launcher.parent.parent)


@pytest.mark.parametrize(
    "declarations",
    [
        b"[console_scripts]\ndemo = wrong:main\n",
        b"[console_scripts]\ndemo = demo:main\ndemo = demo:main\n",
        b"[console_scripts]\ndemo = demo:main\nextra = demo:main\n",
        b"",
        b"[gui_scripts]\ndemo = demo:main\n",
    ],
)
def test_v2_manifest_is_external_and_metadata_must_agree(portable_tmp_path: Path, declarations: bytes) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    distribution, module, record, launcher = _portable_fixture(tmp_path / "env")
    ep = record.parent / "entry_points.txt"
    ep.write_bytes(declarations)
    _rewrite_record(record, module.parent, [module, record.parent / "METADATA", record.parent / "WHEEL", ep, launcher])
    with pytest.raises(ValueError, match="entry.point|manifest"):
        _portable_profile(verifier, distribution, launcher.parent.parent)


def _bind_portable_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    verifier = _verifier()
    prefix = tmp_path / "env"
    distribution, module, record, launcher = _portable_fixture(prefix)
    lock = tmp_path / "release.lock"
    lock.write_text(_lock_text("demo==1.0 --hash=sha256:" + "1" * 64))
    monkeypatch.setattr(verifier.sys, "prefix", str(prefix))
    monkeypatch.setattr(verifier, "runtime_python_version", lambda: "3.12.13")
    monkeypatch.setattr(verifier, "runtime_platform", lambda: "macos-arm64")
    monkeypatch.setattr(verifier.importlib.metadata, "distributions", lambda: [distribution])
    profile = _portable_profile(verifier, distribution, prefix)
    profile["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    # Rebuild the aggregate after changing lock authority, without normalizing any file again.
    profile = verifier.build_content_profile_v2(
        lock_sha256=profile["lock_sha256"],
        python_version="3.12.13",
        platform="macos-arm64",
        evidence=profile["distributions"],
        launcher_manifest=LAUNCHERS,
    )
    sealed = tmp_path / "v2.json"
    sealed.write_text(json.dumps(profile))
    return verifier, lock, sealed, profile, (distribution, module, record, launcher)


def test_v2_cli_and_direct_api_share_sealed_dispatch_and_scope(
    portable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = portable_tmp_path
    verifier, lock, sealed, _profile, fixture = _bind_portable_runtime(tmp_path, monkeypatch)
    binding = verifier.verify_exact_content_profile(lock, sealed)
    output = tmp_path / "receipt.json"
    args = ["--lock", str(lock), "--content-profile", str(sealed), "--output", str(output)]
    assert verifier.main(args) == 0
    receipt = json.loads(output.read_text())
    assert receipt["content_profile"] == binding
    assert binding["profile_schema"] == V2_SCHEMA
    assert binding["verification_scope"] == "locked-distribution-content-with-declared-launcher-prefix-normalization"
    fixture[1].write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="RECORD"):
        verifier.verify_exact_content_profile(lock, sealed)
    assert verifier.main(args) == 1
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ["schema", "normalizer", "extra", "bool-count", "manifest", "runtime", "lock", "duplicate"]
)
def test_v2_schema_drift_is_rejected_before_payload_hashing(
    portable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    tmp_path = portable_tmp_path
    verifier, lock, sealed, profile, _fixture = _bind_portable_runtime(tmp_path, monkeypatch)
    if mutation == "schema":
        profile["schema"] = "unknown"
    elif mutation == "normalizer":
        profile["normalization"]["identifier"] = "unknown"
    elif mutation == "extra":
        profile["extra"] = 1
    elif mutation == "bool-count":
        profile["distributions"]["demo"]["file_count"] = True
    elif mutation == "manifest":
        profile["launcher_manifest"]["demo"]["bin/demo"]["unknown"] = "value"
    elif mutation == "runtime":
        profile["runtime"]["python"] = "3.12.0"
    elif mutation == "lock":
        profile["lock_sha256"] = "0" * 64
    sealed.write_text(json.dumps(profile))
    if mutation == "duplicate":
        sealed.write_text(sealed.read_text().replace("{", '{"schema":"duplicate",', 1))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("schema must fail before content hashing")

    monkeypatch.setattr(verifier, "verify_distribution_contents_v2", forbidden)
    with pytest.raises(ValueError):
        verifier.verify_exact_content_profile(lock, sealed)
    assert verifier.main(["--lock", str(lock), "--content-profile", str(sealed)]) == 1


@pytest.mark.parametrize("destination_kind", ["existing", "symlink", "parent-symlink", "lock-alias"])
def test_profile_initialization_never_overwrites_authority(
    portable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination_kind: str
) -> None:
    tmp_path = portable_tmp_path
    verifier, lock, sealed, _profile, _fixture = _bind_portable_runtime(tmp_path, monkeypatch)
    target = sealed
    if destination_kind == "symlink":
        target = tmp_path / "alias.json"
        target.symlink_to(sealed)
    elif destination_kind == "parent-symlink":
        parent = tmp_path / "alias"
        parent.symlink_to(tmp_path, target_is_directory=True)
        target = parent / "new.json"
    elif destination_kind == "lock-alias":
        target = lock
    before = sealed.read_bytes(), lock.read_bytes()
    assert (
        verifier.main(
            [
                "--lock",
                str(lock),
                "--content-profile",
                str(target),
                "--initialize-content-profile",
                "--profile-schema",
                V2_SCHEMA,
                "--launcher-manifest",
                str(sealed),
            ]
        )
        == 1
    )
    assert (sealed.read_bytes(), lock.read_bytes()) == before
    if destination_kind == "parent-symlink":
        assert not target.exists()


def test_initialization_requires_explicit_schema_and_external_manifest(
    portable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = portable_tmp_path
    verifier, lock, _sealed, _profile, _fixture = _bind_portable_runtime(tmp_path, monkeypatch)
    target = tmp_path / "new.json"
    base = ["--lock", str(lock), "--content-profile", str(target), "--initialize-content-profile"]
    assert verifier.main(base) == 1
    assert verifier.main([*base, "--profile-schema", V2_SCHEMA]) == 1
    assert not target.exists()


def test_v2_rejects_prefix_and_payload_symlinks(portable_tmp_path: Path) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    prefix = tmp_path / "env"
    distribution, module, _record, _launcher = _portable_fixture(prefix)
    alias = tmp_path / "alias"
    alias.symlink_to(prefix, target_is_directory=True)
    with pytest.raises(ValueError, match="prefix"):
        _portable_profile(verifier, distribution, alias)
    actual = prefix / "actual.py"
    module.rename(actual)
    module.symlink_to(actual)
    with pytest.raises(ValueError, match="symlink"):
        _portable_profile(verifier, distribution, prefix)


def test_v2_rejects_cross_distribution_file_ownership(portable_tmp_path: Path) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    distribution, module, record, launcher = _portable_fixture(tmp_path / "env")
    other = module.parent / "other-1.0.dist-info"
    other.mkdir()
    metadata = other / "METADATA"
    metadata.write_bytes(b"Metadata-Version: 2.1\nName: other\nVersion: 1.0\n")
    _rewrite_record(other / "RECORD", module.parent, [metadata, module])
    distributions = list(importlib.metadata.distributions(path=[str(module.parent)]))
    with pytest.raises(ValueError, match="ownership"):
        verifier.verify_distribution_contents_v2(
            distributions,
            expected={"demo": "1.0", "other": "1.0"},
            prefix=launcher.parent.parent,
            launcher_manifest=LAUNCHERS,
        )


def test_v2_requires_entry_point_metadata_in_record_even_without_manifest(portable_tmp_path: Path) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    distribution, module, record, launcher = _portable_fixture(tmp_path / "env")
    _rewrite_record(record, module.parent, [module, record.parent / "METADATA", record.parent / "WHEEL"])
    with pytest.raises(ValueError, match="entry.point"):
        _portable_profile(verifier, distribution, launcher.parent.parent, manifest={})


def test_v2_initializer_authenticates_reference_before_new_profile(
    portable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = portable_tmp_path
    verifier, lock, sealed, _profile, fixture = _bind_portable_runtime(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(LAUNCHERS))
    new_profile = tmp_path / "new.json"
    args = [
        "--lock",
        str(lock),
        "--content-profile",
        str(new_profile),
        "--initialize-content-profile",
        "--profile-schema",
        V2_SCHEMA,
        "--launcher-manifest",
        str(manifest_path),
    ]
    assert verifier.main(args) == 1
    assert not new_profile.exists()
    args += ["--reference-content-profile", str(sealed)]
    assert verifier.main(args) == 0
    assert json.loads(new_profile.read_text()) == json.loads(sealed.read_text())
    new_profile.unlink()
    distribution, module, record, launcher = fixture
    module.write_bytes(b"EVIL = 1\n")
    _rewrite_record(
        record,
        module.parent,
        [module, record.parent / "METADATA", record.parent / "WHEEL", record.parent / "entry_points.txt", launcher],
    )
    assert verifier.main(args) == 1
    assert not new_profile.exists()


@pytest.mark.parametrize("mutation", ["row-order", "extra-native", "embedded-prefix"])
def test_v2_preserves_all_non_normalized_bytes(portable_tmp_path: Path, mutation: str) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    prefix = tmp_path / "env"
    distribution, module, record, launcher = _portable_fixture(prefix)
    baseline = _portable_profile(verifier, distribution, prefix)
    sealed = tmp_path / "seal.json"
    sealed.write_text(json.dumps(baseline))
    paths = [module, record.parent / "METADATA", record.parent / "WHEEL", record.parent / "entry_points.txt", launcher]
    if mutation == "row-order":
        record.write_bytes(b"".join(reversed(record.read_bytes().splitlines(keepends=True))))
    elif mutation == "extra-native":
        extra = prefix / "bin/extra"
        extra.write_bytes(b"\xcf\xfa\xed\xfeEXTRA")
        _rewrite_record(record, module.parent, [*paths, extra])
    else:
        launcher.write_bytes(launcher.read_bytes() + b"# " + os.fsencode(prefix) + b"/bin/python\n")
        _rewrite_record(record, module.parent, paths)
    computed = _portable_profile(verifier, distribution, prefix)
    with pytest.raises(ValueError, match="sealed profile"):
        verifier.verify_content_profile(sealed, computed)


def test_cli_output_cannot_delete_profile_or_lock(portable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = portable_tmp_path
    verifier, lock, sealed, _profile, _fixture = _bind_portable_runtime(tmp_path, monkeypatch)
    for output in (sealed, lock):
        before = output.read_bytes()
        assert verifier.main(["--lock", str(lock), "--content-profile", str(sealed), "--output", str(output)]) == 1
        assert output.read_bytes() == before


def test_v2_initializer_can_convert_authenticated_v1(portable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = portable_tmp_path
    verifier, lock, sealed, _profile, fixture = _bind_portable_runtime(tmp_path, monkeypatch)
    distribution, _module, _record, launcher = fixture
    evidence = verifier.verify_distribution_contents(
        [distribution], expected={"demo": "1.0"}, prefix=launcher.parent.parent
    )
    v1 = verifier.build_content_profile(
        lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        python_version="3.12.13",
        platform="macos-arm64",
        evidence=evidence,
    )
    reference = tmp_path / "v1.json"
    reference.write_text(json.dumps(v1))
    before = reference.read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(LAUNCHERS))
    target = tmp_path / "new-v2.json"
    args = [
        "--lock",
        str(lock),
        "--content-profile",
        str(target),
        "--initialize-content-profile",
        "--profile-schema",
        V2_SCHEMA,
        "--launcher-manifest",
        str(manifest),
        "--reference-content-profile",
        str(reference),
    ]
    assert verifier.main(args) == 0
    assert json.loads(target.read_text()) == json.loads(sealed.read_text())
    assert reference.read_bytes() == before


@pytest.mark.parametrize("fault", ["missing", "aggregate", "lock", "runtime"])
def test_v2_initialization_rejects_bad_reference(
    portable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    tmp_path = portable_tmp_path
    verifier, lock, sealed, profile, _fixture = _bind_portable_runtime(tmp_path, monkeypatch)
    if fault == "missing":
        sealed.unlink()
    else:
        if fault == "aggregate":
            profile["aggregate_sha256"] = "0" * 64
        elif fault == "lock":
            profile["lock_sha256"] = "0" * 64
        else:
            profile["runtime"]["python"] = "3.12.0"
        sealed.write_text(json.dumps(profile))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(LAUNCHERS))
    target = tmp_path / "new.json"
    assert (
        verifier.main(
            [
                "--lock",
                str(lock),
                "--content-profile",
                str(target),
                "--initialize-content-profile",
                "--profile-schema",
                V2_SCHEMA,
                "--launcher-manifest",
                str(manifest),
                "--reference-content-profile",
                str(sealed),
            ]
        )
        == 1
    )
    assert not target.exists()


def test_v2_accepts_empty_gui_section_but_keeps_its_bytes_sealed(portable_tmp_path: Path) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    prefix = tmp_path / "env"
    distribution, module, record, launcher = _portable_fixture(prefix)
    ep = record.parent / "entry_points.txt"
    before = _portable_profile(verifier, distribution, prefix)
    ep.write_bytes(ep.read_bytes() + b"\n[gui_scripts]\n")
    _rewrite_record(record, module.parent, [module, record.parent / "METADATA", record.parent / "WHEEL", ep, launcher])
    after = _portable_profile(verifier, distribution, prefix)
    assert before != after


@pytest.mark.parametrize("fault", ["missing", "duplicate", "version", "interpreter", "missing-launcher", "owner"])
def test_v2_rejects_identity_and_inventory_disagreement(portable_tmp_path: Path, fault: str) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    prefix = tmp_path / "env"
    distribution, _module, _record, launcher = _portable_fixture(prefix)
    distributions = [distribution]
    expected = {"demo": "1.0"}
    manifest = json.loads(json.dumps(LAUNCHERS))
    if fault == "missing":
        distributions = []
    elif fault == "duplicate":
        distributions *= 2
    elif fault == "version":
        expected["demo"] = "2.0"
    elif fault == "interpreter":
        interpreter = prefix / "bin/python"
        interpreter.unlink()
        interpreter.write_bytes(b"unrelated interpreter")
    elif fault == "missing-launcher":
        launcher.unlink()
    else:
        manifest["other"] = manifest.pop("demo")
    with pytest.raises(ValueError):
        verifier.verify_distribution_contents_v2(
            distributions, expected=expected, prefix=prefix, launcher_manifest=manifest
        )


def test_v2_manifest_cannot_grant_two_owners_one_launcher(portable_tmp_path: Path) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    distribution, _module, _record, launcher = _portable_fixture(tmp_path / "env")
    manifest = {"demo": LAUNCHERS["demo"], "other": LAUNCHERS["demo"]}
    with pytest.raises(ValueError, match="ownership"):
        verifier.verify_distribution_contents_v2(
            [distribution],
            expected={"demo": "1.0", "other": "1.0"},
            prefix=launcher.parent.parent,
            launcher_manifest=manifest,
        )


@pytest.mark.parametrize(("owner", "manpage"), [("fonttools", "ttx.1"), ("bandit", "bandit.1"), ("sympy", "isympy.1")])
def test_v2_reviewed_manpages_are_ordinary_sealed_payloads(portable_tmp_path: Path, owner: str, manpage: str) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    prefix = tmp_path / "env"
    _distribution, module, record, launcher = _portable_fixture(prefix)
    metadata_dir = record.parent.with_name(owner + "-1.0.dist-info")
    record.parent.rename(metadata_dir)
    record = metadata_dir / "RECORD"
    (metadata_dir / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {owner}\nVersion: 1.0\n")
    page = prefix / "share/man/man1" / manpage
    page.parent.mkdir(parents=True)
    page.write_bytes(b"manual page\n")
    paths = [
        module,
        metadata_dir / "METADATA",
        metadata_dir / "WHEEL",
        metadata_dir / "entry_points.txt",
        launcher,
        page,
    ]
    _rewrite_record(record, module.parent, paths)
    distribution = list(importlib.metadata.distributions(path=[str(module.parent)]))[0]

    def evidence():
        return verifier.verify_distribution_contents_v2(
            [distribution], expected={owner: "1.0"}, prefix=prefix, launcher_manifest={owner: LAUNCHERS["demo"]}
        )

    before = evidence()
    page.write_bytes(b"modified manual page\n")
    _rewrite_record(record, module.parent, paths)
    assert evidence() != before
    record.write_text(record.read_text().replace("../../../share/man/man1/", "../../../share/man/../man/man1/"))
    with pytest.raises(ValueError, match="alias"):
        evidence()


def test_v2_vendored_record_is_hashed_payload_never_exempt_self_record(portable_tmp_path: Path) -> None:
    tmp_path = portable_tmp_path
    verifier = _verifier()
    prefix = tmp_path / "env"
    distribution, module, record, launcher = _portable_fixture(prefix)
    vendored = module.parent / "demo_vendor/other-1.0.dist-info/RECORD"
    vendored.parent.mkdir(parents=True)
    vendored.write_bytes(b"vendored bytes\n")
    paths = [
        module,
        record.parent / "METADATA",
        record.parent / "WHEEL",
        record.parent / "entry_points.txt",
        launcher,
        vendored,
    ]
    _rewrite_record(record, module.parent, paths)
    before = _portable_profile(verifier, distribution, prefix)
    vendored.write_bytes(b"tampered bytes\n")
    with pytest.raises(ValueError, match="RECORD hash mismatch"):
        _portable_profile(verifier, distribution, prefix)
    _rewrite_record(record, module.parent, paths)
    assert _portable_profile(verifier, distribution, prefix) != before
    raw = record.read_text()
    row = next(line for line in raw.splitlines() if line.startswith("demo_vendor/"))
    record.write_text(raw.replace(row, "demo_vendor/other-1.0.dist-info/RECORD,,"))
    with pytest.raises(ValueError, match="RECORD hash"):
        _portable_profile(verifier, distribution, prefix)


@pytest.mark.parametrize("failure", ["schema-option", "existing-initialization-destination"])
def test_rejected_v2_options_remove_previous_verified_receipt(
    portable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    tmp_path = portable_tmp_path
    verifier, lock, sealed, _profile, fixture = _bind_portable_runtime(tmp_path, monkeypatch)
    output = tmp_path / "receipt.json"
    args = ["--lock", str(lock), "--content-profile", str(sealed), "--output", str(output)]
    assert verifier.main(args) == 0
    assert json.loads(output.read_text())["status"] == "verified"
    authority_before = lock.read_bytes(), sealed.read_bytes()
    fixture[1].write_bytes(b"INVALIDATED SYNTHETIC PACKAGE\n")
    invalid_options = ["--profile-schema", V2_SCHEMA]
    if failure == "existing-initialization-destination":
        invalid_options.append("--initialize-content-profile")
    assert verifier.main([*args, *invalid_options]) == 1
    assert not output.exists(), "failed rerun must remove its previous verified receipt"
    assert (lock.read_bytes(), sealed.read_bytes()) == authority_before


@pytest.mark.parametrize("shebang_length", [127, 128])
def test_v2_launcher_prefix_length_boundary(portable_tmp_path: Path, shebang_length: int) -> None:
    tmp_path = portable_tmp_path
    name_length = shebang_length - len(b"#!" + os.fsencode(tmp_path) + b"/" + b"/bin/python\n")
    assert name_length > 0, "portable_tmp_path must leave room for the exact launcher boundary"
    prefix = tmp_path / ("x" * name_length)
    distribution, _module, _record, launcher = _portable_fixture(prefix)
    assert len(launcher.read_bytes().splitlines(keepends=True)[0]) == shebang_length
    verifier = _verifier()
    if shebang_length == 127:
        profile = _portable_profile(verifier, distribution, prefix)
        assert profile["distributions"]["demo"]["file_count"] == 6
    else:
        with pytest.raises(ValueError, match="compatible short prefix"):
            _portable_profile(verifier, distribution, prefix)
