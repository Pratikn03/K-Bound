from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import importlib.util
import json
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
    assert '--content-profile "$KB/release_python_environment_macos_arm64.json"' in runbook
    source_seal = (ROOT / "docs/research/kbound/scripts/build_release_source_seal.py").read_text(encoding="utf-8")
    assert "docs/research/kbound/release_python_environment_macos_arm64.json" in source_seal
    checksum_verifier = (ROOT / "docs/research/kbound/scripts/verify_release_checksums.py").read_text(encoding="utf-8")
    assert "docs/research/kbound/release_python_environment_macos_arm64.json" in checksum_verifier


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
