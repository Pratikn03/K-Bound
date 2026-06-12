"""Tests for ``uais.registry`` — model cards, manifest I/O, and integrity.

These tests use only the standard library and pytest. They never import torch
and never load a real model artifact; integrity behavior is exercised against
tiny temporary files whose digests are computed at test time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from uais.registry import IntegrityError, ModelCard, ModelRegistry, sha256_file
from uais.registry.build_manifest import (
    PENDING_SHA256,
    build_manifest,
    derive_name,
    discover_artifacts,
    write_manifest,
)
from uais.registry.model_registry import DEFAULT_MANIFEST_PATH, ManifestError


# ---------------------------------------------------------------------------
# ModelCard round-trip
# ---------------------------------------------------------------------------
def test_model_card_round_trip() -> None:
    card = ModelCard(
        name="fraud",
        task="binary fraud classification",
        framework="scikit-learn (HistGradientBoosting)",
        train_data="Kaggle Credit Card Fraud",
        intended_use="research scoring only",
        metrics={"test_roc_auc": 0.892, "calibration": "not recorded in repo"},
        limitations="severe class imbalance",
        version="1.0",
        created="2026-01-01T00:00:00+00:00",
        sha256="see MANIFEST.json",
        path="models/fraud/supervised/fraud_model.pkl",
    )
    restored = ModelCard.from_dict(card.to_dict())
    assert restored == card
    # Round-trips through JSON without loss.
    assert ModelCard.from_dict(json.loads(json.dumps(card.to_dict()))) == card


def test_model_card_defaults_and_unknown_keys() -> None:
    data = {
        "name": "x",
        "task": "t",
        "framework": "f",
        "train_data": "d",
        "intended_use": "u",
        "unexpected_field": "ignored",
    }
    card = ModelCard.from_dict(data)
    assert card.metrics == {}
    assert card.version == "1.0"
    assert card.sha256 == "see MANIFEST.json"


def test_model_card_from_dict_requires_core_fields() -> None:
    with pytest.raises(KeyError):
        ModelCard.from_dict({"name": "x"})


# ---------------------------------------------------------------------------
# sha256 helper
# ---------------------------------------------------------------------------
def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"hello-registry"
    f = tmp_path / "blob.bin"
    f.write_bytes(payload)
    assert sha256_file(f) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# build_manifest helpers
# ---------------------------------------------------------------------------
def test_derive_name_strips_models_prefix_and_suffix() -> None:
    assert derive_name(Path("models/fraud/supervised/fraud_model.pkl")) == ("fraud__supervised__fraud_model")
    assert derive_name(Path("models/fusion/fusion_meta_model.pkl")) == "fusion__fusion_meta_model"


def test_discover_and_build_manifest(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "fraud" / "supervised").mkdir(parents=True)
    (models / "behavior").mkdir(parents=True)
    art1 = models / "fraud" / "supervised" / "fraud_model.pkl"
    art2 = models / "behavior" / "behavior_lof.pkl"
    note = models / "behavior" / "README.txt"  # ignored (wrong suffix)
    art1.write_bytes(b"a")
    art2.write_bytes(b"bb")
    note.write_text("not an artifact")

    discovered = discover_artifacts(models)
    assert art1.resolve() in discovered and art2.resolve() in discovered
    assert note.resolve() not in discovered

    manifest = build_manifest(tmp_path, compute_hashes=True)
    assert manifest["version"] == 1
    names = {a["name"] for a in manifest["artifacts"]}
    assert names == {"fraud__supervised__fraud_model", "behavior__behavior_lof"}
    for entry in manifest["artifacts"]:
        assert len(entry["sha256"]) == 64  # real hex digest
        assert entry["size_bytes"] >= 1
        # POSIX-style relative paths.
        assert entry["path"].startswith("models/")


def test_build_manifest_pending_mode(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.pkl").write_bytes(b"x")
    manifest = build_manifest(tmp_path, compute_hashes=False)
    assert manifest["artifacts"][0]["sha256"] == PENDING_SHA256


# ---------------------------------------------------------------------------
# Manifest read/write round-trip
# ---------------------------------------------------------------------------
def test_manifest_write_read_round_trip(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "fusion").mkdir(parents=True)
    (models / "fusion" / "fusion_meta_model.pkl").write_bytes(b"payload")

    manifest = build_manifest(tmp_path, compute_hashes=True)
    out = write_manifest(manifest, tmp_path / DEFAULT_MANIFEST_PATH)
    assert out.exists()

    reg = ModelRegistry(repo_root=tmp_path)
    entries = reg.load_manifest()  # default path resolves under repo_root
    assert "fusion__fusion_meta_model" in entries
    # The written JSON parses back to an equivalent structure.
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["artifacts"] == manifest["artifacts"]


def test_load_manifest_missing_raises(tmp_path: Path) -> None:
    reg = ModelRegistry(repo_root=tmp_path)
    with pytest.raises(ManifestError):
        reg.load_manifest("models/MANIFEST.json")


def test_load_manifest_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "MANIFEST.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "artifacts": [
                    {"name": "dup", "path": "a.pkl", "sha256": "0"},
                    {"name": "dup", "path": "b.pkl", "sha256": "1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    reg = ModelRegistry(repo_root=tmp_path)
    with pytest.raises(ManifestError):
        reg.load_manifest(path)


# ---------------------------------------------------------------------------
# Card registration
# ---------------------------------------------------------------------------
def test_register_get_and_missing() -> None:
    reg = ModelRegistry(repo_root=Path("."))
    card = ModelCard(
        name="fusion",
        task="late fusion",
        framework="scikit-learn (LogisticRegression)",
        train_data="per-domain scores",
        intended_use="research",
    )
    reg.register(card)
    assert reg.get("fusion") is card
    assert "fusion" in reg.names()
    with pytest.raises(KeyError):
        reg.get("does-not-exist")


# ---------------------------------------------------------------------------
# Integrity: success then corruption
# ---------------------------------------------------------------------------
def _manifest_for(tmp_path: Path, name: str, rel_path: str, sha256: str) -> ModelRegistry:
    manifest = {
        "version": 1,
        "generated": "test",
        "artifacts": [{"name": name, "path": rel_path, "sha256": sha256}],
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    reg = ModelRegistry(repo_root=tmp_path)
    reg.load_manifest(tmp_path / "MANIFEST.json")
    return reg


def test_load_verify_success_then_corruption(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "fraud" / "model.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"genuine-bytes")
    good_digest = sha256_file(artifact)

    reg = _manifest_for(tmp_path, "fraud", "models/fraud/model.pkl", good_digest)

    # Verified load returns the resolved path.
    resolved = reg.load("fraud", verify=True)
    assert resolved == artifact.resolve()

    # Corrupt the file on disk -> digest changes -> IntegrityError.
    artifact.write_bytes(b"tampered-bytes")
    with pytest.raises(IntegrityError) as excinfo:
        reg.load("fraud", verify=True)
    assert excinfo.value.name == "fraud"
    assert excinfo.value.expected == good_digest
    assert excinfo.value.actual == sha256_file(artifact)

    # verify=False skips the check and still returns the path.
    assert reg.load("fraud", verify=False) == artifact.resolve()


def test_load_pending_digest_raises(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "m.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"data")
    reg = _manifest_for(tmp_path, "m", "models/m.pkl", PENDING_SHA256)
    with pytest.raises(IntegrityError):
        reg.load("m", verify=True)


def test_load_missing_artifact_raises_file_not_found(tmp_path: Path) -> None:
    reg = _manifest_for(tmp_path, "ghost", "models/ghost.pkl", "deadbeef")
    with pytest.raises(FileNotFoundError):
        reg.load("ghost", verify=True)


def test_load_unknown_name_raises(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "m.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"data")
    reg = _manifest_for(tmp_path, "m", "models/m.pkl", sha256_file(artifact))
    with pytest.raises(ManifestError):
        reg.load("not-in-manifest", verify=True)


def test_env_override_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "models" / "m.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"data")
    real = sha256_file(artifact)
    # Manifest carries a WRONG digest; env override carries the right one.
    reg = _manifest_for(tmp_path, "m", "models/m.pkl", "0" * 64)
    monkeypatch.setenv("UAIS_MODEL_SHA256_M", real.upper())  # case-insensitive
    assert reg.load("m", verify=True) == artifact.resolve()


# ---------------------------------------------------------------------------
# verify_all report
# ---------------------------------------------------------------------------
def test_verify_all_reports_statuses(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    good = models / "good.pkl"
    bad = models / "bad.pkl"
    good.write_bytes(b"good")
    bad.write_bytes(b"bad")

    manifest = {
        "version": 1,
        "artifacts": [
            {"name": "good", "path": "models/good.pkl", "sha256": sha256_file(good)},
            {"name": "bad", "path": "models/bad.pkl", "sha256": "0" * 64},
            {"name": "missing", "path": "models/missing.pkl", "sha256": "abc"},
            {"name": "pending", "path": "models/good.pkl", "sha256": PENDING_SHA256},
        ],
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    reg = ModelRegistry(repo_root=tmp_path)
    reg.load_manifest(tmp_path / "MANIFEST.json")

    report = reg.verify_all()
    assert report["ok"] is False
    assert report["checked"] == 4
    assert report["artifacts"]["good"]["status"] == "ok"
    assert report["artifacts"]["bad"]["status"] == "mismatch"
    assert report["artifacts"]["missing"]["status"] == "missing"
    assert report["artifacts"]["pending"]["status"] == "pending"


def test_verify_all_requires_manifest(tmp_path: Path) -> None:
    reg = ModelRegistry(repo_root=tmp_path)
    with pytest.raises(ManifestError):
        reg.verify_all()
