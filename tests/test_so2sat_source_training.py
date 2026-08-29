"""Focused tests for the source-only So2Sat training stack."""

# ruff: noqa: E402, I001 -- local modules require the optional Torch skip first.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from experiments.kbound.so2sat.integrity import (
    IntegrityError,
    file_sha256,
    stable_sha256,
    strict_json_load,
)
from experiments.kbound.so2sat.model import (
    CANONICAL_MODEL_SEEDS,
    NUM_CLASSES,
    assert_model_contract,
    build_so2sat_resnet18,
    tensor_state_sha256,
)
from experiments.kbound.so2sat.source_data import (
    ArraySourceContainer,
    H5SourceContainer,
    So2SatSourceDataset,
    build_source_role_inventory,
    fit_band_normalizer,
    load_sealed_band_normalizer,
    seal_band_normalizer,
    synthetic_source_inventory,
)
from experiments.kbound.so2sat.train_source import (
    TrainingConfig,
    build_synthetic_smoke_bundle,
    run_synthetic_smoke,
    train_one_seed,
    verify_complete_source_result,
)


H64 = {
    "population": "a" * 64,
    "manifest": "b" * 64,
    "geo": "c" * 64,
}


@dataclass(frozen=True)
class _GeoRecord:
    row_index: int
    sample_role: str
    official_split: str = "training"
    sample_id: str = "training:000000"
    city_id: str = "sourcecity"
    spatial_block_id: str = "32632:0:0"


class _OnePassGeoIndex:
    def __init__(self, records: list[_GeoRecord]) -> None:
        self.records = records
        self.population_identity_sha256 = H64["population"]
        self.iterations = 0

    def iter_records(self) -> Any:
        self.iterations += 1
        return iter(self.records)

    def record(self, _row_index: int) -> _GeoRecord:
        raise AssertionError("source inventory should use the one-open iter_records path")


class _LoggingArraySource(ArraySourceContainer):
    def __init__(self, pixels: np.ndarray, labels: np.ndarray) -> None:
        super().__init__(pixels, labels, identity_tag="logging-source")
        self.pixel_reads: list[tuple[int, ...]] = []
        self.labeled_reads: list[int] = []

    def read_pixels(self, row_indices: Any) -> np.ndarray:
        self.pixel_reads.append(tuple(row_indices))
        return super().read_pixels(row_indices)

    def read_labeled(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        self.labeled_reads.append(row_index)
        return super().read_labeled(row_index)


def _one_hot_labels(count: int) -> np.ndarray:
    labels = np.zeros((count, NUM_CLASSES), dtype=np.float32)
    labels[np.arange(count), np.arange(count) % NUM_CLASSES] = 1.0
    return labels


def _rewrite_receipted_json(path: Path, document: dict[str, Any]) -> None:
    """Rewrite a temporary test pair while preserving a valid generic receipt."""

    receipt_path = path.with_name(path.name + ".receipt.json")
    payload = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    path.chmod(0o644)
    path.write_text(payload, encoding="ascii")
    receipt = strict_json_load(receipt_path)
    receipt["artifact_bytes"] = path.stat().st_size
    receipt["artifact_sha256"] = file_sha256(path)
    receipt["canonical_document_sha256"] = stable_sha256(document)
    receipt_path.chmod(0o644)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    path.chmod(0o444)
    receipt_path.chmod(0o444)


def test_native_resnet18_is_10_band_32px_full_network_and_seed_independent() -> None:
    hashes = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = build_so2sat_resnet18()
        assert_model_contract(model)
        assert model.conv1.in_channels == 10
        assert model.conv1.kernel_size == (3, 3)
        assert model.conv1.stride == (1, 1)
        assert isinstance(model.maxpool, nn.Identity)
        assert model.fc.out_features == 17
        assert all(parameter.requires_grad for parameter in model.parameters())
        with torch.inference_mode():
            assert model(torch.zeros(2, 10, 32, 32)).shape == (2, 17)
        hashes.append(tensor_state_sha256(model.state_dict()))
    assert hashes[0] != hashes[1]


def test_inventory_uses_sealed_sample_roles_and_never_city_role_names() -> None:
    roles = [
        "source_train",
        "source_monitor",
        "gate_fit_probe",
        "gate_fit_evaluation",
        "gate_cal_probe",
        "gate_cal_evaluation",
    ]
    records = [
        _GeoRecord(
            row_index=index,
            sample_role=role,
            sample_id=f"training:{index:06d}",
            spatial_block_id=f"32632:{index}:0",
        )
        for index, role in enumerate(roles)
    ]
    manifest = {
        "population_identity_sha256": H64["population"],
        "manifest_sha256": H64["manifest"],
        "splits": {
            "training": {
                "observed_samples": len(records),
                "sample_role_counts": dict.fromkeys(roles, 1),
                "geo_artifact": {"sha256": H64["geo"]},
            }
        },
    }
    index = _OnePassGeoIndex(records)
    inventory = build_source_role_inventory(index, manifest)
    assert index.iterations == 1
    assert inventory.source_train_indices == (0,)
    assert inventory.source_monitor_indices == (1,)
    assert set(inventory.identity()).isdisjoint({"source_fit_small", "source_fit_core"})

    bad = records.copy()
    bad[2] = _GeoRecord(row_index=2, sample_role="target_probe")
    with pytest.raises(IntegrityError, match="unknown role"):
        build_source_role_inventory(_OnePassGeoIndex(bad), manifest)


def test_normalizer_and_label_reads_are_strictly_role_gated(tmp_path: Path) -> None:
    population_n = 4
    y = _one_hot_labels(population_n)
    pixels = np.empty((population_n, 32, 32, 10), dtype=np.float32)
    grid = np.arange(32 * 32, dtype=np.float32).reshape(32, 32, 1) / 1024.0
    pixels[0] = grid + np.arange(10, dtype=np.float32)
    pixels[1] = 2.0 * grid + np.arange(10, dtype=np.float32) + 1.0
    pixels[2] = 1_000_000.0  # source_monitor must not affect fitted moments.
    pixels[3] = -1_000_000.0  # an ungranted development row is never read.
    container = _LoggingArraySource(pixels, y)
    inventory = synthetic_source_inventory((0, 1), (2,), population_n=population_n)
    normalizer = fit_band_normalizer(container, inventory, chunk_rows=1)
    expected = pixels[[0, 1]].reshape(-1, 10)
    np.testing.assert_allclose(normalizer.mean, expected.mean(axis=0), rtol=0, atol=1e-7)
    np.testing.assert_allclose(normalizer.std, expected.std(axis=0), rtol=0, atol=1e-7)
    assert container.pixel_reads == [(0,), (1,)]

    seal_path = tmp_path / "normalizer.json"
    seal_band_normalizer(seal_path, normalizer)
    assert load_sealed_band_normalizer(seal_path) == normalizer

    train = So2SatSourceDataset(
        container,
        inventory.source_train_rows,
        normalizer,
        expected_role="source_train",
        augmentation_seed=0,
    )
    monitor = So2SatSourceDataset(
        container,
        inventory.source_monitor_rows,
        normalizer,
        expected_role="source_monitor",
    )
    assert int(train[0][1]) == 0
    assert int(monitor[0][1]) == 2
    assert container.labeled_reads == [0, 2]
    with pytest.raises(IntegrityError, match="exclusively"):
        So2SatSourceDataset(
            container,
            inventory.source_monitor_rows,
            normalizer,
            expected_role="source_train",
        )


def test_source_container_cannot_be_constructed_for_a_target_split(tmp_path: Path) -> None:
    forbidden = tmp_path / "validation.h5"
    forbidden.write_bytes(b"not-opened")
    with pytest.raises(IntegrityError, match="accepts only 'training.h5'"):
        H5SourceContainer(forbidden, expected_rows=1)


def test_epoch_resume_is_exact_and_tampered_resume_receipt_fails(tmp_path: Path) -> None:
    config = TrainingConfig(
        epochs=2,
        batch_size=34,
        learning_rate=1e-3,
        weight_decay=1e-2,
        label_smoothing=0.0,
        scheduler_eta_min=1e-6,
        workers=0,
        run_mode="synthetic_smoke",
    )
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted_bundle = build_synthetic_smoke_bundle(uninterrupted_dir)
    uninterrupted = train_one_seed(
        uninterrupted_bundle,
        uninterrupted_dir,
        model_seed=0,
        config=config,
        device=torch.device("cpu"),
        resume=False,
    )
    assert uninterrupted is not None

    resumed_dir = tmp_path / "resumed"
    resumed_bundle = build_synthetic_smoke_bundle(resumed_dir)
    stopped = train_one_seed(
        resumed_bundle,
        resumed_dir,
        model_seed=0,
        config=config,
        device=torch.device("cpu"),
        resume=False,
        stop_after_epoch=1,
    )
    assert stopped is None
    resume_receipt = resumed_dir / ".so2sat_resnet18_seed0.resume.json"
    original = resume_receipt.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["completed_epochs"] = 2
    resume_receipt.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(IntegrityError, match="receipt epoch mismatch"):
        train_one_seed(
            resumed_bundle,
            resumed_dir,
            model_seed=0,
            config=config,
            device=torch.device("cpu"),
            resume=True,
        )
    resume_receipt.write_text(original, encoding="utf-8")
    resumed = train_one_seed(
        resumed_bundle,
        resumed_dir,
        model_seed=0,
        config=config,
        device=torch.device("cpu"),
        resume=True,
    )
    assert resumed is not None
    assert resumed.checkpoint_tensor_sha256 == uninterrupted.checkpoint_tensor_sha256
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert not (resumed_dir / ".so2sat_resnet18_seed0.resume.pt").exists()
    assert not resume_receipt.exists()


def test_fast_synthetic_smoke_seals_five_independent_receipts(tmp_path: Path) -> None:
    results = run_synthetic_smoke(tmp_path, device_name="cpu")
    assert tuple(row.model_seed for row in results) == CANONICAL_MODEL_SEEDS
    assert len({row.initial_tensor_sha256 for row in results}) == 5
    assert len({row.checkpoint_tensor_sha256 for row in results}) == 5

    collection = strict_json_load(tmp_path / "so2sat_source_checkpoint_collection.json")
    assert collection["status"] == "FIVE_INDEPENDENT_SOURCE_CHECKPOINTS_VERIFIED"
    assert collection["all_checkpoint_tensor_hashes_distinct"] is True
    assert collection["all_initial_tensor_hashes_distinct"] is True
    for seed in CANONICAL_MODEL_SEEDS:
        receipt = strict_json_load(tmp_path / f"so2sat_resnet18_seed{seed}.training.json")
        assert receipt["optimization_data_role"] == "source_train"
        assert receipt["selection_data_role"] == "source_monitor"
        assert receipt["data"]["other_role_label_rows_read"] == 0
        assert receipt["data"]["target_split_pixels_read"] == 0
        assert receipt["data"]["target_split_labels_read"] == 0
        assert receipt["config"]["target_data_inputs"] == []

    verified_again = run_synthetic_smoke(tmp_path, device_name="cpu", resume=True)
    assert [row.checkpoint_tensor_sha256 for row in verified_again] == [
        row.checkpoint_tensor_sha256 for row in results
    ]

    training_receipt = tmp_path / "so2sat_resnet18_seed0.training.json"
    original_training = training_receipt.read_bytes()
    training_byte_receipt = training_receipt.with_name(
        training_receipt.name + ".receipt.json"
    )
    original_training_byte_receipt = training_byte_receipt.read_bytes()

    tampered = strict_json_load(training_receipt)
    tampered["best_epoch"] = 1
    _rewrite_receipted_json(training_receipt, tampered)
    with pytest.raises(IntegrityError, match="best_epoch does not replay"):
        verify_complete_source_result(tmp_path, 0)

    training_receipt.chmod(0o644)
    training_receipt.write_bytes(original_training)
    training_byte_receipt.chmod(0o644)
    training_byte_receipt.write_bytes(original_training_byte_receipt)
    tampered = strict_json_load(training_receipt)
    tampered["history"][0]["source_monitor"]["top1_accuracy"] = 0.123
    _rewrite_receipted_json(training_receipt, tampered)
    with pytest.raises(IntegrityError, match="top-1 accuracy does not replay"):
        verify_complete_source_result(tmp_path, 0)

    training_receipt.chmod(0o644)
    training_receipt.write_bytes(original_training)
    training_byte_receipt.chmod(0o644)
    training_byte_receipt.write_bytes(original_training_byte_receipt)
    tampered = strict_json_load(training_receipt)
    tampered["epochs_completed"] = 2
    _rewrite_receipted_json(training_receipt, tampered)
    with pytest.raises(IntegrityError, match="epochs/history are incomplete"):
        verify_complete_source_result(tmp_path, 0)

    training_receipt.chmod(0o644)
    training_receipt.write_bytes(original_training)
    training_byte_receipt.chmod(0o644)
    training_byte_receipt.write_bytes(original_training_byte_receipt)
    training_receipt.chmod(0o444)
    training_byte_receipt.chmod(0o444)
    checkpoint = tmp_path / "so2sat_resnet18_seed0.pt"
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 1
    checkpoint.write_bytes(payload)
    with pytest.raises(IntegrityError, match="checkpoint file SHA-256 mismatch"):
        run_synthetic_smoke(tmp_path, device_name="cpu", resume=True)
