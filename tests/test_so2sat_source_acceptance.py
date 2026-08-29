"""Synthetic tests for the source-only So2Sat post-run acceptance chain."""

# ruff: noqa: E402 -- skip cleanly when the optional torch dependency is absent.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from experiments.kbound.so2sat import source_acceptance
from experiments.kbound.so2sat.integrity import (
    IntegrityError,
    file_sha256,
    stable_sha256,
    strict_json_load,
    write_immutable_json_with_receipt,
)
from experiments.kbound.so2sat.metadata_manifest import SCHEMA as POPULATION_MANIFEST_SCHEMA
from experiments.kbound.so2sat.model import (
    ARCHITECTURE_ID,
    CANONICAL_MODEL_SEEDS,
    tensor_state_sha256,
)
from experiments.kbound.so2sat.source_data import (
    KNOWN_TRAINING_ROLES,
    NORMALIZER_SCHEMA,
    NORMALIZER_STATUS,
    SENTINEL2_BAND_ORDER,
    SOURCE_TRAIN_ROLE,
    BandNormalizer,
    seal_band_normalizer,
)
from experiments.kbound.so2sat.train_source import (
    CHECKPOINT_SCHEMA,
    COLLECTION_SCHEMA,
    TRAINING_RECEIPT_SCHEMA,
    TrainingConfig,
)


def _rewrite_receipted_json(path: Path, document: dict[str, Any]) -> None:
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


def _normalizer(
    *,
    source_rows_sha256: str,
    source_container_identity_sha256: str,
) -> BandNormalizer:
    unsigned = {
        "schema": NORMALIZER_SCHEMA,
        "status": NORMALIZER_STATUS,
        "fit_role": SOURCE_TRAIN_ROLE,
        "excluded_roles": sorted(KNOWN_TRAINING_ROLES - {SOURCE_TRAIN_ROLE}),
        "method": "float64_parallel_welford_population_moments",
        "band_order": list(SENTINEL2_BAND_ORDER),
        "mean": [0.0] * 10,
        "std": [1.0] * 10,
        "source_train_n": 2,
        "source_train_pixel_n": 2 * 32 * 32,
        "source_rows_sha256": source_rows_sha256,
        "source_container_identity_sha256": source_container_identity_sha256,
    }
    return BandNormalizer(
        mean=tuple(unsigned["mean"]),
        std=tuple(unsigned["std"]),
        source_train_n=unsigned["source_train_n"],
        source_train_pixel_n=unsigned["source_train_pixel_n"],
        source_rows_sha256=source_rows_sha256,
        source_container_identity_sha256=source_container_identity_sha256,
        normalizer_sha256=stable_sha256(unsigned),
    )


def _build_synthetic_source_chain(tmp_path: Path) -> dict[str, Path]:
    training_data = tmp_path / "training.h5"
    training_data.write_bytes(b"synthetic-source-container-v1")
    raw_source_sha = file_sha256(training_data)
    source_identity = {
        "schema": "kbound_so2sat_source_container_identity_v1",
        "basename": "training.h5",
        "bytes": training_data.stat().st_size,
        "file_sha256": raw_source_sha,
        "sen2_shape": [20, 32, 32, 10],
        "sen2_dtype": "float32",
        "label_shape": [20, 17],
        "label_dtype": "float32",
        "accessible_official_split": "training",
        "target_split_paths": [],
    }
    source_identity_sha = stable_sha256(source_identity)

    manifest: dict[str, Any] = {
        "schema": POPULATION_MANIFEST_SCHEMA,
        "status": "LABEL_FREE_METADATA_POPULATION_VERIFIED",
        "population_identity_sha256": stable_sha256({"population": "synthetic"}),
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    manifest_path = tmp_path / "population_manifest.json"
    manifest_receipt = write_immutable_json_with_receipt(manifest_path, manifest)

    support = [0 if class_id in {0, 6} else 1 for class_id in range(17)]
    preflight = {
        "schema": source_acceptance.SOURCE_PREFLIGHT_SCHEMA,
        "status": "SOURCE_DATA_PREFLIGHT_PASSED_WITH_WARNINGS",
        "scope": {
            "official_image_split_opened": "training",
            "target_image_containers_opened": False,
            "target_outcome_arrays_opened": False,
            "target_outcome_arrays_counted": False,
            "target_outcome_arrays_hashed": False,
        },
        "quality_gate": {"ready_for_source_training": True},
        "population_manifest_identity": {
            "basename": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "file_sha256": manifest_receipt["artifact_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "population_identity_sha256": manifest["population_identity_sha256"],
            "receipt_artifact_sha256": manifest_receipt["artifact_sha256"],
        },
        "training_container_identity": {
            "basename": "training.h5",
            "bytes": training_data.stat().st_size,
            "sha256": raw_source_sha,
        },
        "datasets": {
            "sen2": {"shape": source_identity["sen2_shape"], "dtype": "float32"},
            "label": {
                "shape": source_identity["label_shape"],
                "dtype": "float32",
                "class_counts_by_sample_role": {
                    "source_monitor": {
                        str(class_id): count for class_id, count in enumerate(support)
                    }
                },
                "missing_classes_by_sample_role": {"source_monitor": [0, 6]},
            },
        },
    }
    preflight_path = tmp_path / "source_preflight.json"
    write_immutable_json_with_receipt(
        preflight_path,
        preflight,
        receipt_schema=source_acceptance.SOURCE_PREFLIGHT_RECEIPT_SCHEMA,
    )

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    source_rows_sha = stable_sha256({"source_rows": "synthetic"})
    normalizer = _normalizer(
        source_rows_sha256=source_rows_sha,
        source_container_identity_sha256=source_identity_sha,
    )
    seal_band_normalizer(
        checkpoint_dir / source_acceptance.SOURCE_NORMALIZER_BASENAME,
        normalizer,
    )

    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=1e-2,
        label_smoothing=0.0,
        scheduler_eta_min=1e-6,
        workers=0,
        run_mode="synthetic_smoke",
    )
    config_document = config.document()
    config_sha = stable_sha256(config_document)
    inventory = {
        "population_identity_sha256": manifest["population_identity_sha256"],
        "population_manifest_sha256": manifest["manifest_sha256"],
        "training_geo_sha256": stable_sha256({"training_geo": "synthetic"}),
        "training_population_n": 20,
        "source_train_n": 2,
        "source_monitor_n": 15,
        "source_rows_sha256": source_rows_sha,
    }
    data_identity = {
        "inventory": inventory,
        "source_container_identity": source_identity,
        "source_container_identity_sha256": source_identity_sha,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "label_access_roles": ["source_train", "source_monitor"],
        "optimization_role": "source_train",
        "checkpoint_selection_role": "source_monitor",
        "target_split_paths": [],
    }
    data_identity_sha = stable_sha256(data_identity)
    code_files = {"synthetic_source.py": stable_sha256({"source": "code"})}
    code_sha = stable_sha256(code_files)
    runtime = {"device_type": "cpu", "runtime": "synthetic"}
    correct = [0] * 17
    monitor_metrics = {
        "cross_entropy": 1.0,
        "top1_accuracy": 0.0,
        "macro_recall_supported_classes": 0.0,
        "supported_class_count": 15,
        "class_support": support,
        "class_correct": correct,
        "n": 15,
        "role": "source_monitor",
    }
    collection_rows = []
    for model_seed in CANONICAL_MODEL_SEEDS:
        state = {"synthetic_weight": torch.tensor([float(model_seed)], dtype=torch.float32)}
        checkpoint_tensor_sha = tensor_state_sha256(state)
        initial_tensor_sha = stable_sha256({"initial_model_seed": model_seed})
        scientific = {
            "schema": "kbound_so2sat_source_seed_identity_v1",
            "model_seed": model_seed,
            "config": config_document,
            "config_sha256": config_sha,
            "data_identity_sha256": data_identity_sha,
            "population_manifest_sha256": inventory["population_manifest_sha256"],
            "source_rows_sha256": source_rows_sha,
            "source_container_identity_sha256": source_identity_sha,
            "normalizer_sha256": normalizer.normalizer_sha256,
            "code_sha256": code_sha,
            "runtime": runtime,
            "runtime_sha256": stable_sha256(runtime),
            "target_data_inputs": [],
        }
        scientific_sha = stable_sha256(scientific)
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture_id": ARCHITECTURE_ID,
            "model_state": state,
            "model_seed": model_seed,
            "checkpoint_tensor_sha256": checkpoint_tensor_sha,
            "initial_tensor_sha256": initial_tensor_sha,
            "best_epoch": 0,
            "best_source_monitor": monitor_metrics,
            "scientific_identity_sha256": scientific_sha,
            "config_sha256": config_sha,
            "data_identity_sha256": data_identity_sha,
            "code_sha256": code_sha,
            "normalizer_sha256": normalizer.normalizer_sha256,
            "source_rows_sha256": source_rows_sha,
            "target_data_inputs": [],
        }
        checkpoint_path = checkpoint_dir / f"so2sat_resnet18_seed{model_seed}.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint_file_sha = file_sha256(checkpoint_path)
        training_receipt = {
            "schema": TRAINING_RECEIPT_SCHEMA,
            "status": "SOURCE_TRAINING_COMPLETE",
            "model_seed": model_seed,
            "checkpoint_basename": checkpoint_path.name,
            "checkpoint_file_sha256": checkpoint_file_sha,
            "checkpoint_tensor_sha256": checkpoint_tensor_sha,
            "initial_tensor_sha256": initial_tensor_sha,
            "best_epoch": 0,
            "best_source_monitor": monitor_metrics,
            "selection_data_role": "source_monitor",
            "optimization_data_role": "source_train",
            "config": config_document,
            "config_sha256": config_sha,
            "scientific_identity": scientific,
            "scientific_identity_sha256": scientific_sha,
            "data": {
                **inventory,
                "source_container_identity": source_identity,
                "source_container_identity_sha256": source_identity_sha,
                "normalizer": normalizer.document(),
                "source_train_unique_label_rows_authorized": 2,
                "source_train_label_read_passes": 1,
                "source_monitor_unique_label_rows_authorized": 15,
                "source_monitor_label_read_passes": 1,
                "other_role_label_rows_read": 0,
                "target_split_pixels_read": 0,
                "target_split_labels_read": 0,
            },
            "data_identity_sha256": data_identity_sha,
            "code_files_sha256": code_files,
            "code_sha256": code_sha,
            "epochs_completed": 1,
            "history": [
                {
                    "epoch": 0,
                    "source_train": {
                        "role": "source_train",
                        "cross_entropy_with_label_smoothing": 1.0,
                        "n": 2,
                    },
                    "source_monitor": monitor_metrics,
                    "learning_rate_after_scheduler_step": 1e-6,
                    "selected": True,
                }
            ],
            "device": "cpu",
            "wall_seconds": 1.0,
        }
        training_path = checkpoint_dir / f"so2sat_resnet18_seed{model_seed}.training.json"
        write_immutable_json_with_receipt(training_path, training_receipt)
        collection_rows.append(
            {
                "model_seed": model_seed,
                "checkpoint_basename": checkpoint_path.name,
                "training_receipt_basename": training_path.name,
                "checkpoint_file_sha256": checkpoint_file_sha,
                "checkpoint_tensor_sha256": checkpoint_tensor_sha,
                "initial_tensor_sha256": initial_tensor_sha,
                "best_epoch": 0,
                "best_source_monitor_macro_recall": 0.0,
                "best_source_monitor_accuracy": 0.0,
            }
        )

    collection = {
        "schema": COLLECTION_SCHEMA,
        "status": "FIVE_INDEPENDENT_SOURCE_CHECKPOINTS_VERIFIED",
        "model_seeds": list(CANONICAL_MODEL_SEEDS),
        "all_checkpoint_tensor_hashes_distinct": True,
        "all_initial_tensor_hashes_distinct": True,
        "config_sha256": config_sha,
        "data_identity_sha256": data_identity_sha,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "source_rows_sha256": source_rows_sha,
        "checkpoints": collection_rows,
        "target_data_inputs": [],
    }
    write_immutable_json_with_receipt(
        checkpoint_dir / source_acceptance.SOURCE_COLLECTION_BASENAME,
        collection,
    )
    return {
        "training_data": training_data,
        "manifest": manifest_path,
        "preflight": preflight_path,
        "checkpoint_dir": checkpoint_dir,
        "output": tmp_path / source_acceptance.SOURCE_POSTRUN_ACCEPTANCE_BASENAME,
    }


def test_source_postrun_acceptance_replays_full_chain_and_raw_hash_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_synthetic_source_chain(tmp_path)
    monkeypatch.setattr(source_acceptance, "validate_population_manifest", lambda _: None)
    real_file_sha256 = source_acceptance.file_sha256
    training_hash_calls: list[Path] = []

    def counted_file_sha256(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved.name == "training.h5":
            training_hash_calls.append(resolved)
        return real_file_sha256(resolved)

    monkeypatch.setattr(source_acceptance, "file_sha256", counted_file_sha256)
    result = source_acceptance.create_source_postrun_acceptance(
        population_manifest=paths["manifest"],
        training_data=paths["training_data"],
        source_preflight=paths["preflight"],
        checkpoint_dir=paths["checkpoint_dir"],
        output=paths["output"],
    )
    assert len(training_hash_calls) == 1
    document, receipt = source_acceptance.load_verified_source_postrun_acceptance(
        paths["output"]
    )
    assert document == result["document"]
    assert receipt == result["artifact_receipt"]
    assert document["source_checkpoint_selection_disclosure"][
        "source_monitor_absent_class_ids"
    ] == [0, 6]
    assert set(
        document["source_initialization_clarification"][
            "initial_tensor_sha256_by_model_seed"
        ]
    ) == {"0", "1", "2", "3", "4"}
    binding = source_acceptance.source_postrun_acceptance_binding(document, receipt)
    assert binding[source_acceptance.TARGET_SEAL_BINDING_FIELD] == receipt["artifact_sha256"]

    verified, verified_receipt = source_acceptance.verify_source_postrun_acceptance_bindings(
        paths["output"],
        population_manifest_path=paths["manifest"],
        source_preflight_path=paths["preflight"],
        training_data_path=paths["training_data"],
        checkpoint_dir=paths["checkpoint_dir"],
    )
    assert verified == document
    assert verified_receipt == receipt
    assert len(training_hash_calls) == 2

    with pytest.raises(IntegrityError, match="refusing to overwrite"):
        source_acceptance.create_source_postrun_acceptance(
            population_manifest=paths["manifest"],
            training_data=paths["training_data"],
            source_preflight=paths["preflight"],
            checkpoint_dir=paths["checkpoint_dir"],
            output=paths["output"],
        )
    assert len(training_hash_calls) == 2

    training_receipt_path = (
        paths["checkpoint_dir"] / "so2sat_resnet18_seed0.training.json"
    )
    tampered = strict_json_load(training_receipt_path)
    tampered["best_epoch"] = 1
    _rewrite_receipted_json(training_receipt_path, tampered)
    with pytest.raises(IntegrityError, match="best_epoch does not replay"):
        source_acceptance.verify_source_postrun_acceptance_bindings(
            paths["output"],
            population_manifest_path=paths["manifest"],
            source_preflight_path=paths["preflight"],
            training_data_path=paths["training_data"],
            checkpoint_dir=paths["checkpoint_dir"],
        )
    # The verifier rejects the altered receipt/checkpoint pair before another
    # expensive source-container hash pass.
    assert len(training_hash_calls) == 2
