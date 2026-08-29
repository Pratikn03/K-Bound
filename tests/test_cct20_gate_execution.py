"""Prospective CCT-20 gate, firewall, scoring, and inference contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from experiments.kbound.cct20 import label_free_traces as trace_module
from experiments.kbound.cct20.integrity import IntegrityError, file_sha256, stable_sha256
from experiments.kbound.cct20.label_free_traces import (
    FEATURE_NAMES,
    extract_label_free_features,
    sequence_atomic_batches,
    sequence_atomic_partition,
)
from experiments.kbound.cct20.prediction_artifacts import (
    build_prediction_cell,
    build_prediction_collection,
)
from experiments.kbound.cct20.protocol_seal import (
    EXPECTED_MODEL_SEEDS,
    REQUIRED_CODE_DEPENDENCY_NAMES,
    REQUIRED_DATA_DEPENDENCY_NAMES,
    build_execution_seal,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from experiments.kbound.cct20.ridge_gate import apply_gate, fit_calibrate_ridge_gate
from experiments.kbound.cct20.run_development_gate import (
    validate_development_trace_collection,
)
from experiments.kbound.cct20.score_once import cct20_truth_loader, score_once
from experiments.kbound.cct20.target_executor import LabelFreeTargetCell
from experiments.kbound.cct20.tent_official import (
    BN_GAUSSIAN_KL_NUMERIC_CLIPPING,
    BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION,
    BN_GAUSSIAN_KL_SCHEMA,
    BN_GAUSSIAN_KL_TAYLOR_TERMS,
    BN_GAUSSIAN_KL_TAYLOR_THRESHOLD,
    OFFICIAL_TENT_COMMIT,
    OFFICIAL_TENT_FILE_SHA256,
    OFFICIAL_TENT_TREE,
    FrozenBatchNormMomentAccumulator,
    install_locked_root_bn_cpu_fallback,
    new_checkpoint_location_session,
    stable_gaussian_kl_probe_to_source,
    verify_official_tent,
)
from experiments.kbound.cct20.two_way_inference import analyze_score_document

ROOT = Path(__file__).resolve().parents[1]
TENT_REPO = ROOT / "external" / "tent_official"
H64 = {
    name: stable_sha256(name)
    for name in (
        "protocol",
        "gate",
        "manifest",
        "checkpoint",
        "backbone",
        "recipe",
        "data",
        "code",
        "shared_runtime",
    )
}


class _TinyBN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(4, 16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(value).flatten(1))


class _TinyHybridBN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.bn1(self.conv1(value)))
        return self.classifier(self.pool(self.bn2(value)).flatten(1))


def test_official_tent_binding_is_exact_and_scoped_to_checkpoint_location() -> None:
    provenance = verify_official_tent(TENT_REPO)
    assert provenance["git_commit"] == OFFICIAL_TENT_COMMIT
    binding = new_checkpoint_location_session(
        _TinyBN(),
        repo_root=TENT_REPO,
        checkpoint_tensor_sha256=H64["checkpoint"],
        location_id=7,
    )
    assert binding.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert binding.optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert binding.optimizer.param_groups[0]["weight_decay"] == 0.0
    assert binding.adapter.steps == 1
    assert binding.adapter.episodic is False
    assert binding.receipt()["reset_scope"] == f"{H64['checkpoint']}:7"
    parameter = next(iter(binding.optimizer.param_groups[0]["params"]))
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(3.0)
    assert binding.normalized_update_norm() > 0.0
    assert binding.probe_update_receipt()["normalized_tent_update_norm"] > 0.0
    binding.reset()
    assert torch.equal(parameter, original)
    assert binding.normalized_update_norm() == pytest.approx(0.0)
    online_batch = torch.randn(4, 3, 5, 5)
    with torch.no_grad():
        expected_pre_update_logits = binding.adapter.model(online_batch).detach().clone()
    observed_official_logits = binding.adapter(online_batch).detach()
    torch.testing.assert_close(observed_official_logits, expected_pre_update_logits)
    assert binding.normalized_update_norm() > 0.0
    binding.reset()

    frozen = _TinyBN().eval()
    probe_a = torch.randn(2, 3, 5, 5)
    probe_b = torch.randn(2, 3, 5, 5)
    accumulator = FrozenBatchNormMomentAccumulator(frozen)
    with torch.no_grad():
        frozen(probe_a)
        frozen(probe_b)
    moments = accumulator.finalize()
    assert moments["channel_count"] == 4
    assert moments["batchnorm_batch_source_statistic_divergence"] >= 0.0
    assert moments["schema"] == BN_GAUSSIAN_KL_SCHEMA
    assert moments["numeric_implementation"] == BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION
    assert moments["numeric_clipping"] == BN_GAUSSIAN_KL_NUMERIC_CLIPPING
    single_batch = FrozenBatchNormMomentAccumulator(frozen)
    with torch.no_grad():
        frozen(torch.cat([probe_a, probe_b], dim=0))
    combined = single_batch.finalize()
    assert moments["batchnorm_batch_source_statistic_divergence"] == pytest.approx(
        combined["batchnorm_batch_source_statistic_divergence"], rel=1e-6, abs=1e-9
    )
    with pytest.raises(IntegrityError, match="already finalized"):
        accumulator.finalize()


def test_stable_gaussian_kl_handles_exact_cancellation_case_without_clipping() -> None:
    source_mean = torch.tensor([0.0], dtype=torch.float64)
    probe_mean = torch.tensor([1.6456096582952314e-11], dtype=torch.float64)
    source_var = torch.tensor([1.7822087096186028e-19], dtype=torch.float64)
    probe_var = torch.tensor([1.4512675073115715e-19], dtype=torch.float64)
    kl, taylor_mask = stable_gaussian_kl_probe_to_source(
        probe_mean,
        probe_var,
        source_mean,
        source_var,
        eps=1.0e-5,
    )
    assert taylor_mask.tolist() == [True]
    assert kl.item() == pytest.approx(1.3540155737375239e-17, rel=1e-12, abs=1e-30)
    assert kl.item() >= 0.0

    equal, equal_mask = stable_gaussian_kl_probe_to_source(
        source_mean,
        source_var,
        source_mean,
        source_var,
        eps=1.0e-5,
    )
    assert equal_mask.tolist() == [True]
    assert equal.item() == 0.0
    with pytest.raises(IntegrityError, match="variances must be nonnegative"):
        stable_gaussian_kl_probe_to_source(
            source_mean,
            torch.tensor([-1.0], dtype=torch.float64),
            source_mean,
            source_var,
            eps=1.0e-5,
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS regression requires Apple Silicon acceleration",
)
def test_mps_bn_moments_and_tent_update_receipt_transfer_before_float64() -> None:
    device = torch.device("mps")
    frozen = _TinyBN().to(device).eval()
    accumulator = FrozenBatchNormMomentAccumulator(frozen)
    with torch.no_grad():
        frozen(torch.randn(2, 3, 5, 5, device=device))
    moments = accumulator.finalize()
    assert moments["channel_count"] == 4
    assert moments["batchnorm_batch_source_statistic_divergence"] >= 0.0

    adapted_source = _TinyHybridBN().to(device)
    backend_receipt = install_locked_root_bn_cpu_fallback(adapted_source)
    binding = new_checkpoint_location_session(
        adapted_source,
        repo_root=TENT_REPO,
        checkpoint_tensor_sha256=H64["checkpoint"],
        location_id=7,
        backend_installation_receipt=backend_receipt,
    )
    assert binding.receipt()["backend_installation"]["official_tent_parameter_devices"] == [
        "cpu",
        "mps",
    ]
    assert binding.receipt()["initial_bn_affine_l2"] > 0.0
    assert binding.normalized_update_norm() == pytest.approx(0.0)
    binding.adapter(torch.randn(2, 3, 5, 5, device=device))
    assert binding.normalized_update_norm() > 0.0


def _metadata(location: str, count: int = 30) -> list[dict]:
    return [
        {
            "image_id": f"{location}-image-{index}",
            "sequence_id": f"{location}-sequence-{index}",
            "location_id": location,
            "file_name": f"{location}/{index}.jpg",
            "frame_num": 0,
            "date_captured": f"2020-01-{1 + index // 24:02d}T{index % 24:02d}:00:00",
        }
        for index in range(count)
    ]


def test_sequence_partition_and_batches_are_deterministic_atomic_and_label_free() -> None:
    rows = _metadata("7", 40)
    first = sequence_atomic_partition(rows, probe_fraction=0.30, salt="KBOUND_CCT20_PROBE_EVAL_v1")
    second = sequence_atomic_partition(list(reversed(rows)), probe_fraction=0.30, salt="KBOUND_CCT20_PROBE_EVAL_v1")
    assert first == second
    probe_sequences = {row["sequence_id"] for row in first["roles"]["probe"]}
    evaluation_sequences = {row["sequence_id"] for row in first["roles"]["evaluation"]}
    assert probe_sequences and evaluation_sequences
    assert probe_sequences.isdisjoint(evaluation_sequences)

    sequences = []
    for sequence, length in enumerate((3, 3, 1)):
        for frame in range(length):
            sequences.append(
                {
                    "image_id": f"i-{sequence}-{frame}",
                    "sequence_id": f"s-{sequence}",
                    "location_id": "7",
                    "file_name": f"{sequence}-{frame}.jpg",
                    "frame_num": frame,
                    "date_captured": f"2020-01-0{sequence + 1}T00:00:00",
                }
            )
    batches = sequence_atomic_batches(sequences, max_images=3, order="native")
    assert [len(batch) for batch in batches] == [3, 4]
    membership = {}
    for batch_index, batch in enumerate(batches):
        for row in batch:
            membership.setdefault(row["sequence_id"], set()).add(batch_index)
    assert all(len(indices) == 1 for indices in membership.values())
    with pytest.raises(IntegrityError, match="label/outcome field"):
        sequence_atomic_partition(
            [dict(rows[0], category_id=3), *rows[1:]],
            probe_fraction=0.30,
            salt="KBOUND_CCT20_PROBE_EVAL_v1",
        )


def test_probe_fraction_hash_comparison_is_exact_at_the_rational_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = (3 * (1 << 256)) // 10

    class _Digest:
        def __init__(self, value: int) -> None:
            self._value = value

        def digest(self) -> bytes:
            return self._value.to_bytes(32, "big")

        def hexdigest(self) -> str:
            return self.digest().hex()

    def controlled_sha256(payload: bytes) -> _Digest:
        return _Digest(boundary if b"boundary" in payload else (1 << 256) - 1)

    monkeypatch.setattr(trace_module.hashlib, "sha256", controlled_sha256)
    result = sequence_atomic_partition(
        [
            dict(_metadata("7", 1)[0], sequence_id="boundary"),
            dict(
                _metadata("7", 1)[0],
                image_id="evaluation-image",
                sequence_id="evaluation",
            ),
        ],
        probe_fraction=0.30,
        salt="KBOUND_CCT20_PROBE_EVAL_v1",
    )
    assert [row["sequence_id"] for row in result["roles"]["probe"]] == ["boundary"]
    assert [row["sequence_id"] for row in result["roles"]["evaluation"]] == ["evaluation"]


def test_fixed_eleven_label_free_features_are_finite_and_deterministic() -> None:
    frozen = np.zeros((5, 16), dtype=float)
    adapted = frozen.copy()
    adapted[:, 0] = 2.0
    result = extract_label_free_features(
        frozen,
        adapted,
        normalized_tent_update_norm=0.2,
        batchnorm_batch_source_statistic_divergence=0.3,
    )
    assert tuple(result["feature_names"]) == FEATURE_NAMES
    assert len(result["features"]) == 11
    assert result["features"]["normalized_tent_update_norm"] == 0.2
    assert result["features"]["batchnorm_batch_source_statistic_divergence"] == 0.3
    assert 0.0 <= result["features"]["marginal_jensen_shannon_divergence"] <= 1.0
    with pytest.raises(IntegrityError, match="non-negative"):
        extract_label_free_features(
            frozen,
            adapted,
            normalized_tent_update_norm=-1.0,
            batchnorm_batch_source_statistic_divergence=0.3,
        )


def _features(offset: float) -> dict[str, float]:
    return {name: float(offset + index * 0.01) for index, name in enumerate(FEATURE_NAMES)}


def _gate_rows() -> tuple[list[dict], list[dict]]:
    fit = []
    for unit_index, unit in enumerate(("trans_val:125", "cis_test:33")):
        for seed in EXPECTED_MODEL_SEEDS:
            fit.append(
                {
                    "role": "development_fit",
                    "trace_id": f"fit-{unit}-{seed}",
                    "calibration_unit": unit,
                    "checkpoint_id": str(seed),
                    "checkpoint_tensor_sha256": stable_sha256({"tensor": seed}),
                    "checkpoint_file_sha256": stable_sha256({"file": seed}),
                    "shared_runtime_sha256": H64["shared_runtime"],
                    "trace_sha256": stable_sha256({"trace": "fit", "unit": unit, "seed": seed}),
                    "partition_sha256": stable_sha256({"partition": unit}),
                    "features": _features(0.1 * unit_index + 0.01 * seed),
                    "observed_benefit": -0.15 + 0.06 * seed + 0.05 * unit_index,
                }
            )
    calibration = []
    for unit_index, location in enumerate((38, 43, 51, 61, 88, 90, 108, 115, 120)):
        for seed in EXPECTED_MODEL_SEEDS:
            calibration.append(
                {
                    "role": "development_calibration",
                    "trace_id": f"cal-{unit_index}-{seed}",
                    "calibration_unit": f"cis_test:{location}",
                    "checkpoint_id": str(seed),
                    "checkpoint_tensor_sha256": stable_sha256({"tensor": seed}),
                    "checkpoint_file_sha256": stable_sha256({"file": seed}),
                    "shared_runtime_sha256": H64["shared_runtime"],
                    "trace_sha256": stable_sha256({"trace": "cal", "unit": location, "seed": seed}),
                    "partition_sha256": stable_sha256({"partition": location}),
                    "features": _features(0.02 * unit_index + 0.005 * seed),
                    "observed_benefit": -0.1 + 0.02 * unit_index + 0.02 * seed,
                }
            )
    return fit, calibration


def _gate_document() -> dict:
    fit, calibration = _gate_rows()
    return fit_calibrate_ridge_gate(fit, calibration)


def test_ridge_gate_uses_fixed_penalty_exact_rank_location_max_and_fails_closed() -> None:
    gate = _gate_document()
    assert gate["ridge"]["penalty"] == 10.0
    assert gate["calibration"]["n_independent_units"] == 9
    assert gate["calibration"]["epsilon"] == max(gate["calibration"]["residuals_sorted"])
    assert gate["support"]["primary"] == "finite_values_and_exact_feature_schema"
    result = apply_gate(gate, _features(100.0))
    assert result["support_status"] == "IN_SUPPORT"  # Mahalanobis is diagnostic only.
    assert result["decision"] in {"ADAPT", "FREEZE", "ABSTAIN"}
    assert result["mahalanobis_diagnostic"] >= 0.0
    missing = _features(0.0)
    missing.pop(FEATURE_NAMES[-1])
    failed = apply_gate(gate, missing)
    assert failed["decision"] == "ABSTAIN"
    assert failed["support_status"] == "FAIL_CLOSED"


def test_development_collection_replays_gate_and_binds_all_trace_receipts(
    tmp_path: Path,
) -> None:
    fit, calibration = _gate_rows()
    gate = fit_calibrate_ridge_gate(fit, calibration)
    rows = sorted(fit + calibration, key=lambda row: row["trace_id"])
    artifacts = []
    for index, row in enumerate(rows):
        path = str((tmp_path / f"trace-{index:02d}.json").resolve())
        artifacts.append(
            {
                "trace_id": row["trace_id"],
                "trace_sha256": row["trace_sha256"],
                "artifact_path": path,
                "artifact_receipt": {
                    "schema": "kbound_cct20_artifact_receipt_v1",
                    "artifact_path": path,
                    "artifact_bytes": 1,
                    "artifact_sha256": stable_sha256({"artifact": index}),
                    "canonical_document_sha256": stable_sha256({"document": index}),
                },
            }
        )
    audit = _checkpoint_audit()
    collection = {
        "schema": "kbound_cct20_development_trace_collection_v1",
        "status": "SEALED_BEFORE_TARGET_INFERENCE",
        "fit_trace_count": 10,
        "calibration_trace_count": 45,
        "checkpoint_audit_sha256": stable_sha256(audit),
        "shared_runtime_identity": {
            "shared_runtime_sha256": H64["shared_runtime"],
            "artifact_path": str((tmp_path / "shared-runtime.json").resolve()),
            "artifact_receipt": {
                "schema": "kbound_cct20_artifact_receipt_v1",
                "artifact_path": str((tmp_path / "shared-runtime.json").resolve()),
                "artifact_bytes": 1,
                "artifact_sha256": stable_sha256("shared-runtime-artifact"),
                "canonical_document_sha256": stable_sha256("shared-runtime-document"),
            },
        },
        "trace_sha256": sorted(row["trace_sha256"] for row in rows),
        "trace_artifacts": artifacts,
        "gate_rows": rows,
        "gate_sha256": gate["gate_sha256"],
    }
    collection["collection_sha256"] = stable_sha256(collection)
    validate_development_trace_collection(
        collection,
        gate_document=gate,
        checkpoint_audit=audit,
        verify_trace_files=False,
        verify_runtime_file=False,
    )
    tampered = json.loads(json.dumps(collection))
    tampered["gate_rows"][0]["observed_benefit"] += 0.1
    tampered["collection_sha256"] = stable_sha256(
        {key: value for key, value in tampered.items() if key != "collection_sha256"}
    )
    with pytest.raises(IntegrityError, match="replay the sealed gate"):
        validate_development_trace_collection(
            tampered,
            gate_document=gate,
            checkpoint_audit=audit,
            verify_trace_files=False,
            verify_runtime_file=False,
        )


def _checkpoint_audit(checkpoint_root: Path | None = None) -> dict:
    rows = []
    for seed in EXPECTED_MODEL_SEEDS:
        file_hash = stable_sha256({"file": seed})
        row = {
            "model_seed": seed,
            "tensor_sha256": stable_sha256({"tensor": seed}),
            "initial_tensor_sha256": stable_sha256({"initial": seed}),
            "imagenet_backbone_tensor_sha256": H64["backbone"],
            "file_sha256": file_hash,
            "config_sha256": stable_sha256({"config": seed}),
            "config_recipe_sha256": H64["recipe"],
            "data_sha256": H64["data"],
            "code_sha256": H64["code"],
        }
        if checkpoint_root is not None:
            path = checkpoint_root / f"synthetic-checkpoint-{seed}.bin"
            path.write_bytes(f"checkpoint-{seed}".encode())
            row.update(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "file_sha256": file_sha256(path),
                }
            )
        rows.append(row)
    return {
        "schema": "kbound_cct20_independent_checkpoint_audit_v1",
        "status": "PASS",
        "required_model_seeds": list(EXPECTED_MODEL_SEEDS),
        "n_checkpoints": 5,
        "all_file_hashes_distinct": True,
        "all_tensor_hashes_distinct": True,
        "all_initial_tensor_hashes_distinct": True,
        "all_config_hashes_distinct": True,
        "shared_config_recipe_sha256": H64["recipe"],
        "shared_imagenet_backbone_tensor_sha256": H64["backbone"],
        "shared_data_sha256": H64["data"],
        "shared_code_sha256": H64["code"],
        "checkpoints": rows,
    }


def _authoritative_protocol_config() -> dict:
    return yaml.safe_load((ROOT / "experiments/kbound/cct20/prospective_protocol_v1.yaml").read_text(encoding="utf-8"))


def _dependency_files(
    tmp_path: Path,
    *,
    truth_source: Path | None = None,
) -> tuple[dict[str, Path], dict[str, Path]]:
    root = tmp_path / "sealed-dependencies"
    root.mkdir(exist_ok=True)
    dataset: dict[str, Path] = {}
    for name in sorted(REQUIRED_DATA_DEPENDENCY_NAMES):
        path = truth_source if name == "target_annotations_json" and truth_source else root / f"{name}.bin"
        if not path.exists():
            path.write_bytes(f"dataset:{name}\n".encode())
        dataset[name] = path
    code: dict[str, Path] = {}
    for name in sorted(REQUIRED_CODE_DEPENDENCY_NAMES):
        path = root / f"{name}.py"
        path.write_text(f"# code dependency: {name}\n", encoding="utf-8")
        code[name] = path
    return dataset, code


def test_execution_seal_requires_distinct_checkpoints_and_receipts_detect_tamper(
    tmp_path: Path,
) -> None:
    dataset_dependencies, code_dependencies = _dependency_files(tmp_path)
    seal = build_execution_seal(
        target_location_ids=[0, 7, 28, 40, 46, 78, 100, 105, 130],
        target_manifest_sha256=H64["manifest"],
        dataset_dependencies=dataset_dependencies,
        code_dependencies=code_dependencies,
        checkpoint_audit=_checkpoint_audit(),
        gate_sha256=H64["gate"],
        protocol_config=_authoritative_protocol_config(),
    )
    destination = tmp_path / "seal.json"
    receipt = write_immutable_json_with_receipt(destination, seal)
    assert verify_artifact_receipt(destination) == receipt
    with pytest.raises(IntegrityError, match="overwrite"):
        write_immutable_json_with_receipt(destination, seal)

    authoritative_recipe = _authoritative_protocol_config()
    recipe_seal = build_execution_seal(
        target_location_ids=[0, 7, 28, 40, 46, 78, 100, 105, 130],
        target_manifest_sha256=H64["manifest"],
        dataset_dependencies=dataset_dependencies,
        code_dependencies=code_dependencies,
        checkpoint_audit=_checkpoint_audit(),
        gate_sha256=H64["gate"],
        protocol_config=authoritative_recipe,
    )
    assert recipe_seal["protocol_config"]["gate"]["features"] == list(FEATURE_NAMES)

    bad_audit = _checkpoint_audit()
    bad_audit["checkpoints"][1]["tensor_sha256"] = bad_audit["checkpoints"][0]["tensor_sha256"]
    with pytest.raises(IntegrityError, match="all_tensor_hashes_distinct"):
        build_execution_seal(
            target_location_ids=[0, 7, 28, 40, 46, 78, 100, 105, 130],
            target_manifest_sha256=H64["manifest"],
            dataset_dependencies=dataset_dependencies,
            code_dependencies=code_dependencies,
            checkpoint_audit=bad_audit,
            gate_sha256=H64["gate"],
            protocol_config=_authoritative_protocol_config(),
        )

    bad_recipe = _checkpoint_audit()
    bad_recipe["checkpoints"][4]["config_recipe_sha256"] = stable_sha256("different-recipe")
    with pytest.raises(IntegrityError, match="shared_config_recipe_sha256"):
        build_execution_seal(
            target_location_ids=[0, 7, 28, 40, 46, 78, 100, 105, 130],
            target_manifest_sha256=H64["manifest"],
            dataset_dependencies=dataset_dependencies,
            code_dependencies=code_dependencies,
            checkpoint_audit=bad_recipe,
            gate_sha256=H64["gate"],
            protocol_config=_authoritative_protocol_config(),
        )


def _binding_receipt(checkpoint_hash: str, location: str) -> dict:
    root_bn_hash = stable_sha256("synthetic-root-bn-state")
    return {
        "schema": "kbound_cct20_official_tent_binding_v1",
        "checkpoint_tensor_sha256": checkpoint_hash,
        "location_id": location,
        "reset_scope": f"{checkpoint_hash}:{location}",
        "parameter_names": ["bn.weight", "bn.bias"],
        "n_parameters": 2,
        "initial_bn_affine_l2": 1.0,
        "update_norm_formula": ("l2(after_probe-before_probe)/max(l2(before_probe),1e-12)"),
        "backend_installation": {
            "schema": "kbound_cct20_backend_installation_v1",
            "strategy": "mps_resnet50_official_tent_cpu_root_bn1_v1",
            "fallback_layer": "bn1",
            "source_module_class": "torch.nn.BatchNorm2d",
            "fallback_module_class": "KBoundCPUFallbackBatchNorm2d",
            "fallback_input_device": "mps",
            "fallback_compute_device": "cpu",
            "fallback_parameter_device": "cpu",
            "fallback_output_device": "mps",
            "num_features": 64,
            "eps": 1.0e-5,
            "momentum": 0.1,
            "affine": True,
            "preconfigure_track_running_stats": True,
            "source_bn_state_sha256": root_bn_hash,
            "installed_bn_state_sha256": root_bn_hash,
            "state_hash_equal": True,
            "official_tent_parameter_devices": ["cpu", "mps"],
            "configured_track_running_stats": False,
            "configured_running_moments_absent": True,
        },
        "provenance": {
            "git_commit": OFFICIAL_TENT_COMMIT,
            "git_tree": OFFICIAL_TENT_TREE,
            "tent_py_sha256": OFFICIAL_TENT_FILE_SHA256,
            "tracked_worktree_clean": True,
            "configure_function": "tent.configure_model",
            "parameter_function": "tent.collect_params",
            "adapter_class": "tent.Tent",
            "reset_scope": "source_checkpoint_x_camera_location",
            "optimizer": {
                "class": "torch.optim.Adam",
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
            },
            "steps": 1,
            "episodic": False,
        },
    }


def test_target_cell_seals_probe_decision_before_evaluation_and_never_accepts_labels(
    tmp_path: Path,
) -> None:
    target_source = (ROOT / "experiments/kbound/cct20/target_executor.py").read_text(encoding="utf-8")
    assert "score_once" not in target_source
    assert "cct20_truth_loader" not in target_source
    gate = _gate_document()
    metadata = _metadata("7", 40)
    bad_backend = _binding_receipt(H64["checkpoint"], "7")
    bad_backend["backend_installation"]["strategy"] = "unsealed-backend"
    with pytest.raises(IntegrityError, match="hybrid-backend receipt"):
        LabelFreeTargetCell(
            metadata,
            checkpoint_seed=0,
            checkpoint_tensor_sha256=H64["checkpoint"],
            location_id="7",
            protocol_seal_sha256=H64["protocol"],
            target_manifest_sha256=H64["manifest"],
            gate_document=gate,
            tent_binding_receipt=bad_backend,
        )
    cell = LabelFreeTargetCell(
        metadata,
        checkpoint_seed=0,
        checkpoint_tensor_sha256=H64["checkpoint"],
        location_id="7",
        protocol_seal_sha256=H64["protocol"],
        target_manifest_sha256=H64["manifest"],
        gate_document=gate,
        tent_binding_receipt=_binding_receipt(H64["checkpoint"], "7"),
    )
    evaluation_batch = cell.batch_plan("evaluation")[0]
    with pytest.raises(IntegrityError, match="until the probe-only gate action"):
        cell.record_batch(
            role="evaluation",
            batch_index=0,
            image_ids=[row["image_id"] for row in evaluation_batch],
            frozen_logits=np.zeros((len(evaluation_batch), 16)),
            tent_logits=np.zeros((len(evaluation_batch), 16)),
        )
    for role in ("probe",):
        for index, batch in enumerate(cell.batch_plan(role)):
            frozen = np.zeros((len(batch), 16))
            adapted = frozen.copy()
            adapted[:, 0] = 1.0
            cell.record_batch(
                role=role,
                batch_index=index,
                image_ids=[row["image_id"] for row in batch],
                frozen_logits=frozen,
                tent_logits=adapted,
            )
    tent_update_receipt = {
        "schema": "kbound_cct20_tent_probe_update_v1",
        "checkpoint_tensor_sha256": H64["checkpoint"],
        "location_id": "7",
        "reset_scope": f"{H64['checkpoint']}:7",
        "parameter_names": ["bn.weight", "bn.bias"],
        "formula": "l2(after_probe-before_probe)/max(l2(before_probe),1e-12)",
        "normalized_tent_update_norm": 0.1,
    }
    bn_moment_receipt = {
        "schema": BN_GAUSSIAN_KL_SCHEMA,
        "formula": "channel_weighted_mean_gaussian_kl_probe_to_source",
        "numeric_implementation": BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION,
        "taylor_threshold": BN_GAUSSIAN_KL_TAYLOR_THRESHOLD,
        "taylor_terms": BN_GAUSSIAN_KL_TAYLOR_TERMS,
        "numeric_clipping": BN_GAUSSIAN_KL_NUMERIC_CLIPPING,
        "channel_count": 64,
        "batchnorm_batch_source_statistic_divergence": 0.2,
        "taylor_branch_channels": 0,
        "minimum_channel_kl": 0.2,
        "layers": [
            {
                "layer": "bn",
                "channels": 64,
                "values_per_channel": 10,
                "bn_eps": 1.0e-5,
                "mean_kl": 0.2,
                "min_kl": 0.2,
                "taylor_branch_channels": 0,
            }
        ],
    }
    action_path = tmp_path / "action.json"
    action = cell.seal_probe_action(
        tent_update_receipt=tent_update_receipt,
        frozen_bn_probe_moment_receipt=bn_moment_receipt,
        action_output_path=action_path,
    )
    assert action["decision"] in {"ADAPT", "FREEZE", "ABSTAIN"}
    assert (tmp_path / "action.json.receipt.json").is_file()
    replay = LabelFreeTargetCell(
        metadata,
        checkpoint_seed=0,
        checkpoint_tensor_sha256=H64["checkpoint"],
        location_id="7",
        protocol_seal_sha256=H64["protocol"],
        target_manifest_sha256=H64["manifest"],
        gate_document=gate,
        tent_binding_receipt=_binding_receipt(H64["checkpoint"], "7"),
    )
    for index, batch in enumerate(replay.batch_plan("probe")):
        frozen = np.zeros((len(batch), 16))
        adapted = frozen.copy()
        adapted[:, 0] = 1.0
        replay.record_batch(
            role="probe",
            batch_index=index,
            image_ids=[row["image_id"] for row in batch],
            frozen_logits=frozen,
            tent_logits=adapted,
        )
    restored = replay.restore_sealed_probe_action(
        tent_update_receipt=tent_update_receipt,
        frozen_bn_probe_moment_receipt=bn_moment_receipt,
        action_output_path=action_path,
    )
    assert restored["action_sha256"] == action["action_sha256"]
    for index, batch in enumerate(cell.batch_plan("evaluation")):
        frozen = np.zeros((len(batch), 16))
        adapted = frozen.copy()
        adapted[:, 1] = 1.0
        cell.record_batch(
            role="evaluation",
            batch_index=index,
            image_ids=[row["image_id"] for row in batch],
            frozen_logits=frozen,
            tent_logits=adapted,
        )
    artifact = cell.finalize()
    serialized = json.dumps(artifact).lower()
    assert '"label"' not in serialized
    assert '"accuracy"' not in serialized
    assert artifact["gate"]["probe_feature_record"]["n_probe_images"] > 0
    unsafe = {
        "stream_index": 0,
        "image_id": "x",
        "sequence_id": "s",
        "location_id": "7",
        "role": "evaluation",
        "frozen_prediction": 0,
        "adapted_prediction": 1,
        "label": 0,
    }
    with pytest.raises(IntegrityError, match="label/outcome"):
        build_prediction_cell(
            protocol_seal_sha256=H64["protocol"],
            gate_sha256=H64["gate"],
            target_manifest_sha256=H64["manifest"],
            checkpoint_seed=0,
            checkpoint_tensor_sha256=H64["checkpoint"],
            location_id="7",
            gate_result={
                "decision": "ADAPT",
                "support_status": "IN_SUPPORT",
                "delta_hat": 0.2,
                "epsilon": 0.1,
                "lower": 0.1,
                "upper": 0.3,
            },
            rows=[unsafe],
        )


def _scoring_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, list[Path], dict[str, set[int]], Path, Path]:
    locations = ["0", "7", "28", "40", "46", "78", "100", "105", "130"]
    truth_source = tmp_path / "trans_test_annotations.json"
    truth_source.write_text("sealed synthetic truth source\n", encoding="utf-8")
    dataset_dependencies, code_dependencies = _dependency_files(tmp_path, truth_source=truth_source)
    execution_seal = build_execution_seal(
        target_location_ids=locations,
        target_manifest_sha256=H64["manifest"],
        dataset_dependencies=dataset_dependencies,
        code_dependencies=code_dependencies,
        checkpoint_audit=_checkpoint_audit(tmp_path),
        gate_sha256=H64["gate"],
        protocol_config=_authoritative_protocol_config(),
    )
    execution_seal_path = tmp_path / "execution-seal.json"
    execution_receipt = write_immutable_json_with_receipt(execution_seal_path, execution_seal)
    protocol_artifact_hash = execution_receipt["artifact_sha256"]
    target_index = []
    truth = {}
    selected_by_location: dict[str, dict[str, dict]] = {}
    partition_hash_by_location: dict[str, str] = {}
    for location in locations:
        candidates = _metadata(location, 60)
        candidate_partition = sequence_atomic_partition(
            candidates,
            probe_fraction=0.30,
            salt="KBOUND_CCT20_PROBE_EVAL_v1",
        )
        selected = {role: dict(candidate_partition["roles"][role][0]) for role in ("probe", "evaluation")}
        selected_by_location[location] = selected
        selected_partition = sequence_atomic_partition(
            list(selected.values()),
            probe_fraction=0.30,
            salt="KBOUND_CCT20_PROBE_EVAL_v1",
        )
        partition_hash_by_location[location] = stable_sha256(selected_partition)
        for role, metadata in selected.items():
            target_index.append(metadata)
            truth[metadata["image_id"]] = {0, 1} if role == "evaluation" else {0}
    cells = []
    cell_paths = []
    synthetic_probe_features = extract_label_free_features(
        np.zeros((1, 16)),
        np.zeros((1, 16)),
        normalized_tent_update_norm=0.1,
        batchnorm_batch_source_statistic_divergence=0.2,
    )
    for seed in EXPECTED_MODEL_SEEDS:
        checkpoint_hash = stable_sha256({"tensor": seed})
        for location_index, location in enumerate(locations):
            helpful = (seed + location_index) % 2 == 0
            decision = "ADAPT" if helpful else "FREEZE"
            frozen_evaluation_logits = [0.0] * 16
            adapted_evaluation_logits = [0.0] * 16
            frozen_evaluation_logits[2 if helpful else 0] = 1.0
            adapted_evaluation_logits[0 if helpful else 2] = 1.0
            rows = [
                {
                    "stream_index": 0,
                    "image_id": selected_by_location[location]["probe"]["image_id"],
                    "sequence_id": selected_by_location[location]["probe"]["sequence_id"],
                    "location_id": location,
                    "role": "probe",
                    "frozen_prediction": 0,
                    "adapted_prediction": 0,
                    "frozen_logits": [0.0] * 16,
                    "adapted_logits": [0.0] * 16,
                },
                {
                    "stream_index": 1,
                    "image_id": selected_by_location[location]["evaluation"]["image_id"],
                    "sequence_id": selected_by_location[location]["evaluation"]["sequence_id"],
                    "location_id": location,
                    "role": "evaluation",
                    "frozen_prediction": 2 if helpful else 0,
                    "adapted_prediction": 0 if helpful else 2,
                    "frozen_logits": frozen_evaluation_logits,
                    "adapted_logits": adapted_evaluation_logits,
                },
            ]
            probe_trace = [
                {
                    "image_id": rows[0]["image_id"],
                    "sequence_id": rows[0]["sequence_id"],
                    "location_id": rows[0]["location_id"],
                    "frozen_logits": rows[0]["frozen_logits"],
                    "adapted_logits": rows[0]["adapted_logits"],
                }
            ]
            gate_result = {
                "decision": decision,
                "support_status": "IN_SUPPORT",
                "support_reasons": [],
                "delta_hat": 0.2 if helpful else -0.2,
                "epsilon": 0.1,
                "lower": 0.1 if helpful else -0.3,
                "upper": 0.3 if helpful else -0.1,
                "mahalanobis_diagnostic": 0.0,
                "probe_feature_record": synthetic_probe_features,
                "partition_sha256": partition_hash_by_location[location],
                "probe_trace_sha256": stable_sha256(probe_trace),
            }
            action_document = {
                "schema": "kbound_cct20_label_free_action_v1",
                "status": "SEALED_BEFORE_EVALUATION_STREAM",
                "protocol_seal_sha256": protocol_artifact_hash,
                "gate_sha256": H64["gate"],
                "target_manifest_sha256": H64["manifest"],
                "partition_sha256": gate_result["partition_sha256"],
                "probe_trace_sha256": gate_result["probe_trace_sha256"],
                "checkpoint_seed": seed,
                "checkpoint_tensor_sha256": checkpoint_hash,
                "location_id": location,
                "gate_result": gate_result,
            }
            action_document["action_sha256"] = stable_sha256(action_document)
            action_path = tmp_path / f"action-{seed}-{location}.json"
            action_receipt = write_immutable_json_with_receipt(action_path, action_document)
            gate_result = {
                **gate_result,
                "action_sha256": action_document["action_sha256"],
                "action_artifact_sha256": action_receipt["artifact_sha256"],
                "action_receipt": action_receipt,
            }
            cell = build_prediction_cell(
                protocol_seal_sha256=protocol_artifact_hash,
                gate_sha256=H64["gate"],
                target_manifest_sha256=H64["manifest"],
                checkpoint_seed=seed,
                checkpoint_tensor_sha256=checkpoint_hash,
                location_id=location,
                gate_result=gate_result,
                rows=rows,
            )
            cells.append(cell)
            path = tmp_path / f"cell-{seed}-{location}.json"
            write_immutable_json_with_receipt(path, cell)
            cell_paths.append(path)
    collection = build_prediction_collection(
        cells,
        target_index=target_index,
        target_location_ids=locations,
        expected_target_images=len(target_index),
    )
    collection_path = tmp_path / "collection.json"
    write_immutable_json_with_receipt(collection_path, collection)
    return (
        execution_seal_path,
        collection_path,
        cell_paths,
        truth,
        truth_source,
        tmp_path / "score.spent.json",
    )


def test_official_truth_loader_is_lazy_and_retains_complete_distinct_sets(tmp_path: Path) -> None:
    annotations = tmp_path / "trans_test_annotations.json"
    annotations.write_text("NOT OPENED AT CONSTRUCTION", encoding="utf-8")
    loader = cct20_truth_loader(annotations)
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1}, {"id": 2}],
                "annotations": [
                    {"image_id": 1, "category_id": 1},
                    {"image_id": 1, "category_id": 1},
                    {"image_id": 1, "category_id": 3},
                    {"image_id": 2, "category_id": 30},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert loader() == {"1": {0, 1}, "2": {11}}


def test_one_shot_set_valued_scorer_and_locked_two_way_inference(tmp_path: Path) -> None:
    execution_seal, collection_path, cell_paths, truth, truth_source, marker = _scoring_fixture(tmp_path)
    calls = []

    def load_truth() -> dict[str, set[int]]:
        assert marker.exists(), "marker must be spent before labels are loaded"
        calls.append("loaded")
        return truth

    load_truth.kbound_truth_source_path = str(truth_source.resolve())

    score_path = tmp_path / "score.json"
    score = score_once(
        execution_seal_path=execution_seal,
        prediction_collection_path=collection_path,
        prediction_cell_paths=cell_paths,
        truth_loader=load_truth,
        output_path=score_path,
        spent_marker_path=marker,
        expected_target_images=len(truth),
    )
    assert calls == ["loaded"]
    assert all(row["probe_predictions_scored"] is False for row in score["cells"])
    assert all(row["n_evaluation_images"] == 1 for row in score["cells"])
    assert {row["adaptation_benefit"] for row in score["cells"]} == {-1.0, 1.0}
    with pytest.raises(IntegrityError, match="already spent"):
        score_once(
            execution_seal_path=execution_seal,
            prediction_collection_path=collection_path,
            prediction_cell_paths=cell_paths,
            truth_loader=load_truth,
            output_path=tmp_path / "second-score.json",
            spent_marker_path=marker,
            expected_target_images=len(truth),
        )

    inference = analyze_score_document(score)
    assert inference["design"]["matrix_shape"] == [5, 9]
    assert inference["paired_two_way_product_bootstrap"]["replicates"] == 20_000
    for comparison in ("versus_always_adapt", "versus_always_freeze"):
        exact = inference["exact_nine_location_sign_flip_and_holm"][comparison]
        assert exact["enumerated_sign_patterns"] == 512
        assert 0.0 <= exact["holm_adjusted_p"] <= 1.0
        interval = inference["paired_two_way_product_bootstrap"]["results"][comparison][
            "simultaneous_bonferroni_97_5_ci"
        ]
        assert interval[0] > 0.0
    assert inference["adaptation_effect_mix"]["mixed_helpful_and_harmful_present"] is True
    assert inference["adaptation_effect_mix"] == {
        "helpful_cells_strictly_positive": 23,
        "neutral_cells_exactly_zero": 0,
        "harmful_cells_strictly_negative": 22,
        "mixed_helpful_and_harmful_present": True,
    }
    assert inference["action_exposure_at_checkpoint_location_unit"]["rates"]["ADAPT"] > 0.1
    assert inference["action_exposure_at_checkpoint_location_unit"]["rates"]["FREEZE"] > 0.1
    assert inference["strong_success_checks"]["protocol_strong_success"] is True
