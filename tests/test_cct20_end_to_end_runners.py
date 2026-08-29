"""Focused tests for the CCT-20 development and label-free target runners."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from experiments.kbound.cct20 import runner_runtime as runtime_module
from experiments.kbound.cct20.integrity import IntegrityError, stable_sha256
from experiments.kbound.cct20.label_free_traces import (
    TARGET_BATCH_SIZE,
    extract_label_free_features,
)
from experiments.kbound.cct20.protocol_seal import REQUIRED_CODE_DEPENDENCY_NAMES
from experiments.kbound.cct20.run_development_gate import (
    BATCH_SIZE,
    _trace_document,
    validate_development_trace,
    validate_resumed_development_trace,
)
from experiments.kbound.cct20.run_locked_target import (
    _artifact_pair_state,
    normalize_target_manifest,
)
from experiments.kbound.cct20.runner_runtime import (
    SHARED_RUNTIME_DEPENDENCY_NAMES,
    VerifiedImageStore,
    build_shared_runtime_identity,
    configure_deterministic_inference,
    expected_backend_strategy,
    paired_forward,
    select_inference_device,
    validate_runtime_addendum,
    validate_shared_runtime_identity,
    verify_checkpoint_audit_document,
    write_or_verify_immutable_json,
)
from experiments.kbound.cct20.seal_cct20_execution import _code_dependencies
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
)

ROOT = Path(__file__).resolve().parents[1]


def _hybrid_backend_receipt() -> dict[str, object]:
    state_hash = stable_sha256("root-bn-state")
    return {
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
        "source_bn_state_sha256": state_hash,
        "installed_bn_state_sha256": state_hash,
        "state_hash_equal": True,
        "official_tent_parameter_devices": ["cpu", "mps"],
        "configured_track_running_stats": False,
        "configured_running_moments_absent": True,
    }


def test_checkpoint_audit_replay_enforces_all_independence_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = {
        "imagenet_backbone_tensor_sha256": stable_sha256("backbone"),
        "config_recipe_sha256": stable_sha256("recipe"),
        "data_sha256": stable_sha256("data"),
        "code_sha256": stable_sha256("code"),
    }
    rows = [
        {
            "model_seed": seed,
            "path": f"/synthetic/checkpoint-{seed}.pt",
            "bytes": 100 + seed,
            "file_sha256": stable_sha256({"file": seed}),
            "tensor_sha256": stable_sha256({"tensor": seed}),
            "initial_tensor_sha256": stable_sha256({"initial": seed}),
            "config_sha256": stable_sha256({"config": seed}),
            **shared,
        }
        for seed in range(5)
    ]
    identities = {row["path"]: dict(row) for row in rows}
    monkeypatch.setattr(
        runtime_module,
        "checkpoint_identity",
        lambda path: dict(identities[str(path)]),
    )
    audit = {
        "schema": "kbound_cct20_independent_checkpoint_audit_v1",
        "status": "PASS",
        "required_model_seeds": list(range(5)),
        "n_checkpoints": 5,
        "all_file_hashes_distinct": True,
        "all_tensor_hashes_distinct": True,
        "all_initial_tensor_hashes_distinct": True,
        "all_config_hashes_distinct": True,
        "shared_config_recipe_sha256": shared["config_recipe_sha256"],
        "shared_imagenet_backbone_tensor_sha256": shared["imagenet_backbone_tensor_sha256"],
        "shared_data_sha256": shared["data_sha256"],
        "shared_code_sha256": shared["code_sha256"],
        "checkpoints": rows,
    }
    assert verify_checkpoint_audit_document(audit) == rows
    tampered = copy.deepcopy(audit)
    tampered["shared_code_sha256"] = stable_sha256("different-code")
    with pytest.raises(IntegrityError, match="shared_code_sha256"):
        verify_checkpoint_audit_document(tampered)


def _write_image(path: Path, colour: tuple[int, int, int] = (20, 30, 40)) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), colour).save(path, format="JPEG")
    return path.read_bytes()


def test_verified_image_store_checks_exact_bytes_and_safe_population(tmp_path: Path) -> None:
    image_path = tmp_path / "camera" / "one.jpg"
    payload = _write_image(image_path)
    store = VerifiedImageStore(
        tmp_path,
        [
            {
                "image_id": "one",
                "file_name": "camera/one.jpg",
                "image_bytes": len(payload),
                "image_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    )
    tensor = store.tensor_batch(
        [
            {
                "image_id": "one",
                "sequence_id": "sequence-one",
                "location_id": "7",
                "file_name": "camera/one.jpg",
                "frame_num": 0,
                "date_captured": "2020-01-01T00:00:00",
            }
        ]
    )
    assert tensor.shape == (1, 3, 224, 224)
    _write_image(image_path, (200, 10, 10))
    with pytest.raises(IntegrityError, match="byte count changed|SHA-256 changed"):
        store.tensor_batch(
            [
                {
                    "image_id": "one",
                    "sequence_id": "sequence-one",
                    "location_id": "7",
                    "file_name": "camera/one.jpg",
                    "frame_num": 0,
                    "date_captured": "2020-01-01T00:00:00",
                }
            ]
        )


def test_bn_probe_sufficient_statistics_match_concatenated_population_moments() -> None:
    model = nn.Sequential(nn.BatchNorm2d(3)).eval()
    generator = torch.Generator().manual_seed(20260828)
    chunks = [torch.randn(size, 3, 7, 5, generator=generator) for size in (3, 5, 7)]
    split = FrozenBatchNormMomentAccumulator(model)
    with torch.no_grad():
        for chunk in chunks:
            model(chunk)
    split_receipt = split.finalize()
    combined = FrozenBatchNormMomentAccumulator(model)
    with torch.no_grad():
        model(torch.cat(chunks, dim=0))
    combined_receipt = combined.finalize()
    assert split_receipt["layers"][0]["values_per_channel"] == 15 * 7 * 5
    assert split_receipt["batchnorm_batch_source_statistic_divergence"] == pytest.approx(
        combined_receipt["batchnorm_batch_source_statistic_divergence"],
        rel=1e-6,
        abs=1e-9,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS regression requires Apple Silicon acceleration",
)
def test_paired_forward_transfers_mps_logits_before_float64() -> None:
    device = torch.device("mps")
    frozen = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 16)).to(device).eval()
    adapted = copy.deepcopy(frozen)
    frozen_logits, adapted_logits = paired_forward(
        frozen,
        adapted,
        torch.randn(3, 3, 4, 4),
        device=device,
    )
    assert frozen_logits.shape == adapted_logits.shape == (3, 16)
    assert frozen_logits.dtype == adapted_logits.dtype == np.float64


def test_target_manifest_normalizer_exposes_only_label_free_stream_fields() -> None:
    manifest = {
        "images": [
            {
                "id": 1,
                "file_name": "0/one.jpg",
                "location": 0,
                "seq_id": "s-1",
                "frame_num": 0,
                "date_captured": "2020-01-01T00:00:00",
            }
        ],
        "samples": [
            {
                "id": 1,
                "file_name": "0/one.jpg",
                "image_bytes": 10,
                "image_sha256": stable_sha256("image"),
            }
        ],
    }
    # Add one minimal row per remaining sealed camera.
    for location in (7, 28, 40, 46, 78, 100, 105, 130):
        manifest["images"].append(
            {
                "id": location,
                "file_name": f"{location}/one.jpg",
                "location": location,
                "seq_id": f"s-{location}",
                "frame_num": 0,
                "date_captured": "2020-01-01T00:00:00",
            }
        )
        manifest["samples"].append(
            {
                "id": location,
                "file_name": f"{location}/one.jpg",
                "image_bytes": 10,
                "image_sha256": stable_sha256({"image": location}),
            }
        )
    metadata, expected = normalize_target_manifest(manifest)
    assert len(metadata) == len(expected) == 9
    assert set(metadata[0]) == {
        "image_id",
        "sequence_id",
        "location_id",
        "file_name",
        "frame_num",
        "date_captured",
    }


def test_development_trace_replays_features_and_benefit() -> None:
    frozen = np.zeros((2, 16), dtype=float)
    adapted = frozen.copy()
    adapted[:, 1] = 1.0
    features = extract_label_free_features(
        frozen,
        adapted,
        normalized_tent_update_norm=0.1,
        batchnorm_batch_source_statistic_divergence=0.2,
    )
    tensor_hash = stable_sha256("tensor")
    trace = _trace_document(
        split_name="trans_val",
        role="development_fit",
        location_id="125",
        checkpoint_row={
            "model_seed": 0,
            "tensor_sha256": tensor_hash,
            "file_sha256": stable_sha256("file"),
        },
        partition={
            "schema": "partition",
            "roles": {
                "probe": [{"image_id": f"p-{index}", "sequence_id": f"ps-{index}"} for index in range(2)],
                "evaluation": [{"image_id": f"e-{index}", "sequence_id": f"es-{index}"} for index in range(2)],
            },
        },
        binding_receipt={
            "schema": "kbound_cct20_official_tent_binding_v1",
            "checkpoint_tensor_sha256": tensor_hash,
            "location_id": "125",
            "reset_scope": f"{tensor_hash}:125",
            "parameter_names": ["bn.weight", "bn.bias"],
            "n_parameters": 2,
            "initial_bn_affine_l2": 1.0,
            "update_norm_formula": ("l2(after_probe-before_probe)/max(l2(before_probe),1e-12)"),
            "backend_installation": _hybrid_backend_receipt(),
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
        },
        feature_record={
            **features,
            "diagnostic_receipts": {
                "tent_update": {
                    "schema": "kbound_cct20_tent_probe_update_v1",
                    "checkpoint_tensor_sha256": tensor_hash,
                    "location_id": "125",
                    "reset_scope": f"{tensor_hash}:125",
                    "parameter_names": ["bn.weight", "bn.bias"],
                    "normalized_tent_update_norm": 0.1,
                    "formula": ("l2(after_probe-before_probe)/max(l2(before_probe),1e-12)"),
                },
                "frozen_bn_probe_moments": {
                    "schema": BN_GAUSSIAN_KL_SCHEMA,
                    "batchnorm_batch_source_statistic_divergence": 0.2,
                    "channel_count": 64,
                    "numeric_implementation": BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION,
                    "taylor_threshold": BN_GAUSSIAN_KL_TAYLOR_THRESHOLD,
                    "taylor_terms": BN_GAUSSIAN_KL_TAYLOR_TERMS,
                    "numeric_clipping": BN_GAUSSIAN_KL_NUMERIC_CLIPPING,
                    "taylor_branch_channels": 0,
                    "minimum_channel_kl": 0.2,
                    "layers": [
                        {
                            "layer": "bn",
                            "channels": 64,
                            "values_per_channel": 2,
                            "bn_eps": 1.0e-5,
                            "mean_kl": 0.2,
                            "min_kl": 0.2,
                            "taylor_branch_channels": 0,
                        }
                    ],
                    "formula": "channel_weighted_mean_gaussian_kl_probe_to_source",
                },
            },
        },
        probe_rows=[
            {
                "image_id": f"p-{index}",
                "sequence_id": f"ps-{index}",
                "frozen_logits": frozen[index].tolist(),
                "adapted_logits": adapted[index].tolist(),
            }
            for index in range(2)
        ],
        evaluation_rows=[
            {
                "image_id": "e-0",
                "sequence_id": "es-0",
                "ground_truth_output_indices": [1],
                "frozen_prediction": 0,
                "adapted_prediction": 1,
                "frozen_logits": frozen[0].tolist(),
                "adapted_logits": adapted[0].tolist(),
                "frozen_correct": False,
                "adapted_correct": True,
            },
            {
                "image_id": "e-1",
                "sequence_id": "es-1",
                "ground_truth_output_indices": [0],
                "frozen_prediction": 0,
                "adapted_prediction": 1,
                "frozen_logits": frozen[1].tolist(),
                "adapted_logits": adapted[1].tolist(),
                "frozen_correct": True,
                "adapted_correct": False,
            },
        ],
        shared_runtime_sha256=stable_sha256("shared-runtime"),
    )
    validate_development_trace(trace)
    assert trace["observed_benefit"] == 0.0
    tampered = dict(trace)
    tampered["observed_benefit"] = 0.5
    tampered["trace_sha256"] = stable_sha256({key: value for key, value in tampered.items() if key != "trace_sha256"})
    with pytest.raises(IntegrityError, match="benefit does not replay"):
        validate_development_trace(tampered)
    bad_tent = copy.deepcopy(trace)
    bad_tent["official_tent_binding"]["provenance"]["git_commit"] = "0" * 40
    bad_tent["trace_sha256"] = stable_sha256({key: value for key, value in bad_tent.items() if key != "trace_sha256"})
    with pytest.raises(IntegrityError, match="pinned official Tent"):
        validate_development_trace(bad_tent)
    bad_update = copy.deepcopy(trace)
    bad_update["probe_feature_record"]["diagnostic_receipts"]["tent_update"]["normalized_tent_update_norm"] = 0.9
    bad_update["trace_sha256"] = stable_sha256(
        {key: value for key, value in bad_update.items() if key != "trace_sha256"}
    )
    with pytest.raises(IntegrityError, match="update norm does not reconcile"):
        validate_development_trace(bad_update)
    with pytest.raises(IntegrityError, match="identity differs"):
        validate_resumed_development_trace(
            trace,
            split_name="trans_val",
            location_id="125",
            checkpoint_row={
                "model_seed": 0,
                "tensor_sha256": tensor_hash,
                "file_sha256": stable_sha256("file"),
            },
            shared_runtime_sha256=stable_sha256("different-runtime"),
        )


def test_immutable_resume_is_exact_and_partial_pairs_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    document = {"schema": "test", "value": 1}
    first = write_or_verify_immutable_json(path, document)
    second = write_or_verify_immutable_json(path, document)
    assert first == second
    with pytest.raises(IntegrityError, match="differs from replay"):
        write_or_verify_immutable_json(path, {"schema": "test", "value": 2})
    partial = tmp_path / "partial.json"
    partial.write_text("{}\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="incomplete immutable"):
        _artifact_pair_state(partial)


def test_target_entrypoint_has_no_scorer_import_and_seal_freezes_exact_code_set() -> None:
    source_path = ROOT / "experiments/kbound/cct20/run_locked_target.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    assert all("score_once" not in name for name in imports)
    assert all("two_way_inference" not in name for name in imports)
    args = argparse.Namespace(
        checkpoint_audit=Path("audit.json"),
        gate=Path("gate.json"),
        tent_repo=ROOT / "external/tent_official",
        shared_runtime_identity=Path("runtime.json"),
        runtime_addendum=Path("runtime-addendum.yaml"),
    )
    names = set(_code_dependencies(ROOT, args))
    assert names == REQUIRED_CODE_DEPENDENCY_NAMES


def test_shared_runtime_replays_backend_threads_and_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = {}
    for name in sorted(SHARED_RUNTIME_DEPENDENCY_NAMES):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"dependency:{name}\n", encoding="utf-8")
        dependencies[name] = path
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 10)
    configure_deterministic_inference()
    runtime = build_shared_runtime_identity(
        torch.device("mps"),
        source_training_seal_artifact_sha256=stable_sha256("source-artifact"),
        source_training_seal_document_sha256=stable_sha256("source-document"),
        dependency_paths=dependencies,
    )
    assert runtime["backend_strategy"] == expected_backend_strategy()
    assert runtime["torch_intraop_threads"] == 4
    assert runtime["torch_interop_threads"] == 10
    validate_shared_runtime_identity(
        runtime,
        device=torch.device("mps"),
        dependency_paths=dependencies,
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    with pytest.raises(IntegrityError, match="fresh-process torch thread counts"):
        validate_shared_runtime_identity(
            runtime,
            device=torch.device("mps"),
            dependency_paths=dependencies,
        )
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    dependencies["integrity"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="live shared runtime differs"):
        validate_shared_runtime_identity(
            runtime,
            device=torch.device("mps"),
            dependency_paths=dependencies,
        )


def test_execution_batch_size_is_one_shared_runtime_contract() -> None:
    assert BATCH_SIZE == TARGET_BATCH_SIZE == 32
    assert expected_backend_strategy()["sequence_atomic_max_images"] == TARGET_BATCH_SIZE


def test_runtime_addendum_and_explicit_device_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addendum = ROOT / "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml"
    tent_repo = ROOT / "external/tent_official"
    validate_runtime_addendum(
        ROOT,
        tent_repo=tent_repo,
        runtime_addendum=addendum,
    )
    with pytest.raises(IntegrityError, match="explicit --device mps"):
        select_inference_device("auto")

    original_file_sha256 = runtime_module.file_sha256

    def changed_target_hash(path: str | Path) -> str:
        if Path(path).name == "run_locked_target.py":
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(runtime_module, "file_sha256", changed_target_hash)
    with pytest.raises(IntegrityError, match="code identity mismatch: .*run_locked_target.py"):
        validate_runtime_addendum(
            ROOT,
            tent_repo=tent_repo,
            runtime_addendum=addendum,
        )


def test_cct20_runtime_never_combines_cpu_transfer_with_float64_conversion() -> None:
    unsafe: list[str] = []
    source_root = ROOT / "experiments/kbound/cct20"
    for source_path in sorted(source_root.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "to"):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            device = keywords.get("device")
            dtype = keywords.get("dtype")
            sends_to_cpu = isinstance(device, ast.Constant) and device.value == "cpu"
            widens_to_float64 = dtype is not None and ast.unparse(dtype) == "torch.float64"
            if sends_to_cpu and widens_to_float64:
                unsafe.append(f"{source_path.name}:{node.lineno}")
    assert unsafe == [], f"MPS-unsafe combined CPU/float64 transfers: {unsafe}"
