"""Synthetic tests for the So2Sat development-only adapter campaign."""

from __future__ import annotations

import ast
import copy
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from experiments.kbound.so2sat import adapters as adapters_module
from experiments.kbound.so2sat import development as development_module
from experiments.kbound.so2sat.adapters import (
    CANDIDATE_IDS,
    SAR_CANDIDATE_ID,
    TENT_CANDIDATE_ID,
    adapt_on_probe,
    candidate_spec,
    fixed_model_logits,
    frozen_logits_and_bn_divergence,
)
from experiments.kbound.so2sat.development import (
    GATE_CAL_ROLE,
    GATE_FIT_ROLE,
    NO_FEASIBLE_CANDIDATE_EXIT_CODE,
    CityPartition,
    DevelopmentData,
    DevelopmentRow,
    NoFeasibleCandidateError,
    _inspect_candidate_output_state,
    _load_reusable_candidate_bundle,
    _partition_hash,
    build_candidate_bundle,
    build_gate_authorization,
    calibrate_selected_candidate,
    candidate_feasibility,
    development_environment_identity,
    run_candidate_selection,
    run_gate_calibration,
    select_candidate,
    validate_candidate_bundle,
    validate_gate_authorization,
    validate_selection,
)
from experiments.kbound.so2sat.features import extract_label_free_features
from experiments.kbound.so2sat.gate import (
    CHECKPOINT_IDS,
    STUDY_BINDING_SCHEMA,
    trace_identity_sha256,
    validate_study_binding,
)
from experiments.kbound.so2sat.integrity import (
    IntegrityError,
    stable_sha256,
    write_immutable_json_with_receipt,
)
from experiments.kbound.so2sat.protocol import PROTOCOL_ID

SOURCE_ACCEPTANCE_BINDING = {
    "source_postrun_acceptance_artifact_basename": (
        "so2sat_source_postrun_acceptance.json"
    ),
    "source_postrun_acceptance_artifact_sha256": "a" * 64,
    "source_postrun_acceptance_canonical_document_sha256": "b" * 64,
}


class _TinyBatchNormClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(10)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(10, 17)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.fill_(-5.0)
            self.fc.bias[0] = 5.0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.bn(images)).flatten(1))


def _binding() -> dict[str, Any]:
    document = {
        "schema": STUDY_BINDING_SCHEMA,
        "status": "SEALED_LABEL_FREE_POPULATION_BINDING",
        "protocol_id": PROTOCOL_ID,
        "manifest_artifact_sha256": "1" * 64,
        "manifest_canonical_document_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "population_identity_sha256": "4" * 64,
        "protocol_file_sha256": "5" * 64,
        "protocol_document_sha256": "6" * 64,
        "gate_fit_cities": [f"fit{i:02d}" for i in range(9)],
        "gate_cal_cities": [f"cal{i:02d}" for i in range(19)],
        "target_cities": [f"target{i:02d}" for i in range(10)],
    }
    document["binding_sha256"] = stable_sha256(document)
    validate_study_binding(document)
    return document


def _feature(signal: float, checkpoint: int) -> dict[str, Any]:
    frozen = np.zeros((8, 17), dtype=np.float64)
    frozen[:, 0] = 1.0 + checkpoint * 0.01
    adapted = frozen.copy()
    adapted[:, 0] += signal * 12.0
    adapted[:, 1] -= signal * 2.0
    return extract_label_free_features(
        frozen,
        adapted,
        normalized_adapter_update_norm=abs(signal) + 0.001 * checkpoint,
        batchnorm_source_statistic_divergence=0.2 + signal,
    )


def _cells(
    candidate_id: str,
    binding: dict[str, Any],
    *,
    benefits: list[float],
    role: str = "gate_fit",
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    spec = candidate_spec(candidate_id)
    code_files = {"synthetic.py": "7" * 64}
    code_hash = stable_sha256(code_files)
    cities = binding["gate_fit_cities"] if role == "gate_fit" else binding["gate_cal_cities"]
    for city_index, city in enumerate(cities):
        partition_hash = stable_sha256({"city": city, "partition": "west-east"})
        for checkpoint in CHECKPOINT_IDS:
            checkpoint_number = int(checkpoint)
            benefit = benefits[city_index] + (checkpoint_number - 2) * 0.0002
            frozen_accuracy = 0.70
            adapted_accuracy = frozen_accuracy + benefit
            feature = _feature(benefits[city_index], checkpoint_number)
            checkpoint_tensor = stable_sha256({"checkpoint": checkpoint, "kind": "tensor"})
            checkpoint_file = stable_sha256({"checkpoint": checkpoint, "kind": "file"})
            trace_id = f"{candidate_id}:{role}:{city}:checkpoint{checkpoint}"
            trace_sha = trace_identity_sha256(
                role=role,
                city_id=city,
                checkpoint_id=checkpoint,
                checkpoint_tensor_sha256=checkpoint_tensor,
                checkpoint_file_sha256=checkpoint_file,
                trace_id=trace_id,
                partition_sha256=partition_hash,
                feature_sha256=feature["feature_sha256"],
                manifest_sha256=binding["manifest_sha256"],
                population_identity_sha256=binding["population_identity_sha256"],
                protocol_file_sha256=binding["protocol_file_sha256"],
                protocol_document_sha256=binding["protocol_document_sha256"],
            )
            gate_row = {
                "role": role,
                "city_id": city,
                "checkpoint_id": checkpoint,
                "checkpoint_tensor_sha256": checkpoint_tensor,
                "checkpoint_file_sha256": checkpoint_file,
                "trace_id": trace_id,
                "trace_sha256": trace_sha,
                "partition_sha256": partition_hash,
                "manifest_sha256": binding["manifest_sha256"],
                "population_identity_sha256": binding["population_identity_sha256"],
                "protocol_file_sha256": binding["protocol_file_sha256"],
                "protocol_document_sha256": binding["protocol_document_sha256"],
                "feature_document": feature,
                "observed_benefit": benefit,
            }
            cell = {
                "schema": "kbound_so2sat_development_adapter_cell_v1",
                "status": "DEVELOPMENT_ONLY_COMPLETE",
                "candidate_id": candidate_id,
                "candidate_config_sha256": spec["candidate_config_sha256"],
                "role": role,
                "city_id": city,
                "checkpoint_id": checkpoint,
                "probe_n": 8,
                "evaluation_n": 1000,
                "frozen_evaluation_accuracy": frozen_accuracy,
                "adapted_evaluation_accuracy": adapted_accuracy,
                "observed_benefit": benefit,
                "adapter_diagnostics": {
                    "candidate_id": candidate_id,
                    "selected_parameter_names": ["bn1.weight", "bn1.bias"],
                    "probe_batches": 1,
                    "optimizer_updates": 1,
                    "reliable_examples": 8,
                    "skipped_empty_reliable_batches": 0,
                    "model_recovery_resets": 0,
                    "normalized_adapter_update_norm": 0.01,
                    "batchnorm_source_statistic_divergence": 0.2,
                },
                "gate_row": gate_row,
                "source_training_receipt_sha256": stable_sha256({"checkpoint": checkpoint, "receipt": True}),
                "source_normalizer_sha256": "8" * 64,
                "source_container_identity_sha256": "9" * 64,
                "runner_code_sha256": code_hash,
                "probe_labels_read": 0,
                "evaluation_label_read_passes": 2,
                "target_pixels_read": 0,
                "target_labels_read": 0,
                "target_inputs": [],
            }
            cell["cell_sha256"] = stable_sha256(cell)
            cells.append(cell)
    return cells


def _bundle(
    candidate_id: str,
    benefits: list[float],
    binding: dict[str, Any],
    *,
    role: str = "gate_fit",
) -> dict[str, Any]:
    return build_candidate_bundle(
        candidate_id=candidate_id,
        role=role,
        cells=_cells(candidate_id, binding, benefits=benefits, role=role),
        study_binding=binding,
        checkpoint_collection={"schema": "synthetic-five-checkpoint-collection"},
        source_container_identity_sha256="9" * 64,
        normalizer_sha256="8" * 64,
        code_identity={
            "files_sha256": {"synthetic.py": "7" * 64},
            "code_sha256": stable_sha256({"synthetic.py": "7" * 64}),
        },
        development_environment=development_environment_identity(
            torch.device("cpu")
        ),
    )


def _partition(role: str, city_id: str) -> CityPartition:
    probe_role = f"{role}_probe"
    evaluation_role = f"{role}_evaluation"
    probe = (
        DevelopmentRow(
            row_index=0,
            sample_id="training:00000000",
            city_id=city_id,
            spatial_block_id="epsg:block-west",
            sample_role=probe_role,
        ),
    )
    evaluation = (
        DevelopmentRow(
            row_index=1,
            sample_id="training:00000001",
            city_id=city_id,
            spatial_block_id="epsg:block-east",
            sample_role=evaluation_role,
        ),
    )
    return CityPartition(
        role=role,
        city_id=city_id,
        probe_rows=probe,
        evaluation_rows=evaluation,
        partition_sha256=_partition_hash(role, city_id, probe, evaluation),
    )


def test_phase_specific_data_authority_rejects_gate_cal_before_hdf5_read() -> None:
    class _TrapContainer:
        def read_pixels(self, _rows: Any) -> np.ndarray:
            raise AssertionError("HDF5 pixel read occurred before the role check")

    data = object.__new__(DevelopmentData)
    data.authorized_role = GATE_FIT_ROLE
    data._authorized = {}
    data.container = _TrapContainer()
    partition = _partition(GATE_CAL_ROLE, "cal00")
    with pytest.raises(IntegrityError, match="phase-specific authority"):
        list(data.pixel_batches(partition, half="probe"))


def test_gate_fit_bundles_bind_one_self_hashed_runtime() -> None:
    binding = _binding()
    effects = [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
    tent = _bundle(TENT_CANDIDATE_ID, effects, binding)
    sar = _bundle(SAR_CANDIDATE_ID, [value * 0.7 for value in effects], binding)
    assert (
        tent["development_environment_identity"]
        == sar["development_environment_identity"]
    )

    changed = copy.deepcopy(sar)
    environment = changed["development_environment_identity"]
    environment["torch_num_threads"] += 1
    environment["environment_identity_sha256"] = stable_sha256(
        {
            key: value
            for key, value in environment.items()
            if key != "environment_identity_sha256"
        }
    )
    changed["bundle_sha256"] = stable_sha256(
        {key: value for key, value in changed.items() if key != "bundle_sha256"}
    )
    validate_candidate_bundle(changed, study_binding=binding)
    with pytest.raises(IntegrityError, match="different environments"):
        select_candidate(
            [tent, changed],
            study_binding=binding,
            source_postrun_acceptance=SOURCE_ACCEPTANCE_BINDING,
        )


def test_candidate_output_state_rejects_incomplete_and_unknown_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "development"
    assert not any(_inspect_candidate_output_state(destination).values())
    destination.mkdir()
    incomplete = destination / f"so2sat_{TENT_CANDIDATE_ID}.gate_fit.json"
    incomplete.write_text("{}\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="incomplete artifact/receipt pair"):
        _inspect_candidate_output_state(destination)

    other = tmp_path / "other-development"
    other.mkdir()
    (other / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="unknown state"):
        _inspect_candidate_output_state(other)

    non_regular = tmp_path / "non-regular-development"
    non_regular.mkdir()
    (non_regular / f"so2sat_{TENT_CANDIDATE_ID}.gate_fit.json").mkdir()
    with pytest.raises(IntegrityError, match="non-regular member"):
        _inspect_candidate_output_state(non_regular)


def test_complete_candidate_pair_is_reusable_only_with_exact_live_bindings(
    tmp_path: Path,
) -> None:
    binding = _binding()
    effects = [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
    bundle = _bundle(TENT_CANDIDATE_ID, effects, binding)
    destination = tmp_path / "development"
    destination.mkdir()
    path = destination / f"so2sat_{TENT_CANDIDATE_ID}.gate_fit.json"
    write_immutable_json_with_receipt(path, bundle)
    state = _inspect_candidate_output_state(destination)
    assert state[TENT_CANDIDATE_ID] is True
    assert state[SAR_CANDIDATE_ID] is False
    reused = _load_reusable_candidate_bundle(
        path,
        candidate_id=TENT_CANDIDATE_ID,
        study_binding=binding,
        checkpoint_collection={"schema": "synthetic-five-checkpoint-collection"},
        source_container_identity_sha256="9" * 64,
        normalizer_sha256="8" * 64,
        code_identity=bundle["runner_code"],
        development_environment=bundle["development_environment_identity"],
    )
    assert reused == bundle
    with pytest.raises(IntegrityError, match="differs from the current sealed run"):
        _load_reusable_candidate_bundle(
            path,
            candidate_id=TENT_CANDIDATE_ID,
            study_binding=binding,
            checkpoint_collection={"schema": "synthetic-five-checkpoint-collection"},
            source_container_identity_sha256="0" * 64,
            normalizer_sha256="8" * 64,
            code_identity=bundle["runner_code"],
            development_environment=bundle["development_environment_identity"],
        )


def test_frozen_tent_and_sar_specs_are_hash_pinned_and_probe_only() -> None:
    for candidate_id in CANDIDATE_IDS:
        spec = candidate_spec(candidate_id)
        assert spec["adaptation_data_role"] == "development_probe_only"
        assert spec["adaptation_labels_read"] == 0
        assert spec["target_inputs"] == []
        assert spec["target_tuning"] is False
        assert spec["deployment"]["evaluation_batch_statistics_used"] is False
        assert spec["official_commit"] in {
            "e9e926a668d85244c66a6d5c006efbd2b82e83e8",
            "20f6e24b17525f34503510afccedc0629b67b7c4",
        }


@pytest.mark.parametrize("candidate_id", [TENT_CANDIDATE_ID, SAR_CANDIDATE_ID])
def test_adapter_updates_only_probe_then_becomes_fixed(candidate_id: str) -> None:
    torch.manual_seed(11)
    source = _TinyBatchNormClassifier().eval()
    probe = [torch.randn(6, 10, 32, 32), torch.randn(5, 10, 32, 32) + 0.4]
    frozen_logits, divergence = frozen_logits_and_bn_divergence(
        source,
        (batch.clone() for batch in probe),
        device=torch.device("cpu"),
    )
    adapted, diagnostics = adapt_on_probe(
        source,
        (batch.clone() for batch in probe),
        candidate_id=candidate_id,
        device=torch.device("cpu"),
        batchnorm_source_statistic_divergence=divergence,
    )
    assert frozen_logits.shape == (11, 17)
    assert diagnostics.probe_batches == 2
    assert diagnostics.reliable_examples >= 0
    assert adapted.training is False
    assert adapted.bn.track_running_stats is True
    assert adapted.bn.running_mean is not None
    assert adapted.bn.running_var is not None
    assert all(not parameter.requires_grad for parameter in adapted.parameters())
    before = {name: tensor.clone() for name, tensor in adapted.state_dict().items()}
    logits = fixed_model_logits(
        adapted,
        [torch.randn(4, 10, 32, 32)],
        device=torch.device("cpu"),
    )
    assert logits.shape == (4, 17)
    assert all(torch.equal(before[name], tensor) for name, tensor in adapted.state_dict().items())
    assert torch.equal(source.bn.running_mean, torch.zeros_like(source.bn.running_mean))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS regression coverage requires Apple Silicon",
)
def test_batchnorm_divergence_is_finite_and_cpu_consistent_on_mps() -> None:
    torch.manual_seed(17)
    cpu_model = _TinyBatchNormClassifier().eval()
    mps_model = copy.deepcopy(cpu_model).to(torch.device("mps")).eval()
    probe = [
        torch.randn(6, 10, 32, 32) + 0.4,
        torch.randn(5, 10, 32, 32) - 0.2,
    ]

    cpu_logits, cpu_divergence = frozen_logits_and_bn_divergence(
        cpu_model,
        (batch.clone() for batch in probe),
        device=torch.device("cpu"),
    )
    mps_logits, mps_divergence = frozen_logits_and_bn_divergence(
        mps_model,
        (batch.clone() for batch in probe),
        device=torch.device("mps"),
    )

    assert cpu_logits.shape == mps_logits.shape == (11, 17)
    assert torch.isfinite(mps_logits).all()
    assert mps_divergence >= 0.0
    assert mps_divergence == pytest.approx(cpu_divergence, rel=1e-7, abs=1e-9)
    parameter = nn.Parameter(torch.tensor([3.0, 4.0], device=torch.device("mps")))
    assert adapters_module._parameter_norm([parameter]) == pytest.approx(5.0)


def test_batchnorm_divergence_preserves_primary_hook_error() -> None:
    model = _TinyBatchNormClassifier().eval()
    with pytest.raises(IntegrityError, match="hook received invalid activations"):
        frozen_logits_and_bn_divergence(
            model,
            [torch.randn(6, 10, 32)],
            device=torch.device("cpu"),
        )


def test_so2sat_runtime_uses_mps_safe_cpu_float64_conversions() -> None:
    unsafe: list[str] = []
    source_root = Path(__file__).parents[1] / "experiments/kbound/so2sat"
    for source_path in sorted(source_root.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr == "to":
                keywords = {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg
                }
                device = keywords.get("device")
                dtype = keywords.get("dtype")
                sends_to_cpu = (
                    isinstance(device, ast.Constant) and device.value == "cpu"
                )
                widens_to_float64 = (
                    dtype is not None and ast.unparse(dtype) == "torch.float64"
                )
                if sends_to_cpu and widens_to_float64:
                    unsafe.append(f"combined-to:{source_path.name}:{node.lineno}")
            if node.func.attr == "cpu" and isinstance(node.func.value, ast.Call):
                inner = node.func.value.func
                if isinstance(inner, ast.Attribute) and inner.attr == "double":
                    unsafe.append(f"double-before-cpu:{source_path.name}:{node.lineno}")
    assert unsafe == [], f"MPS-unsafe float64 conversions: {unsafe}"


def test_mixed_effect_feasibility_and_deterministic_selection_use_gate_fit_only() -> None:
    binding = _binding()
    strong = [-0.040, -0.030, -0.020, -0.010, 0.010, 0.020, 0.030, 0.040, 0.050]
    weaker = [value * 0.65 for value in strong]
    tent = _bundle(TENT_CANDIDATE_ID, strong, binding)
    sar = _bundle(SAR_CANDIDATE_ID, weaker, binding)
    assert candidate_feasibility(tent["cells"], study_binding=binding)["feasible"] is True
    selection = select_candidate(
        [sar, tent],
        study_binding=binding,
        source_postrun_acceptance=SOURCE_ACCEPTANCE_BINDING,
    )
    assert selection["status"] == "EXACTLY_ONE_CANDIDATE_SELECTED_BEFORE_GATE_CAL"
    assert selection["selected_candidate_id"] == TENT_CANDIDATE_ID
    assert selection["gate_cal_rows_read_before_selection"] == 0
    validate_selection(selection, study_binding=binding)

    tampered = copy.deepcopy(tent)
    tampered["candidate_feasibility"]["feasible"] = False
    tampered["bundle_sha256"] = stable_sha256({key: value for key, value in tampered.items() if key != "bundle_sha256"})
    with pytest.raises(IntegrityError, match="does not replay"):
        validate_candidate_bundle(tampered, study_binding=binding)


def test_all_one_direction_effects_seal_honest_stop_before_gate_calibration() -> None:
    binding = _binding()
    all_helpful = [0.010 + index * 0.001 for index in range(9)]
    bundles = [
        _bundle(TENT_CANDIDATE_ID, all_helpful, binding),
        _bundle(SAR_CANDIDATE_ID, all_helpful, binding),
    ]
    selection = select_candidate(
        bundles,
        study_binding=binding,
        source_postrun_acceptance=SOURCE_ACCEPTANCE_BINDING,
    )
    assert selection["status"] == "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
    assert selection["selected_candidate_id"] is None
    with pytest.raises(NoFeasibleCandidateError, match="gate-calibration access is forbidden"):
        calibrate_selected_candidate(selection, {}, {}, study_binding=binding)


def test_gate_authorization_binds_selection_candidate_bundles_and_gate() -> None:
    binding = _binding()
    fit_effects = [-0.040, -0.030, -0.020, -0.010, 0.010, 0.020, 0.030, 0.040, 0.050]
    tent = _bundle(TENT_CANDIDATE_ID, fit_effects, binding)
    sar = _bundle(SAR_CANDIDATE_ID, [value * 0.65 for value in fit_effects], binding)
    selection = select_candidate(
        [tent, sar],
        study_binding=binding,
        source_postrun_acceptance=SOURCE_ACCEPTANCE_BINDING,
    )
    selected_fit = tent if selection["selected_candidate_id"] == TENT_CANDIDATE_ID else sar
    calibration_effects = np.linspace(-0.045, 0.045, 19).tolist()
    calibration = _bundle(
        selection["selected_candidate_id"],
        calibration_effects,
        binding,
        role="gate_cal",
    )
    gate = calibrate_selected_candidate(
        selection,
        selected_fit,
        calibration,
        study_binding=binding,
    )
    authorization = build_gate_authorization(
        selection,
        selected_fit,
        calibration,
        gate,
        study_binding=binding,
    )
    validate_gate_authorization(
        authorization,
        selection=selection,
        fit_bundle=selected_fit,
        calibration_bundle=calibration,
        gate=gate,
        study_binding=binding,
    )
    tampered = copy.deepcopy(authorization)
    tampered["selected_gate_cal_bundle_sha256"] = "f" * 64
    tampered["authorization_sha256"] = stable_sha256(
        {key: value for key, value in tampered.items() if key != "authorization_sha256"}
    )
    with pytest.raises(IntegrityError, match="replay selected_gate_cal_bundle_sha256 mismatch"):
        validate_gate_authorization(
            tampered,
            selection=selection,
            fit_bundle=selected_fit,
            calibration_bundle=calibration,
            gate=gate,
            study_binding=binding,
        )


def test_real_cli_surfaces_only_training_data_and_two_separate_phases() -> None:
    select_parameters = set(inspect.signature(run_candidate_selection).parameters)
    calibration_parameters = set(inspect.signature(run_gate_calibration).parameters)
    assert "training_data" in select_parameters
    assert "training_data" in calibration_parameters
    assert all("validation" not in name and "testing" not in name for name in select_parameters)
    assert all("validation" not in name and "testing" not in name for name in calibration_parameters)
    assert "selection_path" not in select_parameters
    assert "selection_path" in calibration_parameters


def test_source_acceptance_replays_before_gate_fit_data_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"acceptance": 0, "inventory": 0}

    def reject_acceptance(
        *_args: Any, **_kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        calls["acceptance"] += 1
        raise IntegrityError("acceptance-first-stop")

    def forbidden_inventory(*_args: Any, **_kwargs: Any) -> Any:
        calls["inventory"] += 1
        raise AssertionError("gate-fit inventory constructed before source acceptance")

    monkeypatch.setattr(
        development_module,
        "verify_source_postrun_acceptance_bindings",
        reject_acceptance,
    )
    monkeypatch.setattr(
        development_module,
        "_load_manifest_and_inventory",
        forbidden_inventory,
    )
    with pytest.raises(IntegrityError, match="acceptance-first-stop"):
        run_candidate_selection(
            population_manifest=tmp_path / "population.json",
            source_postrun_acceptance_path=tmp_path / "acceptance.json",
            source_preflight_path=tmp_path / "preflight.json",
            training_geo=tmp_path / "training_geo.h5",
            training_data=tmp_path / "training.h5",
            checkpoint_dir=tmp_path / "checkpoints",
            output_dir=tmp_path / "development",
            device=torch.device("cpu"),
        )
    assert calls == {"acceptance": 1, "inventory": 0}


def test_incomplete_output_stops_before_source_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "development"
    output.mkdir()
    (output / f"so2sat_{TENT_CANDIDATE_ID}.gate_fit.json").write_text(
        "{}\n", encoding="utf-8"
    )
    calls = {"acceptance": 0}

    def forbidden_acceptance(*_args: Any, **_kwargs: Any) -> Any:
        calls["acceptance"] += 1
        raise AssertionError("source chain was rehashed before output-state validation")

    monkeypatch.setattr(
        development_module,
        "verify_source_postrun_acceptance_bindings",
        forbidden_acceptance,
    )
    with pytest.raises(IntegrityError, match="incomplete artifact/receipt pair"):
        run_candidate_selection(
            population_manifest=tmp_path / "population.json",
            source_postrun_acceptance_path=tmp_path / "acceptance.json",
            source_preflight_path=tmp_path / "preflight.json",
            training_geo=tmp_path / "training_geo.h5",
            training_data=tmp_path / "training.h5",
            checkpoint_dir=tmp_path / "checkpoints",
            output_dir=output,
            device=torch.device("cpu"),
        )
    assert calls == {"acceptance": 0}


def test_no_candidate_stops_gate_calibration_before_source_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()
    all_helpful = [0.010 + index * 0.001 for index in range(9)]
    selection = select_candidate(
        [
            _bundle(TENT_CANDIDATE_ID, all_helpful, binding),
            _bundle(SAR_CANDIDATE_ID, all_helpful, binding),
        ],
        study_binding=binding,
        source_postrun_acceptance=SOURCE_ACCEPTANCE_BINDING,
    )
    selection_path = tmp_path / "so2sat_candidate_selection.json"
    write_immutable_json_with_receipt(selection_path, selection)
    calls = {"acceptance": 0}

    monkeypatch.setattr(development_module, "load_study_binding", lambda _path: binding)

    def forbidden_acceptance(*_args: Any, **_kwargs: Any) -> Any:
        calls["acceptance"] += 1
        raise AssertionError("source chain was rehashed after a sealed no-candidate stop")

    monkeypatch.setattr(
        development_module,
        "verify_source_postrun_acceptance_bindings",
        forbidden_acceptance,
    )
    with pytest.raises(NoFeasibleCandidateError, match="enter gate calibration"):
        run_gate_calibration(
            selection_path=selection_path,
            source_postrun_acceptance_path=tmp_path / "acceptance.json",
            source_preflight_path=tmp_path / "preflight.json",
            precalibration_seal_path=tmp_path / "preseal.json",
            target_boundary_amendment_path=tmp_path / "amendment.json",
            population_manifest=tmp_path / "population.json",
            training_geo=tmp_path / "training_geo.h5",
            training_data=tmp_path / "training.h5",
            checkpoint_dir=tmp_path / "checkpoints",
            output_dir=tmp_path / "development",
            device=torch.device("cpu"),
        )
    assert calls == {"acceptance": 0}


def test_cli_exits_twenty_after_sealing_no_candidate_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        development_module,
        "run_candidate_selection",
        lambda **_kwargs: {
            "status": "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL",
            "selected_candidate_id": None,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "development.py",
            "select",
            "--population-manifest",
            str(tmp_path / "population.json"),
            "--training-geo",
            str(tmp_path / "training_geo.h5"),
            "--training-data",
            str(tmp_path / "training.h5"),
            "--source-postrun-acceptance",
            str(tmp_path / "acceptance.json"),
            "--source-preflight",
            str(tmp_path / "preflight.json"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-dir",
            str(tmp_path / "development"),
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        development_module.main()
    assert stopped.value.code == NO_FEASIBLE_CANDIDATE_EXIT_CODE
    assert "STOP: no feasible candidate" in capsys.readouterr().out
