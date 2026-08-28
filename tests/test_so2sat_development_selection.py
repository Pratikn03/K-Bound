"""Synthetic tests for the So2Sat development-only adapter campaign."""

from __future__ import annotations

import copy
import inspect
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

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
    NoFeasibleCandidateError,
    build_candidate_bundle,
    build_gate_authorization,
    calibrate_selected_candidate,
    candidate_feasibility,
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
from experiments.kbound.so2sat.integrity import IntegrityError, stable_sha256
from experiments.kbound.so2sat.protocol import PROTOCOL_ID


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


def test_mixed_effect_feasibility_and_deterministic_selection_use_gate_fit_only() -> None:
    binding = _binding()
    strong = [-0.040, -0.030, -0.020, -0.010, 0.010, 0.020, 0.030, 0.040, 0.050]
    weaker = [value * 0.65 for value in strong]
    tent = _bundle(TENT_CANDIDATE_ID, strong, binding)
    sar = _bundle(SAR_CANDIDATE_ID, weaker, binding)
    assert candidate_feasibility(tent["cells"], study_binding=binding)["feasible"] is True
    selection = select_candidate([sar, tent], study_binding=binding)
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
    selection = select_candidate(bundles, study_binding=binding)
    assert selection["status"] == "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
    assert selection["selected_candidate_id"] is None
    with pytest.raises(NoFeasibleCandidateError, match="gate-calibration access is forbidden"):
        calibrate_selected_candidate(selection, {}, {}, study_binding=binding)


def test_gate_authorization_binds_selection_candidate_bundles_and_gate() -> None:
    binding = _binding()
    fit_effects = [-0.040, -0.030, -0.020, -0.010, 0.010, 0.020, 0.030, 0.040, 0.050]
    tent = _bundle(TENT_CANDIDATE_ID, fit_effects, binding)
    sar = _bundle(SAR_CANDIDATE_ID, [value * 0.65 for value in fit_effects], binding)
    selection = select_candidate([tent, sar], study_binding=binding)
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
