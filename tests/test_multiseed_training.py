import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")
pytest.importorskip("torchvision")

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "experiments" / "kbound" / "training"
sys.path.insert(0, str(TRAINING))

from analysis import multicandidate_route  # noqa: E402
from calibration import exact_rank_radius  # noqa: E402
from iwildcam_label_free_stream import FEATURES  # noqa: E402
from iwildcam_label_free_stream import main as stream_main  # noqa: E402
from pacs_vlcs_runner import split_train_validation  # noqa: E402
from run_multiseed import (  # noqa: E402
    command_matrix,
    expected_paths,
    load_protocol,
    selected_jobs,
)
from theory_primitives import rankone_fit_offdiag  # noqa: E402
from uniform_multiseed import evaluate_method, route_without_target_labels  # noqa: E402

PROTOCOL_PATH = ROOT / "research_lock" / "MULTISEED_COMPLETION_PROTOCOL_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_seed(run_dir: Path, seed: int, *, corrupt_b: bool = False) -> None:
    benefits = np.asarray([0.08, -0.06, 0.04, -0.03, 0.10, -0.09])
    records = []
    for index, benefit in enumerate(benefits):
        frozen = 0.60 + 0.002 * seed
        adapted = frozen + benefit
        records.append(
            {
                "condition": f"condition-{index}",
                "Z": [benefit + 0.001 * seed, abs(benefit), index / 10.0],
                "Z_names": ["signed", "magnitude", "index"],
                "B": float(benefit + (0.01 if corrupt_b and index == 0 else 0.0)),
                "a0": frozen,
                "a_adapted": adapted,
            }
        )
    path = run_dir / f"seed{seed}" / f"per_condition_demo_sar_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}) + "\n")


def write_windows(path: Path, session: str, count: int, *, evidence: bool, outcomes: bool) -> None:
    windows = []
    for index in range(count):
        row = {"session_id": session, "window_end_batch": index}
        if evidence:
            row.update(
                {
                    FEATURES[0]: 0.1 + index * 0.01,
                    FEATURES[1]: 0.2 + index * 0.01,
                    FEATURES[2]: 2.0 + index,
                    FEATURES[3]: 0.3 + index * 0.01,
                }
            )
        if outcomes:
            benefit = 0.04 if index % 2 == 0 else -0.03
            row["frozen_window_f1"] = 0.60
            row["tent_window_f1"] = 0.60 + benefit
        windows.append(row)
    path.write_text(json.dumps({"windows": windows}) + "\n")


def test_exact_rank_radius_uses_observed_order_statistic() -> None:
    values = np.asarray([0.9, 0.1, 0.6, 0.2, 0.7, 0.3, 0.8, 0.4, 0.5, 1.0])
    assert exact_rank_radius(values, alpha=0.2) == 0.9
    assert exact_rank_radius(values[:9], alpha=0.1) == 0.9
    with pytest.raises(ValueError):
        exact_rank_radius(np.asarray([]), alpha=0.1)


def test_rankone_fit_uses_declared_positive_anchor() -> None:
    truth = np.asarray([0.7, 0.5, -0.4, 0.3])
    matrix = np.outer(truth, truth)
    np.fill_diagonal(matrix, 0.0)
    estimate, residual = rankone_fit_offdiag(matrix)
    assert estimate[0] >= 0.0
    assert residual < 1e-5
    assert np.sign(estimate[1:]).tolist() == np.sign(truth[1:]).tolist()


def test_multicandidate_route_is_self_contained() -> None:
    predictions = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    result = multicandidate_route(predictions, tau_star=10.0, min_D=2)
    assert result["decision"] in {"ADAPT", "FREEZE", "ABSTAIN"}
    assert "tau" in result


def test_pacs_source_split_has_no_shared_images() -> None:
    samples = [(f"class-{class_id}/image-{index}.jpg", class_id) for class_id in range(3) for index in range(10)]
    train, validation = split_train_validation(samples, nC=3, seed=7)
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(samples)
    assert {label for _, label in train} == {0, 1, 2}
    assert {label for _, label in validation} == {0, 1, 2}


def test_uniform_outer_seed_evaluation_is_disjoint_and_schema_checked(tmp_path: Path) -> None:
    for seed in (0, 1, 2):
        write_seed(tmp_path, seed)
    result = evaluate_method(tmp_path, "demo", "sar", [0, 1, 2], 0.10, nboot=100)
    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert fold["test_seed"] != fold["calibration_seed"]
        assert fold["test_seed"] not in fold["fit_seeds"]
        assert fold["calibration_seed"] not in fold["fit_seeds"]
    pooled = result["pooled"]
    assert "FA_u_interval" in pooled
    assert "beats_both_gain_ci" in pooled
    assert "beats_both_ci_robust" in pooled
    assert set(inspect.signature(route_without_target_labels).parameters) == {
        "estimator",
        "target_z",
        "calibration_residuals",
        "alpha",
    }


def test_uniform_outer_seed_rejects_inconsistent_benefit(tmp_path: Path) -> None:
    for seed in (0, 1):
        write_seed(tmp_path, seed)
    write_seed(tmp_path, 2, corrupt_b=True)
    with pytest.raises(ValueError, match="B is inconsistent"):
        evaluate_method(tmp_path, "demo", "sar", [0, 1, 2], 0.10, nboot=10)


def test_protocol_expands_complete_locked_matrix(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    jobs = selected_jobs(protocol, ["cifar10c_sar,imagenetc_sar", "pacs", "imagenetr"])
    assert jobs == ["cifar10c_sar", "imagenetc_sar", "pacs", "imagenetr"]
    datasets = {name: tmp_path / name for name in ("cifar", "imagenetc", "pacs", "imagenetr")}
    matrix = command_matrix(
        protocol,
        jobs,
        ROOT / "experiments" / "kbound" / "runs" / "test-plan",
        datasets,
        tmp_path / "imagenet_class_index.json",
        Path("/research/python"),
        "mps",
    )
    assert {name: len(rows) for name, rows in matrix.items()} == {
        "cifar10c_sar": 5,
        "imagenetc_sar": 4,
        "pacs": 3,
        "imagenetr": 1,
    }
    pacs_expected = expected_paths(
        tmp_path, "pacs", protocol["jobs"]["pacs"], seed=0
    )
    imagenetr_expected = expected_paths(
        tmp_path, "imagenetr", protocol["jobs"]["imagenetr"], seed=3
    )
    assert len(pacs_expected) == 3
    assert len(imagenetr_expected) == 10
    encoded = json.dumps(matrix, default=str)
    assert "AutoML_Flagship_V8" not in encoded


def test_imported_imagenetc_seed_zero_hashes_are_locked() -> None:
    root = ROOT / "experiments" / "kbound" / "results" / "imagenetc_seed0_v1"
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    for name, expected in provenance["files_sha256"].items():
        assert sha256(root / name) == expected


def test_iwildcam_writes_live_decisions_before_separate_outcome_join(tmp_path: Path) -> None:
    development = tmp_path / "development.json"
    calibration = tmp_path / "calibration.json"
    heldout_evidence = tmp_path / "heldout_evidence.json"
    heldout_outcomes = tmp_path / "heldout_outcomes.json"
    final_output = tmp_path / "result.json"
    live_output = tmp_path / "live.json"
    write_windows(development, "dev", 10, evidence=True, outcomes=True)
    write_windows(calibration, "cal", 10, evidence=True, outcomes=True)
    write_windows(heldout_evidence, "test", 6, evidence=True, outcomes=False)
    write_windows(heldout_outcomes, "test", 6, evidence=False, outcomes=True)

    assert stream_main(
        [
            "--development",
            str(development),
            "--calibration",
            str(calibration),
            "--heldout-evidence",
            str(heldout_evidence),
            "--heldout-outcomes",
            str(heldout_outcomes),
            "--output",
            str(final_output),
            "--live-output",
            str(live_output),
        ]
    ) == 0
    live_text = live_output.read_text()
    assert "frozen_window_f1" not in live_text
    assert "tent_window_f1" not in live_text
    assert json.loads(live_text)["outcomes_opened"] is False
    final = json.loads(final_output.read_text())
    assert final["live_decisions_sha256_before_outcome_open"] == sha256(live_output)
    assert final["split_integrity"]["files_disjoint"] is True


def test_migrated_certificate_paths_do_not_interpolate_residual_quantiles() -> None:
    for name in (
        "analysis.py",
        "cifar_tent_mps_v2.py",
        "pacs_vlcs_runner.py",
        "per_condition_serialize.py",
    ):
        source = (TRAINING / name).read_text()
        assert "np.quantile(np.abs" not in source
        assert "np.percentile(np.abs" not in source
