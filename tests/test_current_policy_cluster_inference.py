from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/analyze_current_policy_cluster_inference.py"
SPEC = importlib.util.spec_from_file_location("current_policy_cluster", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_generation_refreshes_cluster_before_safe_compatibility_outputs() -> None:
    path = ROOT / "docs/research/kbound/runbooks/release_candidate.sh"
    text = path.read_text()
    generation = text.split("step_generate() {", 1)[1].split("\n}", 1)[0]
    reconcile_at = generation.index("scripts/reconcile_result_panels.py")
    cluster_at = generation.index(
        "scripts/analyze_current_policy_cluster_inference.py",
        reconcile_at,
    )
    compatibility_at = generation.index("scripts/build_results_source_compat.py", cluster_at)
    assert reconcile_at < cluster_at < compatibility_at
    assert '"$PY" scripts/sync_reconciled_panels.py' not in generation


def test_paper_build_validates_but_does_not_regenerate_release_authority() -> None:
    path = ROOT / "docs/research/kbound/scripts/build_pdfs.sh"
    text = path.read_text()

    assert "validate_canonical_release_data.py" in text
    assert "validate_manuscript_claims.py" in text
    assert "scripts/reconcile_result_panels.py" not in text
    assert "scripts/analyze_current_policy_cluster_inference.py" not in text
    assert "scripts/sync_reconciled_panels.py" not in text


def test_saved_artifact_source_hashes_match_disk() -> None:
    artifact = json.loads(MODULE.DEFAULT_OUTPUT.read_text())
    for candidate in ("tent", "eata", "sar"):
        sources = artifact["candidates"][candidate]["sources"]
        assert sources
        for source in sources:
            path = ROOT / source["path"]
            assert path.is_file(), path
            assert path.stat().st_size == source["bytes"], path
            assert _sha256(path) == source["sha256"], path


def test_exact_sign_flip_has_expected_resolution() -> None:
    effects = np.ones(6)
    assert MODULE.exact_sign_flip_pvalue(effects) == 1 / 64


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = MODULE.holm_adjust({"a": 0.01, "b": 0.04})
    assert adjusted == {"a": 0.02, "b": 0.04}


def test_canonical_tent_uses_family_grain_and_positive_sign_convention() -> None:
    row = MODULE.analyze_candidate(
        MODULE.DEFAULT_SOURCE_DIR,
        "tent",
        n_boot=2_000,
        seed=20_260_827,
        ci_level=0.95,
    )
    assert row["grain"]["n_inference_units"] == 6
    assert row["grain"]["n_run_seeds"] == 5
    assert row["decision_counts"]["ADAPT"] > 0
    assert row["decision_counts"]["FREEZE"] > 0
    assert row["comparisons"]["always_adapt"]["point"] > 0
    assert row["comparisons"]["always_freeze"]["point"] > 0


def test_replayed_actions_match_canonical_panel_exactly() -> None:
    canonical = json.loads(
        (
            ROOT
            / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
        ).read_text()
    )["panels"]["cifar10c"]["panel"]["candidates"]
    expected = {
        candidate: {
            "ADAPT": row["adapt_count"],
            "FREEZE": row["freeze_count"],
            "ABSTAIN": row["abstain_count"],
        }
        for candidate, row in canonical.items()
    }

    for candidate in ("tent", "eata", "sar"):
        replayed = MODULE.analyze_candidate(
            MODULE.DEFAULT_SOURCE_DIR,
            candidate,
            n_boot=100,
            seed=20_260_827,
            ci_level=0.95,
        )
        assert replayed["decision_counts"] == expected[candidate]


def test_candidate_loader_uses_canonical_cells_not_raw_historical_bhat(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    conditions = ["family-a|condition-0", "family-b|condition-1"]

    def write_source(bhat: list[float]) -> None:
        records = [
            {
                "benchmark": "cifar10c",
                "method": "tent",
                "seed": 0,
                "condition": condition,
                "Z": [float(index), 1.0],
                "B": benefit,
                "a0": 0.5,
                "a_adapted": 0.5 + benefit,
                "b_hat": bhat[index],
                "eps_conformal": 99.0,
                "kga_decision": "ABSTAIN",
            }
            for index, (condition, benefit) in enumerate(zip(conditions, (0.2, -0.2), strict=True))
        ]
        (source_dir / "per_condition_cifar10c_tent_seed0.json").write_text(
            json.dumps(
                {
                    "schema": "kbound-compact-panel-source-v1",
                    "metadata": {"method": "tent", "seed": 0},
                    "records": records,
                }
            ),
            encoding="utf-8",
        )

    canonical_cells = [
        {
            "sample_id": MODULE.controlled_grid_sample_id(
                track="cifar10c", candidate="tent", seed=0, condition=condition
            ),
            "condition": condition,
            "prediction": prediction,
            "radius": 0.05,
            "action": action,
        }
        for index, (condition, prediction, action) in enumerate(
            zip(conditions, (0.3, -0.3), ("ADAPT", "FREEZE"), strict=True)
        )
    ]
    authority_sha256 = hashlib.sha256(
        (json.dumps(canonical_cells, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    ).hexdigest()
    predictions = [cell["prediction"] for cell in canonical_cells]
    radii = [cell["radius"] for cell in canonical_cells]
    actions = [cell["action"] for cell in canonical_cells]
    value_hash = lambda values: hashlib.sha256(
        (json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    ).hexdigest()
    protocol = {
        "schema": "kga-controlled-grid-crossfit-v1",
        "status": "ok",
        "alpha": 0.1,
        "requested_n_folds": 5,
        "calibration_fraction_target": 0.3,
        "gbrt": {
            "n_estimators": 250,
            "max_depth": 2,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 0,
        },
        "input_sha256": MODULE.controlled_grid_input_sha256(
            [[0.0, 1.0], [1.0, 1.0]],
            [0.2, -0.2],
            [cell["sample_id"] for cell in canonical_cells],
        ),
        "implementation_sha256": {
            name: _sha256(ROOT / MODULE.CURRENT_POLICY_BINDING_PATHS[name])
            for name in ("crossfit", "certificate", "policy", "numeric_validation")
        },
    }
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(
        json.dumps(
            {
                "panels": {
                    "cifar10c": {
                        "panel": {
                            "candidates": {
                                "tent": {
                                    "per_file": [
                                        {
                                            "seed": 0,
                                            "current_prediction_sha256": value_hash(predictions),
                                            "current_radius_sha256": value_hash(radii),
                                            "current_action_sha256": value_hash(actions),
                                            "crossfit_protocol": protocol,
                                            "current_cell_authority": {
                                                "schema": "kbound-controlled-grid-cell-authority-v1",
                                                "sha256": authority_sha256,
                                                "cells": canonical_cells,
                                            },
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    write_source([0.2, -0.2])
    expected, _ = MODULE.load_candidate(source_dir, "tent", canonical_path=canonical_path)
    write_source([1000.0, 1000.0])
    actual, _ = MODULE.load_candidate(source_dir, "tent", canonical_path=canonical_path)

    for observed in (expected, actual):
        assert [row["current_policy_prediction"] for row in observed] == [0.3, -0.3]
        assert [row["current_policy_epsilon"] for row in observed] == [0.05, 0.05]
        assert [row["current_policy_decision"] for row in observed] == ["ADAPT", "FREEZE"]

    path = source_dir / "per_condition_cifar10c_tent_seed0.json"
    changed = json.loads(path.read_text())
    changed["records"][0]["B"] = 0.25
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="input hash mismatch"):
        MODULE.load_candidate(source_dir, "tent", canonical_path=canonical_path)


def test_retrospective_holm_over_six_prospectively_named_contrasts_does_not_promote_tent() -> None:
    args = MODULE.argparse.Namespace(
        source_dir=MODULE.DEFAULT_SOURCE_DIR,
        output=MODULE.DEFAULT_OUTPUT,
        candidates=["tent", "eata", "sar"],
        n_boot=100,
        seed=20_260_827,
        ci_level=0.95,
    )
    artifact = MODULE.build_artifact(args)
    tent = artifact["candidates"]["tent"]

    assert artifact[MODULE.FAMILY_FIELD]["family_size"] == 6
    assert not tent["gate"][
        MODULE.GATE_PASS_FIELD
    ]
    for baseline in MODULE.BASELINES:
        adjusted = tent["comparisons"][baseline][MODULE.COMPARISON_P_FIELD]
        assert adjusted > 0.05
        assert adjusted * 64 == round(adjusted * 64)
