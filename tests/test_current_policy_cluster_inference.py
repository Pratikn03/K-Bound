from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/analyze_current_policy_cluster_inference.py"
SPEC = importlib.util.spec_from_file_location("current_policy_cluster", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_generation_paths_refresh_cluster_before_sync() -> None:
    paths = (
        ROOT / "docs/research/kbound/runbooks/release_candidate.sh",
        ROOT / "docs/research/kbound/scripts/build_pdfs.sh",
    )
    for path in paths:
        text = path.read_text()
        reconcile_at = text.index("scripts/reconcile_result_panels.py")
        cluster_at = text.index(
            "scripts/analyze_current_policy_cluster_inference.py",
            reconcile_at,
        )
        sync_at = text.index("scripts/sync_reconciled_panels.py", cluster_at)
        assert reconcile_at < cluster_at < sync_at, path


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


def test_preregistered_six_comparison_holm_does_not_promote_tent() -> None:
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

    assert artifact["preregistered_six_comparison_holm"]["family_size"] == 6
    assert not tent["gate"][
        "preregistered_six_comparison_cluster_sensitivity_pass"
    ]
    for baseline in MODULE.BASELINES:
        assert tent["comparisons"][baseline][
            "p_value_holm_preregistered_six_comparison_family"
        ] == 0.09375
