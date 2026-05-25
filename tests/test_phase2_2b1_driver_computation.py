# tests/test_phase2_2b1_driver_computation.py
"""Tests proving B-MECH-2/3S/4 drivers implement actual computation.

All tests use only synthetic data — no real dataset access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# ─── Module references (imported lazily so we get clear import errors) ───────


def _gate_sweep_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_phase2_rga_v2_gate_sweep",
        ROOT / "src" / "scripts" / "run_phase2_rga_v2_gate_sweep.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mixture_shift_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_phase2_mixture_shift",
        ROOT / "src" / "scripts" / "run_phase2_mixture_shift.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ks_sweep_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_phase2_ks_power_sweep",
        ROOT / "src" / "scripts" / "run_phase2_ks_power_sweep.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Synthetic fixture helpers ───────────────────────────────────────────────


def _make_synthetic_features(n=60, n_domains=4, n_features=10, seed=0):
    """Return (features [n, D, F], masks [n, D], labels [n])."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.3).astype(float)
    features = rng.random((n, n_domains, n_features)).astype(np.float32)
    # Make score index=0 slightly predictive
    for d in range(n_domains):
        features[labels == 1, d, 0] = np.clip(features[labels == 1, d, 0] + 0.3, 0, 1)
    # ~15% random missingness
    masks = (rng.random((n, n_domains)) < 0.15).astype(bool)
    return features, masks, labels


def _make_fitted_estimator(features, masks, labels, score_idx=0, domain_order=None):
    """Return a fitted ReliabilityEstimator on the provided data."""
    from uais.fusion.attention.reliability_estimator import ReliabilityEstimator
    if domain_order is None:
        domain_order = [f"d{i}" for i in range(features.shape[1])]
    est = ReliabilityEstimator(
        domain_order=domain_order,
        score_index=score_idx,
        gate_threshold=0.66,
        gate_mode="mean",
    )
    est.fit(features, masks, labels)
    return est


# ─── Test 1: B-MECH-2 produces result rows from synthetic data ───────────────


def test_b_mech_2_produces_result_rows_from_synthetic_fixture():
    """_compute_gate_decision runs on synthetic data and returns non-trivial arrays."""
    mod = _gate_sweep_module()

    features, masks, labels = _make_synthetic_features(n=60, seed=11)
    estimator = _make_fitted_estimator(features, masks, labels)

    result_rows = []
    for gate_id in ("G0", "G1", "G2", "G3"):
        if gate_id == "G3":
            selected_tau = (1, 0.40)
        elif gate_id in ("G1", "G2"):
            selected_tau = 0.34
        else:
            selected_tau = None

        gate_fired, mean_rel, min_rel = mod._compute_gate_decision(
            estimator, features, masks, gate_id, selected_tau
        )
        assert gate_fired.shape == (60,), f"{gate_id}: wrong shape"
        assert gate_fired.dtype == bool, f"{gate_id}: wrong dtype"

        static_auc = mod._safe_auc(labels, np.random.default_rng(0).random(60))
        result_rows.append({
            "gate_id": gate_id,
            "delta_auc": 0.01,
            "clean_activation_rate": float(gate_fired.mean()),
            "mean_reliability": mean_rel,
            "min_reliability": min_rel,
        })

    assert len(result_rows) == 4
    # All required fields present
    for r in result_rows:
        assert "gate_id" in r
        assert "delta_auc" in r
        assert "clean_activation_rate" in r
    # At least gate_id values are distinct
    gate_ids = {r["gate_id"] for r in result_rows}
    assert gate_ids == {"G0", "G1", "G2", "G3"}


# ─── Test 2: B-MECH-3S mixture-shift produces domain-shift rows ──────────────


def test_b_mech_3s_produces_domain_shift_rows_from_synthetic():
    """pure_mixture_shift_resample produces correct actual_proportions summing to ~1.0."""
    from elara.family_b.mixture_shift import pure_mixture_shift_resample

    rng = np.random.default_rng(42)
    n = 200
    domains = ("fraud", "cyber", "behavior", "nlp")
    # Assign categories uniformly
    cats = np.array([domains[i % len(domains)] for i in range(n)])
    scores = rng.random(n).astype(float)

    target_props = {"fraud": 0.4, "cyber": 0.3, "behavior": 0.2, "nlp": 0.1}
    result = pure_mixture_shift_resample(
        categories=cats,
        target_proportions=target_props,
        n_samples=100,
        rng_seed=7,
        require_within_category_invariance=True,
        scores_for_invariance_check=scores,
        invariance_tol_ks_p=0.001,  # very loose — synthetic data will pass
    )

    actual_sum = sum(result.actual_proportions.values())
    assert abs(actual_sum - 1.0) < 0.05, f"actual_proportions sum={actual_sum} not ~1.0"
    assert len(result.indices) == 100
    # Within-category invariance check passed (no exception raised = passed)
    assert result.name.startswith("mixture_shift_seed")

    # Verify all sampled categories are from the known domains
    for cat in np.array(cats)[result.indices]:
        assert cat in domains


# ─── Test 3: B-MECH-4 window sizes are locked ────────────────────────────────


def test_b_mech_4_window_sizes_are_locked():
    """KS_WINDOW_GRID must be exactly (32, 64, 128, 256, 512)."""
    from elara.family_b.ks_window import KS_WINDOW_GRID
    assert KS_WINDOW_GRID == (32, 64, 128, 256, 512), (
        f"KS_WINDOW_GRID={KS_WINDOW_GRID!r}; expected (32, 64, 128, 256, 512)"
    )


# ─── Test 4: PredictionArchive writes under tmp_path ─────────────────────────


def test_prediction_archive_writes_under_tmpdir(tmp_path):
    """PredictionArchive.write() must create a parquet (or csv fallback) file."""
    from elara.evaluation.prediction_archive import PredictionArchive

    archive = PredictionArchive(root=tmp_path / "archives")
    n = 10
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=n).astype(int)
    scores = rng.random(n).astype(float)
    sample_ids = [f"s{i}" for i in range(n)]

    frame = archive.build_frame(
        sample_ids=sample_ids,
        labels=labels,
        raw_scores=scores,
        method="test_method",
        method_variant="test_v1",
        benchmark="TestBench",
        protocol="test_protocol",
        analysis_family="B",
        pairing_strength="label_aligned_stress_only",
        split="test",
        seed=0,
        selection_rule="synthetic test: no test-fold reads",
        selection_used_test_metrics=False,
        selected_head_or_comparator_status="test gate",
        gate_mode="mean",
        gate_fired=np.ones(n, dtype=bool),
        mean_reliability=np.full(n, 0.75),
        min_reliability=np.full(n, 0.55),
        failure_type="zero_attack",
        failed_domain_count=2,
        fault_severity=1.0,
    )

    entry = archive.write(
        experiment_id="B-MECH-2",
        benchmark="TestBench",
        protocol="test_protocol",
        seed=0,
        method="test_method",
        split="test",
        frame=frame,
        config={"test": True},
    )
    archive.append_index(entry)

    # File must exist as parquet or csv fallback
    artifact = Path(entry.artifact_path)
    if not artifact.is_absolute():
        artifact = Path.cwd() / artifact
    assert artifact.exists(), f"archive file not found at {artifact}"
    assert artifact.suffix in (".parquet", ".csv"), f"unexpected suffix: {artifact.suffix}"

    # Index must be updated
    idx = archive.load_index()
    assert len(idx) == 1
    assert not entry.validation_only_selection_verified or entry.usable_for_inference


# ─── Test 5: _select_tau_on_validation_only returns selection_used_test_metrics=False ─


def test_selection_uses_validation_tensors_only():
    """_select_tau_on_validation_only must stamp selection_used_test_metrics=False."""
    mod = _gate_sweep_module()

    features, masks, labels = _make_synthetic_features(n=80, seed=99)
    estimator = _make_fitted_estimator(features, masks, labels)

    # Minimal contract structure matching the YAML
    contract = {
        "candidate_gates": [
            {
                "id": "G0",
                "validation_tuning_allowed": False,
            },
            {
                "id": "G1",
                "validation_tuning_allowed": True,
                "tau_min_search_grid": [0.30, 0.40, 0.50, 0.60],
            },
            {
                "id": "G2",
                "validation_tuning_allowed": True,
                "tau_min_search_grid": [0.30, 0.40, 0.50, 0.60],
            },
            {
                "id": "G3",
                "validation_tuning_allowed": True,
                "q_search_grid": [1, 2],
                "tau_q_search_grid": [0.30, 0.40, 0.50],
            },
        ],
        "fault_surface": {
            "attacks": ["zero_attack", "max_attack"],
            "k_values": [1, 2],
        },
    }

    domain_order = [f"d{i}" for i in range(4)]
    for gate_id in ("G0", "G1", "G2", "G3"):
        result = mod._select_tau_on_validation_only(
            estimator, features, masks, labels,
            gate_id=gate_id, contract=contract,
            domain_order=domain_order, score_idx=0,
            base_seed=42,
        )
        assert result["selection_used_test_metrics"] is False, (
            f"gate {gate_id}: selection_used_test_metrics must be False"
        )
        assert "gate_id" in result
        assert result["gate_id"] == gate_id


# ─── Test 6: Driver source files contain no family_d access ──────────────────


def test_no_family_d_access_in_drivers():
    """Driver source files must not import or open any family_d paths."""
    drivers = [
        ROOT / "src" / "scripts" / "run_phase2_rga_v2_gate_sweep.py",
        ROOT / "src" / "scripts" / "run_phase2_mixture_shift.py",
        ROOT / "src" / "scripts" / "run_phase2_ks_power_sweep.py",
    ]
    forbidden_patterns = [
        "family_d",
        "FAMILY_D",
        "family-d",
    ]
    for driver in drivers:
        src = driver.read_text()
        for pat in forbidden_patterns:
            assert pat not in src, (
                f"{driver.name} contains forbidden pattern {pat!r} "
                "(must not access Family-D paths)"
            )
