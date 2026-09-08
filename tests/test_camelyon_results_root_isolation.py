from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WILDS = ROOT / "experiments/kbound/wilds"
if str(WILDS) not in sys.path:
    sys.path.insert(0, str(WILDS))

import run_camelyon17_kbound as cam  # noqa: E402


def _install_synthetic_completed_run(monkeypatch):
    calls = {}
    records = [{"method": "tent", "seed": 0}]
    conditions = []
    meta = {"wall_sec": 0.0}
    manifest = {
        "execution_complete": True,
        "config_sha8": "deadbeef",
        "kbound_summary": {
            "classification": "synthetic",
            "mean_B": None,
            "detectability_verdict": "synthetic",
            "best_single_feature_harm_AUC": None,
            "gamma_S_proxy_indist_advantage": None,
            "gamma_T_proxy_oracle_advantage": None,
            "gamma_gap_within_delta": None,
        },
        "routing_b_multicandidate": {},
        "routing_c_smooth_drift": {},
    }

    def fake_run(args, partial_path=None):
        calls["args"] = args
        calls["partial_path"] = partial_path
        return records, conditions, meta

    def fake_dump(payload, output_path):
        assert payload is manifest
        calls["main_path"] = output_path

    def fake_serialize(run_records, *, dataset, out_dir, seeds, methods):
        assert run_records is records
        calls["serialize"] = {
            "dataset": dataset,
            "out_dir": out_dir,
            "seeds": seeds,
            "methods": methods,
        }
        return {"written": [], "kga_backend": "synthetic"}

    monkeypatch.setattr(cam, "run", fake_run)
    monkeypatch.setattr(cam, "build_manifest", lambda *_args: manifest)
    monkeypatch.setattr(cam.ri, "atomic_json_dump", fake_dump)
    monkeypatch.setattr(cam.pcs, "serialize_run", fake_serialize)
    return calls


def test_results_root_isolates_normal_partial_per_condition_and_main(monkeypatch, tmp_path):
    calls = _install_synthetic_completed_run(monkeypatch)
    results_root = tmp_path / "isolated-results"

    returned = cam.main([
        "--results-root", str(results_root),
        "--run-name", "bounded-camelyon",
        "--seeds", "7",
    ])

    run_dir = results_root / "bounded-camelyon"
    assert Path(calls["partial_path"]) == run_dir / "_partial.json"
    assert Path(calls["serialize"]["out_dir"]) == run_dir
    assert Path(calls["main_path"]) == run_dir / "diagnostic_deadbeef.json"
    assert Path(returned) == run_dir / "diagnostic_deadbeef.json"
    assert run_dir.is_dir()


def test_results_root_isolates_smoke_artifacts_under_fixed_smoke_name(monkeypatch, tmp_path):
    calls = _install_synthetic_completed_run(monkeypatch)
    results_root = tmp_path / "isolated-smoke"

    returned = cam.main([
        "--results-root", str(results_root),
        "--run-name", "ignored-for-smoke",
        "--smoke",
    ])

    run_dir = results_root / "wilds_kbound_smoke"
    assert Path(calls["partial_path"]) == run_dir / "_partial.json"
    assert Path(calls["serialize"]["out_dir"]) == run_dir
    assert Path(calls["main_path"]) == run_dir / "diagnostic_deadbeef.json"
    assert Path(returned) == run_dir / "diagnostic_deadbeef.json"


def test_explicit_out_remains_final_manifest_override(monkeypatch, tmp_path):
    calls = _install_synthetic_completed_run(monkeypatch)
    results_root = tmp_path / "isolated-results"
    explicit_out = tmp_path / "explicit" / "camelyon.json"

    returned = cam.main([
        "--results-root", str(results_root),
        "--run-name", "bounded-camelyon",
        "--out", str(explicit_out),
    ])

    run_dir = results_root / "bounded-camelyon"
    assert Path(calls["partial_path"]) == run_dir / "_partial.json"
    assert Path(calls["serialize"]["out_dir"]) == run_dir
    assert Path(calls["main_path"]) == explicit_out
    assert Path(returned) == explicit_out


def test_results_root_default_preserves_repository_results_location(monkeypatch, tmp_path):
    calls = _install_synthetic_completed_run(monkeypatch)
    synthetic_repo = tmp_path / "synthetic-repo"
    monkeypatch.setattr(cam, "REPO", str(synthetic_repo))

    returned = cam.main(["--run-name", "default-root"])

    run_dir = synthetic_repo / "experiments/kbound/results/default-root"
    assert Path(calls["partial_path"]) == run_dir / "_partial.json"
    assert Path(calls["serialize"]["out_dir"]) == run_dir
    assert Path(calls["main_path"]) == run_dir / "diagnostic_deadbeef.json"
    assert Path(returned) == run_dir / "diagnostic_deadbeef.json"
