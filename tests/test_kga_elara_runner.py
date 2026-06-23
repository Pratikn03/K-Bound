"""End-to-end tests for the canonical KGA-ELARA protocol runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.scripts.kbound import run_kga_elara_integration as runner

run_protocol = runner.run_protocol


def test_portable_path_hides_local_repository_prefix() -> None:
    path = runner.ROOT / "research_lock" / "KGA_ELARA_INTEGRATION_v1.yaml"

    assert runner._portable_path(path) == "research_lock/KGA_ELARA_INTEGRATION_v1.yaml"


def _write_cache(path: Path) -> None:
    rng = np.random.default_rng(12)
    y_val = np.array([0, 1] * 20)
    y_test = np.array([0, 1] * 25)
    s_val = np.column_stack(
        [
            np.clip(0.2 + 0.6 * y_val + rng.normal(0, 0.05, y_val.size), 0, 1),
            np.clip(0.3 + 0.4 * y_val + rng.normal(0, 0.08, y_val.size), 0, 1),
        ]
    )
    s_test = np.column_stack(
        [
            np.clip(0.2 + 0.6 * y_test + rng.normal(0, 0.05, y_test.size), 0, 1),
            np.clip(0.3 + 0.4 * y_test + rng.normal(0, 0.08, y_test.size), 0, 1),
        ]
    )
    np.savez(
        path,
        Sval=s_val,
        yval=y_val,
        Stest=s_test,
        ytest=y_test,
        valauc=np.array([0.9, 0.8]),
    )


def _write_protocol(path: Path, cache_dir: Path, output_dir: Path) -> None:
    path.write_text(
        f"""schema: kga_elara_integration_protocol_v1
date_declared: 2026-06-23
status: RETROSPECTIVE_OPENED_DATA
mode: retrospective_audit
alpha: 0.10
router_action: hybrid
abstain_fallback: freeze
output_dir: {output_dir}
claim_scope: retrospective_multimodal_instantiation_not_headline
tracks:
  - name: synthetic-track
    cache: {cache_dir}
    pattern: "*.npz"
integrity:
  data_previously_opened: true
  frozen_before_target_scoring: false
  label_free_claim_allowed: false
  headline_claim_allowed: false
""",
        encoding="utf-8",
    )


def test_runner_writes_versioned_claim_ineligible_artifacts(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache / "category.npz")
    out = tmp_path / "results"
    protocol = tmp_path / "protocol.yaml"
    _write_protocol(protocol, cache, out)

    summary = run_protocol(protocol, output_dir=out)

    for name in ("results.json", "results_table.tex", "FINDINGS.md", "run_manifest.json"):
        assert (out / name).exists()
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "kga_elara_integrated_results_v1"
    assert payload["mode"] == "retrospective_audit"
    assert payload["claim_eligibility"]["eligible"] is False
    assert payload["tracks"][0]["n_valid_categories"] == 1
    assert summary["n_matched_files"] == 1


def test_runner_dry_run_does_not_write_scored_results(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache / "category.npz")
    out = tmp_path / "results"
    protocol = tmp_path / "protocol.yaml"
    _write_protocol(protocol, cache, out)

    summary = run_protocol(protocol, output_dir=out, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["tracks"][0]["matched_files"] == 1
    assert not (out / "results.json").exists()


def test_launcher_exposes_full_and_dry_run_commands() -> None:
    launcher = Path("docs/research/kbound/scripts/kbtrain.sh").read_text(encoding="utf-8")
    assert "kga-elara-integrated)" in launcher
    assert "kga-elara-integrated-dry-run)" in launcher
