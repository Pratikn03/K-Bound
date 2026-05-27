"""Phase 1.E — manifest integrity tests.

The manifest is the single source of truth for every manuscript-quoted
result. These tests assert:
  - the manifest exists and parses;
  - every Family A confirmatory cell has both RGA+ and primary comparator entries;
  - selection rules are recorded;
  - source-artifact paths are present;
  - forbidden inferential statuses (confirmatory / pre-registered) do not appear
    on existing-cell claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MANIFEST = Path("docs/research/metrics_manifest.json")
MACROS = Path("docs/research/generated/elara_verified_metrics_macros.tex")


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip(f"manifest missing: {MANIFEST}")
    return json.loads(MANIFEST.read_text())


def test_manifest_has_claims(manifest):
    assert isinstance(manifest.get("claims"), list)
    assert len(manifest["claims"]) >= 11, "expected at least 11 claims (A1-A8 + C1-C3)"


def test_family_a_confirmatory_cells_have_required_fields(manifest):
    for c in manifest["claims"]:
        if c["analysis_status"] != "audited primary reanalysis":
            continue
        for f in (
            "rga_plus_head",
            "rga_plus_test_roc_auc",
            "rga_plus_validation_roc_auc",
            "primary_comparator",
            "primary_comparator_test_roc_auc",
            "primary_comparator_validation_roc_auc",
            "delong_p_raw_single_rep_seed",
            "delong_p_holm_family_A_K5",
        ):
            assert c.get(f) is not None, f"cell {c['cell_id']} missing required field {f}"


def test_no_claim_uses_confirmatory_language(manifest):
    """Existing inspected cells must NOT carry a 'confirmatory' or 'pre-registered' status."""
    for c in manifest["claims"]:
        status = c["analysis_status"]
        assert status in {
            "protocol-diagnostic",
            "audited primary reanalysis",
            "exploratory",
        }, f"cell {c['cell_id']} has forbidden status {status!r} — AR-11 reserves confirmatory/pre-registered for Family D only."


def test_selection_rules_recorded(manifest):
    for c in manifest["claims"]:
        if c["analysis_status"] == "audited primary reanalysis":
            assert "validation_frozen" in (
                c.get("rga_plus_selection_rule") or ""
            ), f"cell {c['cell_id']} rga_plus_selection_rule must contain 'validation_frozen'"
            assert "validation_frozen" in (
                c.get("primary_comparator_selection_rule") or ""
            ), f"cell {c['cell_id']} primary_comparator_selection_rule must contain 'validation_frozen'"


def test_source_artifacts_present(manifest):
    for c in manifest["claims"]:
        s = c.get("source_artifacts") or {}
        for k in (
            "rga_plus_selection_csv",
            "comparator_selection_csv",
            "audited_inference_csv",
            "descriptive_variability_csv",
        ):
            assert k in s and s[k], f"cell {c['cell_id']} missing source_artifact {k}"


def test_pairing_strength_recorded(manifest):
    valid = {
        "independent_modalities",
        "naturally_structured_views",
        "derived_view_proxy",
        "label_aligned_stress_only",
        "local_replay",
    }
    for c in manifest["claims"]:
        ps = c.get("pairing_strength")
        assert ps in valid, f"cell {c['cell_id']} pairing_strength={ps!r} is not in the locked taxonomy"


def test_macros_file_exists():
    assert MACROS.exists(), f"generated macros file missing: {MACROS}"


def test_macros_contain_no_underscore_in_macro_name():
    """LaTeX \\newcommand{\\name}{value} disallows digits and underscores in the name."""
    if not MACROS.exists():
        pytest.skip("macros file missing")
    text = MACROS.read_text()
    import re

    for m in re.finditer(r"\\newcommand\{\\([^}]*)\}", text):
        name = m.group(1)
        assert "_" not in name and not any(
            ch.isdigit() for ch in name
        ), f"macro name {name!r} contains digit or underscore (invalid LaTeX command name)"
