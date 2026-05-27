"""Phase 2.2A — the Family-A analysis driver must compare RGA+ only
against static_attention, not against any other comparator."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "src" / "scripts" / "run_phase2_family_a_analysis.py"


def _source() -> str:
    return DRIVER.read_text()


def test_analysis_driver_hard_codes_static_attention_as_only_comparator():
    src = _source()
    # The driver must mention static_attention as the primary comparator
    assert '"static_attention"' in src or "'static_attention'" in src
    # The driver must not iterate over a list of comparators that includes
    # late_fusion_ensemble / random_forest / tent / sar / eata / ttt etc.
    forbidden_comparator_iter = [
        "late_fusion_ensemble",
        "random_forest",
        "tent_score_adapter",
        "sar_score_adapter",
        "eata_score_adapter",
        "ttt_pseudo_label_adapter",
        "confidence_weighted_mean",
        "early_fusion_mlp",
        "craf_attention",
    ]
    for c in forbidden_comparator_iter:
        # the analysis driver must NOT loop over these comparators
        assert c not in src, f"Family-A analysis driver references {c!r}; primary surface allows only static_attention"


def test_analysis_driver_uses_k_equals_5_holm():
    src = _source()
    assert "K=5" in src or "K = 5" in src
    assert "holm_bonferroni" in src
    assert ("K=5" in src and "holm_bonferroni" in src) or "holm_bonferroni" in src.replace("K=5", "")


def test_analysis_driver_writes_v2_primary_paths():
    src = _source()
    assert "family_a_v2_primary_cell_level_raw.csv" in src
    assert "family_a_v2_primary_cell_level_holm_k5.csv" in src


def test_analysis_driver_does_not_overwrite_historical_pilot_paths():
    src = _source()
    assert (
        "family_a_powered_ensemble_inference.csv"
        not in src.replace(
            "family_a_powered_ensemble_inference.csv\n",
            "",
        )
        or "Does NOT overwrite" in src
    ), "analysis driver must not write to family_a_powered_ensemble_inference.csv"
    assert (
        "family_a_powered_holm_results.csv"
        not in src.replace(
            "family_a_powered_holm_results.csv\n",
            "",
        )
        or "Does NOT overwrite" in src
    )
