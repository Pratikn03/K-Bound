from unified_result_audit import (
    ci_robust_beats_both_from_comparisons,
    ci_robust_from_row,
    extract_regrets,
    point_beats_both,
    summarize_result_row,
)


def test_point_beats_both_uses_strict_lower_regret():
    assert point_beats_both(0.002, 0.04, 0.01)
    assert not point_beats_both(0.0, 0.25, 0.0)


def test_extract_regrets_supports_direct_and_decisive_tta_schemas():
    direct = {"regret_kga": 0.1, "regret_adapt": 0.2, "regret_freeze": 0.3}
    assert extract_regrets(direct) == (0.1, 0.2, 0.3)

    decisive = {
        "regret_vs_oracle": {
            "K_Bound": 0.01,
            "always_adapt": 0.06,
            "always_freeze": 0.03,
        }
    }
    assert extract_regrets(decisive) == (0.01, 0.06, 0.03)


def test_ci_robust_beats_both_requires_both_comparisons_to_survive():
    comparisons = [
        {"candidate": "tent", "trivial": "always-adapt", "survives_holm": True},
        {"candidate": "tent", "trivial": "always-freeze", "survives_holm": True},
        {"candidate": "sar", "trivial": "always-adapt", "survives_holm": False},
        {"candidate": "sar", "trivial": "always-freeze", "survives_holm": True},
    ]
    assert ci_robust_beats_both_from_comparisons(comparisons, "tent")
    assert not ci_robust_beats_both_from_comparisons(comparisons, "sar")


def test_summary_keeps_point_win_separate_from_ci_win():
    row = summarize_result_row(
        dataset="officehome_M_v2",
        artifact="dummy.json",
        location="$.test_locked",
        row={"regret_kga": 0.002, "regret_adapt": 0.04, "regret_freeze": 0.01},
    )
    assert row["point_beats_both"] is True
    assert row["ci_robust_beats_both"] == "unknown"


def test_paper_source_zero_ci_lower_bound_is_not_ci_robust():
    assert ci_robust_from_row({"ci_vs_adapt": [0.004, 0.062], "ci_vs_freeze": [0.0, 0.0003]}) == "False"


def test_paper_source_ci_robust_verdict_is_ci_robust():
    assert ci_robust_from_row({"verdict": "beats-both-CI-robust"}) == "True"


def test_explicit_ci_robust_field_takes_precedence():
    assert ci_robust_from_row({"ci_robust_beats_both": False, "verdict": "beats-both-CI-robust"}) == "False"
    assert ci_robust_from_row({"ci_robust_beats_both": None}) == "unknown"
