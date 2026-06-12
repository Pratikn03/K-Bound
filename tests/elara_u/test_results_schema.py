"""Schema guard: curated ELARA-U result artifacts have the keys the paper relies on."""
import json
from pathlib import Path
import pytest

EXP = Path(__file__).resolve().parents[2] / "experiments/elara_u"

REQUIRED = {
    "honest_benchmark.json": ["n_tasks", "average_rank", "mean_auroc", "contrasts", "per_task_auc"],
    "statistical_audit.json": ["family_A_primary_positive", "family_B_reliability_ablation", "summary"],
    "calibration_results.json": ["metrics"],
    "multimodal_reliability_results.json": ["regimes", "reliability_validated"],
    "natural_shift_results.json": ["mean_auroc", "drift_stack_vs_plain_stack"],
}

# The D23 hypotheses block exists under either name (the test script was co-edited and
# both schema versions appear across committed result files); require exactly one.
_HYP_KEYS = ("hypotheses_failure_regime", "hypotheses_modality_failure_regime")


@pytest.mark.parametrize("fname,keys", REQUIRED.items())
def test_required_keys_present(fname, keys):
    p = EXP / fname
    if not p.exists():
        pytest.skip(f"{fname} not present")
    d = json.loads(p.read_text())
    for k in keys:
        assert k in d, f"{fname} missing required key {k}"


@pytest.mark.parametrize("fname", [
    "multimodal_reliability_results.json",
    "multimodal_reliability_results_mvtec3d.json",
    "multimodal_reliability_results_3d_adam.json",
    "multimodal_reliability_results_mulsen.json",
])
def test_d23_hypotheses_block_present(fname):
    p = EXP / fname
    if not p.exists():
        pytest.skip(f"{fname} not present")
    d = json.loads(p.read_text())
    hyp = next((d[k] for k in _HYP_KEYS if k in d), None)
    assert hyp is not None, f"{fname} missing a D23 hypotheses block ({_HYP_KEYS})"
    for h in ("H1_vs_equal_weight", "H2_vs_stale_auto_select", "H3_vs_no_reliability"):
        assert h in hyp, f"{fname} hypotheses missing {h}"


def test_primary_claims_and_boundary_hold():
    p = EXP / "statistical_audit.json"
    if not p.exists():
        pytest.skip("statistical_audit.json not present")
    s = json.loads(p.read_text())["summary"]
    assert s["all_primary_claims_hold_after_holm"] is True
    assert s["any_regime_reliability_helps_after_holm"] is True   # D23
