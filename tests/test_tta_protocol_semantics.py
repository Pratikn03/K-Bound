from __future__ import annotations

from pathlib import Path

import pytest

from experiments.kbound.wilds import tta_methods as tm


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("mode", "gradient_reads_eval"),
    [("online", False), ("episodic", True)],
)
def test_gradient_tta_contract_discloses_transductive_evaluation(
    mode: str, gradient_reads_eval: bool
) -> None:
    contract = tm.tta_protocol_contract(mode)
    assert contract["schema"] == "kbound_tta_candidate_protocol_v1"
    assert contract["candidate_evaluation_is_transductive"] is True
    assert contract["candidate_adaptation_eval_disjoint"] is False
    assert contract["prediction_uses_eval_batch_statistics"] is True
    assert contract["gradient_update_reads_eval_x"] is gradient_reads_eval
    assert contract["target_labels_used_for_adaptation_or_prediction"] is False


@pytest.mark.parametrize("kind", ["labelshift", "conservative"])
def test_stream_prior_contract_is_inductive_and_label_free(kind: str) -> None:
    contract = tm.stream_prior_protocol_contract(kind)
    assert contract["candidate_evaluation_is_transductive"] is False
    assert contract["candidate_adaptation_eval_disjoint"] is True
    assert contract["prediction_uses_eval_batch_statistics"] is False
    assert contract["target_labels_used_for_adaptation_or_prediction"] is False


@pytest.mark.parametrize(
    "relative_path",
    [
        "experiments/kbound/wilds/run_camelyon17_kbound.py",
        "experiments/kbound/wilds/run_geoshift_kbound.py",
        "experiments/kbound/wilds/run_imagenetr_kbound.py",
        "experiments/kbound/wilds/run_iwildcam_kbound.py",
        "experiments/kbound/wilds/run_rxrx1_kbound.py",
        "experiments/kbound/officehome/run_officehome_kbound.py",
    ],
)
def test_natural_runner_persists_candidate_protocol(relative_path: str) -> None:
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    assert "tta_protocol" in source
    assert "tta_protocol_contract" in source or "stream_prior_protocol_contract" in source


def test_live_manuscripts_disclose_transductive_candidate_semantics() -> None:
    for relative_path in (
        "docs/research/kbound/kbound_submission_body.tex",
        "docs/research/kbound/kbound_short_body.tex",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "candidate TTA" in text
        assert "transductive" in text
        assert "evaluation-batch BatchNorm statistics" in text
