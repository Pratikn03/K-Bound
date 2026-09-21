import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts"))


def module():
    return importlib.import_module("domainnet_feasibility_score")


def packet(episode=0):
    return {
        "episode": episode,
        "sample_ids": [f"painting/a/{episode}_{i}.jpg" for i in range(4096)],
        "frozen": [0] * 4096,
        "candidate_direct": [1] * 4096,
        "candidate_refined": [2] * 4096,
    }


def test_helpful_harmful_ties_and_conditional_interval():
    m = module()
    p = packet()
    truth = dict.fromkeys(p["sample_ids"], 1)
    out = m.score_episode(p, truth)
    assert out["direct"]["benefit"] == 1 and out["refined"]["benefit"] == 0
    assert out["direct"]["improvements"] == 4096 and out["direct"]["degradations"] == 0
    assert out["direct"]["interval"][1] == 1
    assert math.isclose(out["sampling_halfwidth"], 0.0517309696824647)
    bad = m.score_episode(p, dict.fromkeys(p["sample_ids"], 0))
    assert bad["direct"]["benefit"] == -1 and bad["direct"]["interval"][0] == -1


@pytest.mark.parametrize("fault", ["missing", "nan", "bool", "repeat", "out_of_class"])
def test_invalid_prediction_or_truth_is_not_silently_dropped(fault):
    m = module()
    p = packet()
    truth = dict.fromkeys(p["sample_ids"], 0)
    if fault == "missing":
        truth.pop(p["sample_ids"][0])
    if fault == "nan":
        p["candidate_direct"][0] = float("nan")
    if fault == "bool":
        p["frozen"][0] = True
    if fault == "repeat":
        p["sample_ids"][0] = p["sample_ids"][1]
    if fault == "out_of_class":
        truth[p["sample_ids"][0]] = 126
    with pytest.raises(ValueError):
        m.score_episode(p, truth)


def test_all_three_required_and_no_prospective_or_routing_promotion():
    m = module()
    scores = []
    for e in range(3):
        p = packet(e)
        scores.append(m.score_episode(p, dict.fromkeys(p["sample_ids"], 1)))
    result = m.summarize_episodes(scores)
    assert result["development_support"] is True
    assert result["independent_source_models"] == 1 and result["domains"] == 1
    assert result["prospective"] is False and result["kga_routing_evaluated"] is False
    assert result["coverage_after_screening_established"] is False
    for score in result["scores"]:
        assert score["coverage_after_screening_established"] is False
        assert "completion screening" in score["interval_qualification"]
    for bad in [scores[:2], [scores[0], scores[0], scores[2]]]:
        with pytest.raises(ValueError):
            m.summarize_episodes(bad)
