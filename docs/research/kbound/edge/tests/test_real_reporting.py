"""Synthetic report computations, not evidence of a physical study.

No saved metrics, logs, NPZs, or real labels are opened. Physical publication and
custody remain a separate authorized workflow; passing tests cannot establish them.
"""

from copy import deepcopy

import numpy as np
import pytest
from kbound_edge.reporting import compile_latex_macros


@pytest.fixture(params=["top_level", "extra"])
def report_inputs(request, tmp_path):
    # A small unit-test protocol, not a registered study artifact: one invented
    # object per session; even sessions include 32 derived mixed windows.
    cfg = {
        "classes": ["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d"],
        "window_size": 32,
        "sessions": {
            f"S{i:02d}": {"objects": [f"synthetic-object-{i}"], "windows": n}
            for i, n in enumerate([20, 20, 16, 44, 16, 44, 32, 56, 32, 56], start=1)
        },
    }
    bootstrap = {
        "balanced_acc": {"val": 0.8123, "ci": [0.7, 0.9]},
        "macro_f1": {"val": 0.7234, "ci": [0.6, 0.8]},
        "mean_regret": {"val": 0.01234, "ci": [0.00123, 0.02345]},
        "false_adapt_uncond": {"val": 0.02567, "ci": [0.01, 0.04]},
        "false_adapt_cond": {"val": 0.10234, "ci": [0.05, 0.2]},
        "adapt_rate": {"val": 0.375, "ci": [0.25, 0.5]},
        "abstain_rate": {"val": 0.125, "ci": [0.1, 0.15]},
        "mean_latency": {"val": 12.34, "ci": [10.12, 14.56]},
    }
    policies = {
        name: deepcopy(bootstrap)
        for name in (
            "always_freeze",
            "always_adapt",
            "confidence_gate",
            "entropy_gate",
            "kga_no_radius",
            "kga_full",
        )
    }
    policies["always_freeze"]["balanced_acc"] = {"val": 0.5, "ci": [0.4, 0.6]}
    ablations = {
        name: {
            "regret": 0.01234,
            "false_adapt_uncond": 0.02567,
            "adapt_rate": 0.375,
            "abstain_rate": 0.125,
            "eps": 0.23456,
        }
        for name in (
            "full_kga",
            "no_radius",
            "no_blur_brightness",
            "no_disagreement",
            "confidence_only",
            "entropy_only",
        )
    }
    ablations["no_radius"]["regret"] = 0.56789
    runtime = {
        name: {"mean_ms": 12.34, "p95_ms": 23.46}
        for name in (
            "frozen_inference",
            "tent_update",
            "candidate_inference",
            "evidence",
            "gate",
            "end_to_end",
            "capture_preprocess",
        )
    }
    runtime["end_to_end"] = {"mean_ms": 43.21, "p95_ms": 50.04}
    runtime["metadata"] = {"rss_mem_before_mb": 123.456}

    # Four invented 32-frame windows. Hand-counted (frozen, candidate) accuracies:
    # mild_light=(1/2,1), glare=(1,1/2), shadow=(3/4,1/4), background=(1,1/2).
    frozen = ([0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, 1])
    candidate = ([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, 1])
    shifts = ("mild_light", "glare", "side_shadow", "new_background")
    decisions = ("adapt", "freeze", "abstain", "adapt")
    records = []
    metadata = {}
    for i in range(4):
        window_id = f"synthetic-W{i + 1:03d}"
        row = {"window_id": window_id, "frozen_pred": list(frozen[i]) * 8, "decision": decisions[i]}
        if request.param == "top_level":
            row["shadow_candidate_pred"] = list(candidate[i]) * 8
        else:
            row["extra"] = {"shadow_candidate_pred": list(candidate[i]) * 8}
        records.append(row)
        metadata[window_id] = {"shift_id": shifts[i]}
    return {
        "cfg": cfg,
        "results_dir": str(tmp_path),
        "heldout_metrics": {"bootstrap_results": policies},
        "replication_metrics": {},
        "ablation_results": ablations,
        "runtime_profile": runtime,
        "anti_leakage_audit": {
            "checks": [{"passed": value} for value in (True, False, True, True, False, True, True, False)]
        },
        "recording_inventory": {},
        "calibration_summary": {},
        "records": records,
        "win_meta_map": metadata,
        "true_labels": [np.array([0, 0, 1, 1] * 8) for _ in range(4)],
    }


def test_compile_latex_macros_formats_policy_statistics(report_inputs):
    macros = compile_latex_macros(**report_inputs)
    assert len(macros) == 193
    assert macros["CameraROneFitSessions"] == "S03, S04"
    assert macros["CameraROneConformalSessions"] == "S05, S06"
    assert macros["CameraROneHeldoutSessions"] == "S07, S08"
    assert macros["CameraRTwoBalAccAlwaysFreeze"] == r"50.0\% [40.0, 60.0]"
    assert macros["CameraRTwoBalAccKgaFull"] == r"81.2\% [70.0, 90.0]"
    assert macros["CameraRTwoMacroFKgaFull"] == r"72.3\% [60.0, 80.0]"
    assert macros["CameraRTwoRegretKgaFull"] == "0.0123 [0.0012, 0.0234]"
    assert macros["CameraRTwoFAuKgaFull"] == "0.0257 [0.0100, 0.0400]"
    assert macros["CameraRTwoFAcKgaFull"] == "0.1023 [0.0500, 0.2000]"
    assert macros["CameraRTwoAdaptRateKgaFull"] == "0.375 [0.250, 0.500]"
    assert macros["CameraRTwoAbstainRateKgaFull"] == "0.125 [0.100, 0.150]"
    assert macros["CameraRTwoLatencyKgaFull"] == "12.3 [10.1, 14.6]"


def test_compile_latex_macros_computes_regret_and_decisions(report_inputs):
    macros = compile_latex_macros(**report_inputs)
    # Benefits +1/2 and -1/2: always-adapt loses 1/2 on the second window,
    # whereas adapting only on the first incurs zero regret.
    assert macros["CameraRThreeWindowsMildLight"] == "2"
    assert macros["CameraRThreeAlwaysAdaptRegretMildLight"] == "0.2500"
    assert macros["CameraRThreeKgaRegretMildLight"] == "0.0000"
    assert macros["CameraRThreeDecisionPatternMildLight"] == "1/1/0"
    assert macros["CameraRThreeAlwaysAdaptRegretSideShadow"] == "0.5000"
    assert macros["CameraRThreeKgaRegretSideShadow"] == "0.0000"
    assert macros["CameraRThreeDecisionPatternSideShadow"] == "0/0/1"
    # This invented adaptation is harmful: the report must not round it to safety.
    assert macros["CameraRThreeAlwaysAdaptRegretNewBackground"] == "0.5000"
    assert macros["CameraRThreeKgaRegretNewBackground"] == "0.5000"
    assert macros["CameraRThreeDecisionPatternNewBackground"] == "1/0/0"


@pytest.mark.parametrize(
    ("shift", "expected"),
    [
        ("HOne", ("+0.5000", "Adapt", "0.5000", "1.0000", "Adapt", "Yes", "0.0000")),
        ("HTwo", ("-0.5000", "Freeze", "0.7500", "0.2500", "Abstain", "Yes", "0.0000")),
        ("HFour", ("-0.5000", "Freeze", "1.0000", "0.5000", "Adapt", "No", "0.5000")),
    ],
)
def test_compile_latex_macros_computes_shift_examples(report_inputs, shift, expected):
    macros = compile_latex_macros(**report_inputs)
    suffixes = ("Delta", "Oracle", "Freeze", "Adapt", "Kga", "Correct", "Regret")
    assert tuple(macros[f"CameraSThree{shift}{suffix}"] for suffix in suffixes) == expected


def test_compile_latex_macros_keeps_missing_shifts_pending(report_inputs):
    macros = compile_latex_macros(**report_inputs)
    for shift in ("MotionBlur", "BatchComposition"):
        assert macros[f"CameraRThreeWindows{shift}"] == "0"
        assert macros[f"CameraRThreeAlwaysAdaptRegret{shift}"] == r"\CamPending"
        assert macros[f"CameraRThreeKgaRegret{shift}"] == r"\CamPending"
        assert macros[f"CameraRThreeDecisionPattern{shift}"] == "0/0/0"
    for suffix in ("Delta", "Oracle", "Freeze", "Adapt", "Kga", "Correct", "Regret"):
        assert macros[f"CameraSThreeHThree{suffix}"] == r"\CamPending"


def test_compile_latex_macros_exports_inventory_audit_runtime_and_ablations(report_inputs):
    macros = compile_latex_macros(**report_inputs)
    assert macros["CameraSOneWindowsSourceTrain"] == "20"
    assert macros["CameraSOneFramesSourceTrain"] == "640"
    assert macros["CameraSOneLabelsSourceTrain"] == "5/5/5/5"
    assert macros["CameraSOneWindowsCalibrationFit"] == "60"
    assert macros["CameraSOneFramesCalibrationFit"] == "1,920"
    assert macros["CameraSOneLabelsCalibrationFit"] == "7/7/7/7 + 32 mixed"
    assert macros["CameraSOneWindowsHeldoutTest"] == "88"
    assert macros["CameraSOneFramesHeldoutTest"] == "2,816"
    assert macros["CameraSOneLabelsHeldoutTest"] == "14/14/14/14 + 32 mixed"
    assert macros["CameraSTwoCheckOne"] == "PASS"
    assert macros["CameraSTwoCheckTwo"] == "FAIL"
    assert macros["CameraSTwoCheckEight"] == "FAIL"
    assert macros["CameraSFourMeanFrozenInference"] == "12.3"
    assert macros["CameraSFourPNinetyFiveFrozenInference"] == "23.5"
    assert macros["CameraSFourMeanFullWindow"] == "43.2"
    assert macros["CameraSFourPNinetyFiveFullWindow"] == "50.0"
    assert macros["CameraSFourMemoryFullWindow"] == "123.5"
    assert macros["CameraSFiveRegretFullKga"] == "0.0123"
    assert macros["CameraSFiveRegretNoRadius"] == "0.5679"
    assert macros["CameraSFiveFAuFullKga"] == "0.0257"
    assert macros["CameraSFiveAdaptFullKga"] == "0.375"
    assert macros["CameraSFiveAbstainFullKga"] == "0.125"
    assert macros["CameraSFiveEpsFullKga"] == "0.2346"


def test_compile_latex_macros_rejects_inconsistent_declared_inventory(report_inputs):
    report_inputs["cfg"]["sessions"]["S01"]["windows"] = 21
    with pytest.raises(ValueError, match="generator produced 20 windows, lock declares 21"):
        compile_latex_macros(**report_inputs)


@pytest.mark.parametrize("missing", ["bootstrap_results", "kga_full", "candidate_prediction"])
def test_compile_latex_macros_does_not_issue_macros_with_missing_required_inputs(report_inputs, missing):
    if missing == "bootstrap_results":
        report_inputs["heldout_metrics"] = {}
    elif missing == "kga_full":
        del report_inputs["heldout_metrics"]["bootstrap_results"]["kga_full"]
    else:
        row = report_inputs["records"][0]
        row.pop("shadow_candidate_pred", None)
        row["extra"] = {}
    with pytest.raises(KeyError):
        compile_latex_macros(**report_inputs)
