import os
import pytest
import numpy as np

import _common as C
from kbound_edge.reporting import compile_latex_macros


def test_compile_latex_macros():
    cfg = C.load_config("edge_real_phone_v1.yaml")
    results_dir = os.path.normpath(os.path.join(C.EDGE_ROOT, cfg["paths"]["results_dir"]))
    
    # Check if files exist, if not skip test
    required_files = [
        "heldout_metrics.json",
        "replication_metrics.json",
        "ablation_results.json",
        "runtime_profile.json",
        "anti_leakage_audit.json",
        "recording_inventory.json",
        "calibration_summary.json"
    ]
    for rf in required_files:
        if not os.path.exists(os.path.join(results_dir, rf)):
            pytest.skip(f"Missing required result file: {rf}")
            
    heldout_metrics = C.load_json(os.path.join(results_dir, "heldout_metrics.json"))
    replication_metrics = C.load_json(os.path.join(results_dir, "replication_metrics.json"))
    ablation_results = C.load_json(os.path.join(results_dir, "ablation_results.json"))
    runtime_profile = C.load_json(os.path.join(results_dir, "runtime_profile.json"))
    anti_leakage_audit = C.load_json(os.path.join(results_dir, "anti_leakage_audit.json"))
    recording_inventory = C.load_json(os.path.join(results_dir, "recording_inventory.json"))
    calibration_summary = C.load_json(os.path.join(results_dir, "calibration_summary.json"))

    from kbound_edge.logging import read_jsonl
    log_path = C.resolve(cfg["paths"]["heldout_log"])
    records = read_jsonl(log_path)
    
    # Load true labels
    from kbound_edge.real_dataset import load_window
    windows_dir = C.resolve(cfg["paths"]["windows_dir"])
    split_dir = os.path.join(windows_dir, "heldout")
    files = sorted([f for f in os.listdir(split_dir) if not f.startswith(".") and f.endswith(".npz") and (f.startswith("S07_") or f.startswith("S08_"))])
    
    true_labels = []
    for fname in files:
        _, off_load = load_window(os.path.join(split_dir, fname))
        true_labels.append(off_load["labels"])

    from kbound_edge.real_manifest import expected_windows
    win_meta_map = {}
    for s_id in ["S07", "S08"]:
        for w in expected_windows(cfg, s_id):
            w_copy = dict(w)
            w_copy["session_id"] = s_id
            win_meta_map[w["window_id"]] = w_copy

    macros = compile_latex_macros(
        cfg=cfg,
        results_dir=results_dir,
        heldout_metrics=heldout_metrics,
        replication_metrics=replication_metrics,
        ablation_results=ablation_results,
        runtime_profile=runtime_profile,
        anti_leakage_audit=anti_leakage_audit,
        recording_inventory=recording_inventory,
        calibration_summary=calibration_summary,
        records=records,
        win_meta_map=win_meta_map,
        true_labels=true_labels,
    )
    
    assert isinstance(macros, dict)
    assert len(macros) >= 190
    assert "CameraROneFitSessions" in macros
    assert "CameraRTwoBalAccKgaFull" in macros
    assert "CameraRThreeWindowsMildLight" in macros
    assert "CameraSOneSessionSourceTrain" in macros
    assert "CameraSTwoCheckOne" in macros
    assert "CameraSThreeHOneDelta" in macros
    assert "CameraSFourMeanFullWindow" in macros
    assert "CameraSFiveRegretFullKga" in macros
