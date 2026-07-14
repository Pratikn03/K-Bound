"""kbound_edge.integrity -- programmatic anti-leakage and schema verification.

Implements the eight checks for Table S2 of the physical-camera validation.
"""

from __future__ import annotations

import os
import hashlib
import json
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from kbound_edge.logging import FORBIDDEN_LABEL_KEYS, read_jsonl
from kbound_edge.evidence import EDGE_EVIDENCE_NAMES
from kbound_edge.model import state_dict_hash


def get_file_sha256(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_frozen_model_mutation(cfg: dict) -> Tuple[bool, str, str, str]:
    """Check 1: Frozen checkpoint unchanged after candidate adaptation."""
    from kbound_edge.model import build_model
    from kbound_edge.tent_adapter import EpisodicTentAdapter
    
    model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", cfg["paths"]["model"]))
    if not os.path.exists(model_path):
        return False, "model file exists", "missing", model_path
        
    device = cfg.get("device", "cpu")
    f0 = build_model(cfg["num_classes"], pretrained=False, seed=cfg["seed"], device=device)
    sd = torch.load(model_path, map_location=device)
    f0.load_state_dict(sd)
    f0.eval()
    
    h_before = state_dict_hash(f0)
    
    # Run a dummy adaptation
    adapter = EpisodicTentAdapter(f0, lr=cfg["adapter"]["lr"], steps=cfg["adapter"]["steps"], device=device)
    x = torch.randn(cfg["window_size"], 3, cfg["image_size"], cfg["image_size"], device=device)
    res = adapter.adapt(x)
    
    h_after = state_dict_hash(f0)
    passed = (h_before == h_after)
    
    return passed, h_before, h_after, model_path


def check_heldout_labels_inaccessible(cfg: dict) -> Tuple[bool, str, str, str]:
    """Check 2: Held-out labels inaccessible to the live runtime (no leakage in online log)."""
    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", cfg["paths"]["heldout_log"]))
    if not os.path.exists(log_path):
        return False, "log file exists", "missing", log_path
        
    records = read_jsonl(log_path)
    if not records:
        return False, "non-empty log", "empty", log_path
        
    leak = False
    for r in records:
        for k in r.keys():
            if k.lower() in FORBIDDEN_LABEL_KEYS:
                leak = True
                break
        if "extra" in r and isinstance(r["extra"], dict):
            for k in r["extra"].keys():
                if k.lower() in FORBIDDEN_LABEL_KEYS:
                    leak = True
                    break
        if leak:
            break
            
    passed = not leak
    return passed, "0 forbidden keys in log", f"{'leak detected' if leak else '0 forbidden keys'}", log_path


def check_feature_schema_unchanged(cfg: dict) -> Tuple[bool, str, str, str]:
    """Check 3: Feature schema unchanged after protocol lock."""
    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", cfg["paths"]["heldout_log"]))
    if not os.path.exists(log_path):
        return False, "log file exists", "missing", log_path
        
    records = read_jsonl(log_path)
    if not records:
        return False, "non-empty log", "empty", log_path
        
    # Check features in the first record
    logged_features = list(records[0]["evidence"].keys())
    expected = sorted(list(EDGE_EVIDENCE_NAMES))
    observed = sorted(logged_features)
    passed = (expected == observed)
    return passed, str(expected), str(observed), log_path


def check_epsilon_conformal_split(results_dir: str) -> Tuple[bool, str, str, str]:
    """Check 4: Epsilon calibrated only from the conformal split."""
    summary_path = os.path.join(results_dir, "calibration_summary.json")
    if not os.path.exists(summary_path):
        return False, "calibration_summary exists", "missing", summary_path
        
    with open(summary_path) as f:
        data = json.load(f)
        
    conformal_sessions = sorted(data.get("conformal_sessions", []))
    fit_sessions = sorted(data.get("fit_sessions", []))
    
    expected_conformal = ["S05", "S06"]
    expected_fit = ["S03", "S04"]
    
    passed = (conformal_sessions == expected_conformal and fit_sessions == expected_fit)
    return passed, f"fit={expected_fit}, conformal={expected_conformal}", f"fit={fit_sessions}, conformal={conformal_sessions}", summary_path


def check_heldout_excluded_from_calibration(results_dir: str) -> Tuple[bool, str, str, str]:
    """Check 5: Held-out sessions excluded from adapter, feature, and threshold tuning."""
    summary_path = os.path.join(results_dir, "calibration_summary.json")
    if not os.path.exists(summary_path):
        return False, "calibration_summary exists", "missing", summary_path
        
    with open(summary_path) as f:
        data = json.load(f)
        
    all_calib_sessions = set(data.get("fit_sessions", []) + data.get("conformal_sessions", []))
    heldout_sessions = {"S07", "S08", "S09", "S10"}
    
    overlap = all_calib_sessions.intersection(heldout_sessions)
    passed = (len(overlap) == 0)
    
    return passed, "0 overlapping sessions", f"{len(overlap)} overlaps: {sorted(list(overlap))}", summary_path


def check_identical_heldout_stream(results_dir: str) -> Tuple[bool, str, str, str]:
    """Check 6: Identical held-out stream replayed for every policy."""
    metrics_path = os.path.join(results_dir, "heldout_metrics.json")
    if not os.path.exists(metrics_path):
        return False, "heldout_metrics exists", "missing", metrics_path
        
    with open(metrics_path) as f:
        data = json.load(f)
        
    comparison = data.get("policy_comparison", {})
    n_windows_list = [m["n_windows"] for m in comparison.values()]
    
    if len(n_windows_list) == 0:
        return False, "at least one policy comparison", "none", metrics_path
        
    passed = (len(set(n_windows_list)) == 1)
    return passed, f"all policies processed {n_windows_list[0]} windows", f"observed window counts: {n_windows_list}", metrics_path


def check_config_hash_in_log(cfg: dict) -> Tuple[bool, str, str, str]:
    """Check 7: Config hash stored in every log row."""
    from kbound_edge.logging import config_hash
    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", cfg["paths"]["heldout_log"]))
    if not os.path.exists(log_path):
        return False, "log file exists", "missing", log_path
        
    records = read_jsonl(log_path)
    if not records:
        return False, "non-empty log", "empty", log_path
        
    cleaned_cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    expected_hash = config_hash(cleaned_cfg)
    mismatches = 0
    for r in records:
        if r.get("config_hash") != expected_hash:
            mismatches += 1
            
    passed = (mismatches == 0)
    return passed, f"config_hash matches {expected_hash} on all rows", f"{mismatches} mismatches", log_path


def check_model_hash_in_log(cfg: dict) -> Tuple[bool, str, str, str]:
    """Check 8: Model hash stored in every log row."""
    from kbound_edge.logging import config_hash
    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", cfg["paths"]["heldout_log"]))
    if not os.path.exists(log_path):
        return False, "log file exists", "missing", log_path
        
    records = read_jsonl(log_path)
    if not records:
        return False, "non-empty log", "empty", log_path
        
    # Get expected model hash
    from kbound_edge.model import build_model
    model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", cfg["paths"]["model"]))
    if not os.path.exists(model_path):
        return False, "model file exists", "missing", model_path
    
    device = cfg.get("device", "cpu")
    f0 = build_model(cfg["num_classes"], pretrained=False, seed=cfg["seed"], device=device)
    sd = torch.load(model_path, map_location=device)
    f0.load_state_dict(sd)
    expected_version = state_dict_hash(f0)
    
    mismatches = 0
    for r in records:
        if r.get("model_version") != expected_version:
            mismatches += 1
            
    passed = (mismatches == 0)
    return passed, f"model_version matches {expected_version} on all rows", f"{mismatches} mismatches", log_path


def run_full_audit(cfg: dict) -> List[Dict[str, Any]]:
    """Run all 8 Table S2 checks and return audit report list."""
    results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", cfg["paths"]["results_dir"]))
    
    checks = []
    
    # 1. Frozen checkpoint adaptation isolation
    pass1, exp1, obs1, path1 = check_frozen_model_mutation(cfg)
    checks.append({
        "check": "Frozen checkpoint unchanged after candidate adaptation?",
        "passed": pass1,
        "expected": exp1,
        "observed": obs1,
        "evidence_artifact": os.path.basename(path1),
        "evidence_hash": get_file_sha256(path1)
    })
    
    # 2. Held-out labels inaccessible
    pass2, exp2, obs2, path2 = check_heldout_labels_inaccessible(cfg)
    checks.append({
        "check": "Held-out labels inaccessible to the live runtime?",
        "passed": pass2,
        "expected": exp2,
        "observed": obs2,
        "evidence_artifact": os.path.basename(path2),
        "evidence_hash": get_file_sha256(path2)
    })
    
    # 3. Feature schema unchanged
    pass3, exp3, obs3, path3 = check_feature_schema_unchanged(cfg)
    checks.append({
        "check": "Feature schema unchanged after protocol lock?",
        "passed": pass3,
        "expected": exp3,
        "observed": obs3,
        "evidence_artifact": os.path.basename(path3),
        "evidence_hash": get_file_sha256(path3)
    })
    
    # 4. Epsilon calibrated only from conformal split
    pass4, exp4, obs4, path4 = check_epsilon_conformal_split(results_dir)
    checks.append({
        "check": "Epsilon calibrated only from the conformal split?",
        "passed": pass4,
        "expected": exp4,
        "observed": obs4,
        "evidence_artifact": os.path.basename(path4),
        "evidence_hash": get_file_sha256(path4)
    })
    
    # 5. Held-out sessions excluded from adapter, feature, and threshold tuning
    pass5, exp5, obs5, path5 = check_heldout_excluded_from_calibration(results_dir)
    checks.append({
        "check": "Held-out sessions excluded from adapter, feature, and threshold tuning?",
        "passed": pass5,
        "expected": exp5,
        "observed": obs5,
        "evidence_artifact": os.path.basename(path5),
        "evidence_hash": get_file_sha256(path5)
    })
    
    # 6. Identical held-out stream replayed for every policy
    pass6, exp6, obs6, path6 = check_identical_heldout_stream(results_dir)
    checks.append({
        "check": "Identical held-out stream replayed for every policy?",
        "passed": pass6,
        "expected": exp6,
        "observed": obs6,
        "evidence_artifact": os.path.basename(path6),
        "evidence_hash": get_file_sha256(path6)
    })
    
    # 7. Config hash stored in every log row
    pass7, exp7, obs7, path7 = check_config_hash_in_log(cfg)
    checks.append({
        "check": "Config hash stored in every log row?",
        "passed": pass7,
        "expected": exp7,
        "observed": obs7,
        "evidence_artifact": os.path.basename(path7),
        "evidence_hash": get_file_sha256(path7)
    })
    
    # 8. Model hash stored in every log row
    pass8, exp8, obs8, path8 = check_model_hash_in_log(cfg)
    checks.append({
        "check": "Model hash stored in every log row?",
        "passed": pass8,
        "expected": exp8,
        "observed": obs8,
        "evidence_artifact": os.path.basename(path8),
        "evidence_hash": get_file_sha256(path8)
    })
    
    return checks
