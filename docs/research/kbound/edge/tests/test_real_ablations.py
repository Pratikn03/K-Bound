import pytest
import os
import sys
import importlib

# Put scripts on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
m = importlib.import_module("09_run_real_ablations")
ABLATIONS = m.ABLATIONS
ABLATION_FEATURES = m.ABLATION_FEATURES


def test_ablation_registry_is_locked():
    assert tuple(ABLATIONS) == (
        "full_kga", "no_radius", "no_blur_brightness",
        "no_disagreement", "confidence_only", "entropy_only",
    )


def test_ablation_features_contain_subset_of_names():
    from kbound_edge.evidence import EDGE_EVIDENCE_NAMES
    
    for variant, features in ABLATION_FEATURES.items():
        assert len(features) > 0, f"Variant {variant} has no features"
        for f in features:
            assert f in EDGE_EVIDENCE_NAMES, f"Feature {f} used in {variant} is not in EDGE_EVIDENCE_NAMES"
