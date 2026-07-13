"""test_no_live_labels -- ground-truth labels can never reach the online path.

The runtime guards every online window payload and the logger guards every
record; a label-like key raises LabelLeakError instead of being processed/written.
"""

import numpy as np
import pytest

from kbound_edge.logging import assert_no_labels, LabelLeakError, WindowLogger, FORBIDDEN_LABEL_KEYS
from kbound_edge.replay import run_window


class TestAssertNoLabels:
    @pytest.mark.parametrize("key", sorted(FORBIDDEN_LABEL_KEYS))
    def test_each_forbidden_key_raises(self, key):
        with pytest.raises(LabelLeakError):
            assert_no_labels({key: [1, 2, 3]})

    def test_case_insensitive(self):
        with pytest.raises(LabelLeakError):
            assert_no_labels({"LABELS": [1]})
        with pytest.raises(LabelLeakError):
            assert_no_labels({"Ground_Truth": 0})

    def test_nested_payload(self):
        with pytest.raises(LabelLeakError):
            assert_no_labels({"frames": np.zeros(3), "meta": {"y_true": [1]}})
        with pytest.raises(LabelLeakError):
            assert_no_labels({"frames": [{"target": 2}]})

    def test_clean_payload_ok(self):
        # frozen_pred is a model OUTPUT, not a label -> allowed
        assert_no_labels({"frames": np.zeros(3), "window_id": 0,
                          "decision": "adapt", "frozen_pred": [0, 1, 2]})


class TestRuntimeRejectsLabels:
    def test_run_window_rejects_label_in_payload(self):
        payload = {
            "frames": np.zeros((4, 3, 48, 48), dtype="float32"),
            "labels": np.array([0, 1, 2, 3]),    # <- forbidden on the online path
        }
        # raises in the payload guard, before any model is touched (f0=None is fine)
        with pytest.raises(LabelLeakError):
            run_window(0, payload, f0=None, adapter=None, estimator=None, eps=0.1, image_size=48)

    def test_logger_rejects_label_in_extra(self, tmp_path):
        lg = WindowLogger(str(tmp_path / "w.jsonl"), model_version="v", config_hash="c")
        decision = {"decision": "adapt", "bhat": 0.1, "eps": 0.1,
                    "lower": 0.0, "upper": 0.2, "reason": "x"}
        try:
            with pytest.raises(LabelLeakError):
                lg.log(0, decision, {"pre_entropy": 0.1}, latency_ms=1.0, extra={"labels": [1]})
        finally:
            lg.close()
