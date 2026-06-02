from __future__ import annotations

import numpy as np

from src.scripts.scenario_c.prepare_realiad_d3_headroom_inputs import (
    D3Row,
    _apply_validation_score_orientation,
    _sample_id,
    _score_and_reliability,
)


def test_d3_sample_id_keeps_anomalous_images_distinct() -> None:
    first = {
        "image_path": "scratch/common_mode_filter_0001_scratch_RGBL05_00000000000000.jpg",
    }
    second = {
        "image_path": "scratch/common_mode_filter_0002_scratch_RGBL05_00000000000000.jpg",
    }

    assert _sample_id("common_mode_filter", first) != _sample_id("common_mode_filter", second)


def test_d3_sample_id_preserves_category_and_logical_image_path() -> None:
    entry = {
        "image_path": "OK/S0153/audio_jack_socket_0153_OK_RGBL05_00000000000000.jpg",
    }

    assert _sample_id("audio_jack_socket", entry) == (
        "audio_jack_socket/OK/S0153/audio_jack_socket_0153_OK_RGBL05_00000000000000"
    )


def test_one_class_scoring_preserves_far_outlier_order() -> None:
    rows = [
        D3Row("c", "t0", "train", "rgb", 0, "OK", "c/t0.jpg"),
        D3Row("c", "t1", "train", "rgb", 0, "OK", "c/t1.jpg"),
        D3Row("c", "t2", "train", "rgb", 0, "OK", "c/t2.jpg"),
        D3Row("c", "a1", "test", "rgb", 1, "scratch", "c/a1.jpg"),
        D3Row("c", "a2", "test", "rgb", 1, "scratch", "c/a2.jpg"),
    ]
    features = {
        ("c", "t0", "rgb"): np.array([0.0], dtype=np.float32),
        ("c", "t1", "rgb"): np.array([0.1], dtype=np.float32),
        ("c", "t2", "rgb"): np.array([-0.1], dtype=np.float32),
        ("c", "a1", "rgb"): np.array([10.0], dtype=np.float32),
        ("c", "a2", "rgb"): np.array([20.0], dtype=np.float32),
    }
    qualities = {
        ("c", "t0", "rgb"): np.array([0.0], dtype=np.float32),
        ("c", "t1", "rgb"): np.array([0.1], dtype=np.float32),
        ("c", "t2", "rgb"): np.array([-0.1], dtype=np.float32),
        ("c", "a1", "rgb"): np.array([10.0], dtype=np.float32),
        ("c", "a2", "rgb"): np.array([20.0], dtype=np.float32),
    }

    scores, reliabilities, _ = _score_and_reliability(rows=rows, features=features, qualities=qualities)

    assert scores[("c", "a2", "rgb")] > scores[("c", "a1", "rgb")]
    assert scores[("c", "a2", "rgb")] < 1.0
    assert reliabilities[("c", "a2", "rgb")] < reliabilities[("c", "a1", "rgb")]
    assert reliabilities[("c", "a1", "rgb")] > 0.0


def test_validation_score_orientation_flips_only_from_validation_labels() -> None:
    rows = [
        D3Row("c", "v0", "validation", "rgb", 0, "OK", "c/v0.jpg"),
        D3Row("c", "v1", "validation", "rgb", 1, "scratch", "c/v1.jpg"),
        D3Row("c", "x0", "test", "rgb", 0, "OK", "c/x0.jpg"),
        D3Row("c", "x1", "test", "rgb", 1, "scratch", "c/x1.jpg"),
    ]
    scores = {
        ("c", "v0", "rgb"): 0.9,
        ("c", "v1", "rgb"): 0.1,
        ("c", "x0", "rgb"): 0.8,
        ("c", "x1", "rgb"): 0.2,
    }

    orientations = _apply_validation_score_orientation(rows, scores)

    assert orientations[("c", "rgb")]["direction"] == "inverted"
    assert orientations[("c", "rgb")]["used_split"] == "validation"
    assert scores[("c", "x1", "rgb")] > scores[("c", "x0", "rgb")]
