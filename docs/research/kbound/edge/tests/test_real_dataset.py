import pytest
from pathlib import Path
import numpy as np

from kbound_edge.real_dataset import audit_dataset, load_window, AuditReport

def manifest_with_same_sha_in(split1_prefix: str, split2_prefix: str) -> dict:
    # Build a mock manifest/inventory where two clips in different splits have the same SHA-256 hash.
    # S03 is calibration_fit_a, S07 is heldout_a.
    clips = [
        {
            "clip_id": "S03_P01_ok_mild_light_R01",
            "session_id": "S03",
            "phone_id": "phone_a",
            "object_id": "P01",
            "class_id": "ok",
            "shift_id": "mild_light",
            "repetition": 1,
            "sha256": "same-sha-value-12345",
            "frame_count": 32,
            "captured_at": "2026-07-03T09:00:00-05:00"
        },
        {
            "clip_id": "S07_P09_ok_mild_light_R01",
            "session_id": "S07",
            "phone_id": "phone_a",
            "object_id": "P09",
            "class_id": "ok",
            "shift_id": "mild_light",
            "repetition": 1,
            "sha256": "same-sha-value-12345",  # Reused hash across splits!
            "frame_count": 32,
            "captured_at": "2026-07-03T10:00:00-05:00"
        }
    ]
    return {"clips": clips}

def test_audit_rejects_clip_reused_across_splits():
    manifest = manifest_with_same_sha_in("calibration_fit", "heldout")
    report = audit_dataset(manifest)
    assert not report.passed
    assert any("cross-split duplicate" in f for f in report.failures)

def test_online_window_excludes_labels(tmp_path):
    # Create a mock npz window
    win_path = tmp_path / "mock_window.npz"
    frames = np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)
    labels = np.array([0] * 32, dtype=np.int64)
    np.savez(
        win_path,
        frames=frames,
        labels=labels,
        window_id="S03_W001",
        source_hashes=np.array(["hash1"], dtype=object)
    )
    
    payload, offline = load_window(win_path)
    assert set(payload) == {"frames", "window_id", "source_hashes"}
    assert "labels" in offline
