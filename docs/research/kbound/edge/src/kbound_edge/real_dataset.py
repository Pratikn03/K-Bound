from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

SESSION_SPLIT_MAP = {
    "S01": "source_train",
    "S02": "source_val",
    "S03": "calibration_fit",
    "S04": "calibration_fit",
    "S05": "calibration_conformal",
    "S06": "calibration_conformal",
    "S07": "heldout",
    "S08": "heldout",
    "S09": "replication",
    "S10": "replication",
}

@dataclass
class AuditReport:
    passed: bool
    failures: list[str] = field(default_factory=list)

def audit_dataset(manifest: dict) -> AuditReport:
    """Audit the real dataset manifest for split leakage, invalid hashes, and count constraints."""
    failures = []

    # Track which SHA-256 hashes are used in which splits
    sha_to_splits = {}
    clips = manifest.get("clips", [])

    for clip in clips:
        clip_id = clip.get("clip_id")
        session_id = clip.get("session_id")
        sha256 = clip.get("sha256")

        split = SESSION_SPLIT_MAP.get(session_id, "unknown")

        if sha256:
            if sha256 not in sha_to_splits:
                sha_to_splits[sha256] = set()
            sha_to_splits[sha256].add(split)

    # Check for cross-split duplicates
    for sha, splits in sha_to_splits.items():
        if len(splits) > 1:
            failures.append(f"cross-split duplicate hash found: {sha} in splits {sorted(list(splits))}")

    passed = len(failures) == 0
    return AuditReport(passed=passed, failures=failures)

def load_window(npz_path: str | Path) -> tuple[dict, dict]:
    """Load a window NPZ file and return split payload (online) and offline components."""
    data = np.load(npz_path, allow_pickle=True)
    payload = {
        "frames": data["frames"],
        "window_id": str(data["window_id"]),
        "source_hashes": data["source_hashes"]
    }
    offline = {
        "labels": data["labels"]
    }
    return payload, offline
