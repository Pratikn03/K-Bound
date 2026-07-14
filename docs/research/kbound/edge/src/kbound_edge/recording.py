import hashlib
import random
from kbound_edge.real_manifest import expected_windows

def build_session_checklist(config: dict, session_id: str) -> list[dict]:
    """Build a deterministic, randomized checklist of physical clips to record for a session."""
    # 1. Get all expected windows
    all_wins = expected_windows(config, session_id)
    # 2. Keep only physical ones
    physical_wins = [w for w in all_wins if not w.get("is_derived", False)]
    # 3. Sort them to have a stable starting point
    physical_wins.sort(key=lambda w: (w["object_id"], w["class_id"], w["shift_id"], w["repetition"]))
    
    # 4. Populate clip_id for each item
    for w in physical_wins:
        w["clip_id"] = f"{session_id}_{w['object_id']}_{w['class_id']}_{w['shift_id']}_R{w['repetition']:02d}"
    
    # 5. Shuffle deterministically using a seed derived from config seed and session_id
    seed_str = f"{config.get('seed', 0)}_{session_id}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) & 0xffffffff
    rng = random.Random(seed)
    rng.shuffle(physical_wins)
    
    return physical_wins

def make_clip_record(
    clip_id: str,
    session_id: str,
    phone_id: str,
    object_id: str,
    class_id: str,
    shift_id: str,
    repetition: int,
    captured_at: str,
    sha256: str,
    frame_count: int,
    capture_mode: str = "physical",
    camera_index: int | None = None,
) -> dict:
    """Build a clip record dictionary containing all reproducibility metadata fields."""
    return {
        "clip_id": clip_id,
        "session_id": session_id,
        "phone_id": phone_id,
        "object_id": object_id,
        "class_id": class_id,
        "shift_id": shift_id,
        "repetition": repetition,
        "captured_at": captured_at,
        "sha256": sha256,
        "frame_count": frame_count,
        "capture_mode": capture_mode,
        "camera_index": camera_index,
    }
