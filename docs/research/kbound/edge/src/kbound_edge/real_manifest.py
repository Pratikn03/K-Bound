import hashlib
import json
from pathlib import Path
import yaml

class ProtocolError(ValueError):
    """Exception raised for protocol validation errors."""
    pass

def load_real_protocol(path: str | Path) -> dict:
    """Load and validate the real protocol configuration from a YAML file."""
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        validate_protocol(cfg)
        return cfg
    except Exception as e:
        if not isinstance(e, ProtocolError):
            raise ProtocolError(f"Failed to load/parse protocol: {e}") from e
        raise e

def validate_protocol(config: dict) -> None:
    """Validate that the config dictionary satisfies all protocol constraints."""
    if not isinstance(config, dict):
        raise ProtocolError("Config must be a dictionary")

    # 1. Check protocol name
    if config.get("protocol") != "edge_real_phone_v1":
        raise ProtocolError("Invalid protocol name, expected 'edge_real_phone_v1'")
    
    # 2. Check classes
    expected_classes = ["ok", "missing_label", "misaligned_label", "damaged_label"]
    if config.get("classes") != expected_classes:
        raise ProtocolError(f"Invalid classes, expected {expected_classes}")
        
    # 3. Check window_size
    if config.get("window_size") != 32:
        raise ProtocolError("Invalid window_size, expected 32")
        
    # 4. Check alpha
    if config.get("alpha") != 0.10:
        raise ProtocolError("Invalid alpha, expected 0.10")

    # 5. Check object splits
    obj_cfg = config.get("objects", {})
    source_train_calib = obj_cfg.get("source_train_calib", [])
    source_val_calib = obj_cfg.get("source_val_calib", [])
    held_out_replication = obj_cfg.get("held_out_replication", [])
    
    if source_train_calib != ["P01", "P02", "P03", "P04", "P05", "P06"]:
        raise ProtocolError("Invalid source_train_calib objects")
    if source_val_calib != ["P07", "P08"]:
        raise ProtocolError("Invalid source_val_calib objects")
    if held_out_replication != ["P09", "P10"]:
        raise ProtocolError("Invalid held_out_replication objects")
        
    # 6. Check phone splits
    phones = config.get("phones", {})
    if "phone_a" not in phones or "phone_b" not in phones:
        raise ProtocolError("Missing phone_a or phone_b configuration")
        
    # 7. Check sessions
    sessions = config.get("sessions", {})
    expected_session_ids = [f"S{i:02d}" for i in range(1, 11)]
    for s_id in expected_session_ids:
        if s_id not in sessions:
            raise ProtocolError(f"Missing session {s_id}")
            
    # Check duplicate session split types
    splits = [sess.get("split") for sess in sessions.values() if sess.get("split")]
    if len(splits) != len(set(splits)):
        raise ProtocolError(f"duplicate session split types found: {splits}")
        
    # Validate each session configuration
    for s_id, sess in sessions.items():
        split = sess.get("split")
        phone_id = sess.get("phone_id")
        objects = sess.get("objects")
        windows = sess.get("windows")
        
        if phone_id == "phone_b" and s_id not in ["S09", "S10"]:
            raise ProtocolError(f"phone_b cannot be used in session {s_id} (outside replication)")
            
        if s_id == "S01":
            if split != "source_train" or phone_id != "phone_a" or objects != ["P01", "P02", "P03", "P04", "P05", "P06"] or windows != 240:
                raise ProtocolError(f"S01 configuration mismatch: {sess}")
        elif s_id == "S02":
            if split != "source_val" or phone_id != "phone_a" or objects != ["P07", "P08"] or windows != 80:
                raise ProtocolError(f"S02 configuration mismatch: {sess}")
        elif s_id in ["S03", "S04", "S05", "S06"]:
            expected_objs = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08"]
            if phone_id != "phone_a" or objects != expected_objs or windows != 128:
                raise ProtocolError(f"{s_id} configuration mismatch: {sess}")
        elif s_id in ["S07", "S08"]:
            if phone_id != "phone_a" or objects != ["P09", "P10"] or windows != 128:
                raise ProtocolError(f"{s_id} configuration mismatch: {sess}")
        elif s_id in ["S09", "S10"]:
            if phone_id != "phone_b" or objects != ["P09", "P10"] or windows != 128:
                raise ProtocolError(f"{s_id} configuration mismatch: {sess}")
        else:
            raise ProtocolError(f"Unknown session {s_id}")

    # Explicitly check phone_b restriction
    for s_id, sess in sessions.items():
        if s_id not in ["S09", "S10"] and sess.get("phone_id") == "phone_b":
            raise ProtocolError(f"phone_b cannot be used in session {s_id} (outside replication)")

def canonical_protocol_hash(config: dict) -> str:
    """Compute a stable SHA-256 hash of the protocol configuration."""
    def sort_struct(item):
        if isinstance(item, dict):
            return {k: sort_struct(item[k]) for k in sorted(item.keys())}
        elif isinstance(item, list):
            return [sort_struct(x) for x in item]
        return item
    sorted_cfg = sort_struct(config)
    data = json.dumps(sorted_cfg, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def expected_windows(config: dict, session_id: str) -> list[dict]:
    """Return the list of all expected windows (physical and derived) for a session."""
    sessions = config.get("sessions", {})
    sess = sessions.get(session_id)
    if not sess:
        raise ProtocolError(f"Session {session_id} not found in protocol")
    
    objects = sess["objects"]
    classes = config["classes"]
    
    is_a_session = session_id in ["S01", "S03", "S05", "S07", "S09"]
    
    wins = []
    
    if session_id == "S01":
        # source_train: 6 objects, 4 classes, 10 repetitions, stable shift
        for obj in objects:
            for cls in classes:
                for rep in range(1, 11):
                    wins.append({
                        "object_id": obj,
                        "class_id": cls,
                        "shift_id": "stable",
                        "repetition": rep,
                        "is_derived": False
                    })
    elif session_id == "S02":
        # source_val: 2 objects, 4 classes, 10 repetitions, stable shift
        for obj in objects:
            for cls in classes:
                for rep in range(1, 11):
                    wins.append({
                        "object_id": obj,
                        "class_id": cls,
                        "shift_id": "stable",
                        "repetition": rep,
                        "is_derived": False
                    })
    elif is_a_session:
        # S03, S05, S07, S09
        # odd sessions use A_sessions shifts: [mild_light, side_shadow, new_background, glare]
        shifts = ["mild_light", "side_shadow", "new_background", "glare"]
        reps = 1 if session_id in ["S03", "S05"] else 4
        for obj in objects:
            for cls in classes:
                for shift in shifts:
                    for rep in range(1, reps + 1):
                        wins.append({
                            "object_id": obj,
                            "class_id": cls,
                            "shift_id": shift,
                            "repetition": rep,
                            "is_derived": False
                        })
    else:
        # S04, S06, S08, S10
        # 96 physical windows with [motion_blur, viewpoint_45, distance_scale]
        # plus 32 derived batch_composition windows.
        shifts = ["motion_blur", "viewpoint_45", "distance_scale"]
        reps = 1 if session_id in ["S04", "S06"] else 4
        
        # 96 physical windows
        for obj in objects:
            for cls in classes:
                for shift in shifts:
                    for rep in range(1, reps + 1):
                        wins.append({
                            "object_id": obj,
                            "class_id": cls,
                            "shift_id": shift,
                            "repetition": rep,
                            "is_derived": False
                        })
                        
        # 32 derived batch_composition windows
        # We assign recipe and index to make them identifiable.
        recipes = [
            ("balanced", [8, 8, 8, 8]),
            ("ok_heavy", [20, 4, 4, 4]),
            ("missing_heavy", [4, 20, 4, 4]),
            ("fault_mixed", [4, 8, 10, 10])
        ]
        
        idx = 1
        for recipe_name, counts in recipes:
            for _ in range(8):  # 8 windows per recipe
                wins.append({
                    "object_id": "mixed",
                    "class_id": "mixed",
                    "shift_id": "batch_composition",
                    "repetition": idx,
                    "is_derived": True,
                    "recipe": recipe_name,
                    "counts": counts
                })
                idx += 1
                
    # Assign deterministic window_ids: SXX_W001, SXX_W002, etc.
    for i, w in enumerate(wins):
        w["window_id"] = f"{session_id}_W{i+1:03d}"
        
    return wins
