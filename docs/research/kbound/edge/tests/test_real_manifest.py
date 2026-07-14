import copy
from pathlib import Path
import pytest
import yaml

from kbound_edge.real_manifest import ProtocolError, validate_protocol, load_real_protocol

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE.parent / "configs" / "edge_real_phone_v1.yaml"

@pytest.fixture
def real_protocol():
    return load_real_protocol(CONFIG_PATH)

def test_protocol_rejects_phone_b_outside_replication(real_protocol):
    cfg = copy.deepcopy(real_protocol)
    # S07 is heldout_a, phone must be phone_a (primary). phone_b is only for replication (S09/S10)
    cfg["sessions"]["S07"]["phone_id"] = "phone_b"
    with pytest.raises(ProtocolError, match="phone_b.*replication"):
        validate_protocol(cfg)

def test_protocol_requires_disjoint_session_ids(real_protocol):
    cfg = copy.deepcopy(real_protocol)
    # sessions dict is keyed by session_id, so duplicate session_id structure is tested
    # by ensuring a validation check or when parsing we enforce session constraints.
    # Wait, the sessions are keys of a dict, so they are unique. But we can modify the inner split IDs
    # or ensure they match expected session keys. Let's see:
    # "validate_protocol must enforce duplicate session_id constraints, exact class list, object split, phone split..."
    # Let's test that if we modify session split ids to be duplicate or invalid it fails.
    cfg["sessions"]["S08"]["split"] = "heldout_a"  # S07 is heldout_a, S08 must be heldout_b
    with pytest.raises(ProtocolError, match="duplicate.*split"):
        validate_protocol(cfg)

def test_protocol_requires_exact_counts(real_protocol):
    cfg = copy.deepcopy(real_protocol)
    cfg["sessions"]["S01"]["windows"] = 100  # S01 must have 120 windows
    with pytest.raises(ProtocolError, match="S01.*windows"):
        validate_protocol(cfg)

def test_protocol_requires_alpha_0_10(real_protocol):
    cfg = copy.deepcopy(real_protocol)
    cfg["alpha"] = 0.05
    with pytest.raises(ProtocolError, match="alpha"):
        validate_protocol(cfg)

def test_protocol_requires_window_size_32(real_protocol):
    cfg = copy.deepcopy(real_protocol)
    cfg["window_size"] = 16
    with pytest.raises(ProtocolError, match="window_size"):
        validate_protocol(cfg)
