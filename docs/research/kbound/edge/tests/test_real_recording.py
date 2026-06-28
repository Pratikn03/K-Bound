from collections import Counter
import copy
from pathlib import Path
import pytest
import yaml

from kbound_edge.real_manifest import load_real_protocol
from kbound_edge.recording import build_session_checklist, make_clip_record

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE.parent / "configs" / "edge_real_phone_v1.yaml"

@pytest.fixture
def real_protocol():
    return load_real_protocol(CONFIG_PATH)

def test_checklist_is_deterministic_and_balanced(real_protocol):
    a = build_session_checklist(real_protocol, "S03")
    b = build_session_checklist(real_protocol, "S03")
    assert a == b
    assert len(a) == 64
    assert set(Counter(x["class_id"] for x in a).values()) == {16}

def test_clip_metadata_has_reproducibility_fields():
    row = make_clip_record(
        clip_id="S03_P01_ok_mild_light_R01",
        session_id="S03",
        phone_id="phone_a",
        object_id="P01",
        class_id="ok",
        shift_id="mild_light",
        repetition=1,
        captured_at="2026-07-03T09:00:00-05:00",
        sha256="a" * 64,
        frame_count=32,
    )
    required = {"clip_id", "session_id", "phone_id", "object_id", "class_id",
                "shift_id", "repetition", "captured_at", "sha256", "frame_count"}
    assert required <= row.keys()
