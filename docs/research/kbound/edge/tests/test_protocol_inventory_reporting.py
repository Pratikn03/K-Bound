from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from kbound_edge.real_manifest import load_real_protocol
from kbound_edge.reporting import protocol_inventory_macros

EDGE = Path(__file__).resolve().parents[1]


def test_table_s1_inventory_is_derived_from_locked_protocol() -> None:
    cfg = load_real_protocol(EDGE / "configs/edge_real_phone_v1.yaml")
    macros = protocol_inventory_macros(cfg)
    expected = {
        "SourceTrain": ("Day 1 / S01", "P01--P06", "120", "3,840", "30/30/30/30"),
        "SourceVal": ("Day 1 / S02", "P07, P08", "40", "1,280", "10/10/10/10"),
        "CalibrationFit": (
            "Day 2 / S03, S04",
            "P01--P04",
            "144",
            "4,608",
            "28/28/28/28 + 32 mixed",
        ),
        "CalibrationConformal": (
            "Day 3 / S05, S06",
            "P01--P04",
            "144",
            "4,608",
            "28/28/28/28 + 32 mixed",
        ),
        "HeldoutTest": (
            "Day 4 / S07, S08",
            "P09, P10",
            "144",
            "4,608",
            "28/28/28/28 + 32 mixed",
        ),
        "Replication": (
            "Day 5 / S09, S10",
            "P09, P10",
            "144",
            "4,608",
            "28/28/28/28 + 32 mixed",
        ),
    }
    for name, (session, objects, windows, frames, labels) in expected.items():
        assert macros[f"CameraSOneSession{name}"] == session
        assert macros[f"CameraSOneObjects{name}"] == objects
        assert macros[f"CameraSOneWindows{name}"] == windows
        assert macros[f"CameraSOneFrames{name}"] == frames
        assert macros[f"CameraSOneLabels{name}"] == labels


def test_inventory_frames_change_with_window_size() -> None:
    cfg = yaml.safe_load((EDGE / "configs/edge_real_phone_v1.yaml").read_text(encoding="utf-8"))
    cfg["window_size"] = 16
    macros = protocol_inventory_macros(cfg)
    assert macros["CameraSOneWindowsSourceTrain"] == "120"
    assert macros["CameraSOneFramesSourceTrain"] == "1,920"


def test_inventory_rejects_generator_lock_count_mismatch() -> None:
    cfg = yaml.safe_load((EDGE / "configs/edge_real_phone_v1.yaml").read_text(encoding="utf-8"))
    cfg["sessions"]["S01"]["windows"] = 121
    with pytest.raises(ValueError, match=r"S01.*produced 120 windows.*declares 121"):
        protocol_inventory_macros(cfg)
