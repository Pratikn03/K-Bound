"""Tests for K-Bound HTML dashboard snapshot builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / "docs" / "research" / "kbound" / "scripts" / "build_dashboard_snapshot.py"
SNAPSHOT = REPO / "docs" / "research" / "kbound" / "dashboard" / "data" / "snapshot.json"


def test_build_dashboard_snapshot_runs():
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=REPO)


def test_snapshot_has_required_sections():
    if not SNAPSHOT.is_file():
        test_build_dashboard_snapshot_runs()
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for key in (
        "meta",
        "evidence_strip",
        "theory_ledger",
        "regime_map",
        "evidence_board",
        "edge_validation",
        "safety",
        "reproduce",
        "provenance",
    ):
        assert key in data, key

    edge = data["edge_validation"]
    assert edge["study_status"] == "pending"
    assert edge["study_label"].startswith("Pre-registered")

    controlled = data["evidence_board"]["controlled_wins"]
    assert any("CIFAR-10-C" in r["name"] for r in controlled)

    # Edge development metrics must not be promoted as verified study.
    dev = edge.get("development_metrics")
    assert dev is not None
    assert dev.get("phone_a_balanced_acc") == 0.25

    helpful = data["evidence_board"].get("helpful_dominated") or []
    assert any("65 cells" in r["name"] for r in helpful)

    boundary = data["evidence_board"].get("boundary_negative") or []
    inr = next((b for b in boundary if b["name"].startswith("ImageNet-R")), None)
    assert inr is not None
    assert inr.get("kga") is not None

    assert data["edge_validation"].get("unblock") is not None
