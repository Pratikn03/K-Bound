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


def test_build_dashboard_snapshot_is_byte_stable():
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=REPO)
    first = SNAPSHOT.read_bytes()
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=REPO)
    assert SNAPSHOT.read_bytes() == first


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
    assert all(row["status"] != "verified" for row in controlled)
    assert all(row["ci_robust_beats_both"] is False for row in controlled)
    assert all(row["beats_both_artifact"] is False for row in controlled)
    tent = next(row for row in controlled if row["name"].endswith("Tent"))
    assert (
        "retrospective Holm adjustment over the six prospectively named contrasts"
        in tent["framing"]
    )
    assert "cluster-robust" not in json.dumps(data).lower()

    assert edge.get("session_progress")
    assert not edge["unblock"]["all_pass"]
    assert edge.get("development_metrics") is None

    boundary = data["evidence_board"].get("boundary_negative") or []
    inr = next((b for b in boundary if b["name"].startswith("ImageNet-R")), None)
    assert inr is not None
    assert inr["status"] == "diagnostic"

    natural = data["evidence_board"].get("natural_shift_no_harm") or []
    assert not any(row["name"].startswith("iWildCam") for row in natural)
    assert not any(row["name"].startswith("Camelyon17") for row in natural)
    iwild = next((b for b in boundary if b["name"].startswith("iWildCam")), None)
    assert iwild is not None
    assert iwild["status"] == "withheld"
    for field in ("freeze", "adapt", "kga", "regret_kga", "regret_adapt", "regret_freeze", "false_adapt"):
        assert iwild.get(field) is None

    camelyon = next((b for b in boundary if b["name"].startswith("Camelyon17")), None)
    assert camelyon is not None
    assert camelyon["status"] == "diagnostic"
    assert camelyon["point_beats_both"] is False
    assert camelyon["beats_both_artifact"] is False

    manifest = json.loads(
        (REPO / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text()
    )
    binding = manifest["reconciliation_source"]["current_policy_family_sensitivity"]
    assert data["meta"]["current_policy_sha256"] == binding["artifact_sha256"]

    assert data["edge_validation"].get("unblock") is not None
    assert data["provenance"]["manifest"].endswith("kbound_result_manifest.json")
    assert "legacy_elara" not in json.dumps(data)
