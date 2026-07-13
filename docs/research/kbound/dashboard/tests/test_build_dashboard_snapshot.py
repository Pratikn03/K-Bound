"""Tests for K-Bound HTML dashboard snapshot builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / "docs" / "research" / "kbound" / "scripts" / "build_dashboard_snapshot.py"
SNAPSHOT = REPO / "docs" / "research" / "kbound" / "dashboard" / "data" / "snapshot.json"
MANIFEST = REPO / "docs" / "research" / "kbound" / "paper" / "generated" / "kbound_result_manifest.json"
ENTRYPOINT = REPO / "docs" / "research" / "kbound" / "kbound_dashboard.html"
COMPONENTS = REPO / "docs" / "research" / "kbound" / "dashboard" / "src" / "components.ts"


def test_dashboard_entrypoint_loads_compiled_application():
    html = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'id="app-root"' in html
    assert 'src="dashboard/js/app.js"' in html
    assert 'href="dashboard/css/app.css"' in html


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

    assert edge.get("session_progress")
    assert not edge["unblock"]["all_pass"]
    assert edge.get("development_metrics") is None

    boundary = data["evidence_board"].get("boundary_negative") or []
    inr = next((b for b in boundary if b["name"].startswith("ImageNet-R")), None)
    assert inr is not None
    assert inr["status"] == "diagnostic"

    assert data["edge_validation"].get("unblock") is not None
    assert data["provenance"]["manifest"].endswith("kbound_result_manifest.json")
    assert data["provenance"]["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert data["meta"]["paper_pages"] == 21
    assert "legacy_elara" not in json.dumps(data)


def test_snapshot_rebuild_is_deterministic():
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=REPO)
    first = SNAPSHOT.read_bytes()
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=REPO)
    assert SNAPSHOT.read_bytes() == first


def test_dashboard_architecture_has_no_retired_external_asset():
    source = COMPONENTS.read_text(encoding="utf-8")
    assert "fig_architecture.svg" not in source
    assert "Population layer." in source
    assert "Real-data" in source
