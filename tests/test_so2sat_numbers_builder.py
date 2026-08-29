from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_so2sat_numbers as builder


def test_released_so2sat_numbers_are_receipt_bound_and_complete() -> None:
    numbers = builder.load_validated_numbers()
    assert numbers["DevelopmentCityCount"] == 9
    assert numbers["CheckpointCount"] == 5
    assert numbers["CellCountPerCandidate"] == 45
    assert numbers["OpenedOutcomeCount"] == 90
    assert numbers["TentOracleGainPP"] == "0.8848"
    assert numbers["TentLocoSignPct"] == "51.11"
    assert numbers["TentLossBestFixedPP"] == "0.1544"
    assert numbers["TentPassedCheckCount"] == 7
    assert numbers["SarOracleGainPP"] == "0.5640"
    assert numbers["SarAdaptCellCount"] == 5
    assert numbers["SarLossBestFixedPP"] == "0.0842"
    assert numbers["SarPassedCheckCount"] == 6
    assert numbers["GateCalRowsRead"] == 0
    assert numbers["TargetPixelsRead"] == 0
    assert numbers["TargetLabelsRead"] == 0
    rendered = builder.render_numbers_tex(numbers)
    assert r"\newcommand{\SoTwoOpenedOutcomeCount}{90}" in rendered
    assert r"\newcommand{\SoTwoTargetLabelsRead}{0}" in rendered


def test_builder_rejects_rebound_study_shape_drift(tmp_path: Path) -> None:
    selection = json.loads(builder.SELECTION.read_text(encoding="utf-8"))
    candidate_id = "tent_adam_bn_affine_probe_transfer_v1"
    selection["candidate_summaries"][candidate_id]["feasibility"]["cell_count"] = 44
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    receipt = {
        "schema": "kbound_so2sat_artifact_receipt_v2",
        "artifact_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "artifact_bytes": selection_path.stat().st_size,
    }
    receipt_path = tmp_path / "selection.json.receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(builder.AuthorityError, match="study shape drifted"):
        builder.load_validated_numbers(selection_path, receipt_path)
