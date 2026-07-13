from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json"
PAPER = ROOT / "docs/research/kbound/kbound_short.tex"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sensitivity_artifact_and_inputs_match_manifest():
    manifest = json.loads(MANIFEST.read_text())
    declared = manifest["sensitivity_ablations"]
    artifact_path = ROOT / declared["artifact"]
    artifact = json.loads(artifact_path.read_text())

    assert sha256(artifact_path) == declared["artifact_sha256"]
    assert artifact["config"]["validity"] == "not exact split-conformal coverage"

    input_dir = ROOT / declared["inputs"]
    for adapter, expected in artifact["config"]["input_sha256"].items():
        path = input_dir / f"per_condition_cifar10c_{adapter}_seed0.json"
        assert sha256(path) == expected


def test_paper_uses_recomputed_ablation_values_and_scope():
    paper = PAPER.read_text()
    section = paper.split("\\section{Ablations and Sensitivity}", 1)[1].split(
        "\\section{Discussion and Failure Modes}", 1
    )[0]

    assert "$0.10$ & $0.0017$ & $0.000$ & $0.51$ & $0.68$" in section
    assert "Tent $\\to$ EATA & $0.0021$" in section
    assert "not exact\nsplit-conformal coverage" in section
    assert "breaks the guarantee" not in section
    assert "false-adapt guarantee intact" not in section
