"""The separately compiled Prop3 evidence must stay linked to exact sources."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_population_transfer_link_authenticates_sources_and_recorded_axioms():
    path = ROOT / "docs/research/kbound/formal/population_transfer_receipt_link.json"
    assert path.is_file(), "missing public Proposition 3 receipt link"
    link = json.loads(path.read_text())
    for relative, digest in link["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    source = (ROOT / "docs/research/kbound/formal/KBound/Probability/PopulationTransfer.lean").read_text()
    names = ["KBound.PopulationTransfer.pointwise", "KBound.PopulationTransfer.population_coverage",
             "KBound.PopulationTransfer.false_direction_probability"]
    assert link["declarations"] == names
    for name in names:
        assert "#print axioms " + name in source
        assert "'" + name + "' depends on axioms: [propext, Classical.choice, Quot.sound]" in link["recorded_axiom_output"]
    assert link["recorded_exit_code"] == 0
    assert link["fresh_build_performed"] is False
    assert link["historical_registered_declarations"] == 150
    original = ROOT / link["original_receipt"]
    if original.exists():
        assert hashlib.sha256(original.read_bytes()).hexdigest() == link["original_receipt_sha256"]
        receipt = json.loads(original.read_text())
        result = next(r for r in receipt["results"] if r.get("module") == link["module"])
        assert result["exit_code"] == link["recorded_exit_code"]
        assert result["output"] == link["recorded_axiom_output"]
