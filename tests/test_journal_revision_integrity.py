"""Journal evidence identities and release inclusion, without study execution."""

import hashlib

import pytest

from docs.research.kbound.scripts import build_release_source_seal as seal
from docs.research.kbound.scripts import validate_journal_revision as journal


def fixture_register(root):
    (root / "paper.tex").write_text("The fixed-pair frontier under the declared class.")
    (root / "proof.tex").write_text("proof bytes")
    return {
        "schema": "kbound-journal-claim-register-v1",
        "claims": [
            {
                "id": "T1",
                "claim": "Conditional frontier",
                "status": "conditional",
                "assumptions": ["declared rich binary class"],
                "metric": "population risk difference",
                "unit": "fixed predictor pair",
                "evidence_type": "analytic",
                "justified_wording": "Exact within the declared class",
                "locations": [{"path": "paper.tex", "anchor": "fixed-pair frontier"}],
                "evidence": [{"path": "proof.tex", "sha256": hashlib.sha256(b"proof bytes").hexdigest()}],
            }
        ],
    }


def test_claim_register_checks_evidence_bytes_and_locations(tmp_path):
    payload = fixture_register(tmp_path)
    assert journal.validate_register(tmp_path, payload) == []
    (tmp_path / "proof.tex").write_text("changed proof")
    assert any("hash mismatch" in x for x in journal.validate_register(tmp_path, payload))
    (tmp_path / "paper.tex").write_text("different claim")
    assert any("anchor missing" in x for x in journal.validate_register(tmp_path, payload))


@pytest.mark.parametrize("bad", ["../outside", "/etc/passwd", "a/../../b"])
def test_claim_register_rejects_out_of_scope_references(tmp_path, bad):
    payload = fixture_register(tmp_path)
    payload["claims"][0]["evidence"][0]["path"] = bad
    assert any("unsafe" in x for x in journal.validate_register(tmp_path, payload))


def test_register_rejects_duplicate_ids_and_validator_as_scientific_status(tmp_path):
    payload = fixture_register(tmp_path)
    payload["claims"].append(dict(payload["claims"][0], status="PASS"))
    errors = journal.validate_register(tmp_path, payload)
    assert any("duplicate" in x for x in errors)
    assert any("scientific status" in x for x in errors)


def test_journal_reports_and_builders_are_in_source_seal_scope():
    prefix = "docs/research/kbound/journal_revision/"
    suffixes = {ext for _category, p, extensions in seal.SOURCE_PREFIX_RULES if p == prefix for ext in extensions}
    assert {".md", ".json", ".tex", ".csv"} <= suffixes
    assert prefix.rstrip("/") in seal.MAINTAINED_WORKTREE_SCOPES
    explicit = {p for values in seal.EXPLICIT_FILES.values() for p in values}
    assert "docs/research/kbound/scripts/validate_journal_revision.py" in explicit


def test_journal_tex_closure_and_metadata_have_unique_inventory_roles(tmp_path, monkeypatch):
    tex = "docs/research/kbound/journal_revision/empirical_tables.tex"
    csv = "docs/research/kbound/journal_revision/empirical_rows_v1.csv"
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {})
    monkeypatch.setattr(seal, "_validated_manuscript_closure", lambda repo: [tex])
    monkeypatch.setattr(seal, "_tree_blobs", lambda *args: {tex: "a" * 40, csv: "b" * 40})
    inventory = seal._inventory(tmp_path, "source")
    assert inventory.count(("paper_source", tex)) == 1
    assert ("journal_revision", csv) in inventory
