import copy
import json

import pytest

from docs.research.kbound.scripts import build_result_manifest as builder


def public_records():
    return json.loads(builder.LEDGER.read_text()), json.loads(builder.OUT.read_text())


def test_public_natural_preservation_does_not_open_authorities(monkeypatch):
    ledger, manifest = public_records()
    before = copy.deepcopy(manifest)
    def forbidden(*args, **kwargs):
        raise AssertionError("separate authority opened")
    monkeypatch.setattr(builder, "digest", forbidden)
    authorities, rows = builder.preserved_public_natural_state(ledger, manifest)
    assert set(rows) == {"KB-CLAIM-051", "KB-CLAIM-052"}
    assert authorities == ledger["reconciliation_source"]["separate_receipt_linked_authorities"]
    assert manifest == before


@pytest.mark.parametrize("mutation", ["duplicate", "index_drift", "promotion", "target_score"])
def test_public_natural_preservation_rejects_drift(mutation):
    ledger, manifest = public_records()
    rows = {r["claim_id"]: r for r in manifest["results"]}
    if mutation == "duplicate":
        manifest["results"].append(copy.deepcopy(rows["KB-CLAIM-051"]))
    elif mutation == "index_drift":
        manifest["reconciliation_source"]["separate_receipt_linked_authorities"]["cct20"]["artifact_sha256"] = "0" * 64
    elif mutation == "promotion":
        rows["KB-CLAIM-051"]["metrics"]["point_beats_both"] = True
    else:
        rows["KB-CLAIM-052"]["metrics"]["target_score"] = 0.9
    with pytest.raises(ValueError):
        builder.preserved_public_natural_state(ledger, manifest)


def test_public_cli_preserves_rows_before_any_authority_access(tmp_path, monkeypatch):
    ledger, manifest = public_records()
    ids = {"KB-CLAIM-051", "KB-CLAIM-052"}
    ledger["claims"] = [r for r in ledger["claims"] if r["claim_id"] in ids]
    expected = [r for r in manifest["results"] if r["claim_id"] in ids]
    ledger_path, out = tmp_path / "ledger.json", tmp_path / "results.json"
    ledger_path.write_text(json.dumps(ledger))
    out.write_text(json.dumps(manifest))
    monkeypatch.setattr(builder, "LEDGER", ledger_path)
    monkeypatch.setattr(builder, "OUT", out)
    def forbidden(*args, **kwargs):
        raise AssertionError("natural result recomputed")
    monkeypatch.setattr(builder, "validated_separate_authorities", forbidden)
    monkeypatch.setattr(builder, "special_metrics", forbidden)
    builder.main(["--public-only"])
    result = json.loads(out.read_text())
    assert result["results"] == expected
