import copy
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "public_ledger_sync", Path(__file__).resolve().parents[1] / "scripts/sync_reconciled_panels.py"
)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_public_ledger_sync_preserves_natural_claims_without_reading_authorities(monkeypatch):
    ledger = sync._load(sync.LEDGER_PATH)
    cluster = sync._load(sync.CURRENT_CLUSTER_PATH)
    panel = sync._load(sync.PANEL_PATH)
    protected_ids = {"KB-CLAIM-051", "KB-CLAIM-052"}
    preserved = copy.deepcopy([r for r in ledger["claims"] if r["claim_id"] in protected_ids])
    index = copy.deepcopy(ledger["reconciliation_source"]["separate_receipt_linked_authorities"])

    def forbidden(*args, **kwargs):
        raise AssertionError("protected natural authority was read")

    monkeypatch.setattr(sync, "_validated_separate_natural_authorities", forbidden)
    original_hash = sync._sha256
    def guarded_hash(path):
        if path in {sync.CCT20_RELEASE_PATH, sync.SO2SAT_SELECTION_PATH}:
            forbidden()
        return original_hash(path)
    monkeypatch.setattr(sync, "_sha256", guarded_hash)
    sync._sync_ledger(ledger, cluster, panel, refresh_separate_natural_authorities=False)
    assert [r for r in ledger["claims"] if r["claim_id"] in protected_ids] == preserved
    assert ledger["reconciliation_source"]["separate_receipt_linked_authorities"] == index
    first = copy.deepcopy(ledger)
    sync._sync_ledger(ledger, cluster, panel, refresh_separate_natural_authorities=False)
    assert ledger == first


def test_public_cli_runs_all_sync_stages_without_natural_authority_access(monkeypatch):
    original_load = sync._load
    writes = {}
    def forbidden(*args, **kwargs):
        raise AssertionError("protected natural authority was accessed")
    def load_public(path):
        if path == sync.CCT20_RELEASE_PATH or path == sync.SO2SAT_SELECTION_PATH or sync.SO2SAT_DEVELOPMENT_DIR in path.parents:
            forbidden()
        return original_load(path)
    monkeypatch.setattr(sync, "_load", load_public)
    monkeypatch.setattr(sync, "_validated_separate_natural_authorities", forbidden)
    monkeypatch.setattr(sync, "_write", lambda path, value: writes.__setitem__(path, copy.deepcopy(value)))
    monkeypatch.setattr(sys, "argv", ["sync_reconciled_panels.py", "--public-only", "--json-only"])
    sync.main()
    assert set(writes) == {sync.TABLE_PATH, sync.LEDGER_PATH, sync.FRONTIER_PATH,
                           sync.UNIFORM_VERDICTS_PATH, sync.DECISION_METRICS_PATH}
    assert "kbound_full.tex" in writes[sync.TABLE_PATH]["generated_for"]
