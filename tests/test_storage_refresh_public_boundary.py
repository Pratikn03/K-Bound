import copy
import json

import pytest

from docs.research.kbound.scripts import refresh_storage_manifest as refresh


def test_duplicate_authority_rows_fail_before_hashing(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    monkeypatch.setattr(refresh, "REFRESHABLE_AUTHORITIES", frozenset({"public.json"}))
    (tmp_path / "public.json").write_text("{}")
    manifest = {"artifacts": [{"expected_location": "public.json"}] * 2}
    before = copy.deepcopy(manifest)
    with pytest.raises(ValueError, match="duplicate"):
        refresh.refresh(manifest)
    assert manifest == before


def test_public_refresh_never_reads_protected_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    monkeypatch.setattr(refresh, "REFRESHABLE_AUTHORITIES", frozenset({"public.json"}))
    (tmp_path / "public.json").write_text("{}")
    protected = {"expected_location": "private/outcomes.json", "sha256": "historical", "size_bytes": 42}
    manifest = {"artifacts": [{"expected_location": "public.json"}, copy.deepcopy(protected)]}
    refresh.refresh(manifest)
    assert manifest["artifacts"][1] == protected
    assert manifest["artifacts"][0]["size_bytes"] == 2
    assert manifest["artifacts"][0]["sha256"] == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


def test_public_cli_refreshes_only_explicit_authorities(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    monkeypatch.setattr(refresh, "REFRESHABLE_AUTHORITIES", frozenset({"public.json"}))
    (tmp_path / "public.json").write_text("{}")
    manifest_path = tmp_path / "storage.json"
    private = {"expected_location": "private/absent.json", "sha256": "preserved"}
    manifest_path.write_text(json.dumps({"artifacts": [{"expected_location": "public.json"}, private]}))
    monkeypatch.setattr(refresh, "MANIFEST", manifest_path)
    assert refresh.main(["--write-public-only"]) == 0
    assert json.loads(manifest_path.read_text())["artifacts"][1] == private
