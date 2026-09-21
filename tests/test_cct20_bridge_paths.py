from __future__ import annotations

from pathlib import Path

import pytest

from docs.research.kbound.scripts import verify_cct20_prospective_evidence as bridge


def test_unconfigured_root_refuses_before_artifact_access(monkeypatch):
    monkeypatch.delenv('KBOUND_CCT20_ROOT', raising=False)
    def deny(*args, **kwargs):
        pytest.fail('unconfigured verifier read an artifact')
    monkeypatch.setattr(bridge, '_verify_receipted', deny)
    with pytest.raises(bridge.BridgeIntegrityError, match='KBOUND_CCT20_ROOT'):
        bridge.verify_cct20_prospective_bundle(verify_large_files=False)


def test_environment_root_is_used_without_a_machine_default(tmp_path, monkeypatch):
    monkeypatch.setenv('KBOUND_CCT20_ROOT', str(tmp_path))
    seen = []
    def inspect(path, *, label):
        seen.append(path)
        raise bridge.BridgeIntegrityError('synthetic stop')
    monkeypatch.setattr(bridge, '_verify_receipted', inspect)
    with pytest.raises(bridge.BridgeIntegrityError, match='synthetic stop'):
        bridge.verify_cct20_prospective_bundle(verify_large_files=False)
    assert seen == [tmp_path / 'cct20_execution_seal_v1.json']


def test_full_verification_requires_explicit_source_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv('KBOUND_CCT20_SOURCE_SNAPSHOT', raising=False)
    def deny(*args, **kwargs):
        pytest.fail('source-unconfigured verifier read an artifact')
    monkeypatch.setattr(bridge, '_verify_receipted', deny)
    with pytest.raises(bridge.BridgeIntegrityError, match='KBOUND_CCT20_SOURCE_SNAPSHOT'):
        bridge.verify_cct20_prospective_bundle(tmp_path)


def test_source_remap_uses_sealed_protocol_identity_and_exact_prefix():
    seal = {'authoritative_protocol_file': {
        'path': '/old/repo/experiments/kbound/cct20/prospective_protocol_v1.yaml'
    }}
    source = bridge._sealed_source_root(seal)
    assert source == Path('/old/repo')
    assert bridge._mapped_sealed_path('/old/repo/models/a.pt', Path('/snapshot'), source) == Path('/snapshot/models/a.pt')
    assert bridge._mapped_sealed_path('/old/repo-other/a.pt', Path('/snapshot'), source) == Path('/old/repo-other/a.pt')


@pytest.mark.parametrize('path', ['relative.pt', '/old/repo/../secret.pt'])
def test_remap_rejects_ambiguous_sealed_paths(path):
    with pytest.raises(bridge.BridgeIntegrityError, match='path'):
        bridge._mapped_sealed_path(path, Path('/snapshot'), Path('/old/repo'))


def test_source_remap_rejects_wrong_protocol_location():
    with pytest.raises(bridge.BridgeIntegrityError, match='protocol.*path'):
        bridge._sealed_source_root({'authoritative_protocol_file': {'path': '/elsewhere/protocol.yaml'}})
