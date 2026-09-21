from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'docs/research/kbound/scripts/seal_nine_track_lock.py'


def load_verifier():
    spec = importlib.util.spec_from_file_location('historical_readonly', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_mode_refuses_before_creating_any_evidence(tmp_path):
    # Broken behavior: default invocation writes seals. Use an isolated script.
    script = tmp_path / 'docs/research/kbound/scripts/seal_nine_track_lock.py'
    script.parent.mkdir(parents=True)
    script.write_bytes(SCRIPT.read_bytes())
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'read-only historical verifier' in result.stderr.lower()
    assert not (tmp_path / 'experiments').exists()
    assert not (tmp_path / 'research_lock').exists()


def test_historical_verification_never_rewrites_files(monkeypatch):
    module = load_verifier()
    before = {p: p.read_bytes() for p in (module.SEAL_JSON, module.SEAL_SHA, module.LOCK_YAML)}
    def forbid_write(*args, **kwargs):
        pytest.fail('historical verification attempted a write')
    monkeypatch.setattr(Path, 'write_bytes', forbid_write)
    monkeypatch.setattr(Path, 'write_text', forbid_write)
    seal, errors = module.verify_historical_seal()
    assert not errors, errors
    assert seal['seal_id'] == 'NINE_TRACK_LOCK_SEAL_v1'
    assert {p: p.read_bytes() for p in before} == before


def test_baseline_unavailable_is_failure_not_current_file_fallback(monkeypatch):
    module = load_verifier()
    def missing_baseline(*args, **kwargs):
        return subprocess.CompletedProcess(args, 128, b'', b'missing object')
    monkeypatch.setattr(module.subprocess, 'run', missing_baseline)
    _, errors = module.verify_historical_seal()
    assert errors
    assert any('baseline' in e.lower() for e in errors)


def test_archive_symlink_is_rejected_before_payload_read(tmp_path, monkeypatch):
    module = load_verifier()
    linked = tmp_path / 'manifest.json'
    linked.symlink_to(module.ARCHIVE_MANIFEST)
    monkeypatch.setattr(module, 'ARCHIVE_MANIFEST', linked)
    _, errors = module.verify_historical_seal()
    assert errors
    assert any('symlink' in e.lower() for e in errors)


def test_missing_archive_record_cannot_silently_fall_back(tmp_path, monkeypatch):
    module = load_verifier()
    value = json.loads(module.ARCHIVE_MANIFEST.read_text())
    value['records'] = value['records'][1:]
    changed = tmp_path / 'manifest.json'
    changed.write_text(json.dumps(value))
    monkeypatch.setattr(module, 'ARCHIVE_MANIFEST', changed)
    _, errors = module.verify_historical_seal()
    assert errors
