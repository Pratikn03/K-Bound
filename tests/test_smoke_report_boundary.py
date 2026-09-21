from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_summary_is_not_release_authority(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        'smoke_report', ROOT / 'docs/research/kbound/scripts/smoke_pipeline_report.py'
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    manifest = tmp_path / 'smoke.json'
    manifest.write_text(json.dumps({
        'rows': [{'dataset': d, 'n': 1, 'regret_kga': '0.1', 'beats_both': True}
                 for d in report.EXPECTED],
        'files_scanned': ['stress_grid_multiseed_v1/seed0/a.json',
                          'stress_grid_multiseed_v1/seed1/a.json',
                          'stress_grid_multiseed_v1/seed1/b.json'],
    }))
    # Substitute only external/historical input locations, not report behavior.
    for entry in report.LOCKED.values():
        entry['path'] = tmp_path / 'absent.json'
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['smoke_pipeline_report.py', '--manifest', str(manifest)])
    result = report.main()
    output = capsys.readouterr().out
    assert result == 0  # dataset presence, not promotion
    assert 'CIFAR seed dirs seen: 2 ' in output
    assert 'lower-regret point flag=True' in output
    assert 'release_candidate.sh all' in output
    assert 'run_final_showcase.sh' not in output
    assert 'not release verification' in output.lower()
    assert 'locked Holm WIN' not in output
    assert 'beats-both(pt)' not in output
    assert not (tmp_path / 'experiments').exists()
