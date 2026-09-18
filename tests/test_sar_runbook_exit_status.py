import os
import subprocess
from pathlib import Path
import pytest


@pytest.mark.parametrize("historical_result_present", [False, True])
def test_failed_seed_stops_launcher_and_never_reports_complete(tmp_path, historical_result_present):
    root = tmp_path / 'repo'
    script = root / 'docs/research/kbound/scripts/cifar_tent_mps_v2.py'
    script.parent.mkdir(parents=True)
    script.write_text('# synthetic entrypoint\n')
    external = tmp_path / 'data'
    for c in ['gaussian_noise', 'shot_noise', 'impulse_noise']:
        for s in [1, 3, 5]:
            (external / 'imagenetc_local' / c / str(s)).mkdir(parents=True)
    bins = tmp_path / 'bin'; bins.mkdir()
    py = bins / 'python'
    py.write_text('#!/bin/sh\nif [ "$1" = "-" ]; then cat >/dev/null; exit 0; fi\nexit 17\n')
    py.chmod(0o755)
    caffeine = bins / 'caffeinate'
    caffeine.write_text('#!/bin/sh\nshift\nexec "$@"\n')
    caffeine.chmod(0o755)
    runbook = Path(__file__).resolve().parents[1] / 'docs/research/kbound/runbooks/run_sar_official_control.sh'
    if historical_result_present:
        for seed in (0, 1):
            output = root / 'experiments/kbound/results/imagenetc_sar_bncorrected_v2' / f'seed{seed}'
            output.mkdir(parents=True)
            (output / 'decisive_tta_results.json').write_text('{}')
    env = dict(os.environ, KBOUND_REPO_ROOT=str(root), KBOUND_EXTERNAL_ROOT=str(external),
               KBOUND_PYTHON=str(py), KBOUND_SEEDS='0 1', PATH=str(bins)+os.pathsep+os.environ['PATH'])
    run = subprocess.run(['/bin/bash', str(runbook)], env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 17
    log = (root / 'experiments/kbound/results/imagenetc_sar_bncorrected_v2/launch.log').read_text()
    assert 'CONTROL COMPLETE' not in log
    assert 'SEED 1 START' not in log


def test_preflight_resolves_fresh_corrected_output(tmp_path):
    runbook = Path(__file__).resolve().parents[1] / 'docs/research/kbound/runbooks/run_sar_official_control.sh'
    env = dict(os.environ, KBOUND_REPO_ROOT=str(tmp_path), KBOUND_EXTERNAL_ROOT=str(tmp_path),
               KBOUND_PYTHON='/usr/bin/true', KBOUND_SAR_OUTPUT=str(tmp_path / 'corrected'))
    run = subprocess.run(['/bin/bash', str(runbook), '--preflight'], env=env,
                         capture_output=True, text=True, timeout=20)
    assert 'output dir         : ' + str(tmp_path / 'corrected') in run.stdout
