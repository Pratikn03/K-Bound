from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "docs/research/kbound/scripts/run_rxrx1_9plus.sh"
KBTRAIN = ROOT / "docs/research/kbound/scripts/kbtrain.sh"


def test_rxrx1_9plus_dry_run_prints_locked_protocol_command() -> None:
    env = os.environ.copy()
    env.update(
        {
            "RXRX1_MODEL_SEEDS": "0",
            "RXRX1_CONDITION_SEEDS": "0 1",
            "RXRX1_N_EVAL": "64",
            "RXRX1_RESULTS_ROOT": "/tmp/rxrx1_results_test",
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "DRY RUN" in proc.stdout
    assert "--ckpt /Users/pratik_n/kbound_rxrx1_ckpt/rxrx1_seed:0_epoch:best_model.pth" in proc.stdout
    assert "--seeds 0 1" in proc.stdout
    assert "--n-eval 64" in proc.stdout
    assert "--n-batches 4" in proc.stdout
    assert "--run-name rxrx1_protocol_c_9plus_modelseed0" in proc.stdout
    assert "tent_online" not in proc.stdout  # adapters are selected by the runner, not shell text.


def test_kbtrain_exposes_rxrx1_9plus_one_command() -> None:
    text = KBTRAIN.read_text()
    assert "rxrx1-9plus)" in text
    assert "run_rxrx1_9plus.sh" in text
