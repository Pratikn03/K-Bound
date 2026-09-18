"""Import-boundary checks isolated from numerical/OpenMP test processes."""

from __future__ import annotations

import subprocess
import sys


def test_offline_scorer_import_is_separate_from_live_runner_and_inference() -> None:
    command = (
        "import sys; "
        "import experiments.kbound.so2sat.target_scorer; "
        "assert 'experiments.kbound.so2sat.target_runner' not in sys.modules; "
        "assert 'experiments.kbound.so2sat.target_inference' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_target_seal_import_does_not_import_offline_scorer_process() -> None:
    command = (
        "import sys; "
        "import experiments.kbound.so2sat.target_seal; "
        "assert 'experiments.kbound.so2sat.target_scorer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
