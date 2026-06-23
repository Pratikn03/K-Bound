#!/usr/bin/env python3
"""Unit tests for the hard-dataset K-Bound win loop."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = ROOT / "docs/research/kbound/scripts"
sys.path.insert(0, str(SCRIPT_DIR))


class FinderCliTests(unittest.TestCase):
    def test_no_known_panels_allows_targeted_empty_scan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "finder"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "find_kbound_wins.py"),
                    "--records",
                    str(Path(td) / "missing_*.json"),
                    "--no-known-panels",
                    "--output-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            payload = json.loads((out_dir / "finder_results.json").read_text())
            self.assertEqual(payload["n_sources"], 0)
            self.assertEqual(payload["n_rows"], 0)


class HardDatasetLoopTests(unittest.TestCase):
    def test_hard_dataset_report_separates_replicated_and_split_specific_wins(self) -> None:
        import run_hard_dataset_win_loop as loop

        results = [
            {
                "name": "imagenetr_light_split01_23",
                "dataset": "imagenet-r",
                "stability_group": "imagenetr_light_tent_cqr",
                "test_locked": {
                    "regret_kga": 0.004,
                    "regret_adapt": 0.005,
                    "regret_freeze": 0.018,
                    "false_adapt": 0.07,
                    "coverage": 0.58,
                    "n_test": 24,
                },
            },
            {
                "name": "imagenetr_light_split02_13",
                "dataset": "imagenet-r",
                "stability_group": "imagenetr_light_tent_cqr",
                "test_locked": {
                    "regret_kga": 0.011,
                    "regret_adapt": 0.003,
                    "regret_freeze": 0.021,
                    "false_adapt": 0.0,
                    "coverage": 0.37,
                    "n_test": 24,
                },
            },
            {
                "name": "rxrx1_modelseed0_tent_mondrian",
                "dataset": "rxrx1",
                "stability_group": "rxrx1_tent_mondrian",
                "test_locked": {
                    "regret_kga": 0.0008,
                    "regret_adapt": 0.052,
                    "regret_freeze": 0.0011,
                    "false_adapt": 0.0,
                    "coverage": 0.78,
                    "n_test": 60,
                },
            },
            {
                "name": "rxrx1_modelseed1_tent_mondrian",
                "dataset": "rxrx1",
                "stability_group": "rxrx1_tent_mondrian",
                "test_locked": {
                    "regret_kga": 0.0002,
                    "regret_adapt": 0.059,
                    "regret_freeze": 0.0002,
                    "false_adapt": 0.0,
                    "coverage": 1.0,
                    "n_test": 60,
                },
            },
        ]
        summarized = loop.summarize_stability(results)
        self.assertFalse(summarized["imagenetr_light_tent_cqr"]["replicated_win"])
        self.assertFalse(summarized["rxrx1_tent_mondrian"]["replicated_win"])
        self.assertEqual(summarized["imagenetr_light_tent_cqr"]["wins"], 1)
        self.assertEqual(summarized["rxrx1_tent_mondrian"]["wins"], 1)


if __name__ == "__main__":
    unittest.main()
