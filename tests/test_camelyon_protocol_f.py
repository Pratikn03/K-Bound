from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WILDS_DIR = ROOT / "experiments" / "kbound" / "wilds"
SCRIPTS_DIR = ROOT / "docs" / "research" / "kbound" / "scripts"


def test_full_camelyon_runner_exposes_protocol_f_evidence_panel_flag():
    runner = WILDS_DIR / "run_camelyon17_kbound.py"
    proc = subprocess.run(
        [sys.executable, str(runner), "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "--evidence-panel" in proc.stdout
    assert "rich" in proc.stdout


def test_rich_evidence_vector_is_finite_and_named():
    sys.path.insert(0, str(WILDS_DIR))
    import tta_methods as tm

    logits_f0 = np.array([[2.0, 0.2], [0.1, 1.8], [1.2, 0.7]], dtype=float)
    logits_fa = np.array([[1.4, 0.4], [0.8, 1.1], [0.4, 1.3]], dtype=float)
    logits_src = np.array([[2.1, 0.1], [0.2, 1.7], [1.9, 0.3]], dtype=float)
    y_src = np.array([0, 1, 0], dtype=int)

    rich = tm.rich_evidence_vector(
        logits_f0_tgt=logits_f0,
        logits_adapt_tgt=logits_fa,
        logits_src=logits_src,
        y_src=y_src,
        bn_kl=0.25,
    )

    assert rich.shape == (len(tm.RICH_EVIDENCE_NAMES),)
    assert tm.RICH_EVIDENCE_NAMES == [
        "disagreement_rate",
        "entropy_gap",
        "energy_shift",
        "bn_kl",
        "atc_acc_est",
        "conf_drop",
    ]
    assert np.all(np.isfinite(rich))


def test_protocol_f_analyzer_loads_full_camelyon_manifest(tmp_path):
    sys.path.insert(0, str(SCRIPTS_DIR))
    import analyze_F

    manifest = {
        "schema": "kbound_wilds_camelyon17_v0.6",
        "config": {"evidence_panel": "rich"},
        "records": [
            {
                "seed": 0,
                "Z": [0.1] * 17,
                "B": 0.05,
                "a0": 0.70,
                "aa": 0.75,
                "comp": "iid",
                "candidate": "tent_online",
            },
            {
                "seed": 2,
                "Z": [0.2] * 17,
                "B": -0.04,
                "a0": 0.72,
                "aa": 0.68,
                "comp": "single_class",
                "candidate": "sar_online",
            },
        ],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    records, panel = analyze_F.load_records(path)

    assert panel == "rich"
    assert len(records) == 2
    assert records[0]["comp"] == "iid"
    assert records[1]["comp"] == "single_class"
