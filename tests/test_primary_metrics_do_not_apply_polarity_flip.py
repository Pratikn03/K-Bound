"""Phase 1.F — assert the runner does not flip primary predictions."""

from __future__ import annotations

import re
from pathlib import Path

RUNNER = Path("src/scripts/run_breakthrough_experiment.py")


def test_runner_does_not_flip_static_or_craf_in_primary_path():
    text = RUNNER.read_text()
    # The four forbidden assignment patterns from before the Phase 1.F lock.
    forbidden = [
        re.compile(r"^\s*static_val_probs\s*=\s*1\.0\s*-\s*static_val_probs", re.MULTILINE),
        re.compile(r"^\s*static_probs\s*=\s*1\.0\s*-\s*static_probs", re.MULTILINE),
        re.compile(r"^\s*craf_val_probs\s*=\s*1\.0\s*-\s*craf_val_probs", re.MULTILINE),
        re.compile(r"^\s*craf_probs\s*=\s*1\.0\s*-\s*craf_probs", re.MULTILINE),
    ]
    for pat in forbidden:
        m = pat.search(text)
        assert m is None, (
            f"runner still applies a polarity flip to primary predictions at offset {m.start()}: "
            f"{text[m.start():m.end()]!r}. Phase 1.F lock requires the flip to be removed."
        )


def test_runner_contains_phase_1f_lock_comment():
    text = RUNNER.read_text()
    assert "Phase 1.F lock" in text, (
        "Phase 1.F lock comment missing from runner — the polarity-flip removal must be documented inline."
    )
