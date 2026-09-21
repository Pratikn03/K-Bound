"""The reviewed helper-only change cannot authorize arbitrary replay drift."""

from pathlib import Path

import pytest

from docs.research.kbound.scripts import policy_source_reconciliation as reconciliation

ROOT = Path(__file__).resolve().parents[1]


def test_authenticates_historical_and_current_policy_without_rewriting_results():
    receipt = reconciliation.verify_policy_reconciliation(ROOT)
    assert receipt["scope"] == "decide_batch replay only"
    assert receipt["historical_sha256"] != receipt["current_sha256"]


@pytest.mark.parametrize("which", ["historical", "current", "crossfit"])
def test_reconciliation_rejects_any_changed_bound_input(tmp_path, which):
    paths = {
        "historical": reconciliation.HISTORICAL_PATH,
        "current": "kga/policy.py",
        "crossfit": "kga/crossfit.py",
    }
    for name, relative in paths.items():
        output = tmp_path / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((ROOT / relative).read_bytes() + (b"\n# drift\n" if name == which else b""))
    with pytest.raises(ValueError, match="hash mismatch"):
        reconciliation.verify_policy_reconciliation(tmp_path)
