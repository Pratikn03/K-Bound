from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCORER = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/official_baselines_headtohead.py"


@pytest.fixture
def stream_case(tmp_path):
    spec = importlib.util.spec_from_file_location("selected_stream_under_test", SCORER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [{"condition": f"c{i}", "Z": [float((i + j) % 13) / 13 for j in range(11)],
                "B": 0.1, "a0": 0.5, "a_adapted": 0.6, "a_oracle": 0.6} for i in range(40)]
    stream = tmp_path / "selected.json"
    stream.write_text(json.dumps({"records": records}), encoding="utf-8")
    return module, stream, records, tmp_path / "score.json"


def test_stream_loader_hashes_the_bytes_it_parsed(stream_case, monkeypatch):
    module, path, records, _ = stream_case
    original = path.read_bytes()
    records[0].update(B=-0.1, a_adapted=0.4, a_oracle=0.5)
    replacement = json.dumps({"records": records}).encode()
    reads = 0

    def racing_open(filename, mode="r", *args, **kwargs):
        nonlocal reads
        if Path(filename) == path and "r" in mode:
            reads += 1
            if reads == 2:
                path.write_bytes(replacement)
        return builtins.open(filename, mode, *args, **kwargs)

    monkeypatch.setattr(module, "open", racing_open, raising=False)
    _, _, benefit, _, adapted, _, digest = module.load("tent", stream_path=str(path))
    assert benefit[0] == pytest.approx(0.1)
    assert adapted[0] == 0.6
    assert hashlib.sha256(original).hexdigest().startswith(digest)


def test_stream_cli_rejects_change_after_the_first_read(stream_case, monkeypatch):
    module, path, records, output = stream_case
    records[0].update(B=-0.1, a_adapted=0.4, a_oracle=0.5)
    replacement = json.dumps({"records": records}).encode()
    changed = False

    class MutatingRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, *args, **kwargs):
            nonlocal changed
            snapshot = self.handle.read(*args, **kwargs)
            if not changed:
                changed = True
                path.write_bytes(replacement)
            return snapshot

    def racing_open(filename, mode="r", *args, **kwargs):
        handle = builtins.open(filename, mode, *args, **kwargs)
        return MutatingRead(handle) if Path(filename) == path and "r" in mode else handle

    monkeypatch.setattr(module, "open", racing_open, raising=False)
    monkeypatch.setattr(sys, "argv", [str(SCORER), "--stream", str(path), "--out", str(output)])
    with pytest.raises(SystemExit, match="stream changed"):
        module.main()
    assert changed
    assert not output.exists()


def test_stream_cli_rechecks_after_scoring_before_publication(stream_case, monkeypatch):
    module, path, records, output = stream_case
    original_gate = module.kga_exact_rank

    def score_then_change(*args, **kwargs):
        result = original_gate(*args, **kwargs)
        records[0].update(B=-0.1, a_adapted=0.4, a_oracle=0.5)
        path.write_text(json.dumps({"records": records}), encoding="utf-8")
        return result

    monkeypatch.setattr(module, "kga_exact_rank", score_then_change)
    monkeypatch.setattr(sys, "argv", [str(SCORER), "--stream", str(path), "--out", str(output)])
    with pytest.raises(SystemExit, match="stream changed"):
        module.main()
    assert not output.exists()


@pytest.mark.parametrize("defect", [
    "benefit", "oracle", "empty", "duplicate_id", "blank_id", "short_features", "long_features",
    "nan_feature", "string_feature", "boolean_feature", "invalid_frozen", "invalid_adapted",
    "boolean_accuracy", "string_accuracy", "boolean_benefit", "nan_benefit", "nan_oracle",
    "overflow_feature", "duplicate_json_key",
])
def test_stream_loader_rejects_invalid_scoring_inputs(stream_case, defect):
    module, path, records, _ = stream_case
    if defect == "benefit":
        records[0]["B"] = -0.1
    elif defect == "oracle":
        records[0]["a_oracle"] = 0.9
    elif defect == "empty":
        records.clear()
    elif defect == "duplicate_id":
        records[1]["condition"] = records[0]["condition"]
    elif defect == "blank_id":
        records[0]["condition"] = "  "
    elif defect == "short_features":
        records[0]["Z"].pop()
    elif defect == "long_features":
        records[0]["Z"].append(0.0)
    elif defect == "nan_feature":
        records[0]["Z"][0] = float("nan")
    elif defect == "string_feature":
        records[0]["Z"][0] = "0.5"
    elif defect == "boolean_feature":
        records[0]["Z"][0] = True
    elif defect == "invalid_frozen":
        records[0]["a0"] = -0.1
    elif defect == "invalid_adapted":
        records[0]["a_adapted"] = 1.1
    elif defect == "boolean_accuracy":
        records[0]["a0"] = True
    elif defect == "string_accuracy":
        records[0]["a_adapted"] = "0.6"
    elif defect == "boolean_benefit":
        records[0]["B"] = True
    elif defect == "nan_benefit":
        records[0]["B"] = float("nan")
    elif defect == "nan_oracle":
        records[0]["a_oracle"] = float("nan")
    elif defect == "overflow_feature":
        records[0]["Z"][0] = 1.23456789
    payload = json.dumps({"records": records})
    if defect == "overflow_feature":
        payload = payload.replace("1.23456789", "1e400")
    elif defect == "duplicate_json_key":
        payload = payload.replace('"condition": "c0"', '"condition": "hidden", "condition": "c0"')
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid.*stream"):
        module.load("tent", stream_path=str(path))


def test_stream_outcomes_use_score_arithmetic_with_existing_absolute_tolerance(stream_case):
    module, path, records, _ = stream_case
    records[0].update(B=0.1 + 5e-11, a_oracle=0.6 + 5e-11)
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    original = path.read_bytes()
    _, _, benefit, _, _, oracle, digest = module.load("tent", stream_path=str(path))
    assert benefit[0] == pytest.approx(0.1, abs=1e-15, rel=0)
    assert oracle[0] == 0.6
    assert digest == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original


@pytest.mark.parametrize("field, value", [("B", 0.1 + 2e-10), ("a_oracle", 0.6 + 2e-10)])
def test_stream_outcomes_reject_disagreement_beyond_existing_tolerance(stream_case, field, value):
    module, path, records, _ = stream_case
    records[0][field] = value
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid.*stream"):
        module.load("tent", stream_path=str(path))
