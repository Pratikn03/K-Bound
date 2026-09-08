"""Regression tests: arithmetic, fail-closed inputs, portable CLI, and extraction."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "docs/research/kbound/scripts"
BUNDLE = ROOT.parent / "paper/release/empirical_macro_inputs"


@pytest.fixture(autouse=True)
def local_script_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))


def module(name):
    path = ROOT / (name + ".py")
    assert path.exists(), "Required portable implementation does not exist yet"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def row_id(row):
    return digest(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def fixture(tmp_path):
    # Unequal n prevents an accidental switch to image-weighted regret.
    rows = [
        {
            "cell_id": f"c{i}",
            "n_images": 2 if i % 2 == 0 else 4,
            "frozen_correct": 0 if i % 2 == 0 else 1,
            "candidate_correct": 1 if i % 2 == 0 else 0,
            "action": "ABSTAIN",
        }
        for i in range(24)
    ]
    oh = [
        {"cell_id": f"o{i}", "a0": 0.25, "a_adapted": 0.5, "update_norm": 0.0 if i % 2 == 0 else 1e-30}
        for i in range(54)
    ]
    payloads = {
        "entropy": rows,
        "bridge": copy.deepcopy(rows),
        "officehome": oh,
        "smoke": [{"cell_id": "whole-run", "wall_seconds": 20.434765332989627}],
    }
    manifest = {"schema": "kbound-empirical-macro-bundle-v1", "extraction_version": 1, "records": {}}
    for role, payload in payloads.items():
        for row in payload:
            row["content_id"] = row_id(row)
        data = {"schema": "kbound-empirical-macro-extract-v1", "role": role, "rows": payload}
        raw = encoded(data)
        (tmp_path / f"{role}.json").write_bytes(raw)
        manifest["records"][role] = {
            "path": f"{role}.json",
            "sha256": digest(raw),
            "source_id": role,
            "original_sha256": "a" * 64,
            "row_count": len(payload),
            "ordered_cell_ids": [x["cell_id"] for x in payload],
        }
    (tmp_path / "manifest.json").write_bytes(encoded(manifest))
    return digest(encoded(manifest))


def mutate(tmp_path, role, change, rebind=True):
    path = tmp_path / f"{role}.json"
    data = json.loads(path.read_text())
    change(data)
    # Rebind checksums so semantic rejection cannot be masked by hash rejection.
    for row in data["rows"]:
        row.pop("content_id", None)
        row["content_id"] = row_id(row)
    raw = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if rebind:
        manifest["records"][role]["sha256"] = digest(raw)
    manifest_path.write_bytes(encoded(manifest))
    return digest(encoded(manifest))


def test_equal_cell_weighting_exact_zero_and_rounding(tmp_path):
    # Missing fsum/equal-cell weighting or tolerance-based zero would change these literals.
    root_hash = fixture(tmp_path)
    values = module("empirical_macros").compute(tmp_path, root_hash)
    assert values == {
        "KBDNEntropyGatePP": "25.000000",
        "KBDNEntropyAdaptPP": "12.500000",
        "KBDNBridgeGatePP": "25.000000",
        "KBDNBridgeAdaptPP": "12.500000",
        "KBOHZeroRegret": "0.12500000",
        "KBOHSmokeWallSeconds": "20.43",
    }


@pytest.mark.parametrize(
    "role,change",
    [
        ("entropy", lambda d: d["rows"].pop()),
        ("entropy", lambda d: d["rows"][1].update(cell_id="c0")),
        ("entropy", lambda d: d["rows"][0].update(action="SKIP")),
        ("entropy", lambda d: d["rows"][0].update(action=True)),
        ("entropy", lambda d: d["rows"][0].update(n_images=True)),
        ("entropy", lambda d: d["rows"][0].update(n_images=0)),
        ("entropy", lambda d: d["rows"][0].update(frozen_correct=-1)),
        ("entropy", lambda d: d["rows"][0].update(candidate_correct=3)),
        ("entropy", lambda d: d["rows"][0].update(candidate_correct=1.0)),
        ("entropy", lambda d: d["rows"].reverse()),
        ("officehome", lambda d: d["rows"][0].update(a0="0.25")),
        ("officehome", lambda d: d["rows"][0].update(a_adapted=1.1)),
        ("officehome", lambda d: d["rows"][0].update(update_norm=-0.1)),
        ("officehome", lambda d: d["rows"][0].update(update_norm=True)),
        ("smoke", lambda d: d["rows"][0].update(wall_seconds=0)),
    ],
)
def test_invalid_semantics_rejected_even_after_checksum_rebinding(tmp_path, role, change):
    fixture(tmp_path)
    root_hash = mutate(tmp_path, role, change)
    with pytest.raises(ValueError):
        module("empirical_macros").compute(tmp_path, root_hash)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_rejected(tmp_path, number):
    fixture(tmp_path)
    path = tmp_path / "officehome.json"
    path.write_text(path.read_text().replace("0.25", number, 1))
    mpath = tmp_path / "manifest.json"
    m = json.loads(mpath.read_text())
    m["records"]["officehome"]["sha256"] = digest(path.read_bytes())
    mpath.write_bytes(encoded(m))
    with pytest.raises(ValueError):
        module("empirical_macros").compute(tmp_path, digest(mpath.read_bytes()))


def test_tampered_input_rejected(tmp_path):
    root_hash = fixture(tmp_path)
    mutate(tmp_path, "entropy", lambda d: d["rows"][0].update(action="ADAPT"), rebind=False)
    with pytest.raises(ValueError):
        module("empirical_macros").compute(tmp_path, root_hash)


def test_missing_file_rejected(tmp_path):
    root_hash = fixture(tmp_path)
    (tmp_path / "bridge.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        module("empirical_macros").compute(tmp_path, root_hash)


def test_wrong_root_manifest_hash_rejected(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError):
        module("empirical_macros").compute(tmp_path, "0" * 64)


def test_content_id_tampering_rejected_when_file_hash_rebound(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "entropy.json"
    d = json.loads(path.read_text())
    d["rows"][0]["content_id"] = "0" * 64
    path.write_bytes(encoded(d))
    mpath = tmp_path / "manifest.json"
    m = json.loads(mpath.read_text())
    m["records"]["entropy"]["sha256"] = digest(path.read_bytes())
    mpath.write_bytes(encoded(m))
    with pytest.raises(ValueError):
        module("empirical_macros").compute(tmp_path, digest(mpath.read_bytes()))


@pytest.mark.parametrize("path", ["../entropy.json", "/tmp/entropy.json"])
def test_nonportable_input_path_rejected(tmp_path, path):
    fixture(tmp_path)
    mpath = tmp_path / "manifest.json"
    m = json.loads(mpath.read_text())
    m["records"]["entropy"]["path"] = path
    mpath.write_bytes(encoded(m))
    with pytest.raises(ValueError):
        module("empirical_macros").compute(tmp_path, digest(mpath.read_bytes()))


def test_cli_writes_only_macro_output_and_refuses_overwrite(tmp_path):
    root_hash = fixture(tmp_path)
    output = tmp_path / "numbers.tex"
    cmd = [
        sys.executable,
        str(ROOT / "empirical_macros.py"),
        "--bundle",
        str(tmp_path),
        "--manifest-sha256",
        root_hash,
        "--output",
        str(output),
    ]
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text().count("\\providecommand") == 6
    assert all((tmp_path / name).read_bytes() == raw for name, raw in before.items())
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode != 0


def test_extractor_strips_nonselected_fields_and_binds_source(tmp_path):
    m = module("extract_macro_bundle")
    source = {
        "cells": [
            {
                "cell_id": f"DEV_check:{i}",
                "n_images": 2,
                "frozen_correct": 1,
                "candidate_correct": 0,
                "action": "ABSTAIN",
                "unused_private": "excluded",
            }
            for i in range(24)
        ]
    }
    selected = m.select_domainnet(source)
    assert len(selected) == 24
    assert set(selected[0]) == {"cell_id", "n_images", "frozen_correct", "candidate_correct", "action"}


def test_officehome_extraction_preserves_exact_decimal_value_and_candidate_identity():
    m = module("extract_macro_bundle")
    data = {
        "metadata": {"evidence_names": ["other", "update_norm"]},
        "records": [
            {
                "domain": "Art",
                "comp": "iid",
                "regime": "tiny",
                "seed": 2,
                "candidate": "sar_online_aggressive",
                "metric": "accuracy",
                "split": "test",
                "a0": 0.25,
                "a_adapted": 0.5,
                "Z": [999, 1e-30],
            }
        ],
    }
    row = m.select_officehome(data)[0]
    assert row["update_norm"] == 1e-30
    assert row["cell_id"] == "Art|iid|tiny|2|sar_online_aggressive|accuracy|test"
    assert set(row) == {"cell_id", "a0", "a_adapted", "update_norm"}


@pytest.mark.parametrize("kind", ["version", "row_count"])
def test_boolean_metadata_is_not_an_integer_contract(tmp_path, kind):
    fixture(tmp_path)
    path = tmp_path / "manifest.json"
    d = json.loads(path.read_text())
    if kind == "version":
        d["extraction_version"] = True
    else:
        d["records"]["smoke"]["row_count"] = True
    path.write_bytes(encoded(d))
    with pytest.raises(ValueError):
        module("empirical_macros").compute(tmp_path, digest(path.read_bytes()))


def test_duplicate_json_key_rejected(tmp_path):
    fixture(tmp_path)
    p = tmp_path / "manifest.json"
    p.write_text(p.read_text().replace('"extraction_version": 1,', '"extraction_version": 1, "extraction_version": 1,'))
    with pytest.raises(ValueError):
        module("empirical_macros").compute(tmp_path, digest(p.read_bytes()))


def test_child_symlink_rejected(tmp_path):
    root_hash = fixture(tmp_path)
    (tmp_path / "entropy.json").rename(tmp_path / "actual.json")
    (tmp_path / "entropy.json").symlink_to("actual.json")
    with pytest.raises(ValueError):
        module("empirical_macros").compute(tmp_path, root_hash)


def test_unapproved_extraction_source_rejected_before_output(tmp_path):
    m = module("extract_macro_bundle")
    p = tmp_path / "fake.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        m.extract(dict.fromkeys(("entropy", "bridge", "officehome", "smoke"), p), tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_committed_bundle_reproduces_six_reviewed_values_and_portable_ids():
    m = module("empirical_macros")
    bundle = BUNDLE
    values = m.compute(bundle, "e4c2cff245871a2d2e8ebe626e7c197b59faa4194f921e15eff84a3903e6c3b4")
    assert values == {
        "KBDNEntropyGatePP": "0.390625",
        "KBDNEntropyAdaptPP": "1.074219",
        "KBDNBridgeGatePP": "0.065104",
        "KBDNBridgeAdaptPP": "8.299520",
        "KBOHZeroRegret": "0.00035613",
        "KBOHSmokeWallSeconds": "20.43",
    }
    for path in (
        bundle / name for name in ("manifest.json", "entropy.json", "bridge.json", "officehome.json", "smoke.json")
    ):
        payload = path.read_text()
        assert not any(token in payload for token in ("/Volumes/", "/Users/", "@"))


@pytest.mark.parametrize("mode", ["missing", "match", "mismatch"])
def test_check_mode_is_read_only_and_validates_exact_macro_bytes(tmp_path, mode):
    root_hash = fixture(tmp_path)
    expected = tmp_path / "existing.tex"
    if mode != "missing":
        expected.write_text(
            module("empirical_macros").render(module("empirical_macros").compute(tmp_path, root_hash))
            if mode == "match"
            else "stale manuscript numbers\n"
        )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "empirical_macros.py"),
            "--bundle",
            str(tmp_path),
            "--manifest-sha256",
            root_hash,
            "--check",
            str(expected),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0 if mode == "match" else result.returncode != 0
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_check_and_output_modes_cannot_be_combined(tmp_path):
    root_hash = fixture(tmp_path)
    output = tmp_path / "must-not-exist.tex"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "empirical_macros.py"),
            "--bundle",
            str(tmp_path),
            "--manifest-sha256",
            root_hash,
            "--check",
            str(tmp_path / "missing.tex"),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert not output.exists()


def test_integrated_bundle_matches_manuscript_bytes():
    m = module("empirical_macros")
    expected = m.render(m.compute(BUNDLE, "e4c2cff245871a2d2e8ebe626e7c197b59faa4194f921e15eff84a3903e6c3b4")).encode(
        "ascii"
    )
    assert (ROOT.parent / "paper/generated/empirical_evidence_numbers.tex").read_bytes() == expected


def test_ci_executes_portable_reproduction_regressions():
    import yaml

    workflow = yaml.safe_load((REPO / ".github/workflows/kbound-ci.yml").read_text())
    steps = workflow["jobs"]["kbound-research-tests"]["steps"]
    runners = [step.get("run", "") for step in steps]
    assert any(
        "pytest" in run and "tests/test_kbound_empirical_macros.py" in run and "--collect-only" not in run
        for run in runners
    )
