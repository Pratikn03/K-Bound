"""Table generation must depend on authenticated records, not display constants."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/generate_task1_kga_models_artifacts.py"


def _module():
    spec = importlib.util.spec_from_file_location("task1_table_authority", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, data):
    path.write_text(json.dumps(data))
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _fixture(tmp_path):
    # Toy bytes test file identity, not a real trained-model receipt.
    checkpoint = tmp_path / "toy.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint-not-a-model")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    models = [{"model_id": "toy", "name": "Toy", "source_seed": 7,
               "checkpoint": {"path": checkpoint.name, "sha256": digest},
               "clean_accuracy": 0.75}]
    rows = [
        {"model_id": "toy", "cell_id": "good", "checkpoint_sha256": digest,
         "a0": 0.5, "aa": 0.8, "kga_action": "ADAPT", "aetta_action": "ADAPT"},
        {"model_id": "toy", "cell_id": "bad", "checkpoint_sha256": digest,
         "a0": 0.9, "aa": 0.6, "kga_action": "FREEZE", "aetta_action": "ADAPT"},
    ]
    manifest = {"schema": "kbound_task1_table_authority_v1", "models": models,
                "panels": {}, "expected_cells": {}}
    for panel in ("helpful_only", "mixed_loo", "mixed_b6"):
        selected = rows[:1] if panel == "helpful_only" else rows
        payload = {"schema": "kbound_task1_cell_records_v1", "panel_id": panel,
                   "records": selected}
        manifest["panels"][panel] = _write(tmp_path / f"{panel}.json", payload)
        manifest["expected_cells"][panel] = [
            {"model_id": "toy", "cell_id": r["cell_id"]} for r in selected
        ]
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    return path, manifest


def _change_panel(path, manifest, panel, mutate):
    source = path.parent / manifest["panels"][panel]["path"]
    data = json.loads(source.read_text())
    mutate(data)
    manifest["panels"][panel] = _write(source, data)
    _write(path, manifest)


def test_missing_record_authorities_do_not_generate_literal_results(tmp_path, monkeypatch):
    module = _module()
    # Exercise the old no-argument entry point safely outside canonical outputs.
    for name in ("SUMMARY_JSON", "MIXED_JSON", "AETTA_JSON", "OUT_JSON", "OUT_TEX"):
        monkeypatch.setattr(module, name, tmp_path / name, raising=False)
    monkeypatch.setattr("sys.argv", [str(SCRIPT)])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 2
    assert not (tmp_path / "OUT_JSON").exists()
    assert not (tmp_path / "OUT_TEX").exists()


def test_changed_authenticated_cell_changes_generated_accuracy(tmp_path):
    module = _module()
    path, manifest = _fixture(tmp_path)
    assert hasattr(module, "build_table"), "record-driven table generator is missing"
    first = module.build_table(path)
    assert first["panels"]["helpful_only"]["toy"]["accuracy"]["kga"] == pytest.approx(0.8)
    _change_panel(path, manifest, "helpful_only", lambda d: d["records"][0].update(aa=0.7))
    second = module.build_table(path)
    assert second["panels"]["helpful_only"]["toy"]["accuracy"]["kga"] == pytest.approx(0.7)
    assert "70.00" in module.render_table(second)


def test_paired_arithmetic_comes_from_saved_actions(tmp_path):
    module = _module()
    path, _ = _fixture(tmp_path)
    assert hasattr(module, "build_table"), "record-driven table generator is missing"
    result = module.build_table(path)["panels"]["mixed_b6"]["toy"]
    assert result["accuracy"] == pytest.approx({"always_freeze": .7, "always_adapt": .7,
                                               "oracle": .85, "kga": .85, "aetta": .7})
    assert result["regret"] == pytest.approx({"always_freeze": .15, "always_adapt": .15,
                                             "kga": 0, "aetta": .15})
    assert result["actions"]["kga"] == {"ADAPT": 1, "FREEZE": 1, "ABSTAIN": 0}
    assert result["false_adapt"]["aetta"] == {"nonpositive": 1, "harmful": 1, "ties": 0,
                                              "unconditional": .5, "conditional": .5}


@pytest.mark.parametrize("bad", ["missing", "digest", "panel_id", "duplicate", "no_action",
                                  "range", "nonfinite", "checkpoint", "paired_scores",
                                  "paired_cells", "paired_aetta", "summary_only", "missing_model",
                                  "unknown_action", "helpful_not_positive", "stale_benefit"])
def test_invalid_authorities_fail_without_outputs(tmp_path, bad):
    module = _module()
    path, manifest = _fixture(tmp_path)
    assert hasattr(module, "build_table"), "record-driven table generator is missing"
    if bad == "missing":
        (tmp_path / "mixed_b6.json").unlink()
    elif bad == "digest":
        (tmp_path / "mixed_b6.json").write_text("{}")
    elif bad == "checkpoint":
        (tmp_path / "toy.pt").write_bytes(b"different-model")
    else:
        def mutate(d):
            if bad == "panel_id": d["panel_id"] = "other"
            if bad == "duplicate": d["records"].append(d["records"][0].copy())
            if bad == "no_action": d["records"][0].pop("kga_action")
            if bad == "range": d["records"][0]["a0"] = 1.1
            if bad == "nonfinite": d["records"][0]["aa"] = float("nan")
            if bad == "paired_scores": d["records"][0]["aa"] = .81
            if bad == "paired_cells": d["records"][0]["cell_id"] = "different"
            if bad == "paired_aetta": d["records"][0]["aetta_action"] = "FREEZE"
            if bad == "summary_only": d.pop("records"); d["kga_accuracy"] = .85
            if bad == "missing_model": d["records"][0]["model_id"] = "unknown"
            if bad == "unknown_action": d["records"][0]["kga_action"] = "GUESS"
            if bad == "helpful_not_positive": d["records"][0]["aa"] = .4
            if bad == "stale_benefit": d["records"][0]["B"] = .9
        _change_panel(path, manifest, "helpful_only" if bad == "helpful_not_positive" else "mixed_b6", mutate)
    output = tmp_path / "generated"
    with pytest.raises((ValueError, FileNotFoundError)):
        module.generate(path, output)
    assert not output.exists()


def test_no_adaptation_has_undefined_conditional_error(tmp_path):
    module = _module()
    path, manifest = _fixture(tmp_path)
    assert hasattr(module, "build_table"), "record-driven table generator is missing"
    def mutate(d):
        for row in d["records"]: row["kga_action"] = "ABSTAIN"
    _change_panel(path, manifest, "mixed_b6", mutate)
    result = module.build_table(path)["panels"]["mixed_b6"]["toy"]
    assert result["false_adapt"]["kga"]["conditional"] is None
    assert result["accuracy"]["kga"] == pytest.approx(.7)


def test_generate_binds_inputs_and_never_replaces_existing_output(tmp_path):
    module = _module()
    path, _ = _fixture(tmp_path)
    assert hasattr(module, "build_table"), "record-driven table generator is missing"
    output = tmp_path / "generated"
    module.generate(path, output)
    report = json.loads((output / "task1_table.json").read_text())
    assert report["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["claim_scope"] == "RECORDED_ACTION_ARITHMETIC_NOT_PROTOCOL_VALIDATION"
    before = (output / "tab_task1_independent_models.tex").read_bytes()
    with pytest.raises(FileExistsError):
        module.generate(path, output)
    assert (output / "tab_task1_independent_models.tex").read_bytes() == before


def test_matching_incomplete_exports_are_not_a_complete_panel(tmp_path):
    module = _module()
    path, manifest = _fixture(tmp_path)
    # Exporter loses the same row in both files. Independent reviewed roster stays intact.
    for panel in ("mixed_loo", "mixed_b6"):
        _change_panel(path, manifest, panel, lambda d: d["records"].pop())
    with pytest.raises(ValueError, match="roster"):
        module.generate(path, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


@pytest.mark.parametrize("bad", ["missing", "duplicate", "unknown_model", "missing_panel"])
def test_invalid_expected_roster_fails(tmp_path, bad):
    module = _module()
    path, manifest = _fixture(tmp_path)
    if bad == "missing": manifest.pop("expected_cells")
    if bad == "duplicate": manifest["expected_cells"]["helpful_only"] *= 2
    if bad == "unknown_model": manifest["expected_cells"]["helpful_only"][0]["model_id"] = "other"
    if bad == "missing_panel": manifest["expected_cells"].pop("mixed_b6")
    _write(path, manifest)
    with pytest.raises(ValueError, match="roster"):
        module.build_table(path)
