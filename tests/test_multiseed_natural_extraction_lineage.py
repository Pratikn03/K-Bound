import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
EXTRACT_PATH = REPO / "docs/research/kbound/scripts/extract_multiseed_natural.py"
SERIALIZER_PATH = REPO / "experiments/kbound/wilds/per_condition_serialize.py"
FOREST_PATH = REPO / "docs/research/kbound/scripts/make_multiseed_natural_forest.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extract = _load("test_extract_multiseed_natural", EXTRACT_PATH)
serializer = _load("test_per_condition_serialize", SERIALIZER_PATH)
forest = _load("test_extract_to_forest", FOREST_PATH)


def _record(*, candidate="sar_online_aggressive", seed=0, domain="Art", comp="iid"):
    record = {
        "candidate": candidate,
        "mode": "episodic" if "episodic" in candidate else "online",
        "seed": seed,
        "domain": domain,
        "split": "val",
        "comp": comp,
        "regime": "tiny",
        "a0": 0.50,
        "aa": 0.55,
        "B": 0.05,
        "Z": [0.1, 0.2],
    }
    record["tta_protocol"] = extract._expected_tta_protocol(record)
    return record


def _source(path, seed, checkpoint, partition="target_val"):
    return {
        "path": str(path),
        "sha256": "f" * 64,
        "n_records": 1,
        "partition": partition,
        "model_seed": seed,
        "checkpoint_sha256": checkpoint,
        "checkpoint_path": str(path.with_suffix(".pt")),
        "checkpoint_tensor_sha256": (hex(seed + 1)[2:] * 64)[:64],
        "checkpoint_verified": True,
        "source_contract_validated": True,
        "protocol_fingerprint": "d" * 64,
        "publication": {"publication_ready": False},
    }


def _annotate(record, source):
    record = dict(record)
    record["_source_path"] = source["path"]
    record["_source_sha256"] = source["sha256"]
    record["_source_partition"] = source["partition"]
    record["_source_model_seed"] = source["model_seed"]
    record["_source_checkpoint_sha256"] = source["checkpoint_sha256"]
    record["_source_checkpoint_path"] = source["checkpoint_path"]
    record["_source_checkpoint_tensor_sha256"] = source["checkpoint_tensor_sha256"]
    record["_source_checkpoint_verified"] = source["checkpoint_verified"]
    record["_source_protocol_fingerprint"] = source["protocol_fingerprint"]
    record["_evaluation_population_sha256"] = extract._sha256_json(
        {
            "domain": record.get("domain"),
            "split": record.get("split"),
            "comp": record.get("comp"),
            "regime": record.get("regime"),
        }
    )
    return record


def _office_contract_document(tmp_path, *, seed=0, metric="accuracy", dataset="office-home"):
    checkpoint = tmp_path / f"missing_checkpoint_seed{seed}.pt"
    record = _record(seed=17)
    record["model_seed"] = seed
    record["sample_provenance"] = {"ordered_eval_sample_ids_sha256": "e" * 64}
    return {
        "schema": "kbound_officehome_v4",
        "dataset": dataset,
        "metric": metric,
        "role": "target_val",
        "model_seed": seed,
        "publication_eligible": False,
        "f0_checkpoint": str(checkpoint),
        "f0_checkpoint_sha256": "a" * 64,
        "f0_checkpoint_tensor_sha256": "b" * 64,
        "population_manifest": {
            "Art/val": {"n": 10, "ordered_sample_label_sha256": "c" * 64}
        },
        "resume_contract": {
            "schema": "test_resume_v1",
            "payload": {
                "dataset": "office-home",
                "role": "target_val",
                "candidate_set_ordered": ["sar_online_aggressive"],
                "checkpoint": {
                    "path": str(checkpoint),
                    "file_sha256": "a" * 64,
                    "tensor_sha256": "b" * 64,
                },
                "seed_semantics": {
                    "model_seed": seed,
                    "stream_seeds_ordered": [17],
                    "condition_seed_rule": "model-seed invariant",
                },
                "scientific_config": {"n_eval": 10},
            },
        },
        "records": [record],
    }


def _per_condition(
    path,
    *,
    seed,
    seed_kind="model_seed",
    checkpoint=None,
    feasible=True,
    backend="test_backend",
):
    records = []
    for condition, benefit, decision in [
        ("Art|val|iid|tiny", 0.05, "ADAPT"),
        ("Clipart|val|iid|tiny", -0.02, "FREEZE"),
    ]:
        a0 = 0.50
        adapted = a0 + benefit
        routed = adapted if decision == "ADAPT" else a0
        record = {
                "condition": condition,
                "a0": a0,
                "a_adapted": adapted,
                "B": benefit,
                "a_oracle": max(a0, adapted),
                "a_kbound": routed,
                "kga_decision": decision,
                "calibration_feasible": feasible,
                "radius_status": "FINITE" if feasible else "INFEASIBLE",
                "eps_conformal": 0.01 if feasible else None,
                "benefit_ci": [benefit - 0.01, benefit + 0.01] if feasible else None,
                "gamma_ci": (
                    [(benefit - 0.01) / 2, (benefit + 0.01) / 2]
                    if feasible
                    else None
                ),
            }
        if not feasible:
            record["kga_decision"] = "ABSTAIN"
            record["a_kbound"] = a0
        records.append(record)
    payload = {
        "seed": seed,
        "model_seed": seed if seed_kind == "model_seed" else None,
        "seed_kind": seed_kind,
        "stream_seed": 0 if seed_kind == "model_seed" else seed,
        "checkpoint_sha256": checkpoint,
        "checkpoint_tensor_sha256": (
            ((hex(seed + 1)[2:] * 64)[:64]) if seed_kind == "model_seed" else None
        ),
        "tta_protocol": extract._expected_tta_protocol(
            {"candidate": "sar_online_aggressive", "mode": "online"}
        ),
        "evaluation_partition": "target_val",
        "benchmark": "officehome",
        "method": "sar_online_aggressive",
        "kga_backend": backend,
        "estimator_publication_eligible": False,
        "n_calibration_infeasible": 0 if feasible else len(records),
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_duplicate_scientific_condition_keys_hard_fail_before_estimation(tmp_path):
    source = _source(tmp_path / "run.json", 0, "a" * 64)
    duplicated = [_annotate(_record(), source), _annotate(_record(), source)]
    with pytest.raises(extract.LineageError, match="duplicate scientific condition"):
        extract._prepare_records(
            duplicated,
            [source],
            "officehome",
            ["sar_online_aggressive"],
            [0],
            "model",
        )


@pytest.mark.parametrize("partition", ["ood_test", "validation_test", "mystery_split"])
def test_unknown_or_test_like_partition_cannot_enter_crossfit(tmp_path, partition):
    source = _source(tmp_path / "run.json", 0, "a" * 64, partition=partition)
    with pytest.raises(extract.LineageError, match="closed development allowlist"):
        extract._prepare_records(
            [_annotate(_record(), source)],
            [source],
            "officehome",
            ["sar_online_aggressive"],
            [0],
            "model",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("dataset", "wilds-iwildcam"), ("metric", "macro_f1"), ("schema", "legacy_v1")],
)
def test_track_source_contract_rejects_wrong_dataset_metric_or_schema(tmp_path, field, value):
    document = _office_contract_document(tmp_path)
    document[field] = value
    with pytest.raises(extract.LineageError, match="dataset/schema/metric contract mismatch"):
        extract._validate_source_contract(document, tmp_path / "source.json", "officehome")


def test_rxrx1_extractor_requires_current_balanced_accuracy_contract(tmp_path):
    assert extract._SOURCE_CONTRACTS["rxrx1"]["metric"] == "balanced_accuracy"
    legacy = {
        "schema": "kbound_rxrx1_v0.6",
        "dataset": "wilds-rxrx1",
        "metric": "accuracy",
    }
    with pytest.raises(extract.LineageError, match="metric contract mismatch"):
        extract._validate_source_contract(legacy, tmp_path / "legacy_rx.json", "rxrx1")


def _rxrx1_contract_document(tmp_path, *, model_seed, checkpoint_path, tensor_sha256):
    checkpoint_sha256 = extract._sha256_file(checkpoint_path)
    record = {
        "candidate": "sar_online",
        "method": "sar",
        "mode": "online",
        "model_seed": model_seed,
        "seed": 17,
        "split": "val",
        "domain": "rxrx1",
        "comp": "iid",
        "regime": "tiny",
        "aggr": "mild",
        "a0": 0.5,
        "aa": 0.55,
        "B": 0.05,
        "Z": [0.1, 0.2],
        "sample_provenance": {"ordered_eval_sample_ids_sha256": "e" * 64},
    }
    record["tta_protocol"] = extract._expected_tta_protocol(record)
    return {
        "schema": "kbound_rxrx1_v0.6",
        "dataset": "wilds-rxrx1",
        "metric": "balanced_accuracy",
        "model_identity": {
            "model_seed": model_seed,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_tensor_sha256": tensor_sha256,
        },
        "config": {
            "split": "val",
            "model_seed": model_seed,
            "stream_seeds": [17],
            "candidate_set": ["sar_online"],
            "metric": "balanced_accuracy",
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "tensor_sha256": tensor_sha256,
                "model_seed": model_seed,
            },
            "model_identity": {
                "model_seed": model_seed,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_tensor_sha256": tensor_sha256,
            },
            "implementation_sha256": {"runner": "r" * 64},
        },
        "data": {"split": "val", "population_sha256": "p" * 64},
        "records": [record],
        "publication_eligible": False,
    }


def test_rxrx1_shared_protocol_excludes_verified_model_identity(tmp_path):
    torch = pytest.importorskip("torch", exc_type=ImportError)
    documents = []
    for model_seed in [0, 1]:
        checkpoint_path = tmp_path / f"rx_seed{model_seed}.pt"
        torch.save(
            {"algorithm": {"model.weight": torch.tensor([float(model_seed), 1.0])}},
            checkpoint_path,
        )
        tensor_sha256 = extract._recompute_checkpoint_tensor_sha256(checkpoint_path)
        documents.append(
            _rxrx1_contract_document(
                tmp_path,
                model_seed=model_seed,
                checkpoint_path=checkpoint_path,
                tensor_sha256=tensor_sha256,
            )
        )
    fingerprints = [
        extract._validate_source_contract(document, tmp_path / f"source{index}.json", "rxrx1")[0]
        for index, document in enumerate(documents)
    ]
    assert fingerprints[0] == fingerprints[1]
    assert extract._model_seed(documents[1]) == 1
    assert extract._checkpoint_path(documents[1], tmp_path / "source1.json") == str(
        (tmp_path / "rx_seed1.pt").resolve()
    )
    assert extract._checkpoint_sha256(documents[1]) == extract._sha256_file(
        tmp_path / "rx_seed1.pt"
    )


def test_rxrx1_algorithm_checkpoint_tensor_state_is_hashed(tmp_path):
    torch = pytest.importorskip("torch", exc_type=ImportError)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    state = {"model.weight": torch.tensor([1.0, 2.0])}
    torch.save({"algorithm": state, "metadata": {"seed": 0}}, first)
    torch.save({"algorithm": state, "metadata": {"seed": 1}}, second)
    assert extract._sha256_file(first) != extract._sha256_file(second)
    assert (
        extract._recompute_checkpoint_tensor_sha256(first)
        == extract._recompute_checkpoint_tensor_sha256(second)
    )


def test_extractor_rejects_missing_or_contradictory_tta_protocol(tmp_path):
    source = _source(tmp_path / "run.json", 0, "a" * 64)
    record = _annotate(_record(seed=17), source)
    record["tta_protocol"] = {**record["tta_protocol"], "gradient_update_reads_eval_x": True}
    with pytest.raises(extract.LineageError, match="exact TTA data-use protocol"):
        extract._prepare_records(
            [record], [source], "officehome", ["sar_online_aggressive"], [0], "model"
        )


def test_model_seed_protocol_and_evaluation_population_must_match(tmp_path):
    sources = [
        _source(tmp_path / "seed0.json", 0, "a" * 64),
        _source(tmp_path / "seed1.json", 1, "b" * 64),
    ]
    sources[1]["protocol_fingerprint"] = "f" * 64
    records = [_annotate(_record(seed=17), source) for source in sources]
    with pytest.raises(extract.LineageError, match="scientific protocol"):
        extract._prepare_records(
            records, sources, "officehome", ["sar_online_aggressive"], [0, 1], "model"
        )

    sources[1]["protocol_fingerprint"] = sources[0]["protocol_fingerprint"]
    records = [_annotate(_record(seed=17), source) for source in sources]
    records[1]["_evaluation_population_sha256"] = "9" * 64
    with pytest.raises(extract.LineageError, match="evaluation sample population differs"):
        extract._prepare_records(
            records, sources, "officehome", ["sar_online_aggressive"], [0, 1], "model"
        )


def test_nonexistent_declared_checkpoint_cannot_enable_model_seed_inference(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(_office_contract_document(tmp_path)), encoding="utf-8")
    with pytest.raises(extract.LineageError, match="checkpoint.*existing artifact"):
        extract._load_records(
            [str(path)], track="officehome", verify_checkpoints=True
        )


def test_extractor_strict_loader_rejects_overflow_number(tmp_path):
    path = tmp_path / "overflow.json"
    path.write_text('{"config": {"x": 1e999}}\n', encoding="utf-8")
    with pytest.raises(extract.LineageError, match="non-finite JSON number"):
        extract._json_load(path)


def test_serializer_also_rejects_duplicate_condition_keys():
    records = [_record(), _record()]
    with pytest.raises(ValueError, match="duplicate scientific condition"):
        serializer.build_per_condition_records(
            records,
            "sar_online_aggressive",
            0,
            "officehome",
            prefer="numpy",
            method_field="candidate",
        )


def test_infeasible_exact_rank_radius_serializes_as_null_with_status(tmp_path):
    records = []
    for index, benefit in enumerate([-0.03, 0.01, 0.04]):
        record = _record(comp=f"cell{index}")
        record["aa"] = record["a0"] + benefit
        record["B"] = benefit
        record["Z"] = [float(index), benefit]
        records.append(record)
    result = serializer.serialize_run(
        records,
        dataset="officehome",
        out_dir=str(tmp_path),
        seeds=[0],
        methods=["sar_online_aggressive"],
        alpha=0.10,
        prefer="numpy",
        method_field="candidate",
    )
    document_path = Path(result["written"][0])
    raw = document_path.read_text(encoding="utf-8")
    document = json.loads(
        raw,
        parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)),
    )
    assert "Infinity" not in raw
    assert document["n_calibration_infeasible"] == 3
    for record in document["records"]:
        assert record["calibration_feasible"] is False
        assert record["radius_status"] == "INFEASIBLE"
        assert record["eps_conformal"] is None
        assert record["benefit_ci"] is None
        assert record["gamma_ci"] is None
        assert record["zone"] == "INFEASIBLE"
        assert record["kga_decision"] == "ABSTAIN"


def test_target_label_selected_panel_serializer_is_retired(tmp_path):
    records = [
        {
            "seed": 0,
            "domain": "imagenet_r",
            "comp": "iid",
            "regime": "tiny",
            "aggr": "mild",
            "candidate": "tent",
            "a0": 0.4,
            "aa": 0.5,
            "Z": [0.1, 0.2],
        }
    ]
    conditions = [
        {
            "seed": 0,
            "comp": "iid",
            "regime": "tiny",
            "aggr": "mild",
            "a0": 0.4,
            "aa_all": [0.4, 0.5],
            "cand_names": ["freeze_f0", "tent"],
        }
    ]
    with pytest.raises(RuntimeError, match="target-label-best candidate"):
        serializer.serialize_panel_run(
            records,
            conditions,
            dataset="imagenet-r",
            out_dir=str(tmp_path),
            seeds=[0],
            candidate_order=["tent"],
        )
    assert list(tmp_path.iterdir()) == []


def test_model_seed_inference_requires_unique_checkpoint_per_seed(tmp_path):
    sources = [
        _source(tmp_path / "seed0.json", 0, "a" * 64),
        _source(tmp_path / "seed1.json", 1, "a" * 64),
    ]
    records = [
        _annotate(_record(seed=0), sources[0]),
        _annotate(_record(seed=0), sources[1]),
    ]
    with pytest.raises(extract.LineageError, match="not unique across model seeds"):
        extract._prepare_records(
            records,
            sources,
            "officehome",
            ["sar_online_aggressive"],
            [0, 1],
            "model",
        )


def test_byte_different_checkpoint_containers_with_identical_tensor_state_are_rejected(
    tmp_path,
):
    torch = pytest.importorskip("torch", exc_type=ImportError)
    checkpoint_paths = []
    for model_seed in [0, 1]:
        path = tmp_path / f"same_state_seed{model_seed}.pt"
        torch.save(
            {
                "model": {"weight": torch.tensor([1.0, 2.0])},
                "container_metadata": {"declared_seed": model_seed},
            },
            path,
        )
        checkpoint_paths.append(path)
    file_hashes = [extract._sha256_file(path) for path in checkpoint_paths]
    tensor_hashes = [
        extract._recompute_checkpoint_tensor_sha256(path) for path in checkpoint_paths
    ]
    assert len(set(file_hashes)) == 2
    assert len(set(tensor_hashes)) == 1

    sources = [
        _source(checkpoint_paths[index].with_suffix(".json"), index, file_hashes[index])
        for index in [0, 1]
    ]
    for index, source in enumerate(sources):
        source["checkpoint_path"] = str(checkpoint_paths[index])
        source["checkpoint_tensor_sha256"] = tensor_hashes[index]
    records = [_annotate(_record(seed=17), source) for source in sources]
    with pytest.raises(extract.LineageError, match="same model are not independent"):
        extract._prepare_records(
            records,
            sources,
            "officehome",
            ["sar_online_aggressive"],
            [0, 1],
            "model",
        )


def test_serializer_requires_complete_cartesian_grid_before_writing(tmp_path):
    with pytest.raises(ValueError, match="exact requested method x seed grid"):
        serializer.serialize_run(
            [_record(seed=0)],
            dataset="officehome",
            out_dir=str(tmp_path),
            seeds=[0, 1],
            methods=["sar_online_aggressive"],
            prefer="numpy",
            method_field="candidate",
        )
    assert list(tmp_path.iterdir()) == []


def test_serializer_generation_manifest_commits_exact_files_and_refuses_reuse(tmp_path):
    records = []
    for cell in range(12):
        record = _record(seed=0, comp=f"cell{cell}")
        record["B"] = (cell - 5.5) / 100.0
        record["aa"] = record["a0"] + record["B"]
        record["Z"] = [cell / 12.0, record["B"]]
        records.append(record)
    result = serializer.serialize_run(
        records,
        dataset="officehome",
        out_dir=str(tmp_path),
        seeds=[0],
        methods=["sar_online_aggressive"],
        prefer="numpy",
        method_field="candidate",
    )
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["generation_committed"] is True
    assert manifest["generation_id"] == result["generation_id"]
    assert manifest["expected_cells"] == 1
    for name, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == serializer._file_sha256(tmp_path / name)
    with pytest.raises(ValueError, match="prior serialization generation"):
        serializer.serialize_run(
            records,
            dataset="officehome",
            out_dir=str(tmp_path),
            seeds=[0],
            methods=["sar_online_aggressive"],
            prefer="numpy",
            method_field="candidate",
        )


def test_model_seed_is_not_confused_with_fixed_stream_seed(tmp_path):
    sources = [
        _source(tmp_path / "seed0.json", 0, "a" * 64),
        _source(tmp_path / "seed1.json", 1, "b" * 64),
    ]
    records = [
        _annotate(_record(seed=7), sources[0]),
        _annotate(_record(seed=7), sources[1]),
    ]
    prepared, metadata, inference = extract._prepare_records(
        records,
        sources,
        "officehome",
        ["sar_online_aggressive"],
        [0, 1],
        "model",
    )
    assert [record["seed"] for record in prepared] == [0, 1]
    assert [record["stream_seed"] for record in prepared] == [7, 7]
    assert metadata[1]["checkpoint_sha256"] == "b" * 64
    assert inference["inference_unit"] == "independent model checkpoint"


def test_target_test_labels_are_rejected_from_crossfit_pool(tmp_path):
    source = _source(tmp_path / "test.json", 0, "a" * 64, partition="target_test")
    with pytest.raises(extract.LineageError, match="labels cannot enter the cross-fit"):
        extract._prepare_records(
            [_annotate(_record(), source)],
            [source],
            "officehome",
            ["sar_online_aggressive"],
            [0],
            "model",
        )


def test_incomplete_current_source_ledger_is_rejected(tmp_path):
    path = tmp_path / "incomplete.json"
    document = {
        "schema": "kbound_officehome_v4",
        "publication_eligible": True,
        "role": "target_val",
        "completion_ledger": {
            "status": "incomplete",
            "expected": 2,
            "completed": 1,
            "failed": 0,
            "pending": 1,
            "failed_cells": [],
            "failure_history": [],
        },
        "records": [_record()],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(extract.LineageError, match="completion ledger is incomplete"):
        extract._load_records([str(path)])


def test_complete_failure_free_source_ledger_is_marked_publication_ready(tmp_path):
    path = tmp_path / "complete.json"
    document = {
        "schema": "kbound_officehome_v4",
        "publication_eligible": True,
        "role": "target_val",
        "completion_ledger": {
            "status": "complete",
            "expected": 1,
            "completed": 1,
            "failed": 0,
            "pending": 0,
            "failed_cells": [],
            "pending_keys": [],
            "failure_history": [],
        },
        "conditions": [{"_key": ["cell0"]}],
        "records": [_record()],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    _, _, sources = extract._load_records([str(path)])
    assert sources[0]["publication"]["publication_ready"] is True
    assert sources[0]["publication"]["expected"] == 1
    assert sources[0]["publication"]["completed"] == 1


def test_legacy_schema_cannot_self_promote_with_forged_complete_ledger(tmp_path):
    path = tmp_path / "legacy.json"
    document = {
        "schema": "legacy_natural_result_v1",
        "publication_eligible": True,
        "role": "target_val",
        "completion_ledger": {
            "status": "complete",
            "expected": 1,
            "completed": 1,
            "failed": 0,
            "pending": 0,
        },
        "conditions": [{"_key": ["cell0"]}],
        "records": [_record()],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    _, _, sources = extract._load_records([str(path)])
    publication = sources[0]["publication"]
    assert publication["ledger_complete_failure_free"] is True
    assert publication["publication_ready"] is False
    assert publication["diagnostic_only_reason"] == (
        "legacy/unknown schema is diagnostic-only"
    )


def test_candidate_set_must_match_exactly(tmp_path):
    source = _source(tmp_path / "run.json", 0, "a" * 64)
    with pytest.raises(extract.LineageError, match="exact requested set"):
        extract._prepare_records(
            [_annotate(_record(candidate="tent_online_mild"), source)],
            [source],
            "officehome",
            ["sar_online_aggressive"],
            [0],
            "model",
        )


def test_aggregate_consumes_only_current_paths_not_stale_directory_files(tmp_path):
    current = [
        _per_condition(tmp_path / "current_seed0.json", seed=0, checkpoint="a" * 64),
        _per_condition(tmp_path / "current_seed1.json", seed=1, checkpoint="b" * 64),
    ]
    _per_condition(tmp_path / "stale_seed99.json", seed=99, checkpoint="c" * 64)
    inference = {
        "seed_kind": "model_seed",
        "inference_unit": "independent model checkpoint",
        "evaluation_partition": "target_val",
        "claim_scope": "development only",
        "tta_protocol_by_candidate": {
            "sar_online_aggressive": extract._expected_tta_protocol(
                {"candidate": "sar_online_aggressive", "mode": "online"}
            )
        },
    }
    result = extract.aggregate_candidate(
        "officehome",
        "sar_online_aggressive",
        current,
        expected_seeds=[0, 1],
        inference=inference,
    )
    assert result["seeds"] == [0, 1]
    assert result["n_seeds"] == 2
    assert "stale_seed99.json" not in result["files"]
    assert result["model_seed_ci_eligible"] is True
    assert result["confirmatory_ci_eligible"] is False
    assert result["heldout_promotion_eligible"] is False


def test_stream_seed_aggregate_is_explicitly_descriptive(tmp_path):
    current = [
        _per_condition(tmp_path / "stream0.json", seed=0, seed_kind="stream_seed"),
        _per_condition(tmp_path / "stream1.json", seed=1, seed_kind="stream_seed"),
    ]
    inference = {
        "seed_kind": "stream_seed",
        "inference_unit": "stream order (shared model)",
        "evaluation_partition": "val",
        "claim_scope": "descriptive only",
        "tta_protocol_by_candidate": {
            "sar_online_aggressive": extract._expected_tta_protocol(
                {"candidate": "sar_online_aggressive", "mode": "online"}
            )
        },
    }
    result = extract.aggregate_candidate(
        "officehome",
        "sar_online_aggressive",
        current,
        expected_seeds=[0, 1],
        inference=inference,
    )
    assert result["gap_vs_better_ci95"] is None
    assert result["development_beats_both"] is False
    assert result["beats_both_promoted"] is False
    assert result["verdict"] == "DESCRIPTIVE_STREAM_SEED_ONLY"


def test_any_infeasible_radius_withholds_multiseed_statistical_verdict(tmp_path):
    current = [
        _per_condition(
            tmp_path / "seed0.json", seed=0, checkpoint="a" * 64, feasible=False
        ),
        _per_condition(
            tmp_path / "seed1.json", seed=1, checkpoint="b" * 64, feasible=False
        ),
    ]
    inference = {
        "seed_kind": "model_seed",
        "inference_unit": "independent model checkpoint",
        "evaluation_partition": "target_val",
        "claim_scope": "development only",
        "sources_publication_ready": True,
        "tta_protocol_by_candidate": {
            "sar_online_aggressive": extract._expected_tta_protocol(
                {"candidate": "sar_online_aggressive", "mode": "online"}
            )
        },
    }
    result = extract.aggregate_candidate(
        "officehome",
        "sar_online_aggressive",
        current,
        expected_seeds=[0, 1],
        inference=inference,
    )
    assert result["calibration_feasible_all"] is False
    assert result["n_calibration_infeasible_total"] == 4
    assert result["statistical_verdict_withheld"] is True
    assert result["gap_vs_better_ci95"] is None
    assert result["gap_vs_adapt"]["ci95"] is None
    assert result["model_seed_ci_eligible"] is False
    assert result["development_beats_both"] is False
    assert all(result["all_abstain_due_infeasible_per_seed"].values())


def test_constant_positive_pool_below_exact_rank_minimum_is_infeasible(tmp_path):
    records = []
    for index in range(9):
        record = _record(comp=f"cell{index}")
        record["Z"] = [float(index), float(index % 2)]
        records.append(record)
    result = serializer.serialize_run(
        records,
        dataset="officehome",
        out_dir=str(tmp_path),
        seeds=[0],
        methods=["sar_online_aggressive"],
        alpha=0.10,
        prefer="numpy",
        method_field="candidate",
    )
    document = json.loads(Path(result["written"][0]).read_text())
    assert document["kga_backend"] == "infeasible_undersized_exact_rank"
    assert document["n_calibration_infeasible"] == 9
    assert all(record["eps_conformal"] is None for record in document["records"])
    assert all(record["zone"] == "INFEASIBLE" for record in document["records"])
    assert all(record["kga_decision"] == "ABSTAIN" for record in document["records"])


def test_crossfit_boundary_total_twelve_is_finite_when_benefit_varies():
    z = [[float(index), float(index % 3)] for index in range(12)]
    benefit = [index / 100.0 for index in range(12)]
    _, epsilon, _, backend = serializer.decide_benefit(z, benefit, prefer="numpy")
    assert backend == "numpy_knn_fallback"
    assert all(float(value) < float("inf") for value in epsilon)


def test_auto_fallback_is_explicit_diagnostic_and_never_swallows_scientific_errors(
    monkeypatch,
):
    z = [[float(index), float(index % 3)] for index in range(12)]
    benefit = [index / 100.0 for index in range(12)]

    def missing_dependency(*_args, **_kwargs):
        raise ModuleNotFoundError("sklearn unavailable")

    monkeypatch.setattr(serializer, "_decide_kga_sklearn", missing_dependency)
    with pytest.raises(ModuleNotFoundError):
        serializer.decide_benefit(z, benefit, prefer="auto")
    _, _, _, backend = serializer.decide_benefit(
        z,
        benefit,
        prefer="auto",
        allow_diagnostic_fallback=True,
    )
    assert backend == "numpy_knn_fallback_diagnostic"

    def scientific_failure(*_args, **_kwargs):
        raise RuntimeError("scientific estimator failure")

    monkeypatch.setattr(serializer, "_decide_kga_sklearn", scientific_failure)
    with pytest.raises(RuntimeError, match="scientific estimator failure"):
        serializer.decide_benefit(
            z,
            benefit,
            prefer="auto",
            allow_diagnostic_fallback=True,
        )


def test_numpy_fallback_aggregate_is_never_publication_eligible(tmp_path):
    backend = "numpy_knn_fallback_diagnostic"
    current = [
        _per_condition(
            tmp_path / "seed0.json", seed=0, checkpoint="a" * 64, backend=backend
        ),
        _per_condition(
            tmp_path / "seed1.json", seed=1, checkpoint="b" * 64, backend=backend
        ),
    ]
    inference = {
        "seed_kind": "model_seed",
        "inference_unit": "independent model checkpoint",
        "evaluation_partition": "target_val",
        "claim_scope": "development only",
        "sources_publication_ready": True,
        "tta_protocol_by_candidate": {
            "sar_online_aggressive": extract._expected_tta_protocol(
                {"candidate": "sar_online_aggressive", "mode": "online"}
            )
        },
    }
    result = extract.aggregate_candidate(
        "officehome",
        "sar_online_aggressive",
        current,
        expected_seeds=[0, 1],
        inference=inference,
    )
    assert result["kga_backend"] == [backend]
    assert result["estimator_publication_eligible"] is False
    assert result["publication_eligible"] is False
    assert result["heldout_promotion_eligible"] is False


def test_no_current_candidate_files_means_no_aggregate():
    inference = {
        "seed_kind": "stream_seed",
        "inference_unit": "stream order",
        "evaluation_partition": "val",
        "claim_scope": "descriptive only",
        "tta_protocol_by_candidate": {
            "sar_online_aggressive": extract._expected_tta_protocol(
                {"candidate": "sar_online_aggressive", "mode": "online"}
            )
        },
    }
    with pytest.raises(extract.LineageError, match="produced no files"):
        extract.aggregate_candidate(
            "iwildcam",
            "tent_episodic",
            [],
            expected_seeds=[0],
            inference=inference,
        )


def test_atomic_directory_publish_removes_stale_derived_outputs(tmp_path):
    destination = tmp_path / "extracted"
    destination.mkdir()
    (destination / "stale_aggregate.json").write_text("stale", encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "current_manifest.json").write_text("current", encoding="utf-8")
    extract._publish_staged_directory(stage, destination)
    assert not (destination / "stale_aggregate.json").exists()
    assert (destination / "current_manifest.json").read_text(encoding="utf-8") == "current"


def test_end_to_end_model_seed_extraction_publishes_only_fresh_outputs(tmp_path):
    torch = pytest.importorskip("torch", exc_type=ImportError)
    sources = []
    checkpoint_file_hashes = {}
    population_manifest = {
        "Art/val": {"n": 100, "ordered_sample_label_sha256": "e" * 64}
    }
    for model_seed in [0, 1]:
        checkpoint_path = tmp_path / f"checkpoint_seed{model_seed}.pt"
        torch.save(
            {"model": {"weight": torch.tensor([float(model_seed), 1.0])}},
            checkpoint_path,
        )
        checkpoint_file_sha = extract._sha256_file(checkpoint_path)
        checkpoint_file_hashes[str(model_seed)] = checkpoint_file_sha
        checkpoint_tensor_sha = extract._recompute_checkpoint_tensor_sha256(checkpoint_path)
        records = []
        for cell in range(12):
            benefit = (cell - 5.5) / 100.0 + model_seed / 1000.0
            records.append(
                {
                    "model_seed": model_seed,
                    "seed": 17,
                    "candidate": "sar_online_aggressive",
                    "method": "sar",
                    "mode": "online",
                    "domain": "Art",
                    "split": "val",
                    "comp": f"cell{cell}",
                    "regime": "tiny",
                    "a0": 0.5,
                    "aa": 0.5 + benefit,
                        "B": benefit,
                        "Z": [cell / 12.0, benefit],
                        "sample_provenance": {
                            "ordered_eval_sample_ids_sha256": extract._sha256_json(
                                ["Art", "val", f"cell{cell}"]
                            )
                        },
                        "tta_protocol": extract._expected_tta_protocol(
                            {"candidate": "sar_online_aggressive", "mode": "online"}
                        ),
                    }
                )
            document = {
                "schema": "kbound_officehome_v4",
                "dataset": "office-home",
                "metric": "accuracy",
                "role": "target_val",
                "model_seed": model_seed,
                "f0_checkpoint": str(checkpoint_path),
                "f0_checkpoint_sha256": checkpoint_file_sha,
                "f0_checkpoint_tensor_sha256": checkpoint_tensor_sha,
                "publication_eligible": False,
                "population_manifest": population_manifest,
                "resume_contract": {
                    "schema": "test_resume_contract_v1",
                    "payload": {
                        "dataset": "office-home",
                        "implementation_sha256": {"runner": "1" * 64},
                        "role": "target_val",
                        "candidate_set_ordered": ["sar_online_aggressive"],
                        "split_manifest": {"sha256": "2" * 64},
                        "checkpoint": {
                            "path": str(checkpoint_path),
                            "file_sha256": checkpoint_file_sha,
                            "tensor_sha256": checkpoint_tensor_sha,
                        },
                        "seed_semantics": {
                            "model_seed": model_seed,
                            "stream_seeds_ordered": [17],
                            "condition_seed_rule": "model-seed invariant",
                        },
                        "scientific_config": {
                            "n_eval": 100,
                            "metric": "accuracy",
                        },
                    },
                },
                "evidence_names": ["cell", "benefit_proxy"],
                "records": records,
            }
        path = tmp_path / f"source_seed{model_seed}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        sources.append(path)

    destination = tmp_path / "extracted"
    destination.mkdir()
    (destination / "multiseed_officehome_stale.json").write_text("stale", encoding="utf-8")
    assert (
        extract.main(
            [
                "--track",
                "officehome",
                "--result",
                *(str(path) for path in sources),
                "--candidates",
                "sar_online_aggressive",
                "--expected-seeds",
                "0",
                "1",
                "--seed-kind",
                "model",
                "--out-dir",
                str(destination),
                "--prefer",
                "numpy",
            ]
        )
        == 0
    )
    assert not (destination / "multiseed_officehome_stale.json").exists()
    aggregate = json.loads(
        (destination / "multiseed_officehome_sar_online_aggressive.json").read_text()
    )
    manifest = json.loads((destination / "extract_manifest_officehome.json").read_text())
    assert aggregate["seeds"] == [0, 1]
    assert aggregate["schema"] == "kbound_natural_multiseed_aggregate_v2"
    assert aggregate["lineage_contract"] == "leakage_safe_multiseed_v2"
    assert aggregate["checkpoint_sha256_by_seed"] == checkpoint_file_hashes
    assert aggregate["heldout_promotion_eligible"] is False
    assert aggregate["sources_publication_ready"] is False
    assert aggregate["estimator_publication_eligible"] is False
    assert aggregate["publication_eligible"] is False
    assert manifest["requested_candidates"] == ["sar_online_aggressive"]
    assert all(
        source["publication"]["publication_ready"] is False
        for source in manifest["sources"]
    )
    assert manifest["aggregate_sha256"][
        "multiseed_officehome_sar_online_aggressive.json"
    ] == extract._sha256_file(
        destination / "multiseed_officehome_sar_online_aggressive.json"
    )
    loaded = forest._load_aggs(
        [str(destination / "multiseed_officehome_sar_online_aggressive.json")],
        scope="development-diagnostic",
    )
    assert len(loaded) == 1
    assert loaded[0][1]["external_authenticity_verified"] is False
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        [
                "extract_manifest_officehome.json",
                "multiseed_officehome_sar_online_aggressive.json",
                "per_condition_officehome_manifest.json",
                "per_condition_officehome_sar_online_aggressive_seed0.json",
            "per_condition_officehome_sar_online_aggressive_seed1.json",
        ]
    )
