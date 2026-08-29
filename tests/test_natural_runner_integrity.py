from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OH = _load("officehome_integrity_runner", "experiments/kbound/officehome/run_officehome_kbound.py")
IWC = _load("iwildcam_integrity_runner", "experiments/kbound/wilds/run_iwildcam_kbound.py")


def _iwc_sampling_context():
    labels = np.tile(np.array([0, 1], dtype=int), 8)
    locations = np.full(len(labels), 7, dtype=int)
    subset = SimpleNamespace(indices=np.arange(100, 100 + len(labels), dtype=np.int64))
    return subset, labels, locations


def _office_sampling_context():
    target = [[f"art-{index}.jpg", int(index % 2)] for index in range(16)]
    source = [[f"source-{index}.jpg", int(index % 2)] for index in range(4)]
    return {
        "splits": {
            "Art": {"val": target},
            OH.ohd.SOURCE: {"train": source},
        }
    }


def _iwc_contract():
    return {
        "schema": IWC.RESUME_CONTRACT_SCHEMA,
        "sha256": "c",
        "payload": {
            "checkpoint": {"tensor_sha256": "a" * 64},
            "candidate_set_ordered": ["tent_online"],
            "scientific_config": {
                "tau_star": 0.52,
                "kappa": 2.5,
                "batch_sizes": {"tiny": 8},
                "n_batches": 1,
                "n_eval": 4,
            },
        },
    }


def _office_contract():
    return {
        "schema": OH.RESUME_CONTRACT_SCHEMA,
        "sha256": "c",
        "payload": {
            "checkpoint": {"tensor_sha256": "t" * 64},
            "candidate_set_ordered": ["sar_online_mild"],
            "population_manifest": {"splits": {f"{OH.ohd.SOURCE}/train": {"n": 4}}},
            "scientific_config": {
                "tau_star": 0.52,
                "kappa": 2.5,
                "batch_sizes": {"tiny": 8},
                "n_batches": 1,
                "n_eval": 4,
                "source_reference_n": 512,
            },
        },
    }


def _iwc_valid_partial(contract, key, sampling_context=None):
    sub, labels, locations = sampling_context or _iwc_sampling_context()
    expected = IWC._expected_iwc_resume_samples(
        key, contract, sub, labels, locations
    )
    provenance = expected["sample_provenance"]
    eval_y = expected["eval_y"].astype(int).tolist()
    frozen = list(eval_y)
    adapted = list(eval_y)
    adapted[-1] = (adapted[-1] + 1) % IWC.NUM_CLASSES
    a0 = IWC.macro_f1(eval_y, frozen)
    aa = IWC.macro_f1(eval_y, adapted)
    identity = {
        "resume_contract_sha256": key[0], "checkpoint_tensor_sha256": key[2],
        "model_seed": key[3], "seed": key[4], "location": key[5],
        "domain": f"loc{key[5]}", "split": key[1], "comp": key[6],
        "regime": key[7], "aggr": key[8],
    }
    scientific_identity = IWC._iwc_scientific_cell_identity(key)
    cell_id = IWC._iwc_cell_id(key)
    route = IWC.an.multicandidate_route(
        np.asarray([frozen, adapted]), tau_star=0.52, kappa=2.5,
        task_type="multiclass_classification", n_classes=IWC.NUM_CLASSES,
        objective="macro_f1", anchor_above_chance=False,
    )
    record = {
        "_cell_key": list(key), "cell_id": cell_id,
        "scientific_cell_identity": scientific_identity,
        **identity, "candidate": "tent_online",
        "location_n": expected["location_n"],
        "location_classes": expected["location_classes"],
        "method": "tent", "mode": "online", "metric": "macro_f1",
        "tta_protocol": IWC.tm.tta_protocol_contract("online"),
        "a0": a0, "aa": aa, "B": aa - a0, "preds": adapted,
        "upd_norm": 0.0, "Z": [0.0] * len(IWC.tm.EVIDENCE_NAMES),
        "c0": [int(value) for value in np.asarray(frozen) == np.asarray(eval_y)],
        "ca": [int(value) for value in np.asarray(adapted) == np.asarray(eval_y)],
        "sample_provenance": provenance, "regime_label": IWC.an.label_regime(aa - a0),
    }
    condition = {
        "_key": list(key), "cell_id": cell_id,
        "scientific_cell_identity": scientific_identity,
        **identity, "cand_names": ["freeze_f0", "tent_online"],
        "location_n": expected["location_n"],
        "location_classes": expected["location_classes"],
        "aa_all": [a0, aa], "a0": a0, "oracle": max(a0, aa),
        "best_adapt": aa, "true_best": "freeze_f0", "metric": "macro_f1",
        "route": route, "route_b_eligible": False,
        "route_objective": {
            "metric": "macro_f1", "n_classes": IWC.NUM_CLASSES,
            "status": "UNSUPPORTED_BINARY_ACCURACY_IDENTITY",
        },
        "route_c": IWC.rc.unsupported_route_c("macro_f1", IWC.NUM_CLASSES),
        "realized": None, "eval_y": eval_y, "preds_frozen": frozen,
        "sample_provenance": provenance,
        "regime_label": IWC.an.label_regime(aa - a0),
    }
    return record, condition


def _office_valid_partial(contract, key, splits=None):
    splits = splits or _office_sampling_context()
    expected = OH._expected_officehome_resume_samples(key, contract, splits)
    provenance = expected["sample_provenance"]
    eval_y = expected["eval_y"].astype(int).tolist()
    frozen = list(eval_y)
    adapted = list(eval_y)
    adapted[-1] = (adapted[-1] + 1) % OH.NUM_CLASSES
    a0 = 1.0
    aa = 0.75
    identity = {
        "resume_contract_sha256": key[0],
        "checkpoint_tensor_sha256": contract["payload"]["checkpoint"]["tensor_sha256"],
        "model_seed": key[1], "seed": key[2], "role": key[3], "domain": key[4],
        "split": key[5], "comp": key[6], "regime": key[7],
    }
    scientific_identity = OH._officehome_scientific_cell_identity(key)
    cell_id = OH._officehome_cell_id(key)
    route = OH.an.multicandidate_route(
        np.asarray([frozen, adapted]), tau_star=0.52, kappa=2.5,
        objective="accuracy", n_classes=OH.NUM_CLASSES, anchor_above_chance=False,
    )
    record = {
        "_cell_key": list(key), "cell_id": cell_id,
        "scientific_cell_identity": scientific_identity,
        **identity, "candidate": "sar_online_mild",
        "tta_protocol": OH.tm.tta_protocol_contract("online"),
        "metric": "accuracy", "a0": a0, "aa": aa, "B": aa - a0,
        "upd_norm": 0.0, "Z": [0.0] * len(OH.ohc.EVIDENCE_NAMES_OH),
        "preds": adapted, "sample_provenance": provenance,
        "regime_label": OH.an.label_regime(aa - a0),
    }
    condition = {
        "_key": list(key), "cell_id": cell_id,
        "scientific_cell_identity": scientific_identity, **identity,
        "cand_names": ["freeze_f0", "sar_online_mild"], "aa_all": [a0, aa],
        "a0": a0, "oracle": a0, "best_adapt": aa, "true_best": "freeze_f0",
        "route": route, "route_c": OH.rc.unsupported_route_c("accuracy", OH.NUM_CLASSES),
        "route_b_eligible": False, "realized": None,
        "route_objective": {
            "metric": "accuracy", "n_classes": OH.NUM_CLASSES, "anchor_above_chance": False,
        },
        "eval_y": eval_y, "preds_frozen": frozen, "sample_provenance": provenance,
        "source_reference_provenance": expected["source_reference_provenance"],
        "regime_label": OH.an.label_regime(aa - a0),
    }
    return record, condition


def test_iwildcam_macro_f1_uses_only_official_labels_present_in_y_true():
    y_true = np.array([0, 0, 1, 1])
    preds = np.array([0, 2, 1, 2])  # predicted-only class 2 is not a target label

    assert IWC.macro_f1(y_true, preds) == pytest.approx(2.0 / 3.0)


def test_iwildcam_macro_f1_matches_wilds_f1_metric():
    metrics = pytest.importorskip("wilds.common.metrics.all_metrics")
    y_true = torch.tensor([0, 0, 1, 1])
    pred_labels = torch.tensor([0, 2, 1, 2])
    logits = torch.full((len(y_true), 3), -10.0)
    logits[torch.arange(len(y_true)), pred_labels] = 10.0
    official = metrics.F1(
        prediction_fn=metrics.multiclass_logits_to_pred,
        average="macro",
    ).compute(logits, y_true)["F1-macro_all"]

    assert IWC.macro_f1(y_true.numpy(), pred_labels.numpy()) == pytest.approx(
        float(official), abs=1e-7
    )


def test_iwildcam_done_key_includes_split_and_checkpoint():
    base = IWC._iwc_cell_key("contract", "val", "a" * 64, 0, 1, 7, "iid", "tiny", "mild")
    other_split = IWC._iwc_cell_key(
        "contract", "test", "a" * 64, 0, 1, 7, "iid", "tiny", "mild"
    )
    other_checkpoint = IWC._iwc_cell_key(
        "contract", "val", "b" * 64, 0, 1, 7, "iid", "tiny", "mild"
    )

    assert base != other_split
    assert base != other_checkpoint
    assert base[1] == "val"
    assert base[2] == "a" * 64


def test_scientific_contracts_bind_required_identities(tmp_path):
    oh_args = OH.parse_args(["--role", "target_val"])
    oh_args.ckpt = str(tmp_path / "office.pt")
    oh_args.splits = str(tmp_path / "splits.json")
    oh_contract = OH.build_resume_contract(
        oh_args, "s" * 64, "f" * 64, "t" * 64,
        {"sha256": "p" * 64, "splits": {}},
    )
    assert oh_contract["payload"]["role"] == "target_val"
    assert oh_contract["payload"]["split_manifest"]["sha256"] == "s" * 64
    assert oh_contract["payload"]["checkpoint"]["tensor_sha256"] == "t" * 64
    assert oh_contract["payload"]["population_manifest"]["sha256"] == "p" * 64
    assert oh_contract["payload"]["candidate_set_ordered"] == oh_args.candidates
    assert oh_contract["payload"]["seed_semantics"]["model_seed"] == oh_args.model_seed
    assert oh_contract["payload"]["scientific_config"]["route_c_contract"]["status"] == "UNSUPPORTED"

    iwc_args = IWC.parse_args([])
    population = {
        "split": "val", "n": 3, "official_sample_ids_sha256": "i" * 64,
        "labels_sha256": "l" * 64, "locations_sha256": "g" * 64,
        "manifest_sha256": "m" * 64,
        "content_identity_status": "VERIFIED",
        "ordered_content_manifest_sha256": "h" * 64,
    }
    iwc_contract = IWC.build_resume_contract(
        iwc_args, population, tmp_path / "iwc.pt", "f" * 64, "t" * 64
    )
    assert iwc_contract["payload"]["split"] == "val"
    assert iwc_contract["payload"]["split_manifest"] == population
    assert iwc_contract["payload"]["checkpoint"]["tensor_sha256"] == "t" * 64
    assert iwc_contract["payload"]["candidate_set_ordered"] == iwc_args.candidates
    assert iwc_contract["payload"]["seed_semantics"]["model_seed"] == iwc_args.train_seed
    assert iwc_contract["payload"]["scientific_config"]["route_c_contract"]["status"] == "UNSUPPORTED"


@pytest.mark.parametrize("module", [OH, IWC])
def test_resume_contract_refuses_legacy_and_mismatch(module, tmp_path):
    expected = {
        "schema": module.RESUME_CONTRACT_SCHEMA,
        "sha256": "expected",
        "payload": {"split": "val", "candidate_set_ordered": ["sar_online"]},
    }
    with pytest.raises(RuntimeError, match="legacy resume"):
        module.validate_resume_contract({}, expected, tmp_path / "_partial.json")

    mismatched = {
        "resume_contract": {
            **expected,
            "sha256": "different",
            "payload": {"split": "test", "candidate_set_ordered": ["sar_online"]},
        }
    }
    with pytest.raises(RuntimeError, match="mismatched resume"):
        module.validate_resume_contract(mismatched, expected, tmp_path / "_partial.json")


def test_iwildcam_partial_requires_exact_candidate_transaction(tmp_path):
    contract = _iwc_contract()
    key = IWC._iwc_cell_key("c", "val", "a" * 64, 0, 0, 7, "iid", "tiny", "mild")
    sampling_context = _iwc_sampling_context()
    record, condition = _iwc_valid_partial(contract, key, sampling_context)
    partial = tmp_path / "_partial.json"
    partial.write_text(json.dumps(IWC._partial_payload(
        contract, [key], [record], [condition], {key}, {}, [], 0.0,
    )))

    loaded = IWC.load_partial_iwc(
        partial, contract, [key], [("tent", "online")],
        sub=sampling_context[0], y=sampling_context[1], locations=sampling_context[2],
    )
    assert loaded[2] == {key}

    doc = json.loads(partial.read_text())
    doc["records"].append({**record, "candidate": "unexpected"})
    partial.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="incomplete candidate transaction"):
        IWC.load_partial_iwc(
            partial, contract, [key], [("tent", "online")],
            sub=sampling_context[0], y=sampling_context[1], locations=sampling_context[2],
        )


def test_officehome_partial_requires_exact_candidate_transaction(tmp_path):
    contract = _office_contract()
    key = OH._officehome_cell_key("c", 0, 1, "target_val", "Art", "val", "iid", "tiny")
    splits = _office_sampling_context()
    record, condition = _office_valid_partial(contract, key, splits)
    partial = tmp_path / "_partial.json"
    partial.write_text(json.dumps(OH._partial_payload(
        contract, [key], [record], [condition], {key}, {}, [], 0.0,
    )))

    loaded = OH._load_partial(
        partial, contract, [key], ["sar_online_mild"], splits=splits
    )
    assert loaded[2] == {key}

    doc = json.loads(partial.read_text())
    doc["records"] = []
    partial.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="incomplete candidate transaction"):
        OH._load_partial(
            partial, contract, [key], ["sar_online_mild"], splits=splits
        )


@pytest.mark.parametrize("module_name", ["officehome", "iwildcam"])
def test_custom_partial_resume_rejects_score_and_route_tampering(module_name, tmp_path):
    if module_name == "officehome":
        module = OH
        contract = _office_contract()
        key = OH._officehome_cell_key("c", 0, 1, "target_val", "Art", "val", "iid", "tiny")
        sampling_context = _office_sampling_context()
        record, condition = _office_valid_partial(contract, key, sampling_context)
        loader = lambda path: module._load_partial(
            path, contract, [key], ["sar_online_mild"], splits=sampling_context
        )
    else:
        module = IWC
        contract = _iwc_contract()
        key = IWC._iwc_cell_key("c", "val", "a" * 64, 0, 0, 7, "iid", "tiny", "mild")
        sampling_context = _iwc_sampling_context()
        record, condition = _iwc_valid_partial(contract, key, sampling_context)
        loader = lambda path: module.load_partial_iwc(
            path, contract, [key], [("tent", "online")],
            sub=sampling_context[0], y=sampling_context[1], locations=sampling_context[2],
        )

    partial = tmp_path / f"{module_name}.json"
    base = module._partial_payload(
        contract, [key], [record], [condition], {key}, {}, [], 0.0,
    )
    for mutation in ("benefit", "summary", "route", "identity"):
        doc = json.loads(json.dumps(base))
        if mutation == "benefit":
            doc["records"][0]["B"] += 0.01
        elif mutation == "summary":
            doc["conditions"][0]["oracle"] -= 0.1
        elif mutation == "route":
            doc["conditions"][0]["route"]["status"] = "OK"
        else:
            doc["conditions"][0]["sample_provenance"]["stream_eval_disjoint"] = False
        partial.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(RuntimeError):
            loader(partial)


@pytest.mark.parametrize("module_name", ["officehome", "iwildcam"])
@pytest.mark.parametrize("mutation", ["evidence", "update_norm", "protocol", "cell_id", "seed"])
def test_custom_resume_rejects_adversarial_payload_even_with_rehashed_inventory(
    module_name, mutation, tmp_path
):
    if module_name == "officehome":
        module = OH
        contract = _office_contract()
        key = OH._officehome_cell_key("c", 0, 1, "target_val", "Art", "val", "iid", "tiny")
        sampling_context = _office_sampling_context()
        record, condition = _office_valid_partial(contract, key, sampling_context)
        loader = lambda path: module._load_partial(
            path, contract, [key], ["sar_online_mild"], splits=sampling_context
        )
        inventory = module._officehome_record_inventory
    else:
        module = IWC
        contract = _iwc_contract()
        key = IWC._iwc_cell_key("c", "val", "a" * 64, 0, 0, 7, "iid", "tiny", "mild")
        sampling_context = _iwc_sampling_context()
        record, condition = _iwc_valid_partial(contract, key, sampling_context)
        loader = lambda path: module.load_partial_iwc(
            path, contract, [key], [("tent", "online")],
            sub=sampling_context[0], y=sampling_context[1], locations=sampling_context[2],
        )
        inventory = module._iwc_record_inventory

    doc = module._partial_payload(
        contract, [key], [record], [condition], {key}, {}, [], 0.0,
    )
    if mutation == "evidence":
        doc["records"][0]["Z"].pop()
    elif mutation == "update_norm":
        doc["records"][0]["upd_norm"] = 0.25
    elif mutation == "protocol":
        doc["records"][0]["tta_protocol"]["mode"] = "tampered"
    elif mutation == "cell_id":
        doc["records"][0]["cell_id"] = "0" * 64
        doc["conditions"][0]["cell_id"] = "0" * 64
    else:
        doc["records"][0]["sample_provenance"]["condition_seed"] += 1
        doc["conditions"][0]["sample_provenance"]["condition_seed"] += 1
    doc["record_inventory"] = inventory(doc["records"], doc["conditions"])
    partial = tmp_path / f"{module_name}-{mutation}.json"
    partial.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises((RuntimeError, module.ri.RunIntegrityError)):
        loader(partial)


@pytest.mark.parametrize("module_name", ["officehome", "iwildcam"])
@pytest.mark.parametrize("mutation", ["sample_identity", "labels_and_predictions"])
def test_custom_resume_recomputes_live_sample_identity_against_resealed_payload(
    module_name, mutation, tmp_path
):
    if module_name == "officehome":
        module = OH
        contract = _office_contract()
        key = OH._officehome_cell_key("c", 0, 1, "target_val", "Art", "val", "iid", "tiny")
        sampling_context = _office_sampling_context()
        record, condition = _office_valid_partial(contract, key, sampling_context)
        loader = lambda path: module._load_partial(
            path, contract, [key], ["sar_online_mild"], splits=sampling_context
        )
        inventory = module._officehome_record_inventory
    else:
        module = IWC
        contract = _iwc_contract()
        key = IWC._iwc_cell_key("c", "val", "a" * 64, 0, 0, 7, "iid", "tiny", "mild")
        sampling_context = _iwc_sampling_context()
        record, condition = _iwc_valid_partial(contract, key, sampling_context)
        loader = lambda path: module.load_partial_iwc(
            path, contract, [key], [("tent", "online")],
            sub=sampling_context[0], y=sampling_context[1], locations=sampling_context[2],
        )
        inventory = module._iwc_record_inventory

    doc = module._partial_payload(
        contract, [key], [record], [condition], {key}, {}, [], 0.0,
    )
    archived_record = doc["records"][0]
    archived_condition = doc["conditions"][0]
    if mutation == "sample_identity":
        provenance = archived_condition["sample_provenance"]
        if module_name == "officehome":
            for field in (
                "ordered_eval_split_positions_sha256",
                "ordered_eval_requested_split_positions_sha256",
                "ordered_eval_resolved_split_positions_sha256",
            ):
                provenance[field] = "9" * 64
            provenance["ordered_eval_sample_ids_sha256"] = "8" * 64
        else:
            provenance["ordered_eval_requested_subset_positions_sha256"] = "9" * 64
            provenance["ordered_eval_resolved_subset_positions_sha256"] = "9" * 64
            provenance["ordered_eval_official_sample_ids_sha256"] = "8" * 64
        archived_record["sample_provenance"] = json.loads(json.dumps(provenance))
        expected_error = "deterministic sample provenance"
    else:
        altered_labels = [int((value + 1) % 2) for value in archived_condition["eval_y"]]
        frozen = list(altered_labels)
        adapted = list(altered_labels)
        adapted[-1] = int((adapted[-1] + 1) % module.NUM_CLASSES)
        if module_name == "officehome":
            a0 = module._acc(frozen, altered_labels)
            aa = module._acc(adapted, altered_labels)
            route = module.an.multicandidate_route(
                np.asarray([frozen, adapted]), tau_star=0.52, kappa=2.5,
                objective="accuracy", n_classes=module.NUM_CLASSES,
                anchor_above_chance=False,
            )
        else:
            a0 = module.macro_f1(altered_labels, frozen)
            aa = module.macro_f1(altered_labels, adapted)
            route = module.an.multicandidate_route(
                np.asarray([frozen, adapted]), tau_star=0.52, kappa=2.5,
                task_type="multiclass_classification", n_classes=module.NUM_CLASSES,
                objective="macro_f1", anchor_above_chance=False,
            )
            archived_record["c0"] = [
                int(value) for value in np.asarray(frozen) == np.asarray(altered_labels)
            ]
            archived_record["ca"] = [
                int(value) for value in np.asarray(adapted) == np.asarray(altered_labels)
            ]
            archived_record["a0_acc"] = float(
                np.mean(np.asarray(frozen) == np.asarray(altered_labels))
            )
            archived_record["aa_acc"] = float(
                np.mean(np.asarray(adapted) == np.asarray(altered_labels))
            )
            archived_record["a0_bacc"] = module.tm.balanced_acc(
                np.asarray(frozen), np.asarray(altered_labels)
            )
            archived_record["aa_bacc"] = module.tm.balanced_acc(
                np.asarray(adapted), np.asarray(altered_labels)
            )
        archived_record.update({
            "a0": a0,
            "aa": aa,
            "B": aa - a0,
            "preds": adapted,
            "regime_label": module.an.label_regime(aa - a0),
        })
        names = archived_condition["cand_names"]
        scores = [a0, aa]
        archived_condition.update({
            "eval_y": altered_labels,
            "preds_frozen": frozen,
            "a0": a0,
            "aa_all": scores,
            "oracle": max(scores),
            "best_adapt": aa,
            "true_best": names[int(np.argmax(scores))],
            "route": route,
            "regime_label": module.an.label_regime(aa - a0),
        })
        expected_error = "evaluation labels differ from the current"

    doc["record_inventory"] = inventory(doc["records"], doc["conditions"])
    partial = tmp_path / f"{module_name}-{mutation}-resealed.json"
    partial.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(RuntimeError, match=expected_error):
        loader(partial)


@pytest.mark.parametrize("module_name", ["officehome", "iwildcam"])
def test_custom_resume_refuses_completed_cells_without_live_sampling_context(
    module_name, tmp_path
):
    if module_name == "officehome":
        module = OH
        contract = _office_contract()
        key = OH._officehome_cell_key("c", 0, 1, "target_val", "Art", "val", "iid", "tiny")
        record, condition = _office_valid_partial(contract, key)
        loader = lambda path: module._load_partial(
            path, contract, [key], ["sar_online_mild"]
        )
    else:
        module = IWC
        contract = _iwc_contract()
        key = IWC._iwc_cell_key("c", "val", "a" * 64, 0, 0, 7, "iid", "tiny", "mild")
        record, condition = _iwc_valid_partial(contract, key)
        loader = lambda path: module.load_partial_iwc(
            path, contract, [key], [("tent", "online")]
        )
    partial = tmp_path / f"{module_name}-no-live-context.json"
    partial.write_text(json.dumps(module._partial_payload(
        contract, [key], [record], [condition], {key}, {}, [], 0.0,
    )), encoding="utf-8")

    with pytest.raises(module.ri.RunIntegrityError, match="requires the current"):
        loader(partial)


@pytest.mark.parametrize("module_name", ["officehome", "iwildcam"])
def test_custom_resume_detects_live_dataset_index_drift(module_name, tmp_path):
    if module_name == "officehome":
        module = OH
        contract = _office_contract()
        key = OH._officehome_cell_key("c", 0, 1, "target_val", "Art", "val", "iid", "tiny")
        original_context = _office_sampling_context()
        record, condition = _office_valid_partial(contract, key, original_context)
        changed_context = json.loads(json.dumps(original_context))
        changed_context["splits"]["Art"]["val"].reverse()
        loader = lambda path: module._load_partial(
            path, contract, [key], ["sar_online_mild"], splits=changed_context
        )
    else:
        module = IWC
        contract = _iwc_contract()
        key = IWC._iwc_cell_key("c", "val", "a" * 64, 0, 0, 7, "iid", "tiny", "mild")
        original_context = _iwc_sampling_context()
        record, condition = _iwc_valid_partial(contract, key, original_context)
        changed_subset = SimpleNamespace(indices=original_context[0].indices[::-1].copy())
        loader = lambda path: module.load_partial_iwc(
            path, contract, [key], [("tent", "online")],
            sub=changed_subset, y=original_context[1], locations=original_context[2],
        )
    partial = tmp_path / f"{module_name}-live-index-drift.json"
    partial.write_text(json.dumps(module._partial_payload(
        contract, [key], [record], [condition], {key}, {}, [], 0.0,
    )), encoding="utf-8")

    with pytest.raises(RuntimeError, match="current .*index"):
        loader(partial)


@pytest.mark.parametrize("composition", ["iid", "imbalanced", "single_class"])
def test_officehome_archived_sample_ids_match_condition_builder(monkeypatch, composition):
    paths = [str(i) for i in range(24)]
    labels = np.repeat(np.arange(4), 6)
    monkeypatch.setattr(
        OH.ohd,
        "_load",
        lambda path, _transform: torch.tensor([int(path)], dtype=torch.float32),
    )
    expected_stream, expected_eval = OH._officehome_condition_indices(
        labels, composition, 2, 8, 2, np.random.default_rng(17)
    )
    stream_x, stream_y = OH._load_officehome_positions(
        paths, labels, expected_stream, torch.device("cpu")
    )
    eval_x, eval_y = OH._load_officehome_positions(
        paths, labels, expected_eval, torch.device("cpu")
    )
    assert stream_x.reshape(-1).numpy().astype(int).tolist() == expected_stream.tolist()
    assert eval_x.reshape(-1).numpy().astype(int).tolist() == expected_eval.tolist()
    assert stream_y.tolist() == labels[expected_stream].tolist()
    assert eval_y.tolist() == labels[expected_eval].tolist()
    assert len(np.unique(expected_stream)) == len(expected_stream)
    assert np.intersect1d(expected_stream, expected_eval).size == 0


def test_officehome_labelshift_and_evidence_panel_contracts():
    n_classes = 65
    source_prior = np.full(n_classes, 1.0 / n_classes)
    stream_probs = np.full((200, n_classes), 0.001)
    stream_probs[:, 3] = 1.0 - 0.001 * (n_classes - 1)

    weights, target_prior = OH.ohc.estimate_labelshift_weights(stream_probs, source_prior)

    assert np.isfinite(weights).all()
    assert int(np.argmax(target_prior)) == 3
    assert target_prior[3] > 0.8

    frozen = np.full((50, n_classes), 1.0 / n_classes)
    adapted = np.full((50, n_classes), 0.001)
    adapted[:, 3] = 1.0 - 0.001 * (n_classes - 1)
    logits_frozen = np.zeros((50, n_classes))
    evidence = OH.ohc.full_evidence(
        frozen, adapted, logits_frozen, None, None, 0.1, 0.0,
        n_classes, source_prior,
    )

    assert len(evidence) == 17  # the cross-candidate feature is appended later
    assert np.isfinite(evidence).all()
    assert evidence[16] > 0.0  # predicted-prior divergence from the source prior


@pytest.mark.parametrize("module", [OH, IWC])
def test_error_routes_are_explicit_failures(module):
    with pytest.raises(RuntimeError, match="failed"):
        module._require_route({"decision": "ERROR", "reason": "dependency missing"}, "route B")
    with pytest.raises(RuntimeError, match="unknown decision"):
        module._require_route({"decision": "BROKEN"}, "route B")
    with pytest.raises(RuntimeError, match="without an integer candidate choice"):
        module._require_route({"decision": "ADAPT"}, "route B", require_choice=True)
    module._require_route({"decision": "ABSTAIN", "reason": "insufficient evidence"}, "route B")


def test_officehome_multiclass_route_is_unsupported_but_cell_commits():
    predictions = np.array([
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 2, 3],
        [3, 2, 1, 0],
    ])
    route, realized, eligible = OH._officehome_route_b(
        predictions, [0.5, 0.6, 0.4, 0.55], tau_star=0.52, kappa=2.5
    )
    assert route["decision"] == "ABSTAIN"
    assert route["status"] == "UNSUPPORTED"
    assert route["scorable"] is False
    assert realized is None
    assert eligible is False

    key = ("office-cell",)
    records, conditions = [], []
    cell_records = [{"_cell_key": list(key), "candidate": "sar", "aa": 0.6}]
    condition = {
        "_key": list(key), "a0": 0.5, "best_adapt": 0.6,
        "route": route, "realized": None, "route_b_eligible": False,
    }
    OH._commit_cell(records, conditions, cell_records, condition)
    assert records == cell_records
    assert conditions == [condition]


@pytest.mark.parametrize("module", [OH, IWC])
def test_strict_atomic_writer_rejects_nonfinite_json(module, tmp_path):
    destination = tmp_path / "result.json"
    with pytest.raises(ValueError):
        module.atomic_dump({"value": float("inf")}, destination)
    assert not destination.exists()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("module", [OH, IWC])
def test_resume_loader_rejects_nonstandard_json_constants(module, token, tmp_path):
    partial = tmp_path / "_partial.json"
    partial.write_text('{"value": ' + token + '}', encoding="utf-8")
    with pytest.raises(module.ri.RunIntegrityError, match="non-standard JSON constant"):
        if module is OH:
            module._load_partial(partial, {}, [], [])
        else:
            module.load_partial_iwc(partial, {}, [], [])


@pytest.mark.parametrize("module", [OH, IWC])
def test_nonfinite_staged_cell_is_not_committed(module):
    key = ("cell",)
    records, conditions = [], []
    cell_records = [{"_cell_key": list(key), "aa": float("nan")}]
    condition = {"_key": list(key), "a0": 0.5}
    with pytest.raises(ValueError, match="NaN/Infinity"):
        module._commit_cell(records, conditions, cell_records, condition)
    assert records == []
    assert conditions == []


@pytest.mark.parametrize("module", [OH, IWC])
def test_ledger_never_calls_failed_or_pending_grid_complete(module):
    first = ("first",)
    second = ("second",)
    failure = {"key": list(second), "stage": "route_b", "error": "boom"}
    ledger = module._ledger([first, second], {first}, {second: failure}, [failure])

    assert ledger["status"] == "incomplete"
    assert ledger["expected"] == 2
    assert ledger["completed"] == 1
    assert ledger["failed"] == 1


@pytest.mark.parametrize("module", [OH, IWC])
def test_checkpoint_tensor_hash_ignores_container_metadata_but_detects_weights(module, tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    changed = tmp_path / "changed.pt"
    state = {"weight": torch.tensor([[1.0, 2.0]]), "counter": torch.tensor(3)}
    torch.save({"model": state, "note": "one"}, first)
    torch.save({"model": state, "note": "different metadata"}, second)
    torch.save({"model": {**state, "weight": torch.tensor([[1.0, 2.5]])}}, changed)

    assert module.checkpoint_tensor_sha256(first) == module.checkpoint_tensor_sha256(second)
    assert module.checkpoint_tensor_sha256(first) != module.checkpoint_tensor_sha256(changed)
    assert module.file_sha256(first) != module.file_sha256(second)


def test_iwildcam_population_manifest_binds_ids_labels_locations_and_split():
    sub = SimpleNamespace(indices=np.array([10, 11, 12]))
    base = IWC.iwildcam_population_manifest(sub, [1, 2, 3], [7, 7, 8], "val")
    changed_label = IWC.iwildcam_population_manifest(sub, [1, 9, 3], [7, 7, 8], "val")
    changed_split = IWC.iwildcam_population_manifest(sub, [1, 2, 3], [7, 7, 8], "test")

    assert base["manifest_sha256"] != changed_label["manifest_sha256"]
    assert base["manifest_sha256"] != changed_split["manifest_sha256"]


def test_iwildcam_loader_fails_requested_unreadable_position_without_substitution():
    class FakeSubset:
        calls = []

        def __getitem__(self, index):
            self.calls.append(index)
            if index == 0:
                raise OSError("unreadable")
            return torch.tensor([float(index)]), index, None

    subset = FakeSubset()
    with pytest.raises(RuntimeError, match="sample substitution is forbidden"):
        IWC.load_positions(
            subset, [0], torch.device("cpu"), return_positions=True,
        )
    assert subset.calls == [0]


def test_iwildcam_condition_preserves_exact_unique_disjoint_identities():
    class FakeSubset:
        def __getitem__(self, position):
            return torch.tensor([float(position)]), int(position % 2), None

    y = np.tile([0, 1], 8)
    locations = np.zeros(len(y), dtype=int)
    _, _, eval_y, ids = IWC.build_condition(
        FakeSubset(), y, locations, 0, "iid", 2, 4, 2,
        np.random.default_rng(3), torch.device("cpu"), return_ids=True,
    )
    stream_requested = ids["stream_requested_subset_positions"]
    eval_requested = ids["eval_requested_subset_positions"]
    assert np.array_equal(stream_requested, ids["stream_resolved_subset_positions"])
    assert np.array_equal(eval_requested, ids["eval_resolved_subset_positions"])
    assert len(np.unique(stream_requested)) == len(stream_requested)
    assert len(np.unique(eval_requested)) == len(eval_requested)
    assert np.intersect1d(stream_requested, eval_requested).size == 0
    assert eval_y.tolist() == y[eval_requested].tolist()


def test_officehome_target_test_is_disabled_before_data_or_model_work(monkeypatch):
    args = OH.parse_args(["--role", "target_test"])
    monkeypatch.setattr(OH.tm, "pick_device", lambda *_: pytest.fail("device access occurred"))
    monkeypatch.setattr(OH.ohd, "load_or_make_splits", lambda *_: pytest.fail("data access occurred"))
    monkeypatch.setattr(OH, "load_f0", lambda *_: pytest.fail("model access occurred"))
    with pytest.raises(RuntimeError, match="target_test is disabled before data/model access"):
        OH.run(args)


@pytest.mark.parametrize("module", [OH, IWC])
def test_manifest_builder_refuses_incomplete_ledger(module):
    with pytest.raises(RuntimeError, match="incomplete ledger"):
        if module is OH:
            module.build_manifest(None, [], [], {"ledger": {"status": "incomplete"}})
        else:
            module.build_manifest(
                None, "checkpoint.pt", {}, {}, [], [],
                {"ledger": {"status": "incomplete"}},
            )


def test_officehome_complete_manifest_remains_diagnostic(monkeypatch):
    monkeypatch.setattr(
        OH.rc,
        "aggregate_single_candidate",
        lambda records: {"sar_online_aggressive": {"kga": {"radius_feasible": True}}},
    )
    monkeypatch.setattr(OH, "_route_b_summary", lambda conditions: {"scorable": False})
    monkeypatch.setattr(OH.an, "detectability_analysis", lambda records, names: {"status": "ok"})
    args = SimpleNamespace(
        role="target_test", candidates=["sar_online_aggressive"], ckpt="checkpoint.pt",
        model_seed=0,
    )
    record = {"candidate": "sar_online_aggressive", "a0": 0.5, "aa": 0.6}
    condition = {"oracle": 0.6}
    meta = {
        "ledger": {"status": "complete", "expected": 1, "completed": 1, "failed": 0, "pending": 0},
        "resume_contract": {
            "sha256": "a" * 64,
            "payload": {"checkpoint": {"file_sha256": "b" * 64, "tensor_sha256": "c" * 64}},
        },
        "split_manifest": {"sha256": "d" * 64},
        "population_manifest": {"manifest_sha256": "e" * 64},
        "source_reference_samples": {"n": 1},
        "wall_sec": 1.0,
    }

    manifest = OH.build_manifest(args, [record], [condition], meta)

    assert manifest["schema"] == "kbound_officehome_v4"
    assert manifest["execution_complete"] is True
    assert manifest["publication_eligible"] is False
    assert manifest["claim_eligibility"]["route_a_single_candidate"] is False
    assert manifest["claim_eligibility"]["route_b_multicandidate"] is False
    assert manifest["claim_eligibility"]["route_c_smooth_drift"] is False
    assert manifest["routing_c_smooth_drift"]["status"] == "UNSUPPORTED"


def test_iwildcam_complete_manifest_withholds_noncanonical_metric(monkeypatch):
    monkeypatch.setattr(
        IWC.rc,
        "aggregate_single_candidate",
        lambda records: {"tent_online": {"kga": {"radius_feasible": True}}},
    )
    monkeypatch.setattr(IWC.rc, "aggregate_multicandidate", lambda conditions: {"scorable": False})
    monkeypatch.setattr(IWC.rc, "aggregate_smoothdrift", lambda conditions: {"implemented": False})
    args = SimpleNamespace(
        data_root="/data", split="test", candidates=["tent_online"], train_seed=0,
    )
    record = {"candidate": "tent_online", "a0": 0.5, "aa": 0.6, "B": 0.1}
    condition = {"oracle": 0.6}
    meta = {
        "ledger": {"status": "complete", "expected": 1, "completed": 1, "failed": 0, "pending": 0},
        "resume_contract": {
            "sha256": "a" * 64,
            "payload": {"checkpoint": {"file_sha256": "b" * 64, "tensor_sha256": "c" * 64}},
        },
        "population_manifest": {"manifest_sha256": "d" * 64},
        "target_locations": [(7, 20, 3)],
        "wall_sec": 1.0,
    }

    manifest = IWC.build_manifest(
        args, "checkpoint.pt", {}, {}, [record], [condition], meta
    )

    assert manifest["schema"] == "kbound_wilds_iwildcam_finder_v0.5"
    assert manifest["execution_complete"] is True
    assert manifest["publication_eligible"] is False
    assert manifest["official_wilds_metric"] is False
    assert manifest["claim_eligibility"]["route_a_single_candidate"] is False
    assert manifest["claim_eligibility"]["route_b_multicandidate"] is False
    assert manifest["baselines"]["metric"] == "diagnostic_per_cell_sklearn_macro_f1"
