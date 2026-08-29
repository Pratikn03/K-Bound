from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WILDS = ROOT / "experiments/kbound/wilds"
if str(WILDS) not in sys.path:
    sys.path.insert(0, str(WILDS))

import run_camelyon17_kbound as cam  # noqa: E402
import run_imagenetr_kbound as imagenetr  # noqa: E402
import run_integrity as ri  # noqa: E402


NATURAL_RUNNERS = [
    ROOT / "experiments/kbound/wilds/run_camelyon17_kbound.py",
    ROOT / "experiments/kbound/wilds/run_imagenetr_kbound.py",
    ROOT / "experiments/kbound/wilds/run_iwildcam_kbound.py",
    ROOT / "experiments/kbound/officehome/run_officehome_kbound.py",
    ROOT / "experiments/kbound/wilds/run_rxrx1_kbound.py",
    ROOT / "experiments/kbound/wilds/run_geoshift_kbound.py",
]


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_route_realized_never_maps_invalid_state_to_freeze() -> None:
    scores = [0.7, 0.8]
    assert cam.route_realized({"decision": "ERROR", "status": "ERROR", "scorable": False}, scores) is None
    assert cam.route_realized(
        {"decision": "ABSTAIN", "status": "UNSUPPORTED", "scorable": False}, scores
    ) is None
    assert cam.route_realized({"decision": "FREEZE", "status": "OK", "scorable": True}, scores) == 0.7
    assert cam.route_realized(
        {"decision": "ADAPT", "choice": 1, "status": "OK", "scorable": True}, scores
    ) == 0.8
    assert cam.route_realized(
        {"decision": "ADAPT", "choice": 99, "status": "OK", "scorable": True}, scores
    ) is None


def test_route_c_is_explicitly_unsupported_for_runner_objectives() -> None:
    for objective, n_classes in (
        ("balanced_accuracy", 2),
        ("balanced_accuracy", 200),
        ("macro_f1", 182),
        ("accuracy", 65),
        ("balanced_accuracy", 1139),
        ("accuracy_binned_wealth", 5),
    ):
        route = cam.unsupported_route_c(objective, n_classes)
        assert route["decision"] == "ABSTAIN"
        assert route["status"] == "UNSUPPORTED"
        assert route["implemented"] is False
        assert route["scorable"] is False
        assert route["reported_objective"] == objective
        assert route["target_label_selection_used"] is False
        assert "bracket" not in route
        assert "true_B_best" not in route
        cam.validate_unsupported_route_c(route, objective, n_classes)


def test_route_c_rejects_retired_scored_payloads() -> None:
    retired = {
        **cam.unsupported_route_c("balanced_accuracy", 2),
        "implemented": True,
        "bracket": [-0.1, 0.1],
    }
    with np.testing.assert_raises_regex(RuntimeError, "must remain the explicit UNSUPPORTED"):
        cam.validate_unsupported_route_c(retired, "balanced_accuracy", 2)


def test_natural_runners_never_call_binary_brier_route_c() -> None:
    for path in NATURAL_RUNNERS:
        assert "smooth_drift_route(" not in path.read_text(encoding="utf-8"), path


def test_camelyon_route_b_metric_contract_requires_exact_class_balance() -> None:
    labels = np.asarray([0, 0, 1, 1])
    predictions = np.asarray([
        [0, 1, 1, 1],
        [0, 0, 0, 1],
    ])
    scores = np.asarray([0.75, 0.75])

    contract = cam.route_b_metric_contract(labels, predictions, scores)

    assert contract["route_b_metric_eligible"] is True
    assert contract["metric_parity_verified"] is True
    assert contract["route_objective"] == "accuracy"

    imbalanced_labels = np.asarray([0, 0, 0, 1])
    imbalanced_scores = np.asarray([2.0 / 3.0, 2.0 / 3.0])
    ineligible = cam.route_b_metric_contract(imbalanced_labels, predictions, imbalanced_scores)
    assert ineligible["route_b_metric_eligible"] is False
    assert ineligible["route_objective"] == "balanced_accuracy"


def test_camelyon_route_b_metric_contract_rejects_silent_metric_mismatch() -> None:
    labels = np.asarray([0, 0, 1, 1])
    predictions = np.asarray([[0, 1, 1, 1]])

    with np.testing.assert_raises_regex(ValueError, "metric parity failed"):
        cam.route_b_metric_contract(labels, predictions, [0.5])


def test_multicandidate_aggregate_withholds_all_metrics_if_one_cell_is_unscorable() -> None:
    valid = {
        "a0": 0.7,
        "oracle": 0.8,
        "realized": 0.8,
        "route": {"decision": "ADAPT", "choice": 1, "status": "OK", "scorable": True},
        "cand_names": ["freeze_f0", "tent"],
        "aa_all": [0.7, 0.8],
        "regime_label": "helpful",
        "domain": "test",
    }
    invalid = {
        **valid,
        "realized": None,
        "route": {"decision": "ERROR", "choice": None, "status": "ERROR", "scorable": False},
    }

    summary = cam.aggregate_multicandidate([valid, invalid])

    assert summary["status"] == "UNSCORABLE_ROUTE_CELLS"
    assert summary["scorable"] is False
    assert summary["beats_both"] is None
    assert "mean_acc" not in summary


def test_single_candidate_infeasible_radius_is_strict_json(monkeypatch) -> None:
    records = [
        {
            "candidate": "tent_online",
            "a0": 0.7,
            "aa": adapted,
            "B": adapted - 0.7,
            "Z": [float(index), 1.0],
        }
        for index, adapted in enumerate((0.6, 0.8, 0.9))
    ]

    monkeypatch.setattr(
        cam.an,
        "decide_kga",
        lambda Z, B, **_: (
            np.asarray(B, dtype=float),
            np.full(len(B), np.inf),
            np.full(len(B), "ABSTAIN"),
        ),
    )
    summary = cam.aggregate_single_candidate(records)

    kga = summary["tent_online"]["kga"]
    assert kga["radius_status"] == "INFEASIBLE_EXACT_RANK"
    assert kga["eps_conformal"] is None
    json.dumps(summary, allow_nan=False)


def test_camelyon_incomplete_manifest_withholds_partial_aggregates() -> None:
    args = SimpleNamespace(
        evidence_panel="base",
        online_only=False,
        delta=0.05,
        data_root="/data",
    )
    meta = {
        "scientific_config": {"dataset": "camelyon17"},
        "run_config_sha256": "a" * 64,
        "ledger": {
            "status": "INCOMPLETE",
            "execution_complete": False,
            "publication_eligible": False,
            "expected_cells": 2,
            "completed_cells": 1,
            "failed_cells": 1,
            "missing_cells": 0,
        },
        "failures": [{"cell_id": "b" * 64}],
        "checkpoint_identities": {},
        "n_present": 10,
        "n_total": 10,
        "population_identity": {"sha256": "c" * 64, "counts_by_domain": {"test": 10}},
        "wall_sec": 1.0,
    }
    records = [{"candidate": "tent_online", "a0": 0.5, "aa": 0.9, "B": 0.4, "Z": [0.0]}]

    manifest = cam.build_manifest(args, records, [], meta)

    assert manifest["publication_eligible"] is False
    assert manifest["baselines"]["status"] == "NOT_COMPUTED_INCOMPLETE_RUN"
    assert manifest["routing_a_single_candidate"]["status"] == "NOT_COMPUTED_INCOMPLETE_RUN"
    assert manifest["records"] == records  # recovery data are retained, but never scored


def test_camelyon_resume_identity_binds_population() -> None:
    args = SimpleNamespace(
        data_root="/data",
        seeds=[0],
        domains=["test"],
        compositions=["iid"],
        batch_regimes=["tiny"],
        aggressiveness=["mild"],
        n_eval=32,
        n_batches=1,
        tau_star=0.08,
        kappa=2.5,
        device="cpu",
        steps_override=0,
        delta=0.05,
        sd_L=0.6,
        evidence_panel="base",
        smoke=False,
        adapt_lr=None,
        online_only=False,
        anchor_above_chance=False,
    )
    checkpoints = {
        "0": {
            "path": "/checkpoint.pt", "sha256": "a" * 64,
            "tensor_sha256": "t" * 64,
        }
    }
    first = cam._scientific_config(
        args,
        checkpoints,
        population_identity={"sha256": "b" * 64, "counts_by_domain": {"test": 10}},
        resolved_device="cpu",
    )
    second = cam._scientific_config(
        args,
        checkpoints,
        population_identity={"sha256": "c" * 64, "counts_by_domain": {"test": 10}},
        resolved_device="cpu",
    )

    assert ri.stable_sha256(first) != ri.stable_sha256(second)
    assert first["implementation_sha256"]["runner"] == ri.file_sha256(cam.__file__)
    assert first["f0_checkpoints"]["0"]["tensor_sha256"] == "t" * 64


def test_camelyon_exact_loader_fails_unreadable_request_without_substitution() -> None:
    class Subset:
        indices = np.arange(12)

        def __len__(self):
            return 12

        def __getitem__(self, position):
            if int(position) == 0:
                raise OSError("broken patch")
            return torch.tensor([float(position)]), int(position % 2), None

    labels = np.tile([0, 1], 6)
    # Seed 9 selects position 0 in this deterministic small fixture.
    found = False
    for seed in range(100):
        try:
            cam._build_condition_exact(
                Subset(), labels, "iid", 1, 2, np.random.default_rng(seed),
                torch.device("cpu"), n_batches=1,
            )
        except RuntimeError as exc:
            if "sample substitution is forbidden" in str(exc):
                found = True
                break
    assert found


def test_camelyon_condition_archives_exact_unique_disjoint_identities() -> None:
    class Subset:
        indices = np.arange(20) + 100

        def __len__(self):
            return 20

        def __getitem__(self, position):
            return torch.tensor([float(position)]), int(position % 2), None

    labels = np.tile([0, 1], 10)
    _, _, eval_y, provenance = cam._build_condition_exact(
        Subset(), labels, "iid", 2, 4, np.random.default_rng(3),
        torch.device("cpu"), n_batches=2,
    )
    assert provenance["requested_resolved_identity_equal"] is True
    assert provenance["stream_eval_disjoint"] is True
    assert provenance["stream_unique"] is True
    assert provenance["eval_unique"] is True
    assert provenance["stream_eval_overlap_count"] == 0
    assert len(eval_y) == provenance["eval_n"]


def test_imagenetr_declares_route_b_unsupported_and_withholds_incomplete_metrics() -> None:
    args = SimpleNamespace(panel="diverse_backbones", delta=0.05)
    meta = {
        "scientific_config": {
            "dataset": "imagenet-r",
            "seed_semantics": {
                "model_seed": 0,
                "model_replications": 1,
                "args_seeds_role": "stream_seed",
                "independent_model_ci_eligible": False,
            },
        },
        "run_config_sha256": "d" * 64,
        "ledger": {
            "status": "INCOMPLETE",
            "execution_complete": False,
            "publication_eligible": False,
            "expected_cells": 1,
            "completed_cells": 0,
            "failed_cells": 1,
            "missing_cells": 0,
        },
        "failures": [{"cell_id": "e" * 64}],
        "candidate_names": ["resnet101"],
        "f0": "resnet50",
        "f0_identity": {"description": "resnet50", "state_sha256": "1" * 64},
        "n_classes": 200,
        "n_images": 30_000,
        "population_identity": {"sha256": "f" * 64, "n_images": 30_000},
        "wall_sec": 1.0,
    }

    manifest = imagenetr.build_manifest(args, [], [], meta)

    assert manifest["publication_eligible"] is False
    assert manifest["metric"] == "balanced_accuracy"
    assert manifest["metric_contract"]["ordinary_accuracy_alias_allowed"] is False
    assert manifest["claim_eligibility"]["route_b_multicandidate"] is False
    assert "unsupported and unscored" in manifest["multiclass_caveat"]
    assert manifest["baselines"]["status"] == "NOT_COMPUTED_INCOMPLETE_RUN"
    json.dumps(manifest, allow_nan=False)


def test_imagenetr_complete_manifest_uses_only_balanced_accuracy_names(monkeypatch) -> None:
    args = SimpleNamespace(panel="shared_tta", delta=0.05)
    record = {
        "candidate": "tent_online",
        "metric": "balanced_accuracy",
        "a0": 0.4,
        "aa": 0.5,
        "B": 0.1,
        "Z": [0.0],
        "domain": "imagenet_r",
        "regime_label": "helpful",
    }
    condition = {
        "a0": 0.4,
        "oracle": 0.5,
        "best_adapt": 0.5,
        "cand_names": ["freeze_f0", "tent_online"],
        "aa_all": [0.4, 0.5],
        "domain": "imagenet_r",
        "regime_label": "helpful",
        "route": {"decision": "ABSTAIN", "status": "UNSUPPORTED", "scorable": False},
        "route_c": cam.unsupported_route_c("balanced_accuracy", 200),
        "realized": None,
    }
    monkeypatch.setattr(imagenetr.rc, "kbound_summary", lambda *args, **kwargs: {"status": "ok"})
    meta = {
        "scientific_config": {
            "metric": "balanced_accuracy",
            "seed_semantics": {"independent_model_ci_eligible": False},
        },
        "run_config_sha256": "a" * 64,
        "ledger": {
            "status": "COMPLETE",
            "execution_complete": True,
            "publication_eligible": False,
            "expected_cells": 1,
            "completed_cells": 1,
            "failed_cells": 0,
            "missing_cells": 0,
        },
        "failures": [],
        "candidate_names": ["tent_online"],
        "f0_identity": {"description": "f0", "state_sha256": "b" * 64},
        "n_classes": 200,
        "n_images": 1,
        "population_identity": {"sha256": "c" * 64, "n_images": 1},
        "wall_sec": 1.0,
    }

    manifest = imagenetr.build_manifest(args, [record], [condition], meta)

    assert manifest["metric"] == "balanced_accuracy"
    assert "mean_acc" not in set(_all_keys(manifest))
    assert manifest["routing_c_smooth_drift"]["status"] == "UNSUPPORTED"
    assert manifest["routing_c_smooth_drift"]["scorable"] is False


def test_imagenetr_resume_identity_binds_population_and_uses_stream_seed(tmp_path: Path) -> None:
    class_index = tmp_path / "class_index.json"
    class_index.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        imagenetr_dir=str(tmp_path / "imagenet-r"),
        class_index=str(class_index),
        panel="diverse_backbones",
        f0_backbone="resnet50",
        candidate_backbones=["resnet101"],
        seeds=[7],
        compositions=["iid"],
        batch_regimes=["tiny"],
        aggressiveness=["mild"],
        n_eval=10,
        n_batches=1,
        tau_star=0.52,
        kappa=2.5,
        sd_L=0.6,
        delta=0.05,
        steps_override=0,
        max_classes=0,
        episodic_steps=1,
        episodic_batch=8,
        frozen_eval_batch=8,
        smoke=False,
        adapt_lr=None,
        online_only=False,
    )
    f0_identity = {"description": "resnet50", "state_sha256": "a" * 64}
    candidate_identities = {
        "resnet101": {
            "backbone": "resnet101",
            "description": "torchvision resnet101 test weights",
            "tensor_sha256": "c" * 64,
            "state_sha256": "c" * 64,
        }
    }
    first = imagenetr._scientific_config(
        args,
        resolved_device="cpu",
        population_identity={"sha256": "b" * 64, "n_images": 1},
        f0_identity=f0_identity,
        candidate_identities=candidate_identities,
    )
    second = imagenetr._scientific_config(
        args,
        resolved_device="cpu",
        population_identity={"sha256": "c" * 64, "n_images": 1},
        f0_identity=f0_identity,
        candidate_identities=candidate_identities,
    )

    assert ri.stable_sha256(first) != ri.stable_sha256(second)
    assert imagenetr._cell_spec(7, "iid", "tiny", "mild")["model_seed"] == 0
    assert imagenetr._cell_spec(7, "iid", "tiny", "mild")["stream_seed"] == 7
    assert first["seed_semantics"]["independent_model_ci_eligible"] is False
    assert first["implementation_sha256"]["runner"] == ri.file_sha256(imagenetr.__file__)
    assert first["candidate_model_artifacts"] == candidate_identities
    assert first["candidate_tta_protocols"] == {}

    args.panel = "shared_tta"
    shared = imagenetr._scientific_config(
        args,
        resolved_device="cpu",
        population_identity={"sha256": "b" * 64, "n_images": 1},
        f0_identity=f0_identity,
        candidate_identities={},
    )
    assert shared["candidate_tta_protocols"]["tent_online"] == imagenetr.tm.tta_protocol_contract(
        "online"
    )
    assert shared["candidate_tta_protocols"]["tent_episodic"] == imagenetr.tm.tta_protocol_contract(
        "episodic"
    )
    assert '"tta_protocol": tm.tta_protocol_contract(mode)' in Path(imagenetr.__file__).read_text(
        encoding="utf-8"
    )


def test_imagenetr_candidate_backbones_are_bound_by_actual_tensor_state(monkeypatch) -> None:
    class TinyMasked(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([float(value)]))

    values = {"resnet101": 1.0, "resnet152": 2.0}

    def fake_backbone(name, _indices, _device):
        return TinyMasked(values[name]), f"test {name} weights"

    monkeypatch.setattr(imagenetr, "make_masked_backbone", fake_backbone)
    monkeypatch.setattr(imagenetr.tm, "mps_free", lambda: None)
    identities = imagenetr._collect_candidate_identities(
        ["resnet101", "resnet152"], [0, 1], torch.device("cpu")
    )

    assert set(identities) == {"resnet101", "resnet152"}
    assert identities["resnet101"]["tensor_sha256"] != identities["resnet152"]["tensor_sha256"]
    assert identities["resnet101"]["tensor_sha256"] == identities["resnet101"]["state_sha256"]

    values["resnet152"] = 1.0
    with pytest.raises(RuntimeError, match="unique tensor-state identities"):
        imagenetr._collect_candidate_identities(
            ["resnet101", "resnet152"], [0, 1], torch.device("cpu")
        )


def test_imagenetr_resume_rejects_tampered_candidate_tensor_identity() -> None:
    identity = {
        "backbone": "resnet101",
        "description": "test weights",
        "tensor_sha256": "a" * 64,
        "state_sha256": "a" * 64,
    }
    expected = {"resnet101": identity}
    record = {
        "candidate": "resnet101",
        "candidate_model_identity": identity,
        "candidate_tensor_sha256": "a" * 64,
    }
    condition = {"candidate_model_identities": expected}
    imagenetr._validate_diverse_resume_model_identities([record], [condition], expected)

    tampered = {**record, "candidate_tensor_sha256": "b" * 64}
    with pytest.raises(ri.RunIntegrityError, match="configured tensor identity"):
        imagenetr._validate_diverse_resume_model_identities([tampered], [condition], expected)


def test_imagenetr_unreadable_requested_image_fails_without_substitution(tmp_path: Path) -> None:
    valid = tmp_path / "valid.png"
    broken = tmp_path / "broken.png"
    Image.new("RGB", (256, 256), color=(1, 2, 3)).save(valid)
    broken.write_bytes(b"not-an-image")
    index = [(str(broken), 0), (str(valid), 0), (str(valid), 1), (str(valid), 1)]
    labels = np.asarray([0, 0, 1, 1])
    with pytest.raises(RuntimeError, match="sample substitution is forbidden"):
        imagenetr.build_condition(
            index, labels, "iid", 1, 2, np.random.default_rng(0),
            torch.device("cpu"), n_batches=1,
        )


def test_imagenetr_condition_exact_identities_are_unique_and_disjoint(monkeypatch) -> None:
    index = [(f"image-{position}.png", position % 2) for position in range(16)]
    labels = np.asarray([label for _, label in index])
    monkeypatch.setattr(
        imagenetr, "load_img", lambda path: torch.tensor([float(path.split("-")[1].split(".")[0])])
    )
    _, _, eval_y, identities = imagenetr.build_condition(
        index, labels, "iid", 2, 4, np.random.default_rng(4),
        torch.device("cpu"), n_batches=2, return_ids=True,
    )
    stream = identities["stream_requested_positions"]
    evaluation = identities["eval_requested_positions"]
    assert np.array_equal(stream, identities["stream_resolved_positions"])
    assert np.array_equal(evaluation, identities["eval_resolved_positions"])
    assert len(np.unique(stream)) == len(stream)
    assert len(np.unique(evaluation)) == len(evaluation)
    assert np.intersect1d(stream, evaluation).size == 0
    assert eval_y.tolist() == labels[evaluation].tolist()


def _camelyon_semantic_resume_fixture():
    args = SimpleNamespace(
        seeds=[0],
        domains=["test"],
        compositions=["iid"],
        batch_regimes=["tiny"],
        aggressiveness=["mild"],
        n_eval=4,
        n_batches=1,
        evidence_panel="base",
        online_only=True,
    )

    class Subset:
        indices = np.arange(40) + 1000

    labels = np.tile([0, 1], 20)
    dom_cache = {"test": (Subset(), labels)}
    checkpoints = {
        "0": {"sha256": "a" * 64, "tensor_sha256": "b" * 64}
    }
    identity = cam._cell_spec(0, "test", "iid", "tiny", "mild")
    cell_id = ri.make_cell_id(**identity)
    sample_seed = ri.deterministic_seed(cell_id)
    stream_positions, eval_positions = cam._condition_positions(
        labels,
        "iid",
        cam.cd.BATCH_REGIMES["tiny"],
        args.n_eval,
        np.random.default_rng(sample_seed),
        n_batches=args.n_batches,
    )
    provenance = cam._condition_sample_provenance(
        dom_cache["test"][0], stream_positions, eval_positions, sample_seed,
    )
    eval_y = labels[eval_positions].astype(int)
    frozen = 1 - eval_y
    candidates = [
        (method, mode, f"{method}_{mode}")
        for method, mode in cam.CANDIDATES if mode == "online"
    ]
    archived_identity = {
        "model_seed": 0,
        "seed": 0,
        "stream_seed": sample_seed,
        "domain": "test",
        "comp": "iid",
        "regime": "tiny",
        "aggr": "mild",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_tensor_sha256": "b" * 64,
    }
    records = []
    for method, mode, candidate in candidates:
        records.append({
            "cell_id": cell_id,
            "scientific_cell_identity": identity,
            **archived_identity,
            "method": method,
            "mode": mode,
            "candidate": candidate,
            "metric": "balanced_accuracy",
            "a0": 0.0,
            "aa": 1.0,
            "B": 1.0,
            "upd_norm": 0.25,
            "Z": [0.1] * 10 + [0.25],
            "Z_base": [0.1] * 10 + [0.25],
            "evidence_panel": "base",
            "tta_protocol": cam.tm.tta_protocol_contract(mode),
            "sample_provenance": provenance,
            "preds": eval_y.tolist(),
            "regime_label": cam.an.label_regime(1.0),
        })
    names = ["freeze_f0", *[candidate for _, _, candidate in candidates]]
    condition = {
        "cell_id": cell_id,
        "scientific_cell_identity": identity,
        **archived_identity,
        "cand_names": names,
        "aa_all": [0.0] + [1.0] * len(candidates),
        "a0": 0.0,
        "oracle": 1.0,
        "best_adapt": 1.0,
        "true_best": names[1],
        "sample_provenance": provenance,
        "eval_y": eval_y.tolist(),
        "preds_frozen": frozen.tolist(),
        "regime_label": cam.an.label_regime(1.0),
    }
    return args, checkpoints, dom_cache, records, [condition]


@pytest.mark.parametrize("mutation", ["aa", "Z", "update_norm", "tta_protocol", "provenance"])
def test_camelyon_semantic_resume_rejects_self_resealed_exploits(mutation: str) -> None:
    args, checkpoints, dom_cache, records, conditions = _camelyon_semantic_resume_fixture()
    cam._validate_resume_semantics(
        records, conditions, args=args, checkpoints=checkpoints, dom_cache=dom_cache,
    )
    records = json.loads(json.dumps(records))
    conditions = json.loads(json.dumps(conditions))
    if mutation == "aa":
        records[0]["aa"] = 0.5
        records[0]["B"] = 0.5
        conditions[0]["aa_all"][1] = 0.5
        conditions[0]["true_best"] = conditions[0]["cand_names"][2]
    elif mutation == "Z":
        records[0]["Z"].pop()
        records[0]["Z_base"].pop()
    elif mutation == "update_norm":
        records[0]["upd_norm"] = 0.5
    elif mutation == "tta_protocol":
        records[0]["tta_protocol"]["mode"] = "forged"
    else:
        conditions[0]["sample_provenance"]["condition_seed"] += 1
        for record in records:
            record["sample_provenance"] = conditions[0]["sample_provenance"]

    # The attacker can recompute the generic inventory hashes; runner semantics
    # must still reject the scientifically impossible payload.
    ri.partial_document(
        run_config_sha256="c" * 64,
        expected_cell_ids=[conditions[0]["cell_id"]],
        records=records,
        conditions=conditions,
        failures=[],
        progress="1/1",
        require_scientific_cell_identity=True,
        semantic_validator=lambda _records, _conditions: None,
    )
    with pytest.raises(ri.RunIntegrityError):
        cam._validate_resume_semantics(
            records, conditions, args=args, checkpoints=checkpoints, dom_cache=dom_cache,
        )


def _imagenetr_semantic_resume_fixture():
    args = SimpleNamespace(
        panel="shared_tta",
        seeds=[7],
        compositions=["iid"],
        batch_regimes=["tiny"],
        aggressiveness=["mild"],
        n_eval=4,
        n_batches=1,
        online_only=True,
    )
    index = [(f"/imagenet-r/image-{position}.png", position % 2) for position in range(40)]
    labels = np.asarray([label for _, label in index], dtype=int)
    f0_identity = {"description": "f0", "state_sha256": "d" * 64}
    identity = imagenetr._cell_spec(7, "iid", "tiny", "mild")
    cell_id = ri.make_cell_id(**identity)
    sample_seed = ri.deterministic_seed(cell_id)
    stream_positions, eval_positions = imagenetr._condition_positions(
        labels,
        "iid",
        imagenetr.BATCH_REGIMES["tiny"],
        args.n_eval,
        np.random.default_rng(sample_seed),
        n_batches=args.n_batches,
    )
    provenance = imagenetr._condition_sample_provenance(
        index,
        {
            "stream_requested_positions": stream_positions,
            "stream_resolved_positions": stream_positions,
            "eval_requested_positions": eval_positions,
            "eval_resolved_positions": eval_positions,
        },
        sample_seed,
    )
    eval_y = labels[eval_positions]
    frozen = 1 - eval_y
    candidates = [
        (method, mode, f"{method}_{mode}")
        for method, mode in imagenetr.rc.CANDIDATES if mode == "online"
    ]
    archived_identity = {
        "model_seed": 0,
        "stream_seed": 7,
        "sampling_seed": sample_seed,
        "seed": 7,
        "domain": "imagenet_r",
        "comp": "iid",
        "regime": "tiny",
        "aggr": "mild",
        "f0_model_identity": f0_identity,
    }
    records = [{
        "cell_id": cell_id,
        "scientific_cell_identity": identity,
        **archived_identity,
        "method": method,
        "mode": mode,
        "candidate": candidate,
        "metric": "balanced_accuracy",
        "a0": 0.0,
        "aa": 1.0,
        "B": 1.0,
        "upd_norm": 0.25,
        "Z": [0.1] * 10 + [0.25],
        "tta_protocol": imagenetr.tm.tta_protocol_contract(mode),
        "sample_provenance": provenance,
        "preds": eval_y.tolist(),
        "regime_label": imagenetr.an.label_regime(1.0),
    } for method, mode, candidate in candidates]
    names = ["freeze_f0", *[candidate for _, _, candidate in candidates]]
    condition = {
        "cell_id": cell_id,
        "scientific_cell_identity": identity,
        **archived_identity,
        "cand_names": names,
        "aa_all": [0.0] + [1.0] * len(candidates),
        "a0": 0.0,
        "oracle": 1.0,
        "best_adapt": 1.0,
        "true_best": names[1],
        "sample_provenance": provenance,
        "eval_y": eval_y.tolist(),
        "preds_frozen": frozen.tolist(),
        "regime_label": imagenetr.an.label_regime(1.0),
    }
    return args, index, labels, f0_identity, records, [condition]


@pytest.mark.parametrize("mutation", ["aa", "Z", "update_norm", "tta_protocol", "provenance"])
def test_imagenetr_semantic_resume_rejects_self_resealed_exploits(mutation: str) -> None:
    args, index, labels, f0_identity, records, conditions = _imagenetr_semantic_resume_fixture()
    validator = lambda rs, cs: imagenetr._validate_resume_semantics(
        rs,
        cs,
        args=args,
        index=index,
        labels=labels,
        f0_identity=f0_identity,
        candidate_identities={},
    )
    validator(records, conditions)
    records = json.loads(json.dumps(records))
    conditions = json.loads(json.dumps(conditions))
    if mutation == "aa":
        records[0]["aa"] = 0.5
        records[0]["B"] = 0.5
        conditions[0]["aa_all"][1] = 0.5
        conditions[0]["true_best"] = conditions[0]["cand_names"][2]
    elif mutation == "Z":
        records[0]["Z"].pop()
    elif mutation == "update_norm":
        records[0]["upd_norm"] = 0.5
    elif mutation == "tta_protocol":
        records[0]["tta_protocol"]["mode"] = "forged"
    else:
        conditions[0]["sample_provenance"]["condition_seed"] += 1
        for record in records:
            record["sample_provenance"] = conditions[0]["sample_provenance"]
    ri.partial_document(
        run_config_sha256="e" * 64,
        expected_cell_ids=[conditions[0]["cell_id"]],
        records=records,
        conditions=conditions,
        failures=[],
        progress="1/1",
        require_scientific_cell_identity=True,
        semantic_validator=lambda _records, _conditions: None,
    )
    with pytest.raises(ri.RunIntegrityError):
        validator(records, conditions)
