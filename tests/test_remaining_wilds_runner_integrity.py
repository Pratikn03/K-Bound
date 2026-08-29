from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WILDS = ROOT / "experiments/kbound/wilds"
if str(WILDS) not in sys.path:
    sys.path.insert(0, str(WILDS))

rx = pytest.importorskip("run_rxrx1_kbound")
geo = pytest.importorskip("run_geoshift_kbound")


def _rx_args(**overrides):
    values = {
        "split": "test",
        "model_seed": 0,
        "seeds": [0, 1],
        "compositions": ["iid"],
        "batch_regimes": ["tiny"],
        "aggressiveness": ["mild"],
        "n_eval": 32,
        "n_batches": 2,
        "tau_star": 0.52,
        "kappa": 2.5,
        "sd_L": 0.6,
        "delta": 0.05,
        "device": "cpu",
        "steps_override": 4,
        "episodic_steps": 2,
        "episodic_batch": 16,
        "smoke": True,
        "adapt_lr": None,
        "online_only": False,
        "data_root": "/data/rxrx1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _geo_args(**overrides):
    values = {
        "dataset": "fmow",
        "split": "test",
        "seeds": [0],
        "max_groups": 2,
        "compositions": ["iid"],
        "batch_regimes": ["tiny"],
        "aggressiveness": ["mild"],
        "candidates": ["tent_online"],
        "n_eval": 16,
        "n_batches": 1,
        "eval_bs": 16,
        "episodic_steps": 2,
        "episodic_batch": 16,
        "tau_star": 0.52,
        "kappa": 2.5,
        "steps_override": 2,
        "backbone": "resnet18",
        "trainable": "head",
        "train_seed": 0,
        "train_epochs": 1,
        "max_train_batches": 6,
        "train_bs": 16,
        "train_lr": 1e-3,
        "balanced_train": True,
        "device": "cpu",
        "workers": 0,
        "smoke": True,
        "data_root": "/data/wilds",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("module", [rx, geo])
def test_cell_commit_is_atomic_and_rejects_nonfinite(module):
    cell_id = "a" * 64
    records = []
    conditions = []
    failures = [{"cell_id": cell_id, "error": "previous attempt"}]
    staged = [{"cell_id": cell_id, "candidate": "tent_online", "B": float("inf")}]
    condition = {"cell_id": cell_id, "route": {"scorable": False}, "realized": None}

    with pytest.raises(ValueError, match="NaN/Infinity"):
        module._commit_cell(records, conditions, failures, staged, condition)
    assert records == []
    assert conditions == []
    assert len(failures) == 1

    staged[0]["B"] = 0.01
    module._commit_cell(records, conditions, failures, staged, condition)
    assert records == staged
    assert conditions == [condition]
    assert failures == []


def test_rxrx1_config_and_cell_grid_bind_scientific_identity():
    args = _rx_args()
    checkpoint = {
        "path": "/ckpt/f0.pt", "sha256": "1" * 64,
        "tensor_sha256": "t" * 64,
    }
    population = {"sha256": "2" * 64, "n_present": 100, "n_total": 110}
    config = rx._scientific_config(args, checkpoint, population)

    assert config["checkpoint"]["sha256"] == "1" * 64
    assert config["checkpoint"]["tensor_sha256"] == "t" * 64
    assert config["population"]["sha256"] == "2" * 64
    assert config["route_b_contract"] == {
        "objective": "balanced_accuracy",
        "n_classes": 1139,
        "eligibility": "UNSUPPORTED_MULTICLASS",
    }
    assert config["metric"] == "balanced_accuracy"
    assert config["route_c_contract"]["status"] == "UNSUPPORTED"
    config_sha256 = rx.ri.stable_sha256(config)
    expected = rx._expected_cell_ids(args, checkpoint, config_sha256)
    assert len(expected) == len(set(expected)) == 2
    assert config["inference_unit"]["independent_model_ci_eligible"] is False
    assert config["stream_seeds"] == [0, 1]
    assert config["implementation_sha256"]["runner"] == rx.ri.file_sha256(rx.__file__)
    assert rx.ri.stable_sha256(config) != rx.ri.stable_sha256(
        rx._scientific_config(_rx_args(n_eval=64), checkpoint, population)
    )
    changed_population = {**population, "sha256": "9" * 64}
    assert rx.ri.stable_sha256(config) != rx.ri.stable_sha256(
        rx._scientific_config(args, checkpoint, changed_population)
    )
    other_checkpoint = {**checkpoint, "sha256": "7" * 64, "tensor_sha256": "u" * 64}
    other_config_sha256 = rx.ri.stable_sha256(
        rx._scientific_config(args, other_checkpoint, population)
    )
    assert rx._expected_cell_ids(args, checkpoint, config_sha256) != rx._expected_cell_ids(
        args, other_checkpoint, other_config_sha256
    )


def test_rxrx1_completion_context_is_recomputed_from_current_population(
    monkeypatch, tmp_path,
):
    args = _rx_args(
        smoke=False,
        ckpt="/checkpoints/f0.pt",
        results_root=str(tmp_path),
        run_name="bound_run",
        out="",
    )
    checkpoint = {
        "path": "/checkpoints/f0.pt",
        "sha256": "1" * 64,
        "tensor_sha256": "2" * 64,
        "model_seed": 0,
    }
    population = {"sha256": "3" * 64, "n_present": 100, "n_total": 100}
    monkeypatch.setattr(rx, "_checkpoint_identity", lambda _args: checkpoint)
    monkeypatch.setattr(
        rx,
        "load_rxrx1",
        lambda *_args, **_kwargs: (object(), object(), np.array([0, 1]), 2, 2),
    )
    monkeypatch.setattr(rx, "_rxrx1_population_identity", lambda *_args: population)

    context = rx._expected_completion_context(args)
    expected_sha = rx.ri.stable_sha256(rx._scientific_config(args, checkpoint, population))
    assert context == {
        "run_name": "bound_run",
        "run_dir": str((tmp_path / "bound_run").resolve()),
        "result_path": str((tmp_path / "bound_run" / f"result_{expected_sha[:8]}.json").resolve()),
        "config_sha256": expected_sha,
    }

    population["sha256"] = "4" * 64
    assert rx._expected_completion_context(args)["config_sha256"] != expected_sha


def test_geoshift_config_binds_selected_groups_population_and_objective():
    args = _geo_args()
    checkpoint = {
        "path": "/ckpt/f0.pt", "sha256": "3" * 64,
        "tensor_sha256": "t" * 64,
    }
    population = {"sha256": "4" * 64, "n": 500}
    groups = [(1, 100, 20), (2, 80, 18)]
    config = geo._scientific_config(args, checkpoint, groups, 62, population)

    assert config["target_groups"][0] == {"location": 1, "n": 100, "n_classes": 20}
    assert config["population"] == population
    assert config["route_b_contract"] == {
        "objective": "accuracy",
        "n_classes": 62,
        "eligibility": "UNSUPPORTED_MULTICLASS",
    }
    expected = geo._expected_cell_ids(args, groups, checkpoint)
    assert len(expected) == len(set(expected)) == 2
    assert config["model_identity"] == {
        "model_seed": 0, "checkpoint_sha256": "3" * 64,
        "checkpoint_tensor_sha256": "t" * 64,
    }
    assert config["implementation_sha256"]["runner"] == geo.ri.file_sha256(geo.__file__)
    assert config["inference_unit"]["independent_model_ci_eligible"] is False
    changed_population = {**population, "sha256": "8" * 64}
    assert geo.ri.stable_sha256(config) != geo.ri.stable_sha256(
        geo._scientific_config(args, checkpoint, groups, 62, changed_population)
    )
    other_checkpoint = {**checkpoint, "sha256": "6" * 64, "tensor_sha256": "u" * 64}
    assert geo._expected_cell_ids(args, groups, checkpoint) != geo._expected_cell_ids(
        args, groups, other_checkpoint)


def test_geoshift_loader_fails_unreadable_request_without_substitution():
    class UnreadableFirst:
        indices = np.array([100, 101])

        def __len__(self):
            return 2

        def __getitem__(self, position):
            if int(position) == 0:
                raise OSError("missing image")
            return geo.torch.tensor([float(position)]), 1, {}

    subset = UnreadableFirst()
    with pytest.raises(RuntimeError, match="sample substitution is forbidden"):
        geo.load_positions(
            subset, np.array([0]), geo.torch.device("cpu"),
        )


def test_geoshift_eval_and_stream_identities_are_exact_unique_and_disjoint():
    class FakeSubset:
        indices = np.arange(16) + 100

        def __len__(self):
            return 16

        def __getitem__(self, position):
            return geo.torch.tensor([float(position)]), int(position % 2), {}

    y = np.tile([0, 1], 8)
    groups = np.ones(16, dtype=int)
    _, _, eval_y, sample_ids = geo.build_condition(
        FakeSubset(), y, groups, grp=1, comp="iid", bs=2, n_eval=4,
        n_batches=2, rng=np.random.default_rng(5), device=geo.torch.device("cpu"),
    )
    stream_requested = np.asarray(sample_ids["stream_requested_positions"])
    stream_resolved = np.asarray(sample_ids["stream_resolved_positions"])
    eval_requested = np.asarray(sample_ids["eval_requested_positions"])
    eval_resolved = np.asarray(sample_ids["eval_resolved_positions"])
    assert np.array_equal(stream_requested, stream_resolved)
    assert np.array_equal(eval_requested, eval_resolved)
    assert len(np.unique(stream_requested)) == len(stream_requested)
    assert len(np.unique(eval_requested)) == len(eval_requested)
    assert np.intersect1d(stream_requested, eval_requested).size == 0
    assert eval_y.tolist() == y[eval_requested].tolist()
    assert sample_ids["requested_resolved_identity_equal"] is True
    assert sample_ids["stream_eval_disjoint"] is True
    assert sample_ids["stream_substitution_count"] == 0
    assert sample_ids["eval_substitution_count"] == 0


def _geoshift_semantic_resume_fixture():
    args = _geo_args(
        seeds=[3],
        max_groups=1,
        n_eval=4,
        n_batches=1,
        candidates=["tent_online"],
    )

    class FakeSubset:
        indices = np.arange(40) + 2000

    sub = FakeSubset()
    y = np.tile([0, 1], 20)
    groups = np.ones(40, dtype=int)
    grp_rows = [(1, 40, 2)]
    checkpoint = {"sha256": "a" * 64, "tensor_sha256": "b" * 64}
    identity = geo._cell_spec(
        args.dataset,
        args.split,
        args.train_seed,
        checkpoint["sha256"],
        3,
        1,
        "iid",
        "tiny",
        "mild",
    )
    cell_id = geo.ri.make_cell_id(**identity)
    sample_seed = geo.ri.deterministic_seed(cell_id)
    stream_positions, eval_positions = geo._condition_positions(
        y,
        groups,
        1,
        "iid",
        geo.BATCH_REGIMES["tiny"],
        args.n_eval,
        args.n_batches,
        np.random.default_rng(sample_seed),
    )
    provenance = geo._condition_sample_provenance(
        sub, stream_positions, eval_positions, sample_seed,
    )
    eval_y = y[eval_positions].astype(int)
    frozen = 1 - eval_y
    archived_identity = {
        "seed": 3,
        "stream_seed": 3,
        "sampling_seed": sample_seed,
        "model_seed": 0,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_tensor_sha256": "b" * 64,
        "inference_unit": "stream_seed_on_one_fixed_model_checkpoint",
        "independent_model_ci_eligible": False,
        "domain": "region1",
        "location": 1,
        "location_n": 40,
        "location_classes": 2,
        "split": "test",
        "comp": "iid",
        "regime": "tiny",
        "aggr": "mild",
    }
    record = {
        "cell_id": cell_id,
        "scientific_cell_identity": identity,
        **archived_identity,
        "method": "tent",
        "mode": "online",
        "candidate": "tent_online",
        "metric": "accuracy",
        "a0": 0.0,
        "aa": 1.0,
        "B": 1.0,
        "a0_bacc": 0.0,
        "aa_bacc": 1.0,
        "upd_norm": 0.25,
        "Z": [0.1] * 10 + [0.25],
        "tta_protocol": geo.tm.tta_protocol_contract("online"),
        "preds": eval_y.tolist(),
        "sample_provenance": provenance,
        "regime_label": geo.an.label_regime(1.0),
    }
    condition = {
        "cell_id": cell_id,
        "scientific_cell_identity": identity,
        **archived_identity,
        "cand_names": ["freeze_f0", "tent_online"],
        "aa_all": [0.0, 1.0],
        "a0": 0.0,
        "oracle": 1.0,
        "best_adapt": 1.0,
        "true_best": "tent_online",
        "sample_provenance": provenance,
        "eval_y": eval_y.tolist(),
        "preds_frozen": frozen.tolist(),
        "regime_label": geo.an.label_regime(1.0),
    }
    return args, checkpoint, grp_rows, sub, y, groups, [record], [condition]


@pytest.mark.parametrize("mutation", ["aa", "Z", "update_norm", "tta_protocol", "provenance"])
def test_geoshift_semantic_resume_rejects_self_resealed_exploits(mutation):
    args, checkpoint, grp_rows, sub, y, groups, records, conditions = (
        _geoshift_semantic_resume_fixture()
    )
    validator = lambda rs, cs: geo._validate_resume_semantics(
        rs,
        cs,
        args=args,
        checkpoint=checkpoint,
        grp_rows=grp_rows,
        sub=sub,
        y=y,
        groups=groups,
    )
    validator(records, conditions)
    records = json.loads(json.dumps(records))
    conditions = json.loads(json.dumps(conditions))
    if mutation == "aa":
        records[0]["aa"] = 0.5
        records[0]["B"] = 0.5
        conditions[0]["aa_all"][1] = 0.5
        conditions[0]["oracle"] = 0.5
        conditions[0]["best_adapt"] = 0.5
    elif mutation == "Z":
        records[0]["Z"].pop()
    elif mutation == "update_norm":
        records[0]["upd_norm"] = 0.5
    elif mutation == "tta_protocol":
        records[0]["tta_protocol"]["mode"] = "forged"
    else:
        conditions[0]["sample_provenance"]["condition_seed"] += 1
        records[0]["sample_provenance"] = conditions[0]["sample_provenance"]
    geo.ri.partial_document(
        run_config_sha256="c" * 64,
        expected_cell_ids=[conditions[0]["cell_id"]],
        records=records,
        conditions=conditions,
        failures=[],
        progress="1/1",
        require_scientific_cell_identity=True,
        semantic_validator=lambda _records, _conditions: None,
    )
    with pytest.raises(geo.ri.RunIntegrityError):
        validator(records, conditions)


def test_rxrx1_condition_fails_unreadable_request_and_preserves_disjointness():
    class FakeSubset:
        indices = np.arange(20) + 1000

        def __len__(self):
            return 20

        def __getitem__(self, position):
            if int(position) == 0:
                raise OSError("broken image")
            return rx.torch.tensor([float(position)]), int(position % 2), {}

    labels = np.tile([0, 1], 10)
    saw_read_failure = False
    for seed in range(100):
        try:
            rx.build_condition(
                FakeSubset(), labels, "iid", 2, 4, np.random.default_rng(seed),
                rx.torch.device("cpu"), n_batches=2, return_ids=True,
            )
        except RuntimeError as exc:
            if "sample substitution is forbidden" in str(exc):
                saw_read_failure = True
                break
    assert saw_read_failure

    class ReadableSubset(FakeSubset):
        def __getitem__(self, position):
            return rx.torch.tensor([float(position)]), int(position % 2), {}

    _, _, eval_y, identities = rx.build_condition(
        ReadableSubset(), labels, "iid", 2, 4, np.random.default_rng(3),
        rx.torch.device("cpu"), n_batches=2, return_ids=True,
    )
    stream = identities["stream_requested_positions"]
    evaluation = identities["eval_requested_positions"]
    assert np.array_equal(stream, identities["stream_resolved_positions"])
    assert np.array_equal(evaluation, identities["eval_resolved_positions"])
    assert len(np.unique(stream)) == len(stream)
    assert np.intersect1d(stream, evaluation).size == 0
    assert eval_y.tolist() == labels[evaluation].tolist()


def test_multiclass_route_summaries_never_claim_a_realized_score():
    rx_summary = rx._route_b_summary([{"route": {"scorable": False}}])
    geo_summary = geo._unsupported_route_b_summary([{}], 62)
    assert rx_summary["status"] == geo_summary["status"] == "UNSUPPORTED"
    assert rx_summary["scorable"] is geo_summary["scorable"] is False
    assert "mean_acc" not in rx_summary
    assert "mean_acc" not in geo_summary


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_rxrx1_manifest_uses_balanced_accuracy_contract_without_acc_aliases(monkeypatch):
    args = _rx_args()
    record = {
        "candidate": "tent_online",
        "metric": "balanced_accuracy",
        "a0": 0.4,
        "aa": 0.5,
        "B": 0.1,
        "Z": [0.0],
        "domain": "rxrx1",
        "regime_label": "helpful",
    }
    condition = {
        "a0": 0.4,
        "oracle": 0.5,
        "best_adapt": 0.5,
        "cand_names": ["freeze_f0", "tent_online"],
        "aa_all": [0.4, 0.5],
        "domain": "rxrx1",
        "regime_label": "helpful",
        "route": {"decision": "ABSTAIN", "status": "UNSUPPORTED", "scorable": False},
        "route_c": rx.rc.unsupported_route_c("balanced_accuracy", 1139),
        "realized": None,
    }
    monkeypatch.setattr(rx.rc, "kbound_summary", lambda *args, **kwargs: {"status": "ok"})
    config = rx._scientific_config(
        args,
        {
            "path": "/ckpt/f0.pt", "sha256": "1" * 64,
            "tensor_sha256": "t" * 64,
        },
        {"sha256": "2" * 64, "n_present": 1, "n_total": 1},
    )
    meta = {
        "scientific_config": config,
        "config_sha256": "3" * 64,
        "ledger": {"execution_complete": True},
        "n_present": 1,
        "n_total": 1,
        "n_classes": 1,
        "population_sha256": "2" * 64,
        "n_cells_done": 1,
        "n_cells_total": 1,
        "wall_sec": 1.0,
    }

    manifest = rx.build_manifest(args, [record], [condition], meta)

    assert manifest["metric"] == "balanced_accuracy"
    assert manifest["metric_contract"]["ordinary_accuracy_alias_allowed"] is False
    assert "mean_acc" not in set(_all_keys(manifest))
    assert manifest["routing_c_smooth_drift"]["status"] == "UNSUPPORTED"
    assert manifest["routing_c_smooth_drift"]["scorable"] is False


@pytest.mark.parametrize(
    ("objective", "n_classes"),
    [("accuracy", 1139), ("accuracy", 62), ("accuracy_binned_wealth", 5)],
)
def test_invalid_route_b_is_unscorable_not_implicit_freeze(objective, n_classes):
    predictions = np.array([
        [0, 1, 2, 3, 0, 1, 2, 3],
        [1, 1, 2, 0, 0, 2, 3, 3],
        [2, 0, 2, 3, 1, 1, 0, 3],
        [3, 1, 0, 3, 0, 2, 2, 1],
    ])
    route = rx.an.multicandidate_route(
        predictions,
        objective=objective,
        n_classes=n_classes,
        anchor_above_chance=False,
    )
    assert route["status"] == "UNSUPPORTED"
    assert route["scorable"] is False
    assert rx.rc.route_realized(route, [0.4, 0.5, 0.45, 0.42]) is None


def test_rxrx1_completion_receipt_binds_complete_result_bytes(tmp_path):
    run_dir = tmp_path / "expected_run"
    run_dir.mkdir()
    config = {"split": "test", "model_seed": 0}
    config_sha256 = rx.ri.stable_sha256(config)
    result = run_dir / f"result_{config_sha256[:8]}.json"
    done = run_dir / ".done"
    manifest = {
        "schema": "kbound_rxrx1_v0.6",
        "config": config,
        "config_sha256": config_sha256,
        "execution_complete": True,
        "completion_ledger": {"execution_complete": True, "expected_cells": 1},
    }
    rx.ri.atomic_json_dump(manifest, result)
    rx.ri.atomic_json_dump(
        rx._completion_receipt(
            result, manifest, run_name=run_dir.name, run_dir=run_dir,
        ),
        done,
    )
    assert rx.validate_completion_receipt(
        done,
        expected_run_name=run_dir.name,
        expected_run_dir=run_dir,
        expected_result_path=result,
        expected_config_sha256=config_sha256,
    ) == str(result.resolve())

    rx.ri.atomic_json_dump({**manifest, "execution_complete": False}, result)
    with pytest.raises(rx.ri.RunIntegrityError, match="hash mismatch"):
        rx.validate_completion_receipt(
            done,
            expected_run_name=run_dir.name,
            expected_run_dir=run_dir,
            expected_result_path=result,
            expected_config_sha256=config_sha256,
        )


def test_rxrx1_completion_receipt_rejects_stale_config_and_foreign_run(tmp_path):
    old_dir = tmp_path / "old_run"
    expected_dir = tmp_path / "expected_run"
    old_dir.mkdir()
    expected_dir.mkdir()
    old_config = {"split": "test", "model_seed": 0}
    new_config = {"split": "test", "model_seed": 1}
    old_sha = rx.ri.stable_sha256(old_config)
    new_sha = rx.ri.stable_sha256(new_config)
    old_result = old_dir / f"result_{old_sha[:8]}.json"
    old_manifest = {
        "schema": "kbound_rxrx1_v0.6",
        "config": old_config,
        "config_sha256": old_sha,
        "execution_complete": True,
        "completion_ledger": {"execution_complete": True, "expected_cells": 1},
    }
    rx.ri.atomic_json_dump(old_manifest, old_result)
    stale = rx._completion_receipt(
        old_result, old_manifest, run_name=old_dir.name, run_dir=old_dir,
    )

    # Copying a perfectly valid receipt into another run directory must not make
    # that run complete, even if all source/result hashes remain self-consistent.
    copied_done = expected_dir / ".done"
    rx.ri.atomic_json_dump(stale, copied_done)
    with pytest.raises(rx.ri.RunIntegrityError, match="expected run context"):
        rx.validate_completion_receipt(
            copied_done,
            expected_run_name=expected_dir.name,
            expected_run_dir=expected_dir,
            expected_result_path=expected_dir / f"result_{new_sha[:8]}.json",
            expected_config_sha256=new_sha,
        )

    # Even under the same run name/path, a receipt for the previous scientific
    # configuration cannot stop a newly configured supervisor invocation.
    same_path_stale = dict(stale)
    same_path_stale["run_name"] = expected_dir.name
    same_path_stale["run_dir"] = str(expected_dir.resolve())
    rx.ri.atomic_json_dump(same_path_stale, copied_done)
    with pytest.raises(rx.ri.RunIntegrityError, match="expected run context"):
        rx.validate_completion_receipt(
            copied_done,
            expected_run_name=expected_dir.name,
            expected_run_dir=expected_dir,
            expected_result_path=expected_dir / f"result_{new_sha[:8]}.json",
            expected_config_sha256=new_sha,
        )


def test_rxrx1_completion_receipt_refuses_incomplete_manifest(tmp_path):
    result = tmp_path / "incomplete.json"
    manifest = {
        "schema": "kbound_rxrx1_v0.6",
        "config": {"split": "test"},
        "config_sha256": "a" * 64,
        "execution_complete": False,
        "completion_ledger": {"execution_complete": False},
    }
    rx.ri.atomic_json_dump(manifest, result)
    with pytest.raises(rx.ri.RunIntegrityError, match="incomplete"):
        rx._completion_receipt(
            result, manifest, run_name=tmp_path.name, run_dir=tmp_path,
        )


def test_rxrx1_supervisor_validates_receipt_instead_of_trusting_file_existence():
    script = (
        ROOT / "experiments/kbound/results/kbound_rxrx1_results/supervise_rxrx1.sh"
    ).read_text(encoding="utf-8")
    assert "--verify-completion" in script
    assert "completion_receipt_valid" in script
    assert "discard_stale_receipt" in script
    assert '"${RUN_ARGS[@]}"' in script
    assert "[ \"$completed\" -eq 1 ] || exit 1" in script
