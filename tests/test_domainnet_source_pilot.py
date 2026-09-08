"""Explicit one-model pilot boundary, exercised only with small synthetic inputs."""

import hashlib
import json
import math
import stat

import pytest
import torch

from experiments.kbound.domainnet import source_data, train_source
from tests.test_domainnet_source_v2 import POLICY, POLICY_SHA, policy_args, training_args

PILOT = POLICY.with_name("DOMAINNET_SOURCE_PILOT_v1.json")


@pytest.mark.parametrize("mode", ["false", 1, None])
def test_library_mode_rejects_non_boolean_values_before_config_or_source_access(tmp_path, mode):
    with pytest.raises(ValueError, match="boolean"):
        train_source.run_training(
            tmp_path / "absent.zip",
            tmp_path / "absent",
            tmp_path / "absent.json",
            "0" * 64,
            tmp_path / "out",
            backend="cpu",
            development_pilot=mode,
        )
    assert not (tmp_path / "out").exists()


def pilot_args(tmp_path):
    archive, dest, _, _, hooks = training_args(tmp_path)
    hooks.epochs = 20
    hooks.batch_size = 32
    return archive, dest, PILOT, hashlib.sha256(PILOT.read_bytes()).hexdigest(), hooks


def test_completed_pilot_records_one_final_model_and_development_scope(tmp_path, monkeypatch):
    # Catches extra seeds, truncated epochs/counts, scope promotion and unsafe checkpoint metadata.
    archive, dest, config, sha, hooks = pilot_args(tmp_path)
    output = tmp_path / "pilot"
    published = []
    original = train_source._progress

    def observe(path, value):
        published.append(dict(value))
        return original(path, value)

    monkeypatch.setattr(train_source, "_progress", observe)
    result = train_source.run_training(
        archive,
        dest,
        config,
        sha,
        output,
        backend="cpu",
        development_pilot=True,
        test_hooks=hooks,
        **policy_args(),
    )
    assert result["status"] == "COMPLETED"
    assert [run["seed"] for run in result["runs"]] == [0]
    assert sorted(p.name for p in output.glob("seed-*")) == ["seed-0"]
    run = result["runs"][0]
    seed_receipt = json.loads((output / "seed-0/receipt.json").read_text())
    checkpoint = torch.load(output / "seed-0/final.pt", weights_only=True, map_location="cpu")
    aggregate = json.loads((output / "receipt.json").read_text())
    assert seed_receipt == run and aggregate == result
    for record in [result, run, checkpoint, checkpoint["record"], *published]:
        assert record["development_pilot"] is True
        assert record["purpose"] == "SOURCE_ONLY_DEVELOPMENT_PILOT"
        assert record["eligible_for_confirmatory"] is False
        assert record["execution_scope"] == "SYNTHETIC_TEST"
        assert record["source_policy"]["sha256"] == POLICY_SHA
    assert published[0]["status"] == "INCOMPLETE" and published[-1]["status"] == "COMPLETED"
    counts = json.loads((dest / "inventory.json").read_text())["counts"]
    assert run["epochs"] == 20
    assert run["processed_count"] == counts["source_fit"] * 20
    assert run["batches"] == math.ceil(counts["source_fit"] / 32) * 20
    epochs = [json.loads(line) for line in (output / "seed-0/epochs.jsonl").read_text().splitlines()]
    assert [epoch["epoch"] for epoch in epochs] == list(range(1, 21))
    assert all(epoch["source_monitor_count"] == counts["source_monitor"] for epoch in epochs)
    assert run["seed_streams"] == {"initialization": 100, "minibatch": 101, "augmentation": 102}
    assert run["candidate"] == "final_epoch_only"
    assert train_source.tensor_hash(checkpoint["model"]) == run["final_tensor_sha256"]
    assert hashlib.sha256((output / "seed-0/final.pt").read_bytes()).hexdigest() == run["checkpoint_sha256"]
    for path in [output / "receipt.json", output / "seed-0/receipt.json", output / "seed-0/final.pt"]:
        assert not (path.stat().st_mode & stat.S_IWUSR)


@pytest.mark.parametrize(
    "fault",
    [
        "ordinary_mode",
        "ordinary_config",
        "no_policy",
        "bad_policy",
        "bad_hash",
        "seeds",
        "epochs",
        "candidate",
        "optimizer",
        "model",
        "streams",
        "extra",
    ],
)
def test_wrong_mode_recipe_or_policy_rejected_before_source_or_output(tmp_path, fault):
    # Missing archive/inventory makes any fallthrough to dataset access observable.
    config = PILOT
    mode = True
    kwargs = policy_args()
    if fault == "ordinary_mode":
        mode = False
    elif fault == "ordinary_config":
        config = POLICY.with_name("DOMAINNET_SOURCE_TRAINING_v2.json")
    elif fault == "no_policy":
        kwargs = {}
    elif fault == "bad_policy":
        kwargs["expected_policy_sha256"] = "0" * 64
    elif fault not in {"bad_hash"}:
        document = json.loads(PILOT.read_text())
        key, value = {
            "seeds": ("seeds", [1]),
            "epochs": ("epochs", 1),
            "candidate": ("candidate", "best_epoch"),
            "optimizer": ("optimizer", {"name": "Adam"}),
            "model": ("model", {"name": "resnet18", "weights": "DEFAULT"}),
            "streams": ("seed_stream_rule", "initialization=seed"),
            "extra": ("resume", True),
        }[fault]
        document[key] = value
        config = tmp_path / "changed.json"
        config.write_text(json.dumps(document))
    sha = "0" * 64 if fault == "bad_hash" else hashlib.sha256(config.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        train_source.run_training(
            tmp_path / "absent.zip",
            tmp_path / "absent",
            config,
            sha,
            tmp_path / "out",
            backend="cpu",
            development_pilot=mode,
            **kwargs,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("pilot, multiplier", [(True, 20), (False, 100)])
def test_preflight_projection_uses_declared_cohort_and_never_writes_checkpoint(tmp_path, pilot, multiplier):
    archive, dest, config, sha, hooks = pilot_args(tmp_path) if pilot else training_args(tmp_path)
    output = tmp_path / "preflight"
    result = train_source.run_training(
        archive,
        dest,
        config,
        sha,
        output,
        backend="cpu",
        development_pilot=pilot,
        preflight_batches=2,
        test_hooks=hooks,
        **policy_args(),
    )
    fit_count = json.loads((dest / "inventory.json").read_text())["counts"]["source_fit"]
    assert result["status"] == "PREFLIGHT_ONLY" and result["eligible_checkpoint"] is False
    assert result["projected_full_training_step_seconds"] == pytest.approx(
        result["mean_step_seconds"] * math.ceil(fit_count / 32) * multiplier
    )
    assert result["projection_scope"].startswith("1 seeds x 20 epochs" if pilot else "5 seeds x 20 epochs")
    if pilot:
        for record in [
            result,
            json.loads((output / "status.json").read_text()),
            json.loads((output / "preflight.json").read_text()),
        ]:
            assert record["development_pilot"] is True
            assert record["purpose"] == "SOURCE_ONLY_DEVELOPMENT_PILOT"
            assert record["eligible_for_confirmatory"] is False
            assert record["execution_scope"] == "SYNTHETIC_TEST"
    assert not list(output.rglob("*.pt"))


def test_pilot_cli_requires_v2_policy_before_source_access(tmp_path):
    with pytest.raises(ValueError, match="policy"):
        train_source.main(
            [
                "--development-pilot",
                "--archive",
                str(tmp_path / "absent.zip"),
                "--inventory-dir",
                str(tmp_path / "absent"),
                "--config",
                str(PILOT),
                "--expected-config-sha256",
                hashlib.sha256(PILOT.read_bytes()).hexdigest(),
                "--output-dir",
                str(tmp_path / "out"),
                "--backend",
                "cpu",
            ]
        )
    assert not (tmp_path / "out").exists()


def test_pilot_rejects_real_class_identity_in_synthetic_hooks(tmp_path):
    archive, dest, config, sha, hooks = pilot_args(tmp_path)
    hooks.identity = source_data.SourceIdentity()
    with pytest.raises(ValueError, match="synthetic reduced-class"):
        train_source.run_training(
            archive,
            dest,
            config,
            sha,
            tmp_path / "out",
            backend="cpu",
            development_pilot=True,
            test_hooks=hooks,
            **policy_args(),
        )
    assert not (tmp_path / "out").exists()


def test_pilot_failure_keeps_development_scope_and_original_storage_error(tmp_path, monkeypatch):
    # Only storage is fault-injected; all training and receipt publication remain real.
    archive, dest, config, sha, hooks = pilot_args(tmp_path)
    output = tmp_path / "failure"
    original = train_source.write_new_json

    def fail_aggregate(path, value):
        if path == output / "receipt.json":
            raise OSError("synthetic aggregate storage failure")
        return original(path, value)

    monkeypatch.setattr(train_source, "write_new_json", fail_aggregate)
    with pytest.raises(OSError, match="aggregate storage failure"):
        train_source.run_training(
            archive, dest, config, sha, output, backend="cpu", development_pilot=True, test_hooks=hooks, **policy_args()
        )
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["purpose"] == "SOURCE_ONLY_DEVELOPMENT_PILOT"
    assert status["development_pilot"] is True and status["eligible_for_confirmatory"] is False
    assert status["execution_scope"] == "SYNTHETIC_TEST"
    assert not (output / "receipt.json").exists()
    assert len(list(output.glob("seed-*/receipt.json"))) == 1
