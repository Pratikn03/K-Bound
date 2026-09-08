"""Synthetic explicit V2 policy regressions; strict V1 remains available."""

import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

import pytest
import torch

from experiments.kbound.domainnet import source_data, train_source
from tests.test_domainnet_source_v1 import fixture, tensor_transform, tiny_model

POLICY = Path(__file__).resolve().parents[1] / "protocols/confirmatory_v2/DOMAINNET_SOURCE_POLICY_v2.json"
POLICY_SHA = "007f2bfb12e8d74f99332d28f675006c493813119275fe47e4f6631d20663e53"


def policy_args():
    return {"policy_path": POLICY, "expected_policy_sha256": POLICY_SHA}


def conflict_fixture(tmp_path):
    def conflict(entries, lines):
        entries[1] = (entries[1][0], entries[0][1])
        entries.append(("clipart/class1/copy.png", entries[0][1]))
        lines.append("clipart/class1/copy.png 1")

    archive, listing, expected = fixture(tmp_path, conflict)
    return archive, listing, source_data.SourceIdentity(**expected)


def prepared(tmp_path, conflicting=True):
    if conflicting:
        archive, listing, identity = conflict_fixture(tmp_path)
    else:
        archive, listing, expected = fixture(tmp_path)
        identity = source_data.SourceIdentity(**expected)
    dest = tmp_path / "inventory"
    source_data.prepare_source(archive, listing, dest, identity=identity, **policy_args())
    return archive, listing, identity, dest


def training_args(tmp_path, *, conflicting=True):
    archive, listing, identity, dest = prepared(tmp_path, conflicting)
    config = Path(__file__).resolve().parents[1] / "protocols/confirmatory_v2/DOMAINNET_SOURCE_TRAINING_v2.json"
    assert config.exists(), "explicit V2 training config is missing"
    hooks = train_source.TestHooks(
        identity=identity,
        model_factory=tiny_model,
        fit_transform=tensor_transform,
        monitor_transform=tensor_transform,
        epochs=2,
        batch_size=17,
    )
    return archive, dest, config, hashlib.sha256(config.read_bytes()).hexdigest(), hooks


def test_v2_retains_all_rows_labels_and_multiplicity_in_one_group(tmp_path):
    archive, listing, identity, dest = prepared(tmp_path)
    inventory = source_data.load_verified_source(dest, archive, identity=identity, **policy_args())
    expected = Counter(
        (path, int(label)) for path, label in (line.split() for line in listing.read_text().splitlines())
    )
    assert Counter((row["path"], row["label"]) for row in inventory["rows"]) == expected
    assert len(inventory["rows"]) == 81
    assert inventory["schema"] == "kbound-domainnet-source-inventory/2"
    assert inventory["source_policy"]["sha256"] == POLICY_SHA
    assert inventory["source_policy"]["document"] == json.loads(POLICY.read_text())
    group_id = hashlib.sha256(struct.pack(">QQ", 12, 12) + bytes([0, 30, 70]) * 144).hexdigest()
    rank = int(
        hashlib.sha256(("kbound-domainnet-source-v1:decoded-group:20260905:" + group_id).encode()).hexdigest(), 16
    )
    split = "source_monitor" if rank < (1 << 256) // 10 else "source_fit"
    manifest = inventory["conflict_manifest"]
    assert len(manifest["groups"]) == 1
    group = manifest["groups"][0]
    assert group["group_id"] == group_id and group["split"] == split
    assert group["multiplicity"] == 3 and group["label_multiplicity"] == {"0": 1, "1": 2}
    assert {r["path"] for r in group["members"]} == {
        "clipart/class0/image0.png",
        "clipart/class1/image1.png",
        "clipart/class1/copy.png",
    }
    assert {r["split"] for r in group["members"]} == {split}
    assert len(group["encoded_sha256s"]) == 1
    assert manifest["counts"]["rows"] == 3 and manifest["counts"]["groups"] == 1
    assert manifest["counts"]["per_class"]["1"] == {"rows": 2, "groups": 1}
    assert manifest["counts"]["per_split"][split]["rows"] == 3
    assert sum(inventory["counts"].values()) == 81


def test_v1_remains_strict_no_automatic_fallback(tmp_path):
    archive, listing, identity = conflict_fixture(tmp_path)
    with pytest.raises(ValueError, match="conflicting labels"):
        source_data.prepare_source(archive, listing, tmp_path / "v1", identity=identity)
    assert not (tmp_path / "v1").exists()


@pytest.mark.parametrize("fault", ["missing_path", "missing_hash", "wrong_hash", "rehashed_policy", "wrong_schema"])
def test_explicit_policy_pair_and_fixed_authority_required(tmp_path, fault):
    archive, listing, identity = conflict_fixture(tmp_path)
    args = policy_args()
    if fault == "missing_path":
        args["policy_path"] = None
    elif fault == "missing_hash":
        args["expected_policy_sha256"] = None
    elif fault == "wrong_hash":
        args["expected_policy_sha256"] = "0" * 64
    else:
        policy = json.loads(POLICY.read_text())
        policy["relabel" if fault == "rehashed_policy" else "schema"] = True
        changed = tmp_path / "policy.json"
        changed.write_text(json.dumps(policy))
        args = {"policy_path": changed, "expected_policy_sha256": hashlib.sha256(changed.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="policy"):
        source_data.prepare_source(archive, listing, tmp_path / "bad", identity=identity, **args)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "fault", ["manifest_omission", "label", "split", "encoded_hash", "counts", "policy", "source_hash"]
)
def test_rehashed_v2_manifest_tampering_rejected(tmp_path, fault):
    archive, _, identity, dest = prepared(tmp_path)
    inventory_path = dest / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    group = inventory["conflict_manifest"]["groups"][0]
    if fault == "manifest_omission":
        inventory["conflict_manifest"]["groups"] = []
    elif fault in {"label", "encoded_hash"}:
        group["members"][0]["label" if fault == "label" else "image_sha256"] = 99
    elif fault == "split":
        group["split"] = "source_monitor" if group["split"] == "source_fit" else "source_fit"
    elif fault == "counts":
        inventory["conflict_manifest"]["counts"]["rows"] = 0
    elif fault == "policy":
        inventory["source_policy"]["sha256"] = "0" * 64
    else:
        inventory["identity"]["archive_sha256"] = "0" * 64
    inventory_path.chmod(0o644)
    inventory_path.write_text(json.dumps(inventory))
    summary_path = dest / "summary.json"
    summary = {key: value for key, value in inventory.items() if key != "rows"}
    summary["inventory_sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    summary_path.chmod(0o644)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        source_data.load_verified_source(dest, archive, identity=identity, **policy_args())


@pytest.mark.parametrize("fault", ["corrupt", "missing", "inconsistent_class"])
def test_v2_retains_other_integrity_gates(tmp_path, fault):
    def change(entries, lines):
        if fault == "corrupt":
            entries[0] = (entries[0][0], b"invalid image")
        elif fault == "missing":
            entries.pop()
        else:
            lines[0] = lines[0].replace(" 0", " 1")

    archive, listing, expected = fixture(tmp_path, change)
    with pytest.raises((ValueError, OSError)):
        source_data.prepare_source(
            archive, listing, tmp_path / "bad", identity=source_data.SourceIdentity(**expected), **policy_args()
        )


def test_v2_five_seed_safe_checkpoints_preserve_policy_and_per_row_training(tmp_path):
    archive, dest, config, sha, hooks = training_args(tmp_path)
    output = tmp_path / "trained"
    result = train_source.run_training(
        archive, dest, config, sha, output, backend="cpu", test_hooks=hooks, **policy_args()
    )
    assert result["status"] == "COMPLETED" and len(result["runs"]) == 5
    fit_count = json.loads((dest / "inventory.json").read_text())["counts"]["source_fit"]
    for run in result["runs"]:
        checkpoint = torch.load(output / run["checkpoint"], weights_only=True, map_location="cpu")
        assert run["source_policy"] == result["source_policy"] == checkpoint["source_policy"]
        assert run["source_policy"]["sha256"] == POLICY_SHA
        assert run["processed_count"] == fit_count * 2
        assert run["batches"] == ((fit_count + 16) // 17) * 2
        assert train_source.tensor_hash(checkpoint["model"]) == run["final_tensor_sha256"]


def test_v2_preflight_is_never_eligible(tmp_path):
    archive, dest, config, sha, hooks = training_args(tmp_path)
    output = tmp_path / "preflight"
    result = train_source.run_training(
        archive, dest, config, sha, output, backend="cpu", preflight_batches=2, test_hooks=hooks, **policy_args()
    )
    assert result["status"] == "PREFLIGHT_ONLY" and result["eligible_checkpoint"] is False
    assert result["source_policy"]["sha256"] == POLICY_SHA
    assert not list(output.rglob("*.pt"))


@pytest.mark.parametrize(
    "fault",
    ["v2_without_policy", "v1_config_v2_policy", "v1_inventory_v2_config", "v2_inventory_v1_config", "incomplete_pair"],
)
def test_training_versions_cannot_be_cross_used(tmp_path, fault):
    archive, dest, config, sha, hooks = training_args(tmp_path, conflicting=False)
    args = policy_args()
    if fault in {"v2_without_policy", "v2_inventory_v1_config"}:
        args = {}
    if fault in {"v1_config_v2_policy", "v2_inventory_v1_config"}:
        config = config.with_name("DOMAINNET_SOURCE_TRAINING_v1.json")
        sha = hashlib.sha256(config.read_bytes()).hexdigest()
    elif fault == "v1_inventory_v2_config":
        v1 = tmp_path / "v1"
        source_data.prepare_source(archive, dest / "source-list.txt", v1, identity=hooks.identity)
        dest = v1
    elif fault == "incomplete_pair":
        args["expected_policy_sha256"] = None
    with pytest.raises(ValueError):
        train_source.run_training(archive, dest, config, sha, tmp_path / "bad", backend="cpu", test_hooks=hooks, **args)
    assert not (tmp_path / "bad").exists()


def test_v2_preserves_v1_recipe_and_nonconflict_split_assignments(tmp_path):
    archive, listing, identity, dest = prepared(tmp_path, conflicting=False)
    v1_dest = tmp_path / "v1"
    source_data.prepare_source(archive, listing, v1_dest, identity=identity)
    v1 = source_data.load_verified_source(v1_dest, archive, identity=identity)
    v2 = source_data.load_verified_source(dest, archive, identity=identity, **policy_args())
    assert v1["rows"] == v2["rows"]
    assert v1["split_salt"] == v2["split_salt"] and v1["split_rule"] == v2["split_rule"]
    assert v2["conflict_manifest"]["groups"] == []
    protocol_dir = POLICY.parent
    config_v1 = json.loads((protocol_dir / "DOMAINNET_SOURCE_TRAINING_v1.json").read_text())
    config_v2 = json.loads((protocol_dir / "DOMAINNET_SOURCE_TRAINING_v2.json").read_text())
    config_v2.pop("source_policy")
    config_v2["schema"] = config_v1["schema"]
    assert config_v1 == config_v2


@pytest.mark.parametrize("partial", ["path", "hash"])
def test_source_cli_requires_explicit_policy_pair_before_source_read(tmp_path, partial):
    args = [
        "--archive",
        str(tmp_path / "absent.zip"),
        "--source-list",
        str(tmp_path / "absent.txt"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    args += ["--source-policy", str(POLICY)] if partial == "path" else ["--expected-policy-sha256", POLICY_SHA]
    with pytest.raises(ValueError, match="supplied together"):
        source_data.main(args)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("partial", ["path", "hash"])
def test_training_cli_requires_explicit_policy_pair_before_config_read(tmp_path, partial):
    args = [
        "--archive",
        str(tmp_path / "absent.zip"),
        "--inventory-dir",
        str(tmp_path / "absent_inventory"),
        "--config",
        str(tmp_path / "absent.json"),
        "--expected-config-sha256",
        "0" * 64,
        "--output-dir",
        str(tmp_path / "out"),
        "--backend",
        "cpu",
    ]
    args += ["--source-policy", str(POLICY)] if partial == "path" else ["--expected-policy-sha256", POLICY_SHA]
    with pytest.raises(ValueError, match="supplied together"):
        train_source.main(args)
    assert not (tmp_path / "out").exists()


def test_v2_policy_file_missing_or_byte_tampered_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        source_data.load_source_policy(tmp_path / "missing.json", POLICY_SHA)
    wrong = tmp_path / "wrong.json"
    wrong.write_bytes(POLICY.read_bytes() + b" ")
    with pytest.raises(ValueError, match="policy SHA256"):
        source_data.load_source_policy(wrong, POLICY_SHA)


def test_v2_preparation_never_overwrites_existing_output(tmp_path):
    archive, listing, identity, dest = prepared(tmp_path)
    before = {path.name: path.read_bytes() for path in dest.iterdir()}
    with pytest.raises(FileExistsError):
        source_data.prepare_source(archive, listing, dest, identity=identity, **policy_args())
    assert {path.name: path.read_bytes() for path in dest.iterdir()} == before
