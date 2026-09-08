from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "reference_source_preflight.py"
DATASET_NAMES = [
    "cifar10c",
    "cifar101",
    "imagenetc",
    "imagenetr",
    "camelyon17",
    "rxrx1",
    "pacs",
    "iwildcam",
    "officehome",
]


def _artifact(role: str, path: Path) -> dict:
    return {
        "role": role,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_module():
    if not SCRIPT.exists():
        pytest.fail("required reference-source preflight module is missing")
    spec = importlib.util.spec_from_file_location("reference_source_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def contract(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "checkpoint.bin"
    data_manifest = tmp_path / "data-manifest.json"
    receipt = tmp_path / "kga-partition-receipt.json"
    checkpoint.write_bytes(b"synthetic checkpoint identity only\n")
    data_manifest.write_bytes(b'{"synthetic": true}\n')
    receipt.write_bytes(b'{"partition": "synthetic"}\n')

    datasets = []
    for name in DATASET_NAMES:
        checkpoints = [checkpoint]
        if name == "rxrx1":
            checkpoints = []
            for replica in range(5):
                replica_path = tmp_path / f"rxrx1-checkpoint-{replica}.bin"
                replica_path.write_bytes(f"synthetic independent replica {replica}\n".encode())
                checkpoints.append(replica_path)
        datasets.append(
            {
                "name": name,
                "blockers": [],
                "reference": {
                    "url": f"https://example.test/reference/{name}",
                    "revision": "published-revision-1",
                },
                "artifacts": [
                    *[_artifact("checkpoint", path) for path in checkpoints],
                    _artifact("data_manifest", data_manifest),
                ],
                "source_ready": True,
                "kga_partition_receipt": _artifact("kga_partition_receipt", receipt),
            }
        )
    return {
        "schema_version": 1,
        "policy": "published_reference_source_then_kga",
        "output_root": str(tmp_path / "fresh-output"),
        "datasets": datasets,
    }


def _assert_blocked(contract: object, text: str) -> list[str]:
    blockers = _load_module().check_contract(contract)
    assert any(text in blocker for blocker in blockers), blockers
    return blockers


def test_valid_all_nine_contract_is_statically_ready_without_creating_output(contract: dict):
    output_root = Path(contract["output_root"])

    blockers = _load_module().check_contract(contract)

    assert blockers == []
    assert not output_root.exists()


def test_cli_missing_manifest_returns_structured_blocker(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = _load_module().main(["--manifest", str(tmp_path / "missing.json")])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["ready"] is False
    assert any("manifest" in blocker and "read" in blocker for blocker in output["blockers"])


def test_cli_rejects_duplicate_json_keys(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    manifest = tmp_path / "duplicate.json"
    manifest.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")

    exit_code = _load_module().main(["--manifest", str(manifest)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["ready"] is False
    assert any("duplicate JSON key" in blocker for blocker in output["blockers"])


def test_false_source_ready_blocks_dataset(contract: dict):
    contract["datasets"][0]["source_ready"] = False

    _assert_blocked(contract, "source_ready")


def test_incomplete_dataset_set_reports_missing_name(contract: dict):
    missing_name = contract["datasets"].pop()["name"]

    _assert_blocked(contract, f"missing dataset: {missing_name}")


def test_wrong_artifact_hash_blocks_contract(contract: dict):
    contract["datasets"][0]["artifacts"][0]["sha256"] = "0" * 64

    _assert_blocked(contract, "SHA-256 mismatch")


def test_symlink_artifact_is_rejected(contract: dict, tmp_path: Path):
    target = tmp_path / "real-checkpoint.bin"
    target.write_bytes(b"synthetic\n")
    link = tmp_path / "checkpoint-link.bin"
    link.symlink_to(target)
    artifact = contract["datasets"][0]["artifacts"][0]
    artifact["path"] = str(link)
    artifact["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()

    _assert_blocked(contract, "symlink")


def test_existing_output_root_is_rejected(contract: dict):
    Path(contract["output_root"]).mkdir()

    _assert_blocked(contract, "output_root already exists")


def test_absent_kga_partition_receipt_blocks_contract(contract: dict):
    del contract["datasets"][0]["kga_partition_receipt"]

    _assert_blocked(contract, "kga_partition_receipt")


@pytest.mark.parametrize("document", [None, 7, "manifest", [], [1, 2]])
def test_check_contract_rejects_non_object_document(document: object):
    _assert_blocked(document, "manifest must be a JSON object")


@pytest.mark.parametrize("raw", ["null", "42", '"manifest"', "[]"])
def test_cli_rejects_scalar_or_list_manifest(
    raw: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    manifest = tmp_path / "malformed-shape.json"
    manifest.write_text(raw, encoding="utf-8")

    exit_code = _load_module().main(["--manifest", str(manifest)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert any("manifest must be a JSON object" in blocker for blocker in output["blockers"])


def test_bool_schema_version_is_rejected(contract: dict):
    contract["schema_version"] = True

    _assert_blocked(contract, "schema_version")


def test_duplicate_dataset_names_are_rejected(contract: dict):
    contract["datasets"][-1]["name"] = contract["datasets"][0]["name"]

    blockers = _assert_blocked(contract, "duplicate dataset name")
    assert any("missing dataset: officehome" in blocker for blocker in blockers)


def test_unknown_dataset_name_is_rejected(contract: dict):
    contract["datasets"][-1]["name"] = "unknown-dataset"

    blockers = _assert_blocked(contract, "unknown dataset")
    assert any("missing dataset: officehome" in blocker for blocker in blockers)


def test_wrong_policy_is_rejected(contract: dict):
    contract["policy"] = "mixed_source_and_kga"

    _assert_blocked(contract, "policy")


@pytest.mark.parametrize("role", ["checkpoint", "data_manifest"])
def test_required_artifact_role_is_enforced(contract: dict, role: str):
    row = contract["datasets"][0]
    row["artifacts"] = [artifact for artifact in row["artifacts"] if artifact["role"] != role]

    _assert_blocked(contract, f"missing artifact role: {role}")


def test_duplicate_artifact_paths_within_dataset_are_rejected(contract: dict):
    row = contract["datasets"][0]
    row["artifacts"][1]["path"] = row["artifacts"][0]["path"]

    _assert_blocked(contract, "duplicate artifact path")


def test_unreadable_artifact_is_structured_blocker(contract: dict):
    artifact_path = Path(contract["datasets"][0]["artifacts"][0]["path"])
    artifact_path.chmod(0)
    try:
        _assert_blocked(contract, "not readable")
    finally:
        artifact_path.chmod(0o600)


def test_output_root_with_symlink_ancestor_is_rejected(contract: dict, tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    contract["output_root"] = str(linked_parent / "fresh-output")

    _assert_blocked(contract, "symlink ancestor")


def test_explicit_operator_blocker_blocks_contract(contract: dict):
    contract["datasets"][0]["blockers"] = ["published recipe is not bound"]

    _assert_blocked(contract, "published recipe is not bound")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("datasets", {}),
        ("policy", ["published_reference_source_then_kga"]),
        ("output_root", 4),
    ],
)
def test_wrong_top_level_field_types_are_rejected(contract: dict, field: str, value: object):
    contract[field] = value

    _assert_blocked(contract, field)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("blockers", "none"),
        ("reference", []),
        ("artifacts", {}),
        ("source_ready", 1),
        ("kga_partition_receipt", []),
    ],
)
def test_wrong_dataset_field_types_are_rejected(contract: dict, field: str, value: object):
    contract["datasets"][0][field] = value

    _assert_blocked(contract, field)


def test_empty_strings_in_nested_fields_are_rejected(contract: dict):
    row = contract["datasets"][0]
    row["blockers"] = [""]
    row["reference"]["url"] = ""
    row["reference"]["revision"] = ""
    row["artifacts"][0]["role"] = ""

    blockers = _load_module().check_contract(contract)

    assert any("blockers" in blocker for blocker in blockers)
    assert any("reference.url" in blocker for blocker in blockers)
    assert any("reference.revision" in blocker for blocker in blockers)
    assert any("artifact role" in blocker for blocker in blockers)


def test_reference_url_must_be_https(contract: dict):
    contract["datasets"][0]["reference"]["url"] = "http://example.test/reference"

    _assert_blocked(contract, "HTTPS")


def test_reference_url_requires_a_network_location(contract: dict):
    contract["datasets"][0]["reference"]["url"] = "https:///missing-host"

    _assert_blocked(contract, "HTTPS")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_cli_rejects_nonfinite_json_constants(
    constant: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    manifest = tmp_path / "nonfinite.json"
    manifest.write_text(f'{{"schema_version": {constant}}}', encoding="utf-8")

    exit_code = _load_module().main(["--manifest", str(manifest)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert any("nonfinite JSON constant" in blocker for blocker in output["blockers"])


def test_cli_valid_contract_emits_only_static_readiness(contract: dict, tmp_path: Path, capsys):
    manifest = tmp_path / "valid.json"
    manifest.write_text(json.dumps(contract), encoding="utf-8")

    exit_code = _load_module().main(["--manifest", str(manifest)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output == {"ready": True, "blockers": []}
    assert not Path(contract["output_root"]).exists()


def test_output_root_requires_absolute_nonempty_path(contract: dict):
    contract["output_root"] = "relative/output"

    _assert_blocked(contract, "absolute")


def test_output_root_requires_existing_parent(contract: dict, tmp_path: Path):
    contract["output_root"] = str(tmp_path / "missing-parent" / "output")

    _assert_blocked(contract, "parent")


def test_receipt_role_must_be_kga_partition_receipt(contract: dict):
    contract["datasets"][0]["kga_partition_receipt"]["role"] = "checkpoint"

    _assert_blocked(contract, "role must be kga_partition_receipt")


def test_receipt_path_cannot_duplicate_an_artifact_path(contract: dict):
    row = contract["datasets"][0]
    row["kga_partition_receipt"]["path"] = row["artifacts"][0]["path"]

    _assert_blocked(contract, "duplicate artifact path")


@pytest.mark.parametrize("checkpoint_count", [1, 3, 4, 6])
def test_rxrx1_requires_exactly_five_checkpoint_artifacts(
    contract: dict, tmp_path: Path, checkpoint_count: int
):
    row = next(dataset for dataset in contract["datasets"] if dataset["name"] == "rxrx1")
    data_manifests = [artifact for artifact in row["artifacts"] if artifact["role"] == "data_manifest"]
    checkpoints = [artifact for artifact in row["artifacts"] if artifact["role"] == "checkpoint"]
    if checkpoint_count == 6:
        sixth_checkpoint = tmp_path / "rxrx1-checkpoint-5.bin"
        sixth_checkpoint.write_bytes(b"synthetic independent replica 5\n")
        checkpoints.append(_artifact("checkpoint", sixth_checkpoint))
    else:
        checkpoints = checkpoints[:checkpoint_count]
    row["artifacts"] = [*checkpoints, *data_manifests]

    _assert_blocked(contract, "exactly five checkpoint artifacts")


def test_rxrx1_checkpoint_copies_with_same_bytes_are_rejected(contract: dict, tmp_path: Path):
    row = next(dataset for dataset in contract["datasets"] if dataset["name"] == "rxrx1")
    checkpoints = [artifact for artifact in row["artifacts"] if artifact["role"] == "checkpoint"]
    first_path = Path(checkpoints[0]["path"])
    copied_path = tmp_path / "copied-rxrx1-checkpoint.bin"
    copied_path.write_bytes(first_path.read_bytes())
    checkpoints[1].update(_artifact("checkpoint", copied_path))

    _assert_blocked(contract, "checkpoint SHA-256 identities must be distinct")


def test_rxrx1_checkpoint_normalized_path_aliases_are_rejected(contract: dict, tmp_path: Path):
    row = next(dataset for dataset in contract["datasets"] if dataset["name"] == "rxrx1")
    checkpoints = [artifact for artifact in row["artifacts"] if artifact["role"] == "checkpoint"]
    first_path = Path(checkpoints[0]["path"])
    alias_directory = tmp_path / "existing-alias-directory"
    alias_directory.mkdir()
    checkpoints[1]["path"] = str(alias_directory / ".." / first_path.name)
    checkpoints[1]["sha256"] = checkpoints[0]["sha256"]

    _assert_blocked(contract, "duplicate artifact path")
