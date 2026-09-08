from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.kbound.reference_source import cifar_population as cifar


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tiny_inputs(tmp_path: Path, *, n: int = 5) -> tuple[Path, Path, cifar.Cifar10CScope]:
    c10c = tmp_path / "CIFAR-10-C"
    c101 = tmp_path / "CIFAR-10.1"
    c10c.mkdir()
    c101.mkdir()
    scope = cifar.Cifar10CScope(
        standard_corruptions=("gaussian_noise",),
        validation_corruptions=("spatter",),
        severities=(1, 2),
        base_count=n,
        full=False,
    )
    base = np.arange(n * 32 * 32 * 3, dtype=np.uint32).reshape(n, 32, 32, 3)
    blocks = np.concatenate([(base + severity) % 256 for severity in scope.severities]).astype(np.uint8)
    for corruption in scope.corruptions:
        np.save(c10c / f"{corruption}.npy", blocks)
    labels = np.arange(n, dtype=np.uint8) % 10
    np.save(c10c / "labels.npy", np.tile(labels, len(scope.severities)))
    np.save(c101 / "cifar10.1_v6_data.npy", base.astype(np.uint8))
    np.save(c101 / "cifar10.1_v6_labels.npy", labels.astype(np.int32))
    return c10c, c101, scope


def _expected_hashes(c10c: Path, c101: Path, scope: cifar.Cifar10CScope) -> dict:
    files = {
        **{f"cifar10c/{name}.npy": _sha256(c10c / f"{name}.npy") for name in scope.corruptions},
        "cifar10c/labels.npy": _sha256(c10c / "labels.npy"),
        "cifar101/cifar10.1_v6_data.npy": _sha256(c101 / "cifar10.1_v6_data.npy"),
        "cifar101/cifar10.1_v6_labels.npy": _sha256(c101 / "cifar10.1_v6_labels.npy"),
    }
    return {"schema": "kbound_cifar_expected_hashes_v1", "files": files}


def _prepare_tiny(tmp_path: Path, *, seed: int = 17) -> tuple[dict, dict, Path]:
    c10c, c101, scope = _write_tiny_inputs(tmp_path)
    output = tmp_path / "prepared"
    result = cifar.prepare_cifar_inputs(
        cifar10c_root=c10c,
        cifar101_root=c101,
        output_dir=output,
        partition_seed=seed,
        expected_hashes=_expected_hashes(c10c, c101, scope),
        cifar10c_scope=scope,
        cifar101_count=5,
        require_full=False,
    )
    acceptance = json.loads(result.data_acceptance_path.read_text(encoding="utf-8"))
    partition = json.loads(result.partition_receipt_path.read_text(encoding="utf-8"))
    return acceptance, partition, output


def _independent_order(dataset: str, ids: list[int], seed: int) -> list[int]:
    def key(base_id: int) -> tuple[bytes, int]:
        payload = f"kbound-cifar-base-partition-v1\0{seed}\0{dataset}\0{base_id}".encode()
        return hashlib.sha256(payload).digest(), base_id

    return sorted(ids, key=key)


def test_full_scope_declares_all_standard_and_validation_corruptions() -> None:
    scope = cifar.FULL_CIFAR10C_SCOPE

    assert len(scope.standard_corruptions) == 15
    assert len(scope.validation_corruptions) == 4
    assert len(scope.corruptions) == 19
    assert set(scope.standard_corruptions).isdisjoint(scope.validation_corruptions)
    assert scope.severities == (1, 2, 3, 4, 5)
    assert scope.base_count == 10_000
    assert cifar.FULL_CIFAR101_COUNT == 2_000


def test_partition_is_hand_checked_exact_60_20_20_and_deterministic() -> None:
    ids = list(range(10))

    first = cifar.partition_base_ids("cifar10c", ids, seed=17, expected_count=10)
    second = cifar.partition_base_ids("cifar10c", list(reversed(ids)), seed=17, expected_count=10)

    order = _independent_order("cifar10c", ids, 17)
    assert first == second
    assert first["fit"] == sorted(order[:6])
    assert first["cal"] == sorted(order[6:8])
    assert first["check"] == sorted(order[8:])
    assert set(first["fit"]).isdisjoint(first["cal"])
    assert set(first["fit"]).isdisjoint(first["check"])
    assert set(first["cal"]).isdisjoint(first["check"])
    assert sorted(first["fit"] + first["cal"] + first["check"]) == ids


@pytest.mark.parametrize("ids", [[0, 1, 1, 3, 4], [0, 1, 2, 4]])
def test_partition_rejects_duplicate_or_missing_base_ids(ids: list[int]) -> None:
    with pytest.raises(cifar.CifarPopulationError, match="duplicate|exactly cover"):
        cifar.partition_base_ids("cifar10c", ids, seed=1, expected_count=5)


def test_partition_requires_population_divisible_by_five() -> None:
    with pytest.raises(cifar.CifarPopulationError, match="divisible by 5"):
        cifar.partition_base_ids("cifar10c", list(range(6)), seed=1, expected_count=6)


def test_tiny_real_npy_fixture_is_inventoryable_but_never_full(tmp_path: Path) -> None:
    acceptance, partition, output = _prepare_tiny(tmp_path)

    assert acceptance["status"] == "NON_FULL_TEST_FIXTURE"
    assert acceptance["eligible_for_full_campaign"] is False
    assert acceptance["datasets"]["cifar10c"]["base_image_count"] == 5
    assert acceptance["datasets"]["cifar10c"]["expanded_image_count"] == 20
    assert acceptance["datasets"]["cifar101"]["base_image_count"] == 5
    assert partition["eligible_for_full_campaign"] is False
    assert partition["proportions"] == {"fit": 0.6, "cal": 0.2, "check": 0.2}
    assert partition["membership_counts"]["cifar10c"] == {"fit": 3, "cal": 1, "check": 1}
    assert (output / "data-acceptance.json").is_file()
    assert (output / "kga-partition-receipt.json").is_file()


def test_cross_corruption_and_severity_rows_reuse_one_base_assignment(tmp_path: Path) -> None:
    _, partition, _ = _prepare_tiny(tmp_path)

    rows = list(cifar.iter_cifar10c_role_members(partition, "fit"))

    assert len(rows) == 2 * 2 * 3
    roles_by_base: dict[int, set[str]] = {}
    blocks_by_base: dict[int, set[tuple[str, int]]] = {}
    for row in rows:
        roles_by_base.setdefault(row["base_id"], set()).add(row["role"])
        blocks_by_base.setdefault(row["base_id"], set()).add((row["corruption"], row["severity"]))
    assert all(roles == {"fit"} for roles in roles_by_base.values())
    assert all(len(blocks) == 4 for blocks in blocks_by_base.values())


def test_role_expansion_rejects_tampered_membership(tmp_path: Path) -> None:
    _, partition, _ = _prepare_tiny(tmp_path)
    partition["membership"]["cifar10c"]["fit"][0], partition["membership"]["cifar10c"]["cal"][0] = (
        partition["membership"]["cifar10c"]["cal"][0],
        partition["membership"]["cifar10c"]["fit"][0],
    )

    with pytest.raises(cifar.CifarPopulationError, match="assignment mismatch"):
        list(cifar.iter_cifar10c_role_members(partition, "fit"))


def test_missing_corruption_is_a_hard_error_without_output(tmp_path: Path) -> None:
    c10c, c101, scope = _write_tiny_inputs(tmp_path)
    expected = _expected_hashes(c10c, c101, scope)
    (c10c / "spatter.npy").unlink()
    output = tmp_path / "prepared"

    with pytest.raises(FileNotFoundError, match="spatter"):
        cifar.prepare_cifar_inputs(
            c10c,
            c101,
            output,
            3,
            expected,
            cifar10c_scope=scope,
            cifar101_count=5,
            require_full=False,
        )

    assert not output.exists()


def test_missing_severity_block_is_rejected(tmp_path: Path) -> None:
    c10c, c101, scope = _write_tiny_inputs(tmp_path)
    path = c10c / "gaussian_noise.npy"
    array = np.load(path, allow_pickle=False)[:5]
    np.save(path, array)
    expected = _expected_hashes(c10c, c101, scope)

    with pytest.raises(cifar.CifarPopulationError, match="severity blocks|shape"):
        cifar.prepare_cifar_inputs(
            c10c,
            c101,
            tmp_path / "prepared",
            3,
            expected,
            cifar10c_scope=scope,
            cifar101_count=5,
            require_full=False,
        )


def test_hash_mutation_stops_before_output_creation(tmp_path: Path) -> None:
    c10c, c101, scope = _write_tiny_inputs(tmp_path)
    expected = _expected_hashes(c10c, c101, scope)
    path = c101 / "cifar10.1_v6_data.npy"
    changed = np.load(path, allow_pickle=False)
    changed[0, 0, 0, 0] ^= np.uint8(1)
    np.save(path, changed)
    output = tmp_path / "prepared"

    with pytest.raises(cifar.CifarPopulationError, match="SHA-256 mismatch"):
        cifar.prepare_cifar_inputs(
            c10c,
            c101,
            output,
            3,
            expected,
            cifar10c_scope=scope,
            cifar101_count=5,
            require_full=False,
        )

    assert not output.exists()


def test_incorrect_cifar101_population_count_is_rejected(tmp_path: Path) -> None:
    c10c, c101, scope = _write_tiny_inputs(tmp_path)
    data_path = c101 / "cifar10.1_v6_data.npy"
    label_path = c101 / "cifar10.1_v6_labels.npy"
    np.save(data_path, np.load(data_path, allow_pickle=False)[:4])
    np.save(label_path, np.load(label_path, allow_pickle=False)[:4])
    expected = _expected_hashes(c10c, c101, scope)

    with pytest.raises(cifar.CifarPopulationError, match="2000|expected 5|population"):
        cifar.prepare_cifar_inputs(
            c10c,
            c101,
            tmp_path / "prepared",
            3,
            expected,
            cifar10c_scope=scope,
            cifar101_count=5,
            require_full=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [("wrong_dtype", "dtype"), ("nonfinite", "non-finite"), ("bad_labels", "label")],
)
def test_invalid_array_or_label_schema_fails_closed(tmp_path: Path, mutation: str, message: str) -> None:
    c10c, c101, scope = _write_tiny_inputs(tmp_path)
    data_path = c101 / "cifar10.1_v6_data.npy"
    label_path = c101 / "cifar10.1_v6_labels.npy"
    if mutation == "wrong_dtype":
        np.save(data_path, np.zeros((5, 32, 32, 3), dtype=np.int16))
    elif mutation == "nonfinite":
        values = np.zeros((5, 32, 32, 3), dtype=np.float32)
        values[0, 0, 0, 0] = np.nan
        np.save(data_path, values)
    else:
        labels = np.load(label_path, allow_pickle=False)
        labels[0] = 10
        np.save(label_path, labels)
    expected = _expected_hashes(c10c, c101, scope)

    with pytest.raises(cifar.CifarPopulationError, match=message):
        cifar.prepare_cifar_inputs(
            c10c,
            c101,
            tmp_path / "prepared",
            3,
            expected,
            cifar10c_scope=scope,
            cifar101_count=5,
            require_full=False,
        )


def test_existing_output_is_preserved_without_rehash_or_rewrite(tmp_path: Path) -> None:
    c10c, c101, scope = _write_tiny_inputs(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve exactly\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output"):
        cifar.prepare_cifar_inputs(
            c10c,
            c101,
            output,
            3,
            _expected_hashes(c10c, c101, scope),
            cifar10c_scope=scope,
            cifar101_count=5,
            require_full=False,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve exactly\n"
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]


def test_partition_receipt_contains_no_target_labels(tmp_path: Path) -> None:
    acceptance, partition, _ = _prepare_tiny(tmp_path)

    assert "labels" not in json.dumps(partition).lower()
    for dataset in acceptance["datasets"].values():
        label_acceptance = dataset["label_acceptance"]
        assert label_acceptance["values_exported"] is False
        assert set(label_acceptance) == {
            "path",
            "sha256",
            "shape",
            "dtype",
            "class_min",
            "class_max",
            "values_exported",
        }


def test_adaptation_and_evaluation_windows_must_be_distinct_and_in_role() -> None:
    membership = [0, 1, 2, 3]

    assert cifar.validate_distinct_windows([0, 2], [1, 3], membership) == {
        "adaptation_count": 2,
        "evaluation_count": 2,
        "disjoint": True,
    }
    with pytest.raises(cifar.CifarPopulationError, match="overlap"):
        cifar.validate_distinct_windows([0, 1], [1, 2], membership)
    with pytest.raises(cifar.CifarPopulationError, match="outside"):
        cifar.validate_distinct_windows([0], [4], membership)


@pytest.mark.parametrize('adaptation,evaluation,membership', [
    ([True], [2], [1, 2]), ([], [1], [0, 1]), ([0], [], [0, 1]),
    ([0], [1], [0, 1, 1]), ([0.0], [1], [0, 1]),
])
def test_window_validation_rejects_type_coercion_and_empty_windows(adaptation, evaluation, membership):
    with pytest.raises(cifar.CifarPopulationError):
        cifar.validate_distinct_windows(adaptation, evaluation, membership)


def test_role_expansion_rejects_inconsistent_corruption_scope(tmp_path):
    _, partition, _ = _prepare_tiny(tmp_path)
    partition['cifar10c_scope']['corruptions'] = ['gaussian_noise']
    with pytest.raises(cifar.CifarPopulationError, match='scope'):
        list(cifar.iter_cifar10c_role_members(partition, 'fit'))


def test_acceptance_does_not_invent_publisher_or_source_selection_verification(tmp_path):
    acceptance, partition, _ = _prepare_tiny(tmp_path)
    assert acceptance['upstream_download_authenticated'] is False
    assert partition['data_use_boundaries']['source_model_selection_exposure'] == 'NOT_VERIFIED_BY_INPUT_INDEX'


def test_full_cli_refuses_tiny_fixture_and_creates_no_receipt(tmp_path: Path) -> None:
    c10c, c101, scope = _write_tiny_inputs(tmp_path)
    hashes = tmp_path / "expected-hashes.json"
    hashes.write_text(json.dumps(_expected_hashes(c10c, c101, scope)), encoding="utf-8")
    output = tmp_path / "prepared"

    with pytest.raises(cifar.CifarPopulationError, match="full CIFAR"):
        cifar.main(
            [
                "--cifar10c-root",
                str(c10c),
                "--cifar101-root",
                str(c101),
                "--output-dir",
                str(output),
                "--partition-seed",
                "17",
                "--expected-hashes",
                str(hashes),
            ]
        )

    assert not output.exists()
