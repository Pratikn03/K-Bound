"""Behavioral boundaries for the two version-2 identity manifests."""

import importlib
import json
from copy import deepcopy
from typing import Any

import pytest


def validate(kind: str, value: object) -> dict:
    # Import inside a test so the initial missing feature is a test failure.
    try:
        module = importlib.import_module("kga.protocol_identity")
    except ModuleNotFoundError:
        pytest.fail("The protocol identity validators have not been implemented")
    result: dict = getattr(module, f"validate_{kind}_manifest")(value)
    return result


@pytest.fixture
def split() -> dict:
    return {
        "schema_version": 2,
        "dataset_id": "example",
        "dataset_version": "v1",
        "environments": {"development": ["dev"], "calibration": ["cal"], "test": ["test"]},
        "episodes": [
            {
                "episode_id": "dev-episode",
                "environment_id": "dev",
                "split": "development",
                "windows": {"U": ["d1"], "V": ["d2"], "E": ["d3"]},
            },
            {
                "episode_id": "cal-episode",
                "environment_id": "cal",
                "split": "calibration",
                "windows": {"U": ["c1"], "V": ["c2"], "E": ["c3"]},
            },
            {
                "episode_id": "test-episode",
                "environment_id": "test",
                "split": "test",
                "windows": {"U": ["t1"], "V": ["t2"], "E": ["t3"]},
            },
        ],
    }


@pytest.fixture
def checkpoint() -> dict:
    return {
        "schema_version": 2,
        "checkpoints": [
            {
                "checkpoint_id": f"checkpoint-{i}",
                "training_run_id": f"run-{i}",
                "training_seed": i,
                "initialization_seed": i,
                "minibatch_seed": i,
                "augmentation_seed": i,
                "sha256": str(i) * 64,
                "training_manifest_sha256": str(i + 5) * 64,
                "source_accuracy": 0.5,
            }
            for i in range(5)
        ],
    }


@pytest.mark.parametrize("kind", ["split", "checkpoint"])
def test_valid_manifest_round_trips_json_and_returns_independent_copy(kind: str, request: Any) -> None:
    value = request.getfixturevalue(kind)
    expected = deepcopy(value)
    result = validate(kind, value)
    assert result == expected
    assert json.loads(json.dumps(result, allow_nan=False)) == expected
    result["schema_version"] = 1
    if kind == "split":
        result["episodes"][0]["windows"]["U"].append("new")
        result["environments"]["test"].append("new-env")
    else:
        result["checkpoints"][0]["source_accuracy"] = 0.0
    assert value == expected


@pytest.mark.parametrize("kind", ["split", "checkpoint"])
@pytest.mark.parametrize("value", [None, False, 2, 2.0, "manifest", [], (), set(), float("nan")])
def test_rejects_non_object_roots(kind: str, value: object) -> None:
    with pytest.raises(ValueError):
        validate(kind, value)


@pytest.mark.parametrize("kind", ["split", "checkpoint"])
@pytest.mark.parametrize("version", [True, False, 2.0, "2", 1, 3, 2**100, None])
def test_schema_version_requires_exact_integer_two(kind: str, version: object, request: Any) -> None:
    value = request.getfixturevalue(kind)
    value["schema_version"] = version
    with pytest.raises(ValueError):
        validate(kind, value)


def replace(value: Any, path: tuple, replacement: object) -> None:
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


SPLIT_ID_PATHS = [
    ("dataset_id",),
    ("dataset_version",),
    ("environments", "development", 0),
    ("episodes", 0, "episode_id"),
    ("episodes", 0, "environment_id"),
    ("episodes", 0, "windows", "U", 0),
]


@pytest.mark.parametrize("path", SPLIT_ID_PATHS)
@pytest.mark.parametrize(
    "bad_id", ["", " leading", "trailing ", "a\nb", "a\x00b", "a\x7fb", "a\x85b", False, 1, None, []]
)
def test_split_rejects_noncanonical_ids(split: dict, path: tuple, bad_id: object) -> None:
    replace(split, path, bad_id)
    with pytest.raises(ValueError):
        validate("split", split)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("environments", "calibration"), ["dev"]),
        (("environments", "test"), ["cal"]),
        (("environments", "development"), ["test"]),
        (("environments", "development"), ["dev", "dev"]),
        (("environments", "development"), ["dev", "uncovered"]),
        (("episodes", 0, "environment_id"), "unknown"),
        (("episodes", 0, "split"), "test"),
        (("episodes", 0, "split"), "unknown"),
        (("episodes", 1, "episode_id"), "dev-episode"),
        (("episodes", 0, "windows", "V"), ["d1"]),
        (("episodes", 0, "windows", "E"), ["d2"]),
        (("episodes", 1, "windows", "U"), ["d3"]),
        (("episodes", 0, "windows", "U"), ["d1", "d1"]),
    ],
)
def test_split_enforces_partition_membership_coverage_and_global_sample_uniqueness(
    split: dict, path: tuple, replacement: object
) -> None:
    replace(split, path, replacement)
    with pytest.raises(ValueError):
        validate("split", split)


@pytest.mark.parametrize("path", [("episodes",), ("environments", "development"), ("episodes", 0, "windows", "U")])
@pytest.mark.parametrize("value", [[], (), {}, "item", None, False])
def test_split_requires_nonempty_json_lists(split: dict, path: tuple, value: object) -> None:
    replace(split, path, value)
    with pytest.raises(ValueError):
        validate("split", split)


@pytest.mark.parametrize("path", [("environments",), ("episodes", 0), ("episodes", 0, "windows")])
@pytest.mark.parametrize("value", [[], None, False, "object"])
def test_split_rejects_malformed_nested_objects(split: dict, path: tuple, value: object) -> None:
    replace(split, path, value)
    with pytest.raises(ValueError):
        validate("split", split)


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        ("split", ()),
        ("split", ("environments",)),
        ("split", ("episodes", 0)),
        ("split", ("episodes", 0, "windows")),
        ("checkpoint", ()),
        ("checkpoint", ("checkpoints", 0)),
    ],
)
@pytest.mark.parametrize("attack", ["missing", "private_path", "outcome", 1])
def test_every_object_requires_exact_fields(kind: str, path: tuple, attack: object, request: Any) -> None:
    value = request.getfixturevalue(kind)
    target = value
    for key in path:
        target = target[key]
    if attack == "missing":
        del target[next(iter(target))]
    else:
        target[attack] = "unexpected"
    with pytest.raises(ValueError):
        validate(kind, value)


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_id",
        "training_run_id",
        "sha256",
        "training_manifest_sha256",
        "training_seed",
        "initialization_seed",
        "minibatch_seed",
        "augmentation_seed",
    ],
)
def test_checkpoint_requires_unique_recorded_independence_prerequisites(checkpoint: dict, field: str) -> None:
    checkpoint["checkpoints"][1][field] = checkpoint["checkpoints"][0][field]
    with pytest.raises(ValueError):
        validate("checkpoint", checkpoint)


@pytest.mark.parametrize("field", ["training_seed", "initialization_seed", "minibatch_seed", "augmentation_seed"])
@pytest.mark.parametrize("seed", [False, True, -1, 2**63, 2**1000, 1.0, "1", None, []])
def test_checkpoint_seeds_require_nonnegative_signed_64_bit_integers(
    checkpoint: dict, field: str, seed: object
) -> None:
    checkpoint["checkpoints"][0][field] = seed
    with pytest.raises(ValueError):
        validate("checkpoint", checkpoint)


def test_checkpoint_accepts_maximum_seed_and_accuracy_endpoints(checkpoint: dict) -> None:
    for field in ["training_seed", "initialization_seed", "minibatch_seed", "augmentation_seed"]:
        checkpoint["checkpoints"][0][field] = 2**63 - 1
    checkpoint["checkpoints"][0]["source_accuracy"] = 0.0
    checkpoint["checkpoints"][1]["source_accuracy"] = 1.0
    assert validate("checkpoint", checkpoint) == checkpoint


@pytest.mark.parametrize(
    "accuracy", [False, True, 0, 1, -0.01, 1.01, float("nan"), float("inf"), -float("inf"), 2**1000, "0.5", None, []]
)
def test_checkpoint_requires_finite_float_accuracy_in_unit_interval(checkpoint: dict, accuracy: object) -> None:
    checkpoint["checkpoints"][0]["source_accuracy"] = accuracy
    with pytest.raises(ValueError):
        validate("checkpoint", checkpoint)


@pytest.mark.parametrize("field", ["sha256", "training_manifest_sha256"])
@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "0" * 63 + "\n", False, 1, None])
def test_checkpoint_requires_lowercase_sha256(checkpoint: dict, field: str, digest: object) -> None:
    checkpoint["checkpoints"][0][field] = digest
    with pytest.raises(ValueError):
        validate("checkpoint", checkpoint)


@pytest.mark.parametrize("field", ["checkpoint_id", "training_run_id"])
@pytest.mark.parametrize("identifier", ["", " x", "x ", "x\ny", "x\x7fy", False, 1, None])
def test_checkpoint_rejects_noncanonical_ids(checkpoint: dict, field: str, identifier: object) -> None:
    checkpoint["checkpoints"][0][field] = identifier
    with pytest.raises(ValueError):
        validate("checkpoint", checkpoint)


@pytest.mark.parametrize("value", [[], (), {}, None, False, "checkpoints"])
def test_checkpoint_requires_nonempty_json_list(checkpoint: dict, value: object) -> None:
    checkpoint["checkpoints"] = value
    with pytest.raises(ValueError):
        validate("checkpoint", checkpoint)


def test_checkpoint_requires_at_least_five_rows(checkpoint: dict) -> None:
    checkpoint["checkpoints"].pop()
    with pytest.raises(ValueError):
        validate("checkpoint", checkpoint)


@pytest.mark.parametrize("value", [[], None, False, "checkpoint"])
def test_checkpoint_rejects_malformed_row(checkpoint: dict, value: object) -> None:
    checkpoint["checkpoints"][0] = value
    with pytest.raises(ValueError):
        validate("checkpoint", checkpoint)
