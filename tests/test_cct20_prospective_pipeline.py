"""Focused integrity and smoke tests for the prospective CCT-20 path."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from experiments.kbound.cct20 import audit_checkpoints as audit
from experiments.kbound.cct20 import prospective_data as target
from experiments.kbound.cct20 import train_source as train
from experiments.kbound.cct20.integrity import IntegrityError, stable_sha256


def _image(path: Path, color: tuple[int, int, int] = (20, 40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _target_envelope(path: Path, images: list[dict], tail: str = "[]") -> None:
    # The tail can be invalid JSON.  A successful preparation still proves the
    # restricted reader stopped at the images-array boundary instead of parsing
    # the annotation value.
    path.write_text(
        '{"info":{"version":"test"},"categories":[],"images":'
        + json.dumps(images)
        + ',"annotations":'
        + tail
        + "}",
        encoding="utf-8",
    )


def _target_row(image_id: int | str, file_name: str, location: int = 1) -> dict:
    return {
        "id": image_id,
        "file_name": file_name,
        "location": location,
        "date_captured": "2016-01-02 03:04:05",
        "seq_id": f"sequence-{image_id}",
        "frame_num": 1,
        "seq_num_frames": 1,
    }


def test_target_manifest_is_images_only_and_has_exact_full_coverage(tmp_path: Path) -> None:
    root = tmp_path / "images"
    _image(root / "a.jpg", (10, 20, 30))
    _image(root / "nested" / "b.png", (30, 20, 10))
    envelope = tmp_path / target.TARGET_ANNOTATIONS_BASENAME
    _target_envelope(
        envelope,
        [
            _target_row("a", "a.jpg", 1),
            _target_row("b", "nested/b.png", 2),
        ],
        tail="THIS_VALUE_IS_INTENTIONALLY_NOT_JSON",
    )

    manifest = target.build_label_free_target_manifest(envelope, root, expected_count=2)
    target.validate_label_free_manifest(manifest)

    assert manifest["coverage"] == {
        "expected_images": 2,
        "declared_images": 2,
        "unique_ids": 2,
        "unique_file_names": 2,
        "resolved_regular_files": 2,
        "fully_decoded_images": 2,
        "coverage_fraction": 1.0,
        "sample_substitution": False,
    }
    contract = manifest["target_annotations_access_contract"]
    assert contract["parsed_top_level_fields"] == ["images"]
    assert contract["skipped_without_deserialization"] == ["info", "categories"]
    assert contract["annotations_field_parsed"] is False
    assert contract["annotations_field_counted"] is False
    assert contract["annotations_field_hashed"] is False
    assert all("label" not in json.dumps(row).lower() for row in manifest["images"])


@pytest.mark.parametrize(
    "images, message",
    [
        (
            [_target_row(1, "a.jpg"), _target_row(1, "b.jpg")],
            "duplicate target image id",
        ),
        (
            [_target_row(1, "a.jpg"), _target_row(2, "a.jpg")],
            "duplicate target image path",
        ),
        (
            [{**_target_row(1, "a.jpg"), "category_id": 3}],
            "label-bearing fields",
        ),
        (
            [{**_target_row(1, "a.jpg"), "unreviewed": "metadata"}],
            "unapproved metadata fields",
        ),
    ],
)
def test_target_manifest_rejects_non_unique_or_unsafe_rows(
    tmp_path: Path, images: list[dict], message: str
) -> None:
    root = tmp_path / "images"
    _image(root / "a.jpg")
    _image(root / "b.jpg")
    envelope = tmp_path / target.TARGET_ANNOTATIONS_BASENAME
    _target_envelope(envelope, images)
    with pytest.raises(IntegrityError, match=message):
        target.build_label_free_target_manifest(envelope, root, expected_count=len(images))


def test_target_manifest_fails_on_missing_corrupt_or_wrong_count(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    envelope = tmp_path / target.TARGET_ANNOTATIONS_BASENAME

    _target_envelope(envelope, [_target_row(1, "missing.jpg")])
    with pytest.raises(IntegrityError, match="is missing"):
        target.build_label_free_target_manifest(envelope, root, expected_count=1)

    (root / "bad.jpg").write_bytes(b"not an image")
    _target_envelope(envelope, [_target_row(1, "bad.jpg")])
    with pytest.raises(IntegrityError, match="does not decode completely"):
        target.build_label_free_target_manifest(envelope, root, expected_count=1)

    _image(root / "ok.jpg")
    _target_envelope(envelope, [_target_row(1, "ok.jpg")])
    with pytest.raises(IntegrityError, match="population count mismatch"):
        target.build_label_free_target_manifest(envelope, root, expected_count=2)


def test_target_reader_refuses_to_traverse_a_field_before_images(tmp_path: Path) -> None:
    path = tmp_path / target.TARGET_ANNOTATIONS_BASENAME
    path.write_text('{"annotations":DO_NOT_READ,"images":[]}', encoding="utf-8")
    with pytest.raises(IntegrityError, match="places 'annotations' before 'images'"):
        list(target.iter_target_image_metadata(path))


def test_target_manifest_hash_and_id_lock_are_enforced(tmp_path: Path) -> None:
    root = tmp_path / "images"
    _image(root / "a.jpg")
    envelope = tmp_path / target.TARGET_ANNOTATIONS_BASENAME
    _target_envelope(envelope, [_target_row(7, "a.jpg")])
    expected_ids = stable_sha256([json.dumps(7)])
    manifest = target.build_label_free_target_manifest(
        envelope,
        root,
        expected_count=1,
        expected_id_set_sha256=expected_ids,
    )
    manifest["coverage"]["declared_images"] = 0
    with pytest.raises(IntegrityError, match="hash does not match"):
        target.validate_label_free_manifest(manifest)


def test_target_manifest_rejects_false_internal_hash_even_with_new_outer_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    _image(root / "a.jpg")
    envelope = tmp_path / target.TARGET_ANNOTATIONS_BASENAME
    _target_envelope(envelope, [_target_row(7, "a.jpg")])
    manifest = target.build_label_free_target_manifest(envelope, root, expected_count=1)
    manifest["identity"]["population_sha256"] = "0" * 64
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = stable_sha256(manifest)
    with pytest.raises(IntegrityError, match="internal population hashes"):
        target.validate_label_free_manifest(manifest)


def test_target_manifest_validator_rejects_path_traversal_after_rehash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    _image(root / "a.jpg")
    envelope = tmp_path / target.TARGET_ANNOTATIONS_BASENAME
    _target_envelope(envelope, [_target_row(7, "a.jpg")])
    manifest = target.build_label_free_target_manifest(envelope, root, expected_count=1)
    manifest["images"][0]["file_name"] = "../a.jpg"
    manifest["samples"][0]["file_name"] = "../a.jpg"
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = stable_sha256(manifest)
    with pytest.raises(IntegrityError, match="unsafe target image path"):
        target.validate_label_free_manifest(manifest)


def _categories(count: int = 16) -> list[dict]:
    return [
        {"id": category_id, "name": name}
        for category_id, name in sorted(train.FROZEN_CATEGORY_ID_TO_NAME.items())[:count]
    ]


def _labeled_document(
    rows: list[tuple[int, str, int, int, str]], *, category_count: int = 16
) -> dict:
    return {
        "images": [
            {
                "id": image_id,
                "file_name": file_name,
                "location": location,
                "seq_id": seq_id,
            }
            for image_id, file_name, _, location, seq_id in rows
        ],
        "annotations": [
            {"id": index, "image_id": image_id, "category_id": category_id}
            for index, (image_id, _, category_id, _, _) in enumerate(rows)
        ],
        "categories": _categories(category_count),
    }


def _sequence_for_role(role: str) -> str:
    for index in range(10_000):
        seq_id = f"sequence-{role}-{index}"
        if train.source_monitor_role(38, seq_id) == role:
            return seq_id
    raise AssertionError(f"could not construct source partition role {role}")


def _source_fixture(tmp_path: Path, *, category_count: int = 16) -> tuple[Path, Path]:
    root = tmp_path / "images"
    rows_train = [
        (1, "train/a.jpg", 30, 38, _sequence_for_role("source_fit")),
        (2, "train/b.jpg", 1, 38, _sequence_for_role("source_monitor")),
    ]
    for image_id, file_name, _, _, _ in rows_train:
        _image(root / file_name, (image_id * 10, 20, 30))
    train_path = tmp_path / train.TRAIN_ANNOTATIONS_BASENAME
    train_path.write_text(json.dumps(_labeled_document(rows_train, category_count=category_count)))
    return root, train_path


def test_source_bundle_uses_only_train_hash_partition_and_preserves_16_way_protocol(
    tmp_path: Path,
) -> None:
    root, train_path = _source_fixture(tmp_path)
    bundle = train.load_training_bundle(train_path, root)
    assert bundle.source.role == "train"
    assert len(bundle.categories) == 16
    assert [row["id"] for row in bundle.categories] == sorted(train.FROZEN_CATEGORY_ID_TO_NAME)
    assert bundle.categories[11] == {"id": 30, "name": "empty"}
    assert len(bundle.source.samples) == 2
    assert len(bundle.source_fit_samples) == 1
    assert len(bundle.source_monitor_samples) == 1
    assert set(inspect.signature(train.load_training_bundle).parameters) == {
        "train_annotations",
        "image_root",
    }
    assert len(bundle.data_sha256) == 64
    preflight = train.build_source_preflight_manifest(bundle)
    assert preflight["status"] == "PASS"
    assert preflight["unused_annotation_roles"] == [
        "cis_val",
        "cis_test",
        "trans_val",
        "trans_test",
    ]
    assert preflight["train_total_n"] == 2
    assert preflight["source_fit_n"] == 1
    assert preflight["source_monitor_n"] == 1
    claimed = preflight.pop("manifest_sha256")
    assert claimed == stable_sha256(preflight)


def test_source_loader_rejects_target_filename_before_parsing(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    forbidden = tmp_path / target.TARGET_ANNOTATIONS_BASENAME
    forbidden.write_text("THIS MUST NEVER BE PARSED", encoding="utf-8")
    with pytest.raises(IntegrityError, match="must be loaded from a file named"):
        train.load_labeled_split(
            forbidden,
            root,
            role="train",
            expected_basename=train.TRAIN_ANNOTATIONS_BASENAME,
        )


def test_source_bundle_rejects_animal_only_15_class_variant(tmp_path: Path) -> None:
    root, train_path = _source_fixture(tmp_path, category_count=15)
    with pytest.raises(IntegrityError, match="sealed 16-output sparse-id mapping"):
        train.load_training_bundle(train_path, root)


def test_source_loader_collapses_repeats_and_keeps_complete_distinct_category_set(
    tmp_path: Path,
) -> None:
    root, train_path = _source_fixture(tmp_path)
    document = json.loads(train_path.read_text())
    # Image 1 has two boxes of category 6 and one of category 5.  The sealed
    # contract retains {5,6}, never chooses the first annotation.
    document["annotations"] = [
        {"id": 10, "image_id": 1, "category_id": 6},
        {"id": 11, "image_id": 1, "category_id": 6},
        {"id": 12, "image_id": 1, "category_id": 5},
        {"id": 13, "image_id": 2, "category_id": 1},
    ]
    train_path.write_text(json.dumps(document))
    split = train.load_labeled_split(
        train_path,
        root,
        role="train",
        expected_basename=train.TRAIN_ANNOTATIONS_BASENAME,
    )
    assert split.samples[0].labels == (2, 3)  # sparse category ids 5 and 6
    dataset = train.CCTClassificationDataset(split.samples, train.transforms.ToTensor())
    _, soft_target = dataset[0]
    assert torch.count_nonzero(soft_target).item() == 2
    assert soft_target[2].item() == pytest.approx(0.5)
    assert soft_target[3].item() == pytest.approx(0.5)
    assert soft_target.sum().item() == pytest.approx(1.0)


def test_normalized_soft_target_cross_entropy_matches_manual_value() -> None:
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.0, 2.0, 1.0]])
    targets = torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
    expected = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    actual = train.normalized_soft_target_cross_entropy(logits, targets)
    assert actual.item() == pytest.approx(expected.item())


def test_source_monitor_selector_keeps_highest_score_and_earliest_tie() -> None:
    selector = train.BestEpochSelector()
    assert selector.observe(0.50, 0) is True
    assert selector.observe(0.50, 1) is False
    assert selector.observe(0.51, 2) is True
    assert selector.best_epoch == 2
    assert selector.best_metric == 0.51


def test_model_builder_requests_resnet50_imagenet_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class FakeResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 1000)

    def fake_resnet50(*, weights: object) -> FakeResNet:
        observed["weights"] = weights
        return FakeResNet()

    monkeypatch.setattr(train.tvm, "resnet50", fake_resnet50)
    model = train.build_resnet50_imagenet(16)
    assert observed["weights"] is train.tvm.ResNet50_Weights.IMAGENET1K_V2
    assert model.fc.out_features == 16


def test_training_constants_match_frozen_draft_protocol() -> None:
    assert train.TRAINING_EPOCHS == 10
    assert train.PHYSICAL_BATCH_SIZE == 32
    assert train.GRADIENT_ACCUMULATION_STEPS == 4
    assert train.EFFECTIVE_BATCH_SIZE == 128
    assert train.LEARNING_RATE == 0.01
    assert train.MOMENTUM == 0.9
    assert train.WEIGHT_DECAY == 1e-4
    assert train.SCHEDULER_ETA_MIN == 1e-6
    assert train.SOURCE_MONITOR_FRACTION == 0.10
    assert train.SOURCE_PARTITION_UNIT == ("location", "seq_id")
    assert [train.accumulation_divisor(index, 10) for index in range(10)] == [
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        2,
        2,
    ]
    # With 12,083 real source-fit samples, the final optimizer group contains
    # 32 + 19 samples.  Weight the means by samples, not one half each.
    assert train.accumulation_loss_weight(376, 378, 32, 12_083) == pytest.approx(32 / 51)
    assert train.accumulation_loss_weight(377, 378, 19, 12_083) == pytest.approx(19 / 51)


def _checkpoint(path: Path, seed: int, tensor_value: float) -> None:
    config = {"model_seed": seed, "learning_rate": 0.01}
    torch.save(
        {
            "schema": audit.CHECKPOINT_SCHEMA,
            "model_state": {"weight": torch.tensor([tensor_value])},
            "model_seed": seed,
            "checkpoint_tensor_sha256": audit.tensor_state_sha256(
                {"weight": torch.tensor([tensor_value])}
            ),
            "initial_tensor_sha256": stable_sha256({"initial_seed": seed}),
            "imagenet_backbone_tensor_sha256": "b" * 64,
            "config": config,
            "config_sha256": stable_sha256(config),
            "data_sha256": "d" * 64,
            "code_sha256": "c" * 64,
        },
        path,
    )


def test_checkpoint_audit_rejects_duplicate_tensor_hash_even_if_files_differ(tmp_path: Path) -> None:
    for seed in audit.CANONICAL_MODEL_SEEDS:
        _checkpoint(tmp_path / f"seed{seed}.pt", seed, 0.0 if seed < 2 else float(seed))
    with pytest.raises(IntegrityError, match="duplicate checkpoint tensor hash"):
        audit.audit_checkpoint_set(str(tmp_path / "seed{seed}.pt"))


def test_checkpoint_audit_accepts_exact_five_independent_seeds(tmp_path: Path) -> None:
    for seed in audit.CANONICAL_MODEL_SEEDS:
        _checkpoint(tmp_path / f"seed{seed}.pt", seed, float(seed))
    result = audit.audit_checkpoint_set(str(tmp_path / "seed{seed}.pt"))
    assert result["status"] == "PASS"
    assert result["n_checkpoints"] == 5
    assert result["all_file_hashes_distinct"] is True
    assert result["all_tensor_hashes_distinct"] is True
    assert result["all_initial_tensor_hashes_distinct"] is True


def test_checkpoint_audit_recomputes_config_hash(tmp_path: Path) -> None:
    for seed in audit.CANONICAL_MODEL_SEEDS:
        path = tmp_path / f"seed{seed}.pt"
        _checkpoint(path, seed, float(seed))
    path = tmp_path / "seed3.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["learning_rate"] = 99.0
    torch.save(payload, path)
    with pytest.raises(IntegrityError, match="config hash claim is false"):
        audit.audit_checkpoint_set(str(tmp_path / "seed{seed}.pt"))


def test_checkpoint_audit_rejects_seed_specific_recipe_change(tmp_path: Path) -> None:
    for seed in audit.CANONICAL_MODEL_SEEDS:
        path = tmp_path / f"seed{seed}.pt"
        _checkpoint(path, seed, float(seed))
    path = tmp_path / "seed4.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["learning_rate"] = 0.02
    payload["config_sha256"] = stable_sha256(payload["config"])
    torch.save(payload, path)
    with pytest.raises(IntegrityError, match="do not share one training recipe"):
        audit.audit_checkpoint_set(str(tmp_path / "seed{seed}.pt"))


def test_tensor_hash_supports_scalar_batchnorm_counters() -> None:
    digest = audit.tensor_state_sha256(
        {
            "weight": torch.tensor([1.0]),
            "num_batches_tracked": torch.tensor(0, dtype=torch.long),
        }
    )
    assert len(digest) == 64
