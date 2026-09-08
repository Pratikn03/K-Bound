from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


@lru_cache(None)
def module():
    path = Path(__file__).parents[1] / "experiments/kbound/reference_source/train_rxrx1.py"
    assert path.is_file(), "reference RxRx1 source training stage missing"
    spec = importlib.util.spec_from_file_location("rx_source", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def toy():
    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.2), torch.nn.Linear(5, 2))
    ds = TensorDataset(torch.arange(21, dtype=torch.float32).view(7, 3) / 20,
                       torch.tensor([0, 1, 0, 1, 0, 1, 0]))
    loader = DataLoader(ds, batch_size=3, shuffle=True)
    opt, sch = module().optimizer_scheduler(model, total_steps=12, warmup_steps=2)
    return model, loader, opt, sch


def test_warmup_and_half_cosine_endpoints():
    m = module()
    assert [m.lr_factor(s, 10, 2) for s in [0, 1, 2, 6, 10]] == pytest.approx([0, .5, 1, .5, 0])


def test_real_updates_include_short_final_batch_and_account_every_example():
    model, loader, opt, sch = toy()
    before = [p.detach().clone() for p in model.parameters()]
    r = module().run_epoch(model, loader, "cpu", optimizer=opt, scheduler=sch)
    assert r["examples"] == 7 and r["batches"] == 3
    assert sch.last_epoch == 3
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert 0 <= r["accuracy"] <= 1 and r["ce"] > 0


def test_eval_does_not_update_model_or_optimizer():
    model, loader, opt, sch = toy()
    before = [p.detach().clone() for p in model.parameters()]
    module().run_epoch(model, loader, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert len(opt.state) == 0 and sch.last_epoch == 0


def test_eval_standardization_uses_sample_std_and_keeps_constant_channel_finite():
    image = Image.fromarray(np.array([[[0, 0, 9], [255, 127, 9]],
                                     [[0, 255, 9], [255, 0, 9]]], dtype=np.uint8))
    t = module().image_transform(image, training=False)
    assert t.shape == (3, 2, 2)
    assert t[0].flatten().tolist() == pytest.approx([-.8660254, .8660254, -.8660254, .8660254])
    assert torch.equal(t[2], torch.zeros(2, 2)) and torch.isfinite(t).all()


def test_labels_are_accessed_only_for_train_and_val(tmp_path):
    p = tmp_path / "metadata.csv"
    p.write_text("sirna_id\n12\nMUST_NOT_PARSE\n1138\nMUST_NOT_PARSE\n")
    rows = [{"row_id": i, "split": split} for i, split in enumerate(["train", "id_test", "val", "test"])]
    labels = module().source_labels(p, rows)
    assert labels == {0: 12, 2: 1138}
    p.write_text("sirna_id\n-1\nMUST_NOT_PARSE\n1138\nMUST_NOT_PARSE\n")
    with pytest.raises(module().SourceError, match="label"):
        module().source_labels(p, rows)


def test_per_sample_hash_is_checked_before_decoding(tmp_path):
    p = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(p)
    row = {"row_id": 0, "path": "sample.png", "sha256": digest(p), "bytes": p.stat().st_size}
    ds = module().SourceDataset(tmp_path, [row], {0: 2}, training=False)
    assert ds[0][1] == 2
    p.write_bytes(b"changed")
    with pytest.raises(module().SourceError, match="identity"):
        ds[0]
    p.unlink()
    with pytest.raises(module().SourceError, match="file|image"):
        ds[0]


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_source_dataset_converts_valid_image_modes_to_three_channels(tmp_path, mode):
    p = tmp_path / "sample.png"
    Image.new(mode, (4, 4)).save(p)
    row = {"row_id": 0, "path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
    ds = module().SourceDataset(tmp_path, [row], {0: 4}, training=False)
    x, label = ds[0]
    assert x.shape == (3, 4, 4) and torch.isfinite(x).all() and label == 4


@pytest.mark.parametrize("field,value", [("complete", 1), ("reference_population_complete", False), ("family", "camelyon17")])
def test_population_receipt_rejects_wrong_status_before_data_access(tmp_path, field, value):
    receipt = {"complete": True, "reference_population_complete": True, "family": "rxrx1"}
    receipt[field] = value
    (tmp_path / "completion.json").write_text(json.dumps(receipt))
    with pytest.raises(module().SourceError, match="population"):
        module().accept_population(tmp_path)


def test_population_index_rejects_duplicate_or_missing_ids_and_split_changes():
    m = module()
    expected = [{"row_id": 0, "path": "a.png", "split": "train", "group": {}},
                {"row_id": 1, "path": "b.png", "split": "val", "group": {}}]
    rows = [{**r, "sha256": "a" * 64, "bytes": 10, "width": 2, "height": 2} for r in expected]
    m.validate_index(rows, expected)
    for bad in [rows[:1], [rows[0], rows[0]], [rows[0], {**rows[1], "split": "test"}]]:
        with pytest.raises(m.SourceError, match="index"):
            m.validate_index(bad, expected)


def test_initialization_requires_expected_hash_and_historical_lineage(tmp_path):
    p = tmp_path / "initialization.pth"
    p.write_bytes(b"not weights")
    with pytest.raises(module().SourceError, match="initialization"):
        module().load_initialization(p, "19c8e357" + "0" * 56)
    with pytest.raises(module().SourceError, match="lineage"):
        module().load_initialization(p, digest(p))


def test_existing_output_is_preserved(tmp_path):
    p = tmp_path / "run"
    p.mkdir()
    (p / "sentinel").write_text("keep")
    with pytest.raises(module().SourceError, match="fresh"):
        module().prepare_output(p, {"identity": "a"}, resume=False)
    assert (p / "sentinel").read_text() == "keep"


@pytest.mark.parametrize("kind", ["missing", "index_hash", "metadata_hash"])
def test_population_missing_or_altered_identity_stops_before_source_labels(tmp_path, kind):
    m = module()
    if kind != "missing":
        root = tmp_path / "data"
        root.mkdir()
        metadata = root / "metadata.csv"
        metadata.write_text("sirna_id\nMUST_NOT_PARSE\n")
        index = tmp_path / "image-index.jsonl"
        index.write_text("{}\n")
        counts = {"train": 40612, "id_test": 40612, "val": 9854, "test": 34432}
        receipt = {"complete": True, "reference_population_complete": True, "family": "rxrx1",
                   "counts": counts, "expected_counts": counts, "root": str(root),
                   "index_file": "image-index.jsonl", "index_sha256": digest(index),
                   "metadata_sha256": digest(metadata)}
        receipt["index_sha256" if kind == "index_hash" else "metadata_sha256"] = "0" * 64
        (tmp_path / "completion.json").write_text(json.dumps(receipt))
    with pytest.raises(m.SourceError, match="file|SHA256"):
        m.accept_population(tmp_path)


def test_resume_selection_rejects_wrong_best_epoch_even_with_valid_file_hash(tmp_path):
    m = module()
    model, loader, opt, scheduler = toy()
    history = [{"epoch": 0, "val": {"accuracy": .8}}, {"epoch": 1, "val": {"accuracy": .6}}]
    p = tmp_path / "not-best.pt"
    m.save_training_state(p, model, opt, scheduler, {"fixture": True}, history, "cpu")
    pointer = {"checkpoint": p.name, "sha256": digest(p), "epoch": 1, "val_accuracy": .6}
    with pytest.raises(m.SourceError, match="selection"):
        m.validate_best(tmp_path, pointer, history, {"fixture": True})


def test_incomplete_epochs_never_satisfy_reference_completion():
    m = module()
    short = {"epoch": 0, "train": {"examples": 7, "batches": 3},
             "val": {"examples": 4, "batches": 2}}
    assert not m.reference_complete([short])
    full = [{"epoch": e, "train": {"examples": 40612, "batches": 565},
             "val": {"examples": 9854, "batches": 137}} for e in range(90)]
    assert m.reference_complete(full)
    full[12]["train"]["examples"] -= 1
    assert not m.reference_complete(full)


def test_resume_rejects_identity_change_and_restores_real_rng_optimizer_continuation(tmp_path):
    m = module()
    model, loader, opt, sch = toy()
    random.seed(6); np.random.seed(6)
    m.run_epoch(model, loader, "cpu", optimizer=opt, scheduler=sch)
    p = tmp_path / "epoch.pt"
    identity = {"recipe": "synthetic_not_reference", "runtime": "cpu"}
    m.save_training_state(p, model, opt, sch, identity, [{"epoch": 0}], "cpu")
    saved_hash = digest(p)
    expected_draw = (random.random(), float(np.random.rand()), torch.rand(2))
    m.run_epoch(model, loader, "cpu", optimizer=opt, scheduler=sch)
    expected = [t.detach().clone() for t in model.parameters()]
    model2, loader2, opt2, sch2 = toy()
    with pytest.raises(m.SourceError, match="identity"):
        m.restore_training_state(p, saved_hash, model2, opt2, sch2, {**identity, "runtime": "changed"}, "cpu")
    history = m.restore_training_state(p, saved_hash, model2, opt2, sch2, identity, "cpu")
    assert history == [{"epoch": 0}]
    got = (random.random(), float(np.random.rand()), torch.rand(2))
    assert got[:2] == expected_draw[:2] and torch.equal(got[2], expected_draw[2])
    m.run_epoch(model2, loader2, "cpu", optimizer=opt2, scheduler=sch2)
    assert all(torch.equal(a, b) for a, b in zip(expected, model2.parameters()))
    assert sch2.state_dict() == sch.state_dict()
