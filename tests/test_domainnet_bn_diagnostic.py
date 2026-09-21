"""Catch wrong counterfactuals, inference mutation and outcome access order."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def diagnostic():
    class LazyImplementation:
        module = None

        def __getattr__(self, name):
            path = SCRIPTS / "domainnet_bn_diagnostic.py"
            assert path.exists(), "approved saved-state diagnostic is not implemented"
            if self.module is None:
                spec = importlib.util.spec_from_file_location("bn_diagnostic_tested", path)
                self.module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.module)
            return getattr(self.module, name)

    return LazyImplementation()


def model_states():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)).eval()
    source = {k: v.clone() for k, v in model.state_dict().items()}
    adapted = {k: v.clone() for k, v in source.items()}
    adapted["0.weight"].fill_(2.0)
    adapted["1.weight"].fill_(3.0)
    adapted["1.running_mean"].fill_(4.0)
    adapted["1.running_var"].fill_(5.0)
    adapted["1.num_batches_tracked"].fill_(6)
    return model, source, adapted


def test_only_bn_statistics_are_swapped_not_affine_parameters(diagnostic):
    model, source, adapted = model_states()
    mixed = diagnostic.counterfactual_state(model, adapted, source)
    assert torch.equal(mixed["0.weight"], adapted["0.weight"])
    assert torch.equal(mixed["1.weight"], adapted["1.weight"])
    assert torch.equal(mixed["1.running_mean"], source["1.running_mean"])
    assert torch.equal(mixed["1.running_var"], source["1.running_var"])
    assert torch.equal(mixed["1.num_batches_tracked"], source["1.num_batches_tracked"])
    mixed["0.weight"].zero_()
    assert torch.all(adapted["0.weight"] == 2.0)
    reverse = diagnostic.counterfactual_state(model, source, adapted)
    assert torch.equal(reverse["0.weight"], source["0.weight"])
    assert torch.all(reverse["1.running_var"] == 5.0)


@pytest.mark.parametrize("fault", ["missing", "shape", "nan", "negative_var", "dtype"])
def test_invalid_state_is_rejected_before_forward(diagnostic, fault):
    model, source, adapted = model_states()
    if fault == "missing":
        del adapted["1.running_mean"]
    if fault == "shape":
        adapted["0.weight"] = torch.zeros(3, 3)
    if fault == "nan":
        adapted["0.weight"][0, 0] = float("nan")
    if fault == "negative_var":
        adapted["1.running_var"].fill_(-1.0)
    if fault == "dtype":
        adapted["0.weight"] = adapted["0.weight"].double()
    with pytest.raises(ValueError):
        diagnostic.counterfactual_state(model, adapted, source)


def test_four_inference_paths_preserve_bn_rng_and_source_states(diagnostic):
    model, source, adapted = model_states()
    before = {k: v.clone() for k, v in source.items()}
    rng = torch.get_rng_state().clone()
    result = diagnostic.predict_four(model, source, adapted, lambda: iter([torch.ones(4, 2)]), 4)
    assert set(result) == set(diagnostic.ARMS)
    assert all(len(row["predictions"]) == 4 for row in result.values())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], source[k]) for k in source)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(not m.training for m in model.modules())


def test_state_mutating_inference_is_rejected(diagnostic):
    class Mutator(torch.nn.Sequential):
        def forward(self, x):
            with torch.no_grad():
                self[1].running_mean.add_(1)
            return super().forward(x)

    model, source, adapted = model_states()
    bad = Mutator(*list(model.children()))
    with pytest.raises(ValueError, match="state changed"):
        diagnostic.predict_four(bad, source, adapted, lambda: iter([torch.ones(4, 2)]), 4)


def test_exact_subset_rejects_wrong_ids_or_role(diagnostic):
    ids = [f"painting/cat/{i}.jpg" for i in range(512)]
    entries = [{"image_id": name, "episode": 0, "role": "evaluation"} for name in ids]
    assert diagnostic.select_entries({"entries": entries}, ids) == entries
    with pytest.raises(ValueError):
        diagnostic.select_entries({"entries": entries}, ids[:-1])
    entries[10]["role"] = "adaptation"
    with pytest.raises(ValueError):
        diagnostic.select_entries({"entries": entries}, ids)


@pytest.mark.parametrize("changed", [False, True])
def test_score_requires_externally_bound_predictions_before_truth(diagnostic, tmp_path, changed):
    from domainnet_feasibility_contract import file_identity, write_json

    path = tmp_path / "predictions.json"
    packet = {"sample_ids": ["a", "b"], "arms": {a: {"predictions": [0, 1]} for a in diagnostic.ARMS}}
    write_json(path, packet)
    binding = file_identity(path)["sha256"]
    seen = []

    def truth():
        seen.append(True)
        return {"a": 0, "b": 0}

    if changed:
        path.write_text("{}")
        with pytest.raises(ValueError):
            diagnostic.score_bound(path, binding, truth)
        assert not seen
    else:
        score = diagnostic.score_bound(path, binding, truth)
        assert seen and all(v["accuracy"] == 0.5 for v in score.values())


def test_single_and_whole_outcome_perturbations_cannot_change_collected_states(diagnostic, tmp_path):
    from domainnet_feasibility_contract import file_identity, write_json

    model, source, adapted = model_states()

    def full(truth):
        arms = diagnostic.predict_four(model, source, adapted, lambda: iter([torch.ones(4, 2)]), 4)
        packet = {"sample_ids": ["a", "b", "c", "d"], "arms": arms}
        path = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
        write_json(path, packet)
        diagnostic.score_bound(path, file_identity(path)["sha256"], lambda: truth)
        return packet

    before = full(dict.fromkeys("abcd", 0))
    assert full({"a": 1, "b": 0, "c": 0, "d": 0}) == before
    assert full(dict.fromkeys("abcd", 1)) == before


def test_missing_review_refuses_before_collection(diagnostic, tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        diagnostic.authorize(tmp_path / "missing-manifest.json", "0" * 64, tmp_path / "missing-review.json", "0" * 64)


def test_standalone_worker_requires_live_parent_binding(diagnostic, tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(manifest_sha="a" * 64, review_sha="b" * 64)
    with pytest.raises((FileNotFoundError, ValueError)):
        diagnostic.check_supervisor({"output_root": str(tmp_path), "code": {}}, args)


def test_worker_rejects_wrong_parent_and_review_identity(diagnostic, tmp_path):
    import json
    import os
    from types import SimpleNamespace

    args = SimpleNamespace(manifest_sha="a" * 64, review_sha="b" * 64)
    m = {"output_root": str(tmp_path), "code": {}}
    p = tmp_path / "authorization.json"
    data = {
        "manifest_sha256": args.manifest_sha,
        "review_sha256": args.review_sha,
        "code": {},
        "supervisor_pid": os.getppid() + 1,
    }
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        diagnostic.check_supervisor(m, args)
    data["supervisor_pid"] = os.getppid()
    data["review_sha256"] = "c" * 64
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        diagnostic.check_supervisor(m, args)
    data["review_sha256"] = args.review_sha
    p.write_text(json.dumps(data))
    diagnostic.check_supervisor(m, args)


def test_full_worker_persists_all_predictions_before_truth_and_preserves_outcome_exclusion(
    diagnostic, tmp_path, monkeypatch
):
    """Actual worker + image broker + inference + serialization; tiny CPU model."""
    import hashlib
    import io
    import json
    import os
    from types import SimpleNamespace

    import domainnet_feasibility_runner as runner
    import domainnet_feasibility_score as scorer
    import domainnet_reference_adapter as reference
    import domainnet_reference_source as loader
    from domainnet_feasibility_contract import file_identity, write_json
    from domainnet_feasibility_images import pixel_digest
    from PIL import Image
    from torchvision.transforms import ToTensor

    native_reader = runner.read_verified_checkpoint

    class Tiny(torch.nn.Sequential):
        def forward(self, x):
            assert not torch.is_grad_enabled()
            return super().forward(x.mean((2, 3))[:, :2])

    _, source, adapted = model_states()
    source["0.weight"] = torch.eye(2)
    source["0.bias"].zero_()
    adapted = {k: v.clone() for k, v in source.items()}
    adapted["0.bias"][0] = 1.0
    adapted["1.running_mean"][1] = 2.0

    def load(_):
        with torch.random.fork_rng():
            model = Tiny(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)).eval()
        model.load_state_dict(source)
        return model, {"synthetic": True}

    buf = io.BytesIO()
    torch.save({"classifier": adapted}, buf)
    monkeypatch.setattr(loader, "load_clipart2020", load)
    monkeypatch.setattr(loader, "read_verified_checkpoint", lambda *a: buf.getvalue())
    monkeypatch.setattr(reference, "load_reference", lambda *a: SimpleNamespace(get_augmentation=lambda *_: ToTensor()))
    images = tmp_path / "images"
    images.mkdir()
    with Image.new("RGB", (2, 2), (10, 20, 30)) as image:
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        pixels = pixel_digest(image)
    data = encoded.getvalue()
    (images / "image.png").write_bytes(data)
    ids = [f"painting/cat/{i}.jpg" for i in range(512)]
    entries = [
        {
            "image_id": name,
            "episode": 0,
            "role": "evaluation",
            "stage_name": "image.png",
            "encoded_bytes": len(data),
            "encoded_sha256": hashlib.sha256(data).hexdigest(),
            "decoded_rgb_sha256": pixels,
            "width": 2,
            "height": 2,
        }
        for name in ids
    ]
    packets = []
    for attempt in range(3):
        output = tmp_path / str(attempt)
        output.mkdir()
        truth = dict.fromkeys(ids, 0)
        if attempt == 1:
            truth[ids[0]] = 1
        if attempt == 2:
            truth = dict.fromkeys(ids, 1)
        labels = output / "truth-fixture.json"
        labels.write_text(json.dumps(truth))
        args = SimpleNamespace(manifest="unused", manifest_sha="a" * 64, review="unused", review_sha="b" * 64)
        m = {
            "output_root": str(output),
            "original_root": str(tmp_path),
            "sample_ids": ids,
            "entries_digest": "synthetic",
            "code": {},
        }
        base = {
            "checkpoint_path": "synthetic",
            "reference_root": "synthetic",
            "source_list": {"path": str(labels), "bytes": labels.stat().st_size},
        }
        write_json(
            output / "authorization.json",
            {
                "supervisor_pid": os.getppid(),
                "code": {},
                "manifest_sha256": args.manifest_sha,
                "review_sha256": args.review_sha,
            },
        )
        worker = diagnostic.worker
        # Replace external authentication and heavy model loaders only; keep the
        # worker, parent check, file broker, swaps, inference and scorer real.
        monkeypatch.setattr(
            diagnostic.module,
            "authorize",
            lambda *a, m=m, base=base: (
                m,
                base,
                {"root": str(images)},
                {"frozen": [1] * 512, "candidate_direct": [0] * 512},
                entries,
                {"episode-0/inference.pt": {"sha256": "fixture", "bytes": len(buf.getvalue())}},
            ),
        )
        old_read = diagnostic.module.read_bytes

        def guarded(path, *a, labels=labels, output=output, old_read=old_read, **kw):
            if Path(path) == labels:
                assert (output / "predictions-binding.json").is_file()
                binding = json.loads((output / "predictions-binding.json").read_text())
                assert file_identity(output / "predictions.json") == binding
                assert set(json.loads((output / "predictions.json").read_text())["arms"]) == set(diagnostic.ARMS)
            return old_read(path, *a, **kw)

        monkeypatch.setattr(diagnostic.module, "read_bytes", guarded)
        monkeypatch.setattr(scorer, "parse_authenticated_truth", lambda data: json.loads(data))
        worker(args)
        packets.append(json.loads((output / "predictions.json").read_text()))
        monkeypatch.setattr(diagnostic.module, "read_bytes", old_read)
    assert packets[0] == packets[1] == packets[2]
    assert runner.read_verified_checkpoint is native_reader
