"""Full collector path with synthetic models/images, never production substitutes.

The real wrapper, role streaming, 15-epoch orchestration, state hashing,
serialization/reload, prediction validation and arithmetic run here. Native
optimizer arithmetic and ZIP authentication have separate non-mocked tests.
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts"))


def test_full_collector_single_and_episode_outcome_exclusion(tmp_path, monkeypatch):
    import domainnet_feasibility_runner as runner
    import domainnet_reference_adapter as adapter
    import domainnet_reference_source as source
    from domainnet_feasibility_contract import RELEASE_SHA256
    from domainnet_feasibility_score import score_episode

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            with torch.random.fork_rng():
                torch.manual_seed(24)
                self.fc = nn.Linear(3, 126)

        def forward(self, x, return_feats=False):
            feats = x.mean((2, 3))
            logits = self.fc(feats)
            return (feats, logits) if return_feats else logits

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.src_model = Tiny()
            self.queue_ptr = 0

    class View:
        active_adaptation = None

        def __init__(self, inventory, episode, role):
            assert episode == 0
            self.ids = manifest["episodes"][0][role + "_ids"]
            self.role = role
            self.closed = False
            if role == "adaptation":
                View.active_adaptation = self

        @contextmanager
        def image(self, name):
            if self.role == "evaluation":
                assert View.active_adaptation.closed, "adaptation cache overlaps evaluation cache"
            if name not in self.ids:
                raise ValueError("cross-role access")
            with Image.new("RGB", (2, 2), (10 if self.role == "adaptation" else 200, 20, 30)) as im:
                yield im

        def close(self):
            self.closed = True

    class Session:
        def __init__(self):
            self.args = SimpleNamespace()
            self.model = Wrapper()
            self.completed_steps = 0
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            self.banks = None
            self.provenance = {}
            self.ref = SimpleNamespace(
                get_augmentation=lambda name: ToTensor(),
                get_augmentation_versions=lambda args: lambda im: [ToTensor()(im) for _ in range(3)],
                refine_predictions=lambda f, p, b, a: (p.argmax(1), None, None),
            )

        def initialize_bank(self, batches):
            data = torch.cat(list(batches))
            assert data.shape[0] == 2048
            assert bool((data[:, 0] < 0.1).all()), "evaluation pixels entered initialization"
            self.banks = {"features": data.mean((2, 3))}

        def train_epoch(self, batches, epoch):
            indices = []
            for views, idx in batches:
                assert bool((views[0][:, 0] < 0.1).all()), "evaluation pixels entered update"
                indices.extend(idx.tolist())
            assert sorted(indices) == list(range(2048))
            with torch.no_grad():
                self.model.src_model.fc.bias[0].add_(0.1)
            self.completed_steps += 512
            self.model.queue_ptr = self.completed_steps * 4 % 16384
            return {"count": 512, "sum": 1024.0, "mean": 2.0, "last": 3.0}

        def predict(self, batches, refine=False):
            assert self.completed_steps == 7680, "evaluation before final candidate"
            with torch.no_grad():
                return torch.cat([self.model.src_model(x).argmax(1) for x in batches])

    a = [f"painting/a/adapt{i}.jpg" for i in range(2048)]
    v = [f"painting/a/eval{i}.jpg" for i in range(4096)]
    manifest = {
        "episodes": [{"stream_seed": 2020, "adaptation_ids": a, "evaluation_ids": v}],
        "runtime": {},
        "code": {},
        "reference_root": "synthetic-only",
        "checkpoint_path": "synthetic-only",
        "resolved_args": {},
        "source_list": {"path": str(tmp_path / "FORBIDDEN_LABELS")},
    }
    monkeypatch.setattr(runner, "validate_inventory", lambda *a: None)
    monkeypatch.setattr(runner, "validate_runtime", lambda *a: None)
    monkeypatch.setattr(runner, "implementation", lambda: {})
    monkeypatch.setattr(runner, "ImageView", View)
    monkeypatch.setattr(adapter, "build_session", lambda *a: Session())
    monkeypatch.setattr(source, "build_reference_classifier", Tiny)
    monkeypatch.setattr(source, "load_clipart2020", lambda path: (Tiny(), {"release_sha256": RELEASE_SHA256}))
    # An actual file-open trap, not just a hypothetical unused truth argument.
    import os

    original_open = os.open

    def guarded(path, *a, **kw):
        if "FORBIDDEN_LABELS" in str(path):
            raise AssertionError("collector read outcome authority")
        return original_open(path, *a, **kw)

    monkeypatch.setattr(os, "open", guarded)
    packets, scores, loss_summaries = [], [], []
    for attempt in range(3):
        dest = tmp_path / str(attempt)
        dest.mkdir()
        truth = dict.fromkeys(v, 0)
        if attempt == 1:
            truth[v[0]] = 1
        if attempt == 2:
            truth = dict.fromkeys(v, 1)
        packet = runner.collect_episode(manifest, {}, 0, dest)
        packets.append(packet)
        scores.append(score_episode(packet, truth))
        import json

        loss_summaries.append([json.loads((dest / f"epoch-{j:02}.json").read_text())["loss"] for j in range(15)])
    keys = [
        "sample_ids",
        "adaptation_ids",
        "candidate_state_sha256",
        "inference_artifact",
        "frozen",
        "candidate_direct",
        "candidate_refined",
        "source_receipt",
        "source_provenance",
    ]
    assert all({k: p[k] for k in keys} == {k: packets[0][k] for k in keys} for p in packets)
    assert loss_summaries[0] == loss_summaries[1] == loss_summaries[2]
    assert scores[0]["correctness"] != scores[1]["correctness"] != scores[2]["correctness"]
