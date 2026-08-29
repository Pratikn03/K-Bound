import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from kbound_edge.profiling import profile_runtime  # noqa: E402


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        # x is (B, T, C, H, W) or (T, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            out = self.conv(x)
            out = out.mean(dim=[2, 3])
            out = self.fc(out)
            return out.view(B, T, -1).mean(dim=1)
        else:
            T, C, H, W = x.shape
            out = self.conv(x)
            out = out.mean(dim=[2, 3])
            out = self.fc(out)
            return out.mean(dim=0, keepdim=True)


class MockAdapterResult:
    def __init__(self, model, upd_norm=0.1):
        self.model = model
        self.upd_norm = upd_norm


class MockAdapter:
    def __init__(self, model):
        self.model = model

    def adapt(self, x):
        return MockAdapterResult(self.model)


class MockEstimator:
    def predict_one(self, z):
        return 0.05


@pytest.fixture
def profile_setup():
    f0 = MockModel()
    adapter = MockAdapter(f0)
    estimator = MockEstimator()
    eps = 0.01

    # 6 mock windows (each has 32 frames of size 224x224x3)
    windows = []
    for _ in range(6):
        frames = np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)
        windows.append(frames)

    return f0, adapter, estimator, eps, windows


def test_runtime_profile_contains_all_stages(profile_setup):
    f0, adapter, estimator, eps, windows = profile_setup

    # warmup = 2 to make sure we keep at least some windows
    profile = profile_runtime(
        f0=f0,
        adapter=adapter,
        estimator=estimator,
        eps=eps,
        windows=windows,
        image_size=224,
        device="cpu",
        warmup=2,
    )

    assert set(profile.keys()) >= {
        "capture_preprocess",
        "frozen_inference",
        "tent_update",
        "candidate_inference",
        "evidence",
        "gate",
        "end_to_end",
        "metadata",
    }

    # Check that each stage dictionary contains the summary keys
    summary_keys = {"mean_ms", "p50_ms", "p95_ms", "max_ms"}
    for stage in [
        "capture_preprocess",
        "frozen_inference",
        "tent_update",
        "candidate_inference",
        "evidence",
        "gate",
        "end_to_end",
    ]:
        assert set(profile[stage].keys()) == summary_keys

    # Check metadata fields
    metadata = profile["metadata"]
    assert "hardware_platform" in metadata
    assert "os_system" in metadata
    assert "pytorch_version" in metadata
    assert "opencv_version" in metadata
    assert "device_backend" in metadata
    assert "thread_count" in metadata
    assert "rss_mem_delta_mb" in metadata

    # Validate end-to-end vs individual stages
    assert profile["end_to_end"]["mean_ms"] >= profile["gate"]["mean_ms"]
