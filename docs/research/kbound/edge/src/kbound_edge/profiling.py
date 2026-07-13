"""kbound_edge.profiling -- Stage-by-stage runtime and memory instrumentation.

Times capture/preprocess, inference, tent adaptation, evidence extraction, and
the KGA decision gate. Discards warm-up windows, synchronises GPU backends,
and queries system resource statistics.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import psutil
import torch
import cv2

from kbound_edge.model import predict_proba
from kbound_edge.evidence import edge_evidence_vector
from kbound_edge.policy import kga_decide
from kbound_edge.replay import _to_tensor


def sync_device(device: Any) -> None:
    """Synchronise the CUDA or MPS hardware device to ensure accurate latency measurement."""
    device_str = str(device).lower()
    if "cuda" in device_str:
        torch.cuda.synchronize()
    elif "mps" in device_str:
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass


def get_process_memory_mb() -> float:
    """Return the current resident set size (RSS) memory of the process in MB."""
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024 * 1024))


def profile_runtime(
    f0: Any,
    adapter: Any,
    estimator: Any,
    eps: float,
    windows: Sequence[Any],
    image_size: int = 224,
    device: Any = "cpu",
    warmup: int = 5,
) -> Dict[str, Dict[str, float] | str | int]:
    """Run stage-by-stage timed inference and decision-making on windows.

    Parameters
    ----------
    f0, adapter, estimator, eps
        decision-making components
    windows : sequence of windows
    image_size : int
    device : PyTorch device
    warmup : int
        number of initial windows to discard from statistics
    """
    stages = ["capture_preprocess", "frozen_inference", "tent_update", "candidate_inference", "evidence", "gate", "end_to_end"]
    timings: Dict[str, List[float]] = {s: [] for s in stages}

    mem_before = get_process_memory_mb()

    for idx, window in enumerate(windows):
        # 1. Capture & Preprocess
        sync_device(device)
        t_start = time.perf_counter()
        x = _to_tensor(window, image_size)
        sync_device(device)
        t_preprocess = (time.perf_counter() - t_start) * 1000.0

        # 2. Frozen Inference
        sync_device(device)
        t_inf0 = time.perf_counter()
        p0 = predict_proba(f0, x)
        sync_device(device)
        t_frozen = (time.perf_counter() - t_inf0) * 1000.0

        # 3. Tent Update
        sync_device(device)
        t_adapt = time.perf_counter()
        res = adapter.adapt(x)
        sync_device(device)
        t_tent = (time.perf_counter() - t_adapt) * 1000.0

        # 4. Candidate Inference
        sync_device(device)
        t_infa = time.perf_counter()
        pa = predict_proba(res.model, x)
        sync_device(device)
        t_candidate = (time.perf_counter() - t_infa) * 1000.0

        # 5. Evidence
        sync_device(device)
        t_ev = time.perf_counter()
        z = edge_evidence_vector(p0, pa, res.upd_norm)
        sync_device(device)
        t_evidence = (time.perf_counter() - t_ev) * 1000.0

        # 6. Gate
        sync_device(device)
        t_gt = time.perf_counter()
        bhat = estimator.predict_one(z)
        decision = kga_decide(bhat, eps)
        sync_device(device)
        t_gate = (time.perf_counter() - t_gt) * 1000.0

        t_end_to_end = t_preprocess + t_frozen + t_tent + t_candidate + t_evidence + t_gate

        if idx >= warmup:
            timings["capture_preprocess"].append(t_preprocess)
            timings["frozen_inference"].append(t_frozen)
            timings["tent_update"].append(t_tent)
            timings["candidate_inference"].append(t_candidate)
            timings["evidence"].append(t_evidence)
            timings["gate"].append(t_gate)
            timings["end_to_end"].append(t_end_to_end)

    mem_after = get_process_memory_mb()

    # Compute summary stats for each stage
    profile_summary: Dict[str, Any] = {}
    for stage in stages:
        arr = np.asarray(timings[stage])
        if len(arr) > 0:
            profile_summary[stage] = {
                "mean_ms": float(np.mean(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "max_ms": float(np.max(arr)),
            }
        else:
            profile_summary[stage] = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

    # Include hardware/system metadata
    profile_summary["metadata"] = {
        "hardware_platform": platform.processor() or platform.machine(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "pytorch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "device_backend": str(device),
        "thread_count": torch.get_num_threads(),
        "rss_mem_before_mb": mem_before,
        "rss_mem_after_mb": mem_after,
        "rss_mem_delta_mb": mem_after - mem_before,
        "power_mode": "battery" if getattr(psutil, "sensors_battery", None) and psutil.sensors_battery() and not psutil.sensors_battery().power_plugged else "AC_power",
    }

    return profile_summary
