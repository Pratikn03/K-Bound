#!/usr/bin/env python3
"""Measure a complete CIFAR-10-C KGA decision window with a real adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cifar_tent_mps_v2 as runner
import numpy as np
import psutil
import sklearn
import torch
from calibration import exact_rank_radius
from sklearn.ensemble import GradientBoostingRegressor


def synchronize(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, function, *args, **kwargs):
    synchronize(device)
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    synchronize(device)
    return result, (time.perf_counter_ns() - start) / 1e6


@torch.no_grad()
def probabilities(model, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    values = []
    for start in range(0, len(x), batch_size):
        values.append(model(x[start : start + batch_size]).softmax(1))
    return torch.cat(values, dim=0)


def evidence_from_probabilities(
    p0: torch.Tensor,
    pa: torch.Tensor,
    num_classes: int,
    update_norm: float,
) -> np.ndarray:
    def entropy(value):
        return -(value * (value + 1e-9).log()).sum(1)
    e0 = entropy(p0).mean().item()
    ea = entropy(pa).mean().item()
    conf0 = p0.max(1).values.mean().item()
    confa = pa.max(1).values.mean().item()
    mb0 = p0.mean(0)
    mba = pa.mean(0)
    scale = np.log(num_classes)
    pbal0 = (-(mb0 * (mb0 + 1e-9).log()).sum()).item() / scale
    pbala = (-(mba * (mba + 1e-9).log()).sum()).item() / scale
    frac_high = (pa.max(1).values > 0.9).float().mean().item()
    marginal_kl = (mba * ((mba + 1e-9).log() - (mb0 + 1e-9).log())).sum().item()
    return np.asarray([
        e0,
        conf0,
        pbal0,
        ea,
        confa,
        pbala,
        pbal0 - pbala,
        e0 - ea,
        frac_high,
        marginal_kl,
        update_norm,
    ])


def adapt(method: str, model, stream, steps: int, learning_rate: float):
    if method == "tent":
        return runner.tent_adapt(model, stream, steps, learning_rate)
    if method == "eata":
        return runner.eata_adapt(model, stream, steps, learning_rate, 10)
    if method == "sar":
        return runner.sar_adapt(model, stream, steps, learning_rate, 10)
    raise ValueError(method)


def decide(estimate: float, epsilon: float) -> str:
    if estimate - epsilon > 0.0:
        return "ADAPT"
    if estimate + epsilon < 0.0:
        return "FREEZE"
    return "ABSTAIN"


def load_benefit_model(path: Path, alpha: float) -> tuple[GradientBoostingRegressor, float]:
    payload = json.loads(path.read_text())
    records = payload["records"]
    z = np.asarray([row["Z"] for row in records], dtype=float)
    benefit = np.asarray([row["B"] for row in records], dtype=float)
    model = GradientBoostingRegressor(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
    ).fit(z, benefit)
    if all("b_hat" in row for row in records):
        residuals = np.abs(np.asarray([row["b_hat"] for row in records], dtype=float) - benefit)
    else:
        residuals = np.abs(model.predict(z) - benefit)
    return model, exact_rank_radius(residuals, alpha)


def summary(values: list[dict]) -> dict:
    keys = values[0]
    return {
        key: {
            "mean_ms": float(np.mean([row[key] for row in values])),
            "p95_ms": float(np.percentile([row[key] for row in values], 95)),
        }
        for key in keys
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-records", type=Path, required=True)
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--composition", choices=["iid", "imbalanced", "single_class"], default="single_class")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--method", choices=["tent", "eata", "sar"], default="sar")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    device_name = runner.pick_device() if args.device == "auto" else args.device
    device = torch.device(device_name)
    model = runner.get_cifar_model("10", str(args.data_root), device, str(args.checkpoint))
    x_all, y_all = runner.load_cifar_c(str(args.data_root), "10", args.corruption)
    x_severity, y_severity = runner.cifar_c_severity(x_all, y_all, args.severity)
    stream, eval_x, _ = runner.build_stream_and_eval(
        x_severity,
        y_severity,
        x_severity,
        y_severity,
        args.composition,
        args.batch_size,
        10,
        np.random.default_rng(0),
        device,
        eval_pool=args.eval_size,
    )
    benefit_model, epsilon = load_benefit_model(args.calibration_records, args.alpha)
    for _ in range(args.warmup):
        _ = probabilities(model, eval_x[: min(len(eval_x), args.batch_size)], args.batch_size)
        synchronize(device)

    process = psutil.Process()
    rss_before = process.memory_info().rss
    raw = []
    for _ in range(args.repeats):
        total_start = time.perf_counter_ns()
        (candidate, update_norm), proposal_ms = timed(
            device,
            adapt,
            args.method,
            model,
            stream,
            args.steps,
            args.learning_rate,
        )
        p0, frozen_ms = timed(device, probabilities, model, eval_x, args.batch_size)
        pa, candidate_ms = timed(device, probabilities, candidate, eval_x, args.batch_size)
        evidence, evidence_ms = timed(
            device,
            evidence_from_probabilities,
            p0,
            pa,
            10,
            update_norm,
        )
        estimate, controller_ms = timed(
            device,
            benefit_model.predict,
            evidence.reshape(1, -1),
        )
        estimate = float(estimate[0])
        _, decision_ms = timed(device, decide, estimate, epsilon)
        rollback_start = time.perf_counter_ns()
        del candidate
        runner._mps_free()
        synchronize(device)
        rollback_ms = (time.perf_counter_ns() - rollback_start) / 1e6
        total_ms = (time.perf_counter_ns() - total_start) / 1e6
        raw.append({
            "candidate_adaptation_including_copy": proposal_ms,
            "frozen_inference": frozen_ms,
            "candidate_inference": candidate_ms,
            "evidence_extraction": evidence_ms,
            "benefit_estimator": controller_ms,
            "decision_rule": decision_ms,
            "rollback_discard": rollback_ms,
            "full_decision_window": total_ms,
        })
    result = {
        "schema": "kbound_end_to_end_runtime_v1",
        "scope": "real CIFAR-10-C model, real candidate update, cached-log benefit estimator",
        "config": vars(args) | {"data_root": str(args.data_root), "checkpoint": str(args.checkpoint),
                                "calibration_records": str(args.calibration_records),
                                "output": str(args.output)},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "device": str(device),
        },
        "provenance": {
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "calibration_records_sha256": hashlib.sha256(args.calibration_records.read_bytes()).hexdigest(),
        },
        "memory": {
            "rss_before_mb": rss_before / 1e6,
            "rss_after_mb": process.memory_info().rss / 1e6,
        },
        "raw_ms": raw,
        "summary": summary(raw),
        "notes": {
            "candidate_copy": "included once inside candidate_adaptation_including_copy",
            "full_decision_window": "includes proposal, both inference paths, evidence, decision, and rollback",
            "labels": "not used by adaptation, evidence extraction, benefit prediction, or decision",
            "epsilon": epsilon,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
