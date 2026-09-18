#!/usr/bin/env python3
"""Execute the predeclared five-arm matched SAR control study on ImageNet-C.

Strictly follows the experiment specification in TASK4_MATCHED_SAR_SPEC.json:
- ARM 1: BN-statistics only (test BN statistics enabled, zero gradient steps)
- ARM 2: Reference SAR (lr=2.5e-4, layers 1-3 affine, layer 4 frozen)
- ARM 3: Rate only (lr=4.0e-3, layers 1-3 affine, layer 4 frozen)
- ARM 4: Mask only (lr=2.5e-4, layers 1-4 affine adapted)
- ARM 5: Combined aggressive (lr=4.0e-3, layers 1-4 affine adapted)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[4]


def _in_layer4(name: str) -> bool:
    """True if parameter or module is inside ResNet layer4 or top blocks."""
    parts = name.split(".")
    for part in parts:
        if part.startswith("layer4") or part == "layer4":
            return True
        if part in ("fc", "classifier", "heads", "head"):
            return True
    return False


def _bn_affine_params(model: nn.Module, freeze_layer4: bool = False) -> list[nn.Parameter]:
    """Collect trainable BN affine parameters matching canonical cifar_tent_mps_v2."""
    mod_to_name = {id(mod): name for name, mod in model.named_modules()}
    ps = []
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.track_running_stats = False
            mod.running_mean = None
            mod.running_var = None
            if freeze_layer4 and _in_layer4(mod_to_name.get(id(mod), "")):
                if mod.weight is not None:
                    mod.weight.requires_grad_(False)
                if mod.bias is not None:
                    mod.bias.requires_grad_(False)
                continue
            if mod.weight is not None:
                mod.weight.requires_grad_(True)
                ps.append(mod.weight)
            if mod.bias is not None:
                mod.bias.requires_grad_(True)
                ps.append(mod.bias)
    return ps


def _clone_for_tta(base: nn.Module, freeze_layer4: bool = False) -> tuple[nn.Module, list[nn.Parameter], list[torch.Tensor]]:
    m = copy.deepcopy(base)
    m.train()
    for p in m.parameters():
        p.requires_grad_(False)
    ps = _bn_affine_params(m, freeze_layer4=freeze_layer4)
    init = [p.detach().clone() for p in ps]
    return m, ps, init


def _upd_norm(ps: list[nn.Parameter], init: list[torch.Tensor]) -> float:
    if not ps or not init:
        return 0.0
    return float(sum(((p.detach() - q).norm() ** 2).item() for p, q in zip(ps, init)) ** 0.5)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * (p + 1e-9).log()).sum(1)


def sar_adapt(
    base: nn.Module,
    stream: list[torch.Tensor],
    steps: int,
    lr: float,
    num_classes: int = 1000,
    rho: float = 0.05,
    margin_e0: float | None = None,
    reset_constant_em: float = 0.2,
    freeze_layer4: bool = False,
) -> tuple[nn.Module, float]:
    """Faithful SAR matching canonical implementation."""
    if margin_e0 is None:
        margin_e0 = 0.4 * math.log(num_classes)
    m, ps, init = _clone_for_tta(base, freeze_layer4=freeze_layer4)
    if not ps:
        return m, 0.0
    opt = torch.optim.SGD(ps, lr=lr, momentum=0.9)
    model_state = copy.deepcopy(m.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    ema = None

    for _ in range(steps):
        for xb in stream:
            xb = xb.contiguous()
            # 1. First forward + reliable sample selection
            opt.zero_grad()
            out1 = m(xb)
            ent = _entropy(out1.softmax(1))
            keep1 = ent < margin_e0
            if keep1.sum() == 0:
                continue
            ent[keep1].mean().backward()

            # 2. SAM first step: climb to w + e(w)
            with torch.no_grad():
                gnorm = sum((p.grad.detach() ** 2).sum() for p in ps if p.grad is not None) ** 0.5
                scale = rho / (gnorm + 1e-12)
                old_p = [p.data.clone() for p in ps]
                for p in ps:
                    if p.grad is not None:
                        p.add_(p.grad * scale)

            # 3. Second forward at perturbed weights
            opt.zero_grad()
            out2 = m(xb)
            ent2 = _entropy(out2.softmax(1))[keep1]
            keep2 = ent2 < margin_e0
            loss2 = ent2[keep2].mean() if keep2.any() else ent2.mean()

            # 4. Model recovery check
            loss2_val = loss2.item()
            if ema is None:
                ema = loss2_val
            else:
                ema = 0.9 * ema + 0.1 * loss2_val

            if ema < reset_constant_em:
                m.load_state_dict(model_state)
                opt.load_state_dict(opt_state)
                ema = None
                continue

            loss2.backward()
            with torch.no_grad():
                for p, p_old in zip(ps, old_p):
                    p.data.copy_(p_old.data)
            opt.step()

    return m, _upd_norm(ps, init)


def bn_adapt(base: nn.Module, stream: list[torch.Tensor]) -> tuple[nn.Module, float]:
    """BN-statistics only control: zero gradient steps, test BN stats active."""
    m, ps, init = _clone_for_tta(base, freeze_layer4=True)
    m.train()
    with torch.no_grad():
        for xb in stream:
            m(xb.contiguous())
    return m, 0.0


EVIDENCE_NAMES = [
    "pre_entropy",
    "pre_conf",
    "pre_pbal",
    "post_entropy",
    "post_conf",
    "post_pbal",
    "pbal_drop",
    "entropy_drop",
    "frac_highconf",
    "marginal_KL",
    "update_norm",
]


def compute_pre_evidence(
    model_frozen: nn.Module,
    x: torch.Tensor,
    num_classes: int = 1000,
    chunk_size: int = 256,
) -> tuple[float, float, torch.Tensor, float]:
    """Compute pre-adaptation evidence quantities on x."""
    model_frozen.eval()
    p0_list = []
    with torch.no_grad():
        for i in range(0, len(x), chunk_size):
            xb = x[i:i + chunk_size]
            p0_list.append(model_frozen(xb).softmax(1))
    p0 = torch.cat(p0_list)
    e0 = _entropy(p0).mean().item()
    conf0 = p0.max(1).values.mean().item()
    mb0 = p0.mean(0)
    pbal0 = (-(mb0 * (mb0 + 1e-9).log()).sum()).item() / math.log(num_classes)
    return e0, conf0, mb0, pbal0


def compute_post_evidence(
    model_adapted: nn.Module,
    x: torch.Tensor,
    pre_stats: tuple[float, float, torch.Tensor, float],
    upd_norm: float,
    num_classes: int = 1000,
    chunk_size: int = 256,
) -> list[float]:
    """Compute 11-feature evidence vector using precomputed frozen stats."""
    e0, conf0, mb0, pbal0 = pre_stats
    model_adapted.eval()
    pa_list = []
    with torch.no_grad():
        for i in range(0, len(x), chunk_size):
            xb = x[i:i + chunk_size]
            pa_list.append(model_adapted(xb).softmax(1))
    pa = torch.cat(pa_list)
    ea = _entropy(pa).mean().item()
    confa = pa.max(1).values.mean().item()
    mba = pa.mean(0)
    pbala = (-(mba * (mba + 1e-9).log()).sum()).item() / math.log(num_classes)
    frac_hi = (pa.max(1).values > 0.9).float().mean().item()
    klm = (mba * ((mba + 1e-9).log() - (mb0 + 1e-9).log())).sum().item()
    return [e0, conf0, pbal0, ea, confa, pbala, pbal0 - pbala, e0 - ea, frac_hi, klm, upd_norm]


def acc_on(model: nn.Module, x: torch.Tensor, y: torch.Tensor, train_mode: bool = True, chunk_size: int = 256) -> float:
    """Evaluate accuracy on (x, y) in chunk_size-sized batches."""
    model.train() if train_mode else model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(x), chunk_size):
            xb = x[i:i + chunk_size]
            preds.append(model(xb).argmax(1).cpu())
    return (torch.cat(preds) == y.cpu()).float().mean().item()


def compute_state_hash(model: nn.Module) -> str:
    """SHA-256 fingerprint of model state dict."""
    hasher = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        hasher.update(k.encode("utf-8"))
        hasher.update(v.cpu().numpy().tobytes())
    return hasher.hexdigest()


def load_imagenetc_samples(folder: Path, max_images: int, seed: int) -> tuple[list[tuple[str, int]], str]:
    """Sample balanced image list matching canonical ImageNet-C loader."""
    classes = sorted(d.name for d in folder.iterdir() if not d.name.startswith(".") and d.is_dir())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)
    per_class = max(1, max_images // max(1, len(classes)) + 1)
    order = list(classes)
    rng.shuffle(order)
    samples: list[tuple[str, int]] = []
    for c in order:
        if len(samples) >= max_images * 2:
            break
        cdir = folder / c
        if not cdir.is_dir():
            continue
        files = [f for f in cdir.iterdir() if not f.name.startswith(".") and f.suffix.lower() in (".jpeg", ".jpg", ".png")]
        if not files:
            continue
        k = min(len(files), per_class)
        chosen = rng.choice(len(files), size=k, replace=False)
        for idx in chosen:
            samples.append((str(files[int(idx)]), cls_to_idx[c]))
    if len(samples) > max_images:
        sub = rng.choice(len(samples), size=max_images, replace=False)
        samples = [samples[int(i)] for i in sub]

    manifest_hash = hashlib.sha256(json.dumps(samples).encode("utf-8")).hexdigest()
    return samples, manifest_hash


def build_composed_stream(X: torch.Tensor, Y: torch.Tensor, comp: str, bs: int = 16, n_total: int = 128) -> list[torch.Tensor]:
    """Create stream batches matching canonical compositions (iid, imbalanced, single_class)."""
    N = len(X)
    if comp == "imbalanced":
        classes = torch.unique(Y)
        major = classes[torch.randint(len(classes), (1,)).item()]
        maj = (Y == major).nonzero(as_tuple=True)[0]
        oth = (Y != major).nonzero(as_tuple=True)[0]
        if len(maj) == 0:
            maj = oth
        nM = int(n_total * 0.85)
        sel_maj = maj[torch.randint(len(maj), (nM,))]
        nO = n_total - nM
        sel_oth = oth[torch.randint(len(oth), (nO,))] if len(oth) > 0 else maj[torch.randint(len(maj), (nO,))]
        idx = torch.cat([sel_maj, sel_oth])
        idx = idx[torch.randperm(len(idx))]
    elif comp == "single_class":
        classes = torch.unique(Y)
        major = classes[torch.randint(len(classes), (1,)).item()]
        pool = (Y == major).nonzero(as_tuple=True)[0]
        if len(pool) == 0:
            pool = torch.arange(N)
        idx = pool[torch.randint(len(pool), (n_total,))]
    else:  # iid
        idx = torch.randperm(N)[:n_total]
    return [X[idx[i:i + bs]] for i in range(0, len(idx), bs)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Path to TASK4_MATCHED_SAR_SPEC.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume completed cells")
    parser.add_argument("--pilot", action="store_true", help="Run pilot mode (2 cells)")
    parser.add_argument("--max-cells", type=int, default=0, help="Stop after N condition cells")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== K-Bound Task 4 Matched SAR Runner ===")
    print(f"Device: {dev}")
    print(f"Spec: {args.spec}")
    print(f"Output Dir: {args.output_dir}")

    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Source Model
    weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
    tf = weights.transforms()
    base_model = tv.models.resnet50(weights=weights).to(dev).eval()
    base_state_hash = compute_state_hash(base_model)
    print(f"Base ResNet-50 state hash: {base_state_hash[:16]}...")

    # Arms configuration
    arms = spec["arms"]
    arm_files = {
        arm_key: raw_dir / f"{arm_cfg['name']}.jsonl"
        for arm_key, arm_cfg in arms.items()
    }

    # Track completed cells per arm for resume
    done_cells_per_arm: dict[str, set[str]] = {arm_key: set() for arm_key in arms}
    if args.resume:
        for arm_key, fpath in arm_files.items():
            if fpath.exists():
                for line in fpath.read_text().splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        done_cells_per_arm[arm_key].add(rec["condition_id"])
                print(f"Resumed {len(done_cells_per_arm[arm_key])} cells for {arm_key}")

    # Grid definition
    dp = spec["data_panel"]
    ic_root = Path(dp["root"])
    corruptions = dp["corruptions"]
    severities = dp["severities"]
    compositions = dp["compositions"]
    eval_sample_size = dp["eval_sample_size"]
    seed = dp["seed"]

    total_conditions = len(corruptions) * len(severities) * len(compositions)
    print(f"Total panel conditions: {total_conditions} ({len(corruptions)} corr x {len(severities)} sev x {len(compositions)} comp)")
    if args.pilot:
        print("PILOT MODE: Limiting to first 2 conditions.")
        args.max_cells = 2

    cond_counter = 0
    t_global_start = time.time()

    for corr in corruptions:
        for sev in severities:
            # Check if all compositions for this (corr, sev) are done across all arms
            need_load = False
            for comp in compositions:
                cid = f"{corr}|s{sev}|small|aggressive|{comp}"
                if any(cid not in done_cells_per_arm[ak] for ak in arms):
                    need_load = True
                    break
            if not need_load:
                continue

            # Load images for (corr, sev)
            corr_folder = ic_root / corr / str(sev)
            if not corr_folder.exists():
                print(f"Missing corruption folder: {corr_folder}")
                continue

            t_load_0 = time.time()
            sample_list, sample_hash = load_imagenetc_samples(corr_folder, eval_sample_size, seed)
            
            # Load images into preprocessed tensor
            xs, ys = [], []
            for img_path, label in sample_list:
                try:
                    img = Image.open(img_path).convert("RGB")
                    xs.append(tf(img))
                    ys.append(label)
                except Exception as e:
                    pass
            X_all = torch.stack(xs).to(dev)
            Y_all = torch.tensor(ys, dtype=torch.long, device=dev)
            dt_load = time.time() - t_load_0
            print(f"Loaded {len(X_all)} images for {corr} sev{sev} in {dt_load:.1f}s (Sample Hash: {sample_hash[:12]}...)")

            for comp in compositions:
                condition_id = f"{corr}|s{sev}|small|aggressive|{comp}"
                cond_counter += 1

                # Check if all arms done for this condition
                if all(condition_id in done_cells_per_arm[ak] for ak in arms):
                    continue

                if args.max_cells and cond_counter > args.max_cells:
                    print(f"Reached max-cells cap ({args.max_cells}); stopping.")
                    return

                t_cell_start = time.time()
                print(f"\n[{cond_counter:02d}/{total_conditions:02d}] Starting Condition: {condition_id}", flush=True)
                # Create adaptation stream
                torch.manual_seed(seed + cond_counter)
                stream = build_composed_stream(X_all, Y_all, comp, bs=dp["stream_batch_size"], n_total=dp["stream_length"])

                # Measure frozen accuracy once for this condition
                acc_frozen = acc_on(base_model, X_all, Y_all, train_mode=False, chunk_size=256)
                pre_stats = compute_pre_evidence(base_model, X_all[:1024], num_classes=1000, chunk_size=256)

                print(f"   Frozen Acc = {acc_frozen:.4f}", flush=True)

                # Evaluate all 5 arms
                for arm_key, arm_cfg in arms.items():
                    if condition_id in done_cells_per_arm[arm_key]:
                        continue

                    arm_id = arm_cfg["arm_id"]
                    arm_name = arm_cfg["name"]
                    lr = arm_cfg["learning_rate"]
                    freeze_l4 = arm_cfg["freeze_layer4"]
                    is_bn_only = (arm_id == 1)

                    t_arm_0 = time.time()
                    try:
                        if is_bn_only:
                            adapted_m, upd_norm = bn_adapt(base_model, stream)
                        else:
                            adapted_m, upd_norm = sar_adapt(
                                base_model,
                                stream,
                                steps=arm_cfg["steps"],
                                lr=lr,
                                num_classes=1000,
                                freeze_layer4=freeze_l4,
                            )
                        acc_adapted = acc_on(adapted_m, X_all, Y_all, train_mode=True, chunk_size=256)
                        delta = acc_adapted - acc_frozen
                        adapted_state_hash = compute_state_hash(adapted_m)
                        Z = compute_post_evidence(
                            adapted_m,
                            X_all[:1024],
                            pre_stats,
                            upd_norm=upd_norm,
                            num_classes=1000,
                            chunk_size=256,
                        )
                        failure_status = None
                        dt_arm = time.time() - t_arm_0
                        oracle_action = "ADAPT" if delta > 0.0 else "FREEZE"
                        oracle_acc = max(acc_adapted, acc_frozen)
                        always_adapt_regret = max(0.0, acc_frozen - acc_adapted)
                        always_freeze_regret = max(0.0, acc_adapted - acc_frozen)

                        record = {
                            "arm_id": arm_id,
                            "arm_name": arm_name,
                            "seed": seed,
                            "condition_id": condition_id,
                            "corruption": corr,
                            "severity": sev,
                            "composition": comp,
                            "batch_regime": dp["batch_regime"],
                            "sample_hash": sample_hash,
                            "frozen_accuracy": float(acc_frozen),
                            "candidate_accuracy": float(acc_adapted),
                            "candidate_benefit": float(delta),
                            "oracle_action": oracle_action,
                            "oracle_accuracy": float(oracle_acc),
                            "always_adapt_regret": float(always_adapt_regret),
                            "always_freeze_regret": float(always_freeze_regret),
                            "candidate_state_hash_before": base_state_hash,
                            "candidate_state_hash_after": adapted_state_hash,
                            "parameter_update_norm": float(upd_norm),
                            "bn_statistics_active": True,
                            "gradient_update_active": arm_cfg["gradient_updates"],
                            "learning_rate": lr,
                            "freeze_layer4": freeze_l4,
                            "runtime_sec": float(dt_arm),
                            "failure_status": failure_status,
                            "Z": [float(v) for v in Z],
                            "Z_names": EVIDENCE_NAMES,
                        }
                        # Validate no NaN or Inf
                        for rk, rv in record.items():
                            if isinstance(rv, float) and (math.isnan(rv) or math.isinf(rv)):
                                raise ValueError(f"Invalid float {rk}={rv} in record")

                        # Append to arm JSONL file
                        with open(arm_files[arm_key], "a") as af:
                            af.write(json.dumps(record) + "\n")
                            af.flush()
                        done_cells_per_arm[arm_key].add(condition_id)

                        print(f"   Arm {arm_id} ({arm_name:18s}): adapted={acc_adapted:.4f}, Delta={delta:+.4f}, upd_norm={upd_norm:.2f} ({dt_arm:.1f}s)", flush=True)

                        del adapted_m
                        if dev == "mps":
                            torch.mps.empty_cache()

                    except Exception as exc:
                        print(f"   Arm {arm_id} FAILED: {exc}", flush=True)
                        err_record = {
                            "arm_id": arm_id,
                            "arm_name": arm_name,
                            "seed": seed,
                            "condition_id": condition_id,
                            "corruption": corr,
                            "severity": sev,
                            "composition": comp,
                            "batch_regime": dp["batch_regime"],
                            "sample_hash": sample_hash,
                            "frozen_accuracy": float(acc_frozen),
                            "candidate_accuracy": None,
                            "candidate_benefit": None,
                            "oracle_action": None,
                            "oracle_accuracy": None,
                            "always_adapt_regret": None,
                            "always_freeze_regret": None,
                            "candidate_state_hash_before": base_state_hash,
                            "candidate_state_hash_after": None,
                            "parameter_update_norm": None,
                            "bn_statistics_active": True,
                            "gradient_update_active": arm_cfg["gradient_updates"],
                            "learning_rate": lr,
                            "freeze_layer4": freeze_l4,
                            "runtime_sec": float(time.time() - t_arm_0),
                            "failure_status": str(exc),
                            "Z": None,
                            "Z_names": EVIDENCE_NAMES,
                        }
                        with open(arm_files[arm_key], "a") as af:
                            af.write(json.dumps(err_record) + "\n")
                            af.flush()

                dt_cell = time.time() - t_cell_start
                print(f"Finished Condition {condition_id} in {dt_cell:.1f}s", flush=True)

            del X_all, Y_all
            if dev == "mps":
                torch.mps.empty_cache()

    total_time = time.time() - t_global_start
    print(f"\nAll requested conditions executed in {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
