#!/usr/bin/env python3
"""True AETTA-style dropout baseline on the ImageNet-C SAR grid.

This script prepares the missing "real AETTA" decision baseline for the same
ImageNet-C grid used by K-Bound SAR evaluations:
  - adapt candidate with SAR on each condition
  - compute true MC-dropout prediction disagreement on the adapted model
  - tune a single adapt/freeze threshold with leave-one-out calibration
  - report policy regret against always-adapt / always-freeze / oracle

Intended usage (once torch/numpy + ImageNet-C are available):
  python3 docs/research/kbound/scripts/run_aetta_dropout_imagenetc.py \
      --imagenetc-root /path/to/ImageNet-C \
      --out experiments/kbound/results/decision_baselines_sarfix/aetta_dropout_imagenetc.json \
      --quick
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from dataclasses import dataclass
from itertools import combinations

import numpy as np


def _load_harness(script_path: str):
    spec = importlib.util.spec_from_file_location("kbound_harness", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load harness at {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enable_only_dropout_train(model):
    model.eval()
    for mod in model.modules():
        cls = mod.__class__.__name__.lower()
        if "dropout" in cls:
            mod.train()


def _inject_head_dropout(model, arch: str, p: float, torch_module):
    m = copy.deepcopy(model)
    nn = torch_module.nn
    if arch == "resnet50" and hasattr(m, "fc"):
        m.fc = nn.Sequential(nn.Dropout(p=p), m.fc)
        return m
    if arch == "vit_b16" and hasattr(m, "heads") and hasattr(m.heads, "head"):
        m.heads.head = nn.Sequential(nn.Dropout(p=p), m.heads.head)
        return m
    raise ValueError(f"Unsupported architecture for MC-dropout head injection: {arch}")


def _stochastic_predictions(model, x, n_passes: int, batch_size: int, torch_module):
    preds = []
    _enable_only_dropout_train(model)
    with torch_module.no_grad():
        for _ in range(n_passes):
            yhat = []
            for i in range(0, len(x), batch_size):
                yhat.append(model(x[i : i + batch_size]).argmax(1).cpu().numpy())
            preds.append(np.concatenate(yhat, axis=0))
    return preds


def _pairwise_disagreement(preds: list[np.ndarray]) -> float:
    if len(preds) < 2:
        return 0.0
    dis = []
    for i, j in combinations(range(len(preds)), 2):
        dis.append(float(np.mean(preds[i] != preds[j])))
    return float(np.mean(dis))


def _loo_threshold(scores: np.ndarray, a0: np.ndarray, aa: np.ndarray, grid: int = 201):
    lo, hi = float(scores.min()), float(scores.max())
    taus = np.linspace(lo - 1e-9, hi + 1e-9, grid)
    out = np.empty(len(scores), dtype=object)
    for i in range(len(scores)):
        tr = np.arange(len(scores)) != i
        best_tau = taus[0]
        best_regret = np.inf
        for tau in taus:
            dec = np.where(scores[tr] <= tau, "ADAPT", "FREEZE")
            realized = np.where(dec == "ADAPT", aa[tr], a0[tr])
            regret = float((np.maximum(a0[tr], aa[tr]) - realized).mean())
            if regret < best_regret - 1e-12:
                best_regret = regret
                best_tau = tau
        out[i] = "ADAPT" if scores[i] <= best_tau else "FREEZE"
    return out


def _policy_metrics(decisions: np.ndarray, a0: np.ndarray, aa: np.ndarray):
    oracle = np.maximum(a0, aa)
    realized = np.where(decisions == "ADAPT", aa, a0)
    return {
        "n": int(len(a0)),
        "mean_acc": {
            "always_adapt": float(aa.mean()),
            "always_freeze": float(a0.mean()),
            "aetta_dropout": float(realized.mean()),
            "oracle": float(oracle.mean()),
        },
        "regret_vs_oracle": {
            "always_adapt": float((oracle - aa).mean()),
            "always_freeze": float((oracle - a0).mean()),
            "aetta_dropout": float((oracle - realized).mean()),
        },
        "harmful_base_rate": float(np.mean((aa - a0) < 0)),
        "adapt_rate": float(np.mean(decisions == "ADAPT")),
    }


@dataclass
class Row:
    condition: str
    a0: float
    aa: float
    dropout_disagreement: float


def main():
    ap = argparse.ArgumentParser(description="AETTA dropout baseline on ImageNet-C SAR grid")
    ap.add_argument("--imagenetc-root", required=True, help="ImageNet-C root (<corr>/<sev>/<class>/*.JPEG)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--arch", choices=["resnet50", "vit_b16"], default="resnet50")
    ap.add_argument("--quick", action="store_true", help="Use harness quick corruption/severity subset")
    ap.add_argument("--max-images", type=int, default=4000, help="Per-cell sampled eval images")
    ap.add_argument("--severities", nargs="+", type=int, default=None)
    ap.add_argument("--all-batch", action="store_true", help="Use large/small/tiny batches (else large/tiny)")
    ap.add_argument("--dropout-p", type=float, default=0.2)
    ap.add_argument("--mc-passes", type=int, default=6)
    ap.add_argument("--mc-eval-images", type=int, default=1024, help="Subset used for disagreement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sar-lr", type=float, default=2.5e-4)
    ap.add_argument("--sar-freeze-layer4", dest="sar_freeze_layer4", action="store_true")
    ap.add_argument("--no-sar-freeze-layer4", dest="sar_freeze_layer4", action="store_false")
    ap.set_defaults(sar_freeze_layer4=True)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    harness_path = os.path.join(here, "cifar_tent_mps_v2.py")
    H = _load_harness(harness_path)
    torch = H.torch
    tv = H.tv
    if torch is None or tv is None:
        raise RuntimeError("Torch/torchvision unavailable in this environment.")

    H.set_global_seed(args.seed)
    device = H.pick_device()
    print(f"[aetta-dropout] device={device} seed={args.seed}")

    if args.arch == "vit_b16":
        weights = tv.models.ViT_B_16_Weights.IMAGENET1K_V1
        frozen = tv.models.vit_b_16(weights=weights).to(device).eval()
    else:
        weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
        frozen = tv.models.resnet50(weights=weights).to(device).eval()

    tf = weights.transforms()
    if args.quick:
        corrs = list(H.IMAGENET_C_QUICK)
    else:
        corrs = sorted(
            d
            for d in os.listdir(args.imagenetc_root)
            if os.path.isdir(os.path.join(args.imagenetc_root, d)) and not d.startswith(".")
        )
    corrs = [c for c in corrs if os.path.isdir(os.path.join(args.imagenetc_root, c))]
    if not corrs:
        raise RuntimeError(f"No ImageNet-C corruption folders found under {args.imagenetc_root}")

    severities = [1, 5] if args.quick else (args.severities if args.severities else H.SEVERITIES)
    batch_regimes = [("large_iid", 128), ("small", 16), ("tiny", 8)] if args.all_batch else [("large_iid", 64), ("tiny", 8)]

    rows: list[Row] = []
    for corr in corrs:
        for sev in severities:
            if not os.path.isdir(os.path.join(args.imagenetc_root, corr, str(sev))):
                continue
            loader = H.imagenet_c_loader(args.imagenetc_root, corr, sev, tf, args.max_images, device)
            xs, ys = [], []
            for xb, yb in loader:
                xs.append(xb)
                ys.append(yb)
            X = torch.cat(xs).to(device)
            Y = torch.cat(ys)
            for br_name, bs in batch_regimes:
                for ag_name, ag in H.AGGRESSIVENESS.items():
                    key = f"{corr}|s{sev}|{br_name}|{ag_name}"
                    perm = torch.randperm(len(X))[: bs * 8]
                    stream = [X[perm[i : i + bs]] for i in range(0, len(perm), bs)]

                    a0 = H.acc_on(frozen, X, Y, train_mode=False)
                    adapted, _ = H.sar_adapt(
                        frozen,
                        stream,
                        steps=ag["steps"],
                        lr=ag["lr"],
                        num_classes=1000,
                        sar_lr=args.sar_lr,
                        freeze_layer4=args.sar_freeze_layer4,
                    )
                    aa = H.acc_on(adapted, X, Y, train_mode=True)

                    mc_model = _inject_head_dropout(adapted, args.arch, args.dropout_p, torch)
                    X_mc = X[: min(len(X), args.mc_eval_images)]
                    preds = _stochastic_predictions(mc_model, X_mc, args.mc_passes, batch_size=256, torch_module=torch)
                    dis = _pairwise_disagreement(preds)
                    rows.append(Row(condition=key, a0=float(a0), aa=float(aa), dropout_disagreement=float(dis)))
                    print(f"[aetta-dropout] {key} a0={a0:.3f} aa={aa:.3f} disagreement={dis:.4f}")

                    del adapted, mc_model
                    H._mps_free()
            del X, Y
            H._mps_free()

    if not rows:
        raise RuntimeError("No rows produced; check paths and corruption folders.")

    scores = np.array([r.dropout_disagreement for r in rows], dtype=float)
    a0 = np.array([r.a0 for r in rows], dtype=float)
    aa = np.array([r.aa for r in rows], dtype=float)
    dec = _loo_threshold(scores, a0, aa)
    metrics = _policy_metrics(dec, a0, aa)

    out = {
        "method": "aetta_dropout",
        "notes": "Lower disagreement => adapt. Threshold tuned with leave-one-out regret.",
        "config": {
            "imagenetc_root": args.imagenetc_root,
            "arch": args.arch,
            "quick": args.quick,
            "max_images": args.max_images,
            "severities": severities,
            "batch_regimes": batch_regimes,
            "dropout_p": args.dropout_p,
            "mc_passes": args.mc_passes,
            "mc_eval_images": args.mc_eval_images,
            "seed": args.seed,
            "sar_lr": args.sar_lr,
            "sar_freeze_layer4": args.sar_freeze_layer4,
        },
        "metrics": metrics,
        "rows": [
            {
                "condition": r.condition,
                "a0": r.a0,
                "aa": r.aa,
                "benefit": r.aa - r.a0,
                "dropout_disagreement": r.dropout_disagreement,
                "decision": str(d),
            }
            for r, d in zip(rows, dec)
        ],
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[aetta-dropout] wrote {out_path}")


if __name__ == "__main__":
    main()

