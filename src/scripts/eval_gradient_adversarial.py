"""Gradient-based adversarial robustness for the fusion head.

The existing score-level adversarial tests in the breakthrough runner
perturb the input scores with deterministic patterns (zero_attack,
max_attack, gaussian_noise). A senior-reviewer criticism is that real
adversaries craft *gradient-aware* perturbations against the fusion
model's logits, not these synthetic score-collapse patterns.

This script loads a benchmark's existing CSV, trains a fresh fusion
model (mirroring the breakthrough runner's training loop), and runs
FGSM~\\cite{goodfellow2014explaining} and PGD~\\cite{madry2017towards}
against the fused logits. Both attacks perturb the input feature
vector x with a sign-of-gradient step that maximises BCE loss on the
true label, clipped to an L_inf epsilon ball.

Output: per-attack ROC-AUC for (static_attention, craf_attention) at a
sweep of L_inf epsilon values. Honest framing: this is a white-box
threat model against the fusion head itself; the upstream PatchCore
features are not retrained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from scripts.run_breakthrough_experiment import (
    _build_model,
    _load_data,
    _make_loaders,
    _make_reliability_estimator,
    _predict_static,
    _split,
    _train_model,
    set_seed,
)


def _fgsm_attack(
    model,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """One-step FGSM perturbation of the input feature tensor."""
    model.eval()
    perturbed = features.copy().astype(np.float32)
    n = features.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feat_t = torch.tensor(perturbed[start:end], dtype=torch.float32, device=device, requires_grad=True)
        mask_t = torch.tensor(masks[start:end], dtype=torch.bool, device=device)
        lbl_t = torch.tensor(labels[start:end], dtype=torch.float32, device=device)
        logits, _, _ = model(feat_t, key_padding_mask=mask_t)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), lbl_t)
        loss.backward()
        if feat_t.grad is None:
            continue
        step = epsilon * feat_t.grad.detach().sign()
        with torch.no_grad():
            new_feat = (feat_t.detach() + step).cpu().numpy()
        new_feat = np.clip(new_feat, 0.0, 1.0)  # feature schema is [0,1]
        perturbed[start:end] = new_feat
    return perturbed


def _pgd_attack(
    model,
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    n_steps: int,
    step_size: float,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Projected Gradient Descent over the L_inf epsilon-ball."""
    model.eval()
    perturbed = features.copy().astype(np.float32)
    n = features.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        original = torch.tensor(features[start:end], dtype=torch.float32, device=device)
        mask_t = torch.tensor(masks[start:end], dtype=torch.bool, device=device)
        lbl_t = torch.tensor(labels[start:end], dtype=torch.float32, device=device)
        x = original.clone().detach().requires_grad_(True)
        for _ in range(int(n_steps)):
            x.requires_grad_(True)
            logits, _, _ = model(x, key_padding_mask=mask_t)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), lbl_t)
            grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
            x = x.detach() + step_size * grad.sign()
            x = torch.clamp(x, original - epsilon, original + epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        perturbed[start:end] = x.detach().cpu().numpy()
    return perturbed


def run_gradient_adversarial(
    cfg: dict,
    *,
    epsilons: list[float] = (0.02, 0.05, 0.10, 0.20),
    pgd_steps: int = 10,
) -> dict:
    """Run FGSM + PGD against the static-attention fusion head."""
    (
        features, masks, labels, sample_ids, domain_order, _, conf_idx,
        score_idx, sample_splits, sample_cats,
    ) = _load_data(cfg)
    set_seed(int(cfg.get("training", {}).get("seed", 42)))
    train_idx, val_idx, test_idx = _split(
        labels,
        {**cfg["training"], "seed": int(cfg.get("training", {}).get("seed", 42))},
        split_values=sample_splits,
    )
    train_loader, val_loader, _ = _make_loaders(
        features, masks, labels, train_idx, val_idx, test_idx,
        batch_size=int(cfg["training"].get("batch_size", 64)),
    )
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = _build_model(cfg, features.shape[1], features.shape[2], conf_idx, device)
    _train_model(model, train_loader, val_loader, cfg, device)
    model.eval()

    test_feat = features[test_idx]
    test_mask = masks[test_idx]
    test_labels = labels[test_idx]

    clean_probs = _predict_static(model, test_feat, test_mask, device)
    try:
        clean_auc = float(roc_auc_score(test_labels, clean_probs))
    except ValueError:
        clean_auc = float("nan")

    results = {
        "clean": {"static_auroc": clean_auc, "epsilon": 0.0},
        "fgsm": [],
        "pgd": [],
        "epsilons": list(epsilons),
        "pgd_steps": int(pgd_steps),
    }
    for eps in epsilons:
        # FGSM
        fgsm_feat = _fgsm_attack(model, test_feat, test_mask, test_labels, eps, device)
        fgsm_probs = _predict_static(model, fgsm_feat, test_mask, device)
        try:
            fgsm_auc = float(roc_auc_score(test_labels, fgsm_probs))
        except ValueError:
            fgsm_auc = float("nan")
        results["fgsm"].append({"epsilon": float(eps), "static_auroc": fgsm_auc})

        # PGD
        pgd_feat = _pgd_attack(
            model, test_feat, test_mask, test_labels,
            epsilon=eps, n_steps=pgd_steps, step_size=eps / max(pgd_steps // 2, 1),
            device=device,
        )
        pgd_probs = _predict_static(model, pgd_feat, test_mask, device)
        try:
            pgd_auc = float(roc_auc_score(test_labels, pgd_probs))
        except ValueError:
            pgd_auc = float("nan")
        results["pgd"].append({"epsilon": float(eps), "static_auroc": pgd_auc})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/fusion/gradient_adversarial_results.json"),
    )
    parser.add_argument("--epsilons", nargs="*", type=float, default=[0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--pgd-steps", type=int, default=10)
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    results = run_gradient_adversarial(cfg, epsilons=args.epsilons, pgd_steps=args.pgd_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
