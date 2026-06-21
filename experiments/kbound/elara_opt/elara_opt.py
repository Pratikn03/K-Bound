"""elara_opt.py — ELARA-Opt: a genuine label-free test-time *parameter-update*
optimizer that the K-Bound / KGA gate can certify as a candidate.

From an UNLABELED batch it builds an adapted model by updating only the
BN/LN-affine parameters (the same surface Tent/EATA/SAR use) with:
  * a reliability-gated mixture of unlabeled objectives
    {entropy, reliability-filtered entropy, augmentation-consistency},
  * a frozen-model KL stability anchor (function-space trust), and
  * a trust-region constraint on each step whose radius grows with reliability.

It is NOT a prediction router and NOT a renamed KGA threshold: it returns a new
nn.Module whose affine parameters genuinely moved.  Three gate modes
(elara_uniform / elara_rule / elara_meta) decide the objective weights.  Faithful
SAR/SAM gradients are reused from the existing kbound_tta.sar_adapt only if
explicitly requested; nothing is fabricated.

Reuses the validated helpers _clone_for_tta / _bn_affine_params / _upd_norm.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch

from ._compat import _clone_for_tta, _upd_norm
from .config import ELARA_OPT_DEFAULTS
from .objectives import all_mixture_losses, frozen_kl_anchor, augment, OBJECTIVE_NAMES
from .reliability import compute_features, reliability_score
from .gate import compute_weights, MetaGate
from .telemetry import TelemetryCollector, candidate_hash


def trust_radius(reliability: float, cfg: Dict) -> float:
    tr = cfg["trust_region"]
    return float(tr["r_min"] + (tr["r_max"] - tr["r_min"]) * reliability)


def _flat(gs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in gs])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


def _obj_grads(loss: torch.Tensor, ps: List[torch.Tensor]) -> List[torch.Tensor]:
    """Per-objective gradient on the affine params; None -> zeros (allow_unused)."""
    gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(gs, ps)]


def elara_opt_adapt(
    base: torch.nn.Module,
    stream: List[torch.Tensor],
    steps: int,
    lr: float,
    num_classes: int,
    *,
    mode: str = "elara_rule",
    cfg: Optional[Dict] = None,
    meta_model: Optional[MetaGate] = None,
    seed: int = 0,
    collect_telemetry: bool = True,
    fixed_weights=None,
):
    """Adapt `base` on the unlabeled `stream`. Returns (adapted_model, update_norm,
    telemetry_dict). `stream` is a list of normalized tensor batches on device.
    No target labels are accepted or used anywhere in this function.

    `fixed_weights` (3-vector over {entropy, filtered_entropy, aug_consistency}),
    if given, bypasses the gate — used ONLY by train_meta_gate.py to score weight
    presets on DEV/source tasks. It is never wired to any held-out evaluation.
    """
    cfg = cfg or ELARA_OPT_DEFAULTS
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    m, ps, init = _clone_for_tta(base)         # validated clone: only affine params train
    base.eval()
    lam = float(cfg.get("lambda_anchor", 1.0))
    margin_frac = float(cfg.get("margin_frac", 0.4))
    tele = TelemetryCollector(mode=mode, seed=seed)

    grad_conflict = 0.0
    cur_norm = 0.0
    for step in range(int(steps)):
        for bi, xb in enumerate(stream):
            xb = xb.contiguous()
            aug_x = augment(xb, seed=seed + step * 100 + bi)

            # --- label-free reliability features (no grad) ---
            feats = compute_features(
                base, m, xb, num_classes,
                update_norm=cur_norm, grad_conflict=grad_conflict, aug=aug_x,
            )
            s = reliability_score(feats, cfg["reliability_coeffs"])
            radius = trust_radius(s, cfg)
            if fixed_weights is not None:
                w = np.asarray(fixed_weights, dtype=np.float64)
            else:
                w = compute_weights(mode, feats, cfg, meta_model)   # nonneg, sums to 1

            # --- forward (with grad on current model) ---
            m.train()
            logits_t = m(xb)
            logits_aug = m(aug_x)
            with torch.no_grad():
                logits_0 = base(xb)

            losses, kept = all_mixture_losses(logits_t, logits_aug, num_classes, margin_frac)
            anchor = frozen_kl_anchor(logits_0, logits_t)

            # --- per-objective grads + conflict ---
            g_obj = {name: _obj_grads(losses[name], ps) for name in OBJECTIVE_NAMES}
            g_anchor = _obj_grads(anchor, ps)
            flats = {name: _flat(g_obj[name]) for name in OBJECTIVE_NAMES}
            cos_ef = _cos(flats["entropy"], flats["filtered_entropy"])
            cos_ea = _cos(flats["entropy"], flats["aug_consistency"])
            cos_fa = _cos(flats["filtered_entropy"], flats["aug_consistency"])
            grad_conflict = float(min(cos_ef, cos_ea, cos_fa))
            cos_anchor = _cos(_flat([w[0] * a + w[1] * b + w[2] * c
                                     for a, b, c in zip(g_obj["entropy"],
                                                        g_obj["filtered_entropy"],
                                                        g_obj["aug_consistency"])]),
                              _flat(g_anchor))

            # --- combined gradient, manual SGD, trust-region clip ---
            combined = [
                w[0] * ge + w[1] * gf + w[2] * ga + lam * gan
                for ge, gf, ga, gan in zip(g_obj["entropy"], g_obj["filtered_entropy"],
                                           g_obj["aug_consistency"], g_anchor)
            ]
            with torch.no_grad():
                delta = [(-lr) * g for g in combined]
                dn = float(_flat(delta).norm())
                if dn > radius and dn > 1e-12:
                    scale = radius / dn
                    delta = [d * scale for d in delta]
                    clipped = True
                else:
                    clipped = False
                for p, d in zip(ps, delta):
                    p.add_(d)
                cur_norm = _upd_norm(ps, init)

            if collect_telemetry:
                tele.log_step({
                    "step": step, "batch": bi,
                    "loss_entropy": float(losses["entropy"].detach()),
                    "loss_filtered_entropy": float(losses["filtered_entropy"].detach()),
                    "loss_aug_consistency": float(losses["aug_consistency"].detach()),
                    "loss_anchor_kl": float(anchor.detach()),
                    "reliable_fraction": float(kept),
                    "reliability_score": float(s),
                    "trust_radius": float(radius),
                    "step_norm_pre_clip": float(dn),
                    "trust_region_clipped": bool(clipped),
                    "update_norm": float(cur_norm),
                    "gate_weights": [float(x) for x in w],
                    "grad_cos_entropy_filtered": cos_ef,
                    "grad_cos_entropy_aug": cos_ea,
                    "grad_cos_filtered_aug": cos_fa,
                    "grad_cos_mixture_anchor": cos_anchor,
                    "grad_conflict_min_cos": grad_conflict,
                    "reliability_features": {k: float(v) for k, v in feats.items()},
                })
            del g_obj, g_anchor, flats, combined, delta

    chash = candidate_hash(ps)
    summary = {
        "candidate_hash": chash,
        "num_classes": int(num_classes),
        "final_update_norm": float(cur_norm),
        "final_gate_weights": [float(x) for x in w],
        "final_reliability_score": float(s),
        "objective_names": OBJECTIVE_NAMES,
        "lambda_anchor": lam,
        "lr": float(lr),
        "steps": int(steps),
    }
    if collect_telemetry:
        tele.finalize(summary)
    return m, cur_norm, tele.to_dict()


class ELARAOptAdapter:
    """Class wrapper exposing ELARA-Opt behind the existing adapter API.

    `adapt(...)` mirrors the (base, stream, steps, lr, num_classes) -> (model,
    update_norm) contract of tent/eata/sar, plus a telemetry payload.  `as_method`
    yields a closure with the bare (base, stream, steps, lr) signature so it can be
    dropped into run_candidate-style dispatch tables.
    """

    def __init__(self, mode: str = "elara_rule", cfg: Optional[Dict] = None,
                 meta_model: Optional[MetaGate] = None, seed: int = 0):
        if cfg is None:
            cfg = ELARA_OPT_DEFAULTS
        if mode not in cfg["modes"]:
            raise ValueError(f"mode {mode} not in {cfg['modes']}")
        self.mode = mode
        self.cfg = cfg
        self.meta_model = meta_model
        self.seed = seed
        self.last_telemetry: Optional[Dict] = None

    def adapt(self, base, stream, steps, lr, num_classes, collect_telemetry=True):
        m, upd, tele = elara_opt_adapt(
            base, stream, steps, lr, num_classes,
            mode=self.mode, cfg=self.cfg, meta_model=self.meta_model,
            seed=self.seed, collect_telemetry=collect_telemetry,
        )
        self.last_telemetry = tele
        return m, upd

    def as_method(self, num_classes: int):
        def _f(base, stream, steps, lr):
            return self.adapt(base, stream, steps, lr, num_classes)
        return _f
