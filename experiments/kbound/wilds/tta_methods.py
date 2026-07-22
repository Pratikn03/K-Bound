"""
tta_methods.py - faithful test-time adaptation for the WILDS Camelyon17 K-Bound
pipeline.  Tent / EATA / SAR, each usable in ONLINE mode (state carried across the
adaptation stream) or EPISODIC mode (model reset to f0 for every test batch), plus
the label-free evidence vector Z and a balanced-accuracy evaluator.

The method bodies (tent_adapt / eata_adapt / sar_adapt / evidence_vector and the
BN-affine helpers) are ported VERBATIM from the project's validated reference
harness  docs/research/kbound/scripts/cifar_tent_mps_v2.py  so Camelyon17 uses
exactly the same faithful implementations (incl. SAR's SAM + entropy-EMA reset)
as the ImageNet-C / CIFAR sweeps.  Only pick_device and the online/episodic
wrappers are new.  INTEGRITY: every adapted model is produced by a real update;
nothing is fabricated.
"""
from __future__ import annotations
import copy, math
import numpy as np
import torch
import torch.nn as nn


def pick_device(prefer: str = "auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is unavailable")
        return torch.device("mps")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mps_free():
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---- label-free evidence ----------------------------------------------------
def _entropy(p):
    return -(p * (p + 1e-9).log()).sum(1)


def evidence_vector(model_frozen, model_adapted, x, num_classes, upd_norm):
    """All label-free.  x is a normalized batch on device.  Returns 11-dim Z."""
    model_frozen.eval(); model_adapted.eval()
    with torch.no_grad():
        p0 = model_frozen(x).softmax(1)
        e0 = _entropy(p0).mean().item()
        conf0 = p0.max(1).values.mean().item()
        mb0 = p0.mean(0); pbal0 = (-(mb0 * (mb0 + 1e-9).log()).sum()).item() / math.log(num_classes)
        pa = model_adapted(x).softmax(1)
        ea = _entropy(pa).mean().item()
        confa = pa.max(1).values.mean().item()
        mba = pa.mean(0); pbala = (-(mba * (mba + 1e-9).log()).sum()).item() / math.log(num_classes)
        frac_hi = (pa.max(1).values > 0.9).float().mean().item()
        klm = (mba * ((mba + 1e-9).log() - (mb0 + 1e-9).log())).sum().item()
    # [pre_entropy, pre_conf, pre_pbal, post_entropy, post_conf, post_pbal,
    #  pbal_drop, entropy_drop, frac_highconf, marginal_KL, update_norm]
    return [e0, conf0, pbal0, ea, confa, pbala, pbal0 - pbala, e0 - ea, frac_hi, klm, upd_norm]


EVIDENCE_NAMES = ["pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf",
                  "post_pbal", "pbal_drop", "entropy_drop", "frac_highconf",
                  "marginal_KL", "update_norm"]

RICH_EVIDENCE_NAMES = [
    "disagreement_rate",
    "entropy_gap",
    "energy_shift",
    "bn_kl",
    "atc_acc_est",
    "conf_drop",
]


def _softmax_np(x):
    x = np.asarray(x, float)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _entropy_np(logits):
    p = _softmax_np(logits)
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def energy_score(logits):
    """Negative free energy, -logsumexp(logits), per sample."""
    L = np.asarray(logits, float)
    m = L.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(L - m).sum(axis=1)))


def atc_threshold_acc(logits_src, y_src, logits_tgt):
    """ATC target accuracy estimate using source labels only."""
    ps = _softmax_np(logits_src)
    ss = ps.max(axis=1)
    ys = np.asarray(y_src)
    err = float(np.mean(ps.argmax(axis=1) != ys))
    if err <= 0.0:
        t = float(ss.min() - 1e-9)
    elif err >= 1.0:
        t = float(ss.max() + 1e-9)
    else:
        t = float(np.quantile(ss, err))
    st = _softmax_np(logits_tgt).max(axis=1)
    return float(np.mean(st >= t))


def bn_stat_kl_drift(running_mean, running_var, batch_mean, batch_var):
    """Mean KL between BN running Gaussian stats and target-batch Gaussian stats."""
    if running_mean is None or batch_mean is None:
        return 0.0
    rm = np.asarray(running_mean, float)
    rv = np.asarray(running_var, float) + 1e-6
    bm = np.asarray(batch_mean, float)
    bv = np.asarray(batch_var, float) + 1e-6
    k = min(len(rm), len(bm))
    rm, rv, bm, bv = rm[:k], rv[:k], bm[:k], bv[:k]
    kl = 0.5 * (np.log(bv / rv) + (rv + (rm - bm) ** 2) / bv - 1.0)
    return float(np.mean(np.clip(kl, 0.0, 1e6)))


def rich_evidence_vector(logits_f0_tgt, logits_adapt_tgt, logits_src, y_src, bn_kl):
    """Protocol-F rich evidence panel. Target side is label-free; source labels are allowed."""
    p0 = _softmax_np(logits_f0_tgt)
    pa = _softmax_np(logits_adapt_tgt)
    disagreement = float(np.mean(p0.argmax(axis=1) != pa.argmax(axis=1)))
    ent_gap = float(_entropy_np(logits_f0_tgt).mean() - _entropy_np(logits_adapt_tgt).mean())
    energy_shift = float(energy_score(logits_f0_tgt).mean() - energy_score(logits_src).mean())
    atc = atc_threshold_acc(logits_src, y_src, logits_f0_tgt)
    conf_drop = float(p0.max(axis=1).mean() - pa.max(axis=1).mean())
    return np.array([disagreement, ent_gap, energy_shift, float(bn_kl), atc, conf_drop], dtype=float)


@torch.no_grad()
def bn_running_stats(model):
    means, vars_ = [], []
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)) and mod.running_mean is not None:
            means.append(mod.running_mean.detach().cpu().numpy())
            vars_.append(mod.running_var.detach().cpu().numpy())
    if not means:
        return None, None
    return np.concatenate(means), np.concatenate(vars_)


@torch.no_grad()
def bn_batch_stats(model, x):
    """Measure BN input stats on one target batch via forward hooks."""
    feats = {"mean": [], "var": []}
    hooks = []

    def _hook(_mod, inp, _out):
        t = inp[0].detach()
        dims = tuple(i for i in range(t.ndim) if i != 1)
        feats["mean"].append(t.mean(dim=dims).cpu().numpy())
        feats["var"].append(t.var(dim=dims, unbiased=False).cpu().numpy())

    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            hooks.append(mod.register_forward_hook(_hook))
    model.eval()
    _ = model(x)
    for h in hooks:
        h.remove()
    if not feats["mean"]:
        return None, None
    return np.concatenate(feats["mean"]), np.concatenate(feats["var"])


# ---- BN/LN-affine params + clone --------------------------------------------
def _bn_affine_params(m):
    ps = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.track_running_stats = False; mod.running_mean = None; mod.running_var = None
            if mod.weight is not None: mod.weight.requires_grad_(True); ps.append(mod.weight)
            if mod.bias is not None: mod.bias.requires_grad_(True); ps.append(mod.bias)
        elif isinstance(mod, nn.LayerNorm):
            if mod.weight is not None: mod.weight.requires_grad_(True); ps.append(mod.weight)
            if mod.bias is not None: mod.bias.requires_grad_(True); ps.append(mod.bias)
    return ps


def _clone_for_tta(base):
    m = copy.deepcopy(base); m.train()
    for p in m.parameters(): p.requires_grad_(False)
    ps = _bn_affine_params(m)
    init = [p.detach().clone() for p in ps]
    return m, ps, init


def _upd_norm(ps, init):
    return float(sum(((p.detach() - q).norm() ** 2).item() for p, q in zip(ps, init)) ** 0.5)


# ---- the three faithful methods (ported verbatim from cifar_tent_mps_v2.py) --
def tent_adapt(base, stream, steps, lr):
    m, ps, init = _clone_for_tta(base); opt = torch.optim.Adam(ps, lr=lr)
    for _ in range(steps):
        for xb in stream:
            out = m(xb.contiguous()); p = out.softmax(1); loss = _entropy(p).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m, _upd_norm(ps, init)


def eata_adapt(base, stream, steps, lr, num_classes, e_margin=None, fisher_alpha=2000.0):
    """Entropy-filtered adaptation + Fisher anti-forgetting (faithful-ish EATA)."""
    if e_margin is None: e_margin = 0.4 * math.log(num_classes)
    m, ps, init = _clone_for_tta(base); opt = torch.optim.Adam(ps, lr=lr)
    fisher = [torch.zeros_like(p) for p in ps]
    x0 = next(iter(stream))
    out = m(x0.contiguous()); p = out.softmax(1); loss = _entropy(p).mean()
    opt.zero_grad(); loss.backward()
    for k, p_ in enumerate(ps):
        if p_.grad is not None: fisher[k] = p_.grad.detach() ** 2
    for _ in range(steps):
        for xb in stream:
            out = m(xb.contiguous()); p = out.softmax(1); ent = _entropy(p)
            keep = ent < e_margin
            if keep.sum() == 0: continue
            loss = ent[keep].mean()
            reg = sum((f * (p_ - q) ** 2).sum() for f, p_, q in zip(fisher, ps, init))
            loss = loss + fisher_alpha * reg
            opt.zero_grad(); loss.backward(); opt.step()
    return m, _upd_norm(ps, init)


def sar_adapt(base, stream, steps, lr, num_classes, rho=0.05, margin_e0=None, reset_constant_em=0.2):
    """Faithful SAR (Niu et al., ICLR 2023): SAM first/second step, reliable-sample
    selection (entropy < E_0), EMA of the reliable loss as collapse criterion, and
    model+optimizer recovery when the EMA drops below reset_constant_em."""
    if margin_e0 is None: margin_e0 = 0.4 * math.log(num_classes)
    m, ps, init = _clone_for_tta(base)
    opt = torch.optim.SGD(ps, lr=lr, momentum=0.9)
    model_state = copy.deepcopy(m.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    ema = None
    for _ in range(steps):
        for xb in stream:
            xb = xb.contiguous()
            opt.zero_grad()
            ent = _entropy(m(xb).softmax(1)); keep1 = ent < margin_e0
            if keep1.sum() == 0: continue
            ent[keep1].mean().backward()
            with torch.no_grad():
                gnorm = sum((p.grad.detach() ** 2).sum() for p in ps if p.grad is not None) ** 0.5
                scale = rho / (gnorm + 1e-12)
                old_p = [p.data.clone() for p in ps]
                for p in ps:
                    if p.grad is not None: p.add_(p.grad * scale)
            opt.zero_grad()
            ent2 = _entropy(m(xb).softmax(1))[keep1]; keep2 = ent2 < margin_e0
            loss2 = ent2[keep2].mean() if keep2.any() else ent2.mean()
            if not math.isnan(loss2.item()):
                ema = loss2.item() if ema is None else 0.9 * ema + 0.1 * loss2.item()
            loss2.backward()
            with torch.no_grad():
                for p, q in zip(ps, old_p): p.data = q
            opt.step()
            if ema is not None and ema < reset_constant_em:
                with torch.no_grad(): m.load_state_dict(model_state, strict=True)
                opt.load_state_dict(opt_state); ema = None
    return m, _upd_norm(ps, init)


_BASE = {"tent": tent_adapt, "eata": eata_adapt, "sar": sar_adapt}


def _adapt(method, base, stream, steps, lr, num_classes):
    if method == "tent":
        return tent_adapt(base, stream, steps, lr)
    if method == "eata":
        return eata_adapt(base, stream, steps, lr, num_classes)
    if method == "sar":
        return sar_adapt(base, stream, steps, lr, num_classes)
    raise ValueError(f"unknown method {method}")


# ---- evaluation -------------------------------------------------------------
def _predict(model, x, train_mode=True, bs=256):
    model.train() if train_mode else model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(x[i:i + bs]).argmax(1).cpu())
    return torch.cat(out).numpy()


def predict_logits(model, x, train_mode=True, bs=256):
    model.train() if train_mode else model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(x[i:i + bs]).detach().cpu())
    return torch.cat(out).numpy()


def _prob_from_logits(logits, pos_class=1, mode="class"):
    p = _softmax_np(logits)
    return p.max(axis=1) if mode == "max" else p[:, pos_class]


def _predict_prob(model, x, train_mode=True, bs=256, pos_class=1, mode="class"):
    """Label-free scalar prob per sample for the Theorem-1B Brier-view route.
    mode='class' -> P(y=pos_class) (binary); mode='max' -> max-class prob (multi-class)."""
    model.train() if train_mode else model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            p = model(x[i:i + bs]).softmax(1)
            out.append((p.max(1).values if mode == "max" else p[:, pos_class]).cpu())
    return torch.cat(out).numpy()


def balanced_acc(preds, y_np):
    recalls = [float((preds[y_np == c] == c).mean()) for c in np.unique(y_np) if (y_np == c).any()]
    return float(np.mean(recalls)) if recalls else float((preds == y_np).mean())


def run_candidate(method, mode, f0, adapt_stream, eval_x, eval_y_np, num_classes,
                  steps, lr, eval_bs=64, prob_mode="class", episodic_steps=None,
                  return_details=False):
    """Run one candidate = (method, mode) and return (aa_balanced, Z, upd_norm, preds).

    mode='online'   : adapt one model across the whole adaptation stream (state carried),
                      then evaluate the resulting model on the balanced held-out eval pool.
                      Z is measured on the adaptation stream's first batch.
    mode='episodic' : reset to f0 for every eval batch and adapt transductively on that
                      batch before predicting it (standard episodic TTA).  Z is measured
                      from a single-batch episodic probe on the adaptation stream.
    preds are per-eval-sample predictions (for the multi-candidate agreement matrix).
    """
    probe = adapt_stream[0]
    if mode == "online":
        fa, upd = _adapt(method, f0, adapt_stream, steps, lr, num_classes)
        logits_eval = predict_logits(fa, eval_x, train_mode=True, bs=256)
        preds = logits_eval.argmax(axis=1)
        pa_pos = _prob_from_logits(logits_eval, mode=prob_mode)
        Z = evidence_vector(f0, fa, probe, num_classes, upd)
    elif mode == "episodic":
        est = episodic_steps if episodic_steps is not None else steps   # episodic: few steps/test-batch
        preds_chunks, prob_chunks, logits_chunks = [], [], []
        for i in range(0, len(eval_x), eval_bs):
            eb = eval_x[i:i + eval_bs]
            fb, _ = _adapt(method, f0, [eb], est, lr, num_classes)
            logits_b = predict_logits(fb, eb, train_mode=True, bs=eval_bs)
            preds_chunks.append(logits_b.argmax(axis=1))
            prob_chunks.append(_prob_from_logits(logits_b, mode=prob_mode))
            logits_chunks.append(logits_b)
        preds = np.concatenate(preds_chunks)
        pa_pos = np.concatenate(prob_chunks)
        logits_eval = np.concatenate(logits_chunks, axis=0)
        fprobe, upd = _adapt(method, f0, [probe], est, lr, num_classes)
        Z = evidence_vector(f0, fprobe, probe, num_classes, upd)
    else:
        raise ValueError(f"unknown mode {mode}")
    aa = balanced_acc(preds, eval_y_np)
    if return_details:
        return aa, Z, Z[-1], preds, pa_pos, {"logits_eval": logits_eval}
    return aa, Z, Z[-1], preds, pa_pos


def eval_frozen(f0, eval_x, eval_y_np, prob_mode="class", bs=256):
    """Frozen f0 balanced acc + per-sample predictions + scalar prob (eval mode = BN running stats)."""
    preds = _predict(f0, eval_x, train_mode=False, bs=bs)
    p0_pos = _predict_prob(f0, eval_x, train_mode=False, bs=bs, mode=prob_mode)
    return balanced_acc(preds, eval_y_np), preds, p0_pos
