"""
oh_candidates.py - the two NON-gradient candidates for Office-Home plus the unified
label-free evidence panel.

(a) labelshift  : full Maximum-Likelihood-Label-Shift (MLLS/BBSE-EM) target-prior
                  estimate from the frozen model's softmax on the adaptation stream,
                  applied as a logit adjustment  logit'_c = logit_c + log(pi_t,c/pi_s,c).
                  NO parameter update.  Can be HARMFUL when the stream prior is skewed
                  (single_class / imbalanced) and the eval pool is balanced -> the harm
                  is label-free DETECTABLE via the predicted-prior features.
(b) conservative: the SAME estimate but damped (lambda small) and the per-class log
                  adjustment CLIPPED to a small band -> a low-risk "safe" option that
                  helps mildly without collapse risk.  NO parameter update.

Unified EVIDENCE PANEL (label-free except ATC, which may use SOURCE labels): computed
on the eval set for EVERY candidate (gradient and non-gradient) so Z is comparable.
"""
from __future__ import annotations
import math
import numpy as np
import torch

import tta_methods as tm  # energy_score, atc_threshold_acc, bn_* helpers


EVIDENCE_NAMES_OH = [
    "pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf", "post_pbal",
    "pbal_drop", "entropy_drop", "frac_highconf", "marginal_KL", "update_norm",
    "disagreement", "energy_shift", "bn_kl", "atc_acc_est", "conf_drop", "prior_kl_source",
    "cand_disagree_mean",   # appended in a 2nd pass (cross-candidate agreement)
]


def _softmax_np(x):
    x = np.asarray(x, float); x = x - x.max(1, keepdims=True)
    e = np.exp(x); return e / e.sum(1, keepdims=True)


def estimate_labelshift_weights(stream_probs, source_prior, iters=50):
    """MLLS/EM target-prior estimate w_c = pi_t,c / pi_s,c from frozen softmax q (label-free)."""
    q = np.asarray(stream_probs, float)
    ps = np.asarray(source_prior, float); ps = ps / ps.sum()
    w = np.ones_like(ps)
    for _ in range(iters):
        num = q * w[None, :]
        post = num / np.clip(num.sum(1, keepdims=True), 1e-12, None)
        pt = post.mean(0)
        pt = np.clip(pt, 1e-6, None); pt = pt / pt.sum()
        w_new = pt / ps
        if np.max(np.abs(w_new - w)) < 1e-6:
            w = w_new; break
        w = w_new
    return w, (ps * w) / np.sum(ps * w)   # weights, estimated target prior


def apply_prior_correction(logits_eval, w, lam=1.0, clip=None):
    """logit'_c = logit_c + lam*log(w_c), optionally clipping the adjustment band."""
    adj = lam * np.log(np.clip(w, 1e-12, None))
    if clip is not None:
        adj = np.clip(adj, -clip, clip)
    return np.asarray(logits_eval, float) + adj[None, :]


def run_prior_candidate(kind, f0, stream, eval_x, num_classes, source_prior, eval_bs=128):
    """Return (preds, pa_probs_eval, logits_eval_adj, upd_norm=0).  Inference-only."""
    stream_x = torch.cat(stream, 0)
    q_stream = _softmax_np(tm.predict_logits(f0, stream_x, train_mode=False, bs=eval_bs))
    w, _pt = estimate_labelshift_weights(q_stream, source_prior)
    logits_eval = tm.predict_logits(f0, eval_x, train_mode=False, bs=eval_bs)
    if kind == "labelshift":
        adj = apply_prior_correction(logits_eval, w, lam=1.0, clip=None)
    elif kind == "conservative":
        adj = apply_prior_correction(logits_eval, w, lam=0.25, clip=0.7)
    else:
        raise ValueError(kind)
    pa = _softmax_np(adj)
    preds = pa.argmax(1)
    return preds, pa, adj, 0.0


# ---------------- unified evidence panel ----------------
def _ent_rows(p):
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def full_evidence(p0_eval, pa_eval, logits_f0_eval, logits_src, y_src, bn_kl,
                  upd_norm, num_classes, source_prior):
    """17-dim label-free panel on the eval set (ATC may use source labels)."""
    p0 = np.asarray(p0_eval, float); pa = np.asarray(pa_eval, float)
    lnK = math.log(num_classes)
    e0 = float(_ent_rows(p0).mean()); ea = float(_ent_rows(pa).mean())
    conf0 = float(p0.max(1).mean()); confa = float(pa.max(1).mean())
    mb0 = p0.mean(0); mba = pa.mean(0)
    pbal0 = float(-(mb0 * np.log(mb0 + 1e-12)).sum() / lnK)
    pbala = float(-(mba * np.log(mba + 1e-12)).sum() / lnK)
    frac_hi = float((pa.max(1) > 0.9).mean())
    klm = float((mba * (np.log(mba + 1e-12) - np.log(mb0 + 1e-12))).sum())
    disagree = float((p0.argmax(1) != pa.argmax(1)).mean())
    energy_shift = float(tm.energy_score(logits_f0_eval).mean() - tm.energy_score(logits_src).mean()) \
        if logits_src is not None else 0.0
    atc = tm.atc_threshold_acc(logits_src, y_src, logits_f0_eval) if logits_src is not None else 0.0
    conf_drop = conf0 - confa
    ps = np.asarray(source_prior, float); ps = ps / ps.sum()
    prior_kl = float((mba * (np.log(mba + 1e-12) - np.log(ps + 1e-12))).sum())
    return [e0, conf0, pbal0, ea, confa, pbala, pbal0 - pbala, e0 - ea, frac_hi, klm,
            float(upd_norm), disagree, energy_shift, float(bn_kl), float(atc), conf_drop, prior_kl]
