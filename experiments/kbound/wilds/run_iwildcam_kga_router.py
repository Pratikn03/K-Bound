"""
run_iwildcam_kga_router.py - FINAL ceiling-break experiment for the K-Bound project.

QUESTION (pre-registered in results/iwildcam_kga_router/PREREG.md):
  The collapse pilot (run_iwildcam_streaming_pilot.py) PROVED that on iWildCam OOD
  test, native temporal/location order, small online batch, ONLINE Tent collapses
  below the frozen source (frozen macro-F1 0.2554 vs Tent 0.0219, CI excludes 0).
  So "KGA beats always-adapt" is locked.  The open question answered here: can a
  LABEL-FREE certificate ROUTER beat BOTH always-online-Tent AND always-freeze, by
  committing an adapter only in windows where a certified benefit lower bound is
  positive at family-wise false-adapt level alpha?

WHAT THIS RUNS, over the SAME native-order OOD stream (reusing the pilot harness):
  1. FROZEN          - source f0, no adapt (always-freeze baseline).
  2. ONLINE-TENT     - one model, state carried across the whole stream (collapses).
  3. EPISODIC-TENT   - reset to f0 every batch, adapt transductively on that batch,
                       predict it (no accumulation).  Reference candidate.
  4. LAME            - Boudiaf et al. CVPR2022, output-only Laplacian-adjusted MLE on
                       frozen f0 features; NO weight updates.  Reference candidate.
  5. KGA-ROUTER      - per window pick among {freeze, episodic-Tent, LAME} the action
                       whose label-free certified benefit lower bound is > 0 at the
                       Bonferroni-corrected level alpha/K (K = #adaptive candidates);
                       else freeze.

REUSE (no committed file modified):
  * f0 loader, native-order stream, RobustReader, macro_f1, OnlineTent, signal helpers
    -> run_iwildcam_streaming_pilot (imported as P).
  * faithful Tent primitives (_clone_for_tta, _entropy, pick_device, mps_free)
    -> tta_methods (imported as tm).
  * the certificate decide() (LOO gradient-boosted B_hat + split-conformal radius)
    -> analysis.decide_kga; the multicandidate Bonferroni correction (alpha/K across
    K candidates) is applied here per docs/research/kbound/theory_v2/
    multicandidate_theorem.tex.

LEAKAGE / HONESTY:
  * OOD cameras are split DISJOINTLY by camera into DEV and TEST (size-alternation,
    label-free).  The certificate is CALIBRATED on DEV windows only (DEV labels used
    only to form the per-window benefit target B for the estimator).  TEST is scored
    once; on TEST the route decision is a function of label-free Z + the DEV-fit
    estimator ONLY.  This is asserted at the decision site (route_decision()).
  * Native order is mandatory and is the default.  Windows never cross a camera.
  * Metric = official macro-F1.  95% CIs by CLUSTER bootstrap over TEST cameras.
  * No fabricated benefit: if the certificate cannot certify benefit > 0, freeze.

Outputs -> results/iwildcam_kga_router/ (JSON + PNG).  Existing results untouched.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_iwildcam_streaming_pilot as P  # pilot harness: loader, stream, OnlineTent, signals
import run_iwildcam_kbound as R           # get_iwildcam, make_model, NUM_CLASSES
import tta_methods as tm                  # faithful Tent primitives
import analysis as an                     # decide_kga certificate (LOO GBR + conformal)

NUM_CLASSES = R.NUM_CLASSES               # 182
DEFAULT_CKPT = REPO / "experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt"
ORDER_FIELDS = P.ORDER_FIELDS             # native-order sort keys


# =========================================================================== LAME
# Laplacian-Adjusted Maximum-likelihood Estimation (Boudiaf, Mueller, Ben Ayed,
# Bertinetto; "Parameter-free Online Test-time Adaptation", CVPR 2022).
# OUTPUT-ONLY: the backbone is FROZEN.  Within a batch we refine the source softmax
# assignments U (N x C) to maximize the LAME objective
#     max_U  <U, log P>  +  <U, W U>  -  <U, log U>
# (unary = source log-probs; pairwise = label smoothness over a kNN affinity W in
# feature space; entropy barrier keeps U on the simplex).  KKT fixed-point:
#     U <- softmax( log P  +  W U )           (row-wise softmax over classes)
# iterated to convergence.  W is a symmetric k-NN affinity built from L2-normalized
# penultimate features (cosine kNN, binary edges, symmetrized).  No weight updates;
# faithful to the paper's kNN-LAME variant.  ~40 lines.
class FeatLogitHead:
    """Frozen ResNet-50 wrapped to return (penultimate_features, logits) in one pass.
    Backbone is f0; nothing is trained.  Penultimate = global-avg-pooled 2048-d vector
    that feeds model.fc (the standard LAME feature space)."""

    def __init__(self, f0: nn.Module):
        self.f0 = f0
        self.fc = f0.fc

    @torch.no_grad()
    def __call__(self, xb: torch.Tensor):
        m = self.f0
        m.eval()
        x = m.conv1(xb); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        x = m.avgpool(x); feat = torch.flatten(x, 1)         # (N, 2048)
        logits = self.fc(feat)
        return feat, logits


@torch.no_grad()
def lame_refine(feat: torch.Tensor, logits: torch.Tensor, knn: int = 5,
                n_iter: int = 100, tol: float = 1e-4) -> torch.Tensor:
    """Return LAME-refined log-probabilities (N x C) for one batch. Frozen backbone.

    feat   : (N, D) penultimate features from the FROZEN f0.
    logits : (N, C) source logits from the FROZEN f0.
    Builds a cosine k-NN affinity W (binary, symmetrized, no self-loop) and runs the
    LAME fixed point U <- softmax(logP + W U).  Label-free (uses no targets)."""
    N = feat.shape[0]
    logP = torch.log_softmax(logits.float(), dim=1)          # unary potentials
    if N <= 2:
        return logP                                          # too few points for kNN
    fn = torch.nn.functional.normalize(feat.float(), dim=1)
    sim = fn @ fn.t()                                        # cosine similarity (N x N)
    sim.fill_diagonal_(-1.0)                                 # exclude self
    k = int(min(knn, N - 1))
    idx = sim.topk(k, dim=1).indices                        # k nearest per row
    W = torch.zeros_like(sim)
    W.scatter_(1, idx, 1.0)                                  # binary kNN graph
    W = ((W + W.t()) > 0).float()                           # symmetrize (union)
    U = logP.exp()                                          # init assignments = source probs
    for _ in range(n_iter):
        U_new = torch.softmax(logP + W @ U, dim=1)
        if (U_new - U).abs().max().item() < tol:
            U = U_new; break
        U = U_new
    return (U + 1e-12).log()                                # refined log-probs


# =============================================================== episodic Tent (reset)
@torch.no_grad()
def _logits_eval(model, xb):
    model.eval()
    return model(xb.contiguous()).detach()


def episodic_tent_logits(f0, xb, steps: int, lr: float):
    """Reset to f0, adapt transductively on THIS batch only (faithful Tent: BN-affine,
    entropy-min, Adam), then return predicted logits for the batch.  No state carried.
    Reuses tm._clone_for_tta / tm._entropy (the project's validated Tent body).
    Returns (logits_np, mean_grad_l2)."""
    m, ps, _ = tm._clone_for_tta(f0)            # train(); BN affine trainable; running stats off
    opt = torch.optim.Adam(ps, lr=lr)
    gnorms = []
    for _ in range(max(1, steps)):
        m.train()
        out = m(xb.contiguous())
        loss = tm._entropy(out.softmax(1)).mean()
        opt.zero_grad(); loss.backward()
        g2 = 0.0
        for p in ps:
            if p.grad is not None:
                g2 += float((p.grad.detach() ** 2).sum())
        gnorms.append(g2 ** 0.5)
        opt.step()
    with torch.no_grad():
        m.train()                                # predict in train mode (batch BN stats), as in pilot
        logits = m(xb.contiguous()).detach()
    return logits.cpu().numpy(), float(np.mean(gnorms) if gnorms else 0.0)


# ============================================================== native camera stream
def build_camera_stream(data_root, split, seed):
    """Native-order stream with per-sample camera id, reusing the pilot's loader/order.
    Returns (ds, sub, y_ord[int], cam_ord[int]).  Order = (location, timestamp, sequence)
    so the within-camera stream is temporally correlated (the collapse-prone regime).
    Cameras are contiguous because 'location' is the PRIMARY sort key."""
    ds, sub, y, _ = R.get_iwildcam(data_root, split, train_tf=False)
    idx = np.asarray(sub.indices)
    md = ds.metadata_array[idx].numpy()
    fi = {f: i for i, f in enumerate(ds.metadata_fields)}
    keys = [md[:, fi[f]] for f in ORDER_FIELDS if f in fi]
    perm = np.lexsort(tuple(reversed(keys)))         # location primary
    sub.indices = idx[perm]
    y_ord = y[perm].astype(int)
    loc_i = fi["location"]
    cam_ord = md[perm, loc_i].astype(int)
    return ds, sub, y_ord, cam_ord


def dev_test_split(cam_ord):
    """FROZEN pre-registered camera split: order cameras by DESCENDING sample count,
    breaking ties by ASCENDING camera id (fully deterministic, label-free), then
    TEST = even ranks, DEV = odd ranks.  Returns (dev_set, test_set).

    The (count desc, id asc) tie-break is REQUIRED so the split is reproducible even
    when two cameras have equal sample counts (e.g. ids 29 and 76 both have n=20)."""
    cams, counts = np.unique(cam_ord, return_counts=True)
    # lexsort: primary key last -> use (-counts) primary, (cams) secondary tie-break
    order = np.lexsort((cams, -counts))               # count desc, then id asc
    ranked = cams[order]
    test_cams = set(int(c) for c in ranked[0::2])
    dev_cams = set(int(c) for c in ranked[1::2])
    return dev_cams, test_cams


# ===================================================== label-free evidence Z per window
def macro_f1(y_true, preds):
    return float(f1_score(np.asarray(y_true, int), np.asarray(preds, int), average="macro"))


def _softmax_np(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _ent_np(p):
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def _hist_entropy(preds, num_classes):
    c = np.bincount(preds.astype(int), minlength=num_classes).astype(float)
    s = c.sum()
    if s <= 0:
        return 0.0
    p = c / s; nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


# Z feature names (ALL label-free).  Order is the certificate's feature vector.
ZNAMES = [
    "fr_entropy",       # mean softmax entropy of frozen f0 on the window
    "cand_entropy",     # mean softmax entropy of the candidate on the window
    "entropy_drop",     # fr_entropy - cand_entropy
    "fr_conf",          # mean max-prob of frozen
    "cand_conf",        # mean max-prob of candidate
    "conf_drop",        # fr_conf - cand_conf
    "disagreement",     # frac(argmax frozen != argmax candidate)
    "cand_div_H",       # predicted-class histogram entropy of candidate (collapse->0)
    "cand_nuniq_frac",  # #unique predicted classes / window size
    "marginal_KL",      # KL(candidate mean-softmax || frozen mean-softmax)
    "grad_l2",          # candidate adaptation grad L2 (0 for LAME; >0 for episodic-Tent)
]


def evidence_Z(fr_logits, cand_logits, grad_l2, num_classes):
    """Label-free evidence vector from frozen vs candidate logits on one window.
    NO labels are read here."""
    Pf = _softmax_np(fr_logits); Pc = _softmax_np(cand_logits)
    fr_pred = fr_logits.argmax(1); cand_pred = cand_logits.argmax(1)
    fr_ent = float(_ent_np(Pf).mean()); cand_ent = float(_ent_np(Pc).mean())
    fr_conf = float(Pf.max(1).mean()); cand_conf = float(Pc.max(1).mean())
    disagree = float(np.mean(fr_pred != cand_pred))
    div_h = _hist_entropy(cand_pred, num_classes)
    nuniq_frac = float(len(np.unique(cand_pred)) / max(1, len(cand_pred)))
    mf = Pf.mean(0) + 1e-12; mc = Pc.mean(0) + 1e-12
    marg_kl = float(np.sum(mc * (np.log(mc) - np.log(mf))))
    return [fr_ent, cand_ent, fr_ent - cand_ent, fr_conf, cand_conf, fr_conf - cand_conf,
            disagree, div_h, nuniq_frac, marg_kl, float(grad_l2)]


# ===================================================== PASS 1: per-window collection
def collect_windows(args, f0, device):
    """Stream the native-order OOD set ONCE.  For every window (windows never cross a
    camera) record, for each policy, the per-sample predictions + labels + camera, and
    for each adaptive candidate the label-free evidence Z and the window benefit B
    (B = candidate_window_F1 - frozen_window_F1; B uses labels but is ONLY consumed for
    DEV calibration / honest leading-indicator reporting, never for a TEST route).

    ONLINE-TENT carries state across the WHOLE stream (the collapse regime); episodic
    Tent and LAME are per-batch.  Returns a list of window dicts + global meta."""
    ds, sub, y_ord, cam_ord = build_camera_stream(args.data_root, args.split, args.seed)
    dev_cams, test_cams = dev_test_split(cam_ord)
    N_full = len(sub)

    # optional smoke cap: keep the FRONT of the native stream
    n_use = N_full
    if args.frac and 0 < args.frac < 1.0:
        n_use = int(N_full * args.frac)
    if args.max_batches:
        n_use = min(n_use, args.max_batches * args.batch_size)
    n_use = max(args.batch_size, (n_use // args.batch_size) * args.batch_size)
    n_use = min(n_use, N_full)
    use_idx = np.arange(n_use)
    y_use = y_ord[use_idx]; cam_use = cam_ord[use_idx]
    sub_use = P.RobustReader(Subset(sub, use_idx.tolist()))
    n_batches = n_use // args.batch_size
    print(f"[stream] split={args.split} order=native N_full={N_full} using={n_use} "
          f"({n_batches} batches of {args.batch_size})", flush=True)
    print(f"[split] DEV cams={len(dev_cams)} TEST cams={len(test_cams)} "
          f"(disjoint={dev_cams.isdisjoint(test_cams)})", flush=True)

    loader = DataLoader(sub_use, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, drop_last=True, pin_memory=False)
    online = P.OnlineTent(f0, lr=args.lr, steps=args.steps)   # state carried across stream
    head = FeatLogitHead(f0)                                  # frozen feat+logit tap for LAME

    windows = []
    # rolling window buffers (reset on camera change or when window fills)
    buf = _new_buf()
    cur_cam = None
    bi = 0
    t0 = time.time()
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb_np = yb.numpy().astype(int)
        # camera id for this batch: take the stream camera of the batch's first slot.
        # (RobustReader may substitute a temporal neighbour for a rare corrupt file; in
        #  native order that neighbour shares the camera, so batch camera is well-defined.)
        b_cam = int(cam_use[bi * args.batch_size])

        # close the window at a camera boundary BEFORE processing this batch
        if cur_cam is not None and (b_cam != cur_cam or buf["nb"] >= args.window):
            windows.append(_finalize_window(buf, cur_cam, dev_cams, test_cams))
            buf = _new_buf()
        cur_cam = b_cam

        # ---- FROZEN (eval, BN running stats) ----
        f0.eval()
        with torch.no_grad():
            fr_logits = f0(xb.contiguous()).detach()
        fr_logits_np = fr_logits.cpu().numpy()

        # ---- ONLINE-TENT (predict-before-adapt, state carried) ----
        on_logits = online.predict_logits(xb)
        on_grad = online.adapt_step(xb)
        on_logits_np = on_logits.cpu().numpy()

        # ---- EPISODIC-TENT (reset to f0, adapt on this batch only) ----
        ep_logits_np, ep_grad = episodic_tent_logits(f0, xb, args.episodic_steps, args.lr)

        # ---- LAME (frozen backbone, output-only refine) ----
        feat, lame_src_logits = head(xb)
        lame_logits = lame_refine(feat, lame_src_logits, knn=args.lame_knn,
                                  n_iter=args.lame_iter).cpu().numpy()

        # accumulate into the current window buffer
        buf["y"].append(yb_np)
        buf["fr"].append(fr_logits_np.argmax(1))
        buf["on"].append(on_logits_np.argmax(1))
        buf["ep"].append(ep_logits_np.argmax(1))
        buf["la"].append(lame_logits.argmax(1))
        buf["fr_logits"].append(fr_logits_np)
        buf["ep_logits"].append(ep_logits_np)
        buf["la_logits"].append(lame_logits)
        buf["ep_grad"].append(ep_grad)
        buf["nb"] += 1

        bi += 1
        if bi % 100 == 0:
            print(f"  [pass1] batch {bi}/{n_batches}  cam={cur_cam}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        tm.mps_free()

    if buf["nb"] > 0 and cur_cam is not None:
        windows.append(_finalize_window(buf, cur_cam, dev_cams, test_cams))

    meta = {"N_full": int(N_full), "n_used": int(n_use), "n_batches": int(n_batches),
            "dev_cams": sorted(dev_cams), "test_cams": sorted(test_cams),
            "n_windows": len(windows),
            "n_dev_windows": int(sum(w["is_dev"] for w in windows)),
            "n_test_windows": int(sum(w["is_test"] for w in windows)),
            "pass1_wall_sec": round(time.time() - t0, 1)}
    print(f"[pass1] done: {len(windows)} windows "
          f"(DEV={meta['n_dev_windows']} TEST={meta['n_test_windows']}) "
          f"in {meta['pass1_wall_sec']}s", flush=True)
    return windows, meta


def _new_buf():
    return {"y": [], "fr": [], "on": [], "ep": [], "la": [],
            "fr_logits": [], "ep_logits": [], "la_logits": [], "ep_grad": [], "nb": 0}


def _finalize_window(buf, cam, dev_cams, test_cams):
    """Concatenate buffers into a window record: per-policy preds, labels, camera,
    per-candidate label-free Z, and per-window benefit B (labels used only for
    DEV calibration / leading-indicator, NEVER for a TEST route)."""
    y = np.concatenate(buf["y"])
    fr = np.concatenate(buf["fr"]); on = np.concatenate(buf["on"])
    ep = np.concatenate(buf["ep"]); la = np.concatenate(buf["la"])
    fr_logits = np.concatenate(buf["fr_logits"], axis=0)
    ep_logits = np.concatenate(buf["ep_logits"], axis=0)
    la_logits = np.concatenate(buf["la_logits"], axis=0)
    ep_grad = float(np.mean(buf["ep_grad"])) if buf["ep_grad"] else 0.0
    fr_f1 = macro_f1(y, fr)
    rec = {
        "camera": int(cam), "n": int(len(y)), "n_batches": int(buf["nb"]),
        "is_dev": int(cam in dev_cams), "is_test": int(cam in test_cams),
        "y": y.astype(int),
        "preds": {"frozen": fr, "online_tent": on, "episodic_tent": ep, "lame": la},
        # label-free evidence per adaptive candidate
        "Z": {
            "episodic_tent": evidence_Z(fr_logits, ep_logits, ep_grad, NUM_CLASSES),
            "lame": evidence_Z(fr_logits, la_logits, 0.0, NUM_CLASSES),
        },
        # per-window macro-F1 (for calibration target + honest reporting)
        "f1": {"frozen": fr_f1, "episodic_tent": macro_f1(y, ep), "lame": macro_f1(y, la),
               "online_tent": macro_f1(y, on)},
    }
    rec["B"] = {"episodic_tent": rec["f1"]["episodic_tent"] - fr_f1,
                "lame": rec["f1"]["lame"] - fr_f1}
    return rec


# ===================================================== certificate: DEV-fit / TEST-score
from sklearn.ensemble import GradientBoostingRegressor

# defect D10: the certificate radius is the shipped exact-rank rule, not np.quantile.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from kga.certificate import split_conformal_rank_radius as _rank_radius  # noqa: E402

CANDIDATES = ["episodic_tent", "lame"]   # the K adaptive candidates (freeze is the default)


def calibrate_certificate(dev_windows, alpha, seed):
    """Fit, on DEV windows ONLY, a per-candidate benefit certificate:
      - GBR  B_hat(Z)  trained on DEV (Z -> measured per-window benefit B),
      - split-conformal radius eps at the Bonferroni level delta_K = alpha/K, where the
        radius is the EXACT RANK quantile eps = r_(k), k = ceil((n+1)(1 - delta_K)), of the
        DEV leave-one-out |B_hat - B| residuals.

    DEFECT D10.  This line used to be ``float(np.quantile(|B_hat - B|, 1 - delta_K))`` --
    numpy's linearly interpolated quantile, which is not an observed order statistic and
    does not satisfy the finite-sample rank argument.  iWildCam is a promoted panel track,
    so it was the only promoted track still scored under a rule the paper does not declare.
    It now calls the shipped ``kga.certificate.split_conformal_rank_radius``, the same
    function every other track uses.  The DEV/TEST split is genuine, so fix-queue item 4
    (leakage) never applied here; only the rule was wrong.

    Feasibility.  At K = 2 candidates and alpha = 0.10 the Bonferroni level is
    delta_K = 0.05, which needs n >= min_calibration_size(0.05) = 19 DEV windows.  With
    fewer the radius is +inf and every window ABSTAINs -- reported, not clamped.

    The one-sided lower bound is L(Z) = B_hat(Z) - eps.  DEV labels enter ONLY through B.
    Mirrors analysis.decide_kga's estimator/conformal machinery, split across DEV/TEST."""
    K = len(CANDIDATES)
    delta_K = alpha / K
    cert = {"alpha": float(alpha), "K": int(K), "delta_K": float(delta_K),
            "candidates": {}, "znames": ZNAMES}
    for c in CANDIDATES:
        Z = np.array([w["Z"][c] for w in dev_windows], float)
        B = np.array([w["B"][c] for w in dev_windows], float)
        n = len(B)
        # leave-one-out B_hat on DEV for an honest conformal residual
        Bhat_loo = np.zeros(n)
        for i in range(n):
            tr = np.arange(n) != i
            m = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, random_state=seed)
            m.fit(Z[tr], B[tr])
            Bhat_loo[i] = m.predict(Z[i:i + 1])[0]
        eps = float(_rank_radius(np.abs(Bhat_loo - B), delta_K))
        # final estimator fit on ALL DEV windows (used to score TEST)
        full = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                         subsample=0.8, random_state=seed)
        full.fit(Z, B)
        cert["candidates"][c] = {
            "model": full, "eps": eps, "delta_K": float(delta_K),
            "dev_n": int(n), "dev_mean_B": float(B.mean()),
            "dev_frac_B_gt0": float(np.mean(B > 0)),
            "dev_loo_mae": float(np.mean(np.abs(Bhat_loo - B))),
        }
        print(f"  [calib {c:13s}] dev_n={n} eps@a/K={eps:.4f} "
              f"dev_meanB={B.mean():+.4f} dev_frac(B>0)={np.mean(B>0):.3f}", flush=True)
    return cert


def route_decision(cert, window):
    """LABEL-FREE route for ONE window.  Reads ONLY the window's label-free Z and the
    DEV-fit certificate.  Returns (action, info) where action in
    {freeze, episodic_tent, lame}.  *** This function NEVER reads window['y']. ***

    Multicandidate Bonferroni rule (Theorem thm:multicand): each candidate's lower
    bound L_c = B_hat_c(Z) - eps_c is evaluated at the corrected level delta_K=alpha/K;
    the certified-helpful set is S = {c: L_c > 0}; commit argmax_{c in S} L_c; else
    freeze (the always-available zero-benefit action)."""
    lowers = {}
    bhats = {}
    for c in CANDIDATES:
        z = np.asarray(window["Z"][c], float).reshape(1, -1)
        bh = float(cert["candidates"][c]["model"].predict(z)[0])
        eps = cert["candidates"][c]["eps"]
        bhats[c] = bh
        lowers[c] = bh - eps
    S = {c: lowers[c] for c in CANDIDATES if lowers[c] > 0.0}
    if S:
        action = max(S, key=lambda c: S[c])      # arbitrary selector: largest lower bound
    else:
        action = "freeze"
    return action, {"lower_bounds": lowers, "b_hat": bhats,
                    "certified_set": sorted(S.keys()), "action": action}


def apply_router(cert, test_windows):
    """Emit KGA-router predictions per TEST window by routing on label-free Z, then
    splicing in the chosen action's predictions.  Returns (router_preds_per_window,
    decisions)."""
    router_preds = []
    decisions = []
    counts = {"freeze": 0, "episodic_tent": 0, "lame": 0}
    for w in test_windows:
        action, info = route_decision(cert, w)
        counts[action] += 1
        src = "frozen" if action == "freeze" else action
        router_preds.append(w["preds"][src])
        decisions.append({"camera": w["camera"], "n": w["n"], **info})
    return router_preds, decisions, counts


# ===================================================== scoring + camera bootstrap
def _concat_policy(windows, policy_key, router_preds=None):
    """Concatenate per-window predictions + labels into flat arrays for a policy.
    If policy_key == 'kga_router', use router_preds (aligned to windows)."""
    ys, ps = [], []
    for i, w in enumerate(windows):
        ys.append(w["y"])
        if policy_key == "kga_router":
            ps.append(router_preds[i])
        else:
            ps.append(w["preds"][policy_key])
    return np.concatenate(ys).astype(int), np.concatenate(ps).astype(int)


def score_test(test_windows, router_preds):
    """Macro-F1 of every policy on the concatenated TEST predictions."""
    policies = ["frozen", "online_tent", "episodic_tent", "lame", "kga_router"]
    out = {}
    for pol in policies:
        y, p = _concat_policy(test_windows, pol, router_preds)
        out[pol] = macro_f1(y, p)
    return out


def camera_bootstrap(test_windows, router_preds, n_boot, seed):
    """CLUSTER bootstrap over TEST cameras: resample cameras with replacement, pool the
    windows of the drawn cameras, recompute each policy's macro-F1 and the pre-registered
    pairwise deltas.  Returns per-policy CIs and the two decision deltas with CIs."""
    # group window indices by camera
    by_cam = {}
    for i, w in enumerate(test_windows):
        by_cam.setdefault(w["camera"], []).append(i)
    cams = sorted(by_cam.keys())
    rng = np.random.default_rng(seed)
    policies = ["frozen", "online_tent", "episodic_tent", "lame", "kga_router"]

    def f1_on(idxs, pol):
        ys, ps = [], []
        for i in idxs:
            w = test_windows[i]
            ys.append(w["y"])
            ps.append(router_preds[i] if pol == "kga_router" else w["preds"][pol])
        y = np.concatenate(ys).astype(int); p = np.concatenate(ps).astype(int)
        return macro_f1(y, p)

    boot = {pol: np.empty(n_boot) for pol in policies}
    d1 = np.empty(n_boot)   # kga - online_tent
    d2 = np.empty(n_boot)   # kga - frozen
    for b in range(n_boot):
        draw = rng.choice(cams, size=len(cams), replace=True)
        idxs = []
        for c in draw:
            idxs.extend(by_cam[c])
        for pol in policies:
            boot[pol][b] = f1_on(idxs, pol)
        d1[b] = boot["kga_router"][b] - boot["online_tent"][b]
        d2[b] = boot["kga_router"][b] - boot["frozen"][b]

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi),
                "excludes_zero": bool(hi < 0 or lo > 0)}

    return {
        "n_boot": int(n_boot), "n_test_cameras": int(len(cams)),
        "per_policy_ci": {pol: ci(boot[pol]) for pol in policies},
        "delta_kga_minus_online_tent": ci(d1),
        "delta_kga_minus_frozen": ci(d2),
    }


def verdict(boot):
    """Pre-registered verdict (PREREG.md section 8)."""
    d1 = boot["delta_kga_minus_online_tent"]
    d2 = boot["delta_kga_minus_frozen"]
    beats_online = d1["lo"] > 0
    beats_freeze = d2["lo"] > 0
    ties_freeze = (d2["lo"] <= 0 <= d2["hi"])
    harms_freeze = d2["hi"] < 0
    if beats_freeze and beats_online:
        v = "BEATS-BOTH"
    elif ties_freeze and beats_online:
        v = "NO-HARM"
    elif harms_freeze:
        v = "HARM"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "beats_online_tent": bool(beats_online),
            "beats_freeze": bool(beats_freeze), "ties_freeze": bool(ties_freeze),
            "harms_freeze": bool(harms_freeze)}


def leading_indicator(windows, tag):
    """Honest: does episodic-Tent or LAME EVER beat frozen on any window? Fraction of
    windows where each candidate's per-window benefit B > 0 (and > a 0.005 margin)."""
    out = {"tag": tag, "n_windows": len(windows)}
    for c in CANDIDATES:
        B = np.array([w["B"][c] for w in windows], float)
        out[c] = {
            "ever_beats_frozen": bool(np.any(B > 0)),
            "frac_windows_B_gt0": float(np.mean(B > 0)),
            "frac_windows_B_gt_margin": float(np.mean(B > 0.005)),
            "max_window_B": float(B.max()) if len(B) else 0.0,
            "mean_window_B": float(B.mean()) if len(B) else 0.0,
            "n_windows_B_gt0": int(np.sum(B > 0)),
        }
    return out


# ===================================================== plotting
def make_plot(result, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = result["test_scores"]
    boot = result["bootstrap"]
    pols = ["frozen", "online_tent", "episodic_tent", "lame", "kga_router"]
    labels = ["frozen", "online-Tent", "episodic-Tent", "LAME", "KGA-router"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"iWildCam KGA router (TEST OOD, native order, bs={result['config']['batch_size']}) "
        f"-> {result['verdict']['verdict']}", fontsize=12)

    # (1) policy macro-F1 with camera-bootstrap CIs
    ax = axes[0, 0]
    means = [sc[p] for p in pols]
    los = [boot["per_policy_ci"][p]["lo"] for p in pols]
    his = [boot["per_policy_ci"][p]["hi"] for p in pols]
    yerr = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
    ax.bar(range(len(pols)), means, color=colors, yerr=yerr, capsize=4)
    ax.set_xticks(range(len(pols))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("TEST macro-F1"); ax.set_title("Policy macro-F1 (camera-bootstrap 95% CI)")
    ax.grid(alpha=0.3, axis="y")
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    # (2) the two decision deltas with CIs
    ax = axes[0, 1]
    d1 = boot["delta_kga_minus_online_tent"]; d2 = boot["delta_kga_minus_frozen"]
    ds = [d1, d2]; dl = ["KGA - online-Tent", "KGA - freeze"]
    m = [d["mean"] for d in ds]
    yerr = [[d["mean"] - d["lo"] for d in ds], [d["hi"] - d["mean"] for d in ds]]
    ax.bar(range(2), m, color=["#d62728", "#1f77b4"], yerr=yerr, capsize=5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(2)); ax.set_xticklabels(dl, rotation=10, ha="right")
    ax.set_ylabel("delta macro-F1"); ax.set_title("Pre-registered decision deltas (95% CI)")
    ax.grid(alpha=0.3, axis="y")

    # (3) leading indicator: per-window B>0 fractions, DEV vs TEST
    ax = axes[1, 0]
    li_dev = result["leading_indicator"]["dev"]; li_test = result["leading_indicator"]["test"]
    x = np.arange(len(CANDIDATES)); w = 0.35
    dev_f = [li_dev[c]["frac_windows_B_gt0"] for c in CANDIDATES]
    test_f = [li_test[c]["frac_windows_B_gt0"] for c in CANDIDATES]
    ax.bar(x - w / 2, dev_f, w, label="DEV", color="#8c564b")
    ax.bar(x + w / 2, test_f, w, label="TEST", color="#e377c2")
    ax.set_xticks(x); ax.set_xticklabels(CANDIDATES, rotation=10)
    ax.set_ylabel("frac windows beating frozen (B>0)")
    ax.set_title("Leading indicator: does any candidate beat frozen per window?")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # (4) router decision mix on TEST
    ax = axes[1, 1]
    rc = result["router_decision_counts"]
    keys = ["freeze", "episodic_tent", "lame"]
    vals = [rc.get(k, 0) for k in keys]
    ax.bar(range(3), vals, color=["#1f77b4", "#2ca02c", "#9467bd"])
    ax.set_xticks(range(3)); ax.set_xticklabels(["freeze", "episodic-Tent", "LAME"])
    ax.set_ylabel("# TEST windows"); ax.set_title("KGA-router action mix on TEST windows")
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


# ===================================================== orchestrator
def run(args):
    t_run0 = time.time()
    device = tm.pick_device(args.device)
    out_dir = Path(args.results_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"source model not found: {ckpt}")
    f0, backbone, ck_meta = P.load_f0(ckpt, device)
    print(f"[f0] loaded {ckpt.name} backbone={backbone} meta={ck_meta} device={device}", flush=True)

    # PASS 1: collect per-window predictions + label-free Z over the native stream
    windows, meta = collect_windows(args, f0, device)
    dev_windows = [w for w in windows if w["is_dev"]]
    test_windows = [w for w in windows if w["is_test"]]
    if len(dev_windows) < 4:
        raise RuntimeError(f"need >=4 DEV windows to calibrate; got {len(dev_windows)}")
    if len(test_windows) < 1:
        raise RuntimeError("no TEST windows")

    # CALIBRATE certificate on DEV ONLY
    print("[calib] fitting per-candidate certificate on DEV windows (alpha/K Bonferroni)", flush=True)
    cert = calibrate_certificate(dev_windows, alpha=args.alpha, seed=args.seed)

    # ROUTE on TEST (label-free), then SCORE once
    router_preds, decisions, route_counts = apply_router(cert, test_windows)
    test_scores = score_test(test_windows, router_preds)
    boot = camera_bootstrap(test_windows, router_preds, n_boot=args.n_boot, seed=args.seed)
    vd = verdict(boot)
    li = {"dev": leading_indicator(dev_windows, "dev"),
          "test": leading_indicator(test_windows, "test")}

    # serialization-safe certificate summary (drop sklearn model objects)
    cert_summary = {"alpha": cert["alpha"], "K": cert["K"], "delta_K": cert["delta_K"],
                    "znames": cert["znames"],
                    "candidates": {c: {k: v for k, v in cert["candidates"][c].items()
                                       if k != "model"} for c in CANDIDATES}}

    result = {
        "schema": "iwildcam_kga_router_v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"node": platform.node(), "platform": platform.platform(),
                 "torch": torch.__version__, "device": str(device)},
        "config": vars(args),
        "source_model": {"ckpt": str(ckpt), "backbone": backbone, **ck_meta},
        "stream_meta": meta,
        "metric": "macro_f1",
        "policies": ["frozen", "online_tent", "episodic_tent", "lame", "kga_router"],
        "candidates": CANDIDATES,
        "certificate": cert_summary,
        "test_scores": test_scores,
        "bootstrap": boot,
        "verdict": vd,
        "leading_indicator": li,
        "router_decision_counts": route_counts,
        "router_decisions": decisions,
        "route_signal_doc": (
            "route_decision() reads ONLY label-free Z (entropy, predicted-class diversity, "
            "confidence drop, frozen-vs-candidate disagreement, marginal-KL, episodic grad L2) "
            "and the DEV-fit GBR + DEV split-conformal radius at level alpha/K. It never reads "
            "TEST labels. Bonferroni multicandidate correction per thm:multicand."),
        "wall_sec": round(time.time() - t_run0, 1),
    }

    tag = "smoke" if (args.max_batches or args.frac) else "full"
    out_json = out_dir / f"router_{args.split}_bs{args.batch_size}_{tag}.json"
    with out_json.open("w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print("\n" + "=" * 78, flush=True)
    print(f"[TEST macro-F1] frozen={test_scores['frozen']:.4f} "
          f"online_tent={test_scores['online_tent']:.4f} "
          f"episodic_tent={test_scores['episodic_tent']:.4f} "
          f"lame={test_scores['lame']:.4f} KGA={test_scores['kga_router']:.4f}", flush=True)
    print(f"[delta] KGA-online_tent={boot['delta_kga_minus_online_tent']['mean']:+.4f} "
          f"CI[{boot['delta_kga_minus_online_tent']['lo']:+.4f},"
          f"{boot['delta_kga_minus_online_tent']['hi']:+.4f}] "
          f"excl0={boot['delta_kga_minus_online_tent']['excludes_zero']}", flush=True)
    print(f"[delta] KGA-frozen={boot['delta_kga_minus_frozen']['mean']:+.4f} "
          f"CI[{boot['delta_kga_minus_frozen']['lo']:+.4f},"
          f"{boot['delta_kga_minus_frozen']['hi']:+.4f}] "
          f"excl0={boot['delta_kga_minus_frozen']['excludes_zero']}", flush=True)
    print(f"[VERDICT] {vd['verdict']}", flush=True)
    print(f"[router mix on TEST] {route_counts}", flush=True)
    print(f"[leading-indicator DEV] " + ", ".join(
        f"{c}: ever_beats={li['dev'][c]['ever_beats_frozen']} "
        f"frac(B>0)={li['dev'][c]['frac_windows_B_gt0']:.3f}" for c in CANDIDATES), flush=True)
    print(f"[leading-indicator TEST] " + ", ".join(
        f"{c}: ever_beats={li['test'][c]['ever_beats_frozen']} "
        f"frac(B>0)={li['test'][c]['frac_windows_B_gt0']:.3f}" for c in CANDIDATES), flush=True)
    print(f"[json] -> {out_json}", flush=True)

    png = out_dir / f"router_{args.split}_bs{args.batch_size}_{tag}.png"
    try:
        make_plot(result, png)
        print(f"[png]  -> {png}", flush=True)
    except Exception as e:
        print(f"[png]  FAILED: {repr(e)[:160]}", flush=True)
    return result, out_json


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="iWildCam KGA multicandidate router (ceiling-break)")
    p.add_argument("--data-root", default=str(REPO / "experiments/kbound/data/wilds"))
    p.add_argument("--results-root", default=str(REPO / "experiments/kbound/results"))
    p.add_argument("--run-name", default="iwildcam_kga_router")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--window", type=int, default=10, help="batches per routing window")
    p.add_argument("--lr", type=float, default=1e-3, help="Tent LR (Adam, BN-affine)")
    p.add_argument("--steps", type=int, default=1, help="online-Tent steps/batch")
    p.add_argument("--episodic-steps", type=int, default=3, dest="episodic_steps",
                   help="episodic-Tent steps per batch (reset each batch)")
    p.add_argument("--lame-knn", type=int, default=5, dest="lame_knn")
    p.add_argument("--lame-iter", type=int, default=100, dest="lame_iter")
    p.add_argument("--alpha", type=float, default=0.10, help="family-wise false-adapt level")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=1000, dest="n_boot")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    # smoke knobs
    p.add_argument("--max-batches", type=int, default=0, dest="max_batches")
    p.add_argument("--frac", type=float, default=0.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
