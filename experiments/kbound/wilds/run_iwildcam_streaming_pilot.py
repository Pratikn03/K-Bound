"""
run_iwildcam_streaming_pilot.py - DECISIVE pilot: does naive ONLINE Tent COLLAPSE
below the frozen ERM source on iWildCam's held-out (OOD) cameras when the stream is
served in NATIVE temporal/location order with a small online batch?

This is a falsification / de-risking pilot for the K-Bound project.  It does NOT
reinvent any modeling component:
  * the frozen ERM source model f0 is the existing ResNet-50 checkpoint trained by
    train_iwildcam_f0.py (results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt);
  * the iWildCam loader / disk-present filter / transforms are reused from
    run_iwildcam_kbound.py;
  * the Tent update is the SAME faithful entropy-minimization BN-affine adaptation as
    tta_methods.tent_adapt (_clone_for_tta + _bn_affine_params + _entropy), only
    re-expressed as a persistent ONLINE adapter so state is carried across the stream
    and we can predict-before-adapt and log per-window label-free signals.

Two policies are run over the SAME native-order stream:
  (a) FROZEN  : f0 in eval mode, predict each batch, no adaptation.
  (b) ONLINE TENT : one model adapted across the whole stream (state carried),
                    1 gradient step / batch, predict-before-adapt.

Metrics (iWildCam official): macro-F1 via sklearn f1_score(average="macro"),
cumulative and per-window.  For the Tent run we additionally log per-window
LABEL-FREE signals (KGA evidence Z / collapse detector):
  * mean softmax entropy of predictions,
  * entropy of the predicted-class histogram (diversity; collapse -> few classes),
  * mean gradient L2-norm of the Tent step.

INTEGRITY: native order is the default and is mandatory for the real claim; --order
shuffled is only a control.  Every adapted prediction comes from a real update.
Nothing is fabricated.  Existing committed results are never touched: outputs go to
results/iwildcam_streaming_pilot/.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # partial extraction left some truncated JPEGs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_iwildcam_kbound as R  # reuse loader, transforms, disk-present filter, make_model
import tta_methods as tm  # reuse pick_device, _entropy, _clone_for_tta primitives

NUM_CLASSES = R.NUM_CLASSES  # 182
DEFAULT_CKPT = REPO / "experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt"

# native-order sort keys, in priority order; all are present in WILDS metadata_fields
ORDER_FIELDS = ["location", "year", "month", "day", "hour", "minute", "second", "sequence"]


# --------------------------------------------------------------------------- model
def load_f0(ckpt: Path, device: torch.device):
    """Load the existing ERM ResNet-50 source model. Format: {'model': state_dict, ...}."""
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    backbone = obj.get("backbone", "resnet50") if isinstance(obj, dict) else "resnet50"
    model = R.make_model(backbone, device)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state, strict=True)
    model.eval()
    meta = {k: obj[k] for k in obj if k != "model"} if isinstance(obj, dict) else {}
    return model, backbone, meta


# ----------------------------------------------------------------------- stream order
def build_native_stream(data_root: str, split: str, order: str, seed: int):
    """Return (subset, y[int], order_index) with the subset re-indexed into NATIVE order.

    Native order = sort the disk-present OOD samples by (location, timestamp, sequence),
    so consecutive batches are temporally & location correlated (the collapse-prone
    regime).  'shuffled' is a control: a fixed random permutation of the same samples.
    Reuses R.get_iwildcam, which already disk-present-filters the split.
    """
    ds, sub, y, _ = R.get_iwildcam(data_root, split, train_tf=False)
    idx = np.asarray(sub.indices)  # disk-present-filtered via R's in-memory present cache
    # R's present cache can be slightly stale (lists a few files no longer on disk).
    # We do NOT stat every file (that is too slow on the external volume); instead the
    # DataLoader is wrapped in RobustReader below, which substitutes the next decodable
    # neighbour for the rare missing/corrupt file.  Labels are read from the loader batch
    # so alignment is preserved, and in native order a neighbour is the same location/time.
    md = ds.metadata_array[idx].numpy()
    fi = {f: i for i, f in enumerate(ds.metadata_fields)}
    if order == "native":
        keys = [md[:, fi[f]] for f in ORDER_FIELDS if f in fi]
        # lexsort: last key is primary -> reverse so location is primary
        perm = np.lexsort(tuple(reversed(keys)))
    elif order == "shuffled":
        perm = np.random.default_rng(seed).permutation(len(idx))
    else:
        raise ValueError(f"unknown order {order}")
    sub.indices = idx[perm]
    y_ord = y[perm].astype(int)
    return ds, sub, y_ord, perm


class RobustReader(torch.utils.data.Dataset):
    """Order-preserving robust wrapper: if sample i is missing/corrupt, substitute the
    next decodable sample (i+1, i+2, ...).  Returns its (x, y, metadata), so labels stay
    consistent with the substituted image.  In native order the substitute is a temporal
    neighbour (same camera/time window), so the stream's correlation structure is kept."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        n = len(self.base)
        for k in range(16):
            try:
                return self.base[(i + k) % n]
            except Exception:
                continue
        raise RuntimeError("too many consecutive unreadable images")


# --------------------------------------------------------------------- online Tent
class OnlineTent:
    """Persistent Tent adapter: ONE model, BN-affine params, Adam, state carried across
    the whole stream (the collapse-prone online mode).  Uses the EXACT faithful Tent
    primitives from tta_methods (_clone_for_tta -> train mode, BN running stats off,
    only BN/LN affine trainable; entropy-min loss; Adam).  Per batch we predict BEFORE
    the update (so frozen and Tent see the same inputs), then take 1 entropy step and
    record the gradient L2-norm.
    """

    def __init__(self, f0, lr, steps=1):
        self.m, self.ps, _ = tm._clone_for_tta(f0)  # train(); BN affine trainable; running stats off
        self.opt = torch.optim.Adam(self.ps, lr=lr)
        self.steps = steps

    @torch.no_grad()
    def predict_logits(self, xb):
        # train mode is required: BN uses batch stats (that is what online Tent acts on)
        self.m.train()
        return self.m(xb.contiguous()).detach()

    def adapt_step(self, xb):
        """One (or `steps`) entropy-minimization step(s) on this batch; return mean grad L2."""
        gnorms = []
        for _ in range(self.steps):
            self.m.train()
            out = self.m(xb.contiguous())
            loss = tm._entropy(out.softmax(1)).mean()
            self.opt.zero_grad()
            loss.backward()
            g2 = 0.0
            for p in self.ps:
                if p.grad is not None:
                    g2 += float((p.grad.detach() ** 2).sum())
            gnorms.append(g2 ** 0.5)
            self.opt.step()
        return float(np.mean(gnorms)) if gnorms else 0.0


# --------------------------------------------------------------------- signals
def _entropy_np_from_logits(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def _pred_hist_entropy(preds: np.ndarray, num_classes: int) -> float:
    """Entropy (nats) of the predicted-class histogram -> diversity. Collapse -> ~0."""
    c = np.bincount(preds.astype(int), minlength=num_classes).astype(float)
    s = c.sum()
    if s <= 0:
        return 0.0
    p = c / s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def macro_f1(y_true, preds) -> float:
    return float(f1_score(np.asarray(y_true, int), np.asarray(preds, int), average="macro"))


# --------------------------------------------------------------------- main run
def run(args):
    device = tm.pick_device(args.device)
    out_dir = Path(args.results_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"source model not found: {ckpt}")
    f0, backbone, ck_meta = load_f0(ckpt, device)
    print(f"[f0] loaded {ckpt.name} backbone={backbone} meta={ck_meta}", flush=True)

    ds, sub, y_ord, perm = build_native_stream(args.data_root, args.split, args.order, args.seed)
    N_full = len(sub)
    print(f"[stream] split={args.split} order={args.order} N={N_full} "
          f"locations={len(np.unique(ds.metadata_array[np.asarray(sub.indices)][:, ds.metadata_fields.index('location')].numpy()))} "
          f"classes={len(np.unique(y_ord))}", flush=True)

    # cap (smoke) -------------------------------------------------------------
    n_use = N_full
    if args.frac and 0 < args.frac < 1.0:
        n_use = int(N_full * args.frac)
    if args.max_batches:
        n_use = min(n_use, args.max_batches * args.batch_size)
    n_use = max(args.batch_size, (n_use // args.batch_size) * args.batch_size)
    n_use = min(n_use, N_full)
    use_idx = np.arange(n_use)  # already in native order; keep the front of the stream
    sub_use = RobustReader(Subset(sub, use_idx.tolist()))  # order-preserving robust read
    n_batches = n_use // args.batch_size
    print(f"[stream] using {n_use} samples = {n_batches} batches of {args.batch_size}", flush=True)

    # DataLoader MUST NOT shuffle: native order is carried by sub.indices order ------
    loader = DataLoader(sub_use, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, drop_last=True, pin_memory=False)

    tent = OnlineTent(f0, lr=args.lr, steps=args.steps)

    # per-batch accumulators
    fr_preds_all, te_preds_all, y_all = [], [], []
    win = args.window  # batches per window
    windows = []  # list of dicts
    # cumulative macro-F1 trace recorded at each window boundary
    cum_trace = {"batch": [], "frozen_cum_f1": [], "tent_cum_f1": []}

    # rolling window buffers
    w_fr, w_te, w_y = [], [], []
    w_ent, w_grad, w_tepred = [], [], []

    t0 = time.time()
    bi = 0
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb_np = yb.numpy().astype(int)

        # (a) FROZEN: eval mode, BN running stats, no grad
        f0.eval()
        with torch.no_grad():
            fr_logits = f0(xb.contiguous()).detach()
        fr_pred = fr_logits.argmax(1).cpu().numpy()

        # (b) ONLINE TENT: predict-before-adapt (same inputs as frozen), then 1 step
        te_logits = tent.predict_logits(xb)
        te_pred = te_logits.argmax(1).cpu().numpy()
        grad = tent.adapt_step(xb)  # carries state forward

        # label-free signals from Tent's pre-update prediction on this batch
        te_logits_np = te_logits.cpu().numpy()
        ent_mean = float(_entropy_np_from_logits(te_logits_np).mean())

        fr_preds_all.append(fr_pred); te_preds_all.append(te_pred); y_all.append(yb_np)
        w_fr.append(fr_pred); w_te.append(te_pred); w_y.append(yb_np)
        w_ent.append(ent_mean); w_grad.append(grad); w_tepred.append(te_pred)

        bi += 1
        if bi % win == 0 or bi == n_batches:
            wy = np.concatenate(w_y); wfr = np.concatenate(w_fr); wte = np.concatenate(w_te)
            wtp = np.concatenate(w_tepred)
            rec = {
                "window_end_batch": bi,
                "n": int(len(wy)),
                "frozen_window_f1": macro_f1(wy, wfr),
                "tent_window_f1": macro_f1(wy, wte),
                "frozen_window_acc": float((wfr == wy).mean()),
                "tent_window_acc": float((wte == wy).mean()),
                # label-free signals (Tent)
                "tent_mean_entropy": float(np.mean(w_ent)),
                "tent_pred_hist_entropy": _pred_hist_entropy(wtp, NUM_CLASSES),
                "tent_pred_n_unique": int(len(np.unique(wtp))),
                "tent_mean_grad_l2": float(np.mean(w_grad)),
            }
            # cumulative to this point
            cy = np.concatenate(y_all); cfr = np.concatenate(fr_preds_all); cte = np.concatenate(te_preds_all)
            rec["frozen_cum_f1"] = macro_f1(cy, cfr)
            rec["tent_cum_f1"] = macro_f1(cy, cte)
            windows.append(rec)
            cum_trace["batch"].append(bi)
            cum_trace["frozen_cum_f1"].append(rec["frozen_cum_f1"])
            cum_trace["tent_cum_f1"].append(rec["tent_cum_f1"])
            print(f"  [win @ b{bi:>4}] froz_f1={rec['frozen_window_f1']:.3f} "
                  f"tent_f1={rec['tent_window_f1']:.3f} | cum froz={rec['frozen_cum_f1']:.3f} "
                  f"tent={rec['tent_cum_f1']:.3f} | ent={rec['tent_mean_entropy']:.3f} "
                  f"divH={rec['tent_pred_hist_entropy']:.3f} nuniq={rec['tent_pred_n_unique']} "
                  f"grad={rec['tent_mean_grad_l2']:.3f}", flush=True)
            w_fr, w_te, w_y, w_ent, w_grad, w_tepred = [], [], [], [], [], []
        tm.mps_free()

    # final cumulative ---------------------------------------------------------
    y_all = np.concatenate(y_all)
    fr_all = np.concatenate(fr_preds_all)
    te_all = np.concatenate(te_preds_all)
    frozen_f1 = macro_f1(y_all, fr_all)
    tent_f1 = macro_f1(y_all, te_all)
    delta = tent_f1 - frozen_f1

    # paired bootstrap CI on (tent - frozen) cumulative macro-F1 over samples ----
    boot = bootstrap_delta_ci(y_all, fr_all, te_all, n_boot=args.n_boot, seed=args.seed)

    wall = time.time() - t0
    result = {
        "schema": "iwildcam_streaming_pilot_v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {
            "node": platform.node(), "platform": platform.platform(),
            "torch": torch.__version__, "mps": bool(torch.backends.mps.is_available()),
            "device": str(device),
        },
        "config": vars(args),
        "source_model": {"ckpt": str(ckpt), "backbone": backbone, **ck_meta},
        "stream": {
            "split": args.split, "order": args.order, "N_full": int(N_full),
            "n_used": int(n_use), "n_batches": int(n_batches), "batch_size": int(args.batch_size),
        },
        "metric": "macro_f1",
        "frozen_cum_macro_f1": frozen_f1,
        "tent_cum_macro_f1": tent_f1,
        "delta_tent_minus_frozen": delta,
        "collapse_observed": bool(delta < 0),
        "bootstrap_delta_ci": boot,
        "frozen_cum_acc": float((fr_all == y_all).mean()),
        "tent_cum_acc": float((te_all == y_all).mean()),
        "windows": windows,
        "cum_trace": cum_trace,
        "wall_sec": round(wall, 1),
    }
    out_json = out_dir / f"pilot_{args.split}_{args.order}_bs{args.batch_size}.json"
    with out_json.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[RESULT] frozen_macroF1={frozen_f1:.4f} tent_macroF1={tent_f1:.4f} "
          f"delta(tent-frozen)={delta:+.4f} CI95=[{boot['lo']:+.4f},{boot['hi']:+.4f}] "
          f"collapse={'YES' if delta<0 else 'no'} (CI excludes 0: {boot['excludes_zero']})",
          flush=True)
    print(f"[json] -> {out_json}", flush=True)

    png = out_dir / f"pilot_{args.split}_{args.order}_bs{args.batch_size}.png"
    try:
        make_plot(result, png)
        print(f"[png]  -> {png}", flush=True)
    except Exception as e:
        print(f"[png]  FAILED: {repr(e)[:160]}", flush=True)
    return result, out_json, png


def bootstrap_delta_ci(y, fr_pred, te_pred, n_boot=1000, seed=0):
    """Paired bootstrap over samples: resample indices, recompute (tent-frozen) macro-F1.

    Macro-F1 is non-linear in samples; resampling samples gives an honest CI on the
    cumulative-metric gap.  Returns the 2.5/97.5 percentile interval and whether it
    excludes 0 (the decision quantity in PREREG.md)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot, float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        deltas[b] = (f1_score(yb, te_pred[idx], average="macro")
                     - f1_score(yb, fr_pred[idx], average="macro"))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_boot": int(n_boot),
        "mean": float(deltas.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "excludes_zero": bool(hi < 0 or lo > 0),
        "p_delta_lt_0": float(np.mean(deltas < 0)),
    }


def make_plot(result, png_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct = result["cum_trace"]
    wins = result["windows"]
    bx = ct["batch"]
    we = [w["window_end_batch"] for w in wins]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"iWildCam streaming pilot ({result['stream']['split']} OOD, order={result['stream']['order']}, "
        f"bs={result['stream']['batch_size']})  delta={result['delta_tent_minus_frozen']:+.4f}",
        fontsize=11,
    )

    # (1) cumulative macro-F1: frozen vs Tent
    ax = axes[0, 0]
    ax.plot(bx, ct["frozen_cum_f1"], "-o", ms=3, label="frozen (source)", color="#1f77b4")
    ax.plot(bx, ct["tent_cum_f1"], "-s", ms=3, label="Tent online", color="#d62728")
    ax.set_xlabel("stream position (batch)"); ax.set_ylabel("cumulative macro-F1")
    ax.set_title("Cumulative macro-F1 vs stream position"); ax.legend(); ax.grid(alpha=0.3)

    # (2) per-window macro-F1
    ax = axes[0, 1]
    ax.plot(we, [w["frozen_window_f1"] for w in wins], "-o", ms=3, label="frozen", color="#1f77b4")
    ax.plot(we, [w["tent_window_f1"] for w in wins], "-s", ms=3, label="Tent", color="#d62728")
    ax.set_xlabel("stream position (batch)"); ax.set_ylabel("per-window macro-F1")
    ax.set_title("Per-window macro-F1"); ax.legend(); ax.grid(alpha=0.3)

    # (3) label-free collapse signals: entropy + diversity
    ax = axes[1, 0]
    ax.plot(we, [w["tent_mean_entropy"] for w in wins], "-o", ms=3, label="mean pred entropy", color="#9467bd")
    ax.set_xlabel("stream position (batch)"); ax.set_ylabel("mean softmax entropy", color="#9467bd")
    ax.tick_params(axis="y", labelcolor="#9467bd")
    ax2 = ax.twinx()
    ax2.plot(we, [w["tent_pred_hist_entropy"] for w in wins], "-^", ms=3,
             label="pred-class diversity (H)", color="#2ca02c")
    ax2.set_ylabel("predicted-class histogram entropy", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax.set_title("Label-free collapse signals (Tent)"); ax.grid(alpha=0.3)

    # (4) gradient norm
    ax = axes[1, 1]
    ax.plot(we, [w["tent_mean_grad_l2"] for w in wins], "-o", ms=3, color="#ff7f0e")
    ax.set_xlabel("stream position (batch)"); ax.set_ylabel("mean Tent grad L2-norm")
    ax.set_title("Tent gradient L2-norm per window"); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="iWildCam native-order online-Tent collapse pilot")
    p.add_argument("--data-root", default=str(REPO / "experiments/kbound/data/wilds"))
    p.add_argument("--results-root", default=str(REPO / "experiments/kbound/results"))
    p.add_argument("--run-name", default="iwildcam_streaming_pilot")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--split", default="test", choices=["val", "test"],
                   help="OOD stream split (test = Test OOD/Trans; val = Validation OOD/Trans)")
    p.add_argument("--order", default="native", choices=["native", "shuffled"],
                   help="native = (location,timestamp,sequence) order [mandatory for the claim]; "
                        "shuffled = control")
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3, help="Tent learning rate (Adam, BN-affine)")
    p.add_argument("--steps", type=int, default=1, help="Tent gradient steps per batch")
    p.add_argument("--window", type=int, default=50, help="batches per reporting window")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=1000, dest="n_boot")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    # smoke knobs
    p.add_argument("--max-batches", type=int, default=0, dest="max_batches",
                   help="cap stream to this many batches (smoke)")
    p.add_argument("--frac", type=float, default=0.0, help="use only this fraction of the stream (smoke)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
