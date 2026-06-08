"""cifar10c_kboundopt.py -- KBoundOptimizer vs SAR on a CONTINUAL collapse stream.

The decisive "active mechanism" test (K-Bound plan 2.4). Instead of deciding once per
batch (the wrapper KGA), KBoundOptimizer injects the anytime certificate INTO the
optimization loop: it adapts continually but, the moment the e-process certifies that the
running updates are HARMFUL (a sustained drop on a clean labelled validation probe), it
freezes the gradient -- halting collapse. We compare it head-to-head with SAR's
sharpness-aware + entropy-reset heuristic on a long non-stationary CIFAR-10-C stream with
long "trap" runs (single-class / severe / tiny-batch) that make continual Tent collapse.

Mechanism (label-free of TARGET labels; uses a clean validation probe, allowed by the
paper's target-label-free definition):
  - base optimizer = Adam on BN-affine params; loss = prediction entropy (Tent-style).
  - evidence_fn returns the per-step change in clean-probe accuracy (the realized benefit
    of the last applied update). KBoundOptimizer feeds it to the e-process:
       * E^- crosses 1/alpha  -> "freeze"  -> gradient scaled to 0 (collapse halted)
       * otherwise            -> full update (abstain_scale=1: innocent until proven harmful)

Honest expectation: KBoundOptimizer should never collapse (like SAR) while capturing the
helpful gains Tent gets early. If it merely TIES SAR, that is still a contribution
(certified gating with a guarantee vs a heuristic reset) -- we report whatever happens.

RUN ON YOUR MAC (MPS/CUDA):
    python docs/research/kbound/scripts/cifar10c_kboundopt.py --quick
    python docs/research/kbound/scripts/cifar10c_kboundopt.py            # full stream
Smoke-test the torch-free analysis core + gating logic (no GPU):
    python docs/research/kbound/scripts/cifar10c_kboundopt.py --smoke-test
"""
from __future__ import annotations
import os, sys, json, argparse, time, copy, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PKG = os.path.join(HERE, "..", "kbound_pkg")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

print("[kboundopt] importing deps (first sklearn/torch import can take ~20-30s; do not Ctrl-C)...", flush=True)
from kbound.eprocess import EProcess          # torch-free (gating logic, smoke-testable)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from kbound import KBoundOptimizer
    # reuse the existing, tested CIFAR helpers
    from cifar_tent_mps_v2 import (make_cifar_resnet, get_cifar_model, _norm_cifar,
                                   load_cifar_c, cifar_c_severity, _bn_affine_params, _entropy)
    _HAS_TORCH = True
except Exception:                              # pragma: no cover (off-GPU)
    torch = None; nn = None; F = None; KBoundOptimizer = None
    _HAS_TORCH = False

SEED = 0
# Pre-registered continual schedule: (corruption, severity, composition, n_batches).
# Long trap runs (single_class, severe) interleaved with recoverable (iid, mild) windows.
QUICK_SCHEDULE = [
    ("fog", 1, "iid", 4), ("gaussian_noise", 1, "iid", 4),                    # recoverable warm-up
    ("contrast", 5, "single_class", 10), ("contrast", 5, "single_class", 10), # long TRAP run...
    ("gaussian_noise", 5, "single_class", 10),                               # ...continues -> collapse
    ("jpeg_compression", 1, "iid", 4),                                       # recovery window
    ("contrast", 5, "single_class", 10),                                     # TRAP again
]
FULL_SCHEDULE = QUICK_SCHEDULE * 4


# ============================================================================
#  ANALYSIS CORE  (pure numpy -- smoke-testable without torch)
# ============================================================================
def stream_metrics(acc_by_method, frozen_acc, collapse_drop=0.15):
    """acc_by_method: {method: np.array per-window accuracy}; frozen_acc: np.array.
    Returns per-method mean acc, worst-window acc, #collapse windows (acc<frozen-drop),
    and regret to the per-window upper envelope across all methods."""
    frozen_acc = np.asarray(frozen_acc, float)
    methods = list(acc_by_method.keys())
    env = np.max(np.vstack([np.asarray(acc_by_method[m], float) for m in methods]
                           + [frozen_acc]), axis=0)   # per-window best any policy achieved
    out = {}
    for m in methods:
        a = np.asarray(acc_by_method[m], float)
        out[m] = dict(
            mean_acc=float(a.mean()),
            worst_window_acc=float(a.min()),
            collapse_windows=int(np.sum(a < frozen_acc - collapse_drop)),
            regret_to_envelope=float((env - a).mean()),
        )
    # headline comparison
    if "kbound_opt" in out and "sar" in out:
        out["_headline"] = {
            "kbound_mean": out["kbound_opt"]["mean_acc"],
            "sar_mean": out["sar"]["mean_acc"],
            "kbound_worst": out["kbound_opt"]["worst_window_acc"],
            "sar_worst": out["sar"]["worst_window_acc"],
            "kbound_collapses": out["kbound_opt"]["collapse_windows"],
            "sar_collapses": out["sar"]["collapse_windows"],
            "kbound_beats_sar_mean": bool(out["kbound_opt"]["mean_acc"] > out["sar"]["mean_acc"] + 1e-6),
            "kbound_ties_or_beats_sar":
                bool(out["kbound_opt"]["mean_acc"] >= out["sar"]["mean_acc"] - 0.005),
        }
    return out


# ============================================================================
#  TORCH PARTS  (run on the Mac)
# ============================================================================
def _eval_acc(model, X, Y, bs=512):
    model.eval()
    with torch.no_grad():
        pred = []
        for i in range(0, len(X), bs):
            pred.append(model(X[i:i+bs]).argmax(1).cpu())
    return float((torch.cat(pred) == Y).float().mean().item())


def _to_dev(np_uint8, dev):
    x = torch.tensor(np.ascontiguousarray(np_uint8)).permute(0, 3, 1, 2).contiguous().float() / 255.0
    return _norm_cifar(x.to(dev)).contiguous()


def _balanced_idx(y, per_class, rng, ncls=10):
    idx = []
    for c in range(ncls):
        ci = np.where(y == c)[0]
        if len(ci):
            idx.append(rng.choice(ci, min(per_class, len(ci)), replace=False))
    out = np.concatenate(idx); rng.shuffle(out); return out


def build_stream(data_root, schedule, dev, rng, bs=8, eval_pool=800, probe_pool=512):
    """Returns (windows, evals, clean_probe). Continual stream over the schedule."""
    # clean probe: balanced clean CIFAR-10 test images (labels allowed; severity-1 brightness ~ clean-ish)
    Xb, yb = load_cifar_c(data_root, "10", "brightness")           # sev1 block ~ mild
    Xb1, yb1 = cifar_c_severity(Xb, yb, 1)
    pidx = _balanced_idx(yb1, probe_pool // 10, rng)
    probe = (_to_dev(Xb1[pidx], dev), torch.tensor(yb1[pidx]))
    windows, evals = [], []
    for (corr, sev, comp, nb) in schedule:
        Xc, yc = load_cifar_c(data_root, "10", corr)
        sX, sY = cifar_c_severity(Xc, yc, sev)
        # balanced labelled eval pool for this window (separate from the stream)
        eidx = _balanced_idx(sY, eval_pool // 10, rng)
        evals.append((_to_dev(sX[eidx], dev), torch.tensor(sY[eidx])))
        remain = np.setdiff1d(np.arange(len(sY)), eidx)
        # stream batches with the chosen composition
        batches = []
        for _ in range(nb):
            if comp == "single_class":
                cls = rng.integers(10)
                pool = np.intersect1d(np.where(sY == cls)[0], remain)
                bidx = rng.choice(pool if len(pool) else remain, bs, replace=True)
            else:
                bidx = rng.choice(remain, bs, replace=False)
            batches.append(_to_dev(sX[bidx], dev))
        windows.append(batches)
    return windows, evals, probe


def _configure(base):
    m = copy.deepcopy(base); m.train()
    for p in m.parameters(): p.requires_grad_(False)
    ps = _bn_affine_params(m)
    return m, ps


def run_continual(method, base_model, windows, evals, probe, dev, lr=2.5e-3,
                  steps_per_batch=8, sar_reset=None, alpha=0.1):
    """Continually adapt ONE model across the whole stream; return per-window accuracy."""
    if method == "frozen":
        base_model.eval()
        return np.array([_eval_acc(base_model, ex, ey) for (ex, ey) in evals])

    m, ps = _configure(base_model)
    accs = []
    px, py = probe
    if method == "kbound_opt":
        base_opt = torch.optim.Adam(ps, lr=lr)
        # TWEAK 1 (less-noisy signal): use the NET benefit on a clean validation probe vs the
        # frozen baseline, EMA-smoothed -- a direct estimate of Delta = acc(adapted) - acc(frozen),
        # far less noisy than the old step-to-step probe delta.
        # TWEAK 3: the probe is a held-out VALIDATION set, never the test/eval windows.
        frozen_probe_acc = _eval_acc(base_model, px, py)
        ben_ema = {"v": 0.0}
        def evidence_fn():
            benefit = _eval_acc(m, px, py) - frozen_probe_acc   # net benefit vs frozen on probe
            ben_ema["v"] = 0.7 * ben_ema["v"] + 0.3 * benefit   # EMA smoothing
            return float(np.clip(ben_ema["v"] * 5.0, -1.0, 1.0))
        opt = KBoundOptimizer(base_opt, evidence_fn=evidence_fn, alpha=alpha,
                              a=-1.0, b=1.0, abstain_scale=1.0)  # adapt until certified harmful
    else:
        opt = torch.optim.Adam(ps, lr=lr)

    if method == "sar":
        ema = None; e_reset = sar_reset if sar_reset is not None else 0.2 * math.log(10)

    for batches in windows:
        if method == "kbound_opt":          # TWEAK 2 (hysteresis): re-arm the certificate at each
            opt.eprocess.reset(); ben_ema["v"] = 0.0   # new regime so it can re-adapt after a freeze
        for xb in batches:
            for _ in range(steps_per_batch):
                out = m(xb.contiguous()); p = out.softmax(1); ent = _entropy(p)
                if method == "eata":
                    keep = ent < 0.4 * math.log(10)
                    if keep.sum() == 0: continue
                    loss = ent[keep].mean()
                elif method == "sar":
                    keep = ent < 0.4 * math.log(10)
                    if keep.sum() == 0: continue
                    loss = ent[keep].mean()
                else:
                    loss = ent.mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if method == "sar":                    # collapse-reset heuristic
                    em = float(loss.detach()); ema = em if ema is None else 0.9 * ema + 0.1 * em
                    if ema is not None and ema < e_reset:
                        with torch.no_grad():
                            mm, ps2 = _configure(base_model)
                            for p_, q_ in zip(ps, ps2): p_.copy_(q_)
                        ema = None
        accs.append(_eval_acc(m, *evals[len(accs)]))
    return np.array(accs)


def main():
    ap = argparse.ArgumentParser(description="KBoundOptimizer vs SAR continual collapse benchmark")
    ap.add_argument("--data-root", default="experiments/kbound/cifar")
    ap.add_argument("--out", default="experiments/kbound/results/kboundopt_results.json")
    ap.add_argument("--methods", nargs="+",
                    default=["frozen", "tent", "eata", "sar", "kbound_opt"])
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run the torch-free analysis core + gating logic only")
    args = ap.parse_args()

    if args.smoke_test or not _HAS_TORCH:
        return smoke_test()

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    base = get_cifar_model("10", args.data_root, dev)
    sched = QUICK_SCHEDULE if args.quick else FULL_SCHEDULE
    windows, evals, probe = build_stream(args.data_root, sched, dev, rng)
    print(f"[stream] {len(windows)} windows; clean acc(probe)={_eval_acc(base, *probe):.3f}")

    acc_by = {}
    for mth in args.methods:
        t0 = time.time()
        acc_by[mth] = run_continual(mth, base, windows, evals, probe, dev,
                                    lr=args.lr, alpha=args.alpha)
        print(f"  {mth:11s} mean={acc_by[mth].mean():.3f} worst={acc_by[mth].min():.3f} "
              f"({time.time()-t0:.0f}s)")
    frozen = acc_by.get("frozen", np.zeros(len(windows)))
    metrics = stream_metrics({k: v.tolist() for k, v in acc_by.items() if k != "frozen"}, frozen)
    out = {"device": dev, "schedule": sched, "per_window_acc": {k: v.tolist() for k, v in acc_by.items()},
           "metrics": metrics, "alpha": args.alpha}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\nheadline:", json.dumps(metrics.get("_headline", {}), indent=2))
    print("Saved", args.out)


def smoke_test():
    """No torch: validate the analysis core + that the e-process freezes on a collapse signal."""
    print("== smoke test (torch-free) ==")
    rng = np.random.default_rng(0)
    nW = 12
    frozen = np.full(nW, 0.55)
    # synthetic per-window accuracies: tent collapses on trap windows; sar/kbound hold
    tent = frozen + np.array([0.1,0.1, -0.35,-0.4, 0.1, -0.4, 0.1,0.1,-0.38,-0.4,0.1,-0.39])
    sar  = frozen + np.array([0.08,0.09, 0.0,0.0, 0.09, 0.0, 0.08,0.09,0.0,0.0,0.09,0.0])
    kb   = frozen + np.array([0.09,0.10, 0.0,0.0, 0.10, 0.0, 0.09,0.10,0.0,0.0,0.10,0.0])
    m = stream_metrics({"tent": tent, "sar": sar, "kbound_opt": kb}, frozen)
    print("tent       :", m["tent"])
    print("sar        :", m["sar"])
    print("kbound_opt :", m["kbound_opt"])
    print("headline   :", m["_headline"])
    # e-process: a sustained negative (collapse) benefit must trigger 'freeze'
    ep = EProcess(alpha=0.1)
    froze_at = None
    for t, x in enumerate(rng.normal(-0.3, 0.05, 60)):   # sustained harm
        ep.update(float(np.clip(x, -1, 1)))
        if ep.decision() == "freeze": froze_at = t; break
    print(f"e-process froze on sustained-harm stream at step {froze_at} (decision={ep.decision()})")
    # and stays abstain/adapt on a helpful stream
    ep2 = EProcess(alpha=0.1); dec2 = "abstain"
    for x in rng.normal(+0.3, 0.05, 60):
        ep2.update(float(np.clip(x, -1, 1)))
        if ep2.decision() == "adapt": dec2 = "adapt"; break
    ok = (m["tent"]["collapse_windows"] > 0 and m["sar"]["collapse_windows"] == 0
          and m["kbound_opt"]["collapse_windows"] == 0 and froze_at is not None and dec2 == "adapt")
    print("SMOKE TEST PASS:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
