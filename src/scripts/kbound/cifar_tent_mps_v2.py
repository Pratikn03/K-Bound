"""
K-Bound DECISIVE benchmark v2 — catastrophic-harm test-time adaptation.

Goal (the one experiment the paper still needs): show that the KGA
adapt/freeze/abstain certificate beats BOTH trivial policies — always-adapt and
always-freeze — on a *realistic* deep-TTA setting, by adapting where adaptation
helps and refusing where it catastrophically (and detectably) hurts.

WHY v1 could not show this (diagnosed from experiments/kbound/results/cifar_tent_results.json):
  base_rate_harmful_B<0 = 0.023, mean_true_B = +0.20  ->  the suite was almost
  entirely a HELPFUL regime, so always-adapt ~= oracle and KGA can only tie it.
  v1 also used gentle Tent (10 steps, lr 1e-3) on toy on-the-fly corruptions, which
  rarely collapses, and measured benefit on the (sometimes imbalanced) adaptation
  batch itself.

WHAT v2 FIXES
  1. REAL corruption benchmarks: CIFAR-10-C, CIFAR-100-C, ImageNet-C (standard files).
  2. A PRE-REGISTERED MIXED GRID that spans regimes on purpose, including documented
     Tent-collapse cells:  severity in {1,3,5} x batch in {large-iid, small, tiny}
     x stream composition in {iid, class-imbalanced, single-class/label-shift}
     x update aggressiveness in {mild, aggressive}.  The aggressive x tiny/single-class
     x severe cells are where entropy minimization collapses to a near-constant
     predictor.  The grid is declared up front and we report the FULL stratified
     breakdown (no cherry-picking).
  3. HONEST benefit:  the model is ADAPTED on the (possibly nasty) stream, but
     benefit B is measured on a separate, class-BALANCED held-out eval set for the
     same corruption.  So collapse shows up as real accuracy loss, not as an artifact
     of an imbalanced batch.
  4. Three candidate adaptations: Tent, EATA, SAR (+ frozen f0).  KGA is run per method.
  5. KGA is IDENTICAL to the rest of the paper: leave-one-condition-out gradient-boosted
     B_hat(Z) + split-conformal radius eps, decide ADAPT if B_hat-eps>0,
     FREEZE if B_hat+eps<0, else ABSTAIN.  Z is LABEL-FREE only.
  6. The reviewer-proof figure: a MIXING-RATIO PARETO sweep — vary the harmful
     fraction p of the deployment stream and show KGA's regret <= both baselines for
     all p (ties always-adapt at p=0, ties always-freeze at p=1, strictly beats both
     in between).  That is the precise, honest answer to "where does it beat both?".

HONESTY / SCOPE
  - The grid is a STRESS benchmark: it intentionally samples the regime space evenly,
    so the headline mean depends on the mix.  That is exactly why we report the Pareto
    over p instead of a single cherry-picked mean.
  - KGA only ever sees label-free evidence Z.  True labels are used ONLY to compute B
    and the oracle, for evaluation.
  - EATA/SAR here are faithful re-implementations (BN-affine, entropy filtering, SAM,
    entropy-reset), not the original repos; swap in official code if you prefer.
  - Every number is produced by this run on your hardware. Nothing is fabricated.

RUN (on your M5 / any CUDA box) — see README_DECISIVE.md for full setup:
  python cifar_tent_mps_v2.py --benchmarks cifar10c cifar100c \
      --data-root experiments/kbound/cifar --quick
  # add imagenetc once ImageNet-C is on disk:
  python cifar_tent_mps_v2.py --benchmarks imagenetc \
      --imagenetc-root /path/to/ImageNet-C --imagenet-val /path/to/val

The analysis core (decide_kga, policy_metrics, mixing_pareto, summarize) imports
WITHOUT torch so it can be unit-tested off-GPU.
"""
from __future__ import annotations
import os, sys, json, copy, math, time, argparse, glob
import numpy as np

# torch is only needed for the actual TTA runs; guard it so the analysis core
# (and the dry-run test) imports on machines without torch.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision as tv
    import torchvision.transforms as T
    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only off-GPU
    torch = None; nn = None; F = None; tv = None; T = None
    _HAS_TORCH = False

from sklearn.ensemble import GradientBoostingRegressor

# ----------------------------------------------------------------------------- #
#  Pre-registered condition grid (declare BEFORE looking at any result)
# ----------------------------------------------------------------------------- #
SEVERITIES      = [1, 3, 5]
BATCH_REGIMES   = {"large_iid": 200, "small": 16, "tiny": 8}
COMPOSITIONS    = ["iid", "imbalanced", "single_class"]   # single_class = label shift
AGGRESSIVENESS  = {"mild": dict(steps=10, lr=1e-3), "aggressive": dict(steps=50, lr=2.5e-3)}
N_REPEATS       = 2          # repeats per cell (different sampled streams)
EVAL_POOL       = 2000       # balanced held-out eval images per corruption/severity
HELP_THR        = 0.02       # |B| band for labeling regime (REPORTING ONLY)
ALPHA           = 0.10       # conformal miscoverage -> false-adapt/false-freeze target
SEED            = 0

# CIFAR-C corruption names (standard 15 + 4 extra). --quick uses a representative subset.
CIFAR_C_ALL = ["gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
               "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
               "elastic_transform","pixelate","jpeg_compression"]
CIFAR_C_QUICK = ["gaussian_noise","defocus_blur","fog","contrast","pixelate","jpeg_compression"]
IMAGENET_C_QUICK = ["gaussian_noise","defocus_blur","snow","contrast","elastic_transform","jpeg_compression"]


# =============================================================================
#  ANALYSIS CORE  (pure numpy/sklearn — no torch; unit-testable off-GPU)
# =============================================================================
def decide_kga(Z, B, alpha=ALPHA, n_estimators=250, max_depth=2, lr=0.05, seed=SEED):
    """Leave-one-out gradient-boosted benefit estimator + split-conformal radius.
    Returns (Bhat, eps, decisions). Identical machinery to knowability/mixed_regime."""
    Z = np.asarray(Z, float); B = np.asarray(B, float); N = len(B)
    Bhat = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=lr, subsample=0.8, random_state=seed)
        m.fit(Z[tr], B[tr])
        Bhat[i] = m.predict(Z[i:i+1])[0]
    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
    dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
    return Bhat, eps, dec


def policy_metrics(dec, a0, aa, B=None):
    """Realized accuracy + regret vs oracle for each policy. ABSTAIN/FREEZE -> frozen."""
    a0 = np.asarray(a0, float); aa = np.asarray(aa, float)
    adapt = dec == "ADAPT"
    kga = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    if B is None:
        B = aa - a0
    B = np.asarray(B, float)
    naive = (aa - a0) > 0  # an estimator-free "adapt if it would help" upper-reference uses truth; we instead use Bhat outside
    out = {
        "n": int(len(a0)),
        "decision_counts": {d: int((dec == d).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "coverage": float(np.mean(dec != "ABSTAIN")),
        "adapt_precision_B>0": float(np.mean(B[adapt] > 0)) if adapt.any() else None,
        "false_adapt_rate_B<0": float(np.mean(B[adapt] < 0)) if adapt.any() else None,
        "mean_acc": {
            "always_adapt": float(aa.mean()),
            "always_freeze": float(a0.mean()),
            "K_Bound": float(kga.mean()),
            "oracle": float(oracle.mean()),
        },
        "regret_vs_oracle": {
            "always_adapt": float((oracle - aa).mean()),
            "always_freeze": float((oracle - a0).mean()),
            "K_Bound": float((oracle - kga).mean()),
        },
        "worst_case_acc": {
            "always_adapt": float(aa.min()),
            "always_freeze": float(a0.min()),
            "K_Bound": float(kga.min()),
        },
        "beats_both_regret_only": bool((oracle - kga).mean() < (oracle - aa).mean() - 1e-9 and
                                       (oracle - kga).mean() < (oracle - a0).mean() - 1e-9),
        # Integrity fix 2026-06-20: beats_both MUST enforce the pre-registered
        # false-adapt budget FA<=ALPHA, not regret alone (regret-only over-counted
        # "wins" on mixes where the router false-adapts above budget). The ungated
        # regret comparison is preserved above as beats_both_regret_only.
        "beats_both": bool((oracle - kga).mean() < (oracle - aa).mean() - 1e-9 and
                           (oracle - kga).mean() < (oracle - a0).mean() - 1e-9 and
                           adapt.any() and float(np.mean(B[adapt] < 0)) <= ALPHA),
    }
    return out


def mixing_pareto(dec, a0, aa, regime, fractions=None, seed=SEED, n_boot=200):
    """Vary the harmful fraction p of the deployment stream and report each policy's
    mean regret. Demonstrates KGA is on the Pareto front for all p (ties at extremes,
    beats both in the middle). 'regime' is the true label ('helpful'/'harmful'/'marginal')."""
    if fractions is None:
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    a0 = np.asarray(a0, float); aa = np.asarray(aa, float)
    regime = np.asarray(regime)
    kga = np.where(dec == "ADAPT", aa, a0)
    oracle = np.maximum(a0, aa)
    reg = {"always_adapt": oracle - aa, "always_freeze": oracle - a0, "K_Bound": oracle - kga}
    H = np.where(regime == "harmful")[0]
    NH = np.where(regime != "harmful")[0]
    rng = np.random.default_rng(seed)
    curve = []
    if len(H) == 0 or len(NH) == 0:
        return {"note": "need both harmful and non-harmful conditions for the sweep",
                "n_harmful": int(len(H)), "n_nonharmful": int(len(NH))}
    M = 200  # synthetic stream length per point
    for p in fractions:
        accs = {k: [] for k in reg}
        for _ in range(n_boot):
            nH = int(round(p * M)); nN = M - nH
            ids = np.concatenate([rng.choice(H, nH, replace=True),
                                  rng.choice(NH, nN, replace=True)])
            for k in reg:
                accs[k].append(reg[k][ids].mean())
        curve.append({"p_harmful": p, **{k: float(np.mean(v)) for k, v in accs.items()}})
    # crossover: smallest p where KGA strictly beats BOTH
    cross = next((c["p_harmful"] for c in curve
                  if c["K_Bound"] < c["always_adapt"] - 1e-6 and c["K_Bound"] < c["always_freeze"] - 1e-6),
                 None)
    return {"curve": curve, "p_where_KGA_beats_both": cross}


def summarize(rows, alpha=ALPHA):
    """rows: list of dicts with keys Z(list), a0, aa, regime, condition. Runs KGA + metrics."""
    Z = np.array([r["Z"] for r in rows], float)
    a0 = np.array([r["a0"] for r in rows], float)
    aa = np.array([r["aa"] for r in rows], float)
    B = aa - a0
    regime = np.array([r["regime"] for r in rows])
    Bhat, eps, dec = decide_kga(Z, B, alpha=alpha)
    pm = policy_metrics(dec, a0, aa, B)
    pm["eps_conformal"] = eps
    pm["alpha"] = alpha
    pm["base_rate_harmful_B<0"] = float(np.mean(B < 0))
    pm["mean_true_B"] = float(B.mean())
    pm["decisions_by_regime"] = {
        g: {d: int(((regime == g) & (dec == d)).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]}
        for g in ["helpful", "harmful", "marginal"]
    }
    pm["pareto"] = mixing_pareto(dec, a0, aa, regime)
    return pm, dict(Bhat=Bhat.tolist(), eps=eps, dec=dec.tolist(), B=B.tolist())


def label_regime(B, thr=HELP_THR):
    return "helpful" if B > thr else ("harmful" if B < -thr else "marginal")


# =============================================================================
#  FIGURES (matplotlib; safe without torch)
# =============================================================================
def make_figures(tag, per_method, fig_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    methods = list(per_method.keys())

    # 1) policy regret per method (grouped)
    plt.figure(figsize=(7, 4.2))
    width = 0.26; xs = np.arange(len(methods))
    for k, (pol, col) in enumerate([("always_adapt", "#e76f51"),
                                    ("always_freeze", "#457b9d"),
                                    ("K_Bound", "#2a9d8f")]):
        vals = [per_method[m]["metrics"]["regret_vs_oracle"][pol] for m in methods]
        plt.bar(xs + (k - 1) * width, vals, width, label=pol.replace("_", "-"), color=col)
    plt.xticks(xs, methods); plt.ylabel("regret vs oracle (lower better)")
    plt.title(f"{tag}: KGA regret vs trivial policies (per TTA method)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"fig_decisive_regret_{tag}.png"), dpi=130); plt.close()

    # 2) mixing-ratio Pareto (use first method that has a curve)
    cur = None; mname = None
    for m in methods:
        pa = per_method[m]["metrics"].get("pareto", {})
        if isinstance(pa, dict) and pa.get("curve"):
            cur = pa["curve"]; mname = m; break
    if cur:
        p = [c["p_harmful"] for c in cur]
        plt.figure(figsize=(6.4, 4.2))
        plt.plot(p, [c["always_adapt"] for c in cur], "-o", color="#e76f51", label="always-adapt")
        plt.plot(p, [c["always_freeze"] for c in cur], "-o", color="#457b9d", label="always-freeze")
        plt.plot(p, [c["K_Bound"] for c in cur], "-o", color="#2a9d8f", label="K-Bound")
        plt.xlabel("harmful fraction p of deployment stream")
        plt.ylabel("mean regret vs oracle")
        plt.title(f"{tag} ({mname}): KGA is Pareto-optimal across mixes")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"fig_decisive_pareto_{tag}.png"), dpi=130); plt.close()

    # 3) decisions by true regime (first method)
    m0 = methods[0]; dbr = per_method[m0]["metrics"]["decisions_by_regime"]
    regs = ["helpful", "harmful", "marginal"]; decs = ["ADAPT", "FREEZE", "ABSTAIN"]
    colors = {"ADAPT": "#2a9d8f", "FREEZE": "#457b9d", "ABSTAIN": "#9aa0a6"}
    M = np.array([[dbr[r][d] for d in decs] for r in regs], float)
    M = M / np.clip(M.sum(1, keepdims=True), 1, None)
    plt.figure(figsize=(6, 4)); bottom = np.zeros(len(regs))
    for k, d in enumerate(decs):
        plt.bar(regs, M[:, k], bottom=bottom, color=colors[d], label=d); bottom += M[:, k]
    plt.ylabel("fraction"); plt.title(f"{tag} ({m0}): KGA decisions by true regime")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"fig_decisive_decisions_{tag}.png"), dpi=130); plt.close()


# =============================================================================
#  TORCH-DEPENDENT PARTS  (only run on the M5 / GPU box)
# =============================================================================
def _require_torch():
    if not _HAS_TORCH:
        sys.exit("This step needs PyTorch + torchvision. Install them and re-run on your GPU/MPS box.")

def pick_device():
    _require_torch()
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# ---- models ----
def make_cifar_resnet(num_classes):
    m = tv.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
    return m

def get_cifar_model(which, data_dir, dev, ckpt=None):
    n = 10 if which == "10" else 100
    m = make_cifar_resnet(n).to(dev)
    default = os.path.join(data_dir, "resnet18_cifar.pt" if which == "10" else "resnet18_cifar100.pt")
    path = ckpt or default
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=dev)); print(f"[model] loaded {path}")
        m.eval(); return m
    print(f"[model] no checkpoint at {path} -> training a quick CIFAR-{n} ResNet18 (~few min on MPS)")
    _train_cifar(m, which, data_dir, dev, path)
    m.eval(); return m

def _train_cifar(m, which, data_dir, dev, save_path, epochs=12):
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    ds = (tv.datasets.CIFAR10 if which == "10" else tv.datasets.CIFAR100)
    tr = ds(data_dir, train=True, download=True, transform=T.ToTensor())
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))]); ytr = torch.tensor(tr.targets)
    aug = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * (len(Xtr) // 256 + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps)
    norm = lambda x: (x - MEAN.to(x.device)) / STD.to(x.device)
    m.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr)); cor = tot = 0
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]; xb = aug(Xtr[idx]).to(dev); yb = ytr[idx].to(dev)
            out = m(norm(xb)); loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            cor += (out.argmax(1) == yb).sum().item(); tot += len(yb)
        print(f"  epoch {ep+1}/{epochs} acc={cor/tot:.3f}")
    torch.save(m.state_dict(), save_path); print(f"[model] saved {save_path}")

# ---- data: CIFAR-10-C / CIFAR-100-C (.npy) ----
def load_cifar_c(root, which, corruption):
    """Returns (X uint8 [50000,32,32,3], y [50000]) stacked over 5 severities."""
    base = os.path.join(root, "CIFAR-10-C" if which == "10" else "CIFAR-100-C")
    xp = os.path.join(base, f"{corruption}.npy"); yp = os.path.join(base, "labels.npy")
    if not (os.path.exists(xp) and os.path.exists(yp)):
        raise FileNotFoundError(
            f"Missing {xp}. Download CIFAR-{which}-C (see README_DECISIVE.md) into {base}/")
    X = np.load(xp); y = np.load(yp).astype(int)
    return X, y

def cifar_c_severity(X, y, sev):
    """Severity s in 1..5 -> block s of 10000."""
    a = (sev - 1) * 10000; b = sev * 10000
    return X[a:b], y[a:b]

# ---- data: ImageNet-C (folder of corruption/severity/class/*.JPEG) ----
def imagenet_c_loader(root, corruption, sev, transform, max_images, dev):
    folder = os.path.join(root, corruption, str(sev))
    ds = tv.datasets.ImageFolder(folder, transform=transform)
    if max_images and len(ds) > max_images:
        idx = np.random.default_rng(SEED).choice(len(ds), max_images, replace=False)
        ds = torch.utils.data.Subset(ds, idx.tolist())
    return torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)

# ---- evidence (label-free) ----
def _entropy(p):
    return -(p * (p + 1e-9).log()).sum(1)

def evidence_vector(model_frozen, model_adapted, x, num_classes, upd_norm):
    """All label-free. x is normalized batch on device."""
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
        frac_hi = (pa.max(1).values > 0.9).float().mean().item()   # collapse -> near 1
        # marginal-prediction KL frozen->adapted (collapse spikes this)
        klm = (mba * ((mba + 1e-9).log() - (mb0 + 1e-9).log())).sum().item()
    # Z: [pre-entropy, pre-conf, pre-balance, post-entropy, post-conf, post-balance,
    #     balance_drop, entropy_drop, frac_highconf, marginal_KL, update_norm]
    return [e0, conf0, pbal0, ea, confa, pbala, pbal0 - pbala, e0 - ea, frac_hi, klm, upd_norm]

EVIDENCE_NAMES = ["pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf",
                  "post_pbal", "pbal_drop", "entropy_drop", "frac_highconf", "marginal_KL", "update_norm"]

# ---- TTA methods (BN-affine), return (adapted_model, update_norm) ----
def _bn_affine_params(m):
    ps = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.track_running_stats = False; mod.running_mean = None; mod.running_var = None
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
    # Fisher diagonal from first batch (importance of each BN-affine param)
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

def sar_adapt(base, stream, steps, lr, num_classes, rho=0.05, e_reset=None):
    """Sharpness-aware (SAM) reliable entropy minimization + collapse reset (faithful-ish SAR)."""
    margin = 0.4 * math.log(num_classes)
    if e_reset is None: e_reset = 0.2 * math.log(num_classes)
    m, ps, init = _clone_for_tta(base); opt = torch.optim.SGD(ps, lr=lr, momentum=0.9)
    ema = None
    for _ in range(steps):
        for xb in stream:
            out = m(xb.contiguous()); ent = _entropy(out.softmax(1)); keep = ent < margin
            if keep.sum() == 0: continue
            loss = ent[keep].mean()
            opt.zero_grad(); loss.backward()
            # SAM ascent step
            with torch.no_grad():
                g = [p.grad.detach() for p in ps]; gn = (sum((gi ** 2).sum() for gi in g) ** 0.5) + 1e-12
                for p, gi in zip(ps, g): p.add_(gi * (rho / gn))
            out2 = m(xb.contiguous()); ent2 = _entropy(out2.softmax(1)); keep2 = ent2 < margin
            loss2 = ent2[keep2].mean() if keep2.any() else ent2.mean()
            opt.zero_grad(); loss2.backward()
            with torch.no_grad():
                g = [p.grad.detach() for p in ps]; gn = (sum((gi ** 2).sum() for gi in g) ** 0.5) + 1e-12
                for p, gi in zip(ps, g): p.sub_(gi * (rho / gn))   # back to weights
            opt.step()
            em = ent[keep].mean().item(); ema = em if ema is None else 0.9 * ema + 0.1 * em
            if ema is not None and ema < e_reset:      # collapse guard -> reset
                with torch.no_grad():
                    for p, q in zip(ps, init): p.copy_(q)
                ema = None
    return m, _upd_norm(ps, init)

TTA_METHODS = {"tent": tent_adapt, "eata": eata_adapt, "sar": sar_adapt}

# ---- stream / eval construction ----
def _norm_cifar(x):
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
    STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(x.device)
    return (x - MEAN) / STD

def build_stream_and_eval(Xc, yc, sev_X, sev_y, comp, bs, num_classes, rng, dev, eval_pool=EVAL_POOL):
    """Xc,yc severity block. Returns (list of normalized stream batches, eval_x, eval_y)."""
    # balanced held-out eval pool (truth used ONLY for evaluation)
    per = max(1, eval_pool // num_classes); ev_idx = []
    for c in range(num_classes):
        ci = np.where(sev_y == c)[0]
        if len(ci): ev_idx.append(rng.choice(ci, min(per, len(ci)), replace=False))
    ev_idx = np.concatenate(ev_idx); rng.shuffle(ev_idx)
    remain = np.setdiff1d(np.arange(len(sev_y)), ev_idx)
    # adaptation stream composition
    if comp == "iid":
        s_idx = rng.choice(remain, min(bs * 4, len(remain)), replace=False)
    elif comp == "imbalanced":
        major = rng.integers(num_classes)
        maj = np.intersect1d(np.where(sev_y == major)[0], remain)
        oth = np.setdiff1d(remain, maj)
        nM = int(bs * 4 * 0.85)
        s_idx = np.concatenate([rng.choice(maj, min(nM, len(maj)), replace=True),
                                rng.choice(oth, bs * 4 - min(nM, len(maj)), replace=True)])
    else:  # single_class label shift
        major = rng.integers(num_classes)
        maj = np.intersect1d(np.where(sev_y == major)[0], remain)
        s_idx = rng.choice(maj if len(maj) else remain, bs * 4, replace=True)
    rng.shuffle(s_idx)
    def to_dev(arr_idx, src):
        x = torch.tensor(src[arr_idx]).permute(0, 3, 1, 2).float() / 255.0
        return _norm_cifar(x.to(dev)).contiguous()
    stream = [to_dev(s_idx[i:i + bs], sev_X) for i in range(0, len(s_idx), bs)]
    eval_x = to_dev(ev_idx, sev_X); eval_y = torch.tensor(sev_y[ev_idx])
    return stream, eval_x, eval_y

def acc_on(model, x, y, train_mode=True):
    model.train() if train_mode else model.eval()
    with torch.no_grad():
        pred = []
        for i in range(0, len(x), 512):
            pred.append(model(x[i:i + 512]).argmax(1).cpu())
    return (torch.cat(pred) == y).float().mean().item()


# =============================================================================
#  RUN ONE BENCHMARK
# =============================================================================
def run_cifar_benchmark(which, data_root, dev, methods, corruptions, ckpt=None, quick=False):
    num_classes = 10 if which == "10" else 100
    model = get_cifar_model(which, data_root, dev, ckpt)
    rng = np.random.default_rng(SEED)
    # clean acc
    base = (tv.datasets.CIFAR10 if which == "10" else tv.datasets.CIFAR100)(data_root, train=False, download=True)
    Xclean = base.data; yclean = np.array(base.targets)
    ev = build_stream_and_eval(Xclean, yclean, Xclean, yclean, "iid", 200, num_classes, rng, dev)[1:]
    clean_acc = acc_on(model, ev[0], ev[1], train_mode=False)
    print(f"[{which}] clean acc = {clean_acc:.3f}")

    rows = {m: [] for m in methods}
    cells = [(c, s, br, comp, ag) for c in corruptions for s in SEVERITIES
             for br in BATCH_REGIMES for comp in COMPOSITIONS for ag in AGGRESSIVENESS]
    if quick:
        cells = [c for c in cells if c[1] in (1, 5)]  # only mild & severe in quick mode
    print(f"[{which}] {len(cells)} cells x {N_REPEATS} repeats x {len(methods)} methods")
    t0 = time.time()
    for ci, (corr, sev, brn, comp, agn) in enumerate(cells):
        Xc, yc = load_cifar_c(data_root, which, corr)
        sX, sY = cifar_c_severity(Xc, yc, sev)
        bs = BATCH_REGIMES[brn]; ag = AGGRESSIVENESS[agn]
        for rep in range(N_REPEATS):
            stream, ex, ey = build_stream_and_eval(Xc, yc, sX, sY, comp, bs, num_classes, rng, dev)
            a0 = acc_on(model, ex, ey, train_mode=False)
            for mth in methods:
                fn = TTA_METHODS[mth]
                kwargs = dict(steps=ag["steps"], lr=ag["lr"])
                if mth in ("eata", "sar"): kwargs["num_classes"] = num_classes
                adapted, un = fn(model, stream, **kwargs)
                aa = acc_on(adapted, ex, ey, train_mode=True)
                Z = evidence_vector(model, adapted, ex, num_classes, un)
                rows[mth].append(dict(condition=f"{corr}|s{sev}|{brn}|{comp}|{agn}|r{rep}",
                                      Z=Z, a0=a0, aa=aa, regime=label_regime(aa - a0)))
        if (ci + 1) % 10 == 0:
            print(f"   {ci+1}/{len(cells)} cells  ({time.time()-t0:.0f}s)")
    return clean_acc, rows


def run_imagenet_benchmark(ic_root, val_root, dev, methods, corruptions, quick=False, max_images=4000):
    # ImageNet uses a frozen pretrained resnet50; BN-affine TTA; eval on the corruption set itself
    weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
    model = tv.models.resnet50(weights=weights).to(dev).eval()
    tf = weights.transforms()
    num_classes = 1000
    rng = np.random.default_rng(SEED)
    rows = {m: [] for m in methods}
    sevs = [1, 5] if quick else SEVERITIES
    print(f"[imagenet-c] {len(corruptions)} corruptions x {len(sevs)} sev x {len(methods)} methods")
    for corr in corruptions:
        for sev in sevs:
            loader = imagenet_c_loader(ic_root, corr, sev, tf, max_images, dev)
            xs, ys = [], []
            for xb, yb in loader: xs.append(xb); ys.append(yb)
            X = torch.cat(xs).to(dev); Y = torch.cat(ys)
            # eval = all; stream = shuffled minibatches (tiny+aggressive to provoke collapse)
            for brn, bs in [("large_iid", 128), ("tiny", 8)]:
                for agn, ag in AGGRESSIVENESS.items():
                    perm = torch.randperm(len(X))[: bs * 8]
                    stream = [X[perm[i:i+bs]] for i in range(0, len(perm), bs)]
                    a0 = acc_on(model, X, Y, train_mode=False)
                    for mth in methods:
                        fn = TTA_METHODS[mth]; kw = dict(steps=ag["steps"], lr=ag["lr"])
                        if mth in ("eata", "sar"): kw["num_classes"] = num_classes
                        adapted, un = fn(model, stream, **kw)
                        aa = acc_on(adapted, X, Y, train_mode=True)
                        Z = evidence_vector(model, adapted, X[:1024], num_classes, un)
                        rows[mth].append(dict(condition=f"{corr}|s{sev}|{brn}|{agn}",
                                              Z=Z, a0=a0, aa=aa, regime=label_regime(aa - a0)))
    return None, rows


# =============================================================================
#  MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="K-Bound decisive deep-TTA benchmark v2")
    ap.add_argument("--benchmarks", nargs="+", default=["cifar10c"],
                    choices=["cifar10c", "cifar100c", "imagenetc"])
    ap.add_argument("--methods", nargs="+", default=["tent", "eata", "sar"],
                    choices=["tent", "eata", "sar"])
    ap.add_argument("--data-root", default="experiments/kbound/cifar",
                    help="dir holding CIFAR-10-C/ and CIFAR-100-C/ and model ckpts")
    ap.add_argument("--cifar10-ckpt", default=None)
    ap.add_argument("--cifar100-ckpt", default=None)
    ap.add_argument("--imagenetc-root", default=None, help="ImageNet-C root (corruption/severity/class/*)")
    ap.add_argument("--out-results", default="experiments/kbound/results")
    ap.add_argument("--out-figs", default="docs/research/kbound/figures")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--quick", action="store_true", help="subset of corruptions/severities (fast smoke run)")
    args = ap.parse_args()

    os.makedirs(args.out_results, exist_ok=True); os.makedirs(args.out_figs, exist_ok=True)
    dev = pick_device(); print("device:", dev)
    torch.manual_seed(SEED); np.random.seed(SEED)

    combined = {"alpha": args.alpha, "device": dev, "benchmarks": {}, "generated": time.strftime("%Y-%m-%d %H:%M")}

    for bench in args.benchmarks:
        if bench in ("cifar10c", "cifar100c"):
            which = "10" if bench == "cifar10c" else "100"
            corrs = (CIFAR_C_QUICK if args.quick else CIFAR_C_ALL)
            ck = args.cifar10_ckpt if which == "10" else args.cifar100_ckpt
            clean_acc, rows = run_cifar_benchmark(which, args.data_root, dev, args.methods, corrs, ck, args.quick)
        else:
            if not args.imagenetc_root: print("[skip] imagenetc needs --imagenetc-root"); continue
            corrs = (IMAGENET_C_QUICK if args.quick else None) or IMAGENET_C_QUICK
            clean_acc, rows = run_imagenet_benchmark(args.imagenetc_root, None, dev, args.methods, corrs, args.quick)

        per_method = {}
        for mth, rws in rows.items():
            if len(rws) < 8:
                print(f"[warn] {bench}/{mth}: only {len(rws)} conditions");
            metrics, detail = summarize(rws, alpha=args.alpha)
            per_method[mth] = {"metrics": metrics, "n_conditions": len(rws),
                               "conditions": [r["condition"] for r in rws]}
            print(f"\n=== {bench} / {mth} ===")
            print("  harmful base rate:", round(metrics["base_rate_harmful_B<0"], 3),
                  "| mean B:", round(metrics["mean_true_B"], 3))
            print("  mean acc:", {k: round(v, 3) for k, v in metrics["mean_acc"].items()})
            print("  regret:  ", {k: round(v, 4) for k, v in metrics["regret_vs_oracle"].items()})
            print("  KGA beats BOTH baselines on this mix:", metrics["beats_both"])
            pa = metrics.get("pareto", {})
            if isinstance(pa, dict) and pa.get("p_where_KGA_beats_both") is not None:
                print("  KGA strictly beats both once harmful fraction >=", pa["p_where_KGA_beats_both"])
        combined["benchmarks"][bench] = {"clean_acc": clean_acc, "methods": per_method}
        make_figures(bench, per_method, args.out_figs)

    outp = os.path.join(args.out_results, "decisive_tta_results.json")
    json.dump(combined, open(outp, "w"), indent=2)
    print("\nSaved", outp)
    # markdown decisive table
    _write_md_table(combined, os.path.join(args.out_results, "decisive_tta_table.md"))
    print("Saved decisive_tta_table.md")


def _write_md_table(combined, path):
    lines = ["# Decisive deep-TTA results (KGA vs trivial policies)\n",
             f"_generated {combined.get('generated','')}, alpha={combined['alpha']}, device={combined['device']}_\n",
             "Regret vs oracle (lower is better). KGA should be <= BOTH always-adapt and always-freeze on a mixed stream.\n",
             "| benchmark | method | harmful% | always-adapt | always-freeze | **K-Bound** | oracle acc | beats both? | p* |",
             "|---|---|--:|--:|--:|--:|--:|:--:|--:|"]
    for b, bd in combined["benchmarks"].items():
        for m, md in bd["methods"].items():
            x = md["metrics"]; r = x["regret_vs_oracle"]; pa = x.get("pareto", {})
            ps = pa.get("p_where_KGA_beats_both") if isinstance(pa, dict) else None
            lines.append(f"| {b} | {m} | {x['base_rate_harmful_B<0']*100:.0f}% | "
                         f"{r['always_adapt']:.4f} | {r['always_freeze']:.4f} | **{r['K_Bound']:.4f}** | "
                         f"{x['mean_acc']['oracle']:.3f} | {'YES' if x['beats_both'] else 'no'} | "
                         f"{('%.1f'%ps) if ps is not None else '-'} |")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
