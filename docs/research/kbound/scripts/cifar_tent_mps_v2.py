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
import os, sys, io, json, copy, math, time, argparse, glob, random
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
# Official-SAR (Protocol E) knobs. Defaults reproduce the matched-LR SAR exactly
# (sar_lr=None -> shared lr; freeze_layer4=False -> all affine params adapted), so
# every prior run stays byte-identical unless these are explicitly set from the CLI.
SAR_LR          = None
SAR_FREEZE_LAYER4 = False

def set_global_seed(seed):
    """Set the run seed for ALL rng (protocol: 'seed sets ALL rng').
    Reassigns the module-global SEED so every np.random.default_rng(SEED) created
    inside the benchmark functions (batch-composition / eval-pool sampling) and the
    decide_kga/mixing_pareto defaults pick it up, and seeds torch/np/python-random."""
    global SEED
    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    if _HAS_TORCH:
        torch.manual_seed(SEED)
        try:
            torch.cuda.manual_seed_all(SEED)
        except Exception:
            pass
    return SEED

# CIFAR-C corruption names (standard 15 + 4 extra). --quick uses a representative subset.
CIFAR_C_ALL = ["gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
               "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
               "elastic_transform","pixelate","jpeg_compression"]
CIFAR_C_QUICK = ["gaussian_noise","defocus_blur","fog","contrast","pixelate","jpeg_compression"]
IMAGENET_C_QUICK = ["gaussian_noise","defocus_blur","snow","contrast","elastic_transform","jpeg_compression"]
# Full ImageNet-C: standard 15 + 4 extra = 19 corruptions (Zenodo 2235448 grouping).
IMAGENET_C_ALL = ["gaussian_noise","shot_noise","impulse_noise",
                  "defocus_blur","glass_blur","motion_blur","zoom_blur",
                  "snow","frost","fog","brightness",
                  "contrast","elastic_transform","pixelate","jpeg_compression",
                  "speckle_noise","gaussian_blur","spatter","saturate"]


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

# ---- data: CIFAR-10.1 (NATURAL distribution shift; reuses the CIFAR-10 model) ----
CIFAR101_URLS = {
    "data":   "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
    "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy",
}
def load_cifar_101(root):
    """CIFAR-10.1 v6: ~2000 brand-new REAL photos collected by CIFAR-10's own protocol
    (a NATURAL distribution shift -- no synthetic corruption). Returns
    (X uint8 [N,32,32,3], y [N]) -- identical format to load_cifar_c, so it flows through
    the SAME stream/KGA machinery. Auto-downloads the two .npy files (~30 MB) into
    <root>/CIFAR-10.1/ if absent."""
    base = os.path.join(root, "CIFAR-10.1"); os.makedirs(base, exist_ok=True)
    xp = os.path.join(base, "cifar10.1_v6_data.npy")
    yp = os.path.join(base, "cifar10.1_v6_labels.npy")
    for path, url in ((xp, CIFAR101_URLS["data"]), (yp, CIFAR101_URLS["labels"])):
        if not os.path.exists(path):
            try:
                import urllib.request
                print(f"[cifar10.1] downloading {os.path.basename(path)} ...")
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                raise FileNotFoundError(
                    f"Missing {path}; auto-download failed ({e}). Fetch manually:\n"
                    f"  curl -L -o {xp} {CIFAR101_URLS['data']}\n"
                    f"  curl -L -o {yp} {CIFAR101_URLS['labels']}")
    return np.load(xp), np.load(yp).astype(int)

# ---- data: ImageNet-C (folder of corruption/severity/class/*.JPEG) ----
# Speed: pass DataLoader worker tensors over file descriptors (no shm files on exFAT),
# and raise the fd soft limit so many workers don't trip macOS's low default (256).
try:
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    import resource as _resource
    _s, _h = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (max(_s, min(_h, 8192)), _h))
except Exception:
    pass

class _ICSampledDS(torch.utils.data.Dataset):
    """Precomputed (path, label) list; opens images lazily. Module-level so it is
    picklable for DataLoader workers under macOS 'spawn'."""
    def __init__(self, samples, transform):
        self.samples = samples; self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        p, y = self.samples[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        return self.transform(img), y

def imagenet_c_loader(root, corruption, sev, transform, max_images, dev):
    """Fast loader: sample (path,label) directly instead of indexing the whole
    ~50k-image folder (ImageFolder stats every file -> minutes/cell on exFAT).
    Assumes complete class coverage (sorted wnid == ImageNet class order)."""
    folder = os.path.join(root, corruption, str(sev))
    rng = np.random.default_rng(SEED)
    classes = sorted(d for d in os.listdir(folder)
                     if not d.startswith("._") and os.path.isdir(os.path.join(folder, d)))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    target = max_images if max_images else 10**9
    per_class = max(1, target // max(1, len(classes)) + 1)
    order = list(classes); rng.shuffle(order)
    samples = []
    for c in order:
        if max_images and len(samples) >= max_images * 2:
            break  # gathered plenty to subsample from; stop scanning more folders
        cdir = os.path.join(folder, c)
        try:
            files = [f for f in os.listdir(cdir)
                     if not f.startswith("._") and f.lower().endswith((".jpeg", ".jpg", ".png"))]
        except OSError:
            continue
        if not files:
            continue
        k = min(len(files), per_class)
        for f in rng.choice(files, size=k, replace=False):
            samples.append((os.path.join(cdir, f), cls_to_idx[c]))
    if max_images and len(samples) > max_images:
        samples = [samples[i] for i in rng.choice(len(samples), size=max_images, replace=False)]
    # num_workers MUST be 0: macOS + exFAT can't back DataLoader shared memory
    # (torch_shm_manager "Operation not supported"). The speed win here is the
    # direct sampling above (no 50k-file index), not parallelism.
    return torch.utils.data.DataLoader(_ICSampledDS(samples, transform),
                                       batch_size=256, shuffle=False, num_workers=0)


# ---- ImageNet-C TAR-STREAMING (no extraction; exFAT-128KB-slack-safe) ----
# Standard Zenodo 2235448 grouping of the 19 corruptions into 5 tars. A corruption is
# served from its extracted dir if that dir is present (byte-identical to all prior
# runs); otherwise it is streamed DIRECTLY from its tar via python tarfile, reading
# each image by its stored TarInfo offset (random access, no extraction to disk).
CORRUPTION_TO_TAR = {
    "gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise",
    "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur", "zoom_blur": "blur",
    "snow": "weather", "frost": "weather", "fog": "weather", "brightness": "weather",
    "contrast": "digital", "elastic_transform": "digital", "pixelate": "digital", "jpeg_compression": "digital",
    "speckle_noise": "extra", "gaussian_blur": "extra", "spatter": "extra", "saturate": "extra",
}

def _ic_corruption_tar(ic_root, corruption):
    """Path to the tar containing `corruption` if present under ic_root, else None."""
    grp = CORRUPTION_TO_TAR.get(corruption)
    if not grp:
        return None
    p = os.path.join(ic_root, f"{grp}.tar")
    return p if os.path.isfile(p) else None

def _ic_available(ic_root, corruption, sev=None):
    """True if the corruption (optionally a given severity) is loadable from an
    extracted dir OR from its tar (streaming)."""
    if sev is None:
        if os.path.isdir(os.path.join(ic_root, corruption)):
            return True
    elif os.path.isdir(os.path.join(ic_root, corruption, str(sev))):
        return True
    return _ic_corruption_tar(ic_root, corruption) is not None

class _ICTarDS(torch.utils.data.Dataset):
    """Streams (image, label) straight from an open ImageNet-C tar by stored TarInfo
    offset. num_workers MUST be 0 (single shared handle; exFAT can't back DataLoader shm)."""
    def __init__(self, tar_path, infos, labels, transform):
        self.tar_path = tar_path; self.infos = infos; self.labels = labels
        self.transform = transform; self._tf = None
    def __len__(self):
        return len(self.infos)
    def _tar(self):
        import tarfile
        if self._tf is None:
            self._tf = tarfile.open(self.tar_path, "r")
        return self._tf
    def __getitem__(self, i):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            fobj = self._tar().extractfile(self.infos[i])
            img = Image.open(io.BytesIO(fobj.read())).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        return self.transform(img), self.labels[i]

def _imagenet_c_tar_loader(ic_root, corruption, sev, transform, max_images):
    """Tar-streaming counterpart of imagenet_c_loader: scan the corruption's tar for
    members under `<corruption>/<sev>/<wnid>/`, sample ~balanced across classes, and
    return a DataLoader that reads image bytes directly from the tar (NO extraction).
    Same (path-free) sampling discipline + SEED rng as the extracted-dir loader."""
    import tarfile
    tar_path = _ic_corruption_tar(ic_root, corruption)
    if tar_path is None:
        raise FileNotFoundError(f"no tar for corruption '{corruption}' under {ic_root}")
    rng = np.random.default_rng(SEED)
    prefix = f"{corruption}/{sev}/"
    by_class = {}
    seen_prefix = False
    tfs = tarfile.open(tar_path, "r")          # 'r' = transparent; iterates headers, seeks past data
    for m in tfs:
        name = m.name[2:] if m.name.startswith("./") else m.name
        if not name.startswith(prefix):
            # tar members are corruption-major then severity-major: once this prefix
            # has been collected and we have moved to a different corruption, stop.
            if seen_prefix and not name.startswith(corruption + "/"):
                break
            continue
        seen_prefix = True
        if not m.isfile():
            continue
        parts = name.split("/")
        if len(parts) < 4:
            continue
        wnid = parts[2]; base = parts[-1]
        if base.startswith("._") or not base.lower().endswith((".jpeg", ".jpg", ".png")):
            continue
        by_class.setdefault(wnid, []).append(m)
    classes = sorted(by_class)
    if not classes:
        raise RuntimeError(f"tar {os.path.basename(tar_path)} had 0 members under {prefix}")
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    target = max_images if max_images else 10**9
    per_class = max(1, target // max(1, len(classes)) + 1)
    order = list(classes); rng.shuffle(order)
    infos, labels = [], []
    for c in order:
        ms = by_class[c]
        k = min(len(ms), per_class)
        for j in rng.choice(len(ms), size=k, replace=False):
            infos.append(ms[int(j)]); labels.append(cls_to_idx[c])
        if max_images and len(infos) >= max_images * 2:
            break
    if max_images and len(infos) > max_images:
        sub = rng.choice(len(infos), size=max_images, replace=False)
        infos = [infos[int(i)] for i in sub]; labels = [labels[int(i)] for i in sub]
    return torch.utils.data.DataLoader(_ICTarDS(tar_path, infos, labels, transform),
                                       batch_size=256, shuffle=False, num_workers=0)

def imagenet_c_any_loader(ic_root, corruption, sev, transform, max_images, dev):
    """Dispatch: extracted dir (byte-identical legacy path) if present, else tar-stream."""
    if os.path.isdir(os.path.join(ic_root, corruption, str(sev))):
        return imagenet_c_loader(ic_root, corruption, sev, transform, max_images, dev)
    return _imagenet_c_tar_loader(ic_root, corruption, sev, transform, max_images)

def _ic_cell_key(corr, sev, brn, agn, comp):
    """Cell key. comp=None reproduces the legacy key exactly (byte-identical checkpoints)."""
    base = f"{corr}|s{sev}|{brn}|{agn}"
    return base if comp is None else f"{base}|{comp}"

def _ic_compose_stream(X, Y, comp, bs):
    """Non-iid within-cell adaptation stream on preloaded (X,Y), mirroring the CIFAR
    build_stream_and_eval compositions: 'imbalanced' (~85% one class) and 'single_class'
    (label shift), the canonical Tent/SAR collapse triggers. 'iid' = uniform random.
    Length = bs*8, matching the legacy iid stream."""
    n_total = bs * 8
    N = len(X)
    if comp == "imbalanced":
        classes = torch.unique(Y)
        major = classes[torch.randint(len(classes), (1,)).item()]
        maj = (Y == major).nonzero(as_tuple=True)[0]
        oth = (Y != major).nonzero(as_tuple=True)[0]
        if len(maj) == 0: maj = oth
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
        if len(pool) == 0: pool = torch.arange(N)
        idx = pool[torch.randint(len(pool), (n_total,))]
    else:  # "iid"
        idx = torch.randperm(N)[:n_total]
    return [X[idx[i:i + bs]] for i in range(0, len(idx), bs)]

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

# ---- TTA methods (BN- or LayerNorm-affine), return (adapted_model, update_norm) ----
def _bn_affine_params(m, freeze_layer4=False):
    """TTA-adaptable affine params: BatchNorm affine (ResNet/CNN) OR LayerNorm affine
    (ViT/transformers). Tent/EATA/SAR target whichever the backbone uses.

    freeze_layer4=False (DEFAULT) keeps the original behavior: every BN/LN-affine
    param is adaptable, byte-identical to all prior runs. freeze_layer4=True is the
    official-SAR (Niu et al. 2023) setting: exclude the final ResNet stage ('layer4')
    / ViT top blocks from adaptation. Implemented by skipping any affine module whose
    qualified name lies under 'layer4' (ResNet) or 'encoder_layer_11'/'encoder.layers.11'
    (ViT-B/16 last block) -- only used when the caller opts in."""
    def _in_layer4(name):
        return (".layer4" in ("." + name)) or ("encoder_layer_11" in name) or \
               ("encoder.layers.11" in name) or ("encoder.layers.encoder_layer_11" in name)
    named = dict(m.named_modules())
    mod_to_name = {id(mod): nm for nm, mod in named.items()}
    ps = []
    for mod in m.modules():
        if freeze_layer4 and _in_layer4(mod_to_name.get(id(mod), "")):
            continue  # official-SAR: do not adapt the final block
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.track_running_stats = False; mod.running_mean = None; mod.running_var = None
            if mod.weight is not None: mod.weight.requires_grad_(True); ps.append(mod.weight)
            if mod.bias is not None: mod.bias.requires_grad_(True); ps.append(mod.bias)
        elif isinstance(mod, nn.LayerNorm):   # ViT / transformer TTA target (no running stats)
            if mod.weight is not None: mod.weight.requires_grad_(True); ps.append(mod.weight)
            if mod.bias is not None: mod.bias.requires_grad_(True); ps.append(mod.bias)
    return ps

def _clone_for_tta(base, freeze_layer4=False):
    m = copy.deepcopy(base); m.train()
    for p in m.parameters(): p.requires_grad_(False)
    ps = _bn_affine_params(m, freeze_layer4=freeze_layer4)
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

def sar_adapt(base, stream, steps, lr, num_classes, rho=0.05, margin_e0=None,
              reset_constant_em=0.2, sar_lr=None, freeze_layer4=False):
    """Faithful SAR (Niu et al., ICLR 2023; github.com/mr-eggplant/SAR), ported from
    sar.py/sam.py. Four components of the reference, all present here:
      (1) SAM optimizer wrapping SGD: first_step climbs to w+e(w) and SAVES the per-param
          perturbation so second_step can restore w EXACTLY, then applies the base SGD step
          using the gradient evaluated at w+e(w);
      (2) reliable-sample selection: keep only samples with entropy < E_0 = 0.4*ln(K);
      (3) an EMA of the reliable (second-step) entropy loss as the collapse criterion;
      (4) model recovery: when ema < reset_constant_em, restore BOTH the weights and the
          optimizer state (momentum), and reset the EMA.
    Adapted to this harness's BN/LN-affine param set and shared (lr, steps) budget so SAR is
    compared apples-to-apples with tent/eata. (The only deliberate departure from the
    official repo is that it adapts the same affine params as tent/eata rather than excluding
    ResNet-layer4 / ViT top-blocks, to keep the candidate-method comparison controlled.)"""
    if margin_e0 is None: margin_e0 = 0.4 * math.log(num_classes)             # E_0, Eqn. (2)
    # sar_lr=None (DEFAULT) -> use the shared matched lr (byte-identical to prior runs).
    # Official SAR uses its own lr (2.5e-4) + frozen final block; opt in via flags.
    eff_lr = lr if sar_lr is None else sar_lr
    m, ps, init = _clone_for_tta(base, freeze_layer4=freeze_layer4)
    opt = torch.optim.SGD(ps, lr=eff_lr, momentum=0.9)                        # SAM's base optimizer
    # snapshot for the recovery scheme (faithful copy_model_and_optimizer)
    model_state = copy.deepcopy(m.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    ema = None
    for _ in range(steps):
        for xb in stream:
            xb = xb.contiguous()
            # (2) first forward + reliable-sample selection
            opt.zero_grad()
            ent = _entropy(m(xb).softmax(1)); keep1 = ent < margin_e0
            if keep1.sum() == 0: continue
            ent[keep1].mean().backward()
            # (1) SAM first_step: climb to w+e(w), saving old_p to restore w exactly later
            with torch.no_grad():
                gnorm = sum((p.grad.detach() ** 2).sum() for p in ps if p.grad is not None) ** 0.5
                scale = rho / (gnorm + 1e-12)
                old_p = [p.data.clone() for p in ps]
                for p in ps:
                    if p.grad is not None: p.add_(p.grad * scale)
            # second forward at w+e(w); re-filter the SAME first-reliable subset
            opt.zero_grad()
            ent2 = _entropy(m(xb).softmax(1))[keep1]; keep2 = ent2 < margin_e0
            loss2 = ent2[keep2].mean() if keep2.any() else ent2.mean()
            # (3) EMA of the reliable second-step loss = model-recovery criterion
            if not math.isnan(loss2.item()):
                ema = loss2.item() if ema is None else 0.9 * ema + 0.1 * loss2.item()
            loss2.backward()
            # (1) SAM second_step: restore w exactly, then the base SGD update at the w+e(w) gradient
            with torch.no_grad():
                for p, q in zip(ps, old_p): p.data = q
            opt.step()
            # (4) model recovery on detected collapse: restore weights AND optimizer, reset EMA
            if ema is not None and ema < reset_constant_em:
                with torch.no_grad(): m.load_state_dict(model_state, strict=True)
                opt.load_state_dict(opt_state); ema = None
    return m, _upd_norm(ps, init)

def shot_adapt(base, stream, steps, lr):
    """SHOT-IM (Liang et al., ICML 2020): information maximization on the same BN/LN-affine
    params as tent, so it slots in apples-to-apples as a candidate under the shared (steps,lr)
    budget. Per-batch loss = conditional entropy (minimize: sharpen each prediction) MINUS the
    marginal/diversity entropy (maximize: prevent collapse to a single class). This is the
    label-free test-time variant (no source-clustering pseudo-labels), the standard SHOT TTA
    baseline."""
    m, ps, init = _clone_for_tta(base); opt = torch.optim.Adam(ps, lr=lr)
    for _ in range(steps):
        for xb in stream:
            p = m(xb.contiguous()).softmax(1)
            cond_ent = _entropy(p).mean()                 # minimize per-sample entropy
            pbar = p.mean(0)                              # marginal (mean) prediction
            div_ent = -(pbar * (pbar + 1e-6).log()).sum() # maximize diversity entropy
            loss = cond_ent - div_ent
            opt.zero_grad(); loss.backward(); opt.step()
    return m, _upd_norm(ps, init)

TTA_METHODS = {"tent": tent_adapt, "eata": eata_adapt, "sar": sar_adapt, "shot": shot_adapt}

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


def _mps_free():
    """Release cached MPS memory between cells (prevents creeping OOM on Apple GPUs)."""
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# =============================================================================
#  RUN ONE BENCHMARK
# =============================================================================
def run_cifar_benchmark(which, data_root, dev, methods, corruptions, ckpt=None, quick=False, max_cells=0):
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
    if max_cells and max_cells > 0:
        cells = cells[:max_cells]  # SMOKE-TEST ONLY truncation (0 = full grid, untouched)
        print(f"[{which}] [SMOKE] max_cells={max_cells} -> running only {len(cells)} cell(s)")
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
                if mth == "sar": kwargs.update(sar_lr=SAR_LR, freeze_layer4=SAR_FREEZE_LAYER4)
                adapted, un = fn(model, stream, **kwargs)
                aa = acc_on(adapted, ex, ey, train_mode=True)
                Z = evidence_vector(model, adapted, ex, num_classes, un)
                rows[mth].append(dict(condition=f"{corr}|s{sev}|{brn}|{comp}|{agn}|r{rep}",
                                      Z=Z, a0=a0, aa=aa, regime=label_regime(aa - a0)))
        if (ci + 1) % 10 == 0:
            print(f"   {ci+1}/{len(cells)} cells  ({time.time()-t0:.0f}s)")
    try:  # per-cell evidence dump for scripts/gate_baseline_comparison.py (non-breaking)
        import json as _json
        _cand = "tent" if "tent" in rows else next(iter(rows))
        _json.dump(rows[_cand], open(f"cifar10c_percell_{which}.json", "w"))
        print(f"[{which}] per-cell dump ({_cand}, {len(rows[_cand])} cells) -> cifar10c_percell_{which}.json")
    except Exception as _e:
        print(f"[{which}] per-cell dump skipped: {_e}")
    return clean_acc, rows


def run_cifar101_benchmark(data_root, dev, methods, ckpt=None, quick=False):
    """CIFAR-10.1 NATURAL-shift harmful-TTA cells. Reuses the CIFAR-10 model and the
    identical stream/KGA machinery as CIFAR-10-C; the ONLY change is the test set is a
    real natural distribution shift instead of a synthetic corruption (no severity axis).
    CIFAR-10.1 v6 is small (~2000 imgs) so we split it ~half eval / ~half adapt-stream."""
    num_classes = 10
    model = get_cifar_model("10", data_root, dev, ckpt)
    rng = np.random.default_rng(SEED)
    X, y = load_cifar_101(data_root); y = np.asarray(y).astype(int)
    print(f"[cifar101] loaded {len(y)} natural-shift images (CIFAR-10.1 v6)")
    # base (no-adapt) accuracy on the whole natural set
    Xt = _norm_cifar((torch.tensor(X).permute(0, 3, 1, 2).float() / 255.0).to(dev))
    clean_acc = acc_on(model, Xt, torch.tensor(y), train_mode=False)
    del Xt; _mps_free()
    print(f"[cifar101] base (no-adapt) acc on CIFAR-10.1 = {clean_acc:.3f}")
    ep = max(num_classes * 10, len(y) // 2)     # ~half eval, ~half adaptation stream
    bkeys = ["small", "tiny"] if quick else list(BATCH_REGIMES.keys())
    cells = [(br, comp, agn) for br in bkeys for comp in COMPOSITIONS for agn in AGGRESSIVENESS]
    rows = {m: [] for m in methods}
    print(f"[cifar101] {len(cells)} cells x {N_REPEATS} repeats x {len(methods)} methods")
    t0 = time.time()
    for ci, (brn, comp, agn) in enumerate(cells):
        bs = BATCH_REGIMES[brn]; ag = AGGRESSIVENESS[agn]
        for rep in range(N_REPEATS):
            stream, ex, ey = build_stream_and_eval(X, y, X, y, comp, bs, num_classes, rng, dev, eval_pool=ep)
            a0 = acc_on(model, ex, ey, train_mode=False)
            for mth in methods:
                fn = TTA_METHODS[mth]; kwargs = dict(steps=ag["steps"], lr=ag["lr"])
                if mth in ("eata", "sar"): kwargs["num_classes"] = num_classes
                if mth == "sar": kwargs.update(sar_lr=SAR_LR, freeze_layer4=SAR_FREEZE_LAYER4)
                adapted, un = fn(model, stream, **kwargs)
                aa = acc_on(adapted, ex, ey, train_mode=True)
                Z = evidence_vector(model, adapted, ex, num_classes, un)
                rows[mth].append(dict(condition=f"cifar101|{brn}|{comp}|{agn}|r{rep}",
                                      Z=Z, a0=a0, aa=aa, regime=label_regime(aa - a0)))
                del adapted, un; _mps_free()
        if (ci + 1) % 10 == 0:
            print(f"   {ci+1}/{len(cells)} cells  ({time.time()-t0:.0f}s)")
    return clean_acc, rows


def run_imagenet_benchmark(ic_root, val_root, dev, methods, corruptions, quick=False, max_images=4000, arch="resnet50", severities=None, batch_regimes=None, out_dir=None, cooldown=0.0, compositions=None, max_cells=0):
    # Frozen pretrained ImageNet backbone; Tent/EATA/SAR adapt BN-affine (ResNet) or
    # LayerNorm-affine (ViT) params; eval on the corruption set itself.
    # Writes progress.log + checkpoint.json after EVERY cell, so the run is VISIBLE on disk
    # and RESUMABLE: if the Mac sleeps / overheats / shuts down, just relaunch -- finished
    # cells are skipped and it continues from where it stopped (no work lost).
    present = [c for c in corruptions if _ic_available(ic_root, c)]
    _missing = [c for c in corruptions if c not in present]
    if _missing:
        print(f"[imagenet-c] skipping {len(_missing)} absent corruption(s): {_missing}")
    if not present:
        raise SystemExit(f"[imagenet-c] none of the requested corruptions exist under {ic_root}: {corruptions}")
    corruptions = present
    if arch == "vit_b16":
        weights = tv.models.ViT_B_16_Weights.IMAGENET1K_V1
        model = tv.models.vit_b_16(weights=weights).to(dev).eval()
    else:
        weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
        model = tv.models.resnet50(weights=weights).to(dev).eval()
    tf = weights.transforms()
    num_classes = 1000
    rng = np.random.default_rng(SEED)
    sevs = [1, 5] if quick else (severities if severities else SEVERITIES)
    bregimes = batch_regimes if batch_regimes else [("large_iid", 64), ("tiny", 8)]
    comps = list(compositions) if compositions else [None]   # None => legacy single iid random-perm stream
    inner = [(brn, bs, agn, AGGRESSIVENESS[agn], comp) for (brn, bs) in bregimes for agn in AGGRESSIVENESS for comp in comps]

    rows = {m: [] for m in methods}
    done = set()
    prog_path = ckpt_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        prog_path = os.path.join(out_dir, "progress.log")
        ckpt_path = os.path.join(out_dir, "checkpoint.json")
        if os.path.exists(ckpt_path):
            try:
                ck = json.load(open(ckpt_path))
                rows = {m: ck.get("rows", {}).get(m, []) for m in methods}
                done = set(ck.get("done", []))
                print(f"[imagenet-c] RESUMING from checkpoint: {len(done)} cells already done, skipping them")
            except Exception as e:
                print(f"[imagenet-c] checkpoint unreadable ({e}); starting fresh")

    total = len(corruptions) * len(sevs) * len(inner)
    def _log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if prog_path:
            try:
                with open(prog_path, "a") as f: f.write(line + "\n")
            except Exception:
                pass
    def _checkpoint():
        if not ckpt_path: return
        try:
            tmp = ckpt_path + ".tmp"
            json.dump({"rows": rows, "done": sorted(done), "cells_done": len(done),
                       "cells_total": total, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                      open(tmp, "w"))
            os.replace(tmp, ckpt_path)
        except Exception as e:
            print(f"[imagenet-c] checkpoint write failed: {e}")
    _log(f"START imagenet-c backbone={arch} | {len(corruptions)} corr x {len(sevs)} sev x {len(inner)} (batch x aggr x comp) x {len(methods)} methods = {total} cells | {len(done)}/{total} already done")

    t0 = time.time()
    _ran = 0   # cells processed THIS call (for --max-cells smoke cap)
    for corr in corruptions:
        for sev in sevs:
            keys_here = [_ic_cell_key(corr, sev, brn, agn, comp) for (brn, bs, agn, ag, comp) in inner]
            if keys_here and all(k in done for k in keys_here):
                continue  # whole (corruption,severity) finished -> don't even load its images
            if not _ic_available(ic_root, corr, sev):
                _log(f"skip {corr} sev{sev} (no extracted dir and no tar)"); continue
            loader = imagenet_c_any_loader(ic_root, corr, sev, tf, max_images, dev)
            xs, ys = [], []
            for xb, yb in loader: xs.append(xb); ys.append(yb)
            X = torch.cat(xs).to(dev); Y = torch.cat(ys)
            for (brn, bs, agn, ag, comp) in inner:
                key = _ic_cell_key(corr, sev, brn, agn, comp)
                if key in done:
                    continue
                if comp is None:
                    perm = torch.randperm(len(X))[: bs * 8]
                    stream = [X[perm[i:i+bs]] for i in range(0, len(perm), bs)]
                else:
                    stream = _ic_compose_stream(X, Y, comp, bs)
                a0 = acc_on(model, X, Y, train_mode=False)
                cell_msg = []
                for mth in methods:
                    fn = TTA_METHODS[mth]; kw = dict(steps=ag["steps"], lr=ag["lr"])
                    if mth in ("eata", "sar"): kw["num_classes"] = num_classes
                    if mth == "sar": kw.update(sar_lr=SAR_LR, freeze_layer4=SAR_FREEZE_LAYER4)
                    adapted, un = fn(model, stream, **kw)
                    aa = acc_on(adapted, X, Y, train_mode=True)
                    Z = evidence_vector(model, adapted, X[:1024], num_classes, un)
                    rows[mth].append(dict(condition=key, Z=Z, a0=a0, aa=aa, regime=label_regime(aa - a0)))
                    cell_msg.append(f"{mth} a0={a0:.3f} aa={aa:.3f} dB={aa-a0:+.3f}")
                    del adapted, un; _mps_free()
                done.add(key)
                _checkpoint()
                _ran += 1
                _log(f"[{len(done)}/{total}] {key} | " + " | ".join(cell_msg) + f" | {time.time()-t0:.0f}s")
                if max_cells and _ran >= max_cells:
                    _log(f"[smoke] --max-cells {max_cells} reached; stopping early (resumable)")
                    del X, Y; _mps_free()
                    return None, rows
                if cooldown > 0:
                    time.sleep(cooldown)
            del X, Y; _mps_free()
    _log(f"DONE imagenet-c: {len(done)}/{total} cells")
    return None, rows


# =============================================================================
#  MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="K-Bound decisive deep-TTA benchmark v2")
    ap.add_argument("--benchmarks", nargs="+", default=["cifar10c"],
                    choices=["cifar10c", "cifar100c", "cifar101", "imagenetc"])
    ap.add_argument("--methods", nargs="+", default=["tent", "eata", "sar"],
                    choices=["tent", "eata", "sar"])
    ap.add_argument("--data-root", default="experiments/kbound/cifar",
                    help="dir holding CIFAR-10-C/ and CIFAR-100-C/ and model ckpts")
    ap.add_argument("--cifar10-ckpt", default=None)
    ap.add_argument("--cifar100-ckpt", default=None)
    ap.add_argument("--imagenetc-root", default=None, help="ImageNet-C root (corruption/severity/class/*)")
    ap.add_argument("--corruptions", nargs="+", default=None,
                    help="ImageNet-C corruptions to run (default: representative quick subset). "
                         "e.g. --corruptions gaussian_noise shot_noise impulse_noise")
    ap.add_argument("--max-images", type=int, default=4000,
                    help="ImageNet-C images sampled per corruption/severity cell (smoke test: 64)")
    ap.add_argument("--severities", type=int, nargs="+", default=None,
                    help="ImageNet-C severities (default 1,3,5). Use 1 2 3 4 5 for the full sweep.")
    ap.add_argument("--all-batch", action="store_true",
                    help="ImageNet-C: use 3 batch regimes (large=128, small=16, tiny=8) instead of 2")
    ap.add_argument("--out-results", default="experiments/kbound/results")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds to sleep between cells (thermal relief for hot laptops; e.g. 3)")
    ap.add_argument("--out-figs", default="docs/research/kbound/figures")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--quick", action="store_true", help="subset of corruptions/severities (fast smoke run)")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "vit_b16"],
                    help="ImageNet-C backbone: resnet50 (BN-affine Tent) or vit_b16 (LayerNorm-affine Tent)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="run seed; sets ALL rng (torch/np/random + batch-composition sampler). "
                         "Each value is a full independent grid replicate.")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="SMOKE-TEST ONLY: cap number of grid cells (0 = no cap = full grid). "
                         "Leave at 0 for any real/confirmatory run so the grid stays byte-identical.")
    ap.add_argument("--sar-lr", type=float, default=None, dest="sar_lr",
                    help="SAR-arm learning rate. DEFAULT None = use the shared matched lr "
                         "(byte-identical to all prior runs). Set 2.5e-4 for the official "
                         "gentle SAR schedule (Protocol E). Only affects the 'sar' method.")
    ap.add_argument("--sar-freeze-layer4", action="store_true", dest="sar_freeze_layer4",
                    help="SAR-arm only: freeze the final block (ResNet layer4 / ViT top block) "
                         "during adaptation, per Niu et al. 2023 official SAR. DEFAULT off "
                         "(adapts all affine params, identical to prior runs).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="compute device. DEFAULT auto = pick_device() (mps>cuda>cpu). Use cpu "
                         "for a no-GPU smoke that won't contend with another MPS job.")
    ap.add_argument("--imagenetc-composition", nargs="+", default=None, dest="imagenetc_composition",
                    choices=["iid", "imbalanced", "single_class"],
                    help="ImageNet-C within-cell stream composition axis (sequential/non-iid). "
                         "DEFAULT None = legacy single iid random-perm stream (byte-identical to "
                         "prior runs). Protocol E (full): --imagenetc-composition iid imbalanced single_class.")
    args = ap.parse_args()

    os.makedirs(args.out_results, exist_ok=True); os.makedirs(args.out_figs, exist_ok=True)
    dev = (pick_device() if args.device == "auto" else args.device); print("device:", dev)
    set_global_seed(args.seed); print("seed:", SEED)
    global SAR_LR, SAR_FREEZE_LAYER4
    SAR_LR = args.sar_lr; SAR_FREEZE_LAYER4 = args.sar_freeze_layer4
    if SAR_LR is not None or SAR_FREEZE_LAYER4:
        print(f"[SAR-official] sar_lr={SAR_LR} freeze_layer4={SAR_FREEZE_LAYER4} "
              f"(non-default SAR arm; tent/eata unchanged)")
    _t_start = time.time()

    combined = {"alpha": args.alpha, "device": dev, "seed": SEED, "benchmarks": {},
                "generated": time.strftime("%Y-%m-%d %H:%M")}

    for bench in args.benchmarks:
        if bench in ("cifar10c", "cifar100c"):
            which = "10" if bench == "cifar10c" else "100"
            corrs = (CIFAR_C_QUICK if args.quick else CIFAR_C_ALL)
            ck = args.cifar10_ckpt if which == "10" else args.cifar100_ckpt
            clean_acc, rows = run_cifar_benchmark(which, args.data_root, dev, args.methods, corrs, ck, args.quick, max_cells=args.max_cells)
        elif bench == "cifar101":
            clean_acc, rows = run_cifar101_benchmark(args.data_root, dev, args.methods, args.cifar10_ckpt, args.quick)
        else:
            if not args.imagenetc_root: print("[skip] imagenetc needs --imagenetc-root"); continue
            corrs = args.corruptions if args.corruptions else IMAGENET_C_QUICK
            _bregimes = [("large_iid", 64), ("small", 16), ("tiny", 8)] if args.all_batch else None
            clean_acc, rows = run_imagenet_benchmark(args.imagenetc_root, None, dev, args.methods, corrs, args.quick, max_images=args.max_images, arch=args.arch, severities=args.severities, batch_regimes=_bregimes, out_dir=args.out_results, cooldown=args.cooldown, compositions=args.imagenetc_composition, max_cells=args.max_cells)

        per_method = {}
        for mth, rws in rows.items():
            if len(rws) < 8:
                print(f"[warn] {bench}/{mth}: only {len(rws)} conditions");
            metrics, detail = summarize(rws, alpha=args.alpha)
            # ---- PER-CONDITION serialization (protocol serialization_contract) ----
            # Join each condition's raw record with its KGA outputs and write one
            # JSON object per (condition x seed), including condition-level audit fields.
            # This v2 runner is single-candidate (frozen vs adapted), so tau is
            # degenerate (0.0 by construction; no pairwise-agreement residual).
            _per_cond = []
            for i, r in enumerate(rws):
                b_hat_i = float(detail["Bhat"][i])
                eps_i = float(detail["eps"])
                lb_i = b_hat_i - eps_i
                ub_i = b_hat_i + eps_i
                if lb_i > 0:
                    zone_i = "CERTIFIED_ADAPT"
                elif ub_i < 0:
                    zone_i = "CERTIFIED_FREEZE"
                else:
                    zone_i = "BLIND"
                kga_acc_i = float(r["aa"] if detail["dec"][i] == "ADAPT" else r["a0"])
                _per_cond.append({
                    "seed": SEED, "benchmark": bench, "method": mth,
                    "condition": r["condition"],
                    "B": float(detail["B"][i]),            # true benefit aa - a0
                    "a0": float(r["a0"]),                  # frozen accuracy
                    "a_adapted": float(r["aa"]),           # adapted accuracy
                    "a_kbound": kga_acc_i,
                    "a_oracle": float(max(r["a0"], r["aa"])),
                    "regime": r["regime"],                 # helpful/harmful/marginal (oracle action proxy)
                    "oracle_action": ("ADAPT" if r["aa"] > r["a0"] else "FREEZE"),
                    "Z": [float(z) for z in r["Z"]],       # label-free evidence vector
                    "Z_names": EVIDENCE_NAMES,
                    "n_D": None,                           # not computed by v2 KGA
                    "c_ij": None,                          # pairwise agreements: not computed by v2 KGA
                    "tau_hat": 0.0,                        # single-candidate route => identically zero
                    "tau_star": 0.0,                       # same degenerate route convention
                    "b_hat": b_hat_i,                      # GB leave-one-out benefit estimate
                    "eps_conformal": eps_i,                # split-conformal certificate radius
                    "benefit_ci": [float(lb_i), float(ub_i)],
                    "zone": zone_i,                        # CERTIFIED_ADAPT / CERTIFIED_FREEZE / BLIND
                    "gamma_hat": float(0.5 * b_hat_i),     # single-candidate proxy: gamma ~ b_hat/2
                    "gamma_ci": [float(0.5 * lb_i), float(0.5 * ub_i)],
                    "kga_decision": str(detail["dec"][i]), # ADAPT/FREEZE/ABSTAIN
                })
            _pc_path = os.path.join(args.out_results, f"per_condition_{bench}_{mth}_seed{SEED}.json")
            with open(_pc_path, "w") as _f:
                json.dump({"seed": SEED, "benchmark": bench, "method": mth,
                           "alpha": args.alpha, "n_conditions": len(_per_cond),
                           "per_condition_fields_absent": ["n_D", "c_ij"],
                           "per_condition_fields_absent_reason":
                               "v2 KGA = gradient-boosted B_hat(Z) + split-conformal eps; "
                               "it does not compute pairwise agreements c_ij or n_D.",
                           "per_condition_field_notes": {
                               "tau_hat_tau_star":
                                   "single-candidate route (frozen vs adapted), so tau is degenerate and set to 0.0.",
                               "zone":
                                   "computed from benefit_ci: lb>0 => CERTIFIED_ADAPT, ub<0 => CERTIFIED_FREEZE, else BLIND.",
                               "gamma_hat":
                                   "single-candidate proxy (b_hat/2) with gamma_ci = benefit_ci/2."
                           },
                           "records": _per_cond}, _f, indent=2)
            print(f"  [serialize] wrote {len(_per_cond)} per-condition records -> {_pc_path}")
            per_method[mth] = {"metrics": metrics, "n_conditions": len(rws),
                               "per_condition_file": os.path.basename(_pc_path),
                               "conditions": [r["condition"] for r in rws],
                               "detail": {"Bhat": detail["Bhat"].tolist() if hasattr(detail["Bhat"], "tolist") else detail["Bhat"],
                                          "eps": float(detail["eps"]),
                                          "dec": list(detail["dec"]),
                                          "B": detail["B"].tolist() if hasattr(detail["B"], "tolist") else detail["B"]}}
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

    # ---- per-seed manifest (protocol serialization_contract: result_manifest.json) ----
    def _git_hash():
        try:
            import subprocess
            return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=os.path.dirname(os.path.abspath(__file__)),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    manifest = {
        "seed": SEED, "alpha": args.alpha, "device": dev,
        "benchmarks": args.benchmarks, "methods": args.methods,
        "quick": args.quick, "max_cells": args.max_cells,
        "sar_arm": {"sar_lr": SAR_LR, "freeze_layer4": SAR_FREEZE_LAYER4,
                    "default_matched_lr_behavior": (SAR_LR is None and not SAR_FREEZE_LAYER4)},
        "grid": {"severities": SEVERITIES, "quick_severities": [1, 5],
                 "batch_regimes": BATCH_REGIMES, "compositions": COMPOSITIONS,
                 "aggressiveness": AGGRESSIVENESS, "n_repeats": N_REPEATS,
                 "eval_pool": EVAL_POOL, "help_thr": HELP_THR},
        "n_conditions_per_method": {b: {m: combined["benchmarks"][b]["methods"][m]["n_conditions"]
                                        for m in combined["benchmarks"][b]["methods"]}
                                    for b in combined["benchmarks"]},
        "git_hash": _git_hash(),
        "python": sys.version.split()[0],
        "torch": (torch.__version__ if _HAS_TORCH else None),
        "numpy": np.__version__,
        "argv": sys.argv,
        "wall_time_sec": round(time.time() - _t_start, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    man_path = os.path.join(args.out_results, "result_manifest.json")
    json.dump(manifest, open(man_path, "w"), indent=2)
    print("Saved", man_path, f"(wall {manifest['wall_time_sec']}s)")


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
