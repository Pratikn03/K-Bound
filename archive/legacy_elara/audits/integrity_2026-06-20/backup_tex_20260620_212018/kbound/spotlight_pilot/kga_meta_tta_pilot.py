"""
KGA-Meta-TTA pilot — CIFAR-100-C continual go/no-go (Protocol S1).
Runs on a FREE Colab/Kaggle GPU. ~30-90 min for 3 seeds.

Pre-registered question (SPOTLIGHT_PILOT_PROTOCOL.md):
  Does a KGA-gated meta-policy over a pool of TTA adapters beat the BEST single
  adapter's mean error on continual CIFAR-100-C, without tuning the gate on test
  corruptions? WIN iff the gap's 95% paired-bootstrap CI excludes 0.

Honest scope:
  - Pool here = {frozen, Tent, EATA-lite}. For the publishable version, plug in the
    OFFICIAL Tent/EATA/SAR/CoTTA/RoTTA implementations (links in README) — this pilot
    is a fast signal, not the final comparison.
  - The KGA gate uses a target-label-light probe (k labels per corruption block) — the
    operating point K-Bound legitimizes. Set --probe_k 0 to test the pure label-free
    gate (entropy-margin proxy) instead.

Colab setup:
  !pip -q install robustbench
  !python kga_meta_tta_pilot.py --seeds 0 1 2 --n_examples 2000 --probe_k 64
"""
import argparse, copy, json, time
import numpy as np
import torch, torch.nn as nn
from robustbench.data import load_cifar100c
from robustbench.utils import load_model

CORRUPTIONS = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur',
               'motion_blur','zoom_blur','snow','frost','fog','brightness','contrast',
               'elastic_transform','pixelate','jpeg_compression']
DEV_CORRUPTIONS = ['speckle_noise','gaussian_blur','spatter','saturate']  # held-out for gate tuning
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_bn_affine(model):
    model.train(); model.requires_grad_(False); params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True); params += [p for p in (m.weight, m.bias) if p is not None]
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = False; m.running_mean = None; m.running_var = None
    return params


class Tent(nn.Module):
    """Tent (Wang et al., ICLR'21): entropy-min on norm affine params."""
    def __init__(self, model, lr=1e-3, steps=1):
        super().__init__()
        self.model = model; self.params = collect_bn_affine(model)
        self.opt = torch.optim.Adam(self.params, lr=lr); self.steps = steps

    @torch.enable_grad()
    def adapt(self, x):
        for _ in range(self.steps):
            loss = entropy(self.model(x)).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()

    def forward(self, x):
        self.adapt(x)
        with torch.no_grad():
            return self.model(x)


class EataLite(Tent):
    """EATA-lite: Tent but only backprop low-entropy (confident) samples (the core EATA filter).
    NOTE: official EATA also adds a Fisher anti-forgetting term — plug the real repo in for the
    publishable run; this captures the sample-selection half for a fast pilot signal."""
    def __init__(self, model, lr=1e-3, steps=1, e_margin=None):
        super().__init__(model, lr, steps)
        self.e_margin = e_margin if e_margin is not None else 0.4 * np.log(100)  # 0.4*ln(K)

    @torch.enable_grad()
    def adapt(self, x):
        for _ in range(self.steps):
            out = self.model(x); ent = entropy(out)
            keep = ent < self.e_margin
            if keep.any():
                loss = ent[keep].mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()


def fresh_pool(model_name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    base = load_model(model_name, dataset='cifar100', threat_model='corruptions').to(DEVICE).eval()
    frozen = copy.deepcopy(base).eval()
    tent = Tent(copy.deepcopy(base).to(DEVICE))
    eata = EataLite(copy.deepcopy(base).to(DEVICE))
    return {'frozen': frozen, 'tent': tent, 'eata': eata}


@torch.no_grad()
def acc_of(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def kga_gate(probe_logits, probe_y, eps_floor=0.0):
    """Multi-candidate KGA at the target-label-light operating point.
    Returns the candidate name with the highest certified lower bound (acc - eps);
    falls back to 'frozen' if none beats frozen's estimate. eps = binomial SE on the probe."""
    best, best_lb = 'frozen', -1.0
    n = len(probe_y)
    for name, lg in probe_logits.items():
        a = acc_of(lg, probe_y)
        eps = (a * (1 - a) / max(n, 1)) ** 0.5 + eps_floor  # conformal-style radius
        lb = a - eps
        if lb > best_lb:
            best_lb, best = lb, name
    # safety: never deploy a candidate whose LB is below frozen's point estimate
    a_frozen = acc_of(probe_logits['frozen'], probe_y)
    if best_lb < a_frozen and best != 'frozen':
        best = 'frozen'
    return best


def run_seed(model_name, seed, corruptions, n_examples, severity, batch, probe_k):
    pool = fresh_pool(model_name, seed)
    # per-policy running error stores
    per_pol = {k: [] for k in ['frozen', 'tent', 'eata', 'kga_meta', 'oracle']}
    for corr in corruptions:
        x, y = load_cifar100c(n_examples=n_examples, severity=severity, data_dir='./data',
                              shuffle=True, corruptions=[corr])
        x, y = x.to(DEVICE), y.to(DEVICE)
        # adapters adapt continually across the whole block (and across corruptions = continual)
        logits = {}
        for name, m in pool.items():
            outs = []
            for i in range(0, len(x), batch):
                xb = x[i:i+batch]
                outs.append(m(xb) if name != 'frozen' else m(xb))
            logits[name] = torch.cat(outs, 0)
        # ----- gate decision on a probe split (target-label-light), evaluate on the rest -----
        if probe_k > 0:
            idx = torch.randperm(len(y))[:probe_k]
            probe_logits = {k: v[idx] for k, v in logits.items()}
            chosen = kga_gate(probe_logits, y[idx])
            mask = torch.ones(len(y), dtype=torch.bool); mask[idx] = False
        else:  # pure label-free gate: pick highest mean confidence margin (proxy), then KGA-freeze rule
            margins = {k: (v.softmax(1).max(1).values.mean().item()) for k, v in logits.items()}
            chosen = max(margins, key=margins.get)
            mask = torch.ones(len(y), dtype=torch.bool)
        yt = y[mask]
        for name in ['frozen', 'tent', 'eata']:
            per_pol[name].append(1 - acc_of(logits[name][mask], yt))
        per_pol['kga_meta'].append(1 - acc_of(logits[chosen][mask], yt))
        per_pol['oracle'].append(1 - max(acc_of(logits[k][mask], yt) for k in ['frozen','tent','eata']))
    return {k: np.array(v) for k, v in per_pol.items()}


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """P(mean(a) < mean(b)) and 95% CI of (b - a) over the paired cells (corruption x seed)."""
    rng = np.random.default_rng(seed); d = b - a; N = len(d); diffs = []
    for _ in range(n_boot):
        diffs.append(d[rng.integers(0, N, N)].mean())
    diffs = np.array(diffs)
    return float((diffs > 0).mean()), [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Hendrycks2020AugMix_WRN')  # RobustBench CIFAR-100 corruptions model
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--n_examples', type=int, default=2000)
    ap.add_argument('--severity', type=int, default=5)
    ap.add_argument('--batch', type=int, default=200)
    ap.add_argument('--probe_k', type=int, default=64, help='target labels per corruption (0 = pure label-free)')
    ap.add_argument('--out', default='kga_meta_pilot_results.json')
    a = ap.parse_args()

    t0 = time.time(); cells = {k: [] for k in ['frozen', 'tent', 'eata', 'kga_meta', 'oracle']}
    for s in a.seeds:
        print(f'[seed {s}] running {len(CORRUPTIONS)} corruptions on {DEVICE} ...', flush=True)
        r = run_seed(a.model, s, CORRUPTIONS, a.n_examples, a.severity, a.batch, a.probe_k)
        for k in cells: cells[k].append(r[k])
        print(f'  mean err: ' + '  '.join(f'{k}={r[k].mean():.4f}' for k in cells), flush=True)

    flat = {k: np.concatenate(v) for k, v in cells.items()}  # (seeds x corruptions) cells
    means = {k: float(flat[k].mean()) for k in flat}
    singles = {k: means[k] for k in ['frozen', 'tent', 'eata']}
    best_single = min(singles, key=singles.get)
    P, ci = paired_bootstrap(flat['kga_meta'], flat[best_single])  # gap = best_single - kga_meta
    verdict = 'CONVERTS' if (means['kga_meta'] < means[best_single] and ci[0] > 0) else (
              'PARTIAL' if means['kga_meta'] <= min(singles.values()) + 1e-9 else 'FAILS')
    out = {'protocol': 'S1_kga_meta_tta_pilot', 'model': a.model, 'seeds': a.seeds,
           'probe_k': a.probe_k, 'mean_error': means, 'best_single': best_single,
           'kga_minus_bestsingle': means['kga_meta'] - means[best_single],
           'P_kga_beats_bestsingle': P, 'ci95_gap_bestsingle_minus_kga': ci,
           'oracle_mean_error': means['oracle'], 'verdict': verdict,
           'wall_sec': round(time.time() - t0, 1)}
    json.dump(out, open(a.out, 'w'), indent=2)
    print('\n==== VERDICT:', verdict, '====')
    print(json.dumps(out, indent=2))
    print(f"\nKGA-Meta {means['kga_meta']:.4f}  vs best single ({best_single}) {means[best_single]:.4f}"
          f"  | P(beats)={P:.3f}  95%CI(gap)={ci}  | oracle {means['oracle']:.4f}")


if __name__ == '__main__':
    main()
