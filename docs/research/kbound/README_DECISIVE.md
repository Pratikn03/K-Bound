# The Decisive Deep-TTA Benchmark (closing the last gap)

This is the one experiment the paper still needs: showing the KGA
**adapt / freeze / abstain** certificate beats **both** trivial policies
(always-adapt and always-freeze) in a realistic deep test-time-adaptation setting.

Script: `src/scripts/kbound/cifar_tent_mps_v2.py`
Run it on your **M5 (MPS)** or any CUDA box — it needs PyTorch + a GPU; the sandbox
cannot run it (no GPU/torch, 45s cap). The analysis core was unit-tested off-GPU.

---

## Why v1 (`cifar_tent_mps.py`) did not show the result
From `experiments/kbound/results/cifar_tent_results.json`:
`base_rate_harmful_B<0 = 0.023`, `mean_true_B = +0.20`. The suite was ~98% **helpful**,
so `always_adapt (0.585) ≈ oracle (0.586)` and KGA could only tie it. v1 also used gentle
Tent (10 steps, lr 1e-3) on toy on-the-fly corruptions, and measured benefit on the
adaptation batch itself.

## What v2 changes
1. **Real** CIFAR-10-C / CIFAR-100-C / ImageNet-C corruptions.
2. A **pre-registered mixed grid** that deliberately spans regimes, including documented
   Tent-collapse cells: `severity {1,3,5} × batch {large-iid 200, small 16, tiny 8} ×
   composition {iid, imbalanced, single-class/label-shift} × aggressiveness {mild,
   aggressive (50 steps, lr 2.5e-3)}`. The aggressive × tiny/single-class × severe cells
   genuinely collapse.
3. **Honest benefit**: adapt on the (nasty) stream, but measure accuracy on a separate
   **class-balanced held-out eval set** for the same corruption — so collapse = real loss.
4. **Tent, EATA, SAR** as candidate adaptations (KGA wraps each).
5. **KGA is identical to the rest of the paper**: leave-one-condition-out gradient-boosted
   `B̂(Z)` + split-conformal radius `ε`; ADAPT if `B̂−ε>0`, FREEZE if `B̂+ε<0`, else ABSTAIN.
   `Z` is label-free only (entropy, confidence, predicted-class balance pre/post, balance
   drop, entropy drop, fraction-high-confidence, marginal-prediction KL, update norm).
6. The reviewer-proof figure: a **mixing-ratio Pareto sweep** — vary the harmful fraction
   `p` of the deployment stream and show KGA's regret ≤ both baselines for all `p`
   (ties always-adapt at `p=0`, ties always-freeze at `p=1`, strictly beats both in between).

---

## 1. Setup
```bash
cd AutoML_Flagship_V8
python -m venv .venv && source .venv/bin/activate     # or your existing env
pip install torch torchvision numpy scikit-learn matplotlib
python -c "import torch; print('MPS', torch.backends.mps.is_available())"   # expect True on M5
```

## 2. Get the corruption datasets
Standard Hendrycks benchmarks (verify the URLs before downloading):

| Dataset | Zenodo record | File | Size |
|---|---|---|---|
| CIFAR-10-C | `zenodo.org/records/2535967` | `CIFAR-10-C.tar` | ~2.8 GB |
| CIFAR-100-C | `zenodo.org/records/3555552` | `CIFAR-100-C.tar` | ~2.9 GB |
| ImageNet-C | `zenodo.org/records/2235448` (+ extra tars) | per-category tars | ~100+ GB |

Extract so the layout is:
```
experiments/kbound/cifar/
  CIFAR-10-C/   gaussian_noise.npy  ...  labels.npy
  CIFAR-100-C/  gaussian_noise.npy  ...  labels.npy
  resnet18_cifar.pt            # you already have this (CIFAR-10)
  resnet18_cifar100.pt         # optional; script trains one if absent
ImageNet-C/                    # anywhere; pass via --imagenetc-root
  gaussian_noise/1/<class>/*.JPEG ...
```
(`.npy` corruption files are 50000×32×32×3 = 5 severities × 10000.)

## 3. Run
```bash
# fast smoke test (subset of corruptions + severities 1,5) — do this first
python src/scripts/kbound/cifar_tent_mps_v2.py --benchmarks cifar10c --quick

# full CIFAR-10-C + CIFAR-100-C, all methods
python src/scripts/kbound/cifar_tent_mps_v2.py \
    --benchmarks cifar10c cifar100c --methods tent eata sar

# add ImageNet-C once on disk (heavy)
python src/scripts/kbound/cifar_tent_mps_v2.py \
    --benchmarks imagenetc --imagenetc-root /path/to/ImageNet-C
```
Runtime: `--quick` CIFAR-10-C is minutes on M5; the full CIFAR grid (×3 methods) is
roughly 1–3 h; ImageNet-C is the heavy one — start with a few corruptions.

## 4. Outputs
- `experiments/kbound/results/decisive_tta_results.json` — full metrics per benchmark/method.
- `experiments/kbound/results/decisive_tta_table.md` — the headline table.
- `docs/research/kbound/figures/fig_decisive_{regret,pareto,decisions}_{benchmark}.png`.

Each method reports: harmful base rate, mean accuracy and **regret vs oracle** for
always-adapt / always-freeze / **K-Bound** / oracle, worst-case accuracy, adapt-precision,
false-adapt rate, decisions-by-true-regime, the Pareto curve, and a `beats_both` flag.

## 5. What success looks like (and the honest reading)
- On a balanced mixed grid you should see **`beats_both = YES`** for Tent (most
  collapse-prone), and KGA on the Pareto front for EATA/SAR too.
- The Pareto curve gives the precise claim: *"for any deployment stream with harmful
  fraction `p ≳ p*`, KGA strictly beats both trivial policies; it ties always-adapt at
  `p=0` and always-freeze at `p=1`."* Report `p*`, do **not** claim a single cherry-picked mean.
- If a method (e.g. SAR, with its reset) rarely collapses, report that honestly — KGA ties
  it there. KGA's value scales with how often catastrophic, detectable harm occurs, which is
  exactly what the theory says.

## 6. Folding it into the paper
- Replaces the §8 limitation "we could not run the catastrophic-harm deep-TTA benchmark."
- Add `decisive_tta_table.md` as a results table and `fig_decisive_pareto_*` as a figure.
- Update `EXECUTION_STATUS.md` (Phase 4 → done) and the abstract's scope sentence.

## Honesty / pre-registration notes
- The grid is declared **before** any result is seen; we report the full stratified
  breakdown — no condition is dropped to flatter the method.
- True labels are used **only** to compute `B` and the oracle (for evaluation). KGA sees
  label-free `Z` only.
- EATA/SAR here are faithful re-implementations (BN-affine, entropy filtering, SAM,
  entropy-reset), not the official repos; swap them in for camera-ready if you prefer.
- Every number is produced by your run. Nothing is fabricated.
