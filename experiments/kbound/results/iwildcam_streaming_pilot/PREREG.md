# Pre-registration — iWildCam native-order online-Tent collapse pilot

**Project:** K-Bound (label-free freeze gate for test-time adaptation)
**Pilot type:** falsification / de-risking. We are trying to *break* the hypothesis,
not confirm it.
**Date registered:** 2026-06-30
**Runner:** `experiments/kbound/wilds/run_iwildcam_streaming_pilot.py`
**Outputs:** `experiments/kbound/results/iwildcam_streaming_pilot/`
**Status when registered:** protocol + decision rule fixed; smoke test (30 batches,
val OOD, native order, bs16) already executed end-to-end and is reported below. No
existing committed result is modified by this pilot.

---

## 1. Hypothesis (to be falsified)

On iWildCam's held-out (OOD) cameras, when the test stream is served in **native
temporal / location order** with a **small online batch**, naive **online Tent**
(entropy-minimization TTA with state carried across the whole stream) **COLLAPSES**:
its cumulative macro-F1 falls **below** the frozen ERM source model. If true, a
label-free freeze gate can "beat both" (never below source, sometimes above), which
breaks the project's current empirical ceiling.

A second, subordinate claim: the collapse is **visible in label-free signals**
(prediction entropy, predicted-class-histogram diversity, gradient L2-norm), so a gate
can in principle detect it without labels.

## 2. Source model (frozen baseline) — fixed in advance

- ERM ResNet-50, full fine-tune from ImageNet, trained by `train_iwildcam_f0.py`.
- Checkpoint: `experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt`.
- Selected on **id_val** macro-F1 (source-side, leakage-free); OOD val/test never seen
  in training. Recorded source numbers: id_val macro-F1 = 0.408, OOD-val macro-F1 = 0.260.
- This is the *only* model; both policies start from it. No retraining in this pilot.

## 3. Data & stream order — NATIVE order is mandatory

- Dataset: WILDS iWildCam v2.0, on disk at `experiments/kbound/data/wilds/iwildcam_v2.0`.
- Stream split: **OOD test** (`Test (OOD/Trans)`, N≈35,370 present, 48 cameras,
  102 classes) for the headline run; **OOD val** (`Validation (OOD/Trans)`, N≈12,409
  present, 32 cameras, 75 classes) as a secondary confirmation.
- **Native order = sort by `(location, year, month, day, hour, minute, second, sequence)`**
  from WILDS `metadata_array`. Consecutive batches are therefore temporally and
  location correlated — the real deployment regime that makes online TTA collapse-prone.
- `--order shuffled` (a fixed random permutation of the same samples) is a **control
  only**. The win/finding must come from the real temporal structure, never from
  shuffling. We will NOT report a collapse obtained only under shuffling as support for
  the hypothesis.
- Disk-present filtering reuses the existing in-memory present-cache; a robust,
  order-preserving reader substitutes the next decodable temporal neighbour for the rare
  missing/corrupt JPEG (labels are read from the loader batch, so alignment is exact).

## 4. Policies (same stream, both start from f0)

- **(a) FROZEN source.** f0 in eval mode (BN running stats), predict each batch, no update.
- **(b) ONLINE Tent.** ONE model, BN/LN-affine params trainable, BN running stats off,
  Adam, entropy-minimization loss, **1 gradient step per batch, state carried across the
  whole stream** (the collapse-prone online mode). Per batch we predict **before** the
  update, so frozen and Tent see identical inputs. Tent primitives are the project's
  faithful `tta_methods` implementation (`_clone_for_tta` / `_bn_affine_params` /
  `_entropy`), identical to the validated CIFAR / ImageNet-C / Camelyon harness.
- Tent hyperparameters fixed in advance: lr = 1e-3 (Tent default), 1 step/batch.

## 5. Metric (official) — fixed in advance

- **iWildCam official macro-F1**, via `sklearn.metrics.f1_score(average="macro")`
  (identical to the repo's `macro_f1`). Computed **cumulatively** over the whole stream
  and **per window** (every 50 batches). The headline quantity is the **cumulative
  macro-F1 over the entire OOD stream** — not any hand-picked window.

## 6. Label-free signals logged (Tent run, per window)

- mean softmax entropy of Tent's pre-update predictions,
- entropy of the predicted-class histogram (diversity; collapse → few classes → low H),
- number of unique predicted classes,
- mean gradient L2-norm of the Tent step.
These are KGA's evidence Z and the collapse detector.

## 7. DECISION RULE (registered, not to be changed post hoc)

Let Δ = (Tent-online cumulative macro-F1) − (frozen-source cumulative macro-F1) over the
**full OOD stream in native order**, with a **paired sample bootstrap** 95% CI on Δ
(resample stream samples, recompute both macro-F1s; n_boot ≥ 1000).

- **COLLAPSE CONFIRMED** if **Δ < 0** by a margin whose bootstrap 95% CI **excludes 0**
  (i.e. CI upper bound < 0). → The empirical premise of the freeze gate holds on a real
  natural shift. **Proceed to the full 3-policy KGA experiment** (frozen / online-Tent /
  label-free-gated) on this stream.
- **FALSIFIED** otherwise (Δ ≥ 0, or the CI for Δ includes 0). → The collapse premise
  does not hold here. **Report this honestly and stop**; do not manufacture a collapse by
  shuffling, tuning Tent to be unstable, or cherry-picking a window.

Headline split for the decision is **OOD test**. OOD val is a secondary confirmation; if
test and val disagree, both numbers are reported and the weaker (test) governs the decision.

Pre-registered robustness: the decision is reported at **batch size 16 and 8**. The
hypothesis is considered robustly confirmed only if Δ<0 with CI excluding 0 at **both**
batch sizes on the test split; a split between bs16 and bs8 is reported as partial.

## 8. Smoke-test result (already run; reported for transparency)

`--split val --order native --batch-size 16 --max-batches 30 --window 5` (first 480
samples of the native-order OOD-val stream; ~2 min on MPS):

| quantity | frozen | Tent-online |
|---|---|---|
| cumulative macro-F1 | **0.182** | **0.060** |

- Δ = **−0.122**, paired-bootstrap 95% CI = **[−0.167, −0.089]** (excludes 0;
  P(Δ<0)=1.00).
- Every one of the 6 windows had Tent below frozen.
- Label-free collapse signature present from the first window: predicted-class diversity
  pinned at **4–11 unique classes out of 182** (histogram entropy 0.48–1.14 nats),
  entropy high (3.7–4.4), gradient norm 5–24.

This is a 30-batch preview on the *smaller* val split, so it does not by itself satisfy
the registered decision rule (which is the full-stream, test-split, two-batch-size
criterion). It is strong early evidence in the predicted direction. The full pilot
below is what the decision rule is evaluated on.

## 9. Exact full-pilot commands

```bash
cd experiments/kbound/wilds
PY=~/.venv_wilds/bin/python   # torch 2.5.1 + MPS + wilds 2.0.0

# HEADLINE: OOD test, native order, both batch sizes (decision rule split)
$PY -u run_iwildcam_streaming_pilot.py --split test --order native --batch-size 16 --window 50 --n-boot 2000
$PY -u run_iwildcam_streaming_pilot.py --split test --order native --batch-size 8  --window 50 --n-boot 2000

# SECONDARY confirmation: OOD val, native order, both batch sizes
$PY -u run_iwildcam_streaming_pilot.py --split val  --order native --batch-size 16 --window 50 --n-boot 2000
$PY -u run_iwildcam_streaming_pilot.py --split val  --order native --batch-size 8  --window 50 --n-boot 2000

# CONTROL (not for the claim): shuffled order, to show collapse is driven by native structure
$PY -u run_iwildcam_streaming_pilot.py --split test --order shuffled --batch-size 16 --window 50 --n-boot 2000
```

**Expected runtime on MPS** (M-series, ResNet-50, images on external USB volume;
measured smoke throughput ≈ 2.25 s/batch = ~7 img/s, I/O-bound):
- test bs16 (~2,210 batches): **~1.5 h**
- test bs8  (~4,421 batches): **~2.8 h**
- val  bs16 (~775 batches):   **~0.5 h**
- val  bs8  (~1,551 batches): **~1.0 h**
- Full matrix (4 runs + control): **~6 h** (conservatively up to ~1 day if the external
  drive is cold / contended).

Each run writes `pilot_<split>_<order>_bs<bs>.json` and `.png` under
`experiments/kbound/results/iwildcam_streaming_pilot/`.
