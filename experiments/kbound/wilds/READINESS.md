# Camelyon17 K-Bound pipeline — READINESS (v0.5 Part 2)

Status: **pipeline built + CPU smoke PASSED end-to-end.** STOPPED before the MPS
sweep (resumable SAR run PID 80329 owns the GPU; never touched). Host execution via
desktop-commander, `~/.venv_wilds` (torch 2.5.1 + MPS, wilds 2.0.0).

## Deliverables (`experiments/kbound/wilds/`)
- `tta_methods.py` — faithful Tent/EATA/SAR (ported verbatim from `cifar_tent_mps_v2.py`),
  online + episodic; 11-dim label-free Z.
- `cam_data.py` — WILDS loader + disk-filter + natural-shift conditions
  (domain × composition × batch-regime).
- `analysis.py` — (a) `decide_kga` single-cand certificate; (b) `multicandidate_route`
  Thm-1A τ-residual (reuses `val_multicandidate_residual.py`); (c) `smooth_drift_route`
  **TODO STUB** (val_smooth_drift.py absent); `detectability_analysis`.
- `run_camelyon17_kbound.py` — orchestrator + CLI; writes JSON manifest.

## Data
- Full dataset on T9: `experiments/kbound/data/wilds/camelyon17_v1.0` (tar.gz 9.7 GB).
- Internal copy `~/kbound_cam/wilds/camelyon17_v1.0` = **414,389 / 455,954 (90.9%)**.
  **Center 2 (`test` = hardest OOD hospital) is 100% present.** Missing ~41.5k are in
  other centers; disk-filter drops them honestly (logged). TOP-OFF DEFERRED to handoff.
- 4 trained f0 seeds present: `results/wilds/f0_seed{0..3}.pt` (DenseNet-121, 28 MB).

## Smoke (`results/wilds_kbound_smoke/result_5d8b5708.json`, CPU, 118 s)
- 24 records (4 conditions × 6 candidates); routing a/b/c, baselines, detectability,
  manifest all written; every cell traces to `records[]`.
- detectability: **detectable**, best label-free harm-AUC = **0.913** (`marginal_KL`).
- multi-cand τ ≈ 1.5 ⇒ ABSTAIN everywhere — correct at smoke scale (n_eval=32, tiny noisy
  disagreement region). Expect τ to drop at full n_eval; **τ\*=0.08 (sim) needs recalibration
  on real agreements.**

## Estimated full MPS run (CPU per-pass measured; 8–20× speedup assumed)
- DEBUG default (4 seed, 3 domain, 3 comp, `small`, mild+aggr) = 72 cond → **~1.8–4.5 h**
- LEAN (4 seed, test+val, 3 comp, `small`) = 48 cond → ~1.2–3 h
- FULL (+ 3 batch-regimes) = 216 cond → ~10–25 h
Cost driver: aggressive (50-step) + episodic-at-small-bs. Calibrate with one real MPS cond.

## GPU-handoff checklist (SAR paused)
1. Top off internal copy (`cp -R` / `rsync -rlt --ignore-existing`, NOT `-a` → avoids `._`).
2. `run_camelyon17_kbound.py --device mps --batch-regimes small` (+`--domains test val id_val` for γ_S).
3. Real run reports: Camelyon17 class (helpful/harmful/mixed±detectable), measured γ_S,
   implied γ_T (oracle), whether |γ_T−γ_S| ≤ δ, real-data multi-candidate τ.


---

# HANDOFF / LIVE MPS RUN (2026-06-09)

SAR sweep PAUSED (was PID 80329/80331, killed cleanly; checkpoint 4/36 cells done).
Camelyon17 K-Bound DEBUG grid LAUNCHED on MPS.

**SAR resume command** (run after the Camelyon run finishes / GPU freed; resumes from checkpoint):
```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is ~/.venv_wilds/bin/python \
  docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks imagenetc \
  --imagenetc-root experiments/kbound/data/imagenet-c \
  --corruptions gaussian_noise shot_noise impulse_noise --arch resnet50 \
  --methods tent eata sar --out-results experiments/kbound/results/imagenetc_noise_sarfix
```

**Camelyon17 run** (PID 86973 python / 86975 caffeinate):
```bash
caffeinate -is ~/.venv_wilds/bin/python -u experiments/kbound/wilds/run_camelyon17_kbound.py \
  --device mps --seeds 0 1 2 3 --domains test val id_val \
  --compositions iid imbalanced single_class --batch-regimes small \
  --aggressiveness mild aggressive --tau-star 0.52 --run-name wilds_kbound_debug_mps
```
- 72 conditions (4 seed x 3 domain x 3 comp x small x mild+aggr). ETA ~1.8-4.5 h.
- Results dir: `experiments/kbound/results/wilds_kbound_debug_mps/`
- Live log: `.../wilds_kbound_debug_mps/run.log`  (per-condition lines)
- Recoverable partial: `.../wilds_kbound_debug_mps/_partial.json` (rewritten each condition)
- Final manifest: `.../wilds_kbound_debug_mps/result_<sha>.json` (at completion)
- tau* = 0.52 = size-normalized analog of the simulation per-entry tolerance 0.08 for M=7
  (0.08*sqrt(7*6)); per-condition tau also stored for post-hoc re-pick.
- Variant (c) smooth-drift now WIRED (Thm 1B, Brier view, reuses val_smooth_drift.py);
  reported as DIAGNOSTIC (conservative g_S~f0 surrogate). All three routes execute.
- Both OOD hospitals 100% present (center 2 = test, center 1 = val); disk-filter drops
  9% only in id_val/train centers.
