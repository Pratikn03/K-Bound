# GPU-side wiring (run on the Mac; CPU sandbox cannot execute these)

## 1. Evidence v2 features → `kga/evidence.py`

The new features need per-batch **logits**, which the CPU retro cannot recover
from logged aggregates. Wiring:

1. Copy `gapclose_wave5/evidence_v2.py` → `kga/evidence_v2.py`.
2. In `kga/evidence.py`, where the per-batch feature dict is built (entropy,
   confidence, drift, ...), add:
   ```python
   from kga.evidence_v2 import extract_all as _ev2
   feats.update({f"ev2_{k}": v for k, v in _ev2(logits_batch).items()
                 if k not in ("entropy", "msp")})  # avoid duplicates
   ```
   `logits_batch`: the frozen-model logits already computed for entropy —
   no extra forward pass needed.
3. ProjNorm (optional, heavier): fine-tune a probe on pseudo-labels per batch,
   feature = ||theta_probe − theta_ref||. Only add if runtime allows.

## 2. Re-run order (evidence uplift on natural shifts)

```bash
# smallest first: Office-Home protocol M v2 with ev2 features
bash kbtrain.sh protocol-m-v2            # after wiring, new Z includes ev2_*
# then Camelyon17 composition grid (the harm-AUC 0.78→? question)
python experiments/kbound/wilds/run_camelyon_grid.py --features ev2
```

Question each run answers: does harm-AUC (and the margin |M|) rise enough that
the certificate commits where it previously abstained, at FA ≤ α? Compare
against the logged baselines: Camelyon17 best single-feature AUC 0.78,
certificate transfer-AUC 0.91; Office-Home transfer-AUC 0.33/0.49.

## 2b. REQUIRED: serialize agreement matrices (Gap B blocker found in retro)

No logged ImageNet-R artifact stores `c_ij`/`n_D` (checked: multiseed
per-condition files have the keys but null; light-grid/diverse-panel runs never
had them). The self-normalized τ cannot be retro-applied. In the grid runner,
where the panel is evaluated per condition, add to the serialized record:
```python
rec["c_ij"] = C_panel.tolist()   # K x K empirical agreement (2A-1)
rec["n_D"]  = int(n_D)           # disagreement-region sample count
```

## 3. Self-normalized τ in the multicandidate route

Replace the fixed threshold in the router:
```python
from gapclose_wave5.tau_selfnorm import tau_selfnorm
res = tau_selfnorm(C_panel, m=n_D, alpha=0.05, n_sim=400, seed=cond_id)
gate_passes = not res["reject_H"]   # was: tau_hat <= 0.52
```
Cost: ~0.1–0.5 s per condition (vectorized null), CPU-side, negligible next to TTA.

## 4. The decisive pre-registered run (NATURAL_WIN_PROTOCOL_v1 — draft before running!)

Freeze BEFORE execution, in `research_lock/NATURAL_WIN_PROTOCOL_v1.yaml`:
- Panel: the 10 independently-trained backbones (ConvNeXt-B/T, EfficientNet-B0/B3,
  ResNet-101/152, ResNeXt-101, Swin-B/T, ViT-B/16) as multicandidate set —
  independence by construction (no shared backbone), the co-adaptation the CEI
  gate kept catching.
- Gate: self-normalized τ (α = 0.05), radius_v2 best variant (from Wave-5
  validators), evidence incl. ev2_* features.
- Datasets: Camelyon17 composition grid (harm detectable, AUC 0.78–0.91) as
  primary; ImageNet-R diverse panel as secondary.
- Bar: beats-both at FA_u ≤ α with Holm-corrected paired-bootstrap CIs,
  seeds 0–4, held-out test scored once.
- Honest outcomes: WIN / NO-HARM-tie / FAIL all publishable; no re-tuning after
  unblinding.
