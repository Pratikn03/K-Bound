# K-Bound natural-shift pipeline — WILDS Camelyon17 (v0.5 Part 2)

Stands up the natural (hospital) distribution-shift experiment for the K-Bound paper
*"When Is Label-Free Adaptation Knowable?"*. Built to DEBUG end-to-end on Camelyon17
(smallest WILDS dataset) before scaling to RxRx1 / iWildCam.

## Files
- `tta_methods.py` — faithful Tent / EATA / SAR (ported verbatim from
  `docs/research/kbound/scripts/cifar_tent_mps_v2.py`), each in **online** and
  **episodic** modes; 11-dim label-free evidence vector `Z`.
- `cam_data.py` — WILDS loading (disk-filter for partial copies) + natural-shift
  **conditions** = (domain × composition × batch-regime). Domains: `test`=center 2
  (hardest OOD hospital, "EXTENDED"), `val`=center 1 (OOD), `id_val`=centers 0/3/4
  (in-dist control). Harmful cells arise from natural stream pathologies
  (single-class label shift, tiny batches), never from tuned hyperparameters.
- `analysis.py` — routing variants: **(a)** single-candidate KGA certificate
  (`decide_kga`), **(b)** multi-candidate τ-residual route [Theorem 1A], reusing
  `theory_validation/val_multicandidate_residual.py`, **(c)** smooth-drift [Theorem 1B]
  = **TODO STUB** (backing `val_smooth_drift.py` absent). Plus `detectability_analysis`.
- `run_camelyon17_kbound.py` — orchestrator + CLI; writes a JSON manifest where every
  table cell traces to `records[]` / `conditions[]`.

## Run
```bash
# tiny CPU end-to-end smoke (no MPS; safe alongside the SAR sweep)
~/.venv_wilds/bin/python experiments/kbound/wilds/run_camelyon17_kbound.py --smoke

# full run (MPS) — ONLY after the SAR GPU handoff
~/.venv_wilds/bin/python experiments/kbound/wilds/run_camelyon17_kbound.py \
    --device mps --seeds 0 1 2 3 \
    --domains test val id_val --compositions iid imbalanced single_class \
    --batch-regimes large_iid small tiny --aggressiveness mild aggressive
```

## Integrity
Every cell is run for real; labels are used ONLY for benefit `B` / oracle /
detectability. Routers see only `Z` (a) or label-free agreements (b). Camelyon17's
helpful/harmful classification is reported from measured `B`, never tuned to a target.
