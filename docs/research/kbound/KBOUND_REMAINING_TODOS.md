# K-Bound Draft — Remaining TODOs

These are the items a faculty reviewer will still ask for. Each is either a visible `DRAFT TODO` in
`kbound_short_draft.pdf` or an audit-driven action. None fabricates data; missing experiments are
labeled, not invented. Priority: **P1** blocks a serious submission, **P2** strengthens it, **P3**
polish.

> **Status:** **1 DRAFT TODO** remains (streaming figures) — and it is *not* T9-blocked: no saved
> streaming artifact exists anywhere (T9, logs, or the long paper), so it needs a re-run of
> `win_hunt_D_anytime_stream.py`. Everything else is done or documentation.

## Method / reproducibility
- ~~Reconcile the **two evidence panels** into one documented feature schema~~ **DONE** —
  Appendix `app:evidence-schema`, `tab:schema-base` (11-dim) + `tab:schema-rich` (16-dim), exact
  per-feature formulas from `evidence.py` / `run_wilds_camelyon17.py`.
- ~~Tabulate **per-adapter learning rate + steps**~~ **DONE** — `tab:adapter-hparams` (mild 10/1e-3,
  aggressive 50/2.5e-3; EATA filter 0.4·ln2; SAR SAM+reset; backbones verified for
  CIFAR/ImageNet/PACS/Camelyon). Office-Home backbone lives in the T9-side runner (generalized).
- ~~**Deployment semantics**~~ **DONE** — filled from `kbound_pkg` (`decide_from_batch`;
  `abstain_scale=0` → skip update / serve frozen; failure-safe defaults). Only a deployment-specific
  numeric min-batch *threshold* remains to be set per site.
- ~~**Cost table**~~ **DONE** — `tab:cost` (controller +0.20 ms/decision, +44.8 MB rollback copy),
  `scripts/cost_profile.py`. Optional P3: add GPU wall-clock for a full forward+backward step for
  absolute end-to-end numbers.

## Results — locked from restored T9 artifacts
- ~~**Three-source mixture 0.0059**~~ **DONE / LOCKED** — `research_lock/KBOUND_MIXED_STREAM_v2.json`
  (regret 0.0059/0.0632/0.0342, FA_u 0, both CIs exclude zero, `beats_both_robust=true`). DRAFT TODO removed.
- ~~**Office-Home / iWildCam OOF summaries**~~ **DONE / RECONCILED** — backed by
  `research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json`; panel upgraded provisional → reconciled.
- ~~**Provenance**~~ **DONE** — restored `mixed_headtohead_v1/HEADTOHEAD_RESULTS_*.json`,
  `mixed_protocol_oof_v2/`, `KBOUND_WIN_BOOTSTRAP_CIS_oof.json`, and all of `research_lock/` from T9;
  the previously-dangling `\path{}` citations now resolve.
- **P1** **Streaming demo** (35,370-image iWildCam) — the ONE remaining DRAFT TODO. No saved artifact
  exists on T9/logs/long-paper; re-run `gapclose_wave5/win_hunt_D_anytime_stream.py` and commit the JSON.
- **P2** **CIFAR-10.1 regret triple** — still only FA_u=0.444 traceable; low-stakes (a no-claim
  diagnostic row).

## Leakage hygiene (residual)
- **P1** De-register the pooled Camelyon `id_val` config from `scripts/bootstrap_win_cis.py`
  (still hard-codes n=54) and from `WIN_HUNT_v5_PROTOCOL_SHELL.yaml:97` (re-lists `id_val`).
- **P2** Make the seed-split scorer **domain-aware** so an OOD claim cannot silently pool
  in-distribution cells again.

## Theory (from Phase 1 audit)
- **P2** Namespace the duplicate `\label` keys shared by `theory_core_main.tex` (short) and
  `main_theory_5.tex` (long) so the two can never collide if co-included.
- **P3** Prose pass confirming the empirical benefit model is never described as *establishing* risk
  alignment (`def:risk-align` is an assumption, not a result).

## References
- **P2** Add `\cite` at first mention for the uncited dataset/tool sources that the paper names in
  prose (CIFAR-10/100, ImageNet, WILDS/Camelyon17/iWildCam/RxRx1, Office-Home, PACS/DomainBed,
  RobustBench, PyTorch, scikit-learn). 45 entries are currently uncited; most are these sources.
- **P3** Fix `gupta2021toplabel` (key 2021 vs venue ICLR 2022); split `corradaemmanuel2024`
  (three arXiv IDs in one entry); `tempora2026` is forward-dated arXiv-only. (Duplicate
  `vovk2005algorithmic` already removed.)

## Pre-specified ablations
- ~~Evidence/estimator dependence~~ **DONE** — `tab:abl-estimator` (GBR/RF/ridge/MLP) + feature-family dropout.
- ~~α sweep {0.01,0.05,0.10,0.20}~~ **DONE** — `tab:abl-alpha`.
- ~~β sweep~~ **DONE** — the synthetic frontier of §Synthetic (FA_u≤0.014 across β).
- ~~Adapter transfer~~ **DONE (adapters)** — `tab:abl-transfer`; cross-adapter transfer breaks FA_u.
  **Remaining (P2):** *backbone* transfer across architectures still needs compute (a real re-run on a
  second backbone); the logged evidence covers only the ResNet-18 (CIFAR) / ResNet-50 (ImageNet) runs.
All the above run from logged evidence via `scripts/ablation_sweep.py` — no T9.

## Compile QC
- **P3** Five overfull `\hbox` warnings (max 71.8pt), mostly long bibliography lines and two wide
  table cells; cosmetic, non-blocking.
