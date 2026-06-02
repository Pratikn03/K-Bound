# 100% Audit — Real-IAD-3D Strong-Detector + Fusion Work (2026-06-02)

Hostile-reviewer audit run BEFORE accepting the held-out confirmation. Findings
ranked by severity. The headline: one critical leakage bug was caught and fixed;
the held-out run was invalidated and must be re-done.

## CRITICAL — FIXED

### F1. Binary-PCD label leakage via degenerate-tiff fallback
- **What:** 3 of 20 categories store **binary** PCDs (`knob_cap`,
  `fork_crimp_terminal`, `telephone_spring_switch`). `load_pcd_points` only parsed
  `DATA ascii` -> returned 0 points -> `load_modality_image` silently fell back to
  the **degenerate XYZ tiff**, whose X,Y are placeholder constants that differ
  between OK and NG files (capture/export artifact correlated with the label =
  leakage).
- **Impact:** faked `knob_cap` XYZ at **0.999**; real PCD geometry gives **0.8996**.
  The killed held-out run had 2 of 11 categories (fork_crimp, telephone) on
  artifact-XYZ -> **invalid**.
- **Fix:** `_parse_pcd_binary()` parses the structured binary record; all 3 now
  return ~160-190k real points. Regression tests added (5 pass). Commit `9823173`.
- **Consequence:** re-pre-register with the fix and re-run the held-out set.

## GAPS — FIXED / NOTED

### F2. Score cache had no detector-version key (latent staleness) — FIXED
- The cache key was `(cat, caps, coreset)`; a detector code change would leave
  stale caches silently (how the artifact could have persisted). Added
  `DETECTOR_VERSION` to the key.

### F3. RGB uses only 1 of 5 lighting views — NOTED (improvement, not a bug)
- `image_path` is `RGBL05`. Using all 5 views would likely strengthen RGB
  (currently 0.756 mean). Optional hardening before/after confirmation.

### F4. Validation set is small (~25 OK/category) — NOTED (robustness)
- Reliability weights come from per-category val AUROC computed on ~25 OK + ~45 NG.
  Noisy weights -> if anything this *hurts* the gate (suboptimal weights), so it
  does not inflate the result; but the per-category reliability estimate is noisy.

## CONFIRMED CLEAN

- **No reliability-weight leakage:** weights use validation labels only; val and
  test are disjoint splits; the rule is frozen (`--fixed-rule`), not test-selected.
- **z-sigmoid normalization** uses train-OK distances only; monotone, so it does
  not change within-category AUROC and keeps the CW-vs-gated comparison fair.
- **Held-out structure clean:** all 11 categories have all 3 modalities, train_OK=200,
  and both classes in val and test (counts checked; no test scores inspected).
- **Bootstrap** is paired (same resample indices), deterministic (fixed seed),
  10000 iters; primary endpoint is a single pre-registered test (no multiplicity).
- **Sample alignment:** modalities share `sample_id` per entry; stratified caps on
  consistent row order -> intersection is the full set; n_test ~115/category.

## METHODOLOGY NOTE TO DISCLOSE

### M1. Pooled vs within-category AUROC
The pre-registered primary metric is *pooled* gated-vs-CW. In v1 the pooled gain
(+0.037) exceeded the within-category mean (~+0.007), i.e. part of the advantage
was cross-category calibration. v2's refinement raised the within-category benefit
(4 wins). The held-out report MUST include both the pooled delta and the
within-category mean delta so the calibration contribution is visible. To be
re-checked on the corrected dev result.

## STATUS
- F1 fix committed (`9823173`); cache guard (F2) committed.
- Corrected development re-run in progress (knob_cap now real geometry).
- Next: re-pre-register (new commit hash + this audit), then re-run the held-out
  confirmation once. No held-out test scores were inspected during this audit ->
  pre-registration integrity intact.
