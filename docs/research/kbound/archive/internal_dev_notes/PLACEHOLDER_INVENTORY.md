# Placeholder Artifact Inventory (iCloud / NUL-filled files)

**Generated:** 2026-07-26 by a full-tree NUL scan of `/home/claude/kb`.
**Scope:** every file with a text extension
(`.json .py .csv .md .sh .txt .tex .yaml .yml .tsv .lean .cfg .toml .html .js .ipynb`),
excluding `.git`, `__pycache__`, `.pytest_cache`.

## Summary

| test | count |
|---|---|
| files scanned | 1895 |
| **zero-byte or NUL-filled in the first 4 KB (the reliable test)** | **143** |
| whitespace-only (the naive test) | **0** |
| raising `OSError` on open/read | 0 |

By extension: **76 `.json`, 45 `.py`, 10 `.csv`, 9 `.md`, 3 `.sh`.**

Two points a reader must take from this table.

1. **A whitespace-only scan finds nothing.** These files have their full nominal byte length
   (`run_officehome_kbound.py` reports 17 202 bytes) and are filled with NUL. Any release guard
   that tests `content.strip() == ""` passes all 143. The guard must test for NUL bytes.
2. **These are iCloud Drive dataless placeholders, not data loss.** The file bodies live in
   iCloud; the local inodes were evicted. They are recoverable on the author's Mac by the
   procedure in "How to materialize them" below. No fix in this repository can recover them —
   this document exists so the recovery is a single command and the blast radius is explicit.

## What still depends on an unreadable file

Nothing in the promoted nine-track panel is *blocked* by a placeholder. The recompute pass that
produced `NUMBERS_PACK.md` re-derived every recomputable promoted number from readable artifacts;
its explicit finding is *"None of these blocked a number in this pack."* What blocks numbers is a
different failure — files that are **absent**, not placeholders — and those are recorded in
`SUBMISSION_LEDGER.md §8` (Camelyon17 reconciliation directory, the four `bootstrap_win_cis.py`
record files).

The placeholders do block the following, all of which are **secondary**:

| what is blocked | placeholder group | consequence today |
|---|---|---|
| Independent recomputation of the ablation tables (`ablation_{alpha,dropout,estimator,transfer}.json`) and the cost table (`cost_profile.json`) | A | The published ablation/cost numbers cannot be re-derived from the release. `ablation_exactrank.json` is readable and was regenerated. |
| Reading the Office-Home runner and analysis source (`run_officehome_kbound.py`, `oh_analyze.py`, 9 more) | B | The Office-Home protocol cannot be audited or re-run from the release at all; combined with the two absent `officehome_full_*` record files this makes Office-Home the least verifiable promoted track after Camelyon17. |
| The ten edge/physical-camera session checklists and pilot items | C | No promoted claim depends on these (Table XXVI is `RESULT PENDING`), but the physical study cannot be resumed from the release. |
| Run checkpoints for ImageNet-C seeds 1-4 and several ImageNet-R / RxRx1 / iWildCam `_partial.json` files | D, E | Resume/audit trails only. The per-condition dumps that the promoted ImageNet-C row is computed from are **readable** and were re-derived. |
| The `frontier_decisive/**` theory probes cited in `GAP_AUDIT.md` and `INTEGRITY_FIXES.md` | F | Those two documents' 2026-06-14 findings cannot be re-verified from the release; both are stamped SUPERSEDED. |
| The vendored legacy tree and the `kbound_pkg/build/lib/` copy of the library | G, H | No claim depends on either. `kbound_pkg/build/` is a build artifact and should be deleted from the release rather than restored. |

## How to materialize them (author, on the source Mac)

One command, from the repository root:

~~~bash
# Force-download every evicted file in the tree, then wait for iCloud to finish.
find . -type f -print0 | xargs -0 -n 200 brctl download
brctl log --wait --shorten          # watch until the queue drains
~~~

`brctl` is macOS's iCloud Drive control tool (`/usr/bin/brctl`); `brctl download <path>` requests
materialization of a dataless file. Equivalent GUI path: right-click the `kb` folder in Finder ->
**Download Now**.

Then re-run the scan and confirm it returns zero:

~~~bash
python3 - <<'PY'
import os
EXT={'.json','.py','.csv','.md','.sh','.txt','.tex','.yaml','.yml','.tsv','.lean','.cfg','.toml','.html','.js','.ipynb'}
bad=[]
for dp,dns,fns in os.walk('.'):
    dns[:]=[d for d in dns if d not in ('.git','__pycache__','.pytest_cache')]
    for f in fns:
        p=os.path.join(dp,f)
        if os.path.splitext(f)[1].lower() not in EXT: continue
        if os.path.getsize(p)==0 or b'\0' in open(p,'rb').read(4096): bad.append(p)
print(len(bad)); [print(b) for b in bad]
PY
~~~

To stop eviction recurring on the working copy:

~~~bash
# macOS Ventura+: keep the research tree resident
xattr -w com.apple.fileprovider.ignore#P 1 .            # (per-folder, where supported)
# or move the tree out of ~/Documents / ~/Desktop, which are the iCloud-synced roots
~~~

## Release guard — specification

The library agent owns `tests/`; this section is the spec, not the implementation.

**Guard 1 — NUL scan (blocking).**

- *Name:* `tests/test_no_placeholder_artifacts.py::test_no_nul_filled_tracked_text_files`
- *Rule:* for every tracked file whose extension is in the set above, `os.path.getsize(p) > 0`
  **and** `b"\x00" not in open(p,"rb").read(4096)`.
- *Why 4 KB and not the whole file:* a fully dataless file is NUL from byte 0; reading 4 KB is
  O(1) per file and keeps the guard under a second on 1 895 files. A whole-file read is
  acceptable but unnecessary.
- *Why not `content.strip()`:* verified on this tree — the strict whitespace test returns
  **0 files**, the NUL test returns **143**. The naive test provides no protection.
- *Expected state at release:* 0 failures. Today: 143 failures. The guard must therefore ship
  with an explicit, dated allowlist (below) that shrinks to empty, not with the assertion
  weakened.
- *Allowlist policy:* a file may be allowlisted only with (a) a reason, (b) a named owner, and
  (c) an expiry date. `kbound_pkg/build/**` should be **deleted**, not allowlisted.

**Guard 2 — checksum coverage of table-bearing artifacts (blocking).**

- *Name:* `tests/test_empirical_data_quality_audit_remediation.py::test_storage_manifest_internal_hashes_and_summary_match_disk`
- *Problem and remediation:* before the 2026-08-27 remediation,
  `docs/research/kbound/STORAGE_MANIFEST.json` carried checksums for only **3 files**
  (`claim_ledger.json`, `KBOUND_MIXED_STREAM_v2.json`, and
  `KBOUND_WIN_BOOTSTRAP_CIS_oof.json`). That gap allowed `claim_ledger.json` to drift from its
  recorded hash (F4-15) without a gate firing. The manifest now records nine direct publication
  artifacts and 71 sealed-evidence paths; the test above and
  `src/scripts/validate_manuscript_claims.py` verify them against disk.
- *Rule A (coverage):* every path that appears as a `source` in
  `paper/generated/kbound_result_manifest.json`, and every path sealed in
  `experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json`, must have an entry in
  `STORAGE_MANIFEST.json` with a non-null `sha256` and `size_bytes`.
- *Rule B (integrity):* every `STORAGE_MANIFEST.json` entry with `tracked: true` must match the
  file on disk in both size and sha256.
- *Rule C (existence):* every entry with `required: true` must exist. Entries that are known
  absent must carry `"status": "absent"` and a restoration procedure, not a silent null.
- *Freeze procedure:* `STORAGE_MANIFEST.json` is maintained explicitly because it combines
  current local hashes, historical lock-seal metadata, external-data inventory, and restoration
  notes. At freeze time, run `src/scripts/validate_manuscript_claims.py`; it fails closed on any
  size/hash drift, lock-seal disagreement, missing promoted-source coverage, or summary mismatch.

**Guard 3 — placeholder inventory freshness (non-blocking, informational).**

- Re-run the scan in CI and fail only if the count *increases* relative to the number recorded at
  the top of this file. This catches new evictions during a release without blocking on the
  existing backlog.

## Complete list of unreadable files

Paths are relative to the repository root. "bytes on disk" is the nominal size of the dataless
placeholder — it is the size the real file will have once materialized, not a byte count of
readable content (all of these read as NUL).

### A. Ablation tables and cost profile (short paper Tables `tab:abl-*`, `tab:cost`)

12 files.

| file | bytes on disk |
|---|---|
| `docs/research/kbound/experiments/kbound/results/ablation_alpha.json` | 3113 |
| `docs/research/kbound/experiments/kbound/results/ablation_dropout.json` | 1125 |
| `docs/research/kbound/experiments/kbound/results/ablation_estimator.json` | 754 |
| `docs/research/kbound/experiments/kbound/results/ablation_transfer.json` | 1148 |
| `docs/research/kbound/experiments/kbound/results/cost_profile.json` | 385 |
| `docs/research/kbound/experiments/kbound/results/decisive_tta_results.json` | 209632 |
| `docs/research/kbound/experiments/kbound/results/decisive_tta_table.md` | 597 |
| `docs/research/kbound/experiments/kbound/results/multiseed_camelyon17_eata.json` | 752 |
| `docs/research/kbound/experiments/kbound/results/multiseed_camelyon17_sar.json` | 756 |
| `docs/research/kbound/experiments/kbound/results/multiseed_camelyon17_tent.json` | 746 |
| `docs/research/kbound/experiments/kbound/results/official_headtohead.json` | 1858 |
| `docs/research/kbound/experiments/kbound/results/result_manifest.json` | 1327 |

### B. Office-Home track — runner source and analysis (`Office-Home M v2` panel row)

15 files.

| file | bytes on disk |
|---|---|
| `experiments/kbound/officehome/_count.py` | 468 |
| `experiments/kbound/officehome/_selftest.py` | 4994 |
| `experiments/kbound/officehome/make_report_figs.py` | 5270 |
| `experiments/kbound/officehome/materialize_officehome.py` | 2286 |
| `experiments/kbound/officehome/oh_analyze.py` | 18989 |
| `experiments/kbound/officehome/oh_candidates.py` | 5030 |
| `experiments/kbound/officehome/oh_data.py` | 7286 |
| `experiments/kbound/officehome/oh_report.py` | 5448 |
| `experiments/kbound/officehome/run_officehome_kbound.py` | 17202 |
| `experiments/kbound/officehome/supervise_oh.sh` | 953 |
| `experiments/kbound/officehome/train_f0_officehome.py` | 4593 |
| `experiments/kbound/results/officehome_full_source/_partial.json` | 335611 |
| `experiments/kbound/results/officehome_kbound_run/_partial.json` | 419526 |
| `experiments/kbound/results/officehome_protocol_m_repl_targettest/_partial.json` | 262979 |
| `experiments/kbound/results/officehome_protocol_m_repl_targetval/_partial.json` | 262943 |

### C. Edge / physical-camera study (no promoted claim; Table XXVI is RESULT PENDING)

21 files.

| file | bytes on disk |
|---|---|
| `docs/research/kbound/edge/artifacts_real/calibration/kga_edge_meta.json` | 390 |
| `docs/research/kbound/edge/artifacts_real/checklists/S01_checklist.csv` | 6888 |
| `docs/research/kbound/edge/artifacts_real/checklists/S02_checklist.csv` | 2328 |
| `docs/research/kbound/edge/artifacts_real/checklists/S03_checklist.csv` | 4208 |
| `docs/research/kbound/edge/artifacts_real/checklists/S04_checklist.csv` | 3392 |
| `docs/research/kbound/edge/artifacts_real/checklists/S05_checklist.csv` | 4208 |
| `docs/research/kbound/edge/artifacts_real/checklists/S06_checklist.csv` | 3392 |
| `docs/research/kbound/edge/artifacts_real/checklists/S07_checklist.csv` | 4208 |
| `docs/research/kbound/edge/artifacts_real/checklists/S08_checklist.csv` | 3392 |
| `docs/research/kbound/edge/artifacts_real/checklists/S09_checklist.csv` | 4208 |
| `docs/research/kbound/edge/artifacts_real/checklists/S10_checklist.csv` | 3392 |
| `docs/research/kbound/edge/artifacts_real/models/f0_meta.json` | 144 |
| `docs/research/kbound/edge/artifacts_real/pilot/PILOT_item_01.json` | 321 |
| `docs/research/kbound/edge/artifacts_real/pilot/PILOT_item_02.json` | 321 |
| `docs/research/kbound/edge/artifacts_real/pilot/PILOT_item_03.json` | 321 |
| `docs/research/kbound/edge/artifacts_real/pilot/PILOT_item_04.json` | 321 |
| `docs/research/kbound/edge/artifacts_synth/REPORT.md` | 2539 |
| `docs/research/kbound/edge/artifacts_synth/f0_meta.json` | 143 |
| `docs/research/kbound/edge/artifacts_synth/heldout_metrics.json` | 4035 |
| `docs/research/kbound/edge/artifacts_synth/kga_edge_meta.json` | 1250 |
| `docs/research/kbound/edge/artifacts_synth/manifest.json` | 2956 |

### D. ImageNet-C multi-seed run checkpoints (`win_hunt_v5_imagenetc_ms/seed{1..4}`)

11 files.

| file | bytes on disk |
|---|---|
| `experiments/kbound/results/imagenetc_noise/seed0/checkpoint.json` | 4428 |
| `experiments/kbound/results/imagenetc_noise_sarfix/checkpoint.json` | 37828 |
| `experiments/kbound/results/smoke05_20260701_113034/imagenetc_noise/seed0/checkpoint.json` | 9163 |
| `experiments/kbound/results/smoke_ms_20260701_125644/imagenetc_noise/seed0/checkpoint.json` | 25233 |
| `experiments/kbound/results/win_hunt_v5/imagenetc_aggr/checkpoint.json` | 29172 |
| `experiments/kbound/results/win_hunt_v5/imagenetc_aggr_1pct/checkpoint.json` | 0 |
| `experiments/kbound/results/win_hunt_v5/imagenetc_aggr_1pct/per_condition_imagenetc_tent_seed0.json` | 0 |
| `experiments/kbound/results/win_hunt_v5_imagenetc_ms/seed1/checkpoint.json` | 29435 |
| `experiments/kbound/results/win_hunt_v5_imagenetc_ms/seed2/checkpoint.json` | 29199 |
| `experiments/kbound/results/win_hunt_v5_imagenetc_ms/seed3/checkpoint.json` | 29373 |
| `experiments/kbound/results/win_hunt_v5_imagenetc_ms/seed4/checkpoint.json` | 29464 |

### E. ImageNet-R / RxRx1 / iWildCam / Camelyon run artifacts

31 files.

| file | bytes on disk |
|---|---|
| `experiments/kbound/results/camelyon17_fullscale_B_v2/_partial.json` | 326630 |
| `experiments/kbound/results/fmow_protocol_L_dev_partial/_partial.json` | 479903 |
| `experiments/kbound/results/gpu_queue_camelyon_then_iwildcam.sh` | 2485 |
| `experiments/kbound/results/gpu_queue_iwildcam_after_camelyon.sh` | 2272 |
| `experiments/kbound/results/imagenetr_kbound_debug_mps/_partial.json` | 45532 |
| `experiments/kbound/results/imagenetr_kbound_full_mps/_partial.json` | 4792 |
| `experiments/kbound/results/imagenetr_kbound_full_mps_internal/_partial.json` | 4884 |
| `experiments/kbound/results/imagenetr_kbound_light_mps_internal/_partial.json` | 215255 |
| `experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/_partial.json` | 228334 |
| `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/_partial.json` | 304353 |
| `experiments/kbound/results/iwildcam_1pct_val_v1/_partial.json` | 279595 |
| `experiments/kbound/results/iwildcam_aetta_prev_idval/_partial.json` | 43843 |
| `experiments/kbound/results/iwildcam_aetta_prev_val/_partial.json` | 43247 |
| `experiments/kbound/results/iwildcam_kbound_smoke/_partial.json` | 4318 |
| `experiments/kbound/results/kbound_inr_results/imagenetr_kbound_full_mps/_partial.json` | 4792 |
| `experiments/kbound/results/kbound_inr_results/imagenetr_kbound_light_mps_internal/_partial.json` | 215255 |
| `experiments/kbound/results/kbound_rxrx1_results/rxrx1_kbound_light_mps_internal/_partial.json` | 216605 |
| `experiments/kbound/results/kbound_rxrx1_results/rxrx1_kbound_smoke/_partial.json` | 17643 |
| `experiments/kbound/results/poverty_protocol_L_dev/_partial.json` | 361328 |
| `experiments/kbound/results/rxrx1_kbound_light_mps_internal/_partial.json` | 216605 |
| `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/_partial.json` | 548592 |
| `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed1/_partial.json` | 546216 |
| `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/_partial.json` | 546833 |
| `experiments/kbound/results/smoke05_20260701_113034/imagenetr_kbound_smoke/_partial.json` | 25334 |
| `experiments/kbound/results/wilds_kbound/_partial.json` | 147891 |
| `experiments/kbound/results/wilds_kbound_debug_mps/_partial.json` | 318514 |
| `experiments/kbound/results/win_hunt_v5/rxrx1_aggr/_partial.json` | 33442 |
| `experiments/kbound/results/win_hunt_v5_iwildcam/_partial.json` | 87074 |
| `experiments/kbound/theory_validation/frontier_decisive/camelyon_recal/CAMELYON17_RECAL.md` | 4159 |
| `experiments/kbound/theory_validation/frontier_decisive/camelyon_recal/camelyon_recal.py` | 7377 |
| `experiments/kbound/theory_validation/frontier_decisive/camelyon_recal/camelyon_recal_results.json` | 1039 |

### F. Theory-validation probes (`frontier_decisive/**`) — cited in GAP_AUDIT/INTEGRITY_FIXES only

17 files.

| file | bytes on disk |
|---|---|
| `experiments/kbound/theory_validation/frontier_decisive/FRONTIER_DECISIVE.md` | 7778 |
| `experiments/kbound/theory_validation/frontier_decisive/MANUSCRIPT_RECENTER.md` | 13661 |
| `experiments/kbound/theory_validation/frontier_decisive/frontier_decisive.py` | 10556 |
| `experiments/kbound/theory_validation/frontier_decisive/frontier_results.json` | 2791 |
| `experiments/kbound/theory_validation/frontier_decisive/kga_elara/KGA_ELARA_CONVERGENCE_FINDINGS.md` | 3598 |
| `experiments/kbound/theory_validation/frontier_decisive/kga_elara/kga_elara_convergence.py` | 6460 |
| `experiments/kbound/theory_validation/frontier_decisive/kga_elara/kga_elara_convergence_results.json` | 3638 |
| `experiments/kbound/theory_validation/frontier_decisive/probe_relax/PROBE_REAL_PROTOCOL.md` | 3402 |
| `experiments/kbound/theory_validation/frontier_decisive/probe_relax/PROBE_RELAX_FINDINGS.md` | 5115 |
| `experiments/kbound/theory_validation/frontier_decisive/probe_relax/build_dump.py` | 878 |
| `experiments/kbound/theory_validation/frontier_decisive/probe_relax/probe_realdata.py` | 4133 |
| `experiments/kbound/theory_validation/frontier_decisive/probe_relax/probe_relax.py` | 6763 |
| `experiments/kbound/theory_validation/frontier_decisive/probe_relax/probe_relax_results.json` | 6288 |
| `experiments/kbound/theory_validation/frontier_decisive/probe_relax/probe_units.json` | 107294 |
| `experiments/kbound/theory_validation/frontier_decisive/realdata/REAL_DATA_FRONTIER.md` | 6142 |
| `experiments/kbound/theory_validation/frontier_decisive/realdata/realdata_frontier.py` | 11972 |
| `experiments/kbound/theory_validation/frontier_decisive/realdata/realdata_frontier_results.json` | 3166 |

### G. Vendored legacy tree (`vendored_from_elara/**`) — no promoted claim depends on it

18 files.

| file | bytes on disk |
|---|---|
| `experiments/kbound/vendored_from_elara/certification/__init__.py` | 802 |
| `experiments/kbound/vendored_from_elara/certification/risk_dominance.py` | 8858 |
| `experiments/kbound/vendored_from_elara/certification/switching_certificate.py` | 5662 |
| `experiments/kbound/vendored_from_elara/drift/__init__.py` | 33 |
| `experiments/kbound/vendored_from_elara/drift/drift_nlp.py` | 331 |
| `experiments/kbound/vendored_from_elara/drift/drift_tabular.py` | 487 |
| `experiments/kbound/vendored_from_elara/drift/drift_time_series.py` | 483 |
| `experiments/kbound/vendored_from_elara/drift/drift_vision.py` | 298 |
| `experiments/kbound/vendored_from_elara/theory/__init__.py` | 658 |
| `experiments/kbound/vendored_from_elara/theory/gdr_minimax.py` | 6841 |
| `experiments/kbound/vendored_from_elara/theory/novel_theorem_bounds.py` | 8489 |
| `experiments/kbound/vendored_from_elara/theory/t1_impossibility.py` | 12066 |
| `experiments/kbound/vendored_from_elara/theory/t2_mixture_entropy.py` | 9711 |
| `experiments/kbound/vendored_from_elara/theory/t3_mean_gate_miss.py` | 8264 |
| `experiments/kbound/vendored_from_elara/theory/t6_sequential_detection.py` | 7217 |
| `experiments/kbound/vendored_from_elara/theory/t8_certified_heterogeneous_fusion.py` | 8151 |
| `experiments/kbound/vendored_from_elara/theory/t9_clean_transfer_ceiling.py` | 22791 |
| `experiments/kbound/vendored_from_elara/theory/theorem_registry.py` | 13419 |

### H. Packaged library build tree + formal + misc

18 files.

| file | bytes on disk |
|---|---|
| `docs/research/kbound/formal/formal_audit_report.json` | 1453 |
| `docs/research/kbound/kbound_pkg/build/lib/kbound/__init__.py` | 1244 |
| `docs/research/kbound/kbound_pkg/build/lib/kbound/certificate.py` | 4740 |
| `docs/research/kbound/kbound_pkg/build/lib/kbound/eprocess.py` | 7520 |
| `docs/research/kbound/kbound_pkg/build/lib/kbound/evidence.py` | 4948 |
| `docs/research/kbound/kbound_pkg/build/lib/kbound/kga.py` | 7485 |
| `docs/research/kbound/kbound_pkg/build/lib/kbound/optimizer.py` | 6961 |
| `docs/research/kbound/kbound_pkg/build/lib/kbound/router.py` | 5384 |
| `docs/research/kbound/kbound_pkg/build/lib/kbound/routing.py` | 5453 |
| `docs/research/kbound/kbound_pkg/tests/__init__.py` | 0 |
| `docs/research/kbound/theory_v2/realdata/_p1_partial.json` | 107156 |
| `docs/research/kbound/theory_v2/realdata/_p2_partial.json` | 3589 |
| `experiments/kbound/pilots/kga_meta_pilot_results.json` | 398 |
| `experiments/kbound/pilots/kga_meta_tta_pilot.py` | 6021 |
| `experiments/kbound/pilots/results_conj1_validator.json` | 945 |
| `experiments/kbound/results_conj1_validator.json` | 945 |
| `experiments/kbound/results_kga_elara.json` | 2807 |
| `experiments/kbound/results_kga_elara_extra.json` | 1308 |
