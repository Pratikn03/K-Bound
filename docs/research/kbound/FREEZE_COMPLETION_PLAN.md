# K-Bound Freeze Completion Plan

**Created:** 2026-06-25  
**Owner:** Lead research engineer  
**Canonical status:** `PROJECT_STATUS_AND_OPEN_PROBLEMS.md`  
**Verdict baseline:** `reports/KBOUND_10X_FINAL_GATE.md` — PASS WITH LIMITATIONS

This document is the executable checklist to close the production freeze gate. Items marked **AUTO** can be run from this repo; **HUMAN** require a person, camera, or external reviewer.

---

## Phase map

| Phase | Goal | Owner | Status |
|-------|------|-------|--------|
| P0 | Integrity + claim ledger | AUTO | ✅ Done (10× pass) |
| P1 | Mixed-stream OOF re-run | AUTO | ✅ Done (`mixed_protocol_oof_v2`) |
| P2 | Paper sync (short + long) | AUTO | ✅ Done (2026-06-25) |
| P3 | Edge honest framing | AUTO | ✅ Done (RESULT PENDING macros) |
| P4 | Repro script + PDF compile | AUTO | ✅ Done |
| P5 | Repo cleanup (stale MDs) | AUTO | ⬜ Banners added; optional `git rm` |
| P6 | Physical camera R2 | HUMAN | ⬜ Real captures S01–S10 |
| P7 | External repro sign-off | HUMAN | ⬜ `REVIEWER_REPRO_PACKET.md` |
| P8 | Optional strict grid v2 | AUTO | ⬜ Stronger calibration story |
| P9 | Head-to-head vs POEM/AETTA | AUTO | ✅ WIN (all 3 sets) |

---

## P1 — Mixed-stream OOF (completed)

**Protocol:** `research_lock/mixed_protocol_oof_v2.yaml`  
**Script:** `docs/research/kbound/scripts/mixed_stream_kbound.py`  
**Artifacts:**
- `research_lock/KBOUND_MIXED_STREAM_v2.json`
- `experiments/kbound/results/mixed_protocol_oof_v2/mixed_protocol_oof_v2_result.json`

**Headline (OOF, honest):**
- `n=143` (Camelyon17 OOD 36 + Office-Home 35 + iWildCam 72)
- `regret_kga=0.0059` vs `0.0632` adapt / `0.0342` freeze
- vs freeze: `+0.0283` CI `[0.019, 0.038]` — excludes 0
- vs adapt: `+0.0573` CI `[0.043, 0.072]` — excludes 0
- `beats_both_robust=true`, `false_adapt=0`
- **Not** the withdrawn 13–24× in-sample multipliers

**Claim tier:** B — constructed cross-protocol aggregate only (not natural-shift headline).

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
.venv/bin/python docs/research/kbound/scripts/mixed_stream_kbound.py
```

---

## P2 — Paper sync (completed)

| File | Change |
|------|--------|
| `kbound.tex` §`sec:mixedstream` | OOF results; withdrawn in-sample figures |
| `kbound.tex` contributions | Cross-protocol beats-both (constructed) |
| `kbound_short.tex` | Same mixed-stream paragraph |
| `claim_ledger.json` | `KB-CLAIM-024` added; `KB-CLAIM-023` stays withdrawn |

Natural-shift rows remain **no-harm** only. Synthetic stress grids remain the only per-dataset beats-both.

---

## P3 — Edge framing (done in code)

- `docs/experiments/kbound/results/edge_real_phone_v1/camera_tables_values.tex` → `RESULT PENDING`
- `kbound_short.tex` §Real-camera → pre-registered protocol, not empirical claim
- `11_export_camera_tables.py` gates export unless publication-ready

**To populate R2 (HUMAN):**

```bash
# Per edge/PHYSICAL_STUDY_RUNBOOK.md — real camera, no --mock
bash tools/physical_capture/run_capture_session.sh S01 P01
# … S02–S10 …
bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh
```

---

## P4 — Repro + PDF (run after edits)

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
bash docs/research/kbound/scripts/reproduce_submission.sh

cd docs/research/kbound
pdflatex -interaction=nonstopmode kbound_short.tex
pdflatex -interaction=nonstopmode kbound.tex
```

Expected: 10 pytest passes, `make_tables.py` OK, `claim_ledger.json` valid.

---

## P5 — Repo cleanup

Stale files (superseded by `PROJECT_STATUS_AND_OPEN_PROBLEMS.md`):

- `COMPLETION_STATUS_2026-06-19.md` — says Conj. 1 open (**wrong**)
- `LAYOUT_*_2026-06-*.md`, `PAPER_BLUEPRINT_80.md`, `WINNING_PAPER_*.md`
- `RESULTS_PENDING.md`, `ELARA_KGA_MERGE_PLAN.md`, `HEADTOHEAD_VERIFICATION.md`

**Action:** DEPRECATED banner at file top (done). Optional `git rm` when ready.

**Review before delete:** `kbound_submission.tex`, `manuscript/` (stale conj:gen).

---

## P6 — Physical camera R2 (HUMAN)

| Step | Command / note |
|------|----------------|
| Labels | P01–P10 boxes: ok / missing / misaligned / damaged |
| Capture | Two phones, sessions S01–S10, source gate ≥ 0.80 |
| Train | Pipeline step 02 (no `--bypass-gate`) |
| Eval | S07/S08 held-out; S09/S10 replication |
| Export | Only if gate passes → `camera_tables_values.tex` |

Current blocker: `artifacts_real/` contains `--mock` noise clips.

---

## P7 — External sign-off (HUMAN)

1. Theory/stats reviewer: `thm:uncond-weakest`, Lemma 1, coverage theorem
2. Independent reproducer: `REVIEWER_REPRO_PACKET.md`
3. Record sign-off in `reports/external_review_signoff.md` (create when done)

---

## P8 — Optional upgrades (not freeze blockers)

| Item | Lock file | Purpose |
|------|-----------|---------|
| Strict stress grid v2 | `STRESS_GRID_STRICT_PROTOCOL_A_v2.yaml` | Group-level OOF calibration |
| Assumption audit | `assumption_audit_v1.yaml` | Deployment falsification suite |
| `foldin_multiseed_results.py` | — | Restore or remove references |

---

## Freeze gate checklist (live)

| # | Item | Status |
|---|------|--------|
| 1 | All scorers OOF | ✅ |
| 2 | Mixed-stream re-run | ✅ |
| 3 | Long paper reclassified | ✅ |
| 4 | Edge feasibility framing | ✅ (R2 pending) |
| 5 | PDFs recompile clean | ✅ |
| 6 | External sign-off | ⬜ HUMAN |

**Honest headline at freeze:** impossibility theorem + FA_u certificate + stress-grid beats-both + uniform natural-shift no-harm + constructed mixed aggregate beats-both. No camera or universal mixed-deployment win claimed.

---

## Weekly execution order

1. **Today (AUTO):** P1 → P2 → P4 → update `PROJECT_STATUS` §3  
2. **This week (HUMAN):** P6 physical capture if R2 is a goal  
3. **Before submission (HUMAN):** P7 external repro  
4. **Nice-to-have:** P8 strict grid + assumption audit run
