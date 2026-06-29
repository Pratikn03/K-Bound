# K-Bound 10× Release Track — Live Audit Plan

**Created:** 2026-06-25  
**Owner:** Lead research engineer / reproducibility auditor  
**Goal:** Maximize rigor and submission-readiness without tuning on held-out tests or inflating claims.

---

## 1. Repository map

| Area | Path |
|------|------|
| Short paper (submission) | `docs/research/kbound/kbound_short.tex` |
| Long paper | `docs/research/kbound/kbound.tex` |
| Number source of truth | `docs/research/kbound/results_source.json` |
| Generated table macros | `docs/research/kbound/paper/generated/kbound_numbers.tex` |
| Certificate (frozen) | `docs/research/kbound/kbound_pkg/kbound/certificate.py` |
| Edge runtime | `docs/research/kbound/edge/src/kbound_edge/` |
| Protocol locks | `research_lock/*.yaml`, `research_lock/*.json` |
| Headline results | `experiments/kbound/results/` |
| Paper-local mirrors | `docs/research/kbound/results/` |
| Edge results | `docs/experiments/kbound/results/edge_real_phone_v1/` |
| Claim ledger | `docs/research/kbound/claim_ledger.json` |
| Repro script | `docs/research/kbound/scripts/reproduce_submission.sh` |

---

## 2. Protocol inventory

| Protocol ID | Lock file | Status | Claim tier |
|-------------|-----------|--------|------------|
| STRESS_GRID_MULTISEED_PROTOCOL_A_v1 | `research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml` | Locked; 5-seed grid complete | B — beats-both (Tent/EATA) |
| STRESS_GRID_STRICT_PROTOCOL_A_v2 | `research_lock/STRESS_GRID_STRICT_PROTOCOL_A_v2.yaml` | **New** — group-level OOF splits | B — pending re-run |
| imagenetc_protocol_E_v1 | `research_lock/imagenetc_protocol_E_v1.yaml` | Locked SAR harmful point | B |
| OFFICEHOME_PROTOCOL_M_v2 | `research_lock/OFFICEHOME_PROTOCOL_M_v2.yaml` | OOF radius | B — no-harm |
| IWILDCAM_PROTOCOL_H_v2 | `research_lock/IWILDCAM_PROTOCOL_H_v2.yaml` | OOF radius | B — no-harm |
| CAMELYON17_PROTOCOL_G_v1 | `research_lock/CAMELYON17_PROTOCOL_G_v1.yaml` | Genuine OOD only | B — no-harm (beats-both withdrawn) |
| edge_real_phone_v1 | `edge/configs/edge_real_phone_v1.yaml` | Locked design; **Tier C** until real captures pass gate | C |
| assumption_audit_v1 | `research_lock/assumption_audit_v1.yaml` | Pre-registered | C until run |
| mixed_protocol_oof_v2 | `research_lock/mixed_protocol_oof_v2.yaml` | **Complete** — OOF LOO; beats-both on constructed aggregate | B supported (constructed only) |

---

## 3. Headline claims (current)

See `claim_ledger.json` for machine-readable status. Summary:

- **Tier A:** Benefit-sign frontier (iff + impossibility); split-conformal certificate controls **FA_u ≤ α** under stated assumptions (not FA_c).
- **Tier B (supported):** CIFAR-10-C stress grid beats-both (Tent/EATA); ImageNet-C SAR harmful point; natural-shift **no-harm** OOF (Office-Home, iWildCam).
- **Tier B (withdrawn):** Camelyon17 pooled beats-both; mixed-stream 13–24× aggregate; in-sample-radius Office-Home/iWildCam beats-both.
- **Tier C:** Physical camera primary result (R2); mixed-regime real-world win.

---

## 4. Audit-only / superseded (archived)

| Artifact | Archive path | Reason |
|----------|--------------|--------|
| `camera_tables_values.tex` (2026-06-25 dev replay) | `archive/superseded/edge_camera_tables_values_2026-06-25_dev.json` | Mock/helpful-dominated; 25% acc presented as measured — **not publication evidence** |
| `KBOUND_WIN_BOOTSTRAP_CIS.json` | audit-only | In-sample radius |
| Mixed-stream pre-fix figures | `PROJECT_STATUS_AND_OPEN_PROBLEMS.md` §2.59 | Scorer bug; withdrawn |

---

## 5. Reproduction commands (headline tables)

```bash
# Environment
cd /Volumes/T9/uav/AutoML_Flagship_V8
source .venv/bin/activate   # or ~/.venv_wilds per local setup

# Full submission repro (lightweight + cached verify)
bash docs/research/kbound/scripts/reproduce_submission.sh

# CIFAR-10-C gate baseline (CPU, seconds)
cd docs/research/kbound
python scripts/gate_baseline_comparison.py --selftest
python scripts/gate_baseline_comparison.py

# Natural-shift OOF scoring (existing JSON)
python scripts/run_protocol_dev_lock.py --protocol-yaml research_lock/IWILDCAM_PROTOCOL_H_v2.yaml
python scripts/run_protocol_dev_lock.py --protocol-yaml research_lock/OFFICEHOME_PROTOCOL_M_v2.yaml

# Regenerate paper table macros from results_source.json
python scripts/make_tables.py

# Physical edge (only after real S01–S10 capture)
bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh
```

**Known gaps:** `scripts/foldin_multiseed_results.py` referenced but missing; venv split (`kbtrain.sh` vs `.venv`); GPU datasets need prep scripts.

---

## 6. Risk register

### Leakage risk
- **HIGH (historical):** In-sample conformal radius in `protocol_result.json::test_locked` — **do not use for headlines** (`results_source.json` documents OOF fix).
- **MEDIUM:** Stress grid uses leave-one-**condition**-out benefit fitting within seed — exchangeability unit = condition, not random frame.
- **LOW (edge):** 8/8 anti-leakage tests pass; mock clips invalidated for publication.

### Calibration validity risk
- Stress grid: LOO-GBR + split-conformal on LOO residuals — **not jackknife+**; finite-sample guarantee is for **split conformal on independent calibration pairs**, not per-cell LOO without qualification.
- Natural shifts: OOF ε per dev lock — **supported by protocol design**.

### Evidence-feature selection risk
- 14 edge features frozen in `EDGE_EVIDENCE_NAMES` test; paper 11-feature core frozen in `kbound/evidence.py`.

### Adapter-shopping risk
- Dev-lock protocols (M v2, H v2) select adapter on dev only; held-out scored once.

### Result-reporting risk
- **BLOCKER:** `camera_tables_values.tex` had numeric cells from development replay — **reverted to RESULT PENDING**.

### Theorem/implementation mismatch risk
- **β (population drift budget) ≠ ε (empirical radius)** — must not conflate in text (audit added to paper guarantee box).
- Certificate docstring claims split-conformal coverage; stress grid uses LOO construction — wording must match (`claim_ledger.json` KB-CLAIM-012).

---

## 7. Phase checklist

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | This file + `claim_ledger.json` | ✅ |
| 1 | `reports/kbound_claim_consistency_audit.md` + paper fixes | 🔄 |
| 2 | `STRESS_GRID_STRICT_PROTOCOL_A_v2.yaml` + leakage tests | 🔄 |
| 3 | `kbound_pkg/assumption_audit/` + `assumption_audit_v1.yaml` | 🔄 |
| 4 | Protocol re-runs + `claim_eligibility.json` per protocol | ⏳ (human/GPU) |
| 5 | Mixed OOF v2 + physical tooling | 🔄 |
| 6 | Baseline strengthening | ⏳ |
| 7 | Paper rewrite (guarantee box, pending tables) | 🔄 |
| 8 | Figure regeneration from artifacts | ⏳ |
| 9 | `reproduce_submission.sh` + `RELEASE_MANIFEST.json` | 🔄 |
| 10 | `reports/KBOUND_10X_FINAL_GATE.md` | 🔄 |

---

## 8. Calibration method truth table

| Track | Benefit estimator | Radius ε | Exchangeability unit | Guarantee claimed |
|-------|-------------------|----------|----------------------|-------------------|
| CIFAR stress v1 | LOO HistGradientBoosting on Z | Conformal quantile on LOO residuals | Per condition within seed | FA_u ≤ α under split + stated alignment |
| Natural shift OOF | Dev-fit GBR | OOF conformal on held-out groups | Seed / domain per protocol | No-harm empirical |
| Edge real phone | Dev-fit GBR | Split conformal S05/S06 | Session | FA_u ≤ α if sessions exchangeable |

**ε is NOT β.** β is the declared population drift budget in theory; ε is an empirical uncertainty radius fit on calibration residuals.
