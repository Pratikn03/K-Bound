# K-Bound: Theory → Proof → Code → Results

**Purpose:** A newcomer (reviewer, collaborator, future you) can follow this document from
theorem labels in the PDF to the proof location, numeric validator, implementation module,
and locked experiment JSON — without guessing from 10 scattered notebooks.

**Canonical status:** [`PROJECT_STATUS_AND_OPEN_PROBLEMS.md`](PROJECT_STATUS_AND_OPEN_PROBLEMS.md)  
**Live tour:** `bash docs/research/kbound/scripts/kbound_tour.sh` (or `python3 docs/research/kbound/scripts/kbound_tour.py`)

---

## 1. Decision pipeline (how theory becomes a button)

```mermaid
flowchart LR
  X[Test batch X] --> Z[Evidence Z = phi(X)]
  Z --> Bhat[Benefit estimate B_hat]
  Bhat --> Eps[OOF conformal radius epsilon]
  Eps --> Rule{ADAPT / FREEZE / ABSTAIN}
  Rule --> Policy[Regret + false-adapt metrics]
  Policy --> JSON[Locked results JSON]
  JSON --> TeX[Paper tables via make_tables.py]
```

| Step | Theory | Code | Notes |
|------|--------|------|-------|
| Evidence `Z` | `thm:frontier` margin `M(O)` | `analyze_F.py`, dataset runners | Label-free features only |
| Radius `ε` | Conformal / LOO split | `decide_kga()`, `score_kbound_holdout.py` | **Must be OOF** (audit 2026-06-26) |
| Decision | Certificate + impossibility | `kbound_pkg/kbound/certificate.py` | FA_u ≤ α under stated assumptions |
| Evaluation | Regret-to-oracle | `multiseed_paired_ci.py`, head-to-head harness | Pre-registered WIN/TIE/LOSE |

---

## 2. Theory spine (closed for the paper)

### 2.1 Identifiability frontier

| | |
|--|--|
| **Label** | `thm:frontier`, `thm:headline` |
| **Claim** | Benefit sign identifiable iff observable margin exceeds drift budget β |
| **Proof** | `paper/sections/main_theory_5.tex` |
| **Validators** | `experiments/kbound/theory_validation/val_frontier.py`, `val_benefit_frontier.py` |
| **Artifacts** | `experiments/kbound/theory_validation/results_frontier.json` |
| **Ledger** | KB-CLAIM-001 |
| **Empirical hook** | Stress grid: when margin large → KGA commits; when matched → abstains |

### 2.2 Impossibility / one-bit dichotomy

| | |
|--|--|
| **Label** | `thm:imp`, `thm:conj1-dichotomy` |
| **Claim** | Matched evidence + opposite benefit → minimax error ½; one bit is minimal supplement |
| **Proof** | `paper/sections/main_theory_5.tex` |
| **Validators** | `val_thm1_lecam.py`, `val_knowability_dichotomy.py`, `scripts/theory_extensions_validation.py` |
| **Artifacts** | `results_thm1_lecam.json`, `results/witness/witness_clean.json` |
| **Ledger** | KB-CLAIM-002, KB-CLAIM-025 (`conj:gen` **resolved negatively**) |
| **Notebook** | `notebooks/01_Problem_and_Theory.ipynb`, `04_Regression_and_Witness.ipynb` |

### 2.3 Certificate (false-adapt control)

| | |
|--|--|
| **Label** | Guarantee box; `thm:anytime` (extension) |
| **Claim** | FA_u ≤ α under split conformal + assumptions; anytime extension under optional stopping |
| **Proof** | Main text + App. theory extensions in `kbound.tex` |
| **Validators** | `val_thm3_evalue.py`, `theory_v2/val_sequential_anytime.py`, `theory_v2/val_multicandidate.py` |
| **Implementation** | `kbound_pkg/kbound/certificate.py` |
| **Ledger** | KB-CLAIM-003 (**supported**); KB-CLAIM-004 FA_c (**withdrawn**) |
| **Notebook** | `notebooks/07_Certificate_and_Calibration.ipynb` |
| **Empirical hook** | Gate table: only KGA keeps FA_u=0 across 432 cells |

### 2.4 Unconditional weakest one-bit class

| | |
|--|--|
| **Label** | `thm:uncond-weakest` |
| **Claim** | Weakest classes = dominance polytopes; GP is a collapsing face |
| **Proof** | `theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md` |
| **Validator** | `theory_v2/val_unconditional_weakest.py` (machine-checked, 0 mismatches) |
| **Artifact** | `theory_v2/unconditional_weakest_results.json` |
| **Paper** | `kbound.tex` appendix |

---

## 3. Empirical claims (what results prove which story)

### Tier A — Headline wins (beats-both)

| Claim ID | Result | Artifact | Rerun |
|----------|--------|----------|-------|
| KB-CLAIM-010 | CIFAR stress grid Tent/EATA | `stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json` | `kbtrain.sh cifar10c` |
| KB-CLAIM-026 | vs POEM/AETTA mixed WIN | `mixed_headtohead_v1/HEADTOHEAD_RESULTS_*.json` | `run_all_headtohead.sh` |
| KB-CLAIM-011 | ImageNet-C SAR harmful | `results_source.json` | locked SAR run |
| KB-CLAIM-024 | Mixed OOF aggregate | `mixed_protocol_oof_v2/` | `mixed_stream_kbound.py` |
| KB-CLAIM-027 | D33 controlled multimodal | `controlled_multimodal_d33/results.json` | D33 runner |

### Tier B — Natural shifts (no-harm only)

| Claim ID | Dataset | Artifact | Rerun |
|----------|---------|----------|-------|
| KB-CLAIM-020 | Office-Home M v2 | `results_source.json` | `kbtrain.sh protocol-m-v2` |
| KB-CLAIM-021 | iWildCam H v2 | `results_source.json` | `kbtrain.sh protocol-h-v2` |
| (panel) | Camelyon, RxRx1, PACS | `results_source.json` | `kbtrain.sh final-all` |

**Withdrawn:** KB-CLAIM-022 (Camelyon pooled beats-both), KB-CLAIM-023 (in-sample mixed aggregate).

### Tier C — Pending / non-headline

| Claim ID | Status | Notes |
|----------|--------|-------|
| KB-CLAIM-030 | pending | Physical camera R2 — protocol only |
| KB-CLAIM-040 | supported | Assumption audit (falsification-oriented) |

---

## 4. How proof connects to results (concrete example)

**Story:** Certificate prevents harmful adaptation on mixed harmful+helpful CIFAR conditions.

1. **Theory** says: if you cannot identify sign Δ, abstain; if you commit ADAPT with certificate, FA_u ≤ α.
2. **Code** `decide_kga()` computes LOO `B_hat` and conformal ε per condition → ADAPT/FREEZE/ABSTAIN.
3. **Stress grid** logs per-condition decisions for Tent adapter (432 conditions × 5 seeds).
4. **Analysis** `LOCKED_ANALYSIS_RESULTS.json`: KGA regret 0.0016 vs adapt 0.0079, FA_u=0.
5. **Head-to-head** same records + POEM/AETTA ports → `VERDICT: WIN`.
6. **Paper** `tab:headtohead-poem-aetta` via `make_tables.py` / inline TeX.

A reviewer can trace: `kbound_short.tex` table → JSON → `run_mixed_headtohead.py` → `decide_kga` → theorem guarantee box.

---

## 5. Validator inventory

Run all lightweight checks:

```bash
bash docs/research/kbound/scripts/reproduce_submission.sh
```

| Script | Checks |
|--------|--------|
| `val_thm1_lecam.py` | Le Cam lower bound + witness worlds |
| `val_thm2_regret.py` | Regret decomposition identity |
| `val_thm3_evalue.py` | E-value / false-adapt certificate |
| `val_thm5_multiclass.py` | Multiclass sign Δ reduction |
| `val_frontier.py` | Frontier margin thresholding |
| `val_knowability_dichotomy.py` | Dichotomy numerics |
| `theory_v2/val_unconditional_weakest.py` | Polytope weakest-class enumeration |
| `theory_v2/val_sequential_anytime.py` | Anytime FA_u |
| `theory_v2/val_multicandidate.py` | Family-wise FA_u |
| `scripts/theory_extensions_validation.py` | Le Cam + forced abstention + multiclass Δ |

CI job `.github/workflows/kbound-ci.yml` runs `val_thm*.py` on push.

---

## 6. What the 10 notebooks do / do not cover

| Covered well | Partially stale (pre–POEM/AETTA) |
|--------------|-----------------------------------|
| Theorem validator JSON loading (01) | `00_KBound_Reproduction.ipynb` (123-task ELARA archive) |
| Trichotomy demos (02) | Artifact lists in 09 (missing new HEADTOHEAD paths) |
| Harmful/mixed regimes (03) | Some figures predate OOF mixed v2 |
| Certificate intuition (07) | |
| **Master guide (00 new)** | **Fills the gap — use this first** |

---

## 7. Genuinely open (not in scope of current paper)

Listed in `PROJECT_STATUS_AND_OPEN_PROBLEMS.md` §1 — gen-capacity without R1/R2, tight rates,
minimax optimality, multiclass anytime, physical R2 data collection.

These are **research frontier**, not missing implementation for submission.
