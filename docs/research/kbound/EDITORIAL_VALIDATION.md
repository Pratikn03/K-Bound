# Editorial & Scientific Validation Report

**Target Documents:**
1. `kbound_short_final_draft.pdf` (48 pages, named professor-facing submission)
2. `kbound_tmlr.pdf` (55 pages, single-column anonymous TMLR review format)

**Date:** September 16, 2026  
**Auditor:** Antigravity AI  

---

## 1. Executed Commands and Test Results

### 1.1 Source Repository & Snapshot Verification
- **Pre-edit Snapshot:** `docs/research/kbound/snapshot_pre_copyedit_20260916/` verified containing pre-edit `.tex` sources.
- **Git Branch Hygiene:** Sole active branch `main`, cleanly synchronized. No reset, push, or destructive actions taken.

### 1.2 Python Canonical Data & Manuscript Validators
| Script / Command | Interpreter | Exit Status | Output / Verdict |
| :--- | :--- | :--- | :--- |
| `docs/research/kbound/scripts/validate_canonical_release_data.py` | Python 3.9 | **0** | `canonical release data: PASS` |
| `docs/research/kbound/scripts/verify_cifar_current_arithmetic.py` | Python 3.9 | **0** | `OK: current CIFAR arithmetic release binding verified` |
| `docs/research/kbound/scripts/build_so2sat_numbers.py` | Python 3.9 | **0** | `wrote paper/generated/so2sat_numbers.tex` |
| `src/scripts/validate_manuscript_claims.py` | Python 3.9 | **0** | Clean pass (0 claim discrepancies) |
| `docs/research/kbound/scripts/make_tables.py` | Python 3.9 | **0** | Recomputed all primary, auxiliary, and CCT-20 tables |

### 1.3 Theoretical Theorem Validators
Executed across `experiments/kbound/theory_validation/`:
| Validator | Target Theory | Exit Status | Summary of Results |
| :--- | :--- | :--- | :--- |
| `val_thm1_lecam.py` | Theorem 1 / Thm 3 (Le Cam lower bound) | **0** | Minimax audit floor matches $\beta$ exactly; anytime false adapt controlled |
| `val_thm3_evalue.py` | Theorem 3 (e-values & anytime valid audit) | **0** | Worst-case anytime $\mathrm{FA} = 0.0626 \le \alpha=0.10$; supermartingale $E[E_t^+] \le 1$ holds |
| `val_thm5_multiclass.py` | Theorem 5 (multiclass 0/1 & regression) | **0** | Pointwise identity max err $5.6 \times 10^{-13}$; 100% sign match; covariate vs concept shift separation exact |
| `val_thm9prime_drift.py` | Theorem 9' (anytime drift control) | **0** | Corrected policy controls $\mathrm{FA} \le 0.0605 \le \alpha=0.10$ across all $\beta$; detection power $=0.997$ |

### 1.4 PDF Compiler & Log Verification
| Target PDF | Engine | Flags | Pages | Undefined Refs | Exit Status | Output SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `kbound_short_final_draft.pdf` | pdfTeX (latexmk) | `-g -pdf -halt-on-error` | **48** | **0** | **0** | `e05904980811d9053bb047d861b6bf07501f331f072919a1bf082c588cc0b098` |
| `kbound_tmlr.pdf` | pdfTeX (latexmk) | `-g -pdf -halt-on-error` | **55** | **0** | **0** | `67024ca156900f76ea5bb51074977046ac04886b510f2f4132472d734ee7ca77` |

---

## 2. Exact Numerical Reconciliation Ledger

Every row across Table 15, Table 16, Table 25, and narrative text was reconciled directly against `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`:

| Row / Setting | $n$ | A | F | U | False A | $\mathrm{FA}_u$ | Commitment | KGA Regret | Always Adapt | Always Freeze |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ImageNet-C Tent** | 135 | 1 | 0 | 134 | 0 | 0.0000 | 0.0074 | **0.0139** | 0.0191 | 0.0145 |
| **ImageNet-C EATA** | 135 | 107 | 1 | 27 | 0 | 0.0000 | 0.8000 | **0.0031** | 0.0001 | 0.0342 |
| **ImageNet-C SAR** | 135 | 8 | 2 | 125 | 2 | 0.0148 | 0.0741 | **0.0348** | 0.0529 | 0.0319 |
| **Camelyon17 B-v2 SAR** | 108 | 50 | 0 | 58 | 1 | 0.0093 | 0.4630 | **0.0240** | 0.0016 | 0.1001 |
| **ImageNet-R Panel** | 480 | 176 | 36 | 268 | 3 | 0.0063 | 0.4417 | **0.0136** | 0.0064 | 0.0325 |
| **CIFAR-10-C Tent** | 2160 | 1094 | 334 | 732 | 0 | 0.0000 | 0.6611 | **0.0018** | 0.0080 | 0.1239 |
| **CIFAR-10-C EATA** | 2160 | 1207 | 118 | 835 | 0 | 0.0000 | 0.6134 | **0.0016** | 0.0033 | 0.1313 |
| **CIFAR-10-C SAR** | 2160 | 1430 | 0 | 730 | 1 | 0.0005 | 0.6620 | **0.0158** | 0.0004 | 0.1197 |

### Mandatory Arithmetic Checks
1. $\mathbf{A + F + U = n}$:
   - ImageNet-C Tent: $1 + 0 + 134 = 135$ $\checkmark$
   - ImageNet-C EATA: $107 + 1 + 27 = 135$ $\checkmark$
   - ImageNet-C SAR: $8 + 2 + 125 = 135$ $\checkmark$
   - Camelyon17 B-v2 SAR: $50 + 0 + 58 = 108$ $\checkmark$
   - ImageNet-R: $176 + 36 + 268 = 480$ $\checkmark$
   - CIFAR Tent: $1094 + 334 + 732 = 2160$ $\checkmark$
2. $\mathbf{\text{False A} \le A}$:
   - ImageNet-C Tent: $0 \le 1$ $\checkmark$
   - ImageNet-C EATA: $0 \le 107$ $\checkmark$
   - ImageNet-C SAR: $2 \le 8$ $\checkmark$
   - Camelyon17 B-v2 SAR: $1 \le 50$ $\checkmark$
   - ImageNet-R: $3 \le 176$ $\checkmark$
   - CIFAR SAR: $1 \le 1430$ $\checkmark$
3. $\mathbf{\text{Rates Use Declared Denominators}}$:
   - $\mathrm{FA}_u = \text{False A} / n$:
     - ImageNet-C SAR: $2 / 135 = 0.01481... \approx 0.0148$ $\checkmark$
     - Camelyon17 B-v2 SAR: $1 / 108 = 0.009259... \approx 0.0093$ $\checkmark$
     - ImageNet-R: $3 / 480 = 0.00625 \approx 0.0063$ $\checkmark$
   - $\mathrm{FA}_c = \text{False A} / A$:
     - ImageNet-C SAR: $2 / 8 = 0.250$ $\checkmark$
     - Camelyon17 B-v2 SAR: $1 / 50 = 0.020$ $\checkmark$
     - When $A=0$ (e.g. RxRx1, CCT-20), $\mathrm{FA}_c$ is explicitly reported as undefined, not zero $\checkmark$
4. $\mathbf{\text{Commitment Rate}} = (A + F) / n$:
   - ImageNet-C Tent: $1 / 135 = 0.0074$ $\checkmark$
   - ImageNet-C EATA: $108 / 135 = 0.8000$ $\checkmark$
   - ImageNet-C SAR: $10 / 135 = 0.0741$ $\checkmark$
   - Camelyon17 B-v2 SAR: $50 / 108 = 0.4630$ $\checkmark$
   - ImageNet-R: $212 / 480 = 0.4417$ $\checkmark$
5. $\mathbf{\text{Non-ADAPT Service Defaults to Always-Freeze}}$:
   - Where a policy chooses 0 ADAPT decisions (e.g., RxRx1, CIFAR-10.1, CCT-20), served model accuracy and paired regret match always-freeze identically on the evaluated records $\checkmark$

---

## 3. Page-by-Page Layout and Visual Assessment

- **Pages 1–19 (Main Body):**
  - Text, equations, section heads, and floats fit within declared geometry.
  - Table 4 (Theory–Algorithm Bridge) cleanly formatted, distinct symbol roles preserved.
  - Section 8 results and discussion flow smoothly into Section 9 limitations.
  - Section 10 (Conclusion) ends on Page 19.
- **Pages 19–20 (References):**
  - References [1]–[55] set in `\scriptsize` with tight list separation.
  - Completely fits across the bottom of page 19 and page 20.
  - No spill onto page 21.
- **Pages 21–48 (Supplementary Material):**
  - Appendix A begins cleanly on page 21.
  - Table 17 (p. 28) converted to full-width `table*`, eliminating column clipping for "False A".
  - All supplementary tables (Tables 5–25) render cleanly within printable margins.
  - Figures 4, 5, 6 render with clean linearized PDF 1.4 assets.
  - Final Appendix N ends cleanly on page 48.

---

## 4. Unresolved Items & Scope Status

1. **Integrated Release Pipeline (`release_candidate.sh all`):**
   - *Status:* **PENDING / UNPASSED**.
   - *Reason:* As disclosed in Appendix N.2, the repository-wide integrated release gate requires un-offloaded multi-gigabyte external datasets (e.g. raw Camelyon17 WSI, full So2Sat, WILDS), upstream native CUDA containers for external baseline ports, and multi-checkpoint training clusters. Successful presentation builds (`build_pdfs.sh`) and local Python/Lean validator runs do not constitute an end-to-end release seal.
2. **External Baselines Implementation Outputs:**
   - Kim et al. TTA/ALine: 0/15 available local scored outputs (incomplete local evaluation, code exists).
   - Baek et al. ALine: 11/15 available.
   - POEM / AETTA: require upstream external library versions and native CUDA configurations.
