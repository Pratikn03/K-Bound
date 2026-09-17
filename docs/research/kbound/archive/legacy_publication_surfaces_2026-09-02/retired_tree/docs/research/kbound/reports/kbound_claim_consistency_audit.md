# K-Bound Claim Consistency Audit

**Date:** 2026-06-25  
**Scope:** `kbound_short.tex`, `kbound.tex`, edge camera tables, `results_source.json`

---

## Blockers fixed in this pass

| ID | Location | Issue | Severity | Correction |
|----|----------|-------|----------|------------|
| B-01 | `edge/.../camera_tables_values.tex` | R2 populated with 25% balanced acc from mock/helpful-dominated dev replay | **blocker** | Replaced with `RESULT PENDING` macros; archived prior file |
| B-02 | `11_export_camera_tables.py` | Exported dev metrics as publication TeX | **blocker** | Gated export on `claim_eligibility.json` |
| B-03 | `kbound_short.tex` | Missing explicit FA_u vs FA_c / β vs ε disclosure box | **major** | Added guarantee box (§guarantees) |
| B-04 | `claim_ledger.json` | No machine-readable claim status | **major** | Created ledger |

---

## Major issues (documented; partial fix)

| ID | Location | Issue | Severity | Required correction |
|----|----------|-------|----------|---------------------|
| M-01 | `cifar_tent_mps_v2.py` | LOO-GBR + LOO residuals described as generic "split conformal" without exchangeability unit | major | State unit = condition; do not claim jackknife+ |
| M-02 | `kbound_short.tex` §mixed aggregate | Withdrawn 13–24× figures still narrated | major | Keep "withdrawn pending OOF v2" only |
| M-03 | `KBOUND_WIN_BOOTSTRAP_CIS.json` | In-sample CIs in repo | major | Audit-only; use `_oof.json` |
| M-04 | Physical edge | 1216 clips include mock-noise captures | major | Re-capture S01–S10 without `--mock` |

---

## Moderate issues

| ID | Location | Issue | Severity | Required correction |
|----|----------|-------|----------|---------------------|
| O-01 | `foldin_multiseed_results.py` | Referenced but missing | moderate | Stub or remove references |
| O-02 | `kbtrain.sh` vs `.venv` | Dual venv paths | moderate | Document in `reproduce_submission.sh` |
| O-03 | ImageNet-R | Weak-evidence regime | moderate | Supporting null only (already in limits) |

---

## Metric semantics verified

- **FA_u** = P(adapt ∧ harmful) — theorem targets this.
- **FA_c** = P(harmful | adapt) — descriptive only; **not bounded by α**.
- **β** = population drift budget (theory).
- **ε** = empirical conformal radius from calibration residuals — **not an estimate of β**.

---

## Scientific conclusion impact

Corrections **narrow** empirical claims (physical R2 pending; mixed aggregate withdrawn) but **strengthen** integrity. Core theory + CIFAR stress + natural-shift no-harm claims **preserved** under OOF protocol.
