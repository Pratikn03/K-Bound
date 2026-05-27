# FINAL MANUSCRIPT HOSTILE REVIEW REPORT

**Reviewer Role:** Senior Trustworthy-ML Empirical Researcher and Hostile Reviewer
**Manuscript Version:** Phase-3-final integration (tex drafts updated)
**Decision:** **`MANUSCRIPT_READY_FOR_FINAL_PDF_BUILD`**

---

## Evaluation Checklist

### 1. Abstract Check
- **Question:** Does the abstract overclaim Family D or present it as positive?
- **Finding:** No. The abstract explicitly calls the Eyecandies study an "attempted held-out transfer" that "failed to meet the frozen clean false-fire budget ($\le 0.010$ budget vs $1.0$ observed)" and is "excluded from primary evidence." It reports the corrected DeLong p-values ($0.3323$ and $0.3127$) to show non-significance.
- **Verdict:** PASS.

### 2. Family-A Claims and Comparators
- **Question:** Does any text describe Family A as confirmatory or claim strongest-baseline superiority?
- **Finding:** No. The paper explicitly labels Family A as "Audited static-reference evidence" and states it is compared against a fixed `static_attention` reference only. It emphasizes that this is not confirmatory replication or strongest-baseline superiority.
- **Verdict:** PASS.

### 3. Modality Independence and Effect Sizes
- **Question:** Are VisA and LOCO-AD proxy-view limitations stated? Is the UNSW effect size qualified?
- **Finding:** Yes. VisA (A-POWERED-4) and LOCO-AD (A-POWERED-3) are explicitly qualified as `derived_view_proxy` pairings (Sobel edge and RGB + edge), noting they do not support independent modality claims. UNSW-NB15 (A-POWERED-5) is explicitly qualified as representing a "small practical effect size (+0.0095)."
- **Verdict:** PASS.

### 4. Family-B Replication and Dual Estimators
- **Question:** Is B1 reported as reproduced? Is B2 reported with dual numbers?
- **Finding:** Yes. B1 zero-attack delta is reported as +0.0507 (Holm $p = 4.31\times 10^{-12}$) and marked as reproduced. B2 max-attack is reported side-by-side with the Phase-1 target (+0.0319) and Phase-2 ensemble delta (+0.0939), explicitly noting the estimator change and stating that it is not an exact magnitude reproduction.
- **Verdict:** PASS.

### 5. RGA-v2 and Domain Shift Negatives
- **Question:** Are RGA-v2 gate failures and domain composition shift negatives retained?
- **Finding:** Yes. The sweep shows that G1/G2/G3 failed clean false-fire controls (firing at rate 1.000) and were not promoted. B-MECH-3S domain composition shift also showed that domain-aware gates did not reduce false fire (rate 1.000). These are retained as key negative results.
- **Verdict:** PASS.

### 6. Certificates Frame
- **Question:** Are certificates qualified as retrospective evaluations?
- **Finding:** Yes. The certificates are explicitly labeled as retrospective stress protocol evaluations, and they carry a loud warning that they are NOT production safety certificates or real-world deployment guarantees.
- **Verdict:** PASS.

### 7. Family-D Exclusion and DeLong Bug
- **Question:** Is Family D excluded? Is the double-division bug explained? Are p-values corrected?
- **Finding:** Yes. A dedicated subsection details the exclusion of Family D due to validation clean false-fire budget failure. The mathematical double-division bug in `_delong_auc_variance` and `_delong_paired_test` is explained, noting that the corrected p-values are $0.3323$ (D-EYE-1) and $0.3127$ (D-EYE-2). The buggy $p = 0.0000$ values are completely absent except when labeled as errors.
- **Verdict:** PASS.

### 8. Caption and Wording Compliance
- **Question:** Do captions use required tags?
- **Finding:** Yes. Captions in Family-A tables include "AUDITED STATIC-REFERENCE EVIDENCE". Captions in Family-D tables include "EXCLUDED FROM PRIMARY EVIDENCE". Certificates include "retrospective evaluation certificate".
- **Verdict:** PASS.

### 9. Numerical Traceability
- **Question:** Are all values traceable to authoritative sources?
- **Finding:** Yes, all numbers match `family_a_v2_primary_cell_level_holm_k5.csv`, `family_b_primary_replication_holm_k2.csv`, and the Family-D audit reports exactly.
- **Verdict:** PASS.

---

## Reviewer Reflection

Although Family D did not transfer successfully and RGA-v2 failed to solve partial failures under clean false-fire limits, the paper makes a valuable methodological contribution:
1. It establishes a disciplined, audited score-fusion benchmark schema.
2. It documents the dilution boundaries of mean-pooling gates.
3. It demonstrates that distribution-shift signals (like KS drift) are necessary triggers under coherent collapse.
4. It honestly reports out-of-distribution calibration failure as a central barrier, mapping out clear future research directions.

The formatting is compliant, and the tone is appropriately self-critical and scientifically rigorous. There are no remaining overclaims.

**Final Recommendation:** Proceed to compile the final paper and thesis PDFs.
