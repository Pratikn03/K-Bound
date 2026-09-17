# Reviewer Response: Task T1 (Natural Shifts & Real-World Evaluation)

**Submission Targets:**
- Maintained long manuscript: `docs/research/kbound/kbound_tmlr.pdf`
- Maintained compact manuscript: `docs/research/kbound/kbound_short_final_draft.pdf`
- Section/table numbers below are historical locators; use the current maintained source labels when preparing the final response.

**Task Status:**
- **Previous Verdict:** 🟡 *Delimited; needs transparent scope documentation and empirical retention analysis*
- **Updated Verdict:** Completed CCT-20 evaluation with a negative selective-routing result; stronger natural-shift routing and population-protection requirements remain open. This is not full paper closure.

---

## 1. Executive Summary

The reviewer raised concerns regarding the real-world utility of Knowability-Guided Adaptation (KGA) beyond synthetic benchmark corruptions (such as CIFAR-10-C and ImageNet-C), specifically asking how KGA performs on natural distribution shifts (e.g., camera traps, satellite imagery, histopathology) and whether the method achieves selective adaptation routing in these settings.

We have addressed this rigorously and transparently:
1. **Sealed CCT-20 Empirical Evaluation (§8.4, Table 9 & Appendix B.4, Table 17):** We evaluate KGA across 45 cell evaluation conditions on Caltech Camera Traps (9 cis/trans target locations $\times$ 5 independently trained checkpoints). KGA issued 44 FREEZE and 1 ABSTAIN decisions (0 ADAPT decisions), perfectly retaining the frozen predictor and completely avoiding the severe degradation suffered by Always Adapt (which harmed accuracy on 44 of 45 cells).
2. **Methodological Framing as Conservative Retention:** We explicitly refrain from claiming a "selective routing victory" on CCT-20. One cell was helpful and 44 were harmful; KGA matched always-freeze and missed the helpful cell. This is conservative retention, not successful selective routing.
3. **Forensic Audit of Historical Natural Datasets (Appendix B.1, B.4, and Table 16):** We report a complete, unblinded provenance audit of 8 prior natural-shift candidate benchmarks (PovertyMap, FMoW, iWildCam, Camelyon17, RxRx1, So2Sat-LCZ42, PACS, and CIFAR-10.1), detailing exact development stops, metric misalignments, and feasibility criteria that led to their responsible withholding or delimitation.
4. **Delimited Scientific Scope (§9.2 & §9.3):** The manuscript clearly states that empirical abstention does not prove population non-identifiability, and that finite-sample certificates do not imply universal distribution-free coverage on unobserved natural environments.

---

## 2. Point-by-Point Reviewer Responses

### Comment 1: "Evaluation on real-world natural distribution shifts"
> *Reviewer: "Synthetic corruptions (noise, blur, weather) do not reflect the complex, multi-modal distribution shifts encountered in real deployments. Does K-Bound provide value on real natural datasets?"*

**Response:**
We completely agree that synthetic corruptions alone cannot establish deployment reliability. In response, we conducted a prospective, sealed evaluation on **CCT-20 (Caltech Camera Traps)**, a prominent natural-shift benchmark characterized by geographic, lighting, and environmental variation across wild camera locations.

The results are reported in **Section 8.4 (Table 9) and Appendix B.4 (Table 17)**:
- **Protocol:** 9 target camera locations evaluated across 5 distinct model checkpoints ($N = 45$ cell conditions). Features are extracted using frozen and adapted backbones under identical streaming sequences.
- **Candidate Outcomes:** Across all 45 cells, adaptation was harmful in 44 cells and helpful in one; the earlier statement of one tie and zero helpful cells was incorrect. The maintained generated CCT-20 tables are the numerical authority for accuracy and paired-oracle regret.
- **KGA Safe Retention:** KGA issued **44 FREEZE decisions and 1 ABSTAIN decision** ($0$ ADAPT decisions). By suppressing updates across all 45 cells, KGA preserved 100% of the frozen model's performance, achieving **$0.000$ excess regret** relative to Always Freeze and zero false adaptations ($\mathrm{FA}_u = 0.000$).
- **Scientific Conclusion:** KGA retained the frozen model on all cells, but abstention and certified FREEZE are different actions: the abstaining interval need not lie below zero, and the single helpful update was not selected. The record does not establish valid interval coverage on arbitrary natural shifts.

---

### Comment 2: "Selective routing vs. conservative retention"
> *Reviewer: "If KGA only freezes on CCT-20, does it actually perform selective adaptation, or is it merely behaving as an Always Freeze policy?"*

**Response:**
We address this distinction directly in §8.4 ("Conservative Retention, Not Selective Routing"):
- The synthetic-shift analyses have distinct protocols and outcome-exclusion limitations; their results do not establish natural-shift validity. Use their current generated tables rather than this response's superseded numerical summaries.
- Matching always-freeze on CCT-20 avoids the observed harmful candidate updates but is not paired-oracle optimal: one helpful update was missed.
- CCT-20 supports **conservative retention, not useful selective routing**. The action-sealing receipt records outcomes unopened before execution but prior aggregate metadata inspection (`literal_label_unopened: false`). Camera-location exchangeability and cross-split interval validity remain protocol assumptions, not proven deployment guarantees.

---

### Comment 3: "Audit of prior candidate benchmarks (WILDS, Satellite, Medical)"
> *Reviewer: "Earlier exploratory documents mentioned WILDS datasets (PovertyMap, FMoW, iWildCam, Camelyon17). Why are these not reported as primary confirmatory rows?"*

**Response:**
Rather than quietly omitting unfavorable exploratory runs, we provide a complete, transparent forensic accounting in **Appendix B.1, B.4, and Table 16 (`tab:compact-provenance`)**:
- **iWildCam:** Archived opened test-split records used a macro F1 metric that deviated from the official evaluation protocol, and the population split was unsealed prior to analysis. It is formally withheld as a non-promoted diagnostic.
- **PovertyMap-WILDS:** Preregistered development feasibility criterion required harm-detection AUC $\ge 0.65$. The empirical AUC reached $0.637$, halting the experiment at the development screen before test-set exposure.
- **FMoW-WILDS:** Protocol L reached held-out scoring matching Always Freeze, but failed cross-validation stability checks.
- **So2Sat-LCZ42:** Stopped at the development feasibility boundary due to candidate divergence across urban settlement types (§8.2, Appendix B.2).
- **Camelyon17 & RxRx1:** Retained as replayable diagnostics demonstrating conservative freeze retention under extreme domain shifts.

This level of negative-finding disclosure ensures complete auditability and prevents file-drawer bias.

---

## 3. Summary of Text and Table References in Submission

| Item | Location in Manuscript | Description |
| :--- | :--- | :--- |
| **CCT-20 Primary Table** | §8.4, Table 9 | 45-cell evaluation: 44 FREEZE, 1 ABSTAIN, 0 ADAPT; 0 false adaptations |
| **CCT-20 Breakdown by Location** | Appendix B.4, Table 17 | Location-by-checkpoint breakdown across cis- and trans-camera traps |
| **So2Sat Development Screen** | §8.2, Appendix B.2 | Complete audit of feasibility stop on urban settlement classification |
| **Experimental Provenance Ledger** | Appendix B.4, Table 16 | Complete status of all 12 candidate benchmark tracks across repo |
| **Scope Delimitations** | §9.2, §9.3 | Theoretical and empirical boundary conditions regarding natural shifts |
