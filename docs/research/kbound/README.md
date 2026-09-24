# K-Bound Research Guide

This directory is the maintained research surface for **K-Bound** and **KGA**. It contains the manuscript LaTeX sources, compiled release deliverables, Lean 4 formalization, canonical benchmark manifests, and replication protocols.

---

## 📄 Canonical Manuscripts & Deliverables

| Deliverable | Format | Size / Pages | Location / Link |
| :--- | :---: | :---: | :--- |
| **Complete Unified Manuscript** | PDF | **70 pages** | [`KBound_Complete_70Pages.pdf`](KBound_Complete_70Pages.pdf) |
| **Main Paper (Without Appendices)** | PDF | **24 pages** | [`theorem_assumption_completion/deliverables/KBound_Main_Without_Appendices.pdf`](theorem_assumption_completion/deliverables/KBound_Main_Without_Appendices.pdf) |
| **Supplementary Material** | PDF | **46 pages** | [`theorem_assumption_completion/deliverables/KBound_Appendices.pdf`](theorem_assumption_completion/deliverables/KBound_Appendices.pdf) |
| **Full Manuscript (Word)** | DOCX | **95 pages** | [`theorem_assumption_completion/deliverables/KBound_With_Appendices.docx`](theorem_assumption_completion/deliverables/KBound_With_Appendices.docx) |
| **Compact Camera-Ready PDF** | PDF | **59 pages** | [`kbound_short_final_draft.pdf`](kbound_short_final_draft.pdf) |
| **Long Companion PDF** | PDF | **66 pages** | [`kbound_tmlr.pdf`](kbound_tmlr.pdf) |
| **Delivery Manifest & Hashes** | JSON / TXT | — | [`theorem_assumption_completion/deliverables/DELIVERY_MANIFEST.json`](theorem_assumption_completion/deliverables/DELIVERY_MANIFEST.json) |

---

## 🧭 Navigation Guide

| Goal | Entry point |
|---|---|
| **Documentation Index** | [`DOCS_INDEX.md`](DOCS_INDEX.md) — Comprehensive guide to all research documents |
| **Read the compact submission source** | [`kbound_submission.tex`](kbound_submission.tex) and [`kbound_submission_body.tex`](kbound_submission_body.tex) |
| **Read the long companion source** | [`kbound_tmlr.tex`](kbound_tmlr.tex) and [`kbound_submission_supplement.tex`](kbound_submission_supplement.tex) |
| **Inspect canonical panel numbers** | [`../../../experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`](../../../experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json) |
| **Audit claim-to-artifact links** | [`KBOUND_SHORT_CLAIM_MANIFEST.md`](KBOUND_SHORT_CLAIM_MANIFEST.md) |
| **Obtain the benchmark datasets** | [`../../../DATA.md`](../../../DATA.md) — Per-dataset split, licence, acquisition notes |
| **Reproduce experimental results** | [`REPRODUCE.md`](REPRODUCE.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) |
| **Run independent replication** | [`INDEPENDENT_REPLICATION_PROTOCOL.md`](INDEPENDENT_REPLICATION_PROTOCOL.md) |
| **Run on Apple Silicon / Mac** | [`RUN_ON_MAC.md`](RUN_ON_MAC.md) |
| **Understand data storage policy** | [`EXTERNAL_STORAGE_POLICY.md`](EXTERNAL_STORAGE_POLICY.md) |
| **Explore Lean 4 formal proofs** | [`formal/README.md`](formal/README.md) — 142 scoped machine-checked declarations |
| **Browse historical dev notes & runsheets** | [`archive/internal_dev_notes/`](archive/internal_dev_notes/) |
| **Browse historical forensic audits** | [`archive/audits/`](archive/audits/) |

---

## 📌 Core Theoretical Concepts & Fixed Terminology

* **K-Bound**: Population partial-identification theory and the formal $\text{ADAPT} / \text{FREEZE} / \text{ABSTAIN}$ decision framework.
* **KGA (Knowability-Guided Adaptation)**: Finite-sample empirical wrapper that maps label-free score evidence to certified action intervals.
* **Population Frontier**: Governed by observable margin $M$, hidden residual $\gamma$, and residual bound $\beta$.
* **Empirical Certificate**: Governed by point estimate $\widehat\Delta$, empirical radius $\varepsilon$, and confidence level $\alpha$.
* **Abstain**: Withhold parameter update; continue inference using the frozen pre-adaptation model.

---

## 📊 Summary of Empirical Evidence Tiers

* **Controlled Mixed-Regime Routing:** CIFAR-10-C Tent achieves verified routing over 5 model seeds and 432 cells per seed. Resampling down to 6 corruption-family clusters confirms robustness.
* **Point-Estimate Controlled Routing:** CIFAR-10-C EATA beats both fixed policies at its operating point, with its adapt-side corruption-family interval touching zero.
* **Prospective Natural Shift:** CCT-20 achieves 44 FREEZE, 0 ADAPT, 1 ABSTAIN, passing the locked safe-utility bootstrap check without fabricating artificial adaptation wins.
* **Negative Development Gate:** So2Sat-LCZ42 candidate selection found no feasible adapter and terminated before target evaluation, adhering to a strict fail-closed contract.
* **One-Sided Diagnostics:** Office-Home, iWildCam, Camelyon17 OOD, and RxRx1 reproduce stable baseline behavior or one-sided safe retention.

---

## 🔨 Compiling Manuscripts from Source

```bash
# Compile the main submission driver
latexmk -pdf -interaction=nonstopmode kbound_submission.tex

# Compile the long companion driver
latexmk -pdf -interaction=nonstopmode kbound_tmlr.tex

# Compile the standalone 24-page main paper
latexmk -pdf -interaction=nonstopmode kbound_short_main.tex

# Compile the standalone 46-page supplementary material
latexmk -pdf -interaction=nonstopmode kbound_short_supplement.tex
```
