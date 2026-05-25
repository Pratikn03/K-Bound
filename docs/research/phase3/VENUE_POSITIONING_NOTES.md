# VENUE POSITIONING & SUBMISSION NOTES

This document lists target venues and positioning strategies for the audited ELARA manuscript.

## 1. Target Venues

1. **NeurIPS (Replicability & Benchmark Track)**
   - *Framing:* Position as a highly rigorous, hostile empirical audit of a promising score-fusion framework. Emphasize the methodological rigor: 3-family evidence structure, validation-only calibration sweeps, pre-registered transfer limits, and correction of widely used DeLong variance estimation software bugs.
   
2. **ICML (Machine Learning Trust & Reliability)**
   - *Framing:* Focus on the empirical limits of score-level calibration under OOD shift. Highlight RGA-v2 gate failure as a negative result that proves the inadequacy of simple pooling operations for partial sensor failures, and motivate the need for distribution-drift metrics.

3. **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**
   - *Framing:* A comprehensive journal version extending the conference paper by including full proofs of the theorem stack, detail of all five evidence tables, and complete transcripts of the replication audit logs.

## 2. Positioning Guidelines for Authors

- **Acknowledge the Negative Outcome Honestly:** Do not attempt to spin the Family-D clean false-fire failure or the non-significant DeLong p-values as a success. Frame them as critical boundaries that save future researchers from repeating calibration errors.
- **Explain the DeLong Bug Clearly:** The mathematical explanation of the $250\times$ variance underestimation is a valuable diagnostic contribution to the ML community. Ensure it is featured prominently in the main text as a cautionary tale for reproducibility engineers.
- **Maintain Audited Status:** Refer to Family A as "audited static-reference evidence," Family B as "mechanism evidence," and Family D as "excluded transfer attempt." Do not use terms like "SOTA," "universal," or "deployment-ready."
