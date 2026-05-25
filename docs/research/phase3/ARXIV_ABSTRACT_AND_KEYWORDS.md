# ARXIV SUBMISSION METADATA: ABSTRACT & KEYWORDS

This document contains the formatted abstract and metadata keywords for the arXiv submission of the ELARA manuscript.

## 1. Title
**Empirical Limits of Score-Level Stress-Response Fusion in Multimodal Anomaly Detection**

## 2. Abstract
Supervised multimodal score-level fusion offers a lightweight path to integrate complementary sensor views in anomaly detection (AD). However, under severe out-of-distribution (OOD) shift, score-fusion models are prone to catastrophic validation failures. In this work, we present an audited empirical evaluation of the Reliability-Guided Attention (RGA) framework. Evaluating across nine primary cells (Family A) and two mechanism replication endpoints (Family B), we confirm that while RGA can improve performance relative to a static attention baseline when training normal-only models under moderate corruption, it fails to solve partial sensor failure under clean false-fire rate constraints. Specifically, a pre-registered transfer evaluation on the Eyecandies dataset (Family D) failed its clean false-fire rate budget ($\le 0.010$ target vs. $1.0$ observed), rendering the primary transfer attempt invalid. We present corrected DeLong paired $p$-values ($p=0.3323$ for D-EYE-1 and $p=0.3127$ for D-EYE-2) and mathematically document a double-division variance calculation bug that had previously masked these non-significant outcomes. We conclude that score-level fusion is fundamentally limited under domain shift, and propose that future architectures must incorporate distribution-drift metrics (such as Kolmogorov-Smirnov statistics) directly into the gating decision.

## 3. Subject Classifications
- **Primary:** Computer Vision and Pattern Recognition (cs.CV)
- **Secondary:** Machine Learning (cs.LG)

## 4. Keywords
- Multimodal Anomaly Detection
- Score-Level Fusion
- Reliability-Guided Attention (RGA)
- Empirical Replication Audit
- Out-of-Distribution Calibration
- DeLong Paired Inference
- Domain Shift Limitations
