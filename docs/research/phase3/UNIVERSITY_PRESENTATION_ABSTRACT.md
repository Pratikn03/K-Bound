# UNIVERSITY PRESENTATION ABSTRACT & SEMINAR DETAILS

This document outlines the presentation details and seminar abstract for the doctoral thesis defense / department presentation of this work.

## 1. Seminar Title
**Auditing the Reliability Boundaries of Score-Level Multimodal Fusion under Domain Shift**

## 2. Seminar Abstract
Multimodal fusion has become a standard approach to enhance the robustness of anomaly detection systems in applications ranging from industrial inspection to cyber-security. Score-level fusion, where individual modality detectors output anomaly scores that are subsequently combined, represents a highly scalable and lightweight fusion paradigm. However, the lack of joint modality representation makes score-level gates highly vulnerable to distribution shifts and partial sensor failures.

This seminar presents an empirical audit of the Reliability-Guided Attention (RGA) score-fusion method. Using a rigorous three-family evidence schema, we trace RGA's performance across multiple public benchmarks and replication attempts. While RGA achieves modest improvements over a static attention reference under benign, normal-only conditions (Family A and B), we demonstrate that RGA-v2 gates fail to transfer to unseen domain environments. In particular, we detail the out-of-distribution transfer attempt on the Eyecandies dataset (Family D), where the clean false-fire rate escalated to 100%, far exceeding the 1% target budget. 

Additionally, we review a subtle bug discovered in the DeLong paired variance estimator that led to false claims of statistical significance, and present corrected non-significant results ($p \approx 0.31$--$0.33$). We conclude by highlighting key open challenges in out-of-distribution score calibration and outlining a future research agenda based on Kolmogorov-Smirnov distribution-drift gates to establish robust switching boundaries.

## 3. Recommended Audience
- Senior Trustworthy-ML researchers
- Graduate students in AI/ML reliability and computer vision
- Thesis committee members and academic reviewers
