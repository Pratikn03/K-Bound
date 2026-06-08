# KGA Breadth Table — Existing Score Archives

**Framework**: f0 = best-val-AUC single detector (frozen); f_a = mean-score ensemble of all 6 detectors (adapted); Z = label-free score statistics; B = AUC(f_a) − AUC(f0) on held-out test set.  
KGA decision rule: LOO gradient-boosted Bhat ± split-conformal ε (α=0.10), identical to cifar_tent_mps_v2.py.

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| N tasks | 62 |
| Harmful base rate (B<0) | 0.855 |
| Mean true B | -0.0455 |
| Conformal ε | 0.0920 |
| α (miscoverage target) | 0.1 |
| Coverage | 0.113 |
| ADAPT decisions | 0 |
| FREEZE decisions | 7 |
| ABSTAIN decisions | 55 |
| False-adapt rate (B<0 given ADAPT) | N/A |
| Adapt precision (B>0 given ADAPT) | N/A |
| Mean AUC — always-adapt | 0.6969 |
| Mean AUC — always-freeze | 0.7424 |
| Mean AUC — K-Bound | 0.7424 |
| Mean AUC — oracle | 0.7479 |
| Regret vs oracle — always-adapt | 0.0510 |
| Regret vs oracle — always-freeze | 0.0055 |
| Regret vs oracle — K-Bound | 0.0055 |
| Beats both baselines | False |
| Ties best baseline | False |
| Loses to a baseline | False |

## Per-Task Results

| # | Dataset | Domain | N | f0 | a0 | aa | B | Decision | KGA AUC | Oracle AUC | Regime |
|---|---------|--------|---|----|----|----|----|----------|---------|------------|--------|
| 1 | adb_10_cover | adbench | 2000 | LOF | 0.964 | 0.962 | -0.002 | ABSTAIN | 0.964 | 0.964 | marginal |
| 2 | adb_11_donors | adbench | 2000 | ECOD | 0.901 | 0.865 | -0.036 | FREEZE | 0.901 | 0.901 | harmful |
| 3 | adb_12_fault | adbench | 486 | KNN | 0.704 | 0.553 | -0.151 | ABSTAIN | 0.704 | 0.704 | harmful |
| 4 | adb_13_fraud | adbench | 2000 | LOF | 0.979 | 0.973 | -0.006 | ABSTAIN | 0.979 | 0.979 | marginal |
| 5 | adb_15_Hepatitis | adbench | 20 | COPOD | 0.588 | 0.569 | -0.020 | ABSTAIN | 0.588 | 0.588 | marginal |
| 6 | adb_16_http | adbench | 2000 | IForest | 1.000 | 0.991 | -0.009 | ABSTAIN | 1.000 | 1.000 | marginal |
| 7 | adb_17_InternetAds | adbench | 492 | ECOD | 0.676 | 0.651 | -0.024 | ABSTAIN | 0.676 | 0.676 | harmful |
| 8 | adb_18_Ionosphere | adbench | 88 | KNN | 0.919 | 0.873 | -0.046 | ABSTAIN | 0.919 | 0.919 | harmful |
| 9 | adb_19_landsat | adbench | 1609 | KNN | 0.585 | 0.462 | -0.123 | ABSTAIN | 0.585 | 0.585 | harmful |
| 10 | adb_1_ALOI | adbench | 2000 | LOF | 0.604 | 0.546 | -0.059 | ABSTAIN | 0.604 | 0.604 | harmful |
| 11 | adb_20_letter | adbench | 400 | KNN | 0.854 | 0.701 | -0.152 | ABSTAIN | 0.854 | 0.854 | harmful |
| 12 | adb_22_magic.gamma | adbench | 2000 | KNN | 0.796 | 0.718 | -0.078 | ABSTAIN | 0.796 | 0.796 | harmful |
| 13 | adb_23_mammography | adbench | 2000 | COPOD | 0.873 | 0.842 | -0.031 | FREEZE | 0.873 | 0.873 | harmful |
| 14 | adb_24_mnist | adbench | 1901 | KNN | 0.846 | 0.829 | -0.018 | ABSTAIN | 0.846 | 0.846 | marginal |
| 15 | adb_25_musk | adbench | 766 | IForest | 1.000 | 0.980 | -0.020 | ABSTAIN | 1.000 | 1.000 | marginal |
| 16 | adb_26_optdigits | adbench | 1304 | IForest | 0.714 | 0.609 | -0.105 | ABSTAIN | 0.714 | 0.714 | harmful |
| 17 | adb_27_PageBlocks | adbench | 1349 | OCSVM | 0.914 | 0.921 | +0.007 | ABSTAIN | 0.914 | 0.921 | marginal |
| 18 | adb_28_pendigits | adbench | 1718 | IForest | 0.927 | 0.928 | +0.002 | ABSTAIN | 0.927 | 0.928 | marginal |
| 19 | adb_29_Pima | adbench | 192 | KNN | 0.707 | 0.654 | -0.053 | ABSTAIN | 0.707 | 0.707 | harmful |
| 20 | adb_2_annthyroid | adbench | 1800 | IForest | 0.814 | 0.781 | -0.032 | ABSTAIN | 0.814 | 0.814 | harmful |
| 21 | adb_30_satellite | adbench | 1609 | IForest | 0.688 | 0.692 | +0.004 | ABSTAIN | 0.688 | 0.692 | marginal |
| 22 | adb_31_satimage-2 | adbench | 1451 | OCSVM | 0.987 | 0.973 | -0.013 | ABSTAIN | 0.987 | 0.987 | marginal |
| 23 | adb_32_shuttle | adbench | 2000 | IForest | 0.999 | 0.998 | -0.001 | ABSTAIN | 0.999 | 0.999 | marginal |
| 24 | adb_33_skin | adbench | 2000 | KNN | 0.714 | 0.566 | -0.147 | FREEZE | 0.714 | 0.714 | harmful |
| 25 | adb_34_smtp | adbench | 2000 | KNN | 0.941 | 0.936 | -0.005 | ABSTAIN | 0.941 | 0.941 | marginal |
| 26 | adb_35_SpamBase | adbench | 1052 | COPOD | 0.687 | 0.616 | -0.071 | ABSTAIN | 0.687 | 0.687 | harmful |
| 27 | adb_36_speech | adbench | 922 | COPOD | 0.603 | 0.590 | -0.014 | ABSTAIN | 0.603 | 0.603 | marginal |
| 28 | adb_37_Stamps | adbench | 85 | COPOD | 0.929 | 0.921 | -0.007 | ABSTAIN | 0.929 | 0.929 | marginal |
| 29 | adb_38_thyroid | adbench | 943 | IForest | 0.975 | 0.966 | -0.010 | ABSTAIN | 0.975 | 0.975 | marginal |
| 30 | adb_39_vertebral | adbench | 60 | OCSVM | 0.348 | 0.251 | -0.097 | FREEZE | 0.348 | 0.348 | harmful |
| 31 | adb_3_backdoor | adbench | 2000 | ECOD | 0.821 | 0.784 | -0.036 | ABSTAIN | 0.821 | 0.821 | harmful |
| 32 | adb_40_vowels | adbench | 364 | KNN | 0.983 | 0.861 | -0.122 | ABSTAIN | 0.983 | 0.983 | harmful |
| 33 | adb_41_Waveform | adbench | 861 | COPOD | 0.816 | 0.818 | +0.002 | ABSTAIN | 0.816 | 0.818 | marginal |
| 34 | adb_44_Wilt | adbench | 1205 | LOF | 0.581 | 0.385 | -0.196 | ABSTAIN | 0.581 | 0.581 | harmful |
| 35 | adb_46_WPBC | adbench | 50 | COPOD | 0.509 | 0.546 | +0.037 | ABSTAIN | 0.509 | 0.546 | helpful |
| 36 | adb_47_yeast | adbench | 371 | ECOD | 0.441 | 0.405 | -0.036 | ABSTAIN | 0.441 | 0.441 | harmful |
| 37 | adb_4_breastw | adbench | 171 | COPOD | 0.990 | 0.966 | -0.024 | ABSTAIN | 0.990 | 0.990 | harmful |
| 38 | adb_5_campaign | adbench | 2000 | COPOD | 0.791 | 0.776 | -0.015 | ABSTAIN | 0.791 | 0.791 | marginal |
| 39 | adb_6_cardio | adbench | 458 | ECOD | 0.933 | 0.900 | -0.032 | ABSTAIN | 0.933 | 0.933 | harmful |
| 40 | adb_7_Cardiotocography | adbench | 529 | ECOD | 0.795 | 0.690 | -0.105 | ABSTAIN | 0.795 | 0.795 | harmful |
| 41 | adb_8_celeba | adbench | 2000 | OCSVM | 0.836 | 0.801 | -0.034 | ABSTAIN | 0.836 | 0.836 | harmful |
| 42 | adb_9_census | adbench | 2000 | COPOD | 0.679 | 0.666 | -0.013 | ABSTAIN | 0.679 | 0.679 | marginal |
| 43 | creditcard | fraud | 2000 | LOF | 0.979 | 0.973 | -0.006 | ABSTAIN | 0.979 | 0.979 | marginal |
| 44 | nlp_20news_0 | text | 773 | LOF | 0.743 | 0.678 | -0.065 | ABSTAIN | 0.743 | 0.743 | harmful |
| 45 | nlp_20news_1 | text | 629 | KNN | 0.522 | 0.467 | -0.055 | ABSTAIN | 0.522 | 0.522 | harmful |
| 46 | nlp_20news_2 | text | 625 | LOF | 0.447 | 0.492 | +0.044 | ABSTAIN | 0.447 | 0.492 | helpful |
| 47 | nlp_20news_3 | text | 154 | KNN | 0.852 | 0.883 | +0.031 | ABSTAIN | 0.852 | 0.883 | helpful |
| 48 | nlp_20news_4 | text | 415 | LOF | 0.519 | 0.610 | +0.091 | ABSTAIN | 0.519 | 0.610 | helpful |
| 49 | nlp_20news_5 | text | 383 | KNN | 0.371 | 0.495 | +0.124 | ABSTAIN | 0.371 | 0.495 | helpful |
| 50 | nlp_agnews_0 | text | 2000 | LOF | 0.637 | 0.560 | -0.077 | ABSTAIN | 0.637 | 0.637 | harmful |
| 51 | nlp_agnews_1 | text | 2000 | LOF | 0.764 | 0.644 | -0.120 | ABSTAIN | 0.764 | 0.764 | harmful |
| 52 | nlp_agnews_2 | text | 2000 | LOF | 0.756 | 0.695 | -0.061 | ABSTAIN | 0.756 | 0.756 | harmful |
| 53 | nlp_agnews_3 | text | 2000 | LOF | 0.706 | 0.624 | -0.083 | ABSTAIN | 0.706 | 0.706 | harmful |
| 54 | nlp_amazon | text | 2000 | KNN | 0.620 | 0.586 | -0.033 | ABSTAIN | 0.620 | 0.620 | harmful |
| 55 | nlp_imdb | text | 2000 | COPOD | 0.514 | 0.483 | -0.030 | ABSTAIN | 0.514 | 0.514 | harmful |
| 56 | nlp_yelp | text | 2000 | KNN | 0.687 | 0.653 | -0.034 | ABSTAIN | 0.687 | 0.687 | harmful |
| 57 | online_shoppers | tabular | 2000 | KNN | 0.696 | 0.676 | -0.020 | ABSTAIN | 0.696 | 0.696 | harmful |
| 58 | unsw_dos | cyber | 2000 | ECOD | 0.643 | 0.532 | -0.111 | FREEZE | 0.643 | 0.643 | harmful |
| 59 | unsw_exploits | cyber | 2000 | KNN | 0.550 | 0.521 | -0.029 | FREEZE | 0.550 | 0.550 | harmful |
| 60 | unsw_full | cyber | 2000 | LOF | 0.547 | 0.335 | -0.212 | ABSTAIN | 0.547 | 0.547 | harmful |
| 61 | unsw_fuzzers | cyber | 2000 | LOF | 0.566 | 0.463 | -0.103 | ABSTAIN | 0.566 | 0.566 | harmful |
| 62 | unsw_reconnaissance | cyber | 2000 | LOF | 0.498 | 0.321 | -0.177 | FREEZE | 0.498 | 0.498 | harmful |

## Skipped Archives

| Entry | Reason |
|-------|--------|
| cv_* (image_ood, 61 files) | Image OOD anomaly detection — not the covariate-shift adaptation scenario. Many files have <30 test samples (skew/kurtosis unreliable). f0/fa distinction collapses when only one detector consistently dominates across all class-vs-rest splits. |
