# Paper Draft v1

The manuscript is now maintained as LaTeX:

- Source: `docs/research/PAPER_DRAFT_v1.tex`
- Compiled PDF: `output/pdf/PAPER_DRAFT_v1.pdf`

Current title:

**VERA: Reliability-Gated Multimodal Anomaly Fusion under Domain Stress**

Draft status: VERA-centered multimodal anomaly-fusion manuscript. The code now includes a naturally paired MVTec 3D-AD RGB/depth benchmark path and a local bagel-subset smoke run; quantitative claims remain limited to generated local artifacts.

Current results status:

- Real-domain score-level benchmark: fraud, cyber, behavior, and NLP domains.
- Naturally paired benchmark path: MVTec 3D-AD RGB + depth/XYZ fusion schema, with 78 paired bagel-subset observations in the local smoke artifact.
- Clean evaluation: five seeds with ROC-AUC, PR-AUC, F1, ECE, and Brier score.
- Test-time adaptation baselines: score-level Tent and pseudo-label TTT comparators.
- Added multi-seed ablation support: missing domains, score drift, adversarial score attacks, calibration, and counterfactual domain attribution.
- Added mechanism-isolation hooks: reliability-gate tau sweep and reliability-component ablations for ECE, KS, sharpness, and gate removal.
- Added harder-benchmark prep hook: `--scorer-train-fraction 0.05` to `0.10` intentionally weakens domain scorers before fusion so method comparisons do not saturate.
- Added system-level VERA framing, architecture figure, threats to validity, reproducibility, and stronger discussion.
- Important limitation: the real-domain benchmark aligns records by label, and the MVTec run is currently a small smoke benchmark rather than full paired-data evidence.
