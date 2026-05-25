# FINAL FORBIDDEN CLAIMS CHECKLIST

This checklist is used to scan LaTeX source files and compiled PDFs for forbidden terms or unsupported claims.

## Forbidden Concepts & Phrases
- `[ ]` **Universality Claims:** Do not call ELARA a "universal anomaly detector" or assert "universal performance."
- `[ ]` **SOTA/Leaderboard Claims:** Do not call ELARA "state-of-the-art" (SOTA) or claim competitive superiority over the best vision/point-cloud AD leaderboards.
- `[ ]` **Production/Deployment Readiness:** Do not describe ELARA as "deployment-ready," "production-ready," or a "real-world deployment safety framework."
- `[ ]` **Family-D Confirmation:** Do not claim "Family D confirms ELARA," "successful held-out transfer," or "independent confirmatory validation."
- `[ ]` **Stale p-values or Buggy statistics:** Double-check that no buggy $p \approx 0.0000$ or z-score of $-7163$ or $-5272$ is cited as valid evidence.
- `[ ]` **Family-A baselines:** Verify that Family A is qualified as an audited static-reference evaluation against fixed `static_attention` only, and does not claim strongest-baseline superiority.
- `[ ]` **Modality Independence:** Verify that LOCO-AD (A-POWERED-3) and VisA (A-POWERED-4) are explicitly qualified by `derived_view_proxy` limitations and not claimed as independent modalities.

## Scanning Terms (For pdfgrep / grep)
- `universal` (unless qualified as method name / scope boundary)
- `state-of-the-art` / `SOTA` (unless in related work citing other papers)
- `deployment-ready` / `production-ready` / `safety certificate`
- `confirms ELARA` / `successful validation`
- `0.919` / `0.0003` (stale p-values/metrics)
- `0.0000` (prohibited DeLong p-values, unless cited inside error explanation)

Checked by Phase 3 Integration Runner on 2026-05-25.
