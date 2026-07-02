# K-Bound 85+ Readiness Report

Generated: 2026-07-01T18:12Z

**Score: 43 / 100**

**Verdict: not yet 85+ — complete blockers below.**

## Checklist
- PASS: formal_audit --strict-100
- PASS: reproduce_submission.sh (cached <2h)
- WARN: smoke incomplete — see /tmp/smoke_report.txt
- WARN: full final-all not completed
- WARN: edge tests need review
- WARN: physical R2 pending — run run_edge_source_gate.sh then S03-S10
- WARN: RxRx1 missing — bash prepare_rxrx1_data.sh

## Next commands
```bash
KB_SMOKE_SEEDS="0 1" bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/run_smoke_showcase.sh
bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/edge/scripts/run_edge_source_gate.sh
bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh
bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/prepare_rxrx1_data.sh
KB_SEEDS="0 1 2 3 4" bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/run_final_showcase.sh --device mps --seeds "0 1 2 3 4"
```
