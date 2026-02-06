# Fusion Schema

This schema defines the long-format input for Phase 2 attention fusion.

## Required columns

- sample_id: stable identifier for alignment
- domain: domain key (fraud, cyber, behavior, nlp, vision)
- score: anomaly score in [0, 1]

## Optional columns

- label: binary label for supervised fusion
- confidence: calibrated confidence in [0, 1]
- timestamp: optional timestamp for time-based splits
- embedding_*: optional embedding vector columns (embedding_0, embedding_1, ...)
- metadata_json: optional JSON metadata

## Example (CSV)

sample_id,domain,score,confidence,label,embedding_0,embedding_1,timestamp
001,fraud,0.82,0.91,1,0.12,0.08,2026-02-01T10:00:00Z
001,cyber,0.40,0.55,1,0.33,0.19,2026-02-01T10:00:00Z
002,fraud,0.05,0.30,0,0.04,0.02,2026-02-01T11:00:00Z

## Alignment rules

- Use sample_id + timestamp to align across domains when possible.
- Do not align across train/val/test boundaries.
- Missing domains are allowed; mark them with masks during fusion training.
