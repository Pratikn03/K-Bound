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
- fusion_split: optional predefined split label for supervised fusion runs
  (for example train, validation, test)
- embedding_*: optional embedding vector columns (embedding_0, embedding_1, ...)
- metadata_json: optional JSON metadata

## Example (CSV)

sample_id,domain,score,confidence,label,fusion_split,embedding_0,embedding_1,timestamp
001,fraud,0.82,0.91,1,train,0.12,0.08,2026-02-01T10:00:00Z
001,cyber,0.40,0.55,1,train,0.33,0.19,2026-02-01T10:00:00Z
002,fraud,0.05,0.30,0,test,0.04,0.02,2026-02-01T11:00:00Z

## Alignment rules

- Use sample_id + timestamp to align across domains when possible.
- Do not align across train/validation/test boundaries.
- If fusion_split is present, every row for a sample_id must have the same
  fusion_split value, and the experiment runner should use that column instead
  of creating a random split.
- Label-aligned composite benchmarks must assign source rows to disjoint split
  pools before composite sampling. The same domain/source row key must not
  appear in more than one fusion_split.
- Missing domains are allowed; mark them with masks during fusion training.
