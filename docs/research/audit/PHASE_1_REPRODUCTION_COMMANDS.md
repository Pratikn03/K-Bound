# Phase 1 Reproduction Commands

**Branch:** `fix/elara-phase1-empirical-validity`

Every command issued during Phase 1 is logged here so the entire repair pipeline is replayable. Commands are listed in execution order, grouped by sub-stage.

---

## Phase 1.A — Canonical label / metric semantics audit

```bash
# Run the canonical label / metric semantics audit
PYTHONPATH=src .venv/bin/python src/scripts/audit_canonical_label_semantics.py \
  --output experiments/audit/canonical_label_semantics.json \
  --polarity-output experiments/audit/polarity_diagnostic_log.csv

# Unit tests
PYTHONPATH=src .venv/bin/python -m pytest tests/test_canonical_label_semantics.py -q
```

(Additional commands appended as work proceeds.)


## Phase 1.B / 1.C / 1.D — selection + statistical artifacts
```bash
PYTHONPATH=src .venv/bin/python src/scripts/emit_rga_plus_validation_frozen_selection.py
PYTHONPATH=src .venv/bin/python src/scripts/select_audited_validation_frozen_comparator.py
PYTHONPATH=src .venv/bin/python src/scripts/emit_locked_audited_statistics.py
```

## Phase 1.E — metrics manifest + macros
```bash
PYTHONPATH=src .venv/bin/python src/scripts/build_metrics_manifest.py
```

## Phase 1.G — regenerate manuscript tables
```bash
PYTHONPATH=src .venv/bin/python src/scripts/emit_milestone2_cross_benchmark.py
PYTHONPATH=src .venv/bin/python src/scripts/emit_gradient_adversarial_table.py
PYTHONPATH=src .venv/bin/python src/scripts/emit_rga_plus_ablation.py
PYTHONPATH=src .venv/bin/python src/scripts/audit_switching_certificate_t5.py
PYTHONPATH=src .venv/bin/python src/scripts/emit_switching_certificate_t5_table.py
```

## Phase 1.H — rebuild PDFs + run validators + run tests
```bash
./scripts/rebuild_paper.sh
PYTHONPATH=src .venv/bin/python src/scripts/validate_manuscript_claims.py
PYTHONPATH=src:. .venv/bin/python -m pytest tests/ -q
```
