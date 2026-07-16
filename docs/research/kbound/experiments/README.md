# Nested experiment configs (not the result tree)

**Canonical experiment runners + results live at repo-root:**

[`experiments/kbound/`](../../../../experiments/kbound/)

This folder only keeps lightweight **run_config.yaml** stubs and a small set of
locked JSON copies under `kbound/results/` used by older checksum / fold-in
scripts. Do **not** download datasets here and do **not** write new training
outputs here.

| Want | Use |
|------|-----|
| Train / multi-seed / WILDS | `experiments/kbound/` |
| Promoted JSON evidence | `experiments/kbound/results/` |
| These run_config stubs | this directory |
