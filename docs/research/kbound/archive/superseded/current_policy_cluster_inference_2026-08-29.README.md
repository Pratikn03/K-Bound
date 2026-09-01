# Historical current-policy inference snapshot

This file preserves the 2026-08-29 inference JSON byte for byte. It is
superseded only for current-code provenance by
`experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json`.

The 2026-08-31 verification replayed the same 15 already-opened compact
CIFAR-10-C files with the same NumPy version, bootstrap seeds, 20,000
replicates, and candidate order. All 197 numeric JSON leaves, all 6,480
per-cell radii, and all 6,480 actions were identical. The exact historical
policy/certificate Git bytes were checked against this snapshot's hashes
before the per-cell comparison. No experiment, target access, or
scientific result was added. All retrospective Holm gates remain false.

Historical SHA-256: `5b1887fc7848ca0a23940806643416c231a04abb62bd141eb318a5e43a36fbdb`.

This archival copy is not the current source-binding authority and is
not a clean-commit release seal. The current replay's file hashes bind
the working implementation; they do not assert that HEAD contains its
uncommitted changes.
