# `kbound_pkg/` — frozen K-Bound reproduction snapshot

**This package is a FROZEN SNAPSHOT.** It is the self-contained `kbound` package
that ships with the K-Bound paper reproduction (notebooks, manifests, results).
It is kept byte-stable so published numbers reproduce.

## Canonical source of truth

The maintained, productized implementation of the K-Bound certificate / evidence
/ decision logic is the **top-level `kga/` package** (`kga/certificate.py`,
`kga/evidence.py`, `kga/kga.py`, `kga/policy.py`).

- `kbound/certificate.py::empirical_bernstein_lcb` here is **numerically
  identical** to `kga.certificate.empirical_bernstein` (Maurer–Pontil 2009
  empirical-Bernstein LCB). Verified over thousands of randomized inputs
  (max abs diff ~1e-15).
- The research origin, `src/elara/certification/switching_certificate.py`,
  delegates its copy of that formula to `kga` (single source of truth).

## Rule to stop drift

If the certificate math needs to change, **edit `kga/certificate.py`**, not this
file. Re-vendor here only when the reproduction package is deliberately
re-released. See `docs/research/kbound/ELARA_KGA_MERGE_PLAN.md` for the full merge
plan and the equivalence evidence.
