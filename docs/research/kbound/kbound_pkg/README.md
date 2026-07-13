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

## Rule to stop drift

If the certificate math needs to change, **edit `kga/certificate.py`**, not this
file. Re-vendor here only when the reproduction package is deliberately
re-released and its equivalence tests pass.
