# `kbound_pkg/` — FROZEN K-Bound reproduction snapshot

**Do not edit this package for product work.**

This directory is a **byte-stable** self-contained `kbound` package used by paper
reproduction (notebooks, manifests, historical imports). It exists so published
numbers keep reproducing.

## Canonical source of truth

| Want | Edit |
|------|------|
| Certificate / evidence / decide / routing | **`kga/` at repo root** |
| This frozen snapshot | Only when deliberately re-releasing the reproduction wheel |

- `kbound/certificate.py::empirical_bernstein_lcb` here is numerically identical
  to `kga.certificate.empirical_bernstein` (Maurer–Pontil empirical-Bernstein LCB).
- Layout reference: [`docs/REPO_LAYOUT.md`](../../../REPO_LAYOUT.md).

## Rule to stop drift

If the certificate math needs to change, **edit `kga/certificate.py`**, then
re-vendor into this tree only as an explicit release step.
