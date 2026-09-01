# `kbound_pkg/` — FROZEN K-Bound reproduction snapshot

**Do not edit this package for product work.**

This directory retains the historical self-contained `kbound` implementation used by
paper reproduction (notebooks, manifests, historical imports). Its Python sources
are preserved so archived numerical behavior can be inspected and reproduced.

## Deployment exclusion — reviewed 2026-08-31

This is **not the maintained deployment API**. Use the root `kga` package instead;
the root installation excludes this directory. The legacy `KGA.decide` uses an
entropy/KL heuristic, not a calibrated benefit interval. Its action strings do not
certify the manuscript's strict benefit claims. Legacy numerical paths also do not
uniformly reject missing or nonfinite evidence.

The legacy `KBoundOptimizer` does not guarantee retention of frozen parameters:
nonzero `abstain_scale` permits updates, and calling the base optimizer with zero
gradients on FREEZE can still change parameters through momentum or weight decay.
Do not use this optimizer to implement the paper's ABSTAIN/retain-frozen contract.
These limitations are preserved as historical code, not endorsed by the current
publication package. This audit note changes documentation only.

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
