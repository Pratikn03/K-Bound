# ELARA + KGA Merge Plan — Single Source of Truth for K-Bound Certificates

**Status:** safe core implemented (2026-06-20). Remaining steps listed at the end.
**Mandate:** unify the duplicated K-Bound certificate logic *without* breaking the
working system or changing any numerical behavior. The `kga/` package is
load-bearing (the paper's validators and the API import it); public APIs stay
identical.

---

## 1. The duplication

The K-Bound certificate / evidence / decision logic exists in **three** places:

| # | Location | Role | Verdict |
|---|----------|------|---------|
| 1 | `kga/` (top-level: `certificate.py`, `evidence.py`, `kga.py`, `policy.py`) | Clean, typed, `py.typed`, torch-free productized package. Its module docstring states it was *"vendored_from_elara/certification/switching_certificate.py"* and that its `empirical_bernstein` is *"identical to"* the elara/kbound_pkg formulas. | **CANONICAL — single source of truth.** |
| 2 | `src/elara/certification/switching_certificate.py` | The research origin (Phase 2.G). Carries the empirical-Bernstein LCB **plus** elara-specific machinery (paired bootstrap, `SwitchingCertificate` gate/scenario dataclass, `fired_subset_certificate`). | **Keep file; delegate the duplicated math to `kga`.** |
| 3 | `docs/research/kbound/kbound_pkg/kbound/certificate.py` | A frozen third copy bundled with the paper reproduction package. | **Frozen snapshot — leave code, add a header pointer to `kga/`.** |

There is also a verbatim copy of (2) under
`docs/research/kbound/vendored_from_elara/certification/switching_certificate.py`.
That tree is, by name, a *vendored snapshot* and is treated like (3): frozen, not
edited.

---

## 2. What is a certificate duplicate vs. what is elara-specific

Only the **empirical-Bernstein lower-confidence-bound formula** is genuinely
duplicated across all three trees. Everything else in `src/elara/certification/`
is either elara-specific machinery or unrelated theory/evaluation and is **NOT a
certificate duplicate** — it stays put.

### Duplicated (the merge target)

The Maurer–Pontil (2009) empirical-Bernstein LCB:

```
E[X] >= mean - sqrt(2 * Var_hat * ln(2/alpha) / n) - 7 * R * ln(2/alpha) / (3 (n-1))
```

appears, byte-for-byte in formula, in:

- `kga/certificate.py::empirical_bernstein`            (returns a `Certificate`)
- `src/elara/certification/switching_certificate.py::empirical_bernstein_lcb`  (returns `(mean, lcb, var)`)
- `docs/research/kbound/kbound_pkg/kbound/certificate.py::empirical_bernstein_lcb` (returns scalar `lcb`)

**Numerically identical** — verified over 3000 randomized inputs (varied n,
range, variance, alpha), max abs difference `1.8e-15` (floating-point noise), and
the `n=1` degenerate case agrees (`-inf`). See §4.

### Elara-specific — NOT a certificate dup, stays in `src/elara/certification/`

- `paired_bootstrap_lcb(...)` — paired-bootstrap percentile LCB. **No kga
  equivalent** (kga deliberately ships only deterministic estimators). Stays.
- `SwitchingCertificate` dataclass — gate/scenario domain object
  (`gate_id`, `scenario_id`, `n_fired_samples`, bootstrap + EB fields).
  Different shape from kga's `Certificate(delta_hat, epsilon, method, alpha, n)`.
  Stays.
- `fired_subset_certificate(...)` — the Phase-2.G driver that maps per-sample
  prediction + gate-fire vectors to a `SwitchingCertificate`. Stays.
- `risk_dominance.py` — `estimate_risk_dominance`, `RiskDominanceTerms`.
  Unrelated to the certificate formula. Stays.

### Elara theory / evaluation — explicitly out of scope of this merge

The audit flagged these as *not* certificate duplicates; they are elara's own
contributions and are left untouched:

- **Theory registry T1–T9** (`src/elara/theory/theorem_registry.py`).
- **Evaluation harnesses** (`src/scripts/audit_switching_certificate_t5*.py`,
  `run_phase2_certificate_audit.py`, `run_phase2_b_cert_1_v2.py`).
- **family_b** experiment code.

These are domain/experiment logic, not a re-derivation of the certificate, and
merging them into `kga` would *expand* `kga`'s surface and risk behavior change.
Out of scope.

---

## 3. The merge decision (why a thin delegation, not a re-export)

A naive "make elara re-export `kga`" is **unsafe** because the public contracts
genuinely differ:

| Aspect | elara `empirical_bernstein_lcb` | kga `empirical_bernstein` |
|---|---|---|
| Return type | `tuple(mean, lcb, var)` | `Certificate(delta_hat, epsilon, method, alpha, n)` |
| `alpha` default | `0.05` | `0.1` |
| `n == 0` | returns `(nan, nan, nan)` | **raises** `ValueError` |
| `n == 1` | returns `(mean, -inf, 0.0)` | returns `epsilon=inf` → `lower == -inf` (agrees) |
| non-finite input | silently computes (`nan`) | **raises** `ValueError` |
| `alpha == 1.0` | computes a value | **raises** `ValueError` |

elara's importers (`fired_subset_certificate`, three `audit_*` scripts) unpack the
**3-tuple** and rely on the `nan`-return path for empty/degenerate fired subsets.
Replacing that with kga's raising/`Certificate`-returning function would change
observable behavior and break callers.

**Therefore the conservative merge is a *thin delegation on the agreeing path*:**

- `src/elara/certification/switching_certificate.py::empirical_bernstein_lcb`
  keeps its exact signature, defaults, and 3-tuple return, and keeps its
  `n<2` / empty / degenerate edge-case behavior **unchanged**.
- On the normal path (`n >= 2`, finite, `alpha in (0,1)`) it now computes the LCB
  by calling `kga.certificate.empirical_bernstein(...)` and reading
  `cert.lower` / `cert.delta_hat`, instead of re-deriving the Maurer–Pontil
  arithmetic locally.
- Result: the *formula* lives in exactly **one** place (`kga`), while every
  observable elara behavior — return shape, defaults, edge cases — is byte-for-byte
  preserved. `paired_bootstrap_lcb`, `SwitchingCertificate`, and
  `fired_subset_certificate` are untouched.

This removes the duplicated arithmetic (the actual drift risk) at zero behavioral
cost. The richer surgery (porting the bootstrap / `SwitchingCertificate` into
`kga`, or collapsing the two dataclasses) is deferred — see "Remaining steps".

### `kbound_pkg/` (copy 3) and `vendored_from_elara/` decision

Both are **frozen snapshots** shipped with the paper reproduction. We do **not**
rewire them to import `kga` (that would couple the reproduction artifact to the
live package and could change a published number). Instead we add a one-line
header to `kbound_pkg/kbound/certificate.py` and a `README` note recording that
`kga/` is canonical and this copy is a frozen snapshot — to stop future drift and
tell a future reader which file to edit.

---

## 4. Equivalence check (evidence)

Cross-check harness (sandbox `python3`, numpy/scipy, torch-free):

- 3000 randomized trials: `n ∈ [2, 400)`, range `[a,b] ⊂ [-1,1]`,
  `alpha ∈ {0.01,0.05,0.1,0.2,0.5}`, `benefit_range ∈ {2.0, None}`.
- Compared elara `lcb` vs kga `cert.lower`, and elara `mean` vs kga `delta_hat`.
- **Mismatches (> 1e-12): 0. Max abs diff: 1.78e-15.**
- `n=1`: elara `lcb=-inf`, kga `lower=-inf` — agree.
- `n=0`: elara `(nan,nan,nan)`; kga raises (handled by keeping elara's guard).
- non-finite: elara `(nan,nan,nan)`; kga raises (handled by elara's guard).

**Conclusion: elara's and kga's empirical-Bernstein certificates are numerically
identical on their shared domain.** The delegation is therefore behavior-preserving
on the normal path, and elara retains its own edge-case semantics.

---

## 5. What changed (this PR)

1. `src/elara/certification/switching_certificate.py` — `empirical_bernstein_lcb`
   now delegates the core LCB to `kga.certificate.empirical_bernstein` on the
   normal path; edge cases and the 3-tuple contract unchanged. (See file diff /
   docstring note.)
2. `docs/research/kbound/kbound_pkg/kbound/certificate.py` — added a header line:
   canonical source is `kga/`; this file is a frozen snapshot.
3. (this document).

No public names removed. No defaults changed. No numbers changed.

---

## 6. Remaining steps (deferred, optional, each behind its own verification)

- **R1.** Port `paired_bootstrap_lcb` into `kga` as an *additional* deterministic-
  seeded estimator (new name, e.g. `paired_bootstrap`), then have elara delegate
  it too. Requires its own equivalence test (seed/rng parity).
- **R2.** Decide whether `SwitchingCertificate` (gate/scenario) should become a
  thin adapter over `kga.Certificate`, or stay an independent domain object.
  Currently independent; collapsing risks changing the audit scripts' field
  expectations.
- **R3.** Optionally rewire `kbound_pkg/` and `vendored_from_elara/` to import
  `kga` *iff* the reproduction package is re-released; today they remain frozen.
- **R4.** Consider deleting `docs/research/kbound/vendored_from_elara/` once the
  paper repro is confirmed to no longer reference it directly (out of scope here;
  needs a usage sweep of the repro notebooks).
