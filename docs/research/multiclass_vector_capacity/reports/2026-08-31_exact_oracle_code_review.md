# Exact-oracle code review

Date: 2026-08-31

Disposition: the identified gate/reporting defects were reproduced, corrected by the implementation owner, and independently rechecked. No unresolved blocking defect was found in this bounded review of exact finite-instance discovery and certificate validation. This is not a theorem proof, exhaustive verification of all finite systems, novelty clearance, or promotion into K-Bound.

## Scope

Reviewed the complete implementations of `exact_oracle.py`, `certificate_check.py`, `finite_cases.py`, and `run_gate.py`, plus `tests/test_exact_oracle.py`, under `docs/research/multiclass_vector_capacity`. Review probes used only Python standard-library exact arithmetic and in-memory mutations. No natural datasets, existing K-Bound source files, formal proof files, or Git state were changed by this review.

The checker is independent of the discovery solver: it does not import the solver's vertex search, rank calculation, or action rule. It reconstructs normalization and the mass-weighted contrast, checks primal feasibility, checks both dual inequalities, and requires exact primal-dual objective equality. Thus the interval check does not rely on the solver's vertex count or reported rank diagnostics.

## Findings resolved and independently rechecked

| Finding and original reproduction | Correction and fresh result |
| --- | --- |
| T5 summary was hard-coded. Replacing its candidate costs with `(1/2,0,0)` produced a checked interval `[3/10,7/10]` while the report still claimed `[1/5,3/5]` and passed. | Report values now come from checked certificates, and the intended named-example invariant is explicitly enforced. The same mutation raises `GateFailure`. |
| T2 summary was hard-coded. Changing its observable value to `1/2` produced interval `[0,1/2]`, not point identification, while the report still described `{e3}` and benefit zero. | The gate checks the declared boundary witness and derives the reported interval and point-identification field from its certificate. The same mutation raises `GateFailure`. |
| Rare worlds were checked only for benefit, not feasibility. Adding the restriction `eta_r0=1/2` left the emitted worlds at `5/16` and `3/16`, outside the fiber, but the report passed. | Every rare world must now be nonnegative and satisfy all equalities, including simplex normalization. The same mutation raises `GateFailure`. |
| Scientific `assert` checks disappeared under optimized Python. A constant `1/2` T5 contrast passed under `optimize=1` while the report claimed a nonzero null-direction effect. | Explicit `GateFailure` checks replace scientific assertions. Both the in-memory optimized reproduction and an actual `python -O` gate run were checked. The corrupted witness is rejected. |
| The independent-checker CLI accepted an empty certificate list and returned successfully with `checked=0`. | `check_batch` rejects an empty list. The original CLI-path reproduction now raises `InvalidCertificate`. |
| A binary candidate-versus-frozen ADAPT result could be mistaken for global multi-adapter selection. For constant frozen costs `3/4`, candidate costs `1/2` and `0`, both binary comparisons favor adaptation, but candidate zero is not the globally best candidate. | Certificates and the checker enforce `comparison_scope=fixed_candidate_vs_frozen_only` and `deployment_selection=NOT_IMPLEMENTED`. Attempts to promote either field are rejected. |
| Truthy non-string or whitespace names could be emitted by the producer and rejected only later by the checker; NaN/infinite budgets could bypass the subset-budget comparison. | Nonblank string names and positive integer enumeration budgets are required. Fresh malformed-name and NaN/infinity/Boolean/zero/negative/string budget probes are rejected. |
| The gate classified every `ProtocolFailure` on an expected-inconsistent case as an emptiness rejection. | A dedicated `EmptyFiber` exception now separates infeasibility from other failures. A synthetic non-feasibility `ProtocolFailure` propagates rather than being recorded as an expected rejection. |

The catalog search also now reads at most 17 candidate rows before enforcing its 16-moment limit. An infinite iterator of repeated valid rows was independently tested and rejected with `SearchLimit`, rather than consumed without bound.

## Independent mathematical cases

The following six cases were derived separately from the repository's named/grid fixtures and rerun against the final reviewed source. Each feasible interval was also passed through the independent primal-dual checker.

| Case | Independently derived expected result |
| --- | --- |
| Two-stratum, non-coordinate coupled line with redundant and zero equality rows | Exactly two vertices; benefit interval `[-1/3,7/12]` |
| The same coupled line with first-stratum mass zero | Benefit interval `[-1/4,3/4]` |
| A full-rank, non-coordinate interior singleton with a negative redundant row | Conditional vector `(7/18,2/9,7/18)`; benefit `13/36` |
| A non-singleton simplex-face fiber | Benefit identically `1/4`, despite a surviving unrestricted affine contrast direction |
| A positive rational benefit `1/10^100` | Strict ADAPT, without rounding the endpoint to zero |
| An affine-consistent system requiring `eta_1+eta_2=2` in a three-class simplex | `EmptyFiber`, not an interval or an abstention certificate |

For reproducibility, the coupled-line example has masses `(1/3,2/3)`, frozen cost rows `((1,0,0),(0,1,1/2))`, and candidate rows `((0,1/2,0),(1/4,0,0))`. Constraints impose

\[
\eta_1=(t,1-t,0),\qquad\eta_2=(1-2t,t,t),\qquad0\leq t\leq1/2.
\]

The explicit equality rows are `(0,0,1,0,0,0)`, `(-1,0,0,0,1,0)`, `(-1,0,0,0,0,1)`, `(2,0,0,1,0,0)`, and the zero row, with right-hand side `(0,0,0,1,0)`. The last two rows are redundant. Its benefit is `11t/6-1/3`, giving the stated endpoints. With masses `(0,1)`, it is `2t-1/4`.

For the interior singleton, the extra rows are `(1,-1,0)`, `(1,0,-1)`, and `(-2,2,0)`, with values `(1/6,0,-1/3)`. Frozen costs `(1,0,1/2)` and candidate costs `(0,1,0)` give the stated benefit. The non-singleton boundary construction is recorded in the companion mathematical adversarial review.

Eight further malformed-certificate probes were rejected on the coupled-line certificate: a Boolean mass, a zero rational denominator, a Boolean candidate index, a tuple instead of the schema's list-valued primal point, a changed observable value, changed stratum weights, an extra dual coordinate, and an unsupported scientific-availability claim.

## Fresh verification receipt

The following checks completed successfully on the reviewed source snapshot:

- Normal Python: all 64 repository tests passed.
- Actual optimized Python, `python -B -O`: the same 64 tests passed. These are two runs of the same suite, not 128 distinct tests.
- Six independently hand-derived finite cases passed, including the expected infeasibility rejection.
- Eight independently constructed malformed-certificate mutations were rejected.
- Original report, rare-world, optimized-witness, empty-CLI, failure-classification, and scope-promotion reproductions were independently rerun after the fixes and rejected as intended.
- Both normal and optimized `run_gate.py --check` passed: 67 problems, 103 independently checked certificates, and 24 expected empty-fiber candidate rejections. The corpus contains two extremum certificates per accepted candidate interval, not new empirical experiments.

Commands used for the maintained checks, with bytecode writing disabled:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s docs/research/multiclass_vector_capacity/tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -B -O -m unittest discover -s docs/research/multiclass_vector_capacity/tests
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/research/multiclass_vector_capacity/discovery/run_gate.py --check
PYTHONDONTWRITEBYTECODE=1 python3 -B -O docs/research/multiclass_vector_capacity/discovery/run_gate.py --check
```

## Reviewed source hashes

These SHA-256 values identify this code-review snapshot. They are not a clean-commit source seal or a release-promotion receipt.

| Source under the track | SHA-256 |
| --- | --- |
| `discovery/exact_oracle.py` | `9c46d0bca7d210698240b1bf129e8cc9e7f851db4eb0a5a7c43d1479bc0f448b` |
| `discovery/certificate_check.py` | `45e163c029144afd50175ccbdc941d93286646c3d0ce74cd193f0104935a5e67` |
| `discovery/finite_cases.py` | `ed64bea0d649712a0688b063aa056fa7f53498203583c5821949780bf0952711` |
| `discovery/run_gate.py` | `7572bcacfeed052cb7123f07170da11c239bfd36df723154b5248edbd5162929` |
| `tests/test_exact_oracle.py` | `3604c0975da9fead0bab9c5db928396082385b5679394303983cfef87a1929db` |

## Limits of this disposition

The checker establishes the self-contained rational instance stated in a certificate. It does not establish scientific availability of its toy restrictions, authenticate an external experiment, validate unchecked rank/vertex diagnostics, or verify a general theorem. Individual candidate intervals do not establish the joint identified benefit region or a globally optimal deployment selection.

The review did not implement or verify passive/adaptive concentration, delayed-feedback inference, label-complexity lower bounds, multiplicity control, or the T9 controller. Lean theorem authority, statement parity, novelty review, fresh-checkout reproducibility, and prospective empirical gates remain separate requirements. The finite search limits are deliberate; passing the declared grid must not be described as exhaustive coverage of every rational system or as a natural-shift win.
