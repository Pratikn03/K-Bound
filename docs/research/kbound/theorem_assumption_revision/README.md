# Theorem and statistical-assumption revision

This revision strengthens the maintained K-Bound paper while preserving the frozen 2026-09-23 release. Its baseline is artifact commit `69548cf44bcbede86053ea3ef6572ef57b2a1615`; the revision branch is `codex/kbound-theorem-assumptions-20260923`.

The two original questions have more precise answers after this work:

| Question | Result of this revision | Still not established |
|---|---|---|
| Is every theorem correct? | All 13 named mathematical results and seven supporting claims were reviewed. No substantive contradiction was found under their stated premises. A missing general random-radius coverage-to-action result was mechanized and checked. | A one-to-one formal proof of every complete printed result and every manuscript sentence. Five named results have scoped mechanizations, five have partial mechanizations, and three have reviewed written proofs without exact formal counterparts. |
| Are every study's statistical assumptions valid? | Twenty assumptions were mapped to twenty study groups. A legacy diagnostic falsely promoting non-rejection to a guarantee was repaired; malformed inputs and empty calibration now fail appropriately. | Exchangeability, fresh independent sampling, externally justified residual/transport budgets, and valid future transfer across every deployment. These require a defensible model and study design; code or proof checks alone cannot supply them. |

## What changed

1. **Formal coverage-to-action gap closed.** `RandomRadiusCertificate.lean` permits a radius varying over `[0,infinity]` on the same probability space as the finite estimate and benefit. Infinite radius and zero-touching endpoints abstain. Each directional error and their union have probability at most the supplied coverage-failure budget. No radius/estimate independence is assumed. The coverage premise remains external.
2. **Proof premises made explicit.** Positive evaluation sample size, nonnegative calibration count, fixed predictors and scores, and fresh independent sampling are now stated next to the relevant procedures. The sampled-margin proof conditions on each positive disagreement count and then averages over counts; zero disagreement produces abstention.
3. **Legacy guarantee overstatement repaired.** The old audit had no benefit interval or external coverage argument but returned ADAPT and “guarantee applies” after quiet diagnostics. It now withholds the guarantee and recommends abstention. Missing drift measurements remain missing. The original v1 records are preserved; the separately labeled posthoc replay is not new empirical validation or recovery of original execution provenance.
4. **Current helper made more precise.** Invalid rank inputs, malformed residual vectors and mismatched group counts reject explicitly. Empty calibration produces the existing unbounded/abstaining outcome instead of crashing. Counting groups and validating a coverage-basis record no longer imply independence or authentication of the underlying evidence.
5. **Review correspondence made checkable.** The theorem register binds exact statements, proofs, formal-source references and review scope. The assumption register binds its authorities. The checker detects omitted/duplicate results, stale bytes, incorrect proof locations, missing Lean bindings and invalid assumption links. Its PASS means correspondence and freshness, not mathematical or empirical truth.

## Evidence and verification

[Read the inspected 69-page revision draft](KBound_Theory_Assumptions_Revision.pdf). Its source and PDF hash are recorded in [verification.json](verification.json). It is separate from the frozen release; no new Word or TMLR export is claimed.

- [Theorem register](theorem_register.json) and [proof review](theorem_review.md): 13 named results, seven supporting claims, explicit formal correspondence and historical exclusions.
- [Assumption register](assumption_register.json) and [statistical review](statistical_review.md): 20 assumptions and 20 study groups, including unfavorable and withdrawn findings.
- [Formal result](random_radius_formalization.md), [strict receipt](formal_random_radius_strict_receipt.json), [log](formal_random_radius_strict.log), and [source snapshot](formal_random_radius_source_snapshot.json): 162 registered declarations passed the fresh Lean build and axiom audit. The new snapshot binds 47 formal source/configuration files. The frozen 150-declaration receipt remains unchanged.
- [Independent integration review](integration_review.md): identified two checker gaps that were repaired and retested; reviewed code outside that reviewer's own Lean contribution.
- [Correspondence receipt](correspondence_receipt.json): exact-source register checks. The final verification report records the targeted test count, lint, draft compilation and PDF review.

No experiment was retrained or protected target scored. Principal empirical authorities and generated results tables remain unchanged. This is a new local research revision, not a replacement of the public frozen release or a complete rerun of its release package.

## Remaining proof work, in priority order

| Obligation | What would close it |
|---|---|
| Randomized matched-evidence abstention and the closed-band probability clause | A measure-level capstone connecting equal evidence laws, independent rule randomization, both error bounds, and the abstention probability. Current deterministic and arithmetic components do not by themselves constitute that entire theorem. |
| Exact fibre-supremum audit floor | Formalize the nonempty compatible-law fibre, common audit law, approximation to the bounded supremum, and continuity of threshold-event probabilities. Retain the restricted range for the equality with beta. |
| Full conditional episode population transfer | Assemble the actual conditional fitting/calibration/evaluation model with conformal ranks, fresh-sample Hoeffding concentration, and compound coverage. Do not replace conditional sampling premises with a count of cells. |
| Sampled-disagreement procedure | Formalize fixed-size iid sampling, conditioning on the random positive disagreement count, and the unconditional averaging/zero-count branch. |
| Specific perfect-cell/imperfect-population counterexample | Mechanize its explicit two-point probability law and exact 0.45 false-adapt probability. The printed proof already supplies this construction. |
| Paired-transport theorem and numerical certificate | Mechanize the statistical box-containment/feasible-table argument separately from validated inverse-beta quantiles and exact numerical feasibility/optimization. The current directed objective bounds are not an end-to-end implementation proof. |

These are mechanization obligations, not newly discovered counterexamples to the maintained proofs. The unrestricted historical orbit-selection claim has an actual counterexample and remains excluded; it is not a target to force through a passing audit.

## What stronger statistical claims would need

A proposed deployment must specify its prediction pair, sampling unit, target estimand, calibration and evaluation roles, and the reason its residuals should be exchangeable or transport should obey its external budget. Fresh held-out outcomes can test performance under that design and reveal failure; they cannot establish validity for all future environments.

Acceptance-conditional harm control and repeated/adaptive decisions need separate guarantees and procedures. Zero accepted updates leave conditional error undefined. The current poor held-out inclusion, simpler-rule wins, CCT retention-only finding and stopped So2Sat target remain scientifically relevant. None was changed to make this revision appear successful.

To check the registers from the repository root:

```bash
python docs/research/kbound/scripts/check_theorem_assumption_revision.py \
  --output docs/research/kbound/theorem_assumption_revision/correspondence_receipt.json
```

Use the pinned project environment for tests and Lean. The full historical-foundations gate remains excluded for its documented refuted/unestablished extensions; the strict receipt must not be relabeled full historical closure.
