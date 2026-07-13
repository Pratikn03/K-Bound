# K-Bound External Review and Reproduction Packet

This packet is for a theory reviewer, empirical auditor, or independent
reproducer. It covers only claims retained in the 21-page short draft.

## Review Entry Points

| Question | Artifact |
|---|---|
| What is claimed? | `kbound_short_final_draft.pdf` |
| Which source generated it? | `kbound_short.tex`, `kbound_short_appendix.tex` |
| Where do repeated numbers come from? | `paper/generated/kbound_result_manifest.json` |
| What is theorem-checked? | `formal/KBound/TheoremMap.lean`, `formal/README.md` |
| How do claims map to evidence? | `KBOUND_SHORT_CLAIM_MANIFEST.md` |
| What remains open? | `KBOUND_SHORT_REMAINING_WORK.md` |

## Theory Checklist

- Confirm `Delta = R_T(f0) - R_T(fa)` everywhere.
- Confirm `eta_a(x) = P_T(fa(X)=Y | X=x)` and the disagreement algebra.
- Check that the interior theorem assumes `beta > 0` and `|M| < beta`.
- Check that the equality boundary is handled separately.
- Check that the headline concerns uniformly supportable strict commitments,
  not unconstrained “sign identifiability.”
- Check that `beta=0` is described as the strongest zero-drift assumption.
- Check that `epsilon` is not identified with `beta`, and `Delta_hat` is not
  identified with `M`.
- Check that the false-adapt theorem assumes interval coverage and controls
  marginal `FA_u`, not conditional `FA_c`.
- Check that multiclass reasoning uses `p_a-p_0`, not `p_0=1-p_a`.
- Check that risk alignment is an external structural assumption rather than a
  fact certified by fitting the benefit regressor.

## Empirical Checklist

- Compare every headline number with the canonical result manifest.
- Keep the authoritative ImageNet-C configuration at 27 cells and seed 0.
- Treat CIFAR stress calibration as leave-one-condition-out cross-fitted
  empirical residual calibration, not exact split conformal or jackknife+.
- Treat POEM/AETTA rows as protocol-matched style ports, not official results.
- Treat Office-Home, iWildCam, Camelyon17, and RxRx1 as no-harm evidence.
- Treat the three-source stream as a researcher-constructed aggregate.
- Treat CIFAR-10.1, ImageNet-R, and PACS as diagnostic or incomplete.
- Do not infer structural non-identifiability from empirical abstention alone.

## Reproduction Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[research,test]"
make verify-fast
make paper
```

Optional kernel verification:

```bash
make formal
```

The expected fast-suite result is 221 collected tests with two intentional
skips. The dashboard snapshot must rebuild byte-for-byte from the same manifest.

## Physical-Study Sign-Off

The physical result table is pending. It must remain pending until
`publication_gate.json` reports `passed: true` and the following are verified:

- exactly the locked S01-S10 physical inventory;
- unique clip hashes and physical-capture provenance;
- source-model balanced accuracy and macro-F1 at least 0.80;
- development and conformal seals created before held-out access;
- Phone A held-out replay and Phone B replication replay;
- all eight anti-leakage checks;
- no target labels available to live decisions.

Browser preview, synthetic clips, pilot sessions, and reconstructed logs cannot
satisfy this gate.

## Reviewer Sign-Off

Record the reviewed Git commit, manifest SHA-256, PDF SHA-256, test result, Lean
result, unresolved concern, and reviewer identity in the external review record.
Absence of a concern is not independent validation unless an independent person
actually performs and signs the review.
