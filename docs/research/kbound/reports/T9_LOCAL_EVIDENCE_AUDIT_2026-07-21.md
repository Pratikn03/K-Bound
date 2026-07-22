# T9 vs local K-Bound evidence audit — 2026-07-21

## Scope

Compared publication-shaped files under
`/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results` with the local
`experiments/kbound/results` tree. AppleDouble `._*` metadata files were excluded from evidence.
The audit searched JSON/CSV/Markdown/TeX/PNG/PDF artifacts, then checked candidate findings against
the short manuscript, claim ledger, and generated result manifest.

## Inventory and reconciliation

- T9 candidate evidence files: 1,437 before excluding AppleDouble metadata.
- Local candidate evidence files: 833.
- T9-only real relative paths after excluding `._*`: **0**. All substantive T9 result paths are
  already represented locally.
- The iWildCam streaming JSON is byte-identical on T9 and locally (SHA-256
  `3d1df57a6212e32087b45f3f6fd08507dd0260b895b4eaf98a743ac6da01e152`).
- The local CIFAR-10-C locked aggregate intentionally differs from T9. T9 retains the older SAR
  seed-0 values; local contains the July 19 reconciliation with SAR radius CV 0.390 and the disclosed
  seed-0 instability. **Do not restore the T9 aggregate over the local one.**

## High-value finding that is not integrated

### Controlled multimodal Protocol D33 — safe to add with narrow scope

Authoritative artifact: `experiments/kbound/results/controlled_multimodal_d33/results.json`.
Claim ledger: `KB-CLAIM-027`, status `supported`.

- Pre-registered controlled two-view MNIST experiment.
- 130 conditions: 13 corruption severities x 10 batches.
- KGA mean accuracy 0.8568; always-fuse 0.5832; always-single-A 0.8536; oracle 0.8573.
- Decisions: 9 ADAPT, 119 FREEZE, 2 ABSTAIN.
- Observed false-adapt: 0.
- Paired bootstrap probability of improvement over each fixed policy: 1.0 in the saved result.

This result is absent from `kbound_short.tex` and from
`paper/generated/kbound_result_manifest.json`. It can improve the paper as a controlled
mechanism-confirmation experiment showing that KGA routes correctly when multimodal fusion changes
from helpful to harmful and that change is detectable.

Required wording boundary: controlled/injected modality corruption, not a natural multimodal
benchmark and not evidence of universal accuracy improvement. The +0.0032 gain over always-single is
small and should be reported with the full uncertainty procedure, not marketed by magnitude.

## Valuable existing diagnostic that was incorrectly listed as missing

### Full iWildCam streaming collapse diagnostic

Artifact: `experiments/kbound/results/iwildcam_streaming_pilot/pilot_test_native_bs16.json`.

- Native test stream: 35,370 images; 35,360 used; 2,210 batches of 16.
- Frozen cumulative macro-F1: 0.2554.
- Continual TENT cumulative macro-F1: 0.0219.
- Difference: -0.2335; bootstrap 95% interval [-0.2537, -0.2212].
- The run records prediction collapse to one/few classes during the stream.

**Status (reconciled 2026-07-21):** `KBOUND_REMAINING_TODOS.md` has since been corrected and now
records the streaming artifact as "recovered / diagnostic only"; the earlier "no saved streaming
artifact exists" wording is no longer present. The short paper
already applies the correct scientific boundary: the betting increments use labels, so this is a
label-informed offline failure diagnostic, not a label-free KGA deployment result. It can strengthen
motivation and the case for an abstaining safety controller, but cannot support the streaming KGA
claim.

## Findings that must not be promoted

- FMoW Protocol L: not cleared; held-out false-adapt 0.375 and KGA does not beat both.
- Poverty Protocol L: preregistered dev-screen stop; harm AUC 0.637 below the 0.65 gate; held-out
  val/test correctly not run.
- ImageNet-R hard-dataset loop: only 1/4 seed splits wins; not replicated.
- RxRx1 hard-dataset loop: only model seed 0 wins; model seeds 1/2 and pooled analysis do not.
- Camelyon pooled Protocol G beats-both: withdrawn because the pooled split is invalid.
- Office-Home/iWildCam historical beats-both summaries: retain current OOF no-harm wording unless a
  separate audit proves that the exact result uses the authoritative held-out calibration protocol.
- 3D-ADAM: exploratory 23-category result; insufficient evidence for a natural multimodal headline.

## Recommended paper-value actions

1. **(done)** Add one compact D33 mechanism-confirmation paragraph/table to the short-paper appendix and add it
   to the generated authoritative manifest and empirical claim matrix. — D33 appears in
   `kbound_short_appendix.tex` (App.~D33 table), `paper/generated/kbound_result_manifest.json`, and
   `paper/generated/empirical_audit/claim_matrix.md`; pre-registration artifact now referenced.
2. **(done)** Replace the stale streaming TODO with “artifact recovered; label-informed diagnostic integrated.”
3. Add a compact iWildCam collapse sentence/figure only if page budget permits, preserving the
   explicit non-KGA diagnostic disclaimer.
4. Preserve FMoW and Poverty as preregistered negative/stopping evidence. This improves credibility
   and demonstrates that the protocol does not promote failed searches.
5. Add a regression check that every `supported` empirical claim in `claim_ledger.json` is either
   present in the authoritative manifest or explicitly marked “long-paper only.”

## Net effect

T9 does not contain an unreported natural-shift headline win. Its main contribution is recovery of
one useful full-stream diagnostic. The strongest genuinely under-integrated positive result is D33,
which should raise experimental breadth and mechanism validation, but not the natural-shift claim
level.
