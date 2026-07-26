# K-Bound Short Paper Empirical Consistency Audit

> **Revised 2026-07-26.** Three rows below carried stale values (ImageNet-C SAR was the superseded
> single-seed triple; ImageNet-R and PACS seed counts were out of date) and three evidence scopes
> were overstated. Canonical: `SUBMISSION_LEDGER.md §3`.

## Canonical source

`paper/generated/kbound_result_manifest.json` is the authoritative promoted-number index. It records
policy order, metric definitions, seed counts, source artifacts, quantile provenance, verdicts, and
caveats. Where it and `SUBMISSION_LEDGER.md` disagree, the ledger wins.

## Promoted evidence

Regret triples are KGA / always-adapt / always-freeze.

| Track | Regret | Evidence scope | Final claim (2026-07-26) |
|---|---:|---|---|
| CIFAR-10-C Tent | 0.0016 / 0.0079 / 0.1241 | mixed head-to-head 5-seed aggregate, 432 cells/seed | **CI beats-both**; `FA_u = 0`. Unaffected by the radius fix. |
| CIFAR-10-C EATA | 0.0013 / 0.0033 / 0.1314 | same | **CI beats-both**; `FA_u = 0`. Adapt-gap CI does **not** exclude zero when clustered by corruption family. |
| ImageNet-C SAR | 0.0289 / 0.0529 / 0.0319 | 27 cells/seed x 5 seeds = 135, LOO radius | **point-estimate no-harm vs always-freeze only.** `FA_u = 1/135`. Seed-averaged freeze-gap CI [-0.0085, +0.0038] includes zero. Was 0.0264/0.0529/0.0319 under the in-pool radius. |
| Camelyon17 OOD | 0.0000 / 0.0000 / 0.1381 | n = 18 | **sealed but not recomputable from release.** The triple exists only at `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29`; the `FA_u = 0` is recorded nowhere. |
| iWildCam H v2 | 0.0041 / 0.1028 / 0.0041 | OOF lock, n = 72 | exact tie with freeze; no-harm. **1 ADAPT decision — guarantee untested.** Source record file absent. |
| Office-Home M v2 | 0.0157 / 0.0468 / 0.0158 | OOF lock, n = 35 | no-harm; point edge only. **Both source record files absent; runner source unreadable.** |
| RxRx1 J | 0.0000 / 0.2531 / 0.0000 | locked test, n = 60 | tie with freeze; no-harm. **0 ADAPT decisions — guarantee untested.** |
| Three-source OOF | 0.0059 / 0.0632 / 0.0342 | constructed n = 143 stream | CI beats-both; **constructed routing mixture, not transfer.** |

## Diagnostic, negative or withheld evidence

- **CIFAR-10.1**: `FA_u = 0.167`, `FA_c = 0.444`; transfer bar fails. Pre-declared as a likely
  negative; came out worse than declared.
- **ImageNet-R**: **4 of 4** planned seeds, 10 backbones. No CI-robust beats-both on any backbone.
  The panel mean 0.0112 hides that **KGA is worse than always-adapt on 7 of 10 backbones**, and
  4 of 10 backbones have a 0% harmful base rate (so there is nothing for a certificate to prevent).
  Report min / median / max and the per-backbone harmful base rate.
- **PACS**: **3 of 3** planned seeds, 4 LODO targets. Null diagnostic. Its entire adapt evidence is
  12 ADAPT decisions from one domain-seed cell (art_painting seed 1, `FA_u = 0.1111 > alpha`);
  art_painting seed 2 abstained on all 18 cells. Pooled `FA_u = 2/216`, CP95 upper 0.03305.
  Cannot be re-scored — the released per-cell dumps carry no `b_hat`, no epsilon and no decision.
- **CIFAR-10-C SAR**: withheld because the current raw seed 0 does not replay the archived
  aggregate — and seed 0 is also the only seed on a different Python, torch and commit, so the
  cause is confounded (`CIFAR10C_SAR_QUARANTINE.md`).

## Calibration provenance

- **Declared rule (2026-07-26): exact split-conformal rank quantile, leave-one-out-of-pool.** One
  rule, stated once, at `SUBMISSION_LEDGER.md §1a`.
- Superseded variant 1: interpolated empirical quantile `np.quantile(|Bhat - B|, 1-alpha)`.
- Superseded variant 2: **in-pool** rank quantile — cell *i*'s own residual was in cell *i*'s
  radius. This was live on five shipped scripts and seven `decide_kga` forks until 2026-07-26; it
  made epsilon a function of the test labels the `FA_u` guarantee attaches to.
- Stress grids: leave-one-condition-out cross-fitted empirical residual calibration; approximate
  nominal empirical coverage, not exact split-conformal validity.
- **Structural caveat:** under in-pool rank calibration `FA_u <= (N-k)/N` is an arithmetic identity
  — exactly 0 at n <= 9. Camelyon17 Table VIII (n = 9/seed), RxRx1 and ImageNet-R (n = 12) are in
  that degenerate range, so their `FA_u = 0` is forced, not measured.

## Baseline fidelity

- KGA is the paper's method.
- POEM-style and AETTA-style rows are protocol-matched ports, not official implementations.
- Higher observed false-adapt in these ports is described as consistent with the lack of an explicit marginal certificate, not caused solely by it.

## Consistency checks

- ImageNet-C 27-cell values are not mixed with the superseded 36-cell configuration.
- Natural-shift point wins are not promoted as CI-robust beats-both.
- Always-adapt/freeze are treated as fully decisive policies; adapt rate and decision coverage are not conflated.
- `FA_u` and `FA_c` are separately named.
- No blank camera table is used as evidence.
