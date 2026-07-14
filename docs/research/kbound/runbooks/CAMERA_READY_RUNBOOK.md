# Camera-Ready Empirical Runbook (review items 11–15)

Five GPU-work items, in value order. Each item states: goal, pre-registered criterion, exact
commands, and the paper edit permitted on completion. Nothing in this file is evidence; every
result must land in `experiments/kbound/results/` (or `research_lock/`) before any paper edit.
Evidence-tier vocabulary: locked / reconciled / provisional / diagnostic (paper §Evidence-Status Policy).

Repo root is assumed as `$REPO`; run everything from there. `K=docs/research/kbound`.

---

## Item 11 — Official POEM / AETTA reproduction

**Goal.** Replace the "protocol-matched ports, not official reproductions" caveat
(Table `tab:headtohead-poem-aetta`) with official-code rows.

**Already in place.** `$K/scripts/official_baselines_headtohead.py` ingests external per-condition
decisions via `--decisions poem=... aetta=...` and errors on any missing condition (no fabrication).
The vendored official AETTA repo is at `$REPO/AETTA/` (Lee et al. 2024). Official POEM:
<https://github.com/yarinbar/poem> (Bar, Shaer, Romano, NeurIPS 2024, arXiv:2408.07511) — clone,
**pin the commit hash in the output JSON**.

**Pre-registered mapping (fixed before running).**
- POEM: run the official protector on each locked stream condition's batch sequence;
  decision = `adapt` if the martingale never fires on that condition, `freeze` if it fires.
  POEM has no abstain; record `decisive_rate = 1.0`.
- AETTA: official dropout accuracy estimate for frozen and adapted model on the condition batch;
  decision = `adapt` iff `est_acc(adapted) > est_acc(frozen)`. No abstain.
- Metrics, stream, WIN/TIE/LOSE criterion: unchanged from `MIXED_BENCHMARK_PROTOCOL.md`.

**Commands.**
```bash
bash $K/runbooks/run_item11_official_baselines.sh          # AETTA (vendored) then POEM (cloned)
# produces experiments/kbound/results/official_repro_v1/{aetta,poem}_decisions.json
python3 $K/scripts/official_baselines_headtohead.py \
  --decisions poem=experiments/kbound/results/official_repro_v1/poem_decisions.json \
              aetta=experiments/kbound/results/official_repro_v1/aetta_decisions.json
```

**Paper edit on success.** Table `tab:headtohead-poem-aetta`: relabel rows "POEM (official)" /
"AETTA (official)"; delete the ports caveat from the caption, §Baselines, Limitations, and
`tab:claim-status`. On WIN reversal: report both rows (port + official), keep the caveat, and
downgrade the head-to-head claim to the official result.

---

## Item 12 — Multi-seed ImageNet-C

**Goal.** Upgrade `ImageNet-C SAR` from `locked (single seed)` to `locked (5 seeds)`; remove
"One seed; no cross-seed claim" from `tab:claim-status` and Limitations.

**Protocol.** Identical to the authoritative 2026-07-09 full-scale run: ResNet-50, 3 noise
corruptions × severities {1,3,5} × {iid, imbalanced, single_class} = 27 cells; Tent/EATA/SAR
(SAR mechanism-faithful, shared lr). Only the seed varies (1–4; seed 0 is locked).

**Commands.**
```bash
bash $K/runbooks/run_item12_imagenetc_multiseed.sh   # seeds 1..4, then per-seed paired bootstrap
```

**Criterion.** Beats-both must hold per seed with paired-bootstrap CIs excluding 0 (Holm over the
2-comparison family per seed), FA_u ≤ α every seed. Mixed outcome → report per-seed table, keep the
single-seed wording for the headline.

---

## Item 13 — One natural mixed-regime held-out result

**Goal.** A single-dataset natural shift where KGA beats BOTH fixed policies with CIs — the one
claim the paper currently (honestly) lacks.

**Design (draft prereg — seal `runbooks/ITEM13_PREREG_natural_mixed_v1.yaml` in `research_lock/`
BEFORE computing any number).** iWildCam per-camera conditions with the adapter run at two locked
operating points (mild Tent, aggressive SAR-style) so helpful and harmful conditions plausibly
coexist in one natural stream; domain-split calibration (cameras disjoint), one-shot scoring.

**Fail-honest branch (pre-registered).** If the held-out CIs do not exclude zero for both
comparisons, the result is reported as no-harm or diagnostic under the tier policy; the paper's
"no single-dataset natural beats-both" statements stay.

---

## Item 14 — External reproduction of the main table

**Goal.** An independent person reruns one key experiment on a clean machine and signs off.

**Already in place.** `$K/REVIEWER_REPRO_PACKET.md` (Parts B–D: what to run, expected numbers,
sign-off form). Send the packet + repo snapshot; they run Part B.

**Verification.**
```bash
python3 $K/runbooks/verify_external_repro.py --their-results /path/from/reproducer.json
# tolerance-diffs their numbers against paper/generated/kbound_numbers.tex values; exits nonzero on mismatch
```

**Paper edit on success.** Limitations: replace "no external-lab replication" with a one-line
acknowledgment (name/date); attach the signed Part D to `research_lock/`.

---

## Item 15 — Clean exact-rank rerun for all promoted certificate rows

**Goal.** Every promoted row scored with the released exact-rank radius
`ε = r_(k), k = min{n, ⌈(n+1)(1−α)⌉}`, removing the "archived JSONs used the interpolated
quantile" asterisk (§KGA Method, Reproducibility).

**Scope (promoted rows only).** CIFAR-10-C Tent/EATA (5 seeds), ImageNet-C SAR 27 cells,
Camelyon17 OOD, iWildCam H v2, Office-Home M v2, RxRx1 J, three-source OOF mixture.

**Commands.**
```bash
bash $K/runbooks/run_item15_exactrank_promoted_rows.sh
# per-row: rerun scorer with exact-rank ε on the SAME frozen evidence/estimator artifacts,
# then diff verdict/regret/FA_u against the promoted row; writes exactrank_rerun_v1/report.json
```

**Criterion.** All verdicts unchanged → relabel affected rows "exact-rank" and delete the
interpolated-quantile caveat sentences. Any verdict flip → the exact-rank number replaces the
archived one (the released rule is authoritative), tier per policy.

**Also required for SAR on CIFAR (from `KBOUND_SHORT_REMAINING_WORK.md`).** SAR's stress-grid row
stays withheld until the five-seed tree is rebuilt cleanly; that rebuild is a superset of this item
for SAR (`run_item15` includes the `--sar-rebuild` flag gate).
