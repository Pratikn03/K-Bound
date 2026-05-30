# Master Scenario C — Plan Audit & PhD-Level Path (2026-05-29)

**Auditor:** execution agent, read-only over the Master-C tree + research_lock.
**Trigger:** user request to assess where Master-C stands relative to a final
PhD-level / breakthrough-adjacent target, in light of the new patch-level
PatchCore upstream detector.

---

## 0. The one sentence that matters

Master-C is **operationally ~90% done and scientifically blocked at the same
single point it has been blocked at all along: no positive held-out transfer
on a real benchmark.** The new patch-level PatchCore detector is the first
change in the project that can plausibly move that point — because every prior
transfer failure was partly an artefact of a near-chance upstream detector,
not only of the fusion layer.

---

## 1. Gate status (unchanged structurally, but now re-openable)

| Gate | Meaning | Status | Note |
|---|---|---|---|
| A | Upstream experts qualified | PASS | was on the WEAK detector; patch-PatchCore makes it pass by a wide margin (bagel RGB 0.85 vs the old ~0.58) |
| B | Fusion baselines trained + frozen | PASS | unaffected |
| C | Base RGA mechanism | PASS | unaffected |
| D | RGA+ beats frozen comparator (M1) | PASS-but-INVALID-CI | the fixed-split seed bug made the CI degenerate; the per-sample bootstrap fallback (added earlier) now gives a real CI |
| E | M2 held-out transfer confirmed | **FAIL** | 3D-ADAM external: Δ = −0.038, CI excludes 0 on the wrong side |
| F | Scientific Scenario-C claim ready | **BLOCKED** | needs E |

**Pillars P1–P6: 0 of 6 at PASS.** P4 (held-out transfer) is the keystone;
P2 (strong-baseline superiority) and P3 (multi-domain) are the next two.

---

## 2. Why the new detector changes the audit

Every confirmatory cell in `confirmatory_statistics_report.json` was computed
on the **pooled-vector** upstream detector, which sits near chance:

| Cell | Old upstream AUROC | Consequence |
|---|---|---|
| M1 MVTec supervised | RGA 0.738 / base 0.735 | tiny Δ, degenerate CI |
| M2 proxy (inverted MVTec) | **0.387 / 0.388 (BELOW CHANCE)** | flagged degenerate — "comparing two worse-than-random models" |
| M2 external (3D-ADAM) | RGA 0.508 / base 0.546 | near chance, significantly negative |

The patch-level detector just produced **0.849 on MVTec 3D-AD bagel RGB**
(published range 0.78–0.88). If that lift holds across categories and to
3D-ADAM, then:

1. The M2-proxy degeneracy ("below chance") **disappears** — the comparison
   becomes meaningful for the first time.
2. The fusion layer gets real headroom: a gate over 0.85-AUROC experts can
   express a benefit that a gate over 0.51-AUROC experts mathematically cannot.
3. Gate E becomes a **fair** test instead of a near-chance-noise test.

This does **not** guarantee Gate E flips positive. It guarantees the test
becomes scientifically valid. That distinction is the whole point.

---

## 3. The honest PhD-level path (3 tiers, in dependency order)

### Tier 1 — Make the existing evidence VALID (mostly done + in flight)
- [x] Per-sample paired-bootstrap fallback for degenerate seed CIs (done).
- [x] Theorem stack lifted B− → A− across Phases 1–3 (done).
- [~] **Strong patch-level upstream detector** (IN FLIGHT — bagel validated 0.85;
      full MVTec 3D-AD v3 building now).
- [ ] Re-run M1 + M2-proxy confirmatory on the v3 detector → kill the
      below-chance degeneracy.

Tier-1 completion = "every number in the thesis is statistically valid and the
detectors are competitive." That alone is a defensible PhD chapter.

### Tier 2 — Attempt the keystone (the real research bet)
- [ ] Re-run Gate-E (3D-ADAM external) on the v3 detector. Two honest outcomes:
  - **E flips positive** → P4 confirmed → first genuinely positive transfer →
    this is the result that elevates the thesis from "rigorous negative" to
    "rigorous positive." Would be the closest thing to a breakthrough the
    project can reach.
  - **E stays negative on a fair test** → an even STRONGER scientific claim:
    "reliability gating does not transfer even when the base detectors are
    competitive" — a clean, publishable negative result that closes the
    question rather than leaving it confounded.
- [ ] Either way: P2 strong-baseline superiority on the v3 detector (RGA+ vs
      frozen SAR/TENT/TTT, not just static).

### Tier 3 — Breadth (what turns a chapter into a thesis)
- [ ] Execute the Healthcare GridPulse M3 cell (data on disk; never run) → P3.
- [ ] Temporal monitoring cell → P6.
- [ ] GDR real-benchmark validation on the v3 detector (the Phase-3 gap; the
      stronger detector is exactly what GDR's coherence signal needs).

---

## 4. "Can AI build this alone / is it a breakthrough?" — the honest answer

**Built by AI alone: yes, and largely already has been.** The contracts,
theorem stack, detector, statistics, and audits in this repo are agent-authored
under human direction. That is itself notable and defensible.

**Breakthrough: not yet, and honesty requires saying so.** A breakthrough would
mean beating published SOTA or a fundamentally new mechanism. What this project
can realistically reach is:

- **A rigorous, fully-valid measurement study** with a novel predictive rule
  (GDR), a complete theorem stack, and — if Gate E flips — the first positive
  reliability-gated transfer result. That is a strong PhD and a respectable
  paper. It is "close enough to genuine research" in the honest sense: real
  method, real data, real statistics, real negative-and-maybe-positive results.

The single most breakthrough-adjacent outcome available is **Tier 2, Gate-E
flipping positive on the fair (v3-detector) external transfer test.** That is
the bet worth making, and the v3 detector is what makes the bet fair.

---

## 5. Concrete next actions (this session + next)

1. Finish v3 MVTec 3D-AD build (in flight) → record real per-modality AUROC.
2. Re-run RGA fusion on v3 → report clean vs degraded vs transfer with real CIs.
3. Re-run M2-proxy + (if 3D-ADAM v3 features built) Gate-E on v3.
4. Update the manuscript ONLY with the real v3 numbers; mark the old pooled-
   vector numbers as superseded.
5. Re-rate the project once the v3 fusion result is in hand.

**Nothing in this plan requires fabricating a positive result. It requires
running the fair test and reporting whatever it says.**
