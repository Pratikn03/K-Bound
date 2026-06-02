# Level 3+ Plan: One-Class Evaluation and Natural Degradation

**Status:** one-class remains scoped; natural degradation has now been executed
once under D16 and is **not confirmed**. These are the experiments that would
move the project from "Level ~2.5/5 (strong bounded paper)" to a top-tier
full-paper claim. Each requires new compute; none is faked or implied as done.

Authority: `research_lock/DECISIONS_v1.md` D7 (v3 bounded claim is not the
flagship). Current rating: `FULL_RESEARCH_AUDIT_2026_05_30.md`.

---

## Why these two, specifically

The two remaining caveats that bound the v3 result are:
1. The headline 0.978/0.954 numbers are **supervised-paired** (positives visible
   to the fusion training fold) -> not comparable to the published one-class
   MVTec 3D-AD leaderboard (M3DM 0.945, AST 0.937, PatchCore-3D 0.901).
2. The stress-regime transfer win uses a **controlled synthetic** degradation
   (uniform-noise blend). D16 adds a Real-IAD D3 natural-degradation attempt,
   but its primary stress-subset CI crosses zero, so a reviewer can still ask
   for confirmed natural stress evidence.

Closing both converts "bounded in-domain + synthetic-stress" into
"leaderboard-positioned + natural-stress", which is the Level-3 bar.

---

## Task L3.1 - One-class (leaderboard-comparable) evaluation

**Goal.** Produce an RGA / RGA-gated-CW number under the canonical MVTec 3D-AD
one-class protocol (train+val normal-only, mixed test), per category, averaged,
so it sits next to the published leaderboard rather than the supervised-paired
table.

**Steps.**
1. Build a one-class fusion CSV from the patch-PatchCore v3 detector: memory
   bank = train/good only; score val(good) + test(good+defect); NO positives in
   train/val. (The v3 detector already scores this way per category; the change
   is the fusion protocol, not the detector.)
2. Fuse with the parameter-free reliability-gated / gated-CW rule (no trained
   head, since one-class has no fusion-training positives) and with the
   per-category baselines (max-patch RGB-only, depth-only, CW, static).
3. Report per-category one-class image-AUROC, averaged, with the leaderboard
   demarcation table (`mvtec3d_sota_demarcation.tex`) updated to add an ELARA
   one-class row clearly marked comparable / not-comparable.

**Honest expectation.** RGA-gated-CW one-class will likely land below the
trained supervised-paired number and below M3DM/AST (which use cross-modal patch
interactions ELARA does not). The value is an honest leaderboard position, not a
SOTA claim.

**Effort.** ~1 day (detector scores exist; fusion + per-category AUROC + table).

## Task L3.2 - Natural degradation evidence

**Current D16 result.** Real-IAD D3 was downloaded and evaluated with a
validation-selected stress router on a 19-category holdout excluding the opened
smoke category. The all-test result vs CW is positive (`Δ = +0.0624`, 95% CI
`[+0.0347,+0.0896]`), but the primary stress subset is not confirmed
(`Δ = +0.0351`, 95% CI `[-0.0276,+0.0980]`). Clean no-regression passes.

**Goal.** Replace (or complement) the synthetic uniform-noise blend with a
naturally occurring modality degradation and show the same gate crossover with
a primary stress-subset CI excluding zero.

**Candidate natural degradations (no synthetic noise):**
- **Real depth-sensor artifacts on 3D-ADAM/MVTec**: missing-return regions,
  quantisation, or low-illumination RGB subsets that genuinely degrade one
  modality's per-category AUROC.
- **Cross-category held-out shift**: evaluate on categories whose depth (or RGB)
  modality is intrinsically weaker, so the degradation is a property of the data
  rather than injected noise.
- **Acquisition-condition splits** if metadata permits (e.g., exposure / pose
  subsets that degrade one channel).

**Win condition (same as the synthetic sweep, honestly reported).** On a natural
degradation where one modality's reliability genuinely drops, RGA-gated-CW
should default to CW when no drift is detected and beat CW where it is, with a
per-sample bootstrap CI excluding zero. Report ties/losses where they occur.

**Effort.** ~2-3 days (identify a real degradation axis, build the split, run).
The first Real-IAD D3 execution is complete; future work is method/endpoint
refinement or a fresh natural-degradation holdout, not claiming the D16 attempt
as a pass.

## Task L3.3 - Second naturally paired external transfer benchmark

**Goal.** Harden the stress-regime replication beyond MVTec + 3D-ADAM with a
third naturally paired RGB+depth dataset (e.g., a future untouched M2 set per
D3), processed through the same patch-PatchCore + gated-CW pipeline.

**Effort.** Dataset-dependent; gated by acquiring an untouched set (D3 open).

---

## What Level 3 would yield

| Caveat removed | By |
|---|---|
| "supervised-paired, not leaderboard" | L3.1 one-class row |
| "synthetic degradation" | L3.2 confirmed natural degradation; D16 Real-IAD D3 is supportive but not confirmed |
| "only 2 transfer datasets" | L3.3 third external set |

With L3.1 + L3.2 done and honestly reported, the project would support a
top-tier *measurement* paper: "reliability gating: a one-class-positioned,
natural-degradation-validated characterisation of when score-fusion gating
helps." None of this changes the D7 rule that the Master C flagship remains
separate and unmet until its own gates pass.
