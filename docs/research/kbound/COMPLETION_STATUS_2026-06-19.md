# K-Bound — Completion Status (2026-06-19)

> **DEPRECATED (2026-06-25).** Superseded by `PROJECT_STATUS_AND_OPEN_PROBLEMS.md` and
> `FREEZE_COMPLETION_PLAN.md`. This file incorrectly lists Conjecture 1 (`conj:gen`) as open;
> it is **resolved negatively** via `thm:conj1-dichotomy`.

Honest, component-by-component "how far to 100%" ledger, written after the positioning
+ calibration + fold-in push. Companion docs: `RELATED_WORK_POSITIONING.md`,
`CLAIMS_CALIBRATION.md`, `RESULTS_PENDING.md`, and the resolution logs appended to
`/GAP_AUDIT.md` and `/INTEGRITY_FIXES.md`.

## Where the program stands

| Component | ~Complete | Status |
|---|---|---|
| Theory (5 theorems + new finite-n Le Cam lower bound) | ~92% | All validated, deterministic; **Conjecture 1 general-position genuinely open** |
| Certificate / `kga` core | ~98% | Importable, tested, served; essentially done |
| Experiments | ~80% | Broad + multi-seed; ImageNet-R multi-seed + Camelyon17 SAR **running on Mac now** |
| Integrity / reproducibility | ~100% | Pre-registration, anti-leakage tests, manifests, self-audit; no fabrication found |
| Manuscript writing | ~80% | Positioning added; final-number fold-in + claims pass now fully mapped |
| Positioning / related-work | ~90% | Drafted against the real 2024–2025 neighbors; final 10% = `\input` wiring + `.bib` + a post-experiment pass |
| **Overall (evidence)** | **~85–88%** | up from ~80% |

## What this push closed (closeable-here, done)

- **Related-work positioning** vs AETTA (CVPR'24), Monitoring Risks in TTA (2025),
  Suitability Filter (2025), ATC, agreement-on-the-line — `paper/sections/related_work_positioning.tex`
  (+ readable `RELATED_WORK_POSITIONING.md`). Key finding folded in: the anytime-valid
  **certificate is the least novel part**; lead with the **knowability boundary**.
- **Claims calibration** — `CLAIMS_CALIBRATION.md` maps every headline claim to
  CLOSED / STRENGTHENED / PENDING-GPU / OPEN with the real backing number/file.
- **Fold-in scaffolding** — `scripts/foldin_multiseed_results.py` ingests a real
  `MULTISEED_ANALYSIS_RESULTS.json` → paper-ready LaTeX rows + Markdown; errors on any
  missing field (no placeholders). Verified on the synthetic smoke output.

## The honest residual to true 100% (cannot be faked)

1. **The two GPU runs** (ImageNet-R multi-seed, Camelyon17 SAR) — executing on the
   author's Mac. Fold-in is pre-staged: one command per run
   (`python scripts/foldin_multiseed_results.py …`, see `RESULTS_PENDING.md`).
   *Note:* completing the **evidence** is in our control; whether it yields a
   *certified* natural-shift win is an empirical question. An honest negative there is
   a valid outcome, not a failure to finish.
2. **Conjecture 1, general-position case** — open research mathematics. The theory
   agent could not remove the assumption and pinned the exact obstruction with a
   machine-checked counterexample (`val_conj1_genpos.py`). It is labeled open in the
   paper, correctly.
3. **Peer review** — external by definition; outside what any amount of internal work
   can close.

## Bottom line

Everything closeable without the GPU or a new theorem is now at the line. The work is
~85–88% complete on evidence, with the remaining ~12–15% being (1) a mechanical
fold-in once the runs finish, (2) one genuinely open conjecture, and (3) review.
"100%" in the sense of *all internal, honest work done and the rest pre-staged* is
reached; "100%" in the sense of *a certified headline win + a closed conjecture* is
not something to assert before the evidence and the math actually deliver it.
