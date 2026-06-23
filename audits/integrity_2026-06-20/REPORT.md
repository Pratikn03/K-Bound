# K-Bound Integrity Cleanup Pass — 2026-06-20

Production-readiness integrity pass on `/Volumes/T9/uav/AutoML_Flagship_V8`.
Run on the host (desktop-commander). Every changed verdict was **re-derived from the
underlying numbers**, nothing fabricated, every file backed up before editing, full
audit trail kept in this directory. Git HEAD at start: `a17d744…` (branch `main`).

Artifacts in this folder: `git_HEAD.txt`, `git_status_before.txt`, `backups/`,
`audit_beats_both.py` + `audit_console.txt` + `audit_beats_both_raw.json`,
`audit_bugs_true_to_false.json`, `patch_bugs.py` + `patch_changes.json`,
`verify_after_patch.py`, `extract_benchmark_verdicts.py` + `benchmark_verdicts.json`.

---

## TASK 1 — `beats_both` recomputed to enforce `false_adapt ≤ α` (α = 0.1)

**Correct verdict** = (router_regret < always-freeze_regret) AND (router_regret <
best-fixed-adapt_regret) AND (false_adapt_rate ≤ α) AND (stable across seeds).

### Per-dataset corrected verdict table (machine-extracted from canonical artifacts)

| Dataset | router vs freeze | router vs best-adapt | false_adapt | old flag | corrected verdict |
|---|---|---|---|---|---|
| **CIFAR-10-C** (decisive grid, tent/eata) | 0.0016 < 0.1232 ✓ | 0.0016 < 0.0086 ✓ | 0.000 ✓ | True | **True — WIN (verified)** |
| **ImageNet-C** (noise, sar) | 0.0086 < 0.0277 ✓ | 0.0086 < 0.0606 ✓ | 0.000 ✓ | True | **True — WIN (verified)** |
| **Camelyon17** (Protocol G, eata_online, 5-seed held-out) | 3.6e-5 < 0.0749 ✓ | 3.6e-5 < 0.0013 ✓ | 0.026 ✓ | (verdict_win True) | **True — WIN (verified)** |
| **CIFAR-10.1** (5-seed) | — | — | — | beats_both_count 1/5 | **False — not stable across seeds** |
| **ImageNet-R** (multiseed, 10 arch) | — | — | — | 0/10 candidates | **False — no natural win (only ≥0.83 harmful)** |
| **RxRx1** (Protocol J, 10-seed) | 0.0 = 0.0 (tie) | — | 0.000 | False | **False — null; KGA freezes** |
| **iWildCam** (full-val route-b) | 0.0307 ≈ 0.0310 (tie) | 0.0307 < 0.0546 | **0.500 ✗** | True | **False — FA 0.5 ≫ α (PATCHED)** |
| **Office-Home** (val, route-a) | 0.0281 = 0.0281 (tie) | — | 0.000 | False | **False (val) — KGA freezes** |
| **fMoW** (Protocol L, 5-seed) | 0.0129 = 0.0129 (tie) | 0.0129 > 0.0092 | **0.375 ✗** | False | **False — FA 0.375 ≫ α** |
| **PovertyMap** (Protocol L dev) | — | — | — | dev_screen_stop | **N/A — stopped at dev screen** |
| **ACDC** | — | — | — | — | **N/A — code-only, not run** |

The KNOWN corrections check out: iWildCam → **False** (FA 0.5; the 0.0307-vs-0.0310 near-tie
is the task's "0.0306 vs 0.0309"), Office-Home (val) → **False**. The two synthetic wins
(CIFAR-10-C, ImageNet-C) were **re-verified, not assumed**: both satisfy `FA ≤ α` and both
regret bars, so they correctly **stay True**. Camelyon17 (Protocol G) likewise verified True.

### What was actually wrong vs. what was correct-by-design

The repo computes `beats_both` in two idioms. The audit (`audit_beats_both.py`) found
**327** stored-True `beats_both*` nodes across **110** files; **159** of them are
regret-only-True while violating `FA ≤ α`. Of those:

* **154 are correct-by-design** — they live in files that carry a *separate* gate field
  (`candidate_win` in the win-finder scans `find_kbound_wins.py`; `verdict_win`/`fa_ok`
  in the dev-locked protocol files `iwildcam_protocol_H_v2`, `officehome_protocol_M_v2`).
  In those files `beats_both` is intentionally the regret-only screen and the file's
  **real** verdict already enforces `FA ≤ α`. They were left intact (flipping them would
  corrupt a correctly-designed schema). `verify_after_patch.py` confirms **0** of these
  gate fields are themselves inconsistent (no `candidate_win`/`verdict_win=True` with FA>α).

* **5 were genuine ungated integrity bugs** — `beats_both:true` was itself the stored
  verdict with no gate sibling. These were corrected (each re-derived from its own
  `regret_vs_oracle` + false-adapt rate; original preserved as `beats_both_raw`,
  gated value added as `beats_both_corrected`, live `beats_both` flipped, note added):

  | file | node | regret (router/freeze/adapt) | FA | raw→corrected |
  |---|---|---|---|---|
  | `iwildcam_full_val/result_f08e751c.json` | `routing_b_multicandidate` | 0.0307/0.0310/0.0546 | 0.500 | True→**False** |
  | `iwildcam_full_idval/result_489da28f.json` | `routing_a…/sar_online/kga` | 0.0180/0.0193/0.0994 | 0.500 | True→**False** |
  | `iwildcam_full_idval/result_489da28f.json` | `routing_a…/tent_online/kga` | 0.0210/0.0367/0.0882 | 0.154 | True→**False** |
  | `imagenetc_1pct/decisive_tta_results.json` | `…/eata/metrics` | 0.0028/0.0111/0.0106 | 0.111 | True→**False** |
  | `imagenetr_kbound_light…/result_f4a1293b.json` | `routing_a…/sar_online/kga` | 0.0221/0.0222/0.0400 | 0.500 | True→**False** |

  (The prior iWildCam session `local_8a5d48ea` had **not** left `beats_both_corrected`/
  `_raw` markers in any JSON — confirmed by repo-wide grep — so there was nothing to
  double-edit; these are first-time corrections.)

### Code fix (so future runs are gated)

* `experiments/kbound/wilds/analysis.py::policy_metrics` and
  `packaging/kbound-tta/src/kbound_tta/_analysis.py` **already** gated `beats_both` on
  `FA ≤ α` (and expose `beats_both_regret_only`). The stale JSONs were written by older code.
* `src/scripts/kbound/cifar_tent_mps_v2.py` and its copy
  `docs/research/kbound/scripts/cifar_tent_mps_v2.py` — `policy_metrics` was **ungated**;
  added the `… and adapt.any() and float(np.mean(B[adapt] < 0)) <= ALPHA` gate and
  preserved the old comparison as `beats_both_regret_only`. (This is the writer that
  produced the `decisive_tta` results and the `imagenetc_1pct` bug.)
* `src/scripts/kbound/cifar_tent_online.py` + docs copy — its `beats_both` is an
  accuracy-dominance diagnostic for the harsh streaming sweep (its result JSONs are
  already `false`); added a comment clarifying it is **not** the FA-gated decision verdict.

**Behavioral test (`wilds/analysis.py`, torch-free):** a crafted case with router beating
both on regret but false-adapting 25% returns `beats_both_regret_only=True`,
`beats_both=False` — gate works.

---

## TASK 2 — K-Bound made self-contained from `src/elara` + provenance

**Live cross-imports into `src/elara/` from `kga/` + `experiments/kbound/`:** exactly **one**
(found by runtime import trace — `rg` missed it because the file is git-ignored):

* `experiments/kbound/vendored_from_elara/theory/__init__.py:3` did
  `from elara.theory.theorem_registry import …`, reaching back into the real ELARA package
  while an **identical** vendored sibling (`theorem_registry.py`, 259 lines, stdlib-only,
  same symbols) sat unused. **Fix:** switched to `from .theorem_registry import …`.

All other vendored `__init__`s already use relative imports; the remaining `elara` strings
in the vendored tree are provenance *path literals* in the registry data, not imports.

**Standalone proof:** with an import hook that raises on any `import elara*`, the full `kga`
package + the vendored `theory`/`certification`/`drift` trees import cleanly, the registry
loads (10 theorems), and `kga.certificate.empirical_bernstein` runs →
**"K-Bound imports/runs with src/elara BLOCKED (zero dependency)" PASS**.

**Provenance note** added where the certificate is defined (`kga/certificate.py`): a
paper-ready acknowledgment in the module docstring plus a one-line comment at
`empirical_bernstein`, stating the empirical-Bernstein (Maurer-Pontil 2009) certificate is
**shared with the ELARA companion work** (ELARA's `switching_certificate` delegates to this
`kga` function as the single source of truth) — honest attribution, not hidden. **No ELARA
file was modified.**

---

## TASK 3 — Manifest completeness + Conjecture-1 disambiguation

* **Data manifest (`DATA.md`):** added §3b "Run TTA benchmark suite (vision / WILDS)" — the
  §3 provenance table omitted the vision/WILDS benchmarks that actually ran. §3b lists all
  ten benchmarks + ACDC with their corrected verdicts (the Task-1 table) and the gate
  definition, and notes the dev-locked H_v2/M_v2 held-out protocol wins vs the conservative
  dataset/val nulls for iWildCam/Office-Home.

* **Conjecture 1 disambiguation (naming only):** the paper's **Conjecture 1** is the
  *label-free benefit-sign bracketing* problem (`\label{conj:gen}`,
  `paper/sections/main_theory_5.tex`, already titled "[Label-free bracketing]") — left
  **untouched**. The empirical **p\*-law** was mislabeled "Conjecture 1" in exactly one
  place, `CLAIMS_CALIBRATION.md` (§D); renamed to **"p\*-law conjecture"** with a
  disambiguation note. No statement, threshold, or math changed; the collision is removed.

---

## Verification summary

* `verify_after_patch.py` (post-all-edits): **0** ungated integrity bugs remain; **0**
  gate-consistency violations; the 154 still-regret-only-True nodes all carry a by-design gate.
* All 4 patched JSON files re-validate as well-formed JSON.
* Edited code files `py_compile` clean; gate behavioral test PASS.
* Standalone import (elara blocked) PASS.
* Every edited file diffed against its pre-edit backup → only the intended changes; the
  large `git diff` vs HEAD is **pre-existing** uncommitted work, not from this pass.

Files changed by this pass: 4 result JSONs (verdicts corrected), 4 Python writers (gate /
clarifying comment), 1 vendored `__init__.py` (self-containment), `kga/certificate.py`
(provenance), `DATA.md` (§3b), `CLAIMS_CALIBRATION.md` (rename). No paper claims/numbers
touched beyond these three integrity fixes; no `src/elara` file touched.
