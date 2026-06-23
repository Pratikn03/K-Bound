# K-Bound paper — layout fix + 9-dataset trim (2026-06-20)

**Scope:** layout repair + dataset trimming only. **No result or verdict was changed.**
Run on the host via desktop-commander (latexmk/pdflatex + ghostscript; poppler absent).

- Canonical source: `docs/research/kbound/kbound.tex` (+ `paper/sections/main_theory_5.tex`)
- Canonical output: `docs/research/kbound/K-Bound_paper.pdf`
- Backups (pre-edit): `kbound.tex.bak_layout9trim_20260620_222802`,
  `paper/sections/main_theory_5.tex.bak_layout9trim_20260620_224439`
- Audit trail: `_audit_renders/kbound_trim_layout.diff`, build logs `_audit_renders/build*.log`,
  final contact sheets `_audit_renders/FINAL{1,2,3}.png`, edit scripts
  `_trim_layout_edits.py`, `_fix_pass2.py`, `_fix_pass3.py`

## Result

| | Before | After |
|---|---|---|
| Pages | 33 | **28** |
| Overfull \hbox | 3 (incl. 89.1pt) | **0** |
| Underfull \hbox | 0 | 0 |
| LaTeX errors / undefined refs / multiply-defined | 0 / 0 / 1 | **0 / 0 / 0** |
| Tables exceeding column/text width | several | **0** |

Every one of the 28 pages was rendered and visually verified: no table exceeds its
column/text width, no content runs off any page edge, and the previously broken pages are fixed.
(The only remaining log warnings are pre-existing cosmetic `T1/lmtt` bold/small-caps font-shape
substitutions, unrelated to layout.)

## Layout fixes

1. **Page-7 right-margin overflow** (text running ~37px past the column): caused by an
   unwrapped duplicate of the five-theorem *consolidation* table in `main_theory_5.tex`.
   Removed the duplicate (the identical, already-resized copy in the appendix is kept) —
   this also cleared the `tab:consolidation` *multiply-defined* warning.
2. **89.1pt overfull \hbox** (the Le-Cam minimax-floor display equation, `main_theory_5.tex`):
   broken from one over-wide line into a 3-line `multline*` → 0pt. Equation content unchanged.
3. **`tab:camelyon17-G`** (Camelyon17 Protocol-G table, 3.0pt over): wrapped the tabular in
   `\resizebox{\columnwidth}{!}{…}`.
4. **Discussion paragraph 2.1pt overfull**: an atomic bold-small-caps em-dash token
   (`…regimes}---`) that no spacing could break; replaced `}---small` with `}: small`
   (a colon, ~7pt narrower) and wrapped the paragraph in `sloppypar`.
5. **Half-empty / sparse pages** (Limitations, Discussion, the RxRx1 table sitting almost
   alone): removed 5 section-/float-isolating `\clearpage` commands so text flows
   continuously. This reclaimed the white space and removed 3 pages.

## Trim to the 9 benchmark datasets

Kept (the 9): CIFAR-10-C, ImageNet-C, Camelyon17, iWildCam, Office-Home, RxRx1, ImageNet-R,
CIFAR-10.1, fMoW (FMoW). Theory (impossibility theorem, Conjecture 1, identifiability/knowability,
Thm 6) left fully intact.

Removed (non-9 benchmark content):

- **PovertyMap** — excluded-wins entry, limitations mention, and GPU-inventory row.
- **The "multimodal panel"** = the entire appendix subsection *"Forward work:
  Target-Label-Light Multimodal Safety Guard"* (Real-IAD / MVTec-3D / 3D-ADAM / MulSen-AD /
  healthcare tracks): the multimodal table (`tab:multimodal`), the probe k-sweep table
  (`tab:probe-k`), Proposition *target-label-light escape* (`prop:tll-escape`), the iWildCam
  label-free val-grid null table (`tab:iwildcam`), and the claim-discipline table.
- **MVTec-3D** "corroborating repository evidence" paragraph in the experiments section.
- Their GPU-inventory rows (PovertyMap, Multimodal KGA, target-label-light probe, iWildCam val-grid).
- All **multimodal / target-label-light** prose in the intro contributions, Discussion,
  Conclusion, and appendix intro — dangling references rewired (0 undefined refs).
- The duplicate theory-copy of the five-theorem consolidation table (a true duplicate).

ACDC was already absent. Final occurrence counts: PovertyMap 0, ACDC 0, MVTec 0, Real-IAD 0,
multimodal 0, target-label-light 0.

## Verdicts unchanged (Table I, `tab:regime-summary`, byte-identical)

| Dataset (protocol) | pre-reg WIN | CI-robust |
|---|---|---|
| CIFAR-10-C stress (Tent/EATA) | yes | yes |
| ImageNet-C (SAR) | yes | re-run |
| Office-Home (Prot. M v2) | yes | **yes** |
| iWildCam (Prot. H v2) | yes | no (point-estimate) |
| Camelyon17 (Prot. G) | no (withdrawn, pooling artifact / no-harm) | no |
| RxRx1 (Prot. J) | no | — |
| ImageNet-R (Prot. D) | no | no |
| CIFAR-10.1 (Prot. K) | no | no |
| FMoW (Prot. L) | no | no |

Wins = CIFAR-10-C, ImageNet-C, Office-Home (CI-robust), iWildCam (point-estimate); the rest are
the honestly-reported no-win / null panel. Camelyon17 was **not** re-introduced as a win.
