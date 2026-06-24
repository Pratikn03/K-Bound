# Publication-quality polish — options for figures, tables, pseudocode & algorithm

How the top test-time-adaptation / distribution-shift papers present their work, where K-Bound
currently sits, and a menu of concrete upgrades. Nothing here changes the *results* — it changes how
"finished" the paper looks to a reviewer. Pick the options you want and I'll implement them.

---

## 1. The published-paper standard (what the field actually does)

Grounded in the shared conventions of the most-cited TTA papers (Tent, EATA, SAR, CoTTA, RoTTA, MEMO,
AdaContrast) and recent 2024–25 work (BECoTTA, COME, AR-TTA), plus the NeurIPS/ICML formatting guidance:

| Element | What top papers do |
|---|---|
| **Method** | A numbered **Algorithm box** (`algorithm2e`/`algorithmic`) with explicit Input/Output and 6–12 lines. Almost every TTA paper has one (Tent, EATA, SAR, CoTTA all do). |
| **Teaser** | A **Figure 1 on page 1** — a clean schematic of the method/idea, before the math. Often the most-looked-at object in the paper. |
| **Figures** | **Vector PDF** (not PNG), text ≥ caption size, readable at 100% zoom *and* in grayscale, axes labeled, **colorblind-safe palette** (Okabe–Ito or viridis), figure fonts **matched to the paper's serif**. |
| **Sub-panels** | Proper `(a)/(b)` **subfigures** (`subcaption`), not "Left:/Right:" prose. |
| **Tables** | `booktabs` only (no vertical rules), **best in bold**, `±std`, decimal-aligned numbers (`siunitx`), sometimes a shaded best row. |
| **Notation** | One symbol table or consistent macro set; method name in `\textsc{}`. |

---

## 2. Where K-Bound stands today (scorecard)

| Element | Status | Verdict |
|---|---|---|
| Tables (`booktabs`, no vertical rules) | ✅ already done | **keep** |
| `microtype`, T1 fontenc, `\textsc{KGA}` | ✅ | keep |
| Honest captions, regime framing | ✅ | keep |
| **Algorithm / pseudocode box** | ❌ **none in the paper** | **biggest gap** |
| Figures = vector PDF | ❌ raster PNG @ dpi 130 | upgrade |
| Figure fonts match paper | ❌ matplotlib sans-serif vs. paper serif | upgrade |
| Sub-panels as real subfigures | ❌ `\includegraphics…\hfill` + "Left:/Right:" | upgrade |
| Page-1 teaser (Fig. 1) | ❌ first figure is deep in the body | high upside |
| Colorblind-safe / grayscale-safe palette | ⚠️ partial | nice |
| Decimal-aligned numeric columns (`siunitx`) | ❌ | nice |

Bottom line: your **tables are already publication-grade**; the gaps are the **algorithm box**, **vector
figures**, and a **teaser** — exactly the three things that most cheaply make a paper "look top-tier."

---

## 3. Options — pick what you want

### A. Algorithm / pseudocode  ← highest value, lowest effort, pure LaTeX

- **A1 (recommended): add one `algorithm2e` box for KGA.** Closes the single biggest gap. Draft is in §4 below — ready to paste. Effort: 5 min, no recompile risk (pure LaTeX).
- **A2: add a second mini-box** for the certificate/`ConformalRadius` subroutine (only if you want method depth). Optional.
- **A3: a notation/symbol table** (Δ, ε, φ, α, regimes) — helps reviewers; half a page in the appendix.

### B. Figures

- **B1 (recommended): vector PDF + font match.** Re-save every matplotlib figure as `.pdf` with a shared
  "paper style" header (serif fonts, sizes, 300 dpi fallback). Snippet in §4. *Takes effect when you
  re-run the generators on your Mac* (they need torch/data). Then change `\includegraphics{…png}`→`{…pdf}`.
- **B2 (recommended): a teaser Figure 1.** Turn the decision-flow schematic (or a new TikZ diagram) into a
  one-column page-1 teaser showing the three regimes (helpful → adapt, harmful → freeze, unknowable →
  abstain). I can draft this as **native TikZ** (vector, matches fonts, no external file) — the most
  "published-paper" option.
- **B3: real subfigures.** Add `\usepackage{subcaption}`, convert the side-by-side pairs (mixed, harm,
  ablation, c10c) to `(a)/(b)` with sub-captions; replace "Left:/Right:" prose. Effort: moderate.
- **B4: colorblind-safe palette.** Standardize on Okabe–Ito (`#E69F00 #56B4E9 #009E73 #0072B2 #D55E00`)
  so figures survive grayscale printing. Applied in the same re-run as B1.
- **B5: revive/upgrade the architecture figure.** `fig_architecture` exists but is unused — either drop it
  or redraw it as the teaser (B2). Don't leave it orphaned.

### C. Tables (already good — these are refinements)

- **C1: bold-best + `±std` everywhere.** You do this in some tables; make it uniform across all result
  tables. Low effort, high polish.
- **C2: `siunitx` S-columns** for decimal alignment (numbers line up on the decimal point). `\usepackage{siunitx}`, `S[table-format=1.4]` columns. Nice-to-have.
- **C3: shade the KGA row** (`\rowcolor{gray!12}`) in the headline summary table so the eye lands on it.
- **C4: caption placement** — IEEE convention is caption **above** tables; verify all your `table`
  captions are above the tabular (some may be below).

### D. Cross-cutting

- **D1: consistent decimals** — pick 3 or 4 sig figs and apply everywhere (you mix `0.0019` and `0.07492`).
- **D2: one master figure-style file** so every plot is visually identical (fonts, sizes, palette).

---

## 4. Ready-to-paste starters

### KGA algorithm box (option A1)

Add to the preamble:
```latex
\usepackage[ruled,vlined,linesnumbered]{algorithm2e}
```
Then drop this into the Method section:
```latex
\begin{algorithm}[t]
\caption{\textsc{KGA}: Knowability-Guided Adaptation (one test batch)}\label{alg:kga}
\KwIn{batch $x_{1:n}$; source model $f_0$; adapter $f_a$; evidence map $\phi$;
      calibration set $\mathcal{C}$; level $\alpha$}
\KwOut{decision $d\in\{\textsc{adapt},\textsc{freeze},\textsc{abstain}\}$ and deployed model}
$Z \leftarrow \phi(x_{1:n})$\tcp*{label-free evidence vector}
$\widehat{\Delta} \leftarrow \widehat{g}(Z)$\tcp*{benefit-sign estimate}
$\varepsilon \leftarrow \textsc{ConformalRadius}(\mathcal{C},\alpha)$\tcp*{finite-sample radius}
\uIf{$\widehat{\Delta}-\varepsilon > 0$}{$d\leftarrow\textsc{adapt}$; deploy $f_a$\tcp*{certified helpful}}
\uElseIf{$\widehat{\Delta}+\varepsilon < 0$}{$d\leftarrow\textsc{freeze}$; deploy $f_0$\tcp*{certified harmful}}
\Else{$d\leftarrow\textsc{abstain}$; deploy $f_0$\tcp*{sign unidentifiable (Thm.~\ref{thm:imp})}}
\Return $d$, deployed model
\end{algorithm}
```
This is faithful to your `fig_decision_flow` and the certificate — controls false-adapt at level $\alpha$.

### matplotlib "paper style" header (options B1/B4)

Put at the top of each generator (before plotting):
```python
import matplotlib as mpl
mpl.rcParams.update({
    "savefig.dpi": 300, "savefig.format": "pdf", "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})
OKABE_ITO = ["#E69F00","#56B4E9","#009E73","#0072B2","#D55E00","#CC79A7","#000000"]
# ...save as .pdf instead of .png, then \includegraphics{figures/fig_x.pdf}
```

---

## 5. Suggested priority (best polish per unit effort)

1. **A1** — add the algorithm box. *(Biggest gap, 5 min, I can do it now.)*
2. **B2** — a TikZ teaser Figure 1. *(High reviewer impact; I can draft it now.)*
3. **C1 + C4** — uniform bold-best/±std and caption-above. *(Pure LaTeX, I can do now.)*
4. **B1 + B4** — vector PDF + colorblind palette + font match. *(Needs a Mac re-run of the generators.)*
5. **B3** — real subfigures. *(Moderate LaTeX edit.)*
6. **C2/C3, A2/A3, D1/D2** — refinements, optional.

Items 1–3 I can implement in this session (pure LaTeX). Item 4 needs your Mac (the figure generators
need torch + data); I'll prep the script changes so one re-run does it.

---

*Reminder: this is presentation polish — it makes a clean 80 *look* like a clean 80, which genuinely helps
at review. It does not change the research level; that remains the experimental bet (the pilot).*
