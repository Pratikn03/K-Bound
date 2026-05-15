# arXiv Submission Package — ELARA / RGA

**Author:** Pratik Niroula (independent researcher).
**Status:** Phase A complete; preprint ready for upload.
**Compiled PDF:** [output/pdf/PAPER_DRAFT_v1.pdf](../../../output/pdf/PAPER_DRAFT_v1.pdf) (17 pp, IEEEtran).
**Companion artifact:** [output/pdf/THESIS_CHAPTER_v1.pdf](../../../output/pdf/THESIS_CHAPTER_v1.pdf) (24 pp).

This document is the working checklist + metadata cheat-sheet for the
arXiv upload. Open it next to the arXiv submission form and copy
fields across.

---

## 1. arXiv metadata (copy-paste fields)

### Title
```
ELARA: Reliability-Gated Multimodal Anomaly Fusion under Domain Stress
```

### Author list
```
Pratik Niroula
```
Single author, no institutional affiliation.

### Abstract (≤ 1920 chars including spaces — current version is 1842)

Copy verbatim from the LaTeX abstract in [docs/research/PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex)
(rendered version):

> This manuscript presents ELARA, a reliability-aware fusion pipeline for
> heterogeneous anomaly detection. The system combines domain experts for
> fraud, cyber telemetry, user behavior, and text evidence, then learns a
> fusion layer that produces a unified anomaly risk score and a
> domain-level explanation. The central research question is whether a
> practical fusion layer can preserve clean benchmark accuracy while
> using calibration and drift evidence to reduce brittle behavior under
> missing, shifted, or corrupted domains. The technical contribution is
> Reliability-Gated Attention (RGA), a reliability-aware fusion module
> inside ELARA. RGA couples masked cross-domain attention with a post-hoc
> reliability estimator that combines validation expected calibration
> error, test-time score-distribution drift, and prediction sharpness.
> A conservative gate preserves the static attention path when domains
> appear reliable and injects reliability weights only when batch
> evidence indicates degraded domain quality. The headline empirical
> study is a naturally paired benchmark based on MVTec 3D-AD: eight
> object categories, 3,226 paired RGB and 3D/depth observations, 22.4%
> positive rate, no label alignment required. A contrastive secondary
> benchmark, ELARA-Bench-LA, is built from four real local datasets and
> is explicitly label-aligned. On the naturally paired MVTec 3D-AD
> benchmark, random forest fusion is the strongest baseline, static
> attention is competitive, and the reliability gate of RGA actively
> hurts clean and adversarial performance. On the label-aligned
> secondary benchmark, the same gate improves all-domain coherent
> score-collapse attacks by +0.0506 and +0.0319 ROC-AUC. The contrast
> suggests that validation-derived drift detectors generalize poorly
> across paired-versus-aligned regimes: the KS-drift component
> misfires on legitimate inter-category variation in the natural-paired
> benchmark.

### Primary category
```
cs.LG    Machine Learning (primary)
```

### Cross-list categories
```
cs.AI    Artificial Intelligence
stat.ML  Machine Learning (statistics)
cs.CR    Cryptography and Security  (optional — only if you want the
                                     security-adjacent crowd to see it)
```

### MSC class (Mathematics Subject Classification — optional)
```
68T05    Learning and adaptive systems in artificial intelligence
68T07    Artificial neural networks and deep learning
62H30    Classification and discrimination; cluster analysis
```

### ACM class (optional)
```
I.2.6    Learning
I.5.1    Models — Statistical
```

### Comments (free-text field, visible on arXiv listing)
```
17 pages, 13 figures, 10 tables. Companion 24-page thesis chapter and
runnable codebase: https://github.com/<your-handle>/AutoML_Flagship_V8
(or whatever public repo URL you set). Negative-result paper:
validation-derived drift gates misfire on natural-paired anomaly data.
```

### License
Recommended: `CC BY 4.0` (allows reuse with attribution). Alternative:
`arXiv non-exclusive distribution license` (keeps all your rights but
others can read it).

---

## 2. Upload package contents

What arXiv wants in the tarball, in order of preference:

1. **Source-tarball submission (preferred)** — arXiv recompiles the PDF
   on its end. Pros: arXiv re-renders future updates cleanly; full LaTeX
   source becomes downloadable. Cons: more upload prep.
2. **PDF-only submission** — fastest but locks the PDF; harder to update.

For the source-tarball path, build the upload archive like this:

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
mkdir -p .arxiv_staging
cp docs/research/PAPER_DRAFT_v1.tex .arxiv_staging/main.tex
cp -r docs/research/figures .arxiv_staging/figures
cp -r docs/research/tables .arxiv_staging/tables
# Make graphicspath relative (arXiv unpacks flat by default but our
# tex uses \graphicspath{{figures/}})
cd .arxiv_staging
tar -czf elara_rga_arxiv.tar.gz main.tex figures tables
```

Upload `elara_rga_arxiv.tar.gz` via the arXiv form.

If arXiv complains about the IEEEtran style, include the `.cls` file from
the local TeX Live distribution:

```bash
cp $(kpsewhich IEEEtran.cls) .arxiv_staging/
# re-tar
```

---

## 3. Pre-flight checklist

Run before clicking "submit":

| ✓ | Item | Verification |
|---|---|---|
| ☐ | Build is clean | `./scripts/rebuild_paper.sh` shows zero `Undefined`/`Error` warnings |
| ☐ | All `\cite{}` resolve | check log: 0 `Citation undefined` |
| ☐ | All `\ref{}` resolve | check log: 0 `Reference undefined` |
| ☐ | All bibliography entries cited | verify with `grep -c 'cite{KEY}'` for each `\bibitem{KEY}` |
| ☐ | All tables referenced from prose | none orphan-floating |
| ☐ | All figures referenced from prose | none orphan-floating |
| ☐ | PDF metadata correct | open the PDF, check Title and Author fields |
| ☐ | No identifying CI/email leakage in source | grep for personal email addresses, internal hostnames |
| ☐ | No Anthropic / OpenAI / Claude branding in author block or acknowledgements | review the title-page area |
| ☐ | No `TODO`, `XXX`, `FIXME` in `.tex` source | `grep -nE "TODO\|XXX\|FIXME" PAPER_DRAFT_v1.tex` |
| ☐ | License field set on arXiv form | CC BY 4.0 recommended |
| ☐ | Comments field populated (page count + repo link + negative-result tag) | see §1 above |
| ☐ | Cross-list categories selected | cs.AI + stat.ML at minimum |

Run the verification helper:

```bash
cd docs/research
echo "=== Undefined references / citations? ==="
grep -E "Reference|Citation" ../../.tex_build/PAPER_DRAFT_v1.log | grep -i undefined
echo "=== TODOs in source? ==="
grep -nE "TODO|XXX|FIXME" PAPER_DRAFT_v1.tex
echo "=== Bibliography entries that are never cited? ==="
python3 -c '
import re
text = open("PAPER_DRAFT_v1.tex").read()
defined = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
cited = set()
for group in re.findall(r"\\cite\{([^}]+)\}", text):
    for k in group.split(","):
        cited.add(k.strip())
for key in sorted(defined - cited):
    print(f"UNCITED: {key}")
'
```

All three should return empty. (The previous bash-only helper missed
multi-cite groups with spaces around commas; the Python version splits
on commas correctly.)

---

## 4. arXiv account setup (if you don't have one)

1. Go to `arxiv.org/user/register`.
2. Provide a real institutional or independent-researcher email.
3. arXiv will ask for an **endorsement** for cs.LG if you're a first-time
   submitter. Two ways to get one:
   - Have a previously-published author email arXiv on your behalf (the
     simplest path).
   - Apply for endorsement via the arXiv form; arXiv staff review the
     paper's quality.
4. Endorsement typically takes 1–7 days. Plan for this lead time.

If you don't have a contact for endorsement, **use the arXiv endorsement
request form** referenced from your account dashboard. Mention that the
work is a negative-result paper with a working codebase and that the
companion thesis chapter is available at [output/pdf/THESIS_CHAPTER_v1.pdf](../../../output/pdf/THESIS_CHAPTER_v1.pdf).

---

## 5. Post-submission

Once the preprint is live:

1. **Update repo README** with the arXiv ID (e.g.~`arXiv:2606.NNNNN`).
2. **Tweet thread** (optional but high-leverage): 3–5 tweets summarizing
   the contrast finding, the negative result, the mechanism-isolation
   result, and the code link.
3. **LinkedIn post** with the same content, framed for an applied-ML
   audience.
4. **Email the authors of foundational citations** that you used:
   - Tent (Wang et al. 2021)
   - TTT (Sun et al. 2020)
   - MVTec 3D-AD (Bergmann et al. 2022)
   - M3DM (Wang et al. 2023)
   - Geifman & El-Yaniv 2017 (selective prediction)
   
   One short polite email per author: "Hi, I cited your X paper in a
   recent arXiv preprint on Y; I thought you'd be interested because
   the negative result has implications for Z." Don't expect responses;
   sometimes you get one and it's worth a lot.
5. **Add the preprint to your Google Scholar profile** (set up if you
   haven't).
6. **Submit to the chosen workshop** as soon as the CFP opens
   (recommended: NeurIPS 2026 Workshop on Distribution Shifts, deadline
   typically mid-September).

---

## 6. arXiv-specific style nits to check before upload

Things that don't break the PDF but make the arXiv listing look
unprofessional:

- **No `\thanks{}` macros** in the author block (arXiv strips them
  inconsistently).
- **No `\IEEEspecialpapernotice` markers** ("DRAFT" watermarks).
- **All `\path{}` arguments** should also work if rendered as plain text
  (some arXiv versions don't load the `url` package by default).
- **Embedded fonts**: arXiv's PDF re-render needs all fonts embedded.
  If you upload PDF-only, run `pdffonts output/pdf/PAPER_DRAFT_v1.pdf`
  and verify every font has `(embedded)` or `(embedded subset)`.
- **File names with spaces**: arXiv mangles these. Keep figure / table
  names ASCII-only with underscores.

Current PDF (verified locally):
- 17 pages ✓
- All `\cite{}` resolved ✓
- No `Undefined`/`Error` warnings ✓
- No `TODO`/`XXX`/`FIXME` markers in source ✓

---

## 7. The single line to send to anyone who asks "what is this?"

> *"ELARA is a multimodal anomaly-fusion pipeline. The interesting
> finding is a negative one: a validation-derived drift gate that helps
> under coherent adversarial attack on label-aligned data hurts on
> naturally paired MVTec 3D-AD because it misfires on legitimate
> inter-category variation. arXiv:2606.NNNNN; code on GitHub."*

That's the elevator pitch. Use it in the Twitter post, the LinkedIn
post, the cold emails, and the workshop submission cover letter.

---

## 8. Phase A complete — what's next

After the arXiv preprint is live:

- **Week 3–6:** Phase B — workshop preparation. Add learned-gate row to
  τ-sweep table, per-domain subset attack table, calibration CIs.
- **Week 7–10:** Submit to NeurIPS 2026 workshop.
- **Month 3+:** Phase C — replace MVTec scorer with M3DM-style features,
  add third paired benchmark, target mid-tier conference.

See [PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md) for the full
multi-phase plan.
