# K-Bound — the 80-level paper: write-from-this kit

You decided to bank the strong, honest paper and write it. Good call. This is everything
you need to write from, plus the few cleanups before submission. Nothing here is aspirational —
every claim below is already in the compiled paper and traces to a committed result file.

---

## 0. WHICH FILE IS THE PAPER (read this first — there are stale duplicates)

| file | role | action |
|---|---|---|
| **`docs/research/kbound/kbound.tex`** (2051 lines) | **THE paper** → compiles to `K-Bound_paper.pdf` | **write here** |
| `paper/kbound.tex` (711 lines) | older/partial variant | ignore (or archive) — do **not** edit by accident |
| `kbound_submission.tex` (1226 lines) | earlier trimmed cut, now stale | ignore — carries a "superseded" header |
| `manuscript/main.tex` | the 200+pp thesis/long-form companion | keep in sync, separate target |

→ Before writing, archive the two stale `*.tex` to a `_archive/` folder so you never edit the wrong one.

---

## 1. The thesis, in one sentence

A label-free certificate that decides **adapt / freeze / abstain** — it provably *must* abstain when
the benefit sign is unidentifiable, and on real distribution shifts it **beats both trivial policies
where harm is detectable** (two independent held-out wins), while **never underperforming the better
policy** elsewhere at false-adapt ≤ α.

The contribution is a **decision problem + an identifiability theory + a safe certificate** — not a
new adaptation mechanism, and not a universal accuracy booster.

---

## 2. Verified claims ledger — write these, cross-check every digit against the result JSON

> Discipline (keep it): before a number goes in prose, confirm it against the committed
> `experiments/kbound/results/.../*.json` it came from. The abstract digits below are authoritative.

**Theory (Sec. 3–4):**
- **Impossibility:** two target worlds, identical observable evidence, opposite benefit ⇒ no label-free rule is correct in both ⇒ abstention is *information-theoretically necessary* (not just cautious).
- **Exact benefit-sign frontier:** the sign is identifiable iff an observable margin on the disagreement region exceeds the calibration-drift budget. (Prior label-free heuristics — ATC, agreement-on-the-line — are the budget-=0 face.)
- **Finite-sample certificate:** adapt/freeze/abstain controlling false-adapt and false-freeze at level α, under the stated risk-alignment/calibration assumptions.
- **One-bit dichotomy:** the residual orientation is exactly one bit (resolved the open conjecture; the *unconditional* weakest class stays open — keep that honest).

**Experiments (Sec. 6) — the six-dataset core panel + mixed stream:**

| result | the claim (verify digits vs JSON) | status |
|---|---|---|
| CIFAR-10-C stress grid (5 seeds) | Tent & EATA **beat both** trivial policies, Holm p≈6e-4, **0 false-adapt** / 2160; SAR ties (below the p\* turn-on) | ✅ headline synthetic win |
| **Camelyon17** (held-out, rich evidence) | **beats-both**, regret-gap 95% CIs **exclude 0**, false-adapt ≤ α; harm became detectable once evidence was enriched | ✅ real-shift win #1 |
| **Office-Home** (held-out, dev-locked) | **beats-both**, gap-vs-freeze CI **[0.008, 0.018] excludes 0**, false-adapt 0%, genuinely adapts (~60%) | ✅ real-shift win #2 |
| iWildCam | **damage-prevention, NOT a win** — crushes reckless always-adapt, *ties* the safe baseline | ⚠️ report as safety, never as a 3rd win |
| Mixed-deployment stream | regret **14–27× below either fixed policy**, CIs exclude 0 | ✅ |
| RxRx1 / ImageNet-R / A-POWERED-2 / micro-probe | honest **negatives/abstentions** — all consistent with the theory's unknowable regime | ✅ keep them in |
| Controlled multimodal (detectable modality failure) | KGA **significantly beats both** — mechanism confirmation | ✅ appendix |
| Code | **805 tests pass** | ✅ reproducibility signal |

---

## 3. Section outline (most is drafted — this is a polish/own-the-prose pass, not a rewrite)

1. **Abstract** — ✅ strong & honest, leave it.
2. **Introduction** — ✅ the "should it adapt at all?" framing; tighten to your voice.
3. **Related work** — position vs ATC / agreement-on-the-line / accuracy-on-the-line / AETTA / e-process monitoring (POEM, Schirmer) / selective prediction. Make the "they're the budget-0 face" point explicit.
4. **Setup & three regimes** — definitions of benefit, helpful/harmful/unknowable.
5. **Theory** — the 5 results (impossibility → frontier → certificate → one-bit dichotomy → rate). Keep it to 5; the weakest-class refinement stays in the appendix.
6. **Method (KGA)** — the wrapper, the evidence vector, the adapt/freeze/abstain rule, the operating point (note the rich-evidence + in-domain calibration honestly).
7. **Experiments** — the six-dataset panel table + mixed stream + the honest-negative battery.
8. **Limitations** — ✅ already candid; this is a feature, keep it.
9. **Excluded alternate wins** — ✅ the anti-cherry-pick section; reviewers respect it.
10. **Conclusion + Appendix** (ImageNet-scale SAR, breadth datasets, weakest-class, proofs).

---

## 4. Honesty guardrails (this is the paper's whole credibility — do not break them)

- **Two** real-shift beats-both, not three. iWildCam ties the safe baseline.
- "**regime-specific** wins," never "universal" / "always."
- State plainly the wins use the **rich-evidence panel + an in-domain labeled calibration split** (a permitted target-label-light operating point, distinct from pure label-free).
- The win claim **is the CI/Holm test**, not a point estimate.
- Significance for Camelyon/Office-Home was a **post-hoc paired bootstrap** (the pre-registered endpoint — FA ≤ α with commit ≥ threshold — is independently met). Say both.
- Keep every negative in. The negatives are *evidence the theory is right*, not weakness.

---

## 5. Pre-submission checklist

- [ ] **Venue:** TMLR (best fit — judges correctness, not hype) or a top NeurIPS/ICML/ICLR workshop. Top-tier main track = stretch shot.
- [ ] Archive `paper/kbound.tex` + `kbound_submission.tex` so the canonical file is unambiguous.
- [ ] Fix the duplicate `\label{tab:consolidation}` (defined twice — rename one).
- [ ] Final compile: confirm **0 undefined references / 0 undefined citations** in the *last* pass.
- [ ] Anonymize for double-blind: author block, GitHub/repo URL, hardware names, any ELARA/identity strings.
- [ ] Release an anonymized repo (code + result JSONs + the `research_lock/` protocols) — your pre-registration is a strength; show it.
- [ ] Cover letter / rebuttal kit: you already have `KBOUND_PLAIN_SUMMARY.md` and `KBOUND_2PAGE_SUMMARY.md`.

---

## 6. What is deliberately NOT in this paper (and shouldn't be)

The micro-probe and multimodal "beat-SOTA" extensions are a **separate, future paper** — that's what
`spotlight_pilot/` tests. Keep this paper a clean, complete, honest contribution: the decision
problem, the theory, the certificate, and the regime-specific wins. Ship it.

---

*This kit reflects the verified state as of 2026-06-20. The paper is real, honest, and submission-shaped.
It's an excellent result — write it, polish it in your voice, and send it.*
