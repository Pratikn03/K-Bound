# K-Bound — Released Ceiling Assessment (full-repo sweep, both papers)

*Generated 2026-06-16 after a whole-repo sweep (every result tree, research_lock, both manuscripts),
explicitly checking for artifacts added by other agents. Honest, calibrated, no overclaim.*

## Verdict: **~77 / 100** (strong contribution; up from ~70 once Protocol F landed)

The number moved from ~70 to ~77 for one reason: there is now **one clean, rigorous, real
natural-shift beats-both** (Protocol F, Camelyon17). It is not higher because that win is a single
dataset with a small margin over always-adapt, and every *other* real shift is a null. It is not
"field-shaping" (that is earned by adoption, not declared).

## What the full sweep found (nothing material was missing)
Every result JSON flagged `beats_both:true` was triaged on the metrics that matter (false-adapt ≤ α,
held-out, significance). Only ONE survives as a clean win.

### The positives (verified)
| Result | Shift type | Status |
|---|---|---|
| **Camelyon17 rich-evidence (Protocol F)** | **real natural shift** | **Clean win.** Pre-registered, held-out (dev seeds 0–1 / test 2–4 once), label-free (leakage-checked), false-adapt 0.033≤α, commit 72%, regret 0.0019 < adapt 0.0045 < freeze 0.065, bootstrap P≥0.99 both sides. Enabled by rich evidence (harm-AUC 0.78→0.95). |
| CIFAR-10-C stress grid (Protocol A) | synthetic corruption | Strong. 5-seed STANDS; Tent & EATA beat both, 0/2160 false-adapt. |
| ImageNet-C SAR stream | synthetic corruption | Beats both on the harmful SAR stream (mechanism-faithful). |
| Controlled multimodal (D26) | controlled construction | Significant beats-both (P=1.0, 0 false-adapt) — mechanism confirmation, *not* a natural benchmark. |

### The negatives / nulls (all real shifts except BAF) — consistent with the theory
| Result | Status |
|---|---|
| Camelyon17 full-scale (Protocol B, sparse Z) | FAILS: bias-limited, false-adapt 0.33; **resolved by Protocol F's rich evidence**. |
| iWildCam (val) | Null: weak detectability (harm-AUC 0.62), false-adapt 0.5, abstains 89%. |
| iWildCam (id_val) | `beats_both:true` but **false-adapt 0.78 ≫ α** → safety guarantee violated, NOT a usable win. |
| ImageNet-R diverse panel (Protocol D) | Null: 48/48 abstain, weak detectability. |
| Office-Home (target val/test) | Null: beats_both false; abstains/freezes under undetectable harm. |
| RxRx1 (Protocol C) | Null: harmful-dominated, abstains 99%. |
| A-POWERED-2 ELARA engine (held-out category) | Null: total generalization collapse (test AUROC ~0.50); KGA abstains 30/30. |
| D25 PPI micro-probe (85 cats) | Honest negative: probe doesn't beat-both at k≤64 (rectifier variance dominates). |
| BAF fraud panel (D26-baf) | Honest negative: stacking not > selection on near-chance panel. |

**The pattern across ~9 real probes is remarkably consistent:** KGA avoids harm and abstains under weak
detectability; it *wins* exactly where harm is label-free detectable (CIFAR; Camelyon-with-rich-evidence).
That consistency is strong evidence *for* the theory — and it is why the honest ceiling is a solid strong
contribution, not a fragile one.

## What anchors ~77 (the assets)
- **Theory:** the impossibility result + exact benefit-sign frontier + finite-sample adapt/freeze/abstain
  certificate. Clean, novel, durable — the load-bearing contribution.
- **A real natural-shift beats-both** (Protocol F) — the thing that was missing all along, now honest-to-bank.
- **Multi-seed synthetic win** (CIFAR-10-C STANDS) + ImageNet-C SAR.
- **A pre-registered battery of honest negatives** (research_lock D1–D32) that precisely confirm the
  detectability-gating boundary — reviewers reward this.
- **Reproducibility:** 153-file test suite green (805 passed); every number traces to code.

## What caps it below 80 (the honest discounts)
1. The natural-shift win is **one dataset**, with a **small (though significant) margin over always-adapt**
   (the decisive gain is over always-freeze), and depends on the **rich 17-feature panel + an in-domain
   labeled calibration split** (a permitted operating point, but not the globally-frozen τ\*).
2. Every *other* real shift is a null — no breadth of natural-shift wins yet.
3. The beats-both significance for Protocol F is a **post-hoc** bootstrap (the pre-registered endpoint,
   FA≤α & commit≥0.3, is independently met).
4. No external adoption (field-shaping is earned, not claimed).

## Path upward (honest)
- **~80 (consolidated):** replicate the detectable-harm beats-both on a *second* real shift (the external
  benchmark scoping doc lists MultiBench++/MultiCorrupt; D26 proves the mechanism works once such data is found).
- **~85+:** a deployed, adopted target-label-light guard, or the impossibility result reorganizing how the
  field approaches label-free TTA. Multi-year, adoption-driven.

## Both papers (current, consistent state)
- **Conference** `kbound.tex` → `K-Bound_paper.pdf` (26pp, rebuilt 2026-06-16): abstract + body now feature
  the Protocol F Camelyon win honestly; the D25 micro-probe claim is scoped to an open direction; no claim
  contradicts a run. Tight and submission-ready.
- **Thesis** `manuscript/main.tex` → `main.pdf` (207pp): full battery (incl. Protocol F resolution, D26
  controlled demo, Office-Home safety §, Beyond-label-free future work).

## Minor housekeeping (don't-miss items)
- **Protocol numbering collision:** my controlled-multimodal protocol is filed as `D26` but research_lock
  already had `D26_baf_fraud_panel`. Cosmetic; consider renumbering to D33.
- **`kbound_submission.tex`** (the trimmed cut) is now **stale** vs `kbound.tex` (lacks the Protocol F / D25
  updates). If it's a live submission target, it needs re-syncing; otherwise it's a scratch cut.
- Several older PDF variants (`*_CLEAN`, `*_with_frontier`, `*_officehome`) remain in the folder; the
  canonical outputs are `K-Bound_paper.pdf` and `manuscript/main.pdf`.

## Bottom line
~77: a strong, honest, defensible paper whose every claim now matches a verified result, anchored by a
real natural-shift beats-both. Not field-shaping; not inflatable past ~77 on current evidence. The single
highest-value next step is a **second real-shift detectable-harm win** to consolidate ~80.
