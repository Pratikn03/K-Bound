# K‑Bound — Two‑Page Summary

**When Is Label‑Free Adaptation Knowable? Helpful, Harmful, and Unknowable Regimes Under Distribution Shift**

*One sentence:* before adapting a model on unlabeled data, a system can sometimes **prove** whether adaptation will help, sometimes **prove** it will hurt, and sometimes face an information‑theoretic wall where it must **abstain** — and K‑Bound makes that three‑way decision rigorous, with a finite‑sample certificate that controls the rate of harmful adaptation.

---

## 1. The problem

Test‑time adaptation (TTA) updates a model on unlabeled target data to recover accuracy under distribution shift. But the same unlabeled objective that *helps* under one shift can *silently hurt* under another — sharpening wrong predictions, adapting to label imbalance, or collapsing on non‑stationary streams. Because target labels are unavailable, a deployed system usually **cannot tell which is happening**. Almost all prior work asks "how do we adapt better?" We ask the prior question: **should the system adapt at all, and can it know?**

## 2. The framework: adapt / freeze / abstain

Given a frozen baseline $f_0$, an adapted candidate $f_a$, and an unlabeled batch, define the **benefit** $\Delta = R(f_0) - R(f_a)$ (positive ⇒ adapting helps). The system must choose one of three actions and we separate the world into three label‑free regimes:

- **Knowably helpful** → ADAPT,
- **Knowably harmful** → FREEZE,
- **Unknowable** → ABSTAIN (keep $f_0$, flag for review).

## 3. Theory (the durable core)

1. **Impossibility.** We construct two target worlds that induce *identical* label‑free evidence but *opposite* benefit. No label‑free rule can be correct in both, so in the unknowable regime **abstention is information‑theoretically necessary** — not merely cautious.
2. **Exact benefit‑sign frontier.** The sign of $\Delta$ is identifiable from label‑free observables **iff** an observable margin on the disagreement region exceeds a *calibration‑drift budget*. This is the precise line between knowable and unknowable, and it explains why label‑free heuristics (ATC, agreement‑on‑the‑line) work in benign regimes and fail under drift: they implicitly assume that drift is zero.
3. **Finite‑sample certificate.** Under an observable risk‑alignment assumption we give a certificate $\widehat\Delta \pm \varepsilon$ that routes adapt/freeze/abstain while **controlling the false‑adapt and false‑freeze rates at a chosen level $\alpha$.**
4. **One‑bit dichotomy.** The residual ambiguity at the frontier is exactly *one bit* of orientation, suppliable only by a falsifiable‑but‑not‑verifiable structural assumption — so the boundary is sharp, not a gap to be closed by cleverness.

## 4. KGA: the deployable certificate (ELARA subsumed)

**KGA (Knowability‑Guided Adaptation)** is the algorithm: a *mechanism‑agnostic wrapper* that takes any candidate pool, estimates each candidate's label‑free benefit, attaches a calibrated radius, and returns adapt/freeze/abstain. It does **not** introduce a new adaptation objective — it is a safety/decision layer *around* existing methods (Tent, EATA, SAR, …).

The **multimodal instantiation** (previously developed as ELARA) is now simply a *candidate pool KGA routes over*: when the candidates are reliability‑weighted fusions of multiple sensors/modalities, KGA decides whether to deploy the fused candidate, fall back to the best single modality, or abstain. ELARA is therefore not a separate system but the special case of KGA where the pool is a fusion engine — the same certificate, the same trichotomy.

## 5. Results (honest, pre‑registered)

| Setting | Type | Result |
|---|---|---|
| **CIFAR‑10‑C stress grid (5 seeds)** | synthetic corruption | **Beats both** for Tent & EATA; **0/2160 false‑adapt**; ties collapse‑resistant SAR. Pre‑registered verdict STANDS. |
| **ImageNet‑C, SAR stream** | synthetic corruption | **Beats both** trivial policies on the harmful stream. |
| **Camelyon17 (rich evidence)** | **real natural shift** | **Pre‑registered, held‑out beats‑both:** richer label‑free evidence makes harm detectable (AUC 0.78→0.95); regret 0.0019 vs always‑adapt 0.0045 vs always‑freeze 0.065; **false‑adapt 0.033 ≤ α**, 72% commit; bootstrap P≥0.99 both sides. |
| **Controlled multimodal (MNIST two‑view)** | controlled | Significant beats‑both (P=1.0, zero false‑adapt) — confirms the mechanism when a modality detectably fails. |
| **iWildCam, ImageNet‑R, Office‑Home, RxRx1, BAF, fusion‑engine held‑out** | real shifts / fusion | **Honest nulls:** harm is weakly detectable, so KGA **abstains/freezes** — never false‑adapts. Reported in full. |

The pattern across ~9 real probes is consistent and *is* the thesis: **KGA wins exactly where harmful adaptation is frequent and label‑free detectable, and safely abstains everywhere else.** Every claim is backed by code in the public repo; a 153‑file test suite is green.

## 6. Honest scope

K‑Bound is **not universal**. Where adaptation is already safe, always‑adapt is strong and the certificate only ties or slightly trails it; the certificate's distinctive value is concentrated where catastrophic, detectable harm occurs. The real natural‑shift win is currently **one dataset** with a modest margin over always‑adapt and depends on a richer evidence panel plus a small in‑domain calibration split. The remaining open problem is **detectability on natural shifts**: more *unlabeled* data cannot close the certificate (it is bias‑, not sample‑, limited), so the levers are richer label‑free evidence or a small labeled probe — directions we scope honestly rather than overclaim.

## 7. Contributions (what to take away)

1. A reframing of label‑free TTA as an **adapt/freeze/abstain decision** governed by the sign of benefit.
2. An **impossibility theorem** + an **exact identifiability frontier** for that sign.
3. A **finite‑sample certificate** with false‑adapt control, instantiated as **KGA**, a drop‑in safety layer around any TTA method (multimodal fusion included as a special case).
4. A **pre‑registered evidence base** — one synthetic and one real‑shift beats‑both, plus a battery of honest negatives that precisely confirm the theory's detectability boundary.

*K‑Bound's contribution is knowing the limits of label‑free adaptation — proving where you can act, where you must abstain, and certifying the difference.*
