# MIXED HEAD-TO-HEAD — DATASET EXTENSION PRE-REGISTRATION

Extends `MIXED_BENCHMARK_PROTOCOL.md` (KGA vs POEM vs AETTA) from the CIFAR-10-C
primary set to additional datasets. **Same metric, same win criterion, same frozen
constants.** This file fixes the additional sets and their *expected* verdicts
**before** any unseen number is computed, so the outcome cannot be reverse-engineered
to favor KGA.

Registered: 2026-06-29. Driver: `experiments/kbound/poem_aetta/run_all_headtohead_extended.sh`.

---

## 1. Scope and the honest reason this is bounded

The published win is on CIFAR-10-C (Tent/EATA, mixed harmful+helpful). The natural
question is "does the win generalize to other datasets?" The answer is **gated by a
hard compatibility constraint, not by effort**:

**POEM and AETTA are faithful ports locked to one evidence panel.** `baselines.py`
asserts the records' `Z_names` equal the frozen 11-dim panel

```
Z_NAMES = [pre_entropy, pre_conf, pre_pbal, post_entropy, post_conf, post_pbal,
           pbal_drop, entropy_drop, frac_highconf, marginal_KL, update_norm]
```

because POEM needs the source-entropy stream and AETTA needs the post-entropy +
high-confidence-fraction signals. **Only the stress-grid pipeline (CIFAR-10-C,
ImageNet-C) logs this panel.** The natural-shift runners log a *different* 18-dim
panel with no `Z_names` (verified on `officehome_full_targettest`: `Z` length 18,
`Z_names` absent). So:

- A dataset can be **added with no new compute** only if it already has
  `per_condition_<ds>_<adapter>_seed<S>.json` records with the 11-dim panel.
- Every other dataset must **regenerate records** with that panel (a GPU pass) before
  POEM/AETTA can run on it faithfully. There is no honest shortcut — feeding the ports
  a different feature panel would break their published algorithms.

The extended driver enforces this automatically: it runs only sets whose records pass
the `Z_names` check and marks the rest **ABSENT** (never silently dropped).

---

## 2. Win criterion (unchanged — re-stated so it is locked here too)

For each set, per `multiseed_paired_ci.py` (paired bootstrap over per-condition mean
regret across seeds, `nboot=1e4`, `BOOT_SEED=20260619`, frozen):

- `diff(KGA, X) = mean_cond[regret_KGA − regret_X]`, X ∈ {POEM, AETTA}.
- **WIN** iff both 95% CIs lie entirely below 0 **and** survive Holm over
  {POEM, AETTA, always-adapt, always-freeze} at family-wise α=0.05, **and** KGA's
  false-adapt ≤ α=0.10.
- **TIE** iff at least one diff CI includes 0 and KGA loses to neither.
- **LOSE** iff some diff CI lies entirely above 0.

Metric = regret-to-oracle. α=0.10, τ_regime=0.02, all frozen. **No re-picking the set,
metric, α, threshold, or bootstrap seed after seeing any diff.** All three verdicts are
publishable.

---

## 3. The registered sets, their state, and pre-committed expected verdicts

Expected verdicts are **hypotheses stated in advance**, not assumptions; the run
decides. "beats-both-trivials" = does KGA also beat always-adapt AND always-freeze
(the legacy headline bar), separate from the no-harm-SOTA WIN test.

| Set | Records | State | Pre-committed *expected* verdict (run decides) |
|---|---|---|---|
| CIFAR-10-C **Tent** | cached | **DONE → WIN** (in paper; harmful 16–18%, beats-both) | — |
| CIFAR-10-C **EATA** | cached | **DONE → WIN** (harmful 4–6%, beats-both) | — |
| CIFAR-10-C **Tent+EATA** | cached | **DONE → WIN** (harmful 10–12%) | — |
| CIFAR-10-C **SAR** | cached | **RAN 2026-06-29 → COVERAGE** | benign grid (0% harmful) → beats POEM/AETTA but **loses to always-adapt**; *not* a headline win. The original protocol §1.2 predicted this; the confirming run is reported transparently, not pre-assumed. Not folded into the paper as a win. |
| **ImageNet-C** noise {Tent,EATA} | **needs regen** (11-dim panel) | UNSEEN | likely benign/TIE on the fixed-severity grid (Tent/EATA stay near-helpful), as on CIFAR. |
| **ImageNet-C SAR-collapse** (online) | **needs regen** | UNSEEN | **WIN-plausible** — SAR genuinely goes harmful online; this is the one extension where a new headline win is realistic. The paper's existing ImageNet-C SAR beats-both (vs trivial policies) supports the hypothesis but does **not** assume the POEM/AETTA verdict. |
| **Natural shifts** (Office-Home, iWildCam, Camelyon17, RxRx1, PACS) | **needs 11-dim-panel regen** | UNSEEN | **TIE** — these are uniformly no-harm under the verified out-of-fold scorer (no beats-both anywhere). Run for *coverage* and to show the certificate's FA_u advantage holds; a WIN here would contradict our own verified no-harm finding and is **not** expected. |

**Honest headline this extension can support:** more *coverage* (the certificate keeps
FA_u ≤ α and KGA ≥ the no-harm SOTA across regimes), plus a *plausible* second headline
win on ImageNet-C SAR-collapse. It will **not** manufacture wins on natural shifts; TIE
there is the correct, pre-committed outcome.

---

## 4. Record generation (to enable a "needs regen" set)

Produce, per adapter and seed, `per_condition_<ds>_<adapter>_seed<S>.json` with
`{"records": [ {Z, Z_names, B, a0, a_adapted, oracle_action, kga_decision,
b_hat, eps_conformal, condition}, ... ]}`, where `Z_names == baselines.Z_NAMES`
(the 11-dim panel) and `B = a_adapted − a0` per condition. This is the same schema the
CIFAR-10-C stress grid emits; run the stress-grid / TTA logging pass on the target
dataset's deployed adapter on GPU (see `RUN_ON_MAC_POEM_AETTA.md` for the runner glue
and the optional official-repo POEM/AETTA arm). Place the files under the `records_dir`
you add to the `EXT` list in `run_all_headtohead_extended.sh`, then re-run the driver —
the set runs automatically once the `Z_names` check passes.

---

## 5. Integrity contract (inherited)

Same as `MIXED_BENCHMARK_PROTOCOL.md` §5: frozen hyperparameters (KGA α=0.10, POEM/AETTA
paper constants), no cherry-picking conditions, no re-picking the criterion after a
result, no weakening a baseline, **absent sets marked ABSENT not silently dropped**, no
fabricated numbers. The cached SAR set was executed during authoring (2026-06-29); its
benign outcome was predicted in advance by the original protocol and is reported as
coverage, with that provenance stated plainly here.

---

## 6. How to run (one command)

```
bash experiments/kbound/poem_aetta/run_all_headtohead_extended.sh
```

Runs the full cached CIFAR-10-C suite immediately, auto-runs any extension set whose
records pass the 11-dim-panel check, marks the rest ABSENT with the exact enable command,
and prints a combined honest verdict summary. Pure numpy/CPU; no torch, no GPU for the
cached arm.
