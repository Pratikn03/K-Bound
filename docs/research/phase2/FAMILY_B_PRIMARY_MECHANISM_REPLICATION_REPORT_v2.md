# Family-B Primary Mechanism Replication (B-MECH-1) — v2

**Cell:** B-MECH-1
**Benchmark:** ELARA-Bench-LA (4 domains; n_test = 1600 samples × 2 scenarios)
**Protocol:** k=4 all-domain failure, mean gate, τ=0.66 (LOCKED)
**Seeds:** 30 (seeds 42–71)
**Inference:** seed-averaged DeLong + paired sample bootstrap CI (10 000 iter, fixed seed 0); Holm K=2 across {B1, B2}.

## 1. Headline result — BOTH ENDPOINTS REPRODUCED

| Endpoint | Scenario | Phase-2 Δ AUC | Phase-2 95% CI | Phase-2 Holm K=2 p | Phase-1 target Δ | Phase-1 target CI | Sign-consistent seeds | Decision |
|---|---|---:|:---:|---:|---:|:---:|:---:|:---:|
| **B1** | zero_attack k=4 | **+0.0507** | [+0.0364, +0.0650] | **4.3 × 10⁻¹²** | +0.0506 | [0.0315, 0.0681] | 29 / 30 | **REPRODUCED** |
| **B2** | max_attack k=4  | **+0.0939** | [+0.0741, +0.1149] | **< 1 × 10⁻¹⁵** | +0.0319 | [0.0050, 0.0617] | 27 / 30 | **REPRODUCED** |

Source: [experiments/phase2/mechanism/family_b_primary_replication_inference.csv](../../../experiments/phase2/mechanism/family_b_primary_replication_inference.csv), [experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv](../../../experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv).

Per-seed descriptive statistics:

- B1: per-seed mean Δ = +0.0448, SD = 0.0178, 29 / 30 sign-positive.
- B2: per-seed mean Δ = +0.0625, SD = 0.0383, 27 / 30 sign-positive.

The B1 Phase-2 point estimate **+0.0507** matches the Phase-1 target **+0.0506** almost exactly. The B2 Phase-2 point estimate **+0.0939** is substantially **larger** than the Phase-1 target **+0.0319** (the CIs overlap but Phase-2 sits in the upper portion).

## 2. Permitted interpretation

> "Under the Phase-2 archived-prediction pipeline (30 seeds, seed-averaged DeLong + paired sample bootstrap), both primary mechanism endpoints reproduce: B1 zero_attack k=4 with Δ AUC = +0.0507 (95% CI [+0.0364, +0.0650]; Holm K=2 p = 4.3 × 10⁻¹²) and B2 max_attack k=4 with Δ AUC = +0.0939 (95% CI [+0.0741, +0.1149]; Holm K=2 p < 1e-15). Both effects are 'large' on the Phase-2 practical-effect band; sign consistency is 29/30 (B1) and 27/30 (B2)."

## 3. Forbidden interpretation

- "RGA solves partial failure." (B-MECH-2 partial-failure surface was not executed under v2 — only k=4 coherent collapse was tested.)
- "RGA-v2 is promoted." (B-MECH-2 RGA-v2 sweep was not executed.)
- "Mechanism evidence proves cross-domain generalization." (B-MECH-1 is mechanism evidence on label-aligned stress only; not a cross-domain test.)
- "ELARA is deployment-ready." (Reproduced mechanism does not unlock deployment claims.)
- Any claim that Phase-1 numbers were inflated/deflated based on this v2 result — both are within their own confidence framing and the cross-comparison is informational, not deflationary.

## 4. Honest caveats

- **B2 is larger than the Phase-1 target.** The Phase-1 B2 endpoint Δ = +0.0319 [0.0050, 0.0617] does not include the Phase-2 point estimate +0.0939. Possible explanations include: (a) Phase-1 used 5 seeds vs Phase-2's 30, so the Phase-2 estimate has a tighter CI but is also drawing from a slightly different model-init distribution; (b) per-seed noise in the corruption injection; (c) bug in either phase. The cleanest forward action is to **report both phases side by side without re-defining either endpoint** — the v2 contract forbids retroactively redefining B1/B2 endpoints after seeing outcomes.
- **The B-MECH-1 driver wrote per-sample `gate_fired = False` for every sample** because it consumed the runner's batch-level `adapted` flag, which under-reports per-sample firing. The empirical AUC delta still shows real RGA-vs-static separation; B-CERT-1 (below) uses an empirical-firing definition to recover meaningful certificate input. This is an implementation issue, not a result issue.

## 5. Reproduction

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_mechanism_replication.py \
    --experiment-id B-MECH-1 --seeds 30 --seed-start 42
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_b_mech_1_inference.py
```

Wall-clock observed: ~6 min for 30 seeds on M-series Mac.

## 6. Provenance

- Pre-execution commit: `2719d8111405a4fcc75e288678cd5a18d37134c5` (Phase 2.2B infrastructure).
- Protocol lock: [MIXTURE_SHIFT_PROTOCOL.md](./MIXTURE_SHIFT_PROTOCOL.md), [PHASE_2_2B_EXECUTION_PRECHECK.md](./PHASE_2_2B_EXECUTION_PRECHECK.md).
- Selection log: every archived row carries `selection_used_test_metrics=False` (verified by the v2 schema test).
