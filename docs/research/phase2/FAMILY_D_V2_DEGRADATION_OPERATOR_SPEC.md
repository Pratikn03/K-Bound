# Family-D v2 — Degradation Operator Specification

**Phase:** 2.2C / Step 3
**Status:** FROZEN. Any deviation invalidates the held-out invariant.

This document is the human-readable companion to the YAML at
[configs/phase2/family_d_v2_eyecandies_protocol.yaml](../../../configs/phase2/family_d_v2_eyecandies_protocol.yaml).

## 1. Scope and rationale

The operators below apply only to **validation-fold** evidence injection (for reliability-threshold calibration) and to the **one-time future held-out test execution**. They do **not** alter the raw Eyecandies pixels; they degrade the per-modality score features that feed the fusion model, which is the cleanly-supported degradation surface in the current ELARA architecture.

This is a **score-level degradation** protocol. Raw-sensor pixel-level corruption is out of scope for the first Family-D v2 freeze; the claim ceiling therefore excludes any "raw-sensor robustness" language.

## 2. Primary endpoints

### 2.1 D-EYE-1 — depth-channel evidence collapse (primary)

| Field | Value |
|---|---|
| `id` | D-EYE-1 |
| `target_modality` | depth |
| `transformation_level` | modality-score-level (post anomaly-expert) |
| `operator` | set per-domain depth score to 0.0 on the gated batch |
| `parameters` | mode = "zero_collapse" |
| `seed policy` | deterministic; PRNG seed = global seed + 41000 |
| `validation use` | allowed (for τ calibration on normal-only validation) |
| `future test use` | allowed (one-time held-out evaluation) |

### 2.2 D-EYE-2 — RGB-channel evidence collapse (primary)

| Field | Value |
|---|---|
| `id` | D-EYE-2 |
| `target_modality` | rgb |
| `transformation_level` | modality-score-level (post anomaly-expert) |
| `operator` | set per-domain RGB score to 0.0 on the gated batch |
| `parameters` | mode = "zero_collapse" |
| `seed policy` | deterministic; PRNG seed = global seed + 41001 |
| `validation use` | allowed |
| `future test use` | allowed |

## 3. Optional secondary endpoint (descriptive only)

### 3.1 D-EYE-3 — single-modality missingness (secondary descriptive)

| Field | Value |
|---|---|
| `id` | D-EYE-3 |
| `target_modality` | depth OR rgb (alternated per validation batch) |
| `transformation_level` | mask-level (sets the modality's mask = True) |
| `operator` | set domain mask = True (missing) for the targeted modality |
| `parameters` | mask_count = 1 |
| `seed policy` | deterministic; PRNG seed = global seed + 41002 |
| `validation use` | allowed |
| `future test use` | **descriptive only** — does not count toward Holm K=2 primary multiplicity |

## 4. Operator implementation pointer

All three operators are implementable via the existing infrastructure at:

- `src/elara/family_b/corruption.py` (existing per Phase 2.2B utilities) — uses `AdversarialPerturbationEngine` from `src/uais/fusion/attention/adversarial_robustness.py`.
- For D-EYE-1/D-EYE-2: `inject_corruption(..., attack_name="zero_attack", k_values=[1], ...)` with the target modality specified via the existing `target_domain` argument inside `AdversarialPerturbationEngine.apply_attack`.
- For D-EYE-3: a small new wrapper that flips the mask for the targeted modality without touching scores.

These are **score/mask-level** transformations, not raw-pixel transformations. The freeze acknowledges this in the protocol YAML §rationale.

## 5. Validation-only invariants

For every operator:

1. Validation injections use ONLY the normal-only validation split.
2. Per-injection PRNG seed is computed deterministically from `global_seed + offset_per_operator` (see §2.x and §3.1).
3. After threshold selection on validation, the selected τ is **frozen** before any test-fold access.
4. Operator parameters are not changed in response to test outcomes (test outcomes are not even read).
5. Per-injection hash recording (operator + parameters + seed) is included in the per-seed run log.

## 6. Forbidden operator behaviours

- Operator parameters tuned on test outcomes — forbidden.
- Operator switched after test inspection — forbidden.
- Raw-pixel corruption claimed under this freeze — forbidden (this is score/mask-level only).
- Use of D-EYE-3 in the primary Holm correction — forbidden (descriptive only).

## 7. Provenance

- Frozen at Phase 2.2C.
- SHA256 of this file recorded in the partition manifest's `operator_spec_sha256` field.
- Any future modification requires v2 to be re-versioned as v3.
