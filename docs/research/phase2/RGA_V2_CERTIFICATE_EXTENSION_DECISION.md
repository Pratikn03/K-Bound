# RGA-v2 Certificate Extension Decision

**Phase:** 2.2B.2 / Step 6
**Status:** FINAL. No RGA-v2 partial-failure certificate extension is admissible.

## 1. Contract rule

The locked RGA-v2 promotion contract (`configs/phase2/rga_v2_gate_contract.yaml`) requires certificate evidence (criterion C4) **only if** the candidate also passes C1 (clean false-fire budget). C4 cannot rescue or substitute for C1.

## 2. Current state

All three non-baseline candidates G1, G2, G3 fail C1 at clean false-fire rate 1.0000 (vs locked budget 0.0100). See [RGA_V2_SEED_COUNT_DECISION.md](./RGA_V2_SEED_COUNT_DECISION.md) §2.

## 3. Decision

> **No RGA-v2 partial-failure certificate extension is admissible because no candidate satisfies the prerequisite clean false-fire budget criterion C1.**

The `rga_v2_failure_surface_inference.csv` row for each gate retains `C4_positive_certificate = False (no extension admissible per RGA_V2_CERTIFICATE_EXTENSION_DECISION.md)`.

## 4. Implication for switching-certificate file

`experiments/phase2/certification/switching_certificates_v2.csv` will contain certificate rows **only for the G0 mean-gate baseline scenarios** (B1 zero_attack k=4 and B2 max_attack k=4, plus the new clean k=0 baseline rows added by Step 5). No G1/G2/G3 rows are added.

## 5. Test guard

[tests/test_phase2_rga_v2_certificate_extension_boundary.py](../../../tests/test_phase2_rga_v2_certificate_extension_boundary.py) asserts:
- `switching_certificates_v2.csv` contains only G0 gate_ids when no candidate passes C1.
- No row asserts a positive certificate for G1/G2/G3.
