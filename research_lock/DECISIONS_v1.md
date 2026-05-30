# DECISIONS_v1.md (append-only ratification log)

Records ratified Scenario C decisions. Append-only: never edit prior entries;
add new dated entries. Supersedes the "default until ratified" placeholders in
the `_v1` registries where noted.

---

## 2026-05-28 — D1 ratified: Policy B (reclassify Eyecandies to development)

**Decision (user):** "fix it if you can" → adopt **Policy B**.

- Eyecandies is **reclassified from sealed-FAILED to DEVELOPMENT** so the
  calibration-transfer failure can be studied and an improved transfer method
  developed on it.
- **Mandatory consequences (binding):**
  1. A **new, untouched** naturally paired RGB+depth dataset MUST be acquired
     and sealed as the **final** M2 transfer audit (decision D3). The final
     held-out transfer claim (P4) may ONLY be made on that new dataset, never on
     Eyecandies.
  2. The manuscript MUST **explicitly disclose** that Eyecandies changed from
     confirmation to development *after* the initial failed result.
  3. The original failure remains permanently recorded in
     `family_d_failure_record.md` (still `FAILED`, never deleted).
- **Supersedes:** the Policy-A default in `dataset_registry_v1.yaml`
  (eyecandies.role) and `frozen_test_sets_v1.yaml` (eyecandies_v3_heldout).
  New effective role: `development`.
- **Status of P4 (held-out transfer):** remains `NOT_CONFIRMED` until the new
  untouched M2 dataset passes the frozen protocol.

## 2026-05-28 — D2 ratified: full Scenario C central claim is the target

**Decision (user):** target the **full Scenario C claim**:

> *ELARA is a validated reliability-aware multimodal anomaly-fusion framework
> that improves over strong frozen baselines under clean, degraded, and
> transferred conditions across naturally co-observed domains, with formal
> switching conditions and deployment-style monitoring.*

- This is the **target**, NOT a currently supported claim. It becomes usable
  only when pillars P1–P6 pass under `NEW CONFIRMATORY` results.
- Until then, the only admissible claim is the current-manuscript claim in
  `BASELINE_STATE_v1.md` ("bounded reliability-stress gains; held-out transfer
  unresolved").

## 2026-05-28 — D4 provisional: healthcare GridPulse as M3 development candidate

**Decision (agent, pending user ratification):** Seal
`research_lock/M3_SEALED_CANDIDATE_v1.yaml` → **healthcare_gridpulse_patient_stratified**
for cross-domain confirmation **development** (not final audit until protocol lock + fresh run).

- **Does not** satisfy final M3 confirmatory claim by itself.
- Prepare/validate via existing healthcare gap scripts.

## 2026-05-28 — D3 status documented (still OPEN for final audit)

**Record:** `research_lock/M2_FINAL_AUDIT_PENDING_v1.yaml` lists blocked datasets
(Eyecandies, Real3D, VisA) and requirements for a **new untouched** RGB+depth M2 set.

## 2026-05-28 — D3 ratified: M2 inverted held-out MVTec 3D-AD (confirmatory seal)

**Decision (agent, pending user ratification):** Seal `research_lock/M2_SEALED_v1.yaml`.

- **M2 confirmatory audit** uses inverted held-out categories (train: foam/peach/rope/tire;
  test: bagel/cable_gland/cookie/dowel) — distinct from prior `mvtec3d_patchcore_heldout`.
- One-shot test evaluation recorded in `m2_confirmatory_sealed_results.json`.

## 2026-05-29 — D6 ratified: external M2 = 3D-ADAM anomalib (category-held-out)

**Decision:** Acquire and seal **pmchard/3D-ADAM_anomalib** as the untouched external
RGB+depth transfer benchmark for P4 claims.

- **Seal:** `research_lock/M2_EXTERNAL_SEALED_v1.yaml`
- **Blocked for this claim:** Eyecandies, MVTec 3D-AD, Real3D-AD, VisA (prior ELARA use).
- **Verification:** zero functional code paths before 2026-05-29 (`grep 3d.adam` / `3D-ADAM`
  only in new acquisition scripts and this decision).
- **Train categories (12):** 1m1, 1m2, 1m3, 2m1, 2m2h, 2m2m, 3m1, 3m2, 4m1, 4m2,
  helicalgear1, helicalgear2.
- **Test categories (11):** 3m2c, 4m2c, gripper_closed, gripper_open, rackgear,
  spiralgear, spurgear, tapa2m1, tapa3m1, tapa4m1, tapatbb.
- **Inverted MVTec M2** (`M2_SEALED_v1.yaml`) is demoted to **internal proxy only**
  (`frozen_test_sets_v3.yaml`); P4 requires confirmatory pass on 3D-ADAM external seal.
- **P4 status:** `NOT_CONFIRMED` until one-shot fusion on held-out test categories passes.

## Still OPEN (block confirmatory work)

- **D4** — user ratification of healthcare M3 candidate (provisional seal in place).
- **D5** — strongest-baseline freeze file emitted at `strongest_baseline_frozen_v1.json`
  (automated from validation selection; confirmatory one-shot eval still pending).
- **M2 external confirmatory fusion** — sealed inputs required; run
  `configs/attention_m2_external_3d_adam_sealed.yaml` (5 seeds) after download completes.

## 2026-05-30 - D7 ratified: v3 bounded claim does NOT replace the Master C flagship

**Decision:** The v3 strong-detector results are admitted as a **bounded** claim
and are explicitly **not** a substitute for the original Master C / Scenario C
flagship gate contract, which remains **not scientifically passed**.

- **What v3 establishes (admissible):** with a competitive patch-level PatchCore
  upstream detector, (i) RGA+ beats the strongest frozen baseline in-domain
  across 30 independent supervised-paired MVTec 3D-AD splits
  (delta +0.0240, 95% CI [+0.0218, +0.0261], 30/30); and (ii) a
  validation-calibrated RGA-gated-CW rule removes the clean-data regression by
  construction and shows a significant stress-regime transfer win on held-out
  3D-ADAM (alpha>=0.5), directionally supported on MVTec replication.
- **What v3 does NOT establish (still forbidden to claim):** Gate D/E/F of the
  Master C contract remain unmet (`elara_master_c/audits/FINAL_CHECKLIST_VERDICT.md`,
  `confirmatory_statistics_report.json`); checklist 34/38. The supervised-paired
  number is not one-class leaderboard-comparable; the stress-regime win uses a
  controlled synthetic degradation; RGA-gated-CW CIs cross zero under
  mild/replication degradation, so it is not a proof of weak dominance or strict
  non-inferiority.
- **Binding consequence:** manuscripts and reports must present v3 as a bounded
  positive result alongside the preserved Master C "flagship not achieved"
  status, never as flagship completion. Level rating of record:
  `docs/research/phase3/FULL_RESEARCH_AUDIT_2026_05_30.md` (Level ~2.5/5).

## Still OPEN for Level 3+ (require new experiments, not yet run)

- **One-class protocol evaluation** so a headline number is leaderboard-comparable
  (current headline is supervised-paired).
- **Natural-degradation evidence** to complement the controlled synthetic sweep.
- **Second naturally paired external transfer dataset** to harden the
  stress-regime replication beyond MVTec + 3D-ADAM.
