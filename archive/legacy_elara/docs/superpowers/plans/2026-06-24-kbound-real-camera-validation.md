# K-Bound Real-Camera Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a leakage-resistant, two-phone physical package-inspection validation that automatically populates Tables R1--R3 and S1--S5 from locked artifacts.

**Architecture:** Keep the existing synthetic edge demo intact and add a separate manifest-driven real-data backend. Every raw clip has a clip-level class, object, phone, session, shift, repetition, and SHA-256 identity; online code receives frames only, while an offline evaluator joins labels after decisions are logged. Calibration-fit, calibration-conformal, held-out, and external-device replication are session-disjoint, and table exporters read versioned JSON rather than handwritten values.

**Tech Stack:** Python 3.12, PyTorch/torchvision, OpenCV, NumPy, scikit-learn, joblib, PyYAML, psutil, pytest, LaTeX/IEEEtran.

---

## 1. Scientific Lock

### Physical task

Use ten distinct packages, `P01`--`P10`. Each package must be recordable in all four states:

1. `ok`: intact label, centered and correctly oriented.
2. `missing_label`: package present, target label absent.
3. `misaligned_label`: label shifted, rotated, folded, or partially detached.
4. `damaged_label`: label torn, scratched, stained, wrinkled, or obscured.

Use removable labels so package identity is not equivalent to class. Vary the exact offset/damage within a class. Every class must occur under every physical shift; never record one class only in one lighting/background.

### Object split

| Objects | Permitted use |
|---|---|
| `P01`--`P06` | Source training and calibration |
| `P07`--`P08` | Source validation and calibration; never source training |
| `P09`--`P10` | Held-out and replication only; never training or calibration |

### Phone split

| Device | Permitted use |
|---|---|
| `phone_a` | Source, calibration, conformal, and primary held-out sessions |
| `phone_b` | External-device replication only; do not use for development |

Record the manufacturer/model/OS and OpenCV device index in the session manifest. A changed device index does not define a new phone; the immutable `phone_id` does.

### Ten-session schedule

Each session occurs on a different day. Complete a disposable pilot before `S01`; pilot clips live under `artifacts_real/pilot/` and are excluded from every analysis.

| Session | Split | Phone | Objects | Required windows | Purpose |
|---|---|---|---|---:|---|
| `S01` | source_train | A | P01--P06 | 240 | Clean source model training, 10 windows/object/state |
| `S02` | source_val | A | P07--P08 | 80 | Clean object-held-out source validation |
| `S03` | calibration_fit_a | A | P01--P08 | 128 | Lighting, shadow, background, glare |
| `S04` | calibration_fit_b | A | P01--P08 | 128 | Blur, viewpoint, distance, mixed composition |
| `S05` | calibration_conformal_a | A | P01--P08 | 128 | New-day residual calibration, first shift half |
| `S06` | calibration_conformal_b | A | P01--P08 | 128 | New-day residual calibration, second shift half |
| `S07` | heldout_a | A | P09--P10 | 128 | Untouched primary physical test, first half |
| `S08` | heldout_b | A | P09--P10 | 128 | Untouched primary physical test, second half |
| `S09` | replication_a | B | P09--P10 | 128 | External-phone replication, first half |
| `S10` | replication_b | B | P09--P10 | 128 | External-phone replication, second half |

The calibration-fit, conformal, held-out, and replication totals are each 256 windows. At 32 frames/window, the complete study contains 43,008 analyzed frames: 7,680 source-train, 2,560 source-validation, and 8,192 in each of calibration-fit, conformal, held-out, and replication.

### Shift vocabulary

Use these exact IDs:

| Shift ID | Recording definition |
|---|---|
| `mild_light` | Moderate brightness or color-temperature change; package remains fully visible |
| `side_shadow` | Strong side-lit shadow crossing the target label |
| `motion_blur` | Repeatable hand/camera motion during the 32-frame window |
| `new_background` | Background material/color absent from source sessions |
| `viewpoint_45` | Camera yaw or pitch of approximately 45 degrees |
| `distance_scale` | Package occupies 35--55% rather than 70--85% of frame height |
| `glare` | Specular reflection crosses part of the label |
| `batch_composition` | Derived 32-frame window with a pre-locked class mixture assembled from clips in the same session |

For `batch_composition`, source clips remain one class per clip. The window builder assembles a fixed class proportion from same-session clips and records all source clip hashes.

### Exact shift allocation

- `S03`, `S05`, `S07`, and `S09` contain `mild_light`, `side_shadow`, `new_background`, and `glare`.
- `S04`, `S06`, `S08`, and `S10` contain `motion_blur`, `viewpoint_45`, `distance_scale`, and derived `batch_composition` windows.
- Calibration sessions use one physical window per object/state/shift: `8 objects x 4 states x 4 shifts = 128` in each A session; each B session records 96 physical windows and derives 32 composition windows.
- Held-out/replication A sessions use four repetitions: `2 objects x 4 states x 4 shifts x 4 repetitions = 128`.
- Held-out/replication B sessions record `2 objects x 4 states x 3 shifts x 4 repetitions = 96` physical windows and derive 32 composition windows.
- The 32 composition windows use four fixed recipes, eight windows each: balanced `[8,8,8,8]`, ok-heavy `[20,4,4,4]`, missing-heavy `[4,20,4,4]`, and fault-mixed `[4,8,10,10]`, in class order `[ok, missing_label, misaligned_label, damaged_label]`.

### Recording controls

- Record 1080p at 30 fps when supported; retain raw resolution and preprocess to 224x224 in code.
- Use one 32-frame decision window as the atomic unit. Capture 40 frames and deterministically keep frames 5--36 to avoid start/stop motion.
- Use a tripod or fixed mount except for `motion_blur`.
- Randomize object/state/shift order from a generated session checklist.
- Keep all four classes balanced within each session.
- Do not inspect model decisions during calibration, held-out, or replication capture.
- Store raw clips outside Git; store manifests, hashes, aggregate JSON/CSV, and approved sample frames in Git.

### Pre-registered model and decision settings

- Base: MobileNetV3-Small, ImageNet initialization allowed only if declared in the lock.
- Input: 224x224 RGB.
- Window: 32 frames.
- Candidate: episodic Tent, BatchNorm affine parameters only, one Adam step/window.
- Evidence: the locked 14-feature schema.
- Benefit estimator: HistGradientBoostingRegressor fit on calibration-fit only.
- Radius: conservative split-conformal order statistic from calibration-conformal only.
- Miscoverage: `alpha=0.10`; never change after `S05` begins.
- `adapt`: emit candidate prediction.
- `freeze` or `abstain`: emit frozen prediction.
- Confidence/entropy thresholds: select on calibration-fit only and freeze before `S05`.
- Primary held-out result: `S07+S08`; replication is `S09+S10`; pooled results are secondary.

### Primary acceptance and claim rules

These rules classify the result; they do not determine whether data are retained.

1. Integrity pass: all expected windows present, hashes valid, splits disjoint, no online labels, one model/config hash per run.
2. Source gate: balanced accuracy and macro-F1 on `S02` at least 0.80 before calibration begins.
3. Safety gate: held-out KGA `FA_u <= 0.10`; report the empirical rate and exact 95% binomial interval.
4. Coverage gate: report adapt, freeze, and abstain rates without requiring all three to be nonzero.
5. Beats-both claim: KGA regret is lower than always-freeze and always-adapt, and both paired session/object-block bootstrap 95% improvement intervals exclude zero.
6. Heuristic superiority claim: compare KGA against confidence and entropy gates; do not call KGA best if either has lower regret at the same or lower `FA_u`.
7. Replication: report Phone B independently. Do not hide a failed replication or pool it away.
8. Runtime: target mean end-to-end latency <=250 ms and p95 <=400 ms on the declared Mac. Failure changes the deployment claim, not the data.

---

## 2. Artifact Contract

### Raw, local-only artifacts

```text
docs/research/kbound/edge/artifacts_real/
  protocol_lock.json
  protocol_lock.sha256
  raw/<session_id>/<clip_id>.mp4
  raw/<session_id>/<clip_id>.json
  windows/<split>/<window_id>.npz
  models/f0.pt
  calibration/calibration_fit.npz
  calibration/calibration_conformal.npz
  calibration/kga_edge.joblib
  logs/heldout_online.jsonl
  logs/replication_online.jsonl
```

### Versioned evidence artifacts

```text
experiments/kbound/results/edge_real_phone_v1/
  protocol_snapshot.yaml
  recording_inventory.json
  recording_inventory.csv
  split_audit.json
  model_card.json
  calibration_summary.json
  heldout_primary.json
  replication_phone_b.json
  physical_shift_breakdown.json
  per_condition_results.csv
  per_condition_results.json
  anti_leakage_audit.json
  runtime_profile.json
  ablation_results.json
  bootstrap_intervals.json
  REPORT.md
  camera_tables_values.tex
```

`camera_tables_values.tex` contains only macros generated from the JSON files. The table layouts remain in `docs/research/kbound/edge/kbound_camera_main_tables.tex` and `kbound_camera_supp_tables.tex`.

---

## 3. Table Population Contract

| Table | Source artifact | Population rule |
|---|---|---|
| R1 | `protocol_snapshot.yaml`, `recording_inventory.json` | Protocol values and signed session IDs only |
| R2 | `heldout_primary.json`, `runtime_profile.json`, `bootstrap_intervals.json` | Six policies on identical `S07+S08` windows; cells formatted as estimate `[95% CI]` where applicable |
| R3 | `physical_shift_breakdown.json` | Aggregate pre-registered shift families; no selective row deletion |
| S1 | `recording_inventory.json` | Counts, frames, object IDs, phone IDs, class distribution by split |
| S2 | `anti_leakage_audit.json` | Every row must be machine-generated PASS/FAIL plus evidence hash |
| S3 | `per_condition_results.csv/json` | One row per condition/window group; release full file even if PDF shows four examples |
| S4 | `runtime_profile.json` | Mean/p95/RSS memory for each timed stage and total window |
| S5 | `ablation_results.json` | Locked ablations trained/calibrated on development splits, scored once on held-out |

### R2 metric definitions

- Balanced accuracy and macro-F1: compute over all frame predictions emitted by each policy on the held-out stream.
- Per-window benefit: `Delta = candidate_accuracy - frozen_accuracy` on that labeled offline window.
- Realized benefit: `Delta` when policy adapts, otherwise zero.
- Regret: `max(Delta, 0) - realized_benefit`, averaged over windows.
- `FA_u`: fraction of all windows satisfying `decision == adapt and Delta <= 0`.
- `FA_c`: fraction of adapted windows with `Delta <= 0`; return zero with an explicit `n_adapt=0` flag if no adaptations occur.
- Adapt/abstain rates: fractions of all windows.
- Latency: candidate adaptation + candidate inference + evidence + gate for decision latency; also report capture/preprocessing and full end-to-end latency in S4.

---

## 4. Implementation Tasks

### Task 1: Lock the real protocol schema

**Files:**
- Create: `docs/research/kbound/edge/configs/edge_real_phone_v1.yaml`
- Create: `research_lock/KBOUND_EDGE_REAL_PHONE_v1.yaml`
- Create: `docs/research/kbound/edge/src/kbound_edge/real_manifest.py`
- Test: `docs/research/kbound/edge/tests/test_real_manifest.py`

- [ ] **Step 1: Write the manifest validation tests**

```python
import copy

import pytest
import yaml

from kbound_edge.real_manifest import ProtocolError, validate_protocol


@pytest.fixture
def real_protocol():
    with open("configs/edge_real_phone_v1.yaml") as f:
        return yaml.safe_load(f)


def test_protocol_rejects_phone_b_outside_replication(real_protocol):
    cfg = copy.deepcopy(real_protocol)
    cfg["sessions"]["S07"]["phone_id"] = "phone_b"
    with pytest.raises(ProtocolError, match="phone_b.*replication"):
        validate_protocol(cfg)


def test_protocol_requires_disjoint_session_ids(real_protocol):
    cfg = copy.deepcopy(real_protocol)
    cfg["sessions"]["S08"]["session_id"] = "S07"
    with pytest.raises(ProtocolError, match="duplicate session_id"):
        validate_protocol(cfg)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `source ~/.venv_wilds/bin/activate && pytest docs/research/kbound/edge/tests/test_real_manifest.py -q`

Expected: collection fails because `kbound_edge.real_manifest` does not exist.

- [ ] **Step 3: Implement typed protocol loading, validation, and canonical hashing**

The module must expose `ProtocolError(ValueError)`,
`load_real_protocol(path: str | Path) -> dict`,
`validate_protocol(config: dict) -> None`,
`canonical_protocol_hash(config: dict) -> str`, and
`expected_windows(config: dict, session_id: str) -> list[dict]`.

Validation must enforce the exact class list, object split, phone split, session IDs, 32-frame window, `alpha=0.10`, and expected per-session counts above.

- [ ] **Step 4: Run tests**

Expected: all `test_real_manifest.py` tests pass.

- [ ] **Step 5: Commit**

```bash
git add docs/research/kbound/edge/configs/edge_real_phone_v1.yaml \
  research_lock/KBOUND_EDGE_REAL_PHONE_v1.yaml \
  docs/research/kbound/edge/src/kbound_edge/real_manifest.py \
  docs/research/kbound/edge/tests/test_real_manifest.py
git commit -m "protocol(kbound-edge): lock two-phone physical validation"
```

### Task 2: Add deterministic session checklists and real capture

**Files:**
- Create: `docs/research/kbound/edge/src/kbound_edge/recording.py`
- Create: `docs/research/kbound/edge/scripts/00_prepare_real_protocol.py`
- Create: `docs/research/kbound/edge/scripts/01_capture_real_session.py`
- Test: `docs/research/kbound/edge/tests/test_real_recording.py`

- [ ] **Step 1: Test checklist determinism and metadata completeness**

```python
def test_checklist_is_deterministic_and_balanced(real_protocol):
    a = build_session_checklist(real_protocol, "S03")
    b = build_session_checklist(real_protocol, "S03")
    assert a == b
    assert len(a) == 128
    assert set(Counter(x["class_id"] for x in a).values()) == {32}


def test_clip_metadata_has_reproducibility_fields():
    row = make_clip_record(
        clip_id="S03_P01_ok_mild_light_R01",
        session_id="S03",
        phone_id="phone_a",
        object_id="P01",
        class_id="ok",
        shift_id="mild_light",
        repetition=1,
        captured_at="2026-07-03T09:00:00-05:00",
        sha256="a" * 64,
        frame_count=32,
    )
    required = {"clip_id", "session_id", "phone_id", "object_id", "class_id",
                "shift_id", "repetition", "captured_at", "sha256", "frame_count"}
    assert required <= row.keys()
```

- [ ] **Step 2: Verify the tests fail, then implement**

`00_prepare_real_protocol.py` must write the canonical lock, hash, and randomized CSV checklists. `01_capture_real_session.py --session S03 --camera 0 --phone-id phone_a` must show one checklist item at a time, capture 40 frames, keep frames 5--36, write MP4 plus JSON sidecar, hash both, and resume without overwriting completed clips. A separate `--pilot` mode writes only under `artifacts_real/pilot/` and never enters a protocol inventory.

- [ ] **Step 3: Add capture-time hard failures**

Reject incorrect phone/session assignments, fewer than 40 frames, inconsistent resolution, unknown object/class/shift IDs, duplicate clip IDs, and capture after a session is signed closed.

- [ ] **Step 4: Run tests and one pilot capture**

```bash
python scripts/00_prepare_real_protocol.py --config configs/edge_real_phone_v1.yaml
python scripts/01_capture_real_session.py --config configs/edge_real_phone_v1.yaml \
  --pilot --camera 0 --phone-id phone_a --max-items 4
pytest tests/test_real_recording.py -q
```

Expected: four pilot clips, valid sidecars/hashes, tests pass; pilot does not appear in analyzed inventory.

- [ ] **Step 5: Commit**

```bash
git add docs/research/kbound/edge/src/kbound_edge/recording.py \
  docs/research/kbound/edge/scripts/00_prepare_real_protocol.py \
  docs/research/kbound/edge/scripts/01_capture_real_session.py \
  docs/research/kbound/edge/tests/test_real_recording.py
git commit -m "feat(kbound-edge): add locked real-session capture"
```

### Task 3: Validate, freeze, and inventory the dataset

**Files:**
- Create: `docs/research/kbound/edge/src/kbound_edge/real_dataset.py`
- Create: `docs/research/kbound/edge/scripts/02_validate_real_dataset.py`
- Test: `docs/research/kbound/edge/tests/test_real_dataset.py`

- [ ] **Step 1: Test split leakage and hash failures**

```python
def test_audit_rejects_clip_reused_across_splits(tmp_path):
    manifest = manifest_with_same_sha_in("calibration_fit", "heldout")
    report = audit_dataset(manifest)
    assert not report.passed
    assert "cross-split duplicate" in report.failures[0]


def test_online_window_excludes_labels(real_window):
    payload, offline = load_window(real_window)
    assert set(payload) == {"frames", "window_id", "source_hashes"}
    assert "labels" in offline
```

- [ ] **Step 2: Implement inventory and strict audit**

Audit exact expected counts, SHA-256, class balance, frame count, object/phone/session permissions, source-clip lineage for derived mixed windows, duplicate and near-duplicate frames across splits, and chronology. Output `recording_inventory.json/csv` and `split_audit.json`.

- [ ] **Step 3: Add split sealing**

`--seal-through calibration_conformal` writes a signed inventory hash. `--open-split heldout` refuses to run unless all development artifacts and hashes are already frozen.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/test_real_dataset.py -q
git add docs/research/kbound/edge/src/kbound_edge/real_dataset.py \
  docs/research/kbound/edge/scripts/02_validate_real_dataset.py \
  docs/research/kbound/edge/tests/test_real_dataset.py
git commit -m "feat(kbound-edge): audit and seal real recording splits"
```

### Task 4: Train and freeze the source model from manifests

**Files:**
- Modify: `docs/research/kbound/edge/scripts/03_train_source_model.py`
- Modify: `docs/research/kbound/edge/src/kbound_edge/model.py`
- Test: `docs/research/kbound/edge/tests/test_real_source_training.py`

- [ ] **Step 1: Test that only source splits can reach training**

```python
def test_training_loader_never_reads_calibration_or_test(real_manifest):
    train, val = source_datasets(real_manifest)
    assert {x.session_id for x in train} == {"S01"}
    assert {x.session_id for x in val} == {"S02"}
```

- [ ] **Step 2: Add `dataset.kind: real_manifest` support**

Keep synthetic behavior unchanged. Real training uses `S01`, validates on `S02`, fixed epochs/seed, 224x224 transforms declared in the lock, and writes a model card containing protocol hash, data inventory hash, training command, metrics, and state-dict hash.

- [ ] **Step 3: Add the source gate**

The command exits nonzero if `S02` balanced accuracy or macro-F1 is below 0.80. Any permitted model change must occur before `S03`; after `S03`, a changed model hash starts a new protocol version.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/test_real_source_training.py tests/test_candidate_isolation.py -q
git add docs/research/kbound/edge/scripts/03_train_source_model.py \
  docs/research/kbound/edge/src/kbound_edge/model.py \
  docs/research/kbound/edge/tests/test_real_source_training.py
git commit -m "feat(kbound-edge): train source model from sealed clips"
```

### Task 5: Separate calibration-fit from conformal calibration

**Files:**
- Modify: `docs/research/kbound/edge/scripts/04_generate_calibration_pairs.py`
- Modify: `docs/research/kbound/edge/scripts/05_fit_kga_edge.py`
- Modify: `docs/research/kbound/edge/src/kbound_edge/conformal.py`
- Test: `docs/research/kbound/edge/tests/test_real_calibration.py`

- [ ] **Step 1: Write the leakage test**

```python
def test_estimator_and_radius_use_disjoint_sessions(calibration_bundle):
    result = fit_real_certificate(calibration_bundle)
    assert set(result.fit_sessions) == {"S03", "S04"}
    assert set(result.conformal_sessions) == {"S05", "S06"}
    assert set(result.fit_source_hashes).isdisjoint(result.conformal_source_hashes)
```

- [ ] **Step 2: Replace random calibration splitting for real mode**

Real mode writes separate `calibration_fit.npz` and `calibration_conformal.npz`. The estimator fits only the first; the radius uses residuals only from the second. Confidence/entropy thresholds are also selected from calibration-fit and stored in the lock. Synthetic mode may retain its existing deterministic random split.

- [ ] **Step 3: Persist calibration provenance**

Write fit/conformal session IDs, clip hashes, feature schema hash, model hash, sample counts, alpha, epsilon, estimator parameters, MAE, and empirical conformal coverage to `calibration_summary.json`.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/test_real_calibration.py tests/test_conformal.py tests/test_features.py -q
git add docs/research/kbound/edge/scripts/04_generate_calibration_pairs.py \
  docs/research/kbound/edge/scripts/05_fit_kga_edge.py \
  docs/research/kbound/edge/src/kbound_edge/conformal.py \
  docs/research/kbound/edge/tests/test_real_calibration.py
git commit -m "feat(kbound-edge): enforce session-disjoint conformal fitting"
```

### Task 6: Add real held-out and Phone-B evaluators

**Files:**
- Modify: `docs/research/kbound/edge/scripts/06_replay_heldout.py`
- Modify: `docs/research/kbound/edge/src/kbound_edge/replay.py`
- Modify: `docs/research/kbound/edge/src/kbound_edge/metrics.py`
- Create: `docs/research/kbound/edge/scripts/07_replay_replication.py`
- Test: `docs/research/kbound/edge/tests/test_real_evaluation.py`

- [ ] **Step 1: Test policy output and metric semantics**

```python
def test_abstain_emits_frozen_prediction(example_outcome):
    emitted = emitted_predictions("abstain", example_outcome.p0, example_outcome.pa)
    np.testing.assert_array_equal(emitted, example_outcome.p0)


def test_false_adapt_definitions():
    decisions = ["adapt", "adapt", "freeze", "abstain"]
    delta = np.array([0.2, -0.1, -0.2, 0.3])
    m = evaluate(decisions, delta, np.zeros(4))
    assert m["false_adapt_uncond"] == 0.25
    assert m["false_adapt_cond"] == 0.50
```

- [ ] **Step 2: Implement offline label joining**

The online replay receives frame tensors, IDs, and hashes only. After the JSONL file is closed, the offline evaluator joins labels by window ID, computes frozen/candidate and policy outputs, and verifies that every policy used the identical ordered stream.

- [ ] **Step 3: Add balanced accuracy, macro-F1, and CIs**

Compute frame-level balanced accuracy/macro-F1, window regret, `FA_u`, `FA_c`, rates, exact binomial intervals, and 2,000-replicate paired block bootstrap intervals grouped by session, object, and shift. Use a fixed bootstrap seed from the protocol.

- [ ] **Step 4: Keep primary and replication separate**

`06` accepts only `S07+S08`; `07_replay_replication.py` accepts only `S09+S10`. A secondary pooled artifact may be written but cannot replace either table.

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/test_real_evaluation.py tests/test_policy.py tests/test_no_live_labels.py -q
git add docs/research/kbound/edge/scripts/06_replay_heldout.py \
  docs/research/kbound/edge/scripts/07_replay_replication.py \
  docs/research/kbound/edge/src/kbound_edge/replay.py \
  docs/research/kbound/edge/src/kbound_edge/metrics.py \
  docs/research/kbound/edge/tests/test_real_evaluation.py
git commit -m "feat(kbound-edge): score held-out and external-phone streams"
```

### Task 7: Instrument runtime and memory by stage

**Files:**
- Create: `docs/research/kbound/edge/src/kbound_edge/profiling.py`
- Modify: `docs/research/kbound/edge/src/kbound_edge/shadow_runtime.py`
- Test: `docs/research/kbound/edge/tests/test_runtime_profile.py`

- [ ] **Step 1: Test stage accounting**

```python
def test_runtime_profile_contains_all_stages(profile):
    assert set(profile) >= {"capture_preprocess", "frozen_inference", "tent_update",
                            "candidate_inference", "evidence", "gate", "end_to_end"}
    assert profile["end_to_end"]["mean_ms"] >= profile["gate"]["mean_ms"]
```

- [ ] **Step 2: Instrument synchronized timers**

Synchronize MPS/CUDA before stopping timers, discard five warm-up windows, measure resident memory with psutil, and write mean/p50/p95/max for each stage. Record hardware, OS, PyTorch, OpenCV, device backend, thread counts, and power mode.

- [ ] **Step 3: Run performance checks before held-out**

Optimize only on source validation and calibration-fit. Once the runtime implementation hash is frozen, do not change it for held-out or replication.

- [ ] **Step 4: Test and commit**

```bash
pytest tests/test_runtime_profile.py -q
git add docs/research/kbound/edge/src/kbound_edge/profiling.py \
  docs/research/kbound/edge/src/kbound_edge/shadow_runtime.py \
  docs/research/kbound/edge/tests/test_runtime_profile.py
git commit -m "feat(kbound-edge): profile full physical decision latency"
```

### Task 8: Generate an executable anti-leakage audit

**Files:**
- Create: `docs/research/kbound/edge/src/kbound_edge/integrity.py`
- Create: `docs/research/kbound/edge/scripts/08_audit_real_run.py`
- Test: `docs/research/kbound/edge/tests/test_real_integrity.py`

- [ ] **Step 1: Test deliberate violations**

Inject one changed model buffer, one label key in an online row, one reused clip hash, and one mismatched config hash; each must produce a named FAIL.

- [ ] **Step 2: Implement all S2 checks**

The audit emits PASS/FAIL, expected value, observed value, and evidence artifact/hash for all eight S2 rows. Any FAIL makes report generation exit nonzero.

- [ ] **Step 3: Test and commit**

```bash
pytest tests/test_real_integrity.py tests/test_candidate_isolation.py \
  tests/test_log_integrity.py tests/test_no_live_labels.py -q
git add docs/research/kbound/edge/src/kbound_edge/integrity.py \
  docs/research/kbound/edge/scripts/08_audit_real_run.py \
  docs/research/kbound/edge/tests/test_real_integrity.py
git commit -m "feat(kbound-edge): produce machine-verifiable leakage audit"
```

### Task 9: Run locked ablations without test tuning

**Files:**
- Create: `docs/research/kbound/edge/scripts/09_run_real_ablations.py`
- Test: `docs/research/kbound/edge/tests/test_real_ablations.py`

- [ ] **Step 1: Test the six fixed variants**

```python
def test_ablation_registry_is_locked():
    assert tuple(ABLATIONS) == (
        "full_kga", "no_radius", "no_blur_brightness",
        "no_disagreement", "confidence_only", "entropy_only",
    )
```

- [ ] **Step 2: Fit ablation artifacts on development splits only**

Feature-removal variants refit on calibration-fit and recalibrate radii on conformal. Confidence/entropy thresholds remain those locked before held-out. Score all variants on the already logged frozen/candidate outputs so the input stream is identical and no new adaptation run changes data.

- [ ] **Step 3: Test and commit**

```bash
pytest tests/test_real_ablations.py -q
git add docs/research/kbound/edge/scripts/09_run_real_ablations.py \
  docs/research/kbound/edge/tests/test_real_ablations.py
git commit -m "experiments(kbound-edge): add locked physical ablations"
```

### Task 10: Export reports and populate R1--R3/S1--S5

**Files:**
- Create: `docs/research/kbound/edge/src/kbound_edge/reporting.py`
- Create: `docs/research/kbound/edge/scripts/10_make_real_report.py`
- Create: `docs/research/kbound/edge/scripts/11_export_camera_tables.py`
- Modify: `docs/research/kbound/edge/kbound_camera_main_tables.tex`
- Modify: `docs/research/kbound/edge/kbound_camera_supp_tables.tex`
- Test: `docs/research/kbound/edge/tests/test_real_reporting.py`

- [ ] **Step 1: Test that values come from JSON**

```python
def test_exported_macros_match_primary_json(tmp_path):
    export_camera_tables(fixture_results(), tmp_path / "values.tex")
    text = (tmp_path / "values.tex").read_text()
    assert r"\newcommand{\CameraKGARegret}{0.0411}" in text


def test_final_export_rejects_missing_result_cells(tmp_path):
    with pytest.raises(ReportError, match="missing R2 value"):
        export_camera_tables(incomplete_results(), tmp_path / "values.tex", final=True)
```

- [ ] **Step 2: Define generated macros**

Generate one macro per table cell and make the table source consume those macros. `--draft` renders em dashes; `--final` rejects absent R2/S1--S4 values. S5 may remain omitted only when the preregistered minimum-window gate is not met, in which case the report states that reason.

- [ ] **Step 3: Build the report packet**

Create `REPORT.md`, all JSON/CSV artifacts, and a manifest of their SHA-256 hashes. Copy no raw clips into Git. Include approved representative frames with faces, screens, addresses, and barcodes excluded or blurred.

- [ ] **Step 4: Build and visually verify the paper**

```bash
cd docs/research/kbound
latexmk -pdf -interaction=nonstopmode -halt-on-error kbound_short.tex
```

Expected: no undefined references; R1--R2 remain in the main text; R3/S1--S5 remain in the supplement; no clipping or handwritten numeric cell.

- [ ] **Step 5: Test and commit**

```bash
pytest tests/test_real_reporting.py -q
git add docs/research/kbound/edge/src/kbound_edge/reporting.py \
  docs/research/kbound/edge/scripts/10_make_real_report.py \
  docs/research/kbound/edge/scripts/11_export_camera_tables.py \
  docs/research/kbound/edge/kbound_camera_main_tables.tex \
  docs/research/kbound/edge/kbound_camera_supp_tables.tex \
  docs/research/kbound/kbound_short.pdf
git commit -m "docs(kbound): populate locked real-camera evidence tables"
```

---

## 5. Execution Runbook After Implementation

### Before Day 1

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/edge
source ~/.venv_wilds/bin/activate
python -m pip install opencv-python psutil
python scripts/00_prepare_real_protocol.py --config configs/edge_real_phone_v1.yaml
python scripts/01_capture_real_session.py --config configs/edge_real_phone_v1.yaml \
  --pilot --camera 0 --phone-id phone_a --max-items 4
python -m pytest tests -q
```

Discard the pilot from analysis. Check focus, class-state definitions, camera framing, metadata, and runtime. Commit the protocol lock before `S01`.

### Days 1--2: source data

```bash
python scripts/01_capture_real_session.py --config configs/edge_real_phone_v1.yaml --session S01 --camera 0 --phone-id phone_a
python scripts/01_capture_real_session.py --config configs/edge_real_phone_v1.yaml --session S02 --camera 0 --phone-id phone_a
python scripts/02_validate_real_dataset.py --config configs/edge_real_phone_v1.yaml --through source_val --strict
python scripts/03_train_source_model.py --config configs/edge_real_phone_v1.yaml
```

Proceed only if `S02` balanced accuracy and macro-F1 both reach 0.80. Retraining decisions are legal here because calibration and test sessions do not yet exist.

### Days 3--4: calibration-fit and final engineering lock

Capture `S03` and `S04`, validate, generate fit pairs, select confidence/entropy thresholds, and complete any latency optimization using source validation and calibration-fit only. Freeze the model, adapter, features, thresholds, runtime implementation, and hashes before `S05`. No method change is allowed afterward without incrementing the protocol version and restarting conformal capture.

### Days 5--6: conformal calibration

Capture `S05` and `S06`, generate conformal residuals, compute epsilon at alpha 0.10, benchmark the already-frozen runtime, then seal all development artifacts. Do not optimize or alter code, model, features, thresholds, or adapter after `S05` begins.

### Days 7--8: primary held-out

```bash
python scripts/02_validate_real_dataset.py --config configs/edge_real_phone_v1.yaml --open-split heldout
python scripts/01_capture_real_session.py --config configs/edge_real_phone_v1.yaml --session S07 --camera 0 --phone-id phone_a
python scripts/01_capture_real_session.py --config configs/edge_real_phone_v1.yaml --session S08 --camera 0 --phone-id phone_a
python scripts/06_replay_heldout.py --config configs/edge_real_phone_v1.yaml
```

Do not rerun after inspecting results unless correcting a documented execution failure; preserve the original artifact and protocol hash.

### Days 9--10: external-phone replication

Connect Phone B, record its model/OS, verify its camera index, capture `S09` and `S10`, and run the replication command. Do not tune any component using Phone B.

### Final audit and PDF

```bash
python scripts/08_audit_real_run.py --config configs/edge_real_phone_v1.yaml --strict
python scripts/09_run_real_ablations.py --config configs/edge_real_phone_v1.yaml
python scripts/10_make_real_report.py --config configs/edge_real_phone_v1.yaml
python scripts/11_export_camera_tables.py --config configs/edge_real_phone_v1.yaml --final
cd ..
latexmk -pdf -interaction=nonstopmode -halt-on-error kbound_short.tex
```

Open the generated PDF and inspect every table page. Cross-check a random sample of at least ten PDF cells against the JSON source. Commit and push only after full tests, lint, JSON parsing, PDF build, and visual QA pass.

---

## 6. How to Write the Result Honestly

- **Strong held-out win:** State that KGA lowers regret versus both fixed policies with paired CIs excluding zero, while satisfying `FA_u <= alpha`; report whether Phone B replicates it.
- **No-harm result:** State that KGA matches the better fixed policy and beats the worse while controlling false adaptation.
- **Abstain-heavy result:** State that uncertain physical shifts trigger frozen fallback; report coverage and the missed-helpful adaptation cost.
- **Heuristic wins:** State that KGA is safe but not the best router; do not call it a breakthrough.
- **Safety failure:** Report the failed `FA_u` gate and diagnose using calibration/shift artifacts without replacing the locked held-out result.
- **Replication failure:** Keep the primary result and explicitly limit device transfer claims.

Readiness is a property of protocol integrity, reproducibility, and honest reporting. It does not guarantee favorable numbers.
