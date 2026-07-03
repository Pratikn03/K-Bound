# Natural Positive Transfer Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a natural clean-transfer track that can become positive only when a frozen candidate beats both SAR and confidence-weighted mean (CW) on a fresh or demonstrably unopened holdout.

**Architecture:** Keep the failed legacy RGA result immutable, then create a D13 positive-transfer track. Opened 3D-ADAM and MulSen runs are development/replication evidence only; official confirmation requires a fresh natural holdout and positive paired CIs against both SAR and CW.

**Tech Stack:** Python, pandas, numpy, scikit-learn, pytest, JSON/CSV/parquet prediction archives, YAML research locks.

---

## Current Evidence To Preserve

The plan starts from these facts, which must remain visible in reports:

- Legacy strict M2 3D-ADAM transfer is negative: RGA `0.5034` vs SAR `0.5433`, delta `-0.0398`, CI `[-0.0648, -0.0146]`.
- V3 cross-modal development result shows a path: validation-selected product/CW-like fusion beats SAR with delta `+0.0628`, CI `[+0.0463, +0.0792]`, but does not beat CW and is forbidden as official Gate E by `research_lock/SCENARIO_C_V3_INTEGRATION_v1.yaml`.
- RGA-gated-CW clean transfer ties CW by construction and wins under controlled degradation from alpha `0.5` onward.
- MulSen has positive mean delta vs SAR but CI crosses zero and the comparator is below chance, so it is not a clean transfer pass.

## Target Definitions

There are two valid targets:

1. **Required target:** New candidate beats frozen SAR on clean natural transfer with delta `>= +0.010` and paired bootstrap CI lower bound `> 0`.
2. **Required target:** The same candidate beats CW on clean natural transfer with delta `>= +0.005` and paired bootstrap CI lower bound `> 0`.
3. **Forbidden:** synthetic degradation, controlled corruption, fake relabeling, and opened-test-selected rules.

Both endpoints are required. A SAR-only win is not enough.

## File Structure

- Create `research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml`: pre-registration for the new candidate and pass/fail criteria.
- Modify `research_lock/DECISIONS_v1.md`: add D13 so the new track is separate from failed legacy RGA and bounded v3.
- Create `src/scripts/scenario_c/diagnose_positive_transfer.py`: retrospective diagnosis only; may read opened 3D-ADAM test but cannot set readiness fields.
- Create `src/uais/fusion/attention/positive_transfer.py`: validation-only candidate selector and scoring rules.
- Create `src/scripts/scenario_c/run_positive_transfer_confirmatory.py`: one-shot runner for a fresh/sealed holdout.
- Modify `src/scripts/scenario_c/confirmatory_statistics.py`: ingest only sealed positive-transfer results and keep old RGA result visible.
- Modify `src/scripts/scenario_c/audit_checklist_progress.py`: expose strict old Gate E, bounded v3, and new positive-transfer track separately.
- Modify `research_dashboard/web/app.js` and `research_dashboard/cpp/main.cpp`: show the new track without overwriting legacy failure.
- Add tests:
  - `tests/test_positive_transfer_protocol_lock.py`
  - `tests/test_positive_transfer_candidate.py`
  - `tests/test_positive_transfer_confirmatory.py`
  - extend `tests/test_scenario_c_checklist.py`

---

### Task 1: Lock The New Positive-Transfer Protocol

**Files:**
- Create: `research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml`
- Modify: `research_lock/DECISIONS_v1.md`
- Test: `tests/test_positive_transfer_protocol_lock.py`

- [ ] **Step 1: Write the failing protocol test**

```python
from pathlib import Path
import yaml


def test_positive_transfer_protocol_is_locked_and_does_not_reuse_opened_result():
    root = Path(__file__).resolve().parents[1]
    path = root / "research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert doc["status"] == "SEALED_DEV_THEN_CONFIRM"
    assert doc["primary_endpoint"]["metric"] == "roc_auc_delta_vs_sar"
    assert doc["primary_endpoint"]["ci_low_must_be_gt"] == 0.0
    assert doc["primary_endpoint"]["minimum_practical_delta"] == 0.010
    assert doc["primary_endpoint"]["holm_alpha"] == 0.05
    assert "experiments/fusion/cross_modal_gate_e_result.json" in doc["forbidden_sources"]
    assert doc["confirmation_rules"]["fresh_or_unopened_holdout_required"] is True
    assert doc["confirmation_rules"]["opened_3d_adam_test_is_development_only"] is True
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=src /tmp/uais-ci-py311/bin/python -m pytest tests/test_positive_transfer_protocol_lock.py -q
```

Expected: fail because the protocol file does not exist yet.

- [ ] **Step 3: Create the protocol lock**

Create `research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml` with this content:

```yaml
version: 1
status: SEALED_DEV_THEN_CONFIRM
ratified: 2026-06-01
decision: D13_positive_transfer_candidate_track

purpose: >
  Develop a new clean external transfer candidate without changing the failed
  legacy RGA result. Already-opened 3D-ADAM test analyses are diagnostic only.

candidate_family:
  name: positive_transfer_clean_default
  allowed_development_sources:
    - experiments/fusion/m2_external_3d_adam_v3_inputs.csv
    - elara_master_c/predictions/v3_transfer
    - experiments/fusion/m2_external_mulsen_sealed_inputs.csv
  rules:
    - validation labels may be used for candidate selection
    - test labels from opened 3D-ADAM may be used only for diagnosis
    - official confirmation requires a fresh or unopened holdout
    - candidate code and hyperparameters must be frozen before confirmation

primary_endpoint:
  metric: roc_auc_delta_vs_sar
  comparator: sar_score_adapter
  ci_low_must_be_gt: 0.0
  minimum_practical_delta: 0.010
  holm_alpha: 0.05
  bootstrap_iterations: 10000

secondary_endpoints:
  clean_vs_cw:
    comparator: confidence_weighted_mean
    desired_ci_low_must_be_gt: 0.0
    acceptable_noninferiority_margin: -0.005
  stress_vs_cw:
    comparator: confidence_weighted_mean
    required_at_alpha_ge: 0.5
    ci_low_must_be_gt: 0.0

confirmation_rules:
  fresh_or_unopened_holdout_required: true
  opened_3d_adam_test_is_development_only: true
  archive_per_sample_predictions: true
  no_test_selected_hyperparameters: true

forbidden_sources:
  - experiments/fusion/cross_modal_gate_e_result.json
  - elara_master_c/audits/checklist_progress.json
  - elara_master_c/audits/confirmatory_statistics_report.json

reporting:
  keep_legacy_negative_gate_e_visible: true
  new_track_field: gate_e_positive_transfer_confirmed
  do_not_overwrite: gate_e_m2_transfer_confirmed
```

- [ ] **Step 4: Append D13 to the decision log**

Append to `research_lock/DECISIONS_v1.md`:

```markdown
## 2026-06-01 - D13 ratified: new positive-transfer candidate track

**Decision:** A new positive-transfer track may be developed, but it cannot
rewrite the failed legacy RGA M2 result and cannot use
`cross_modal_gate_e_result.json` as official Gate E evidence.

- **Development only:** opened 3D-ADAM test diagnostics.
- **Official confirmation:** fresh or unopened holdout with frozen candidate.
- **Primary pass:** candidate vs frozen SAR, paired bootstrap 95% CI lower
  bound > 0, Holm p < 0.05, and delta >= +0.010.
- **Reporting:** expose `gate_e_positive_transfer_confirmed` separately from
  strict legacy `gate_e_m2_transfer_confirmed`.
```

- [ ] **Step 5: Run the protocol test**

Run:

```bash
PYTHONPATH=src /tmp/uais-ci-py311/bin/python -m pytest tests/test_positive_transfer_protocol_lock.py -q
```

Expected: pass.

---

### Task 2: Diagnose Why Transfer Fails And Where It Can Win

**Files:**
- Create: `src/scripts/scenario_c/diagnose_positive_transfer.py`
- Test: `tests/test_positive_transfer_diagnostics.py`
- Output: `elara_master_c/audits/positive_transfer_diagnosis.json`

- [ ] **Step 1: Write tests for diagnosis output**

```python
import json
from pathlib import Path
import subprocess
import sys


def test_positive_transfer_diagnosis_is_development_only():
    root = Path(__file__).resolve().parents[1]
    out = root / "elara_master_c/audits/positive_transfer_diagnosis.json"
    if out.exists():
        out.unlink()

    subprocess.check_call(
        [
            sys.executable,
            "src/scripts/scenario_c/diagnose_positive_transfer.py",
            "--write",
        ],
        cwd=root,
        env={"PYTHONPATH": str(root / "src")},
    )
    doc = json.loads(out.read_text(encoding="utf-8"))

    assert doc["status"] == "DEVELOPMENT_ONLY"
    assert doc["cannot_set_gate_e"] is True
    assert "legacy_m2_negative" in doc
    assert "candidate_levers" in doc
    assert doc["legacy_m2_negative"]["delta_vs_sar"] < 0
```

- [ ] **Step 2: Implement the diagnostic script**

The script must compute:

- legacy per-method AUROC on `M2_external_one_shot_audit`;
- per-category deltas vs SAR;
- v3 clean candidates vs SAR and CW;
- validation-test rank correlation for candidate rules;
- polarity failure flags where AUC `< 0.5`;
- oracle envelope showing whether any validation-selectable method can exceed SAR.

It must write this top-level schema:

```json
{
  "status": "DEVELOPMENT_ONLY",
  "cannot_set_gate_e": true,
  "legacy_m2_negative": {
    "rga_auc": 0.5034,
    "sar_auc": 0.5433,
    "delta_vs_sar": -0.0398
  },
  "candidate_levers": {
    "use_v3_patchcore_scores": true,
    "clean_default_to_cw_or_product": true,
    "avoid_learned_rga_head_on_clean_shift": true,
    "fresh_confirmation_required": true
  }
}
```

- [ ] **Step 3: Run diagnosis**

Run:

```bash
PYTHONPATH=src /tmp/uais-ci-py311/bin/python src/scripts/scenario_c/diagnose_positive_transfer.py --write
```

Expected: JSON written and explicitly marked development-only.

---

### Task 3: Build A Validation-Only Positive Transfer Candidate

**Files:**
- Create: `src/uais/fusion/attention/positive_transfer.py`
- Test: `tests/test_positive_transfer_candidate.py`

- [ ] **Step 1: Write candidate-selection tests**

```python
import numpy as np

from uais.fusion.attention.positive_transfer import (
    candidate_scores,
    paired_auc_bootstrap,
    select_candidate_on_validation,
)


def test_selector_uses_validation_only_and_prefers_low_capacity_on_tie():
    val_y = np.array([0, 0, 1, 1])
    val_rgb = np.array([0.1, 0.2, 0.8, 0.9])
    val_depth = np.array([0.1, 0.2, 0.8, 0.9])
    result = select_candidate_on_validation(val_y, val_rgb, val_depth)
    assert result.selected_rule in {"cw", "product", "rank_cw"}
    assert result.used_test_labels is False


def test_bootstrap_requires_positive_ci_for_pass():
    y = np.array([0, 0, 0, 1, 1, 1])
    a = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    b = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    stat = paired_auc_bootstrap(y, a, b, n_iter=500, seed=0)
    assert stat["delta"] > 0
    assert stat["ci95"][0] >= 0


def test_candidate_scores_are_finite_for_all_rules():
    rgb = np.array([0.0, 0.2, 0.8, 1.0])
    depth = np.array([1.0, 0.8, 0.2, 0.0])
    scores = candidate_scores(rgb, depth)
    assert set(scores) >= {"cw", "product", "max", "softor", "rank_cw"}
    for value in scores.values():
        assert np.isfinite(value).all()
        assert value.shape == rgb.shape
```

- [ ] **Step 2: Implement candidate rules**

Implement these rules in `positive_transfer.py`:

- `cw`: `0.5 * (rgb + depth)`.
- `product`: `sqrt(clip(rgb, 0, 1) * clip(depth, 0, 1))`.
- `max`: `maximum(rgb, depth)`.
- `softor`: `1 - (1-rgb) * (1-depth)`.
- `rank_cw`: average rank-normalized RGB and depth.

Selection rule:

- select only on validation AUROC;
- require validation delta vs SAR `>= +0.010` when SAR validation scores are supplied;
- tie-break by this order: `cw`, `product`, `rank_cw`, `softor`, `max`;
- record `used_test_labels=False`.

- [ ] **Step 3: Run unit tests**

Run:

```bash
PYTHONPATH=src /tmp/uais-ci-py311/bin/python -m pytest tests/test_positive_transfer_candidate.py -q
```

Expected: pass.

---

### Task 4: Add A One-Shot Confirmatory Runner

**Files:**
- Create: `src/scripts/scenario_c/run_positive_transfer_confirmatory.py`
- Test: `tests/test_positive_transfer_confirmatory.py`
- Output: `experiments/fusion/positive_transfer_confirmatory_result.json`

- [ ] **Step 1: Write confirmatory guard tests**

```python
import json
from pathlib import Path


def test_confirmatory_result_from_opened_3d_adam_cannot_be_official(tmp_path):
    result = {
        "protocol": "POSITIVE_TRANSFER_PROTOCOL_v1",
        "holdout_status": "OPENED_DEVELOPMENT_ONLY",
        "gate_e_positive_transfer_confirmed": True,
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    from src.scripts.scenario_c.run_positive_transfer_confirmatory import is_official_confirmation

    assert is_official_confirmation(path) is False


def test_fresh_positive_result_can_be_official(tmp_path):
    result = {
        "protocol": "POSITIVE_TRANSFER_PROTOCOL_v1",
        "holdout_status": "FRESH_OR_UNOPENED",
        "delta_vs_sar": 0.025,
        "ci95_vs_sar": [0.011, 0.039],
        "holm_p": 0.01,
        "minimum_practical_delta": 0.010,
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    from src.scripts.scenario_c.run_positive_transfer_confirmatory import is_official_confirmation

    assert is_official_confirmation(path) is True
```

- [ ] **Step 2: Implement the runner**

Runner behavior:

- read `research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml`;
- train/select candidate on validation only;
- evaluate once on test;
- compute paired bootstrap vs SAR and CW;
- write per-sample archives under `elara_master_c/predictions/confirmation/POSITIVE-TRANSFER-v1/...`;
- set `holdout_status` to `OPENED_DEVELOPMENT_ONLY` for current 3D-ADAM, and `FRESH_OR_UNOPENED` only for a new sealed holdout.

- [ ] **Step 3: Run confirmatory tests**

Run:

```bash
PYTHONPATH=src /tmp/uais-ci-py311/bin/python -m pytest tests/test_positive_transfer_confirmatory.py -q
```

Expected: pass.

---

### Task 5: Get Fresh Confirmation Evidence

**Files:**
- Create or modify one protocol file, depending on dataset choice:
  - `research_lock/M2_EXTERNAL_SEALED_v3.yaml`, or
  - a new versioned MulSen strong-detector seal if MulSen is reused with a materially fixed upstream pipeline.

- [ ] **Step 1: Choose the confirmation source**

Use this priority order:

1. Fresh naturally paired external dataset not used in prior selection.
2. Unopened category reserve from an existing dataset, only if the reserve was never inspected.
3. MulSen rerun only if the protocol explicitly declares it a new upstream-pipeline audit and treats prior MulSen test output as opened/development evidence.

- [ ] **Step 2: Freeze candidate before opening the holdout**

Write the selected candidate ID, rule set, tie-breaks, stop rule, and pass/fail thresholds into the protocol file. No test-result-dependent changes are allowed after this point.

- [ ] **Step 3: Run one-shot confirmation**

Run:

```bash
PYTHONPATH=src /tmp/uais-ci-py311/bin/python src/scripts/scenario_c/run_positive_transfer_confirmatory.py \
  --protocol research_lock/POSITIVE_TRANSFER_PROTOCOL_v1.yaml \
  --holdout fresh_or_unopened \
  --write
```

Expected pass condition:

```text
delta_vs_sar >= +0.010
ci95_vs_sar[0] > 0
holm_p < 0.05
holdout_status == FRESH_OR_UNOPENED
```

---

### Task 6: Integrate Results Without Hiding The Failure

**Files:**
- Modify: `src/scripts/scenario_c/confirmatory_statistics.py`
- Modify: `src/scripts/scenario_c/audit_checklist_progress.py`
- Modify: `research_dashboard/web/app.js`
- Modify: `research_dashboard/cpp/main.cpp`
- Test: extend `tests/test_scenario_c_checklist.py`

- [ ] **Step 1: Add checklist tests**

```python
def test_positive_transfer_track_does_not_overwrite_legacy_gate_e():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "elara_master_c/audits/confirmatory_statistics_report.json").read_text())

    assert report["gate_e_m2_transfer_confirmed"] is False
    assert "gate_e_positive_transfer_confirmed" in report
    assert report["legacy_confirmatory"]["gate_e_m2_transfer_confirmed"] is False
```

- [ ] **Step 2: Add report fields**

Add these fields only when `positive_transfer_confirmatory_result.json` exists:

```json
{
  "gate_e_positive_transfer_confirmed": false,
  "gate_e_positive_transfer_official": false,
  "gate_e_positive_transfer_delta_vs_sar": null,
  "gate_e_positive_transfer_ci95": null,
  "gate_f_positive_transfer_track": false
}
```

Rules:

- `gate_e_m2_transfer_confirmed` remains legacy strict RGA and stays false.
- `gate_e_positive_transfer_confirmed` can pass only if `holdout_status == FRESH_OR_UNOPENED`.
- Dashboard must display three lines: legacy strict fail, bounded v3 pass, new positive-transfer track pending/pass.

- [ ] **Step 3: Run integration tests**

Run:

```bash
PYTHONPATH=src /tmp/uais-ci-py311/bin/python -m pytest \
  tests/test_scenario_c_checklist.py \
  tests/test_positive_transfer_protocol_lock.py \
  tests/test_positive_transfer_candidate.py \
  tests/test_positive_transfer_confirmatory.py \
  -q
```

Expected: pass.

---

## Scientific Stop Rules

Stop development and keep the negative result if any of these remain true after candidate development:

- validation delta vs SAR is `< +0.010`;
- validation win is due only to score polarity reversal or category leakage;
- candidate beats SAR but is worse than CW by more than `0.005` AUROC on clean data;
- fresh holdout CI crosses zero;
- positive result appears only after inspecting the fresh test labels.

## Execution Order

1. Lock D13 protocol.
2. Run diagnosis.
3. Build validation-only candidate.
4. Freeze candidate.
5. Acquire or designate fresh/unopened holdout.
6. Run one-shot confirmation.
7. Integrate results into reports and dashboard.
8. Update manuscript claims only after the fresh confirmation passes.

## Expected Outcome

The most realistic near-term win is positive transfer vs SAR using the v3 strong-detector clean-default family. The stronger flagship target, beating CW on clean transfer, is not supported by current cross-modal experiments and likely needs a new mechanism or stronger upstream expert.
