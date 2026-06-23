# KGA-ELARA Full Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose ELARA-U as an optional, claim-safe KGA integration with strict retrospective, target-label-light, and label-free modes; run it through one canonical command; and connect the verified retrospective output to the short paper without changing KGA's theory or headline claims.

**Architecture:** `kga.integrations.elara` owns the integration boundary and delegates candidate generation to the existing ELARA-U router and certification to the existing KGA facade. Evaluation labels are separated from decision inputs by mode: retrospective audit may use all labels, target-label-light may use only fixed probe indices, and label-free rejects target labels and requires a frozen disjoint estimator. A versioned runner consumes a locked YAML protocol, writes JSON/LaTeX/findings/manifest artifacts, and applies a deterministic promotion guard.

**Tech Stack:** Python 3.12, NumPy, SciPy, scikit-learn through the existing ELARA-U router, PyYAML, pytest, Bash, LaTeX.

---

## File Structure

- Create `kga/integrations/__init__.py`: public optional-integration exports.
- Create `kga/integrations/elara.py`: evaluation modes, frozen linear benefit estimator, ELARA candidate construction, KGA decisions, post-decision evaluation, and JSON records.
- Create `kga/integrations/claims.py`: deterministic research-promotion guard.
- Create `tests/test_kga_elara_integration.py`: mode separation, no-label leakage, decisions, compatibility, and promotion tests.
- Create `tests/test_kga_elara_runner.py`: protocol, dry-run, full synthetic run, artifact, and launcher tests.
- Create `src/scripts/kbound/run_kga_elara_integration.py`: canonical protocol runner and artifact emitter.
- Create `research_lock/KGA_ELARA_INTEGRATION_v1.yaml`: honest retrospective protocol over the five already-opened multimodal tracks.
- Modify `src/uais/kbound/multimodal_guard.py`: retain legacy imports while delegating decisions to the canonical KGA integration and labeling full-target use retrospective.
- Modify `docs/research/kbound/scripts/kbtrain.sh`: add full and dry-run commands.
- Modify `kga/README.md`: document the optional ELARA integration and mode boundaries.
- Modify `docs/research/kbound/kbound_short_appendix.tex`: add a non-headline ELARA-U instantiation subsection and generated table.
- Generate `experiments/kbound/results/kga_elara_integrated_v1/{results.json,results_table.tex,FINDINGS.md,run_manifest.json}`.
- Rebuild `docs/research/kbound/kbound_short.pdf`.

The shared worktree is already dirty and several paper files contain user changes. Do not make broad commits or stage unrelated content. Use scoped diffs and leave commit creation to an explicit user request.

### Task 1: Canonical Optional KGA-ELARA API

**Files:**
- Create: `tests/test_kga_elara_integration.py`
- Create: `kga/integrations/__init__.py`
- Create: `kga/integrations/elara.py`

- [ ] **Step 1: Write failing mode and decision tests**

Create tests that import the desired API and exercise real arrays:

```python
from __future__ import annotations

import numpy as np
import pytest

from kga import Decision
from kga.integrations.elara import (
    ELARAKGAGuard,
    EvaluationMode,
    FrozenLinearBenefitEstimator,
)


def synthetic_scores(seed: int = 7):
    rng = np.random.default_rng(seed)
    y_val = np.array([0, 1] * 40)
    y_test = np.array([0, 1] * 50)
    s_val = np.column_stack([
        np.clip(0.15 + 0.70 * y_val + rng.normal(0, 0.04, y_val.size), 0, 1),
        np.clip(0.25 + 0.50 * y_val + rng.normal(0, 0.10, y_val.size), 0, 1),
    ])
    s_test = np.column_stack([
        np.clip(0.15 + 0.70 * y_test + rng.normal(0, 0.04, y_test.size), 0, 1),
        np.clip(0.25 + 0.50 * y_test + rng.normal(0, 0.10, y_test.size), 0, 1),
    ])
    return s_val, y_val, s_test, y_test


def test_label_free_rejects_target_labels():
    s_val, y_val, s_test, y_test = synthetic_scores()
    guard = ELARAKGAGuard(alpha=0.1)
    with pytest.raises(ValueError, match="must not receive y_test"):
        guard.decide(s_val=s_val, y_val=y_val, s_test=s_test,
                     mode=EvaluationMode.LABEL_FREE, y_test=y_test)


def test_label_free_fails_closed_without_frozen_estimator():
    s_val, y_val, s_test, _ = synthetic_scores()
    guard = ELARAKGAGuard(alpha=0.1)
    with pytest.raises(ValueError, match="frozen estimator"):
        guard.decide(s_val=s_val, y_val=y_val, s_test=s_test,
                     mode=EvaluationMode.LABEL_FREE)


def test_retrospective_is_never_claim_eligible():
    s_val, y_val, s_test, y_test = synthetic_scores()
    result = ELARAKGAGuard(alpha=0.1).decide(
        s_val=s_val, y_val=y_val, s_test=s_test, y_test=y_test,
        mode=EvaluationMode.RETROSPECTIVE_AUDIT,
    )
    assert result.claim_tier == "retrospective_only"
    assert result.claim_eligible is False
    assert result.labels_used_for_decision == len(y_test)


def test_target_label_light_ignores_nonprobe_labels():
    s_val, y_val, s_test, y_test = synthetic_scores()
    probe = np.arange(20)
    guard = ELARAKGAGuard(alpha=0.1, probe_seed=3)
    first = guard.decide(s_val=s_val, y_val=y_val, s_test=s_test, y_test=y_test,
                         mode=EvaluationMode.TARGET_LABEL_LIGHT, probe_indices=probe)
    permuted = y_test.copy()
    permuted[20:] = permuted[20:][::-1]
    second = guard.decide(s_val=s_val, y_val=y_val, s_test=s_test, y_test=permuted,
                          mode=EvaluationMode.TARGET_LABEL_LIGHT, probe_indices=probe)
    assert first.decision == second.decision
    assert first.certificate == second.certificate
    assert first.labels_used_for_decision == 20


def test_label_free_uses_frozen_estimator_without_labels():
    s_val, y_val, s_test, _ = synthetic_scores()
    estimator = FrozenLinearBenefitEstimator(
        feature_names=("best_val_auc",),
        weights=np.array([0.0]),
        intercept=0.40,
        residuals=np.full(200, 0.01),
        protocol_hash="calibration-protocol-sha256",
    )
    result = ELARAKGAGuard(alpha=0.1).decide(
        s_val=s_val, y_val=y_val, s_test=s_test,
        mode=EvaluationMode.LABEL_FREE, estimator=estimator,
    )
    assert result.decision == Decision.ADAPT
    assert result.labels_used_for_decision == 0
    assert result.claim_tier == "label_free_candidate"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_kga_elara_integration.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'kga.integrations'`.

- [ ] **Step 3: Implement the minimal canonical integration**

Create `kga/integrations/elara.py` with these public types and behavior:

```python
class EvaluationMode(str, Enum):
    RETROSPECTIVE_AUDIT = "retrospective_audit"
    TARGET_LABEL_LIGHT = "target_label_light"
    LABEL_FREE = "label_free"


@dataclass(frozen=True)
class FrozenLinearBenefitEstimator:
    feature_names: tuple[str, ...]
    weights: np.ndarray
    intercept: float
    residuals: np.ndarray
    protocol_hash: str

    def __post_init__(self) -> None:
        if not self.protocol_hash:
            raise ValueError("protocol_hash is required")
        if len(self.feature_names) != np.asarray(self.weights).size:
            raise ValueError("feature_names and weights must have equal length")
        if np.asarray(self.residuals).size == 0:
            raise ValueError("residuals must be non-empty")

    def predict(self, features: Mapping[str, float]) -> float:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"missing estimator features: {missing}")
        x = np.array([features[name] for name in self.feature_names], dtype=float)
        return float(self.intercept + x @ np.asarray(self.weights, dtype=float))


@dataclass
class ELARAKGAResult:
    mode: EvaluationMode
    decision: Decision
    deployed_action: str
    router_action: str
    frozen_expert: int
    frozen_scores: np.ndarray
    candidate_scores: np.ndarray
    deployed_scores: np.ndarray
    certificate: dict[str, float | int | str]
    evidence: dict[str, float]
    labels_used_for_decision: int
    claim_tier: str
    claim_eligible: bool
    claim_reasons: tuple[str, ...]

    def to_record(self) -> dict:
        return {
            "mode": self.mode.value,
            "decision": self.decision.value,
            "deployed_action": self.deployed_action,
            "router_action": self.router_action,
            "frozen_expert": self.frozen_expert,
            "certificate": self.certificate,
            "evidence": self.evidence,
            "labels_used_for_decision": self.labels_used_for_decision,
            "claim_tier": self.claim_tier,
            "claim_eligible": self.claim_eligible,
            "claim_reasons": list(self.claim_reasons),
        }
```

Implement `ELARAKGAGuard.decide` with this exact signature:

```python
def decide(
    self,
    *,
    s_val: np.ndarray,
    y_val: np.ndarray,
    s_test: np.ndarray,
    mode: EvaluationMode | str,
    y_test: np.ndarray | None = None,
    probe_indices: np.ndarray | None = None,
    estimator: FrozenLinearBenefitEstimator | None = None,
) -> ELARAKGAResult:
```

Its body must validate arrays, use ELARA-U validation reliability to select the
frozen expert, call `route(s_val, y_val, s_test, self.policy,
action=self.router_action)` for the candidate, and compute KGA evidence only from
`s_val` and `s_test`. `RETROSPECTIVE_AUDIT` certifies full-target per-example
Brier benefits. `TARGET_LABEL_LIGHT` certifies only Brier benefits at
`probe_indices`. `LABEL_FREE` rejects `y_test` and `probe_indices`, requires the
frozen estimator, predicts `delta_hat` from label-free features, and certifies it
with the frozen residuals. ADAPT deploys the ELARA candidate; FREEZE and ABSTAIN
deploy the frozen expert.

Use bounded per-example Brier benefit
`(frozen_score - y)^2 - (candidate_score - y)^2`, whose range is at most `2.0`.
Do not use placement/AUROC benefits for target-label-light decisions because they
would require labels outside the probe. Import `uais.elara_u.router` lazily inside
the integration so ordinary `import kga` stays independent of ELARA-U.

Create `kga/integrations/__init__.py` exporting `EvaluationMode`,
`FrozenLinearBenefitEstimator`, `ELARAKGAGuard`, `ELARAKGAResult`, and
`evaluate_result`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_kga_elara_integration.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Run existing KGA package tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_kga_package.py -q
```

Expected: existing KGA tests pass unchanged, proving the optional integration did not change the core API.

### Task 2: Promotion Guard and Legacy Compatibility

**Files:**
- Modify: `tests/test_kga_elara_integration.py`
- Create: `kga/integrations/claims.py`
- Modify: `kga/integrations/__init__.py`
- Modify: `src/uais/kbound/multimodal_guard.py`

- [ ] **Step 1: Add failing promotion-guard tests**

Append:

```python
from kga.integrations.claims import assess_promotion


def eligible_summary():
    return {
        "mode": "label_free",
        "frozen_estimator_verified": True,
        "held_out_natural_datasets": 2,
        "frozen_before_scoring": True,
        "independent_splits": 3,
        "regret_kga": 0.01,
        "regret_always_adapt": 0.03,
        "regret_always_freeze": 0.02,
        "false_adapt_rate": 0.05,
        "alpha": 0.10,
        "coverage": 0.30,
        "confidence_intervals_complete": True,
        "strong_baselines_complete": True,
        "required_tracks_complete": True,
        "integrity_failures": [],
    }


def test_promotion_guard_accepts_only_complete_heldout_evidence():
    verdict = assess_promotion(eligible_summary())
    assert verdict["eligible"] is True
    assert verdict["reasons"] == []


@pytest.mark.parametrize("field,value", [
    ("mode", "retrospective_audit"),
    ("held_out_natural_datasets", 1),
    ("independent_splits", 2),
    ("coverage", 0.19),
    ("false_adapt_rate", 0.11),
    ("required_tracks_complete", False),
])
def test_promotion_guard_rejects_each_missing_requirement(field, value):
    summary = eligible_summary()
    summary[field] = value
    verdict = assess_promotion(summary)
    assert verdict["eligible"] is False
    assert verdict["reasons"]
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_kga_elara_integration.py -q
```

Expected: collection fails because `kga.integrations.claims` does not exist.

- [ ] **Step 3: Implement deterministic promotion assessment**

Create `assess_promotion(summary: Mapping[str, object]) -> dict[str, object]`.
It must append stable reason codes for every failed requirement:

```python
requirements = (
    (mode in {"label_free", "target_label_light"}, "invalid_evaluation_mode"),
    (mode != "label_free" or frozen_estimator_verified, "unverified_frozen_estimator"),
    (held_out_natural_datasets >= 2, "fewer_than_two_heldout_natural_datasets"),
    (frozen_before_scoring, "configuration_not_frozen_before_scoring"),
    (independent_splits >= 3, "fewer_than_three_independent_splits"),
    (regret_kga < regret_always_adapt, "does_not_beat_always_adapt_regret"),
    (regret_kga < regret_always_freeze, "does_not_beat_always_freeze_regret"),
    (false_adapt_rate <= alpha, "false_adapt_exceeds_alpha"),
    (coverage >= 0.20, "coverage_below_20_percent"),
    (confidence_intervals_complete, "confidence_intervals_incomplete"),
    (strong_baselines_complete, "strong_baselines_incomplete"),
    (required_tracks_complete, "required_tracks_incomplete"),
    (not integrity_failures, "integrity_failure"),
)
```

Return `{"eligible": not reasons, "reasons": reasons}` and export it from
`kga.integrations`.

- [ ] **Step 4: Convert the old guard to a compatibility facade**

Preserve existing names `MultimodalGuard`, `GuardResult`, `placement_benefits`,
`cw_fuse`, `relgate_fuse`, `auroc`, and `load_track_cache`. Delegate
`guard_category` to `ELARAKGAGuard` using:

- `EvaluationMode.TARGET_LABEL_LIGHT` when `probe_k > 0`, with deterministic
  probe indices drawn once from the target examples;
- `EvaluationMode.RETROSPECTIVE_AUDIT` when no probe is declared;
- never call the retrospective path label-free in code, output keys, or docs.

Map ADAPT to candidate scores and FREEZE/ABSTAIN to frozen scores. Keep the old
return shape so existing experiment scripts continue to import successfully.

- [ ] **Step 5: Run integration and compatibility tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_kga_elara_integration.py tests/test_target_label_light_probe.py -q
```

Expected: all tests pass.

### Task 3: Locked Protocol, Canonical Runner, and Artifacts

**Files:**
- Create: `tests/test_kga_elara_runner.py`
- Create: `research_lock/KGA_ELARA_INTEGRATION_v1.yaml`
- Create: `src/scripts/kbound/run_kga_elara_integration.py`
- Modify: `docs/research/kbound/scripts/kbtrain.sh`

- [ ] **Step 1: Write failing runner and launcher tests**

Create a synthetic two-expert `.npz` cache and a temporary YAML protocol, invoke
the runner through its `main(argv)` function, then assert:

```python
assert (out / "results.json").exists()
assert (out / "results_table.tex").exists()
assert (out / "FINDINGS.md").exists()
assert (out / "run_manifest.json").exists()
payload = json.loads((out / "results.json").read_text())
assert payload["schema"] == "kga_elara_integrated_results_v1"
assert payload["mode"] == "retrospective_audit"
assert payload["claim_eligibility"]["eligible"] is False
assert payload["tracks"][0]["n_valid_categories"] == 1
```

Add a dry-run test asserting no scored `results.json` is written and the returned
summary reports every declared track. Add a source-level launcher test asserting
`kga-elara-integrated` and `kga-elara-integrated-dry-run` both appear in
`kbtrain.sh`.

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_kga_elara_runner.py -q
```

Expected: import fails because `run_kga_elara_integration.py` does not exist.

- [ ] **Step 3: Add the honest retrospective protocol**

Create `research_lock/KGA_ELARA_INTEGRATION_v1.yaml` with:

```yaml
schema: kga_elara_integration_protocol_v1
date_declared: 2026-06-23
status: RETROSPECTIVE_OPENED_DATA
mode: retrospective_audit
alpha: 0.10
router_action: hybrid
abstain_fallback: freeze
output_dir: experiments/kbound/results/kga_elara_integrated_v1
claim_scope: retrospective_multimodal_instantiation_not_headline
tracks:
  - name: 3D-ADAM
    cache: experiments/fusion/3d_adam_score_cache
    pattern: "*.npz"
  - name: Real-IAD-D3
    cache: experiments/fusion/realiad_d3_score_cache
    pattern: "*_v2_binpcd.npz"
  - name: MulSen-AD
    cache: experiments/fusion/mulsen_score_cache
    pattern: "*.npz"
  - name: MVTec-3D
    cache: experiments/fusion/mvtec3d_score_cache
    pattern: "*.npz"
  - name: Real-IAD-NatDeg
    cache: experiments/fusion/realiad_natdeg_score_cache
    pattern: "*.npz"
integrity:
  data_previously_opened: true
  frozen_before_target_scoring: false
  label_free_claim_allowed: false
  headline_claim_allowed: false
```

- [ ] **Step 4: Implement the runner and emitters**

The runner must:

1. Parse YAML and validate schema, mode, alpha, tracks, and output path.
2. Hash the complete protocol bytes with SHA-256.
3. In dry-run mode, list matched/missing files and exit without scored artifacts.
4. Load `Sval`, `yval`, `Stest`, and `ytest` from each category cache.
5. Call `ELARAKGAGuard.decide` in the protocol mode.
6. Call `evaluate_result` only after the decision is frozen.
7. Aggregate mean AUROC, regret, coverage, and unconditional/conditional false-adapt.
8. Call `assess_promotion`, which must reject this retrospective protocol.
9. Refuse to overwrite an existing result whose manifest has a different protocol hash unless `--overwrite` is supplied.
10. Write deterministic JSON using sorted keys and indentation.
11. Generate a compact LaTeX table with track, categories, freeze AUROC, ELARA AUROC, KGA AUROC, decisions, mode, and claim status.
12. Generate findings that begin with `RETROSPECTIVE - NOT A LABEL-FREE OR HEADLINE CLAIM`.
13. Record protocol hash, Git revision when available, Python version, package versions, input paths, and input SHA-256 hashes in the manifest.

Expose these functions with the listed signatures; their bodies implement the
thirteen requirements above and return exit code zero only after all requested
artifacts are written:

```python
def run_protocol(protocol_path: Path, output_dir: Path | None = None,
                 *, dry_run: bool = False, overwrite: bool = False) -> dict:

def main(argv: Sequence[str] | None = None) -> int:
```

- [ ] **Step 5: Wire the one-command launcher**

Add variables and cases to `kbtrain.sh`:

```bash
KGA_ELARA=src/scripts/kbound/run_kga_elara_integration.py

kga-elara-integrated)
  "$VENV/bin/python" "$KGA_ELARA" \
    --protocol research_lock/KGA_ELARA_INTEGRATION_v1.yaml ;;
kga-elara-integrated-dry-run)
  "$VENV/bin/python" "$KGA_ELARA" \
    --protocol research_lock/KGA_ELARA_INTEGRATION_v1.yaml --dry-run ;;
```

Include both names in the usage line.

- [ ] **Step 6: Run runner tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_kga_elara_runner.py -q
```

Expected: all tests pass.

### Task 4: Execute the Real Retrospective Audit and Connect the Paper

**Files:**
- Generate: `experiments/kbound/results/kga_elara_integrated_v1/results.json`
- Generate: `experiments/kbound/results/kga_elara_integrated_v1/results_table.tex`
- Generate: `experiments/kbound/results/kga_elara_integrated_v1/FINDINGS.md`
- Generate: `experiments/kbound/results/kga_elara_integrated_v1/run_manifest.json`
- Modify: `kga/README.md`
- Modify: `docs/research/kbound/kbound_short_appendix.tex`
- Rebuild: `docs/research/kbound/kbound_short.pdf`

- [ ] **Step 1: Run dry-run inventory**

Run:

```bash
bash docs/research/kbound/scripts/kbtrain.sh kga-elara-integrated-dry-run
```

Expected: five declared tracks, 85 matched category cache files in the current workspace, no scored result rewrite.

- [ ] **Step 2: Run the canonical retrospective audit**

Run:

```bash
bash docs/research/kbound/scripts/kbtrain.sh kga-elara-integrated
```

Expected: four versioned artifacts are written and `claim_eligibility.eligible` is `false` with retrospective/opened-data reasons.

- [ ] **Step 3: Add public API documentation**

Document the three complete method signatures in `kga/README.md`, including the
required `FrozenLinearBenefitEstimator` constructor fields for label-free mode
and a target-label-light example that passes fixed `probe_indices`. State
explicitly that ELARA proposes a candidate, KGA certifies deployment, the
core KGA API does not require ELARA, and current real-cache results are
retrospective rather than label-free.

- [ ] **Step 4: Add a non-headline short-paper bridge**

In `kbound_short_appendix.tex`, replace the unexplained two-sentence anomaly
routing subsection with a subsection titled `ELARA-U as a KGA instantiation
(retrospective audit)`. Explain:

- K-Bound is the framework, KGA the decision method, and ELARA-U the candidate router;
- validation labels select/fit ELARA while deployment labels are unavailable in label-free mode;
- the current five-track cache audit is retrospective and claim-ineligible;
- the exact values below are generated from the canonical result artifact.

Input the generated table with:

```tex
\IfFileExists{../../../experiments/kbound/results/kga_elara_integrated_v1/results_table.tex}{%
  \input{../../../experiments/kbound/results/kga_elara_integrated_v1/results_table.tex}%
}{%
  \noindent\emph{Integrated retrospective table unavailable in this build.}%
}
```

Do not alter the abstract, theorem statements, six-dataset headline panel, or
headline status language.

- [ ] **Step 5: Rebuild and render the short PDF**

Run:

```bash
cd docs/research/kbound
latexmk -pdf -interaction=nonstopmode -halt-on-error kbound_short.tex
```

Render the page containing `ELARA-U as a KGA instantiation` to PNG with a white
background and inspect it for clipping, overlap, tiny text, and broken references.

### Task 5: Final Verification

**Files:** all files above.

- [ ] **Step 1: Run focused and regression tests**

```bash
.venv/bin/python -m pytest \
  tests/test_kga_package.py \
  tests/test_kga_elara_integration.py \
  tests/test_kga_elara_runner.py \
  tests/test_target_label_light_probe.py \
  tests/elara_u -q
```

Expected: zero failures.

- [ ] **Step 2: Run lint on changed Python files**

```bash
.venv/bin/python -m ruff check \
  kga/integrations \
  src/scripts/kbound/run_kga_elara_integration.py \
  src/uais/kbound/multimodal_guard.py \
  tests/test_kga_elara_integration.py \
  tests/test_kga_elara_runner.py
```

Expected: zero errors.

- [ ] **Step 3: Verify artifact integrity and claims**

Run a Python assertion that checks:

```python
assert results["schema"] == "kga_elara_integrated_results_v1"
assert results["mode"] == "retrospective_audit"
assert results["claim_eligibility"]["eligible"] is False
assert results["aggregate"]["labels_used_for_decision"] > 0
assert len(results["tracks"]) == 5
assert not results["missing_tracks"]
```

- [ ] **Step 4: Verify the compiled PDF text**

Use `pypdf` to assert the latest PDF contains:

```text
ELARA-U as a KGA instantiation
retrospective audit
not a label-free or headline claim
```

Confirm visually that the rendered page is readable.

- [ ] **Step 5: Inspect the scoped diff**

Run:

```bash
git diff --check
git status --short
```

Review only integration-related paths. Do not revert, stage, or commit unrelated user changes.
