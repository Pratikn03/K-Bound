"""Integration test: one adaptation batch end-to-end, all three modes, plus the
KGA-consumes-the-candidate path and code<->lock consistency."""
import copy
import os

import pytest
import torch

from experiments.kbound.elara_opt import (
    ELARAOptAdapter, elara_opt_adapt, run_elara_candidate, load_meta_gate,
    ELARA_OPT_DEFAULTS, ELARA_MODES,
)
from experiments.kbound.elara_opt._compat import _bn_affine_params
from experiments.kbound.elara_opt.smoke_models import build_f0, synth_cell

_DECISIONS = {"Decision.ADAPT", "Decision.FREEZE", "Decision.ABSTAIN"}


def test_one_batch_moves_affine_params_and_emits_telemetry():
    f0 = build_f0(10, 3, seed=0)
    stream, _, _ = synth_cell(10, 16, 3, 32, seed=0)
    before = [p.detach().clone() for p in _bn_affine_params(copy.deepcopy(f0))]
    m, upd, tele = elara_opt_adapt(f0, stream, 1, 1e-3, 10, mode="elara_rule", seed=0)
    after = _bn_affine_params(m)
    assert upd > 0.0                                  # genuine parameter movement
    moved = any(not torch.equal(a, b) for a, b in zip(after, before))
    assert moved
    s = tele["steps"][0]
    for key in ("gate_weights", "trust_radius", "grad_conflict_min_cos",
                "update_norm", "reliability_features"):
        assert key in s
    assert abs(sum(s["gate_weights"]) - 1.0) < 1e-6   # nonneg weights sum to 1
    assert tele["summary"]["candidate_hash"]


@pytest.mark.parametrize("mode", ELARA_MODES)
def test_kga_consumes_candidate_each_mode(mode):
    f0 = build_f0(10, 3, seed=0)
    stream, eval_x, dev_y = synth_cell(10, 16, 3, 32, seed=0)
    mm = load_meta_gate() if mode == "elara_meta" else None
    if mode == "elara_meta" and mm is None:
        pytest.skip("meta checkpoint not present")
    res = run_elara_candidate(f0, stream, eval_x, dev_y, 10, mode,
                              steps=1, lr=1e-3, meta_model=mm, seed=0)
    assert res["kga_decision"] in _DECISIONS
    assert len(res["Z"]) == 11
    assert res["kga_epsilon"] >= 0.0


def test_adapter_as_method_signature_matches_repo_contract():
    adapter = ELARAOptAdapter(mode="elara_uniform", seed=0)
    method = adapter.as_method(num_classes=10)        # (base, stream, steps, lr) -> (model, upd)
    f0 = build_f0(10, 3, seed=0)
    stream, _, _ = synth_cell(10, 8, 3, 32, seed=0)
    m, upd = method(f0, stream, 1, 1e-3)
    assert isinstance(upd, float) and upd >= 0.0
    assert isinstance(m, torch.nn.Module)


def test_code_matches_lock_if_present():
    """If the lock file exists, its frozen hyperparameters must equal the code."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    lock = os.path.join(repo, "research_lock", "elara_opt_protocol_v1.yaml")
    if not os.path.exists(lock):
        pytest.skip("lock file not yet written")
    yaml = pytest.importorskip("yaml")
    with open(lock) as fh:
        L = yaml.safe_load(fh)
    hp = L["hyperparameters"]
    d = ELARA_OPT_DEFAULTS
    assert hp["seed"] == d["seed"]
    assert float(hp["lr"]) == float(d["lr"])
    assert hp["steps"] == d["steps"]
    assert float(hp["margin_frac"]) == float(d["margin_frac"])
    assert float(hp["lambda_anchor"]) == float(d["lambda_anchor"])
    assert float(hp["trust_region"]["r_min"]) == float(d["trust_region"]["r_min"])
    assert float(hp["trust_region"]["r_max"]) == float(d["trust_region"]["r_max"])
    assert list(hp["modes"]) == list(d["modes"])
