"""Deterministic per-dataset smoke (all 9 runner configs): the adapter loads,
adapts, emits telemetry, KGA consumes the candidate, and runs are reproducible."""
import pytest

from experiments.kbound.elara_opt import run_elara_candidate, load_meta_gate, ELARA_MODES
from experiments.kbound.elara_opt.smoke_models import DATASET_CONFIGS, DATASET_IDS, build_f0, synth_cell

_DECISIONS = {"Decision.ADAPT", "Decision.FREEZE", "Decision.ABSTAIN"}
_META = load_meta_gate()


@pytest.mark.parametrize("dataset", DATASET_IDS)
def test_dataset_smoke_all_modes(dataset):
    dc = DATASET_CONFIGS[dataset]
    nc, hw, in_ch = dc["num_classes"], dc["hw"], dc["in_ch"]
    f0 = build_f0(nc, in_ch, seed=0)
    stream, eval_x, dev_y = synth_cell(nc, 8, in_ch, hw, seed=0)
    for mode in ELARA_MODES:
        mm = _META if mode == "elara_meta" else None
        if mode == "elara_meta" and mm is None:
            continue
        res = run_elara_candidate(f0, stream, eval_x, dev_y, nc, mode,
                                  steps=1, lr=1e-3, meta_model=mm, seed=0)
        assert res["kga_decision"] in _DECISIONS
        assert len(res["telemetry"]["steps"]) >= 1
        assert len(res["Z"]) == 11


@pytest.mark.parametrize("dataset", DATASET_IDS)
def test_dataset_determinism(dataset):
    dc = DATASET_CONFIGS[dataset]
    nc, hw, in_ch = dc["num_classes"], dc["hw"], dc["in_ch"]
    f0 = build_f0(nc, in_ch, seed=0)
    stream, eval_x, dev_y = synth_cell(nc, 8, in_ch, hw, seed=0)
    r1 = run_elara_candidate(f0, stream, eval_x, dev_y, nc, "elara_uniform", steps=1, lr=1e-3, seed=0)
    r2 = run_elara_candidate(f0, stream, eval_x, dev_y, nc, "elara_uniform", steps=1, lr=1e-3, seed=0)
    assert r1["candidate_hash"] == r2["candidate_hash"]
