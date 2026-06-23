"""test_log_integrity -- every window record carries the required audit fields.

Required on EVERY record: model_version, config_hash, decision, latency_ms
(plus window_id, schema_version, the benefit interval, and the 14-feature
evidence block).
"""

import numpy as np

from kbound_edge.logging import WindowLogger, read_jsonl
from kbound_edge.policy import kga_decide
from kbound_edge.evidence import edge_evidence_vector, EDGE_EVIDENCE_NAMES

REQUIRED = (
    "schema_version", "timestamp", "window_id",
    "model_version", "config_hash",
    "decision", "bhat", "eps", "lower", "upper", "reason",
    "latency_ms", "evidence",
)


def test_every_record_has_required_fields(tmp_path):
    path = str(tmp_path / "windows.jsonl")
    rng = np.random.default_rng(0)
    with WindowLogger(path, model_version="mv-abc123", config_hash="cfg-def456") as lg:
        for wid in range(6):
            p0 = rng.dirichlet(np.ones(4), size=8)
            pa = rng.dirichlet(np.ones(4), size=8)
            z = edge_evidence_vector(p0, pa, upd_norm=0.1)
            zdict = {n: float(v) for n, v in zip(EDGE_EVIDENCE_NAMES, z)}
            decision = kga_decide(0.3 * (wid - 3), 0.15)
            lg.log(wid, decision.as_dict(), zdict, latency_ms=10.0 + wid,
                   frozen_pred=p0.argmax(1).tolist())

    records = read_jsonl(path)
    assert len(records) == 6
    for r in records:
        for field in REQUIRED:
            assert field in r, f"missing required field: {field}"
        assert r["model_version"] == "mv-abc123"
        assert r["config_hash"] == "cfg-def456"
        assert r["decision"] in ("adapt", "freeze", "abstain")
        assert isinstance(r["latency_ms"], (int, float))
        assert len(r["evidence"]) == 14


def test_window_ids_are_sequential_and_unique(tmp_path):
    path = str(tmp_path / "windows.jsonl")
    with WindowLogger(path, model_version="v", config_hash="c") as lg:
        for wid in range(4):
            d = kga_decide(0.1, 0.2).as_dict()
            lg.log(wid, d, {"pre_entropy": 0.0}, latency_ms=1.0)
    ids = [r["window_id"] for r in read_jsonl(path)]
    assert ids == [0, 1, 2, 3]
    assert len(set(ids)) == len(ids)
