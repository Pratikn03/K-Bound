"""Lifecycle regressions: tampered evidence must never promote a candidate."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from kga.deployment_audit import DeploymentContract, GateSession, sha256_bytes

SOURCE = b"immutable source checkpoint"
CANDIDATE = b"candidate checkpoint"


def make_contract(**overrides):
    values = {
        "source_sha256": hashlib.sha256(SOURCE).hexdigest(),
        "adapter_sha256": "a" * 64,
        "estimator_sha256": "b" * 64,
        "calibration_sha256": "c" * 64,
        "schema_sha256": "d" * 64,
        "valid_from": 100.0,
        "expires_at": 200.0,
        "alpha": 0.1,
        "max_assessments": 20,
    }
    values.update(overrides)
    return DeploymentContract(**values)


def evidence(contract, candidate, batch, **changes):
    value = {
        "source_sha256": contract.source_sha256,
        "candidate_sha256": sha256_bytes(candidate),
        "batch_sha256": sha256_bytes(batch),
        "adapter_sha256": contract.adapter_sha256,
        "estimator_sha256": contract.estimator_sha256,
        "calibration_sha256": contract.calibration_sha256,
        "schema_sha256": contract.schema_sha256,
        "prediction": 0.2,
        "radius": 0.05,
        "alpha": 0.1,
        "target": "declared_cell_benefit",
    }
    value.update(changes)
    return value


def execute(session, rid="r1", batch=b"batch", **changes):
    return session.assess(
        rid,
        batch,
        lambda source, inputs: CANDIDATE,
        lambda candidate, inputs: evidence(session.contract, candidate, inputs, **changes),
        now=150.0,
    )


def test_candidate_is_request_scoped_and_source_survives_success(tmp_path):
    s = GateSession(make_contract(), SOURCE, journal=tmp_path / "journal.jsonl")
    r = execute(s)
    assert r.action == "ADAPT"
    assert s.selected_model(r) == CANDIDATE
    assert s.source_model == SOURCE
    r2 = execute(s, "freeze", prediction=-0.2)
    assert r2.action == "FREEZE"
    assert s.selected_model(r2) == SOURCE
    assert len((tmp_path / "journal.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize(
    "field",
    [
        "source_sha256",
        "candidate_sha256",
        "batch_sha256",
        "adapter_sha256",
        "estimator_sha256",
        "calibration_sha256",
        "schema_sha256",
    ],
)
def test_any_identity_mismatch_prevents_candidate_selection(field):
    s = GateSession(make_contract(), SOURCE)
    r = execute(s, **{field: "f" * 64})
    assert r.action == "ABSTAIN"
    assert s.selected_model(r) == SOURCE
    assert r.reason == "evidence_identity_mismatch"


@pytest.mark.parametrize(
    "change",
    [
        {"prediction": float("nan")},
        {"radius": -0.1},
        {"radius": float("nan")},
        {"alpha": 0.2},
        {"prediction": float("inf")},
        {"realized_benefit": 1.0},
        {"prediction": True},
        {"target": "population_benefit"},
    ],
)
def test_malformed_or_outcome_bearing_evidence_fails_closed(change):
    s = GateSession(make_contract(), SOURCE)
    r = execute(s, **change)
    assert r.action == "ABSTAIN"
    assert s.selected_model(r) == SOURCE
    json.dumps(r.to_dict(), allow_nan=False)


@pytest.mark.parametrize("now", [99.0, 200.0, float("nan")])
def test_expired_or_not_yet_valid_contract_never_calls_candidate(now):
    s = GateSession(make_contract(), SOURCE)

    def should_not_run(*args):
        raise AssertionError("expired path attempted candidate generation")

    r = s.assess("expired", b"batch", should_not_run, should_not_run, now=now)
    assert r.action == "ABSTAIN"
    assert r.reason == "calibration_outside_validity_window"


def test_failure_preserves_source_and_records_failure_type():
    s = GateSession(make_contract(), SOURCE)

    def bad_update(source, batch):
        raise RuntimeError("candidate generation failed")

    r = s.assess("failed", b"batch", bad_update, lambda *_: {}, now=150.0)
    assert r.action == "ABSTAIN"
    assert r.reason == "candidate_failure:RuntimeError"
    assert s.selected_model(r) == SOURCE


def test_duplicate_requests_are_idempotent_and_conflicting_ids_rejected():
    s = GateSession(make_contract(max_assessments=1), SOURCE)
    r = execute(s)
    assert execute(s).receipt_sha256 == r.receipt_sha256
    assert s.assessment_count == 1
    assert execute(s, batch=b"different").reason == "request_id_conflict"
    assert execute(s, "r2").reason == "assessment_limit_reached"


def test_concurrent_duplicate_requests_do_not_spend_two_assessments():
    s = GateSession(make_contract(), SOURCE)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: execute(s), range(12)))
    assert len({r.receipt_sha256 for r in receipts}) == 1
    assert s.assessment_count == 1


def test_expiry_cannot_be_bypassed_by_cached_adapt():
    s = GateSession(make_contract(), SOURCE)
    execute(s)
    r = s.assess("r1", b"batch", lambda *_: CANDIDATE, lambda *_: {}, now=201.0)
    assert r.action == "ABSTAIN"
    assert s.selected_model(r) == SOURCE


def test_infinite_radius_and_zero_endpoint_abstain():
    s = GateSession(make_contract(), SOURCE)
    for i, change in enumerate([{"radius": float("inf")}, {"prediction": 0.05}, {"prediction": -0.05}]):
        r = execute(s, str(i), **change)
        assert r.action == "ABSTAIN"
        json.dumps(r.to_dict(), allow_nan=False)


def test_forged_receipt_cannot_select_a_candidate():
    s = GateSession(make_contract(), SOURCE)
    r = execute(s, prediction=-0.2)
    with pytest.raises(ValueError):
        s.selected_model(replace(r, action="ADAPT"))


def test_journal_refuses_historical_overwrite_and_tampered_source(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text("old authority\n")
    with pytest.raises(FileExistsError):
        GateSession(make_contract(), SOURCE, journal=path)
    with pytest.raises(ValueError):
        GateSession(make_contract(), b"tampered")


def test_unwritable_journal_prevents_successful_promotion(tmp_path):
    path = tmp_path / "journal.jsonl"
    s = GateSession(make_contract(), SOURCE, journal=path)
    path.unlink()
    path.mkdir()
    receipt = execute(s)
    assert receipt.action == "ABSTAIN"
    assert receipt.reason == "journal_write_failure"
    assert s.selected_model(receipt) == SOURCE


def test_slow_candidate_cannot_outlive_calibration(monkeypatch):
    import kga.deployment_audit as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    session = GateSession(make_contract(), SOURCE)

    def slow_candidate(source, batch):
        clock[0] += 51.0
        return CANDIDATE

    receipt = session.assess(
        "slow",
        b"batch",
        slow_candidate,
        lambda model, batch: evidence(session.contract, model, batch),
        now=150.0,
    )
    assert receipt.action == "ABSTAIN"
    assert session.selected_model(receipt) == SOURCE


def test_selected_candidate_expires_even_without_another_assessment(monkeypatch):
    import kga.deployment_audit as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    session = GateSession(make_contract(), SOURCE)
    receipt = execute(session)
    assert session.selected_model(receipt) == CANDIDATE
    clock[0] += 51.0
    assert session.selected_model(receipt) == SOURCE


def test_nonscalar_identity_fails_closed_instead_of_raising():
    import numpy as np

    session = GateSession(make_contract(), SOURCE)
    receipt = execute(session, calibration_sha256=np.array(["c" * 64, "a" * 64]))
    assert receipt.action == "ABSTAIN"
    assert session.selected_model(receipt) == SOURCE


def test_partial_journal_failure_latches_session_closed(monkeypatch, tmp_path):
    import kga.deployment_audit as module

    session = GateSession(make_contract(), SOURCE, journal=tmp_path / "journal.jsonl")
    old_fsync = module.os.fsync

    def interrupted_fsync(fd):
        raise OSError("injected failure after write")

    monkeypatch.setattr(module.os, "fsync", interrupted_fsync)
    receipt = execute(session)
    assert receipt.action == "ABSTAIN"
    monkeypatch.setattr(module.os, "fsync", old_fsync)
    second = execute(session, "second")
    assert second.action == "ABSTAIN"
    assert second.reason == "journal_unavailable"
    assert not session.journal_healthy
    assert session.selected_model(second) == SOURCE


@pytest.mark.parametrize("field", ["max_retained_receipts", "max_journal_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_session_capacity_limits_must_be_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        make_contract(**{field: value})


def test_receipt_capacity_preserves_duplicates_then_latches_without_growth(tmp_path):
    import kga.deployment_audit as module

    path = tmp_path / "journal.jsonl"
    session = GateSession(make_contract(max_retained_receipts=2), SOURCE, journal=path)
    first, last = execute(session, "first"), execute(session, "last")
    assert execute(session, "first") == first
    assert not session.capacity_exhausted
    size = path.stat().st_size

    def forbidden(*_):
        pytest.fail("capacity-exhausted session ran a callback")

    for index in range(100):
        with pytest.raises(module.SessionCapacityError, match="retained_receipt_capacity"):
            session.assess(str(index), b"batch", forbidden, forbidden, now=150.0)
    assert session.capacity_exhausted
    assert session.assessment_count == 2
    assert session.retained_receipt_count == len(session._deadlines) == 2
    assert len(session._cache) == 2
    assert path.stat().st_size == session.journal_bytes_reserved == size
    assert session.selected_model(first) == session.selected_model(last) == SOURCE
    with pytest.raises(module.SessionCapacityError):
        execute(session, "first")


def test_rejected_requests_are_also_bounded_without_candidate_assessments():
    import kga.deployment_audit as module

    session = GateSession(make_contract(max_retained_receipts=2), SOURCE)
    for index in range(2):
        receipt = session.assess(str(index), b"batch", None, None, now=250.0)
        assert receipt.action == "ABSTAIN"
    for index in range(100):
        with pytest.raises(module.SessionCapacityError):
            session.assess(str(index), b"batch", None, None, now=250.0)
    assert session.assessment_count == 0
    assert session.retained_receipt_count == len(session._deadlines) == 2
    assert session.journal_bytes_reserved == 0


def _fixed_clock_receipt_size(monkeypatch, tmp_path):
    import kga.deployment_audit as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.0)
    path = tmp_path / "reference.jsonl"
    reference = GateSession(make_contract(), SOURCE, journal=path)
    execute(reference, "one")
    return path.stat().st_size


def test_exact_journal_byte_boundary_allows_retry_but_no_further_writes(monkeypatch, tmp_path):
    import kga.deployment_audit as module

    size = _fixed_clock_receipt_size(monkeypatch, tmp_path)
    path = tmp_path / "bounded.jsonl"
    session = GateSession(make_contract(max_journal_bytes=size), SOURCE, journal=path)
    first = execute(session, "one")
    assert path.stat().st_size == session.journal_bytes_reserved == size
    assert execute(session, "one") == first
    with pytest.raises(module.SessionCapacityError, match="journal_byte_capacity"):
        execute(session, "two")
    assert session.assessment_count == 1
    assert session.selected_model(first) == SOURCE
    assert path.stat().st_size == size


def test_receipt_that_exceeds_remaining_journal_bytes_is_never_appended(monkeypatch, tmp_path):
    import kga.deployment_audit as module

    size = _fixed_clock_receipt_size(monkeypatch, tmp_path)
    # A second equally sized request has a 66-byte predecessor hash in place
    # of the first record's 4-byte null, so the budget is short by one byte.
    path = tmp_path / "bounded.jsonl"
    session = GateSession(make_contract(max_journal_bytes=2 * size + 61), SOURCE, journal=path)
    first = execute(session, "one")
    with pytest.raises(module.SessionCapacityError, match="journal_byte_capacity"):
        execute(session, "two")
    assert session.assessment_count == 2
    assert session.retained_receipt_count == len(session._deadlines) == 1
    assert session.journal_bytes_reserved == path.stat().st_size == size
    assert session.selected_model(first) == SOURCE


def test_too_small_journal_budget_fails_without_publishing_any_receipt(monkeypatch, tmp_path):
    import kga.deployment_audit as module

    size = _fixed_clock_receipt_size(monkeypatch, tmp_path)
    path = tmp_path / "bounded.jsonl"
    session = GateSession(make_contract(max_journal_bytes=size - 1), SOURCE, journal=path)
    with pytest.raises(module.SessionCapacityError, match="journal_byte_capacity"):
        execute(session, "one")
    assert session.capacity_exhausted
    assert session.retained_receipt_count == len(session._deadlines) == 0
    assert session.journal_bytes_reserved == path.stat().st_size == 0


def test_concurrent_requests_cannot_overrun_the_last_receipt_slot():
    import kga.deployment_audit as module

    session = GateSession(make_contract(max_retained_receipts=1), SOURCE)

    def request(index):
        try:
            return execute(session, str(index))
        except module.SessionCapacityError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = [receipt for receipt in pool.map(request, range(20)) if receipt is not None]
    assert len(receipts) == session.retained_receipt_count == session.assessment_count == 1
    assert session.capacity_exhausted
    assert session.selected_model(receipts[0]) == SOURCE


def test_expiry_during_final_journal_write_cannot_overrun_receipt_capacity(monkeypatch, tmp_path):
    import kga.deployment_audit as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.os, "fsync", lambda _: clock.__setitem__(0, 1051.0))
    path = tmp_path / "bounded.jsonl"
    session = GateSession(make_contract(max_retained_receipts=1), SOURCE, journal=path)
    with pytest.raises(module.SessionCapacityError, match="retained_receipt_capacity"):
        execute(session)
    assert session.capacity_exhausted
    assert session.retained_receipt_count == len(session._deadlines) == 1
    assert len(path.read_text().splitlines()) == 1
    assert session.selected_model(next(iter(session._receipts.values()))) == SOURCE
