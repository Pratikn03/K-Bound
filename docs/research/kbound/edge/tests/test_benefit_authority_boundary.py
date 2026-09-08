"""Synthetic tests: serialized estimator bytes need independent authorization."""

import hashlib
import io
import json
import os
import platform
from unittest.mock import Mock

import joblib
import numpy as np
import pytest
import sklearn
from kbound_edge.benefit_estimator import EdgeBenefitEstimator
from kbound_edge.evidence import EDGE_EVIDENCE_NAMES
from sklearn.ensemble import HistGradientBoostingRegressor


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, allow_nan=False).encode()


def identities():
    return {
        "protocol_sha256": "1" * 64,
        "frozen_model_sha256": "2" * 64,
        "candidate_sha256": "3" * 64,
        "fit_sha256": "4" * 64,
        "calibration_sha256": "5" * 64,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


@pytest.fixture
def pair(tmp_path):
    rng = np.random.default_rng(31)
    z = rng.normal(size=(30, 14))
    estimator = EdgeBenefitEstimator(max_iter=3).fit(z, z[:, 0] * 0.2)
    payload = {
        "artifact_schema": "kbound-edge-benefit-joblib/2",
        "evidence_schema_version": "kbound-edge-evidence/1",
        "feature_names": list(EDGE_EVIDENCE_NAMES),
        "identities": identities(),
        "params": estimator.params,
        "model": estimator._model,
    }
    path = tmp_path / "estimator.joblib"
    metadata_path = tmp_path / "decision.json"
    metadata = {
        "metadata_schema": "kbound-edge-decision-metadata/2",
        "artifact_schema": "kbound-edge-benefit-joblib/2",
        "evidence_schema_version": "kbound-edge-evidence/1",
        "feature_names": list(EDGE_EVIDENCE_NAMES),
        "identities": identities(),
        "estimator_sha256": "0" * 64,
        "eps": 0.125,
        "policies": {"conf_tau": 0.5, "entropy_tau": 0.05},
    }
    authority = {
        "authority_schema": "kbound-edge-estimator-authority/1",
        "artifact_schema": "kbound-edge-benefit-joblib/2",
        "evidence_schema_version": "kbound-edge-evidence/1",
        "feature_names": list(EDGE_EVIDENCE_NAMES),
        "identities": identities(),
        "estimator_sha256": "0" * 64,
        "metadata_sha256": "0" * 64,
    }
    result = {
        "path": path,
        "metadata_path": metadata_path,
        "payload": payload,
        "metadata": metadata,
        "authority": authority,
        "original": estimator,
        "z": z,
    }
    seal(result)
    return result


def seal(pair, *, artifact=True, metadata=True):
    """Test-only synthetic independent seal; production cannot self-authorize."""
    if artifact:
        buffer = io.BytesIO()
        joblib.dump(pair["payload"], buffer)
        pair["path"].write_bytes(buffer.getvalue())
        pair["authority"]["estimator_sha256"] = digest(buffer.getvalue())
        pair["metadata"]["estimator_sha256"] = digest(buffer.getvalue())
    if metadata:
        pair["metadata_path"].write_bytes(encoded(pair["metadata"]))
    pair["authority"]["metadata_sha256"] = digest(pair["metadata_path"].read_bytes())
    pair["authority_bytes"] = encoded(pair["authority"])
    pair["expected"] = digest(pair["authority_bytes"])


def load(pair, **overrides):
    kwargs = {
        "metadata_path": str(pair["metadata_path"]),
        "authority": pair["authority_bytes"],
        "expected_authority_sha256": pair["expected"],
        "active_identities": identities(),
    }
    kwargs.update(overrides)
    return EdgeBenefitEstimator.load(str(pair["path"]), **kwargs)


def rejected(pair, monkeypatch, **kwargs):
    deserialize = Mock(side_effect=AssertionError("unverified deserialize"))
    monkeypatch.setattr(joblib, "load", deserialize)
    with pytest.raises(Exception) as caught:
        load(pair, **kwargs)
    assert type(caught.value).__name__ in {
        "EdgeBenefitAuthorityError",
        "EdgeBenefitMetadataError",
        "EdgeBenefitIntegrityError",
    }
    assert deserialize.call_count == 0
    assert "/" not in str(caught.value)
    return caught.value


def test_missing_authority_stops_before_any_artifact_io(monkeypatch):
    """Removing the authority guard must expose the prohibited load attempt."""
    deserialize = Mock(side_effect=AssertionError("unverified deserialize"))
    monkeypatch.setattr(joblib, "load", deserialize)
    monkeypatch.setattr(os, "open", Mock(side_effect=AssertionError("premature file open")))
    with pytest.raises(Exception) as caught:
        EdgeBenefitEstimator.load("/synthetic/never-open.joblib")
    assert type(caught.value).__name__ == "EdgeBenefitAuthorityError"
    assert deserialize.call_count == 0


@pytest.mark.parametrize(
    "bad", [None, True, 4, [], {}, "", "a" * 63, "a" * 65, "A" * 64, " " + "a" * 64, "a" * 64 + " ", "x" * 64]
)
@pytest.mark.parametrize("field", ["expected", "estimator_sha256", "metadata_sha256", "protocol_sha256"])
def test_digest_primitives_reject_before_deserialization(pair, monkeypatch, bad, field):
    if field == "expected":
        rejected(pair, monkeypatch, expected_authority_sha256=bad)
        return
    target = pair["authority"]["identities"] if field == "protocol_sha256" else pair["authority"]
    target[field] = bad
    raw = encoded(pair["authority"])
    rejected(pair, monkeypatch, authority=raw, expected_authority_sha256=digest(raw))


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "legacy",
        "evidence",
        "reordered",
        "short",
        "duplicate",
        "protocol",
        "frozen",
        "candidate",
        "fit",
        "calibration",
        "runtime",
    ],
)
def test_authority_schema_and_active_scope_checked_before_io(pair, monkeypatch, change):
    a = pair["authority"]
    if change == "missing":
        del a["metadata_sha256"]
    elif change == "extra":
        a["unknown"] = 1
    elif change == "legacy":
        a["authority_schema"] = "kbound-edge-estimator-authority/0"
    elif change == "evidence":
        a["evidence_schema_version"] = "legacy"
    elif change == "reordered":
        a["feature_names"].reverse()
    elif change == "short":
        a["feature_names"].pop()
    elif change == "duplicate":
        a["feature_names"][0] = a["feature_names"][1]
    elif change == "runtime":
        a["identities"]["runtime_versions"]["sklearn"] = "0.0.0"
    else:
        key = "frozen_model_sha256" if change == "frozen" else change + "_sha256"
        a["identities"][key] = "9" * 64
    raw = encoded(a)
    monkeypatch.setattr(os, "open", Mock(side_effect=AssertionError("premature file open")))
    rejected(pair, monkeypatch, authority=raw, expected_authority_sha256=digest(raw))


def test_valid_pair_roundtrip_preserves_predictions_and_verified_metadata(pair):
    restored = load(pair)
    np.testing.assert_array_equal(restored.predict(pair["z"]), pair["original"].predict(pair["z"]))
    assert restored.authority_receipt.authority_sha256 == pair["expected"]
    assert restored.authority_receipt.feature_names == EDGE_EVIDENCE_NAMES
    assert restored.decision_metadata.eps == 0.125
    assert restored.decision_metadata.conf_tau == 0.5
    assert restored.decision_metadata.entropy_tau == 0.05
    assert restored.authority_receipt.identities.protocol_sha256 == "1" * 64


def test_missing_active_identity_prevents_io(pair, monkeypatch):
    monkeypatch.setattr(os, "open", Mock(side_effect=AssertionError("premature file open")))
    rejected(pair, monkeypatch, active_identities=None)


@pytest.mark.parametrize("field", ["eps", "conf_tau", "entropy_tau"])
@pytest.mark.parametrize("bad", [None, True, "0.1", [], {}, float("nan"), float("inf"), -float("inf")])
def test_metadata_numeric_primitives_reject_before_joblib(pair, monkeypatch, field, bad):
    target = pair["metadata"] if field == "eps" else pair["metadata"]["policies"]
    target[field] = bad
    pair["metadata_path"].write_bytes(json.dumps(pair["metadata"]).encode())
    seal(pair, artifact=False, metadata=False)
    rejected(pair, monkeypatch)


@pytest.mark.parametrize(
    "change",
    [
        "negative_eps",
        "missing_eps",
        "extra",
        "duplicate",
        "corrupt",
        "identity",
        "wrong_pair",
        "bad_conf_tau",
        "missing_policy",
    ],
)
def test_strict_decision_metadata_rejected_before_joblib(pair, monkeypatch, change):
    meta = pair["metadata"]
    if change == "negative_eps":
        meta["eps"] = -0.1
    elif change == "missing_eps":
        del meta["eps"]
    elif change == "extra":
        meta["extra"] = 0
    elif change == "identity":
        meta["identities"]["calibration_sha256"] = "9" * 64
    elif change == "wrong_pair":
        meta["estimator_sha256"] = "9" * 64
    elif change == "bad_conf_tau":
        meta["policies"]["conf_tau"] = 1.1
    elif change == "missing_policy":
        del meta["policies"]["entropy_tau"]
    raw = encoded(meta)
    if change == "duplicate":
        raw = raw[:-1] + b', "eps": 0.125}'
    if change == "corrupt":
        raw = b"{broken"
    pair["metadata_path"].write_bytes(raw)
    seal(pair, artifact=False, metadata=False)
    rejected(pair, monkeypatch)


@pytest.mark.parametrize(
    "change",
    [
        "root",
        "legacy",
        "missing",
        "extra",
        "schema",
        "feature_order",
        "class",
        "subclass",
        "unfitted",
        "params",
        "param_boolean",
        "feature_count",
        "identity",
    ],
)
def test_authorized_but_invalid_payload_returns_no_estimator(pair, change):
    payload = pair["payload"]
    if change == "root":
        pair["payload"] = []
    elif change == "legacy":
        pair["payload"] = {"params": payload["params"], "model": payload["model"]}
    elif change == "missing":
        del payload["feature_names"]
    elif change == "extra":
        payload["unknown"] = 1
    elif change == "schema":
        payload["artifact_schema"] = "old"
    elif change == "feature_order":
        payload["feature_names"].reverse()
    elif change == "class":
        payload["model"] = np.zeros(14)
    elif change == "subclass":
        payload["model"] = UnsupportedRegressor().fit(pair["z"], np.zeros(30))
    elif change == "unfitted":
        payload["model"] = HistGradientBoostingRegressor(**payload["params"])
    elif change == "params":
        payload["params"] = dict(payload["params"], learning_rate=0.1)
    elif change == "param_boolean":
        payload["params"] = dict(payload["params"], max_iter=True)
    elif change == "feature_count":
        payload["model"].n_features_in_ = 13
    elif change == "identity":
        payload["identities"]["candidate_sha256"] = "9" * 64
    seal(pair)
    with pytest.raises(Exception) as caught:
        load(pair)
    assert type(caught.value).__name__ == "EdgeBenefitPayloadError"


class UnsupportedRegressor(HistGradientBoostingRegressor):
    pass


class NonPrimitiveKey(str):
    pass


def test_parameter_keys_require_exact_primitive_strings(pair):
    params = pair["payload"]["params"]
    pair["payload"]["params"] = {NonPrimitiveKey(key): value for key, value in params.items()}
    seal(pair)
    with pytest.raises(Exception) as caught:
        load(pair)
    assert type(caught.value).__name__ == "EdgeBenefitPayloadError"


def test_replacement_pair_cannot_self_authorize(pair, monkeypatch):
    expected = pair["expected"]
    pair["payload"]["model"]._baseline_prediction += 0.25
    seal(pair)
    pair["path"].with_suffix(".authority.json").write_bytes(pair["authority_bytes"])
    rejected(pair, monkeypatch, expected_authority_sha256=expected)


@pytest.mark.parametrize("field", ["path", "metadata_path"])
def test_changed_file_bytes_fail_before_joblib(pair, monkeypatch, field):
    pair[field].write_bytes(pair[field].read_bytes() + b" ")
    rejected(pair, monkeypatch)


@pytest.mark.parametrize(
    "failure", ["missing", "malformed", "digest_mismatch", "metadata", "feature_order", "protocol", "artifact_digest"]
)
def test_executable_gadget_never_runs_without_authorization(pair, monkeypatch, failure):
    sentinel = pair["path"].parent / "executed"
    pair["payload"] = Gadget(str(sentinel))
    seal(pair)
    kwargs = {}
    if failure == "missing":
        kwargs["authority"] = None
    elif failure in {"malformed", "digest_mismatch"}:
        kwargs["expected_authority_sha256"] = "invalid" if failure == "malformed" else "f" * 64
    elif failure == "metadata":
        pair["metadata"]["eps"] = True
        seal(pair, artifact=False)
    elif failure == "feature_order":
        pair["authority"]["feature_names"].reverse()
        seal(pair, artifact=False)
    elif failure == "protocol":
        pair["authority"]["identities"]["protocol_sha256"] = "9" * 64
        seal(pair, artifact=False)
    elif failure == "artifact_digest":
        pair["authority"]["estimator_sha256"] = "f" * 64
        pair["metadata"]["estimator_sha256"] = "f" * 64
        seal(pair, artifact=False)
    rejected(pair, monkeypatch, **kwargs)
    assert not sentinel.exists()


class Gadget:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return os.system, ("touch " + self.path,)


@pytest.mark.parametrize("symlink", [False, True])
def test_path_substitution_after_capture_never_changes_loaded_bytes(pair, monkeypatch, symlink):
    original_load = joblib.load
    expected = pair["original"].predict(pair["z"])
    replacement = pair["path"].parent / "replacement.joblib"
    replacement.write_bytes(b"untrusted replacement")

    def substitute(stream):
        assert type(stream) is io.BytesIO
        pair["path"].unlink()
        if symlink:
            pair["path"].symlink_to(replacement)
        else:
            replacement.rename(pair["path"])
        return original_load(stream)

    monkeypatch.setattr(joblib, "load", substitute)
    np.testing.assert_array_equal(load(pair).predict(pair["z"]), expected)


@pytest.mark.parametrize("field", ["path", "metadata_path"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory", "missing", "oversize"])
def test_nonregular_aliased_missing_or_unbounded_files_rejected(pair, monkeypatch, field, kind):
    path = pair[field]
    content = path.read_bytes()
    path.unlink()
    if kind in {"symlink", "hardlink"}:
        other = path.with_suffix(".other")
        other.write_bytes(content)
        if kind == "symlink":
            path.symlink_to(other)
        else:
            os.link(other, path)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    elif kind == "oversize":
        with path.open("wb") as stream:
            stream.truncate(65 * 1024 * 1024)
    rejected(pair, monkeypatch)


def test_parent_symlink_and_dot_alias_are_rejected(pair, monkeypatch):
    alias = pair["path"].parent / "alias"
    alias.symlink_to(pair["path"].parent, target_is_directory=True)
    pair["path"] = alias / pair["path"].name
    rejected(pair, monkeypatch)


@pytest.mark.parametrize("field", ["path", "metadata_path"])
def test_same_descriptor_mutation_is_rejected(pair, monkeypatch, field):
    original_read = os.read
    changed = False

    def mutate(fd, count):
        nonlocal changed
        data = original_read(fd, count)
        if data and not changed and os.fstat(fd).st_ino == pair[field].stat().st_ino:
            changed = True
            with pair[field].open("ab") as stream:
                stream.write(b" ")
        return data

    monkeypatch.setattr(os, "read", mutate)
    rejected(pair, monkeypatch)


def test_raw_deserializer_error_is_sanitized(pair, monkeypatch):
    monkeypatch.setattr(joblib, "load", Mock(side_effect=ValueError("private path /secret/content")))
    with pytest.raises(Exception) as caught:
        load(pair)
    assert type(caught.value).__name__ == "EdgeBenefitPayloadError"
    assert "private" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_v2_candidate_save_requires_identities_but_does_not_issue_authority(pair):
    candidate = pair["path"].parent / "new-candidate.joblib"
    pair["original"].save(str(candidate), identities=identities())
    assert candidate.exists()
    assert not candidate.with_suffix(".authority.json").exists()
    pair["path"] = candidate
    pair["metadata"]["estimator_sha256"] = digest(candidate.read_bytes())
    pair["authority"]["estimator_sha256"] = digest(candidate.read_bytes())
    seal(pair, artifact=False)
    np.testing.assert_array_equal(load(pair).predict(pair["z"]), pair["original"].predict(pair["z"]))


def test_inmemory_fit_predict_remains_compatible():
    z = np.arange(60, dtype=float).reshape(20, 3)
    est = EdgeBenefitEstimator(max_iter=2).fit(z, np.ones(20))
    assert est.predict_one(z[0]) == 1.0


def test_model_parameter_boolean_cannot_equal_stored_integer(pair):
    pair["payload"]["params"] = dict(pair["payload"]["params"], max_depth=1)
    pair["payload"]["model"].max_depth = True
    seal(pair)
    with pytest.raises(Exception) as caught:
        load(pair)
    assert type(caught.value).__name__ == "EdgeBenefitPayloadError"


@pytest.mark.parametrize("mutation", ["duplicate", "invalid_utf8", "oversize", "nonbytes", "deep_json"])
def test_malformed_authority_content_prevents_all_file_io(pair, monkeypatch, mutation):
    raw = pair["authority_bytes"]
    if mutation == "duplicate":
        raw = raw[:-1] + b', "authority_schema": "kbound-edge-estimator-authority/1"}'
    elif mutation == "invalid_utf8":
        raw = b"\xff"
    elif mutation == "oversize":
        raw = b" " * 65537
    elif mutation == "deep_json":
        raw = b"[" * 2000 + b"]" * 2000
    expected = digest(raw)
    if mutation == "nonbytes":
        raw = bytearray(raw)
    monkeypatch.setattr(os, "open", Mock(side_effect=AssertionError("premature file open")))
    rejected(pair, monkeypatch, authority=raw, expected_authority_sha256=expected)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "bool_digest", "runtime_bool", "runtime_missing", "runtime_old"]
)
def test_active_identity_contract_rejects_malformed_values_before_io(pair, monkeypatch, mutation):
    active = identities()
    if mutation == "missing":
        del active["fit_sha256"]
    elif mutation == "extra":
        active["unknown"] = 1
    elif mutation == "bool_digest":
        active["frozen_model_sha256"] = True
    elif mutation == "runtime_bool":
        active["runtime_versions"]["python"] = True
    elif mutation == "runtime_missing":
        del active["runtime_versions"]["numpy"]
    elif mutation == "runtime_old":
        active["runtime_versions"]["sklearn"] = "0.0.0"
    monkeypatch.setattr(os, "open", Mock(side_effect=AssertionError("premature file open")))
    rejected(pair, monkeypatch, active_identities=active)


@pytest.mark.parametrize(
    "path",
    [
        "relative.joblib",
        "/synthetic//estimator.joblib",
        "/synthetic/./estimator.joblib",
        "/synthetic/../estimator.joblib",
        "/synthetic/estimator.joblib/",
        "/",
    ],
)
def test_exact_native_path_rejects_spelling_aliases(pair, monkeypatch, path):
    pair["path"] = path
    rejected(pair, monkeypatch)


def test_metadata_from_second_sealed_pair_cannot_mix_with_first(pair, monkeypatch):
    expected = pair["expected"]
    original_authority = pair["authority_bytes"]
    original_artifact = pair["path"].read_bytes()
    pair["payload"]["model"]._baseline_prediction += 0.25
    seal(pair)
    pair["path"].write_bytes(original_artifact)
    rejected(pair, monkeypatch, authority=original_authority, expected_authority_sha256=expected)


def test_authorized_nonfinite_prediction_is_rejected(pair):
    pair["payload"]["model"]._baseline_prediction[:] = np.nan
    seal(pair)
    with pytest.raises(Exception) as caught:
        load(pair)
    assert type(caught.value).__name__ == "EdgeBenefitPayloadError"


def test_refit_clears_authority_receipt_and_metadata(pair):
    loaded = load(pair)
    loaded.fit(pair["z"], np.zeros(30))
    assert loaded.authority_receipt is None
    assert loaded.decision_metadata is None


def test_missing_save_identity_leaves_no_candidate(pair):
    candidate = pair["path"].parent / "unscoped.joblib"
    with pytest.raises(Exception) as caught:
        pair["original"].save(str(candidate))
    assert type(caught.value).__name__ == "EdgeBenefitAuthorityError"
    assert not candidate.exists()


def test_io_error_is_fixed_and_context_suppressed(pair, monkeypatch):
    monkeypatch.setattr(os, "open", Mock(side_effect=PermissionError("private path /secret")))
    error = rejected(pair, monkeypatch)
    assert error.__suppress_context__


@pytest.mark.parametrize("field", ["path", "metadata_path"])
def test_unencodable_native_path_is_fixed_and_context_suppressed(pair, monkeypatch, field):
    pair[field] = str(pair[field]) + "\ud800"
    error = rejected(pair, monkeypatch)
    assert type(error).__name__ == "EdgeBenefitIntegrityError"
    assert str(error) == "EDGE_ARTIFACT_INTEGRITY"
    assert error.__cause__ is None
    assert error.__suppress_context__
