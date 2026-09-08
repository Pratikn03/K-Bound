"""Strict external authorization of an edge estimator and its decision metadata.

Callers supply authority JSON *bytes* and an independently trusted SHA-256 of
those exact bytes. Never compute the expected identity from an adjacent receipt.
The receipt pins both candidate files and protocol/model/candidate/fit/calibration
identities. Runtime versions are exact, not a cross-version compatibility claim.

Files use exact absolute native paths, with no dot, symlink, or hardlink aliases.
Each parent is opened relative to a pinned no-follow descriptor. One bounded
regular-file read is checked for same-descriptor stability. This is not a sandbox
against trusted pickle code, process compromise, mount races, or host-owner
write-and-restore attacks. Successful verification does not prove calibration
exchangeability or authenticate the separately loaded frozen checkpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import platform
import re
import stat
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import sklearn

from kbound_edge.evidence import EDGE_EVIDENCE_NAMES

AUTHORITY_SCHEMA = "kbound-edge-estimator-authority/1"
ARTIFACT_SCHEMA = "kbound-edge-benefit-joblib/2"
EVIDENCE_SCHEMA = "kbound-edge-evidence/1"
METADATA_SCHEMA = "kbound-edge-decision-metadata/2"
MAX_JSON_BYTES = 64 * 1024
MAX_ESTIMATOR_BYTES = 64 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_KEYS = {"protocol_sha256", "frozen_model_sha256", "candidate_sha256", "fit_sha256", "calibration_sha256"}
_COMMON_KEYS = {"artifact_schema", "evidence_schema_version", "feature_names", "identities"}


class EdgeBenefitArtifactError(ValueError):
    """Expected unavailable-artifact failure with a fixed, non-sensitive code."""

    code = "EDGE_ARTIFACT_INVALID"

    def __init__(self) -> None:
        super().__init__(self.code)


class EdgeBenefitAuthorityError(EdgeBenefitArtifactError):
    code = "EDGE_AUTHORITY_INVALID"


class EdgeBenefitIntegrityError(EdgeBenefitArtifactError):
    code = "EDGE_ARTIFACT_INTEGRITY"


class EdgeBenefitMetadataError(EdgeBenefitArtifactError):
    code = "EDGE_METADATA_INVALID"


class EdgeBenefitPayloadError(EdgeBenefitArtifactError):
    code = "EDGE_PAYLOAD_INVALID"


@dataclass(frozen=True)
class EdgeActiveIdentities:
    protocol_sha256: str
    frozen_model_sha256: str
    candidate_sha256: str
    fit_sha256: str
    calibration_sha256: str
    runtime_versions: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {**{key: getattr(self, key) for key in _IDENTITY_KEYS}, "runtime_versions": dict(self.runtime_versions)}


@dataclass(frozen=True)
class EdgeEstimatorAuthority:
    authority_sha256: str
    estimator_sha256: str
    metadata_sha256: str
    identities: EdgeActiveIdentities
    feature_names: tuple[str, ...] = EDGE_EVIDENCE_NAMES
    artifact_schema: str = ARTIFACT_SCHEMA
    evidence_schema_version: str = EVIDENCE_SCHEMA
    authority_schema: str = AUTHORITY_SCHEMA


@dataclass(frozen=True)
class EdgeDecisionMetadata:
    eps: float
    conf_tau: float
    entropy_tau: float


def _dictionary(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys or any(type(key) is not str for key in value):
        raise ValueError
    return value


def _digest(value: Any) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError
        obj[key] = value
    return obj


def _constant(_: str) -> None:
    raise ValueError


def _json(data: object) -> Any:
    if type(data) is not bytes or not 0 < len(data) <= MAX_JSON_BYTES:
        raise ValueError
    return json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)


def validate_identities(value: object) -> EdgeActiveIdentities:
    """Copy exact primitive identities and require the current runtime contract."""
    try:
        obj = _dictionary(value, _IDENTITY_KEYS | {"runtime_versions"})
        digests = {key: _digest(obj[key]) for key in _IDENTITY_KEYS}
        runtime = _dictionary(obj["runtime_versions"], {"python", "numpy", "sklearn", "joblib"})
        actual = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        }
        if any(type(version) is not str for version in runtime.values()) or runtime != actual:
            raise ValueError
        return EdgeActiveIdentities(**digests, runtime_versions=tuple(sorted(runtime.items())))
    except (ValueError, TypeError, RecursionError):
        raise EdgeBenefitAuthorityError() from None


def validate_common(obj: dict[str, Any], identities: EdgeActiveIdentities) -> None:
    if (
        type(obj["artifact_schema"]) is not str
        or obj["artifact_schema"] != ARTIFACT_SCHEMA
        or type(obj["evidence_schema_version"]) is not str
        or obj["evidence_schema_version"] != EVIDENCE_SCHEMA
    ):
        raise ValueError
    features = obj["feature_names"]
    if (
        type(features) is not list
        or any(type(item) is not str for item in features)
        or len(features) != 14
        or tuple(features) != EDGE_EVIDENCE_NAMES
    ):
        raise ValueError
    if validate_identities(obj["identities"]) != identities:
        raise ValueError


def verify_authority(data: object, expected_sha256: object, active_identities: object) -> EdgeEstimatorAuthority:
    """No artifact I/O occurs here, including when any required input is absent."""
    try:
        expected = _digest(expected_sha256)
        if type(data) is not bytes or not 0 < len(data) <= MAX_JSON_BYTES:
            raise ValueError
        if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected):
            raise ValueError
        obj = _dictionary(_json(data), _COMMON_KEYS | {"authority_schema", "estimator_sha256", "metadata_sha256"})
        if type(obj["authority_schema"]) is not str or obj["authority_schema"] != AUTHORITY_SCHEMA:
            raise ValueError
        active = validate_identities(active_identities)
        validate_common(obj, active)
        return EdgeEstimatorAuthority(
            expected, _digest(obj["estimator_sha256"]), _digest(obj["metadata_sha256"]), active
        )
    except (ValueError, TypeError, RecursionError):
        raise EdgeBenefitAuthorityError() from None


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError
    return float(value)


def verify_metadata(data: bytes, authority: EdgeEstimatorAuthority) -> EdgeDecisionMetadata:
    try:
        if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), authority.metadata_sha256):
            raise ValueError
        obj = _dictionary(_json(data), _COMMON_KEYS | {"metadata_schema", "estimator_sha256", "eps", "policies"})
        if type(obj["metadata_schema"]) is not str or obj["metadata_schema"] != METADATA_SCHEMA:
            raise ValueError
        validate_common(obj, authority.identities)
        if not hmac.compare_digest(_digest(obj["estimator_sha256"]), authority.estimator_sha256):
            raise ValueError
        policies = _dictionary(obj["policies"], {"conf_tau", "entropy_tau"})
        eps, conf, entropy = _number(obj["eps"]), _number(policies["conf_tau"]), _number(policies["entropy_tau"])
        if eps < 0 or not 0 <= conf <= 1:
            raise ValueError
        return EdgeDecisionMetadata(eps, conf, entropy)
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise EdgeBenefitMetadataError() from None


def _snapshot(st: os.stat_result) -> tuple[int, ...]:
    return (st.st_dev, st.st_ino, st.st_mode, st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _path_parts(path: object) -> list[str]:
    if type(path) is not str or not path.startswith("/") or "\0" in path:
        raise EdgeBenefitIntegrityError()
    parts = path.split("/")[1:]
    if any(part in {"", ".", ".."} for part in parts):
        raise EdgeBenefitIntegrityError()
    return parts


def read_verified_bytes(path: object, expected_sha256: str, max_bytes: int) -> bytes:
    """Read one descriptor into immutable bytes; never reopen for deserialization."""
    parts = _path_parts(path)
    descriptors: list[int] = []
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        parent = os.open("/", directory_flags)
        descriptors.append(parent)
        for component in parts[:-1]:
            parent = os.open(component, directory_flags, dir_fd=parent)
            descriptors.append(parent)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
        descriptors.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= max_bytes:
            raise EdgeBenefitIntegrityError()
        chunks = []
        total = 0
        while True:
            part = os.read(fd, min(65536, max_bytes + 1 - total))
            if not part:
                break
            total += len(part)
            if total > max_bytes:
                raise EdgeBenefitIntegrityError()
            chunks.append(part)
        after = os.fstat(fd)
        if _snapshot(before) != _snapshot(after) or total != after.st_size:
            raise EdgeBenefitIntegrityError()
        data = b"".join(chunks)
        if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected_sha256):
            raise EdgeBenefitIntegrityError()
        return data
    except (OSError, UnicodeError):
        raise EdgeBenefitIntegrityError() from None
    finally:
        failed = False
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                failed = True
        if failed:
            raise EdgeBenefitIntegrityError() from None
