"""kbound_edge.benefit_estimator -- HistGradientBoostingRegressor benefit model.

Predicts the per-window adaptation benefit ``B`` (e.g. accuracy of the adapted
candidate minus accuracy of the frozen model on the same window) from the
label-free evidence vector ``Z``.

The estimator is fit ONLY on the calibration-FIT split.  The conformal radius is
then computed on the held-out calibration-CONFORMAL split (see
:mod:`kbound_edge.conformal`) -- this module deliberately knows nothing about
conformal so the split cannot accidentally leak.

Joblib persistence emits candidates only. Public loading requires independently
authorized v2 bytes and paired decision metadata before executable deserialization.
"""

from __future__ import annotations

import io
import math
from typing import cast

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

from kbound_edge.benefit_authority import (
    ARTIFACT_SCHEMA,
    EVIDENCE_SCHEMA,
    MAX_ESTIMATOR_BYTES,
    MAX_JSON_BYTES,
    EdgeBenefitPayloadError,
    EdgeDecisionMetadata,
    EdgeEstimatorAuthority,
    read_verified_bytes,
    validate_common,
    validate_identities,
    verify_authority,
    verify_metadata,
)
from kbound_edge.evidence import EDGE_EVIDENCE_NAMES


class EdgeBenefitEstimator:
    """Gradient-boosted regressor B_hat = f(Z) for the edge benefit signal.

    Parameters mirror a small, low-variance configuration suitable for the
    modest number of calibration conditions typical of an inspection setup.

    Parameters
    ----------
    max_iter : int, default=200
    learning_rate : float, default=0.05
    max_depth : int, default=3
    l2_regularization : float, default=0.0
    min_samples_leaf : int, default=5
    random_state : int, default=0
    """

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        l2_regularization: float = 0.0,
        min_samples_leaf: int = 5,
        random_state: int = 0,
    ) -> None:
        self.params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "l2_regularization": l2_regularization,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "early_stopping": False,
        }
        self._model: HistGradientBoostingRegressor | None = None
        self.authority_receipt: EdgeEstimatorAuthority | None = None
        self.decision_metadata: EdgeDecisionMetadata | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, Z: np.ndarray, B: np.ndarray) -> EdgeBenefitEstimator:
        """Fit on the calibration-FIT split only.

        Parameters
        ----------
        Z : np.ndarray of shape (n_fit, d)
        B : np.ndarray of shape (n_fit,)
        """
        Z = np.asarray(Z, dtype=float)
        B = np.asarray(B, dtype=float)
        if Z.ndim != 2 or B.ndim != 1 or len(Z) != len(B):
            raise ValueError("Z must be (n,d), B must be (n,), with matching n")
        self.authority_receipt = None
        self.decision_metadata = None
        self._model = HistGradientBoostingRegressor(**self.params).fit(Z, B)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Predict benefit for one or more evidence rows."""
        if self._model is None:
            raise RuntimeError("EdgeBenefitEstimator.fit() must be called before predict()")
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[None, :]
        return cast(np.ndarray, self._model.predict(Z))

    def predict_one(self, z: np.ndarray) -> float:
        """Predict benefit for a single evidence vector -> float."""
        return float(self.predict(np.asarray(z, dtype=float)[None, :])[0])

    # -- persistence -----------------------------------------------------------
    def save(self, path: str, *, identities: object = None) -> None:
        """Write an untrusted v2 candidate, requiring explicit scope identities.

        This does not create an authority or migrate historical artifacts.
        The caller separately writes metadata, reviews both files, and seals
        their exact byte identities through an independent trusted channel.
        """
        import joblib

        if self._model is None:
            raise RuntimeError("Nothing to save: estimator is not fitted")
        scope = validate_identities(identities)
        blob = {
            "artifact_schema": ARTIFACT_SCHEMA,
            "evidence_schema_version": EVIDENCE_SCHEMA,
            "feature_names": list(EDGE_EVIDENCE_NAMES),
            "identities": scope.as_dict(),
            "params": self.params,
            "model": self._model,
        }
        self._validate_payload(blob, scope)
        joblib.dump(blob, path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        metadata_path: object = None,
        authority: object = None,
        expected_authority_sha256: object = None,
        active_identities: object = None,
    ) -> EdgeBenefitEstimator:
        """Authorize strict v2 pair bytes before loading trusted executable code.

        All keyword inputs are required semantically. Defaults produce a typed
        authority error for legacy callers before any artifact I/O. ``authority``
        is strict JSON bytes; its expected SHA-256 and active identities must come
        from independently trusted deployment configuration. This boundary does
        not implement callers' unavailable-ABSTAIN integration.
        """
        import joblib

        receipt = verify_authority(authority, expected_authority_sha256, active_identities)
        metadata_bytes = read_verified_bytes(metadata_path, receipt.metadata_sha256, MAX_JSON_BYTES)
        metadata = verify_metadata(metadata_bytes, receipt)
        verified_bytes = read_verified_bytes(path, receipt.estimator_sha256, MAX_ESTIMATOR_BYTES)
        try:
            blob = joblib.load(io.BytesIO(verified_bytes))
            cls._validate_payload(blob, receipt.identities)
        except Exception:
            raise EdgeBenefitPayloadError() from None
        obj = cls()
        obj.params = blob["params"]
        obj._model = blob["model"]
        obj.authority_receipt = receipt
        obj.decision_metadata = metadata
        return obj

    @staticmethod
    def _validate_payload(blob, identities) -> None:
        try:
            keys = {"artifact_schema", "evidence_schema_version", "feature_names", "identities", "params", "model"}
            if type(blob) is not dict or set(blob) != keys or any(type(key) is not str for key in blob):
                raise ValueError
            validate_common(blob, identities)
            params = blob["params"]
            param_keys = {
                "max_iter",
                "learning_rate",
                "max_depth",
                "l2_regularization",
                "min_samples_leaf",
                "random_state",
                "early_stopping",
            }
            if type(params) is not dict or set(params) != param_keys or any(type(key) is not str for key in params):
                raise ValueError
            for key in ("max_iter", "max_depth", "min_samples_leaf"):
                if type(params[key]) is not int or params[key] < 1:
                    raise ValueError
            if type(params["random_state"]) is not int or not 0 <= params["random_state"] < 2**32:
                raise ValueError
            for key in ("learning_rate", "l2_regularization"):
                if type(params[key]) not in (int, float) or not math.isfinite(params[key]):
                    raise ValueError
            if params["learning_rate"] <= 0 or params["l2_regularization"] < 0 or params["early_stopping"] is not False:
                raise ValueError
            model = blob["model"]
            if type(model) is not HistGradientBoostingRegressor:
                raise ValueError
            check_is_fitted(model)
            if type(model.n_features_in_) is not int or model.n_features_in_ != 14:
                raise ValueError
            if hasattr(model, "feature_names_in_"):
                # The supported save path trains on anonymous ndarray features.
                # Named-frame estimators require a separately reviewed format.
                raise ValueError
            actual_params = model.get_params(deep=False)
            expected_params = HistGradientBoostingRegressor(**params).get_params(deep=False)
            if set(actual_params) != set(expected_params) or any(
                type(actual_params[key]) is not type(value) or actual_params[key] != value
                for key, value in expected_params.items()
            ):
                raise ValueError
            probe = model.predict(np.zeros((1, 14), dtype=float))
            if type(probe) is not np.ndarray or probe.shape != (1,) or not np.isfinite(probe).all():
                raise ValueError
        except Exception:
            raise EdgeBenefitPayloadError() from None
