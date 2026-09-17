#!/usr/bin/env python3
"""Fixed three-way common-panel routing and independent scoring.

Public API: fit_gate(manifest, development_features, fit_outcomes,
calibration_outcomes); decide_gate(gate, score_features, native_estimates=[]);
score_decisions(manifest, packet, score_outcomes). Inputs are JSON-shaped values.
Each row carries its exact manifest identity, including ordered sample IDs and
candidate checkpoint/trajectory. Correctness arrays contain one 0/1 per sample.
Every sample belongs to one cell, and environments cannot cross role boundaries.

The manifest must be fixed before outcome access by the caller. This module
checks consistency and outcome exclusion, not chronology, source authenticity,
or exchangeability. Hashes bind declarations, not independently authenticated
native executions. AETTA and Baek ALine estimates feed a disclosed local
estimated-benefit controller. POEM can be the declared adaptation candidate;
another POEM trajectory requires its own manifest and paired outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from run_decision_baselines import EVIDENCE_NAMES, policy_metrics  # noqa: E402
from kbound_decide import conformal_radius, decide  # noqa: E402

_IDENTITY = {"cell_id", "environment_id", "sample_ids", "model_id",
             "frozen_checkpoint_sha256", "candidate_checkpoint_sha256",
             "trajectory_id", "candidate_method"}
_NATIVE = {"identity", "method", "implementation", "source_sha256",
           "invocation_sha256", "frozen_estimated_accuracy", "candidate_estimated_accuracy"}
_POLICIES = {
    "always_freeze": "always FREEZE",
    "always_adapt": "always ADAPT to the declared candidate",
    "point_benefit": "ADAPT iff the shared fitted benefit prediction is strictly positive",
    "KGA": "positive interval ADAPT; negative interval FREEZE; otherwise ABSTAIN (deploy frozen)",
    "confidence_gain": "ADAPT iff post_conf - pre_conf > 0; fixed threshold zero",
    "entropy_progress": "ADAPT iff entropy_drop > 0; fixed threshold zero",
}


def _keys(value, keys, context):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{context} requires exactly {sorted(keys)}")


def _text(value, context):
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"invalid {context}")


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA256 declaration")


def _number(value, low=-math.inf, high=math.inf):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("expected finite numeric value in allowed range")
    return float(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _manifest(value):
    _keys(value, {"schema", "feature_names", "cells"}, "manifest")
    if value["schema"] != "task3-common-panel-v1" or value["feature_names"] != EVIDENCE_NAMES:
        raise ValueError("unsupported manifest schema or feature ordering")
    if type(value["cells"]) is not list or not value["cells"]:
        raise ValueError("manifest needs cells")
    ids, samples, environment_roles = set(), set(), {}
    counts = {"fit": 0, "calibrate": 0, "score": 0}
    for cell in value["cells"]:
        _keys(cell, {"role", "identity"}, "cell")
        role = cell["role"]
        if type(role) is not str or role not in counts:
            raise ValueError("unknown role")
        identity = cell["identity"]
        _keys(identity, _IDENTITY, "cell identity")
        for key in _IDENTITY - {"sample_ids"}:
            _text(identity[key], key)
        for key in ("frozen_checkpoint_sha256", "candidate_checkpoint_sha256"):
            _sha(identity[key])
        sample_ids = identity["sample_ids"]
        if type(sample_ids) is not list or not sample_ids:
            raise ValueError("sample IDs must be an ordered nonempty list")
        for sample in sample_ids:
            _text(sample, "sample ID")
            if sample in samples:
                raise ValueError("duplicate or overlapping underlying sample identity")
            samples.add(sample)
        if identity["cell_id"] in ids:
            raise ValueError("duplicate cell ID")
        ids.add(identity["cell_id"])
        env = identity["environment_id"]
        if env in environment_roles and environment_roles[env] != role:
            raise ValueError("environment overlaps fit/calibrate/score roles")
        environment_roles[env] = role
        counts[role] += 1
    if counts["fit"] < 2 or not counts["calibrate"] or not counts["score"]:
        raise ValueError("need at least two fit cells, one calibration cell and one score cell")
    return json.loads(_canonical(value))


def _rows(manifest, rows, roles, extra_keys):
    if type(rows) is not list:
        raise ValueError("rows must be a list")
    expected = [c["identity"] for c in manifest["cells"] if c["role"] in roles]
    lookup = {i["cell_id"]: i for i in expected}
    found = {}
    for row in rows:
        _keys(row, {"identity"} | set(extra_keys), "row")
        identity = row["identity"]
        _keys(identity, _IDENTITY, "row identity")
        cell_id = identity["cell_id"]
        _text(cell_id, "cell ID")
        if cell_id not in lookup or identity != lookup[cell_id]:
            raise ValueError("row identity/role does not match exact manifest association")
        if cell_id in found:
            raise ValueError("duplicate row")
        found[cell_id] = row
    if set(found) != set(lookup):
        raise ValueError("missing role rows")
    return [found[i["cell_id"]] for i in expected]


def _features(manifest, rows, roles):
    rows = _rows(manifest, rows, roles, {"Z"})
    values = []
    for row in rows:
        z = row["Z"]
        if type(z) is not list or len(z) != len(EVIDENCE_NAMES):
            raise ValueError("feature width/type mismatch")
        z = [_number(v) for v in z]
        for index in (1, 4, 8):
            _number(z[index], 0, 1)
        for index in (0, 2, 3, 5, 9, 10):
            _number(z[index], 0)
        values.append(z)
    return rows, np.asarray(values)


def _outcomes(manifest, rows, roles):
    rows = _rows(manifest, rows, roles, {"frozen_correct", "candidate_correct"})
    accuracies = []
    for row in rows:
        pair = []
        for key in ("frozen_correct", "candidate_correct"):
            values = row[key]
            if type(values) is not list or len(values) != len(row["identity"]["sample_ids"]):
                raise ValueError("correctness length/type mismatch")
            for value in values:
                _number(value, 0, 1)
                if value not in (0, 1):
                    raise ValueError("correctness must be binary")
            pair.append(float(np.mean(values)))
        accuracies.append(pair)
    return rows, np.asarray(accuracies)


def exact_radius(residuals):
    """Exact alpha=.1 finite-sample rank, returning +inf when k is unavailable."""
    if type(residuals) is not list:
        raise ValueError("residuals must be a list")
    values = [_number(value, 0) for value in residuals]
    if math.ceil((len(values) + 1) * 0.9) > len(values):
        return math.inf
    return conformal_radius(values, alpha=0.1)


@dataclass(frozen=True)
class FittedGate:
    """Snapshot of role declarations and fitted state; never accepts scored labels."""
    manifest_json: str
    fit_binding: str
    calibration_binding: str
    radius: float
    _model: GradientBoostingRegressor


def fit_gate(manifest, development_features, fit_outcomes, calibration_outcomes):
    """Fit on fit outcomes and calibrate on calibration outcomes, exactly once."""
    manifest = _manifest(manifest)
    ordered, _ = _features(manifest, development_features, {"fit", "calibrate"})
    fit_ids = {c["identity"]["cell_id"] for c in manifest["cells"] if c["role"] == "fit"}
    fit_features = [r for r in ordered if r["identity"]["cell_id"] in fit_ids]
    calibration_features = [r for r in ordered if r["identity"]["cell_id"] not in fit_ids]
    _, zfit = _features(manifest, fit_features, {"fit"})
    _, zcal = _features(manifest, calibration_features, {"calibrate"})
    fit_rows, fit_acc = _outcomes(manifest, fit_outcomes, {"fit"})
    calibration_rows, cal_acc = _outcomes(manifest, calibration_outcomes, {"calibrate"})
    model = GradientBoostingRegressor(n_estimators=250, max_depth=2,
        learning_rate=0.05, subsample=0.8, random_state=0)
    model.fit(zfit, fit_acc[:, 1] - fit_acc[:, 0])
    residuals = np.abs(model.predict(zcal) - (cal_acc[:, 1] - cal_acc[:, 0]))
    return FittedGate(_canonical(manifest), _digest([fit_features, fit_rows]),
        _digest([calibration_features, calibration_rows]), exact_radius(residuals.tolist()), model)


def _native(manifest, estimates):
    if type(estimates) is not list:
        raise ValueError("native estimates must be a list")
    grouped = {}
    for row in estimates:
        _keys(row, _NATIVE, "native paired estimate")
        method = row["method"]
        if type(method) is not str or method not in {"AETTA", "Baek_ALine"}:
            raise ValueError("only explicitly named AETTA and Baek_ALine native estimates supported")
        if row["implementation"] != "native_component":
            raise ValueError("ports cannot be declared native")
        for key in ("source_sha256", "invocation_sha256"):
            _sha(row[key])
        for key in ("frozen_estimated_accuracy", "candidate_estimated_accuracy"):
            _number(row[key], 0, 1)
        grouped.setdefault(method, []).append(row)
    result = {}
    for method, rows in sorted(grouped.items()):
        result[method] = _rows(manifest, rows, {"score"}, _NATIVE - {"identity"})
        if len({r["source_sha256"] for r in rows}) != 1:
            raise ValueError("native method source must be consistent across the panel")
    return result


def decide_gate(gate, score_features, native_estimates=None):
    """Feature-only decision stage; scored correctness/labels are not accepted."""
    manifest = json.loads(gate.manifest_json)
    features, zscore = _features(manifest, score_features, {"score"})
    native = _native(manifest, [] if native_estimates is None else native_estimates)
    predictions = gate._model.predict(zscore)
    kga = decide(predictions, np.full(len(predictions), gate.radius), alpha=0.1)
    descriptions = dict(_POLICIES)
    for method in native:
        descriptions[method + "_controller"] = (
            "local estimated-benefit controller: ADAPT iff native candidate estimated "
            "accuracy minus native frozen estimated accuracy > 0; otherwise FREEZE; "
            "not an unchanged native adaptation algorithm")
    rows = []
    for index, (feature, prediction) in enumerate(zip(features, predictions)):
        actions = {"always_freeze": "FREEZE", "always_adapt": "ADAPT",
            "point_benefit": "ADAPT" if prediction > 0 else "FREEZE", "KGA": str(kga[index]),
            "confidence_gain": "ADAPT" if feature["Z"][4] - feature["Z"][1] > 0 else "FREEZE",
            "entropy_progress": "ADAPT" if feature["Z"][7] > 0 else "FREEZE"}
        for method, estimates in native.items():
            estimate = estimates[index]
            actions[method + "_controller"] = "ADAPT" if (
                estimate["candidate_estimated_accuracy"] > estimate["frozen_estimated_accuracy"]) else "FREEZE"
        rows.append({"identity": feature["identity"], "prediction": float(prediction), "actions": actions})
    packet = {"schema": "task3-common-decisions-v1", "manifest_sha256": _digest(manifest),
        "fit_binding": gate.fit_binding, "calibration_binding": gate.calibration_binding,
        "score_features_sha256": _digest(features), "alpha": 0.1,
        "radius": "infinity" if math.isinf(gate.radius) else gate.radius,
        "policy_descriptions": descriptions, "rows": rows,
        "native_estimates": [r for rows in native.values() for r in rows],
        "native_provenance": {"independently_authenticated": False,
            "claim": "source/checkpoint/invocation hashes are caller declarations; no native authentication performed"}}
    packet["packet_sha256"] = _digest(packet)
    return json.loads(_canonical(packet))


def score_decisions(manifest, packet, score_outcomes):
    """Score all policies on the identical declared paired cell accuracies.

    Packet checksum detects accidental mutations, not adversarial re-signing.
    The caller must seal/retain the decision packet before opening score outcomes.
    Metrics are equal-cell averages; ABSTAIN deploys the frozen model.
    """
    manifest = _manifest(manifest)
    _keys(packet, {"schema", "manifest_sha256", "fit_binding", "calibration_binding",
        "score_features_sha256", "alpha", "radius", "policy_descriptions", "rows",
        "native_estimates", "native_provenance", "packet_sha256"}, "decision packet")
    if packet["schema"] != "task3-common-decisions-v1" or type(packet["alpha"]) is not float or packet["alpha"] != 0.1:
        raise ValueError("invalid decision packet")
    for key in ("manifest_sha256", "fit_binding", "calibration_binding", "score_features_sha256", "packet_sha256"):
        _sha(packet[key])
    radius = math.inf if packet["radius"] == "infinity" else _number(packet["radius"], 0)
    body = {k: v for k, v in packet.items() if k != "packet_sha256"}
    if packet.get("packet_sha256") != _digest(body) or packet.get("manifest_sha256") != _digest(manifest):
        raise ValueError("decision packet or manifest binding changed")
    decision_rows = _rows(manifest, packet["rows"], {"score"}, {"prediction", "actions"})
    native = _native(manifest, packet["native_estimates"])
    _keys(packet["policy_descriptions"], set(_POLICIES) | {m + "_controller" for m in native}, "policies")
    outcome_rows, accuracies = _outcomes(manifest, score_outcomes, {"score"})
    frozen, candidate = accuracies[:, 0], accuracies[:, 1]
    metrics = {}
    for index, row in enumerate(decision_rows):
        prediction = _number(row["prediction"], -1, 1)
        _keys(row["actions"], packet["policy_descriptions"], "policy actions")
        if any(type(action) is not str or action not in {"ADAPT", "FREEZE", "ABSTAIN"}
               for action in row["actions"].values()):
            raise ValueError("unknown action")
        expected = {"always_freeze": "FREEZE", "always_adapt": "ADAPT",
            "point_benefit": "ADAPT" if prediction > 0 else "FREEZE",
            "KGA": "ADAPT" if prediction - radius > 0 else "FREEZE" if prediction + radius < 0 else "ABSTAIN"}
        for method, estimates in native.items():
            estimate = estimates[index]
            expected[method + "_controller"] = "ADAPT" if (
                estimate["candidate_estimated_accuracy"] > estimate["frozen_estimated_accuracy"]) else "FREEZE"
        if any(row["actions"][policy] != action for policy, action in expected.items()):
            raise ValueError("actions disagree with declared shared prediction/radius/native estimates")
    for policy in packet["policy_descriptions"]:
        actions = [row["actions"][policy] for row in decision_rows]
        result = policy_metrics(actions, frozen, candidate, candidate - frozen)
        result["mean_accuracy"] = result.pop("mean_acc")
        result["commitment_rate"] = result.pop("coverage")
        metrics[policy] = result
    return {"schema": "task3-common-scores-v1", "manifest_sha256": _digest(manifest),
        "decision_packet_sha256": packet["packet_sha256"], "outcomes_sha256": _digest(outcome_rows),
        "aggregation": "equal cell weight; each cell accuracy averages its ordered samples; ABSTAIN deploys frozen",
        "policies": metrics,
        "scope": "declared paired outcomes only; no source authentication or native benchmark completion claim"}
