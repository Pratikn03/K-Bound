"""Retrospective matched-information calibration comparisons.

These routines do not claim new-environment coverage from dependent saved cells.
Decision APIs accept an explicit outcome-free schema. All data roles are split
before fitting, and rank failures are represented by infinite uncertainty.
"""

from __future__ import annotations

import hashlib
import json
import math

# Serialization only hashes locally fitted models; no pickle input is loaded.
import pickle  # nosec B403
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.ensemble import GradientBoostingRegressor

FEATURE_KEYS = {"id", "group", "environment", "checkpoint", "features"}
DEFAULTS = {
    "alphas": [0.1, 0.05, 0.2],
    "harm_weights": [1, 5, 20],
    "margin_grid": [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 2],
    "risk_budgets": [0.1, 0.2],
    "risk_delta": 0.05,
    "gbrt": {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "subsample": 1.0, "random_state": 20260921},
    "scale_floor": 0.001,
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_sha256(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"input hash mismatch: {path}")


def write_fresh_json(path, value):
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(payload)


def exact_radius(residuals, alpha):
    values = np.asarray(residuals, dtype=float)
    if not 0 < alpha < 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid calibration residuals or alpha")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float("inf") if rank > len(values) else float(np.sort(values)[rank - 1])


def conditional_risk_upper(errors, n, delta):
    """One-sided Clopper-Pearson bound; caller applies family correction.

    Requires independent Bernoulli calibration units for its guarantee. Saved
    historical semantic groups do not establish that assumption.
    """
    if not 0 < delta < 1 or not 0 <= errors <= n:
        raise ValueError("invalid binomial counts")
    return 1.0 if n == 0 or errors == n else float(beta.ppf(1 - delta, errors + 1, n - errors))


def outcome_free(rows):
    return [{key: r[key] for key in sorted(FEATURE_KEYS)} for r in rows]


def validate_roles(**roles):
    all_ids = []
    seen_groups = set()
    for name, rows in roles.items():
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs in {name}")
        groups = {r["group"] for r in rows}
        if seen_groups & groups:
            raise ValueError(f"group overlap in {name}")
        seen_groups |= groups
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate IDs across roles")


def semantic_group(condition):
    """Remove a literal terminal repeat tag without discarding composition."""
    fields = condition.split("|")
    if fields[-1].startswith("r") and fields[-1][1:].isdigit():
        fields = fields[:-1]
    return "|".join(fields)


def grouped_roles(rows, salt):
    groups = sorted({r["group"] for r in rows}, key=lambda x: digest([salt, x]))
    if len(groups) < 4:
        raise ValueError("at least four semantic groups required")
    ncal = max(1, len(groups) // 4)
    ntune = max(1, len(groups) // 4)
    mapping = {g: "cal" if i < ncal else "tune" if i < ncal + ntune else "fit" for i, g in enumerate(groups)}
    result = {role: [r for r in rows if mapping[r["group"]] == role] for role in ("fit", "tune", "cal")}
    validate_roles(**result)
    return result


def _arrays(rows):
    X = np.asarray([r["features"] for r in rows], dtype=float)
    y = np.asarray([r["candidate_score"] - r["frozen_score"] for r in rows], dtype=float)
    if X.ndim != 2 or not len(rows) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("malformed fit/tune/calibration records")
    if any(not 0 <= r[k] <= 1 for r in rows for k in ("frozen_score", "candidate_score")):
        raise ValueError("scores must be unit-interval accuracies")
    return X, y


def _model(config):
    return GradientBoostingRegressor(**config["gbrt"])


def _loss(pred, y, margin, harm_weight):
    use = pred > margin
    return float(np.mean(np.maximum(y, 0) * (~use) + harm_weight * np.maximum(-y, 0) * use))


def _group_max(values, rows, key="group"):
    out = {}
    for value, row in zip(values, rows):
        out[row[key]] = max(out.get(row[key], 0), float(value))
    return list(out.values())


def calibrate_fold(fit, tune, cal, config=None):
    config = DEFAULTS if config is None else config
    validate_roles(fit=fit, tune=tune, cal=cal)
    X, y = _arrays(fit)
    Xt, yt = _arrays(tune)
    Xc, yc = _arrays(cal)
    t = time.perf_counter()
    model = _model(config).fit(X, y)
    model_hash = hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest()
    timing = {"predictor_fit_seconds": time.perf_counter() - t}
    # Scale model is learned only from grouped OOF residuals within FIT.
    t = time.perf_counter()
    groups = sorted({r["group"] for r in fit}, key=lambda g: digest(["scale", g]))
    membership = {g: i % 3 for i, g in enumerate(groups)}
    oof = np.empty(len(fit))
    for fold in range(3):
        held = np.array([membership[r["group"]] == fold for r in fit])
        train = ~held
        if held.any() and train.any():
            oof[held] = _model(config).fit(X[train], y[train]).predict(X[held])
        elif held.any():
            oof[held] = np.mean(y)
    scale_model = _model(config).fit(X, np.log(np.maximum(abs(y - oof), config["scale_floor"])))
    timing["scale_oof_fit_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    pt = model.predict(Xt)
    pc = model.predict(Xc)
    scales = np.maximum(np.exp(np.clip(scale_model.predict(Xc), -7, 0)), config["scale_floor"])
    residual = abs(yc - pc)
    used_environments = {r["environment"] for r in fit + tune}
    independent_environment_rows = [r for r in cal if r["environment"] not in used_environments]
    independent_environment_residuals = [e for e, r in zip(residual, cal) if r["environment"] not in used_environments]
    radii = {}
    for alpha in config["alphas"]:
        radii[str(alpha)] = {
            "constant_cell": exact_radius(residual, alpha),
            "scaled_cell": exact_radius(residual / scales, alpha),
            "constant_group": exact_radius(_group_max(residual, cal), alpha),
            "environment_max": exact_radius(
                _group_max(independent_environment_residuals, independent_environment_rows, "environment"), alpha
            ),
        }
    margins = {
        str(w): min(config["margin_grid"], key=lambda m: (_loss(pt, yt, m, w), -m)) for w in config["harm_weights"]
    }
    risk_table = []
    for margin in config["margin_grid"]:
        group_status = {}
        for row, p, b in zip(cal, pc, yc):
            if p > margin:
                group_status[row["group"]] = group_status.get(row["group"], False) or b <= 0
        errors = int(sum(group_status.values()))
        n = len(group_status)
        risk_table.append(
            {
                "margin": margin,
                "adapted_groups": n,
                "error_groups": errors,
                "conditional_upper": conditional_risk_upper(
                    errors, n, config["risk_delta"] / len(config["margin_grid"])
                ),
            }
        )
    risk_selected = {}
    for budget in config["risk_budgets"]:
        eligible = [r["margin"] for r in risk_table if r["conditional_upper"] <= budget and r["adapted_groups"] > 0]
        risk_selected[str(budget)] = min(eligible, key=lambda m: (_loss(pt, yt, m, 5), -m)) if eligible else None
    timing["tuning_calibration_seconds"] = time.perf_counter() - t
    receipt = {
        "predictor_sha256": model_hash,
        "scale_sha256": hashlib.sha256(pickle.dumps(scale_model, protocol=5)).hexdigest(),
        "role_ids": {role: [r["id"] for r in rows] for role, rows in [("fit", fit), ("tune", tune), ("cal", cal)]},
        "role_groups": {
            role: sorted({r["group"] for r in rows}) for role, rows in [("fit", fit), ("tune", tune), ("cal", cal)]
        },
        "calibration_environments": sorted({r["environment"] for r in cal}),
        "independent_calibration_environment_count": len({r["environment"] for r in independent_environment_rows}),
        "calibration_count": len(cal),
        "radii": {a: {k: v if math.isfinite(v) else None for k, v in rs.items()} for a, rs in radii.items()},
        "radii_null_semantics": "positive infinity: exact rank unavailable",
        "fixed_margins": margins,
        "risk_candidates": risk_table,
        "risk_selected": risk_selected,
        "config_sha256": digest(config),
        "timing": timing,
    }
    return {
        "model": model,
        "scale_model": scale_model,
        "radii": radii,
        "margins": margins,
        "risk_selected": risk_selected,
        "config": config,
        "receipt": receipt,
    }


def decide(trained, rows):
    for r in rows:
        if set(r) != FEATURE_KEYS:
            raise ValueError("decision records must use the exact outcome-free schema")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate score IDs")
    used = set().union(*[set(x) for x in trained["receipt"]["role_groups"].values()])
    if used & {r["group"] for r in rows}:
        raise ValueError("score group overlaps fitting/calibration/tuning")
    X = np.asarray([r["features"] for r in rows], dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("nonfinite score features")
    p = trained["model"].predict(X)
    scales = np.maximum(np.exp(np.clip(trained["scale_model"].predict(X), -7, 0)), trained["config"]["scale_floor"])
    output = []
    for i, row in enumerate(rows):

        def add(arm, action, radius=None, row=row, i=i):
            finite = radius is not None and math.isfinite(radius)
            output.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "environment": row["environment"],
                    "checkpoint": row["checkpoint"],
                    "arm": arm,
                    "action": action,
                    "prediction": float(p[i]),
                    "radius": float(radius) if finite else None,
                    "lower": float(p[i] - radius) if finite else None,
                    "upper": float(p[i] + radius) if finite else None,
                    "interval_status": "finite" if finite else "unbounded" if radius is not None else "not_applicable",
                    "predictor_sha256": trained["receipt"]["predictor_sha256"],
                }
            )

        add("always_adapt", "ADAPT")
        add("always_freeze", "FREEZE")
        add("point", "ADAPT" if p[i] > 0 else "FREEZE")
        for w, m in trained["margins"].items():
            add(f"fixed_margin_harm{w}", "ADAPT" if p[i] > m else "FREEZE" if p[i] < -m else "ABSTAIN")
        for alpha, rs in trained["radii"].items():
            for kind, q in rs.items():
                radius = q * scales[i] if kind == "scaled_cell" else q
                add(
                    f"{kind}_alpha{alpha}",
                    "ADAPT" if p[i] > radius else "FREEZE" if p[i] < -radius else "ABSTAIN",
                    radius,
                )
        for budget, m in trained["risk_selected"].items():
            add(f"conditional_risk{budget}", "ABSTAIN" if m is None else "ADAPT" if p[i] > m else "FREEZE")
    return output


def matched_exposure_ids(scores, ids, n):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))[:n]]


def score_decisions(decisions, outcomes):
    truth = {r["id"]: r["candidate_score"] - r["frozen_score"] for r in outcomes}
    if len(truth) != len(outcomes) or len({r["id"] for r in decisions}) != len(decisions):
        raise ValueError("duplicate scoring IDs")
    if set(truth) != {r["id"] for r in decisions}:
        raise ValueError("scoring ID set mismatch")
    B = np.array([truth[d["id"]] for d in decisions])
    adapt = np.array([d["action"] == "ADAPT" for d in decisions])
    n = len(B)
    if n == 0:
        raise ValueError("empty score")
    finite = [i for i, d in enumerate(decisions) if d.get("lower") is not None and d.get("upper") is not None]
    nA = int(adapt.sum())
    errors = int(((B <= 0) & adapt).sum())
    regret = np.maximum(B, 0) - B * adapt
    return {
        "n": n,
        "adapt": nA,
        "freeze": sum(d["action"] == "FREEZE" for d in decisions),
        "abstain": sum(d["action"] == "ABSTAIN" for d in decisions),
        "oracle_regret": float(regret.mean()),
        "score_gain": float(np.mean(B * adapt)),
        "false_adapt_count": errors,
        "false_adapt_unconditional": errors / n,
        "false_adapt_conditional": errors / nA if nA else None,
        "false_freeze_count": int(((B >= 0) & np.array([d["action"] == "FREEZE" for d in decisions])).sum()),
        "forgone_helpful_nonadapt_count": int(((B > 0) & ~adapt).sum()),
        "harmful_adapt_count": int(((B < 0) & adapt).sum()),
        "zero_benefit_adapt_count": int(((B == 0) & adapt).sum()),
        "beneficial_gain_captured": float(np.sum(np.maximum(B, 0) * adapt)),
        "beneficial_gain_forgone": float(np.sum(np.maximum(B, 0) * ~adapt)),
        "harmful_loss": float(np.sum(np.maximum(-B, 0) * adapt)),
        "harm_weighted_losses": {
            str(w): float(np.mean(np.maximum(B, 0) * ~adapt + w * np.maximum(-B, 0) * adapt)) for w in [1, 5, 20]
        },
        "interval_inclusion": float(np.mean([decisions[i]["lower"] <= B[i] <= decisions[i]["upper"] for i in finite]))
        if finite
        else None,
        "finite_interval_count": len(finite),
        "unbounded_interval_count": sum(d.get("interval_status") == "unbounded" for d in decisions),
        "mean_finite_interval_width": float(np.mean([decisions[i]["upper"] - decisions[i]["lower"] for i in finite]))
        if finite
        else None,
    }


def build_folds(rows, kind):
    """Construct all outer folds using only environment/checkpoint/group IDs.

    Checkpoint-only evaluation rotates four held-out semantic-group partitions:
    this prevents even repeated conditions from crossing score and fit roles.
    Joint evaluation excludes both the scored checkpoint and corruption family.
    """
    if kind not in {"environment", "checkpoint", "joint"}:
        raise ValueError("unknown outer split")
    folds = []
    environments = sorted({r["environment"] for r in rows})
    models = sorted({r["checkpoint"] for r in rows})
    groups = sorted({r["group"] for r in rows}, key=lambda g: digest(["score_groups", g]))
    group_fold = {g: i % 4 for i, g in enumerate(groups)}
    specifications = (
        [(e, None, None) for e in environments]
        if kind == "environment"
        else [(None, m, f) for m in models for f in range(4)]
        if kind == "checkpoint"
        else [(e, m, None) for m in models for e in environments]
    )
    for env, model, rotation in specifications:
        score = [
            r
            for r in rows
            if (env is None or r["environment"] == env)
            and (model is None or r["checkpoint"] == model)
            and (rotation is None or group_fold[r["group"]] == rotation)
        ]
        if not score:
            continue
        score_groups = {r["group"] for r in score}
        rest = [
            r
            for r in rows
            if r["group"] not in score_groups
            and (env is None or r["environment"] != env)
            and (model is None or r["checkpoint"] != model)
        ]
        name = f"{kind}:{env or 'all'}:{model or 'all'}:{rotation if rotation is not None else 'all'}"
        roles = grouped_roles(rest, salt=name)
        validate_roles(**roles, score=score)
        folds.append(dict(name=name, kind=kind, **roles, score_features=outcome_free(score), score_outcomes=score))
    return folds


def score_semantic_groups(decisions, outcomes):
    """Held-out risk on (fold_id, semantic_group), separate from cell FA-C.

    A group is exposed if any cell is ADAPT and erroneous if any accepted cell
    has B <= 0. Repeated cells do not increase the risk denominator. Utility is
    first averaged within each group, then equally across all groups, including
    groups with no ADAPT. These descriptive aggregates do not establish group
    independence or a new risk guarantee.
    """
    if not decisions:
        raise ValueError("empty group score")
    keys = [(d["fold_id"], d["id"]) for d in decisions]
    truth = {(r["fold_id"], r["id"]): r for r in outcomes}
    if len(keys) != len(set(keys)) or len(truth) != len(outcomes):
        raise ValueError("duplicate fold/cell scoring IDs")
    if set(keys) != set(truth):
        raise ValueError("group scoring ID set mismatch")
    groups = {}
    for decision, key in zip(decisions, keys):
        outcome = truth[key]
        if decision["group"] != outcome["group"]:
            raise ValueError("decision/outcome semantic group mismatch")
        if decision["action"] not in {"ADAPT", "FREEZE", "ABSTAIN"}:
            raise ValueError("unknown action")
        frozen = float(outcome["frozen_score"])
        candidate = float(outcome["candidate_score"])
        if not 0 <= frozen <= 1 or not 0 <= candidate <= 1:
            raise ValueError("nonfinite or invalid accuracy")
        group_key = (decision["fold_id"], decision["group"])
        groups.setdefault(group_key, []).append((decision["action"] == "ADAPT", candidate - frozen))
    n_adapt = n_error = n_strict = n_zero_only = 0
    losses = {str(w): [] for w in [1, 5, 20]}
    gains = []
    harmful_losses = []
    for rows in groups.values():
        exposed = any(adapt for adapt, benefit in rows)
        negative = any(adapt and benefit < 0 for adapt, benefit in rows)
        zero = any(adapt and benefit == 0 for adapt, benefit in rows)
        n_adapt += exposed
        n_error += negative or zero
        n_strict += negative
        n_zero_only += zero and not negative
        gains.append(sum(benefit if adapt else 0 for adapt, benefit in rows) / len(rows))
        harmful_losses.append(sum(max(-benefit, 0) if adapt else 0 for adapt, benefit in rows) / len(rows))
        for weight in [1, 5, 20]:
            losses[str(weight)].append(
                sum(weight * max(-benefit, 0) if adapt else max(benefit, 0) for adapt, benefit in rows) / len(rows)
            )
    n_all = len(groups)
    return {
        "unit": "(fold_id, semantic_group)",
        "n_all_groups": n_all,
        "n_adapted_groups": n_adapt,
        "n_error_groups": n_error,
        "false_adapt_conditional": n_error / n_adapt if n_adapt else None,
        "false_adapt_unconditional": n_error / n_all,
        "n_strictly_harmful_adapted_groups": n_strict,
        "n_zero_only_error_groups": n_zero_only,
        "equal_group_harm_weighted_losses": {w: sum(values) / n_all for w, values in losses.items()},
        "equal_group_score_gain": sum(gains) / n_all,
        "equal_group_harmful_loss": sum(harmful_losses) / n_all,
    }
