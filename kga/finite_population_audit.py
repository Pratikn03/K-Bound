"""Labeled design-based audits of a fixed finite frame.

Uniform independent sampling WITH replacement is the statistical premise.
The software records draw ordering and identities but cannot attest an external
frame, label truth, random-source integrity or previous outcome access. These
results do not certify future environments or label-free calibration.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from fractions import Fraction
from pathlib import Path

from kga.transport_numerics import binomial_interval_certificate, rational


def _digest(record):
    raw = {k: v for k, v in record.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _seal(record):
    record["sha256"] = _digest(record)
    return record


def _verify(record):
    if not isinstance(record, dict) or record.get("sha256") != _digest(record):
        raise ValueError("Record digest does not match contents")


def make_plan(
    *,
    ids,
    frozen,
    adapted,
    classes,
    sample_size,
    alpha,
    predictor_identity,
    stratum="all",
    scores=None,
    bins=None,
    bin_universe=None,
    bin_definition=None,
    label_semantics="single_class",
):
    """Freeze outcome-free inputs; class labels must be integers, IDs strings.

    Scores, if supplied, mean adapted correctness probabilities ON DISAGREEMENT
    for a binary task, as in the residual decomposition. They are not calibrated
    by this constructor. Bin semantics and predictor identities are declarations.
    """
    ids, frozen, adapted, classes = map(list, (ids, frozen, adapted, classes))
    if (
        not ids
        or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
        or len(frozen) != len(ids)
        or len(adapted) != len(ids)
    ):
        raise ValueError("Frame requires unique nonempty IDs and aligned predictions")
    if (
        len(classes) < 2
        or any(type(c) is not int for c in classes)
        or len(set(classes)) != len(classes)
        or any(type(c) is not int or c not in classes for c in frozen + adapted)
    ):
        raise ValueError("Invalid class universe or predictions")
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("Positive integer sample size required")
    budget = rational(alpha)
    if label_semantics not in {"single_class", "set_membership"}:
        raise ValueError("Unknown label semantics")
    if not 0 < budget < 1 or stratum not in {"all", "disagreement"}:
        raise ValueError("Invalid budget or stratum")
    if not isinstance(predictor_identity, str) or not predictor_identity.strip():
        raise ValueError("Fixed predictor identity required")
    if scores is not None:
        scores = [rational(s) for s in scores]
        if len(scores) != len(ids) or any(not 0 <= s <= 1 for s in scores):
            raise ValueError("Scores must be aligned and lie in [0,1]")
    if bins is not None:
        bins, bin_universe = list(bins), list(bin_universe or [])
        if (
            len(bins) != len(ids)
            or not bin_universe
            or len(set(bin_universe)) != len(bin_universe)
            or any(not isinstance(b, str) or not b for b in bin_universe)
            or any(b not in bin_universe for b in bins)
            or not isinstance(bin_definition, str)
            or not bin_definition.strip()
        ):
            raise ValueError("Aligned bins, fixed universe and definition required")
    elif bin_universe is not None or bin_definition is not None:
        raise ValueError("Bin metadata without bins")
    return _seal(
        {
            "schema": "kbound-finite-frame-audit-plan-v1",
            "ids": ids,
            "frozen": frozen,
            "adapted": adapted,
            "classes": classes,
            "sample_size": sample_size,
            "alpha_exact": str(budget),
            "predictor_identity": predictor_identity,
            "stratum": stratum,
            "scores_exact": None if scores is None else list(map(str, scores)),
            "bins": bins,
            "bin_universe": bin_universe,
            "bin_definition": bin_definition,
            "label_semantics": label_semantics,
        }
    )


def _parse_fraction(value):
    if not isinstance(value, str) or len(value) > 4096 or not re.fullmatch(r"-?[0-9]+(?:/[0-9]+)?", value):
        raise ValueError("Exact probabilities require bounded canonical rational strings")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Invalid rational string") from error


def _validate_plan(plan):
    """Validate structure as well as digest; neither authenticates its history."""
    _verify(plan)
    fields = {
        "schema",
        "ids",
        "frozen",
        "adapted",
        "classes",
        "sample_size",
        "alpha_exact",
        "predictor_identity",
        "stratum",
        "scores_exact",
        "bins",
        "bin_universe",
        "bin_definition",
        "label_semantics",
        "sha256",
    }
    if set(plan) != fields or plan["schema"] != "kbound-finite-frame-audit-plan-v1":
        raise ValueError("Unsupported or malformed plan schema")
    if any(not isinstance(plan[key], list) for key in ("ids", "frozen", "adapted", "classes")):
        raise ValueError("Plan arrays must be lists")
    for key in ("scores_exact", "bins", "bin_universe"):
        if plan[key] is not None and not isinstance(plan[key], list):
            raise ValueError("Optional plan arrays must be lists or null")
    scores = None if plan["scores_exact"] is None else [_parse_fraction(s) for s in plan["scores_exact"]]
    canonical = make_plan(
        ids=plan["ids"],
        frozen=plan["frozen"],
        adapted=plan["adapted"],
        classes=plan["classes"],
        sample_size=plan["sample_size"],
        alpha=_parse_fraction(plan["alpha_exact"]),
        predictor_identity=plan["predictor_identity"],
        stratum=plan["stratum"],
        scores=scores,
        bins=plan["bins"],
        bin_universe=plan["bin_universe"],
        bin_definition=plan["bin_definition"],
        label_semantics=plan["label_semantics"],
    )
    if canonical != plan:
        raise ValueError("Plan is not in canonical validated form")


def _validate_draws(plan, draws):
    _verify(draws)
    fields = {"schema", "plan_sha256", "indices", "sampling", "randomness_independently_attested", "sha256"}
    if set(draws) != fields or draws["schema"] != "kbound-finite-frame-draws-v1":
        raise ValueError("Unsupported or malformed draw schema")
    if (
        draws["sampling"] != "uniform_with_replacement_os_csprng"
        or draws["randomness_independently_attested"] is not False
    ):
        raise ValueError("Unsupported sampling declaration; random generation is not independently authenticated")
    if draws["plan_sha256"] != plan["sha256"]:
        raise ValueError("Draws belong to a different plan")
    eligible = set(_eligible(plan))
    indices = draws["indices"]
    if (
        not isinstance(indices, list)
        or len(indices) != (plan["sample_size"] if eligible else 0)
        or any(type(i) is not int or i not in eligible for i in indices)
    ):
        raise ValueError("Invalid draw inventory")


def _eligible(plan):
    return [i for i, (f, a) in enumerate(zip(plan["frozen"], plan["adapted"])) if plan["stratum"] == "all" or f != a]


def draw_sample(plan, output_path):
    """Persist a new inventory before scoring. Never overwrite previous draws.

    Repeated frame IDs are legitimate independent draws, not additional unique
    labels. This call has no label argument. The OS random source is not an
    independently witnessed randomization ceremony.
    """
    _validate_plan(plan)
    # Exclusive creation occurs before draws so a retry cannot silently reroll.
    with Path(output_path).open("x", encoding="utf-8") as handle:
        eligible = _eligible(plan)
        indices = [eligible[secrets.randbelow(len(eligible))] for _ in range(plan["sample_size"])] if eligible else []
        record = _seal(
            {
                "schema": "kbound-finite-frame-draws-v1",
                "plan_sha256": plan["sha256"],
                "indices": indices,
                "sampling": "uniform_with_replacement_os_csprng",
                "randomness_independently_attested": False,
            }
        )
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _sample(plan, draws, labels):
    _validate_plan(plan)
    _validate_draws(plan, draws)
    indices = draws["indices"]
    values = []
    for i in indices:
        identifier = plan["ids"][i]
        if identifier not in labels:
            raise ValueError("A sampled label is missing")
        label = labels[identifier]
        valid = (
            (type(label) is int and label in plan["classes"])
            if plan["label_semantics"] == "single_class"
            else (
                isinstance(label, (list, tuple))
                and len(label) > 0
                and all(type(c) is int and c in plan["classes"] for c in label)
            )
        )
        if not valid:
            raise ValueError("Label outside declared class universe")
        values.append((i, label))
    return values


def _interval(x, n, alpha):
    lo, hi, certificate = binomial_interval_certificate(x, n, alpha)
    return Fraction(lo), Fraction(hi), certificate


def assess_sample(plan, draws, labels):
    sample = _sample(plan, draws, labels)
    n = len(sample)
    disagreement = sum(f != a for f, a in zip(plan["frozen"], plan["adapted"]))
    d = Fraction(disagreement, len(plan["ids"]))
    weight = d if plan["stratum"] == "disagreement" else Fraction(1)

    def correct(prediction, truth):
        return prediction == truth if plan["label_semantics"] == "single_class" else prediction in truth

    helpful = sum(correct(plan["adapted"][i], y) and not correct(plan["frozen"][i], y) for i, y in sample)
    harmful = sum(correct(plan["frozen"][i], y) and not correct(plan["adapted"][i], y) for i, y in sample)
    budget = Fraction(plan["alpha_exact"])
    residual_upper = None
    certificates = []
    if disagreement == 0:
        lo = hi = Fraction(0)
    elif plan["stratum"] == "disagreement" and len(plan["classes"]) == 2 and plan["label_semantics"] == "single_class":
        p_lo, p_hi, cert = _interval(helpful, n, budget)
        certificates.append(cert)
        lo, hi = d * (2 * p_lo - 1), d * (2 * p_hi - 1)
        if plan["scores_exact"] is not None:
            scores = [Fraction(plan["scores_exact"][i]) for i in _eligible(plan)]
            mean = sum(scores) / len(scores)
            residual_upper = max(abs(p_lo - mean), abs(p_hi - mean))
    else:
        p_lo, p_hi, cp = _interval(helpful, n, budget / 2)
        q_lo, q_hi, cq = _interval(harmful, n, budget / 2)
        certificates.extend((cp, cq))
        lo, hi = weight * (p_lo - q_hi), weight * (p_hi - q_lo)
    return _seal(
        {
            "schema": "kbound-finite-frame-assessment-v1",
            "plan_sha256": plan["sha256"],
            "draws_sha256": draws["sha256"],
            "scope": "fixed_finite_frame_conditional_on_uniform_independent_draws",
            "prospective_deployment_validated": False,
            "alpha_exact": str(budget),
            "label_semantics": plan["label_semantics"],
            "frame_size": len(plan["ids"]),
            "disagreement_mass_exact": str(d),
            "draw_count": n,
            "unique_labels_used": len({i for i, _ in sample}),
            "helpful_count": helpful,
            "harmful_count": harmful,
            "benefit_interval_exact": [str(lo), str(hi)],
            "residual_absolute_upper_exact": None if residual_upper is None else str(residual_upper),
            "action": "ADAPT" if lo > 0 else "FREEZE" if hi < 0 else "ABSTAIN",
            "probability_certificates": certificates,
        }
    )


def simplex_upper(values, lower, upper):
    """Exact support function of simplex intersected with coordinate boxes."""
    values, lower, upper = [list(map(rational, x)) for x in (values, lower, upper)]
    if (
        not values
        or len(values) != len(lower)
        or len(values) != len(upper)
        or any(not 0 <= low <= high <= 1 for low, high in zip(lower, upper))
        or sum(lower) > 1
        or sum(upper) < 1
    ):
        raise ValueError("Empty or malformed boxed simplex")
    weights = lower.copy()
    remaining = 1 - sum(weights)
    for j in sorted(range(len(values)), key=lambda i: values[i], reverse=True):
        add = min(remaining, upper[j] - weights[j])
        weights[j] += add
        remaining -= add
    assert remaining == 0
    return sum(v * w for v, w in zip(values, weights))


def transport_budget(source_plan, source_draws, source_labels, target_plan, target_draws, target_labels, *, alpha):
    """Audit-assisted upper bound on target-prior-weighted conditional TV.

    Both frames use the same fixed predictor/bin definition and label universe.
    Each frame is uniformly sampled with replacement. Conditional bin counts
    have binomial laws given class counts. Frames need not be independent of
    each other for the union bound. No source/target exchangeability is assumed.
    A subsequent transport certificate must add this bound's failure budget.

    If a class has zero probability in a frame, its conditional bin law is not
    identified. Interpret it as any declared normalized extension; the vacuous
    per-class TV bound covers every such extension. It is not a uniquely
    identified conditional distribution for an absent class.
    """
    _validate_plan(source_plan)
    _validate_plan(target_plan)
    for field in ("predictor_identity", "classes", "bin_universe", "bin_definition"):
        if source_plan[field] != target_plan[field]:
            raise ValueError("Source and target definitions must match")
    if any(p["label_semantics"] != "single_class" for p in (source_plan, target_plan)):
        raise ValueError("Conditional transport requires a single-class partition")
    if any(p["stratum"] != "all" or p["bins"] is None for p in (source_plan, target_plan)):
        raise ValueError("Transport requires full-frame sampling and fixed bins")
    source = _sample(source_plan, source_draws, source_labels)
    target = _sample(target_plan, target_draws, target_labels)
    classes, bins = source_plan["classes"], source_plan["bin_universe"]
    budget = rational(alpha)
    if not 0 < budget < 1:
        raise ValueError("Invalid error budget")
    count = 2 * len(classes) * len(bins) + len(classes)
    per = budget / count
    tv, prior_lo, prior_hi, certificates = [], [], [], []
    for label in classes:
        intervals = []
        for plan, sample in ((source_plan, source), (target_plan, target)):
            selected = [i for i, y in sample if y == label]
            band = []
            for b in bins:
                lo, hi, cert = _interval(sum(plan["bins"][i] == b for i in selected), len(selected), per)
                band.append((lo, hi))
                certificates.append(cert)
            intervals.append(band)
        tv.append(
            min(Fraction(1), sum(max(abs(blo - ahi), abs(bhi - alo)) for (alo, ahi), (blo, bhi) in zip(*intervals)) / 2)
        )
        lo, hi, cert = _interval(sum(y == label for _, y in target), len(target), per)
        prior_lo.append(lo)
        prior_hi.append(hi)
        certificates.append(cert)
    try:
        rho = simplex_upper(tv, prior_lo, prior_hi)
        status = "design_conditional_bound"
    except ValueError:
        # Off the simultaneous event the intersected prior boxes can be empty.
        # A vacuous bound remains safe; an empty box is never a success claim.
        rho, status = Fraction(1), "empty_prior_polytope_vacuous_bound"
    return _seal(
        {
            "schema": "kbound-audit-assisted-transport-budget-v1",
            "status": status,
            "scope": "two_fixed_finite_frames_conditional_on_uniform_independent_draws",
            "source_plan_sha256": source_plan["sha256"],
            "target_plan_sha256": target_plan["sha256"],
            "source_draws_sha256": source_draws["sha256"],
            "target_draws_sha256": target_draws["sha256"],
            "rho_upper_exact": str(rho),
            "per_class_tv_upper_exact": dict(zip(map(str, classes), map(str, tv))),
            "additional_failure_budget_exact": str(budget),
            "per_interval_alpha_exact": str(per),
            "interval_count": count,
            "probability_certificates": certificates,
            "prospective_deployment_validated": False,
        }
    )
