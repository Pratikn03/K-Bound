"""Separate truth-bearing painting feasibility arithmetic, not KGA calibration."""

import math

from domainnet_feasibility_contract import CLASS_MAP_SHA256, LIST_SHA, SCOPE, digest


def validate_predictions(p):
    try:
        ids = p["sample_ids"]
        if len(ids) != 4096 or len(set(ids)) != 4096 or type(p["episode"]) is not int or p["episode"] not in range(3):
            raise ValueError("invalid episode/prediction count")
        for name in ("frozen", "candidate_direct", "candidate_refined"):
            if not isinstance(p[name], list) or len(p[name]) != len(ids):
                raise ValueError("unaligned predictions")
            if any(type(x) is not int or not 0 <= x < 126 for x in p[name]):
                raise ValueError("invalid predicted class")
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed predictions") from exc


def score_episode(packet, truth):
    validate_predictions(packet)
    ids = packet["sample_ids"]
    if any(name not in truth or type(truth[name]) is not int or not 0 <= truth[name] < 126 for name in ids):
        raise ValueError("invalid or missing truth")
    correct = {
        name: [int(value == truth[key]) for key, value in zip(ids, packet[name])]
        for name in ("frozen", "candidate_direct", "candidate_refined")
    }
    n = len(ids)
    b = math.sqrt(2 * math.log(2 / (0.05 / 6)) / n)
    result = {
        "episode": packet["episode"],
        "n": n,
        "sample_ids": ids,
        "correctness": correct,
        "frozen_accuracy": sum(correct["frozen"]) / n,
        "sampling_halfwidth": b,
        "scope": SCOPE,
        "coverage_after_screening_established": False,
        "interval_qualification": (
            "Assumption-based SRSWOR finite-pool design diagnostic; sampling premises not established by tests. "
            "Does not establish 95% simultaneous coverage conditional on passing duplicate, corruption, "
            "resource or completion screening. Not a KGA radius or population certificate."
        ),
    }
    for label in ("direct", "refined"):
        values = correct["candidate_" + label]
        changes = [a - f for a, f in zip(values, correct["frozen"])]
        mean = sum(changes) / n
        result[label] = {
            "accuracy": sum(values) / n,
            "benefit": mean,
            "improvements": changes.count(1),
            "degradations": changes.count(-1),
            "ties": changes.count(0),
            "interval": [max(-1.0, mean - b), min(1.0, mean + b)],
        }
    return result


def summarize_episodes(scores):
    if len(scores) != 3 or sorted(x["episode"] for x in scores) != [0, 1, 2]:
        raise ValueError("exactly three complete unique episodes required")
    if any(x["n"] != 4096 for x in scores):
        raise ValueError("wrong episode size")
    ids = [name for score in scores for name in score["sample_ids"]]
    if len(set(ids)) != 12288:
        raise ValueError("repeated evaluation unit")
    direct = sum(x["direct"]["benefit"] for x in scores) / 3
    refined = sum(x["refined"]["benefit"] for x in scores) / 3
    return {
        "schema": "painting-feasibility-summary-v1",
        "status": "DEVELOPMENT_FEASIBILITY_SCORED",
        "episodes": 3,
        "independent_source_models": 1,
        "domains": 1,
        "scope": SCOPE,
        "prospective": False,
        "kga_routing_evaluated": False,
        "direct_mean_benefit": direct,
        "refined_mean_benefit": refined,
        "aggregate_confidence_interval": None,
        "coverage_after_screening_established": False,
        "development_support": direct >= 0.02 and sum(x["direct"]["interval"][0] > 0 for x in scores) >= 2,
        "interpretation": "Conditional painting development support only; no automatic recipe selection or routing claim",
        "scores": scores,
    }


def parse_authenticated_truth(data):
    """Caller authenticates LIST_SHA before this separate-stage label parser."""
    import hashlib

    if hashlib.sha256(data).hexdigest() != LIST_SHA:
        raise ValueError("label authority digest mismatch")
    truth, classes = {}, {}
    for line in data.decode().splitlines():
        name, value = line.split()
        value = int(value)
        category = name.split("/")[1]
        if name in truth or not 0 <= value < 126:
            raise ValueError("invalid/duplicate native truth")
        if category in classes and classes[category] != value:
            raise ValueError("inconsistent class mapping")
        truth[name] = value
        classes[category] = value
    if (
        len(truth) != 30042
        or len(classes) != 126
        or set(classes.values()) != set(range(126))
        or digest(classes) != CLASS_MAP_SHA256
    ):
        raise ValueError("source class mapping mismatch")
    return truth
