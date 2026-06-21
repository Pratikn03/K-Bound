"""config.py — the single, frozen source of truth for ELARA-Opt hyperparameters.

These defaults are mirrored verbatim in research_lock/elara_opt_protocol_v1.yaml.
Tests assert code==lock so the two can never silently diverge.  All values are
chosen on DEV reasoning only; none is tuned on any held-out test set.
"""
from __future__ import annotations

ELARA_OPT_DEFAULTS = {
    "version": "elara_opt_protocol_v1",
    "seed": 0,
    "steps": 1,
    "lr": 1.0e-3,
    "margin_frac": 0.4,          # reliable-sample threshold = margin_frac * ln(C)
    "lambda_anchor": 1.0,        # weight on the frozen-model KL stability anchor
    "gate_temperature": 1.0,     # softmax temperature for objective weights
    "optimizer": "sgd",          # manual SGD step so the trust region is exact
    "momentum": 0.0,             # 0 -> deterministic, no buffer carryover
    "trust_region": {
        "r_min": 0.05,           # radius when reliability ~ 0 (be conservative)
        "r_max": 0.50,           # radius when reliability ~ 1 (trust the update)
    },
    # monotone reliability scalar s in (0,1); higher = more reliable.
    "reliability_coeffs": {
        "bias": 0.0,
        "conf_mean": 3.0,
        "ent_mean": 3.0,
        "frozen_div": 2.0,
        "bn_drift": 2.0,
        "aug_disagreement": 2.0,
    },
    # deterministic rule-mode logits over {entropy, filtered_entropy, aug_consistency}
    "rule": {
        "a_entropy": 0.0,
        "a_filtered": 0.30,
        "a_aug": 0.0,
        "k_reliable": 2.0,       # reliable -> trust entropy minimization
        "k_unreliable": 2.0,     # unreliable -> lean on augmentation consistency
        "k_disagree": 1.0,       # unstable under aug -> lean on consistency
    },
    "meta": {
        "hidden": 16,
        "checkpoint": "meta/meta_gate_v1.pt",
        "train_ids": "meta/training_data_ids.json",
        # presets the meta-gate chooses among (dev-labeled argmax target):
        "presets": {
            "uniform": [0.3333, 0.3333, 0.3333],
            "entropy_heavy": [0.7, 0.2, 0.1],
            "filtered_heavy": [0.2, 0.7, 0.1],
            "consistency_heavy": [0.1, 0.2, 0.7],
        },
    },
    "modes": ["elara_uniform", "elara_rule", "elara_meta"],
}
