"""K-Bound: the Knowability Boundary of label-free adaptation.

This package is a clean wrapper/index over the real implementations that currently
live in the monorepo (src/scripts/kbound/, src/uais/, and vendored_from_elara/).
Submodules document where each piece lives so the paper code is auditable without
moving load-bearing modules. THIS PACKAGE CONTAINS NO RUNTIME CODE: it is an index.
Canonical implementations: the KGA decision rule (decide_kga, Algorithm 1) is in
docs/research/kbound/scripts/cifar_tent_mps_v2.py; the split-conformal holdout scorer
is docs/research/kbound/scripts/score_kbound_holdout.py; per-condition bootstrap is
docs/research/kbound/scripts/percondition_bootstrap.py; Holm-corrected confirmatory
analysis is experiments/kbound/results/stress_grid_multiseed_v1/_locked_analysis_script.py.
"""
__all__ = ["evidence", "estimators", "decision", "metrics", "theory", "data", "utils"]
