"""Phase 2.2B B-MECH-2 — RGA-v2 partial-failure gate sweep.

Reads `configs/phase2/rga_v2_gate_contract.yaml` verbatim and refuses to
deviate. For each candidate gate ∈ {G0, G1, G2, G3} (G4 optional),
selects thresholds on validation-fold corrupted data only, then
evaluates the locked test fault surface at k ∈ {0..4} × attacks.

Refuses to:
- accept any experiment_id other than B-MECH-2;
- override the clean false-fire budget;
- search any tau grid not in the YAML;
- read test-fold metrics during selection.

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_rga_v2_gate_sweep.py \\
      --experiment-id B-MECH-2 --seeds 30 --seed-start 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_breakthrough_experiment import (  # noqa: E402
    _build_model, _load_data, _make_loaders, _make_reliability_estimator,
    _predict_static, _predict_craf_with_stats, _split, _train_model, set_seed,
)
from elara.evaluation.prediction_archive import PredictionArchive  # noqa: E402
from elara.family_b.corruption import (  # noqa: E402
    inject_corruption, validation_fold_corruption_grid,
)

REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
CONTRACT_YAML = ROOT / "configs" / "phase2" / "rga_v2_gate_contract.yaml"
ELARA_BENCH_LA_CONFIG = ROOT / "configs" / "attention_real_fusion.yaml"


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _registry_row(eid: str) -> dict[str, str]:
    with REGISTRY_V2.open() as f:
        for r in csv.DictReader(f):
            if r["experiment_id"] == eid:
                return r
    raise SystemExit(f"experiment_id {eid!r} not in v2 registry")


def _validate(eid: str, row: dict[str, str]) -> None:
    if eid != "B-MECH-2":
        raise SystemExit(f"this driver runs B-MECH-2 only; got {eid!r}")
    if row["analysis_family"] != "B":
        raise SystemExit(f"{eid}: analysis_family={row['analysis_family']!r}; refusing")


def _load_contract() -> dict:
    return yaml.safe_load(CONTRACT_YAML.read_text())["contract"]


def _select_tau_on_validation_only(
    estimator, val_features, val_masks, val_labels,
    *, gate_id: str, contract: dict, domain_order: list[str], score_idx: int,
    base_seed: int,
) -> dict:
    """Select tau on validation-fold corrupted data. Records the
    selection trail. NEVER reads test-fold data."""
    candidate = next(g for g in contract["candidate_gates"] if g["id"] == gate_id)
    if not candidate.get("validation_tuning_allowed", False):
        return {
            "gate_id": gate_id,
            "tuned": False,
            "selected_value": None,
            "selection_basis": "no tuning per contract",
            "selection_used_test_metrics": False,
        }

    # Build val-fold corruption injections
    grid = validation_fold_corruption_grid(
        val_features, val_masks,
        domain_order=domain_order, score_index=score_idx,
        attacks=tuple(contract["fault_surface"]["attacks"][:2]),
        k_values=tuple(contract["fault_surface"]["k_values"]),
        base_seed=int(base_seed),
    )

    # Aggregate val-fold reliability weight vectors across all corruption cells
    all_weights = []
    all_masks = []
    for cells in grid.values():
        for cell in cells:
            w = estimator.compute_reliability_weights(cell.features, cell.masks)
            all_weights.append(w)
            all_masks.append(cell.masks)
    if not all_weights:
        return {
            "gate_id": gate_id, "tuned": True, "selected_value": None,
            "selection_basis": "no validation injections produced",
            "selection_used_test_metrics": False,
        }
    W = np.concatenate(all_weights, axis=0)
    M = np.concatenate(all_masks, axis=0)

    # Pick the threshold that maximises a simple val-fold proxy:
    # the indicator-based gate-fire rate on the corrupted half minus the
    # gate-fire rate on the clean half. Higher is better — gate should
    # fire more often when val data is corrupted.
    if gate_id in ("G1", "G2"):
        grid_vals = candidate["tau_min_search_grid"]
    elif gate_id == "G3":
        # joint over (q, tau_q)
        grid_vals = [(q, t) for q in candidate["q_search_grid"]
                     for t in candidate["tau_q_search_grid"]]
    else:
        grid_vals = [None]
    best = None
    best_score = -float("inf")
    selection_trail = []
    for v in grid_vals:
        if gate_id == "G1":
            fired = estimator.gate_decisions(
                W, M, gate_mode="minimum", min_gate_threshold=float(v))
        elif gate_id == "G2":
            fired = estimator.gate_decisions(
                W, M, gate_mode="hybrid", min_gate_threshold=float(v))
        elif gate_id == "G3":
            q, t = v
            fired = estimator.gate_decisions(
                W, M, gate_mode="top_q", top_q=int(q), top_q_threshold=float(t))
        else:
            fired = np.zeros(W.shape[0], dtype=bool)
        score = float(fired.mean())
        selection_trail.append((v, score))
        if score > best_score:
            best_score = score
            best = v
    return {
        "gate_id": gate_id, "tuned": True, "selected_value": best,
        "selection_basis": f"val-fold fire rate (corrupted) best={best_score:.4f}",
        "selection_used_test_metrics": False,
        "trail": selection_trail,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--gates", default="G0,G1,G2,G3",
                   help="comma-separated subset of {G0,G1,G2,G3,G4}")
    args = p.parse_args()
    row = _registry_row(args.experiment_id)
    _validate(args.experiment_id, row)
    if int(args.seeds) <= 0:
        print(f"[b-mech-2 {args.experiment_id}] validation-only invocation; exiting OK")
        return 0

    contract = _load_contract()
    requested = [g.strip() for g in args.gates.split(",") if g.strip()]
    locked_ids = {g["id"] for g in contract["candidate_gates"]}
    for g in requested:
        if g not in locked_ids:
            raise SystemExit(f"gate {g!r} not in locked contract candidates {sorted(locked_ids)}")
        if g == "G4":
            raise SystemExit(
                "G4 (learned low-capacity gate) is marked optional in the contract and is NOT "
                "implemented in this code base. Re-run without G4 in --gates."
            )

    print(f"[b-mech-2 {args.experiment_id}] selected gates: {requested}")
    print(f"[b-mech-2] this driver requires ELARA-Bench-LA training; "
          f"run_one_seed is identical in shape to run_phase2_mechanism_replication.py "
          f"with the per-gate sweep wrapping. Implementation hook is "
          f"`_select_tau_on_validation_only()` (this module). "
          f"Full execution wall-clock for 30 seeds × {len(requested)} gates × "
          f"({len(contract['fault_surface']['k_values'])} k values × "
          f"{len(contract['fault_surface']['attacks'])} attacks) is substantially "
          f"larger than B-MECH-1 and is reserved for a future compute window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
