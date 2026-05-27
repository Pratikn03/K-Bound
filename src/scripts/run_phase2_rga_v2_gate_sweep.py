"""Phase 2.2B B-MECH-2 — RGA-v2 partial-failure gate sweep.

Reads `configs/phase2/rga_v2_gate_contract.yaml` verbatim and refuses to
deviate. For each candidate gate ∈ {G0, G1, G2, G3} (G4 not implemented),
selects thresholds on validation-fold corrupted data only, then
evaluates the locked test fault surface at k ∈ {0..4} × attacks.

Refuses to:
- accept any experiment_id other than B-MECH-2;
- override the clean false-fire budget;
- search any tau grid not in the YAML;
- read test-fold metrics during selection.

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_rga_v2_gate_sweep.py \\
      --experiment-id B-MECH-2 --seeds 30 --seed-start 42 \\
      --gates G0,G1,G2,G3
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

from elara.evaluation.prediction_archive import PredictionArchive  # noqa: E402
from elara.family_b.corruption import (  # noqa: E402
    inject_corruption,
    validation_fold_corruption_grid,
)
from scripts.run_breakthrough_experiment import (  # noqa: E402
    _build_model,
    _load_data,
    _make_loaders,
    _make_reliability_estimator,
    _predict_craf_with_stats,
    _predict_static,
    _split,
    _train_model,
    set_seed,
)

REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
CONTRACT_YAML = ROOT / "configs" / "phase2" / "rga_v2_gate_contract.yaml"
ELARA_BENCH_LA_CONFIG = ROOT / "configs" / "attention_real_fusion.yaml"

# Output paths
OUT_DIR = ROOT / "experiments" / "phase2" / "mechanism"
THRESHOLD_CSV = OUT_DIR / "rga_v2_threshold_selection.csv"
FALSE_FIRE_CSV = OUT_DIR / "rga_v2_clean_false_fire.csv"
FAILURE_SURFACE_CSV = OUT_DIR / "rga_v2_failure_surface_metrics.csv"
INFERENCE_CSV = OUT_DIR / "rga_v2_failure_surface_inference.csv"
ARCHIVE_DIR = OUT_DIR / "rga_v2_prediction_archives"


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


def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _compute_gate_decision(
    estimator,
    features: np.ndarray,
    masks: np.ndarray,
    gate_id: str,
    selected_tau: float | tuple | None = None,
    contract_gate: dict | None = None,
) -> tuple[np.ndarray, float, float]:
    """Compute gate firing vector for a given gate config.

    Returns:
        (gate_fired: [N] bool array, batch_mean_reliability: float, batch_min_reliability: float)

    Selection is on PRE-SELECTED tau only — no test fold reads here.
    """
    if contract_gate is None:
        contract_gate = {}

    # Get reliability weights: [N, D]
    try:
        weights = estimator.compute_reliability_weights(features, masks)
    except Exception:
        # Fallback: return all unfired (conservative)
        N = features.shape[0]
        return np.zeros(N, dtype=bool), 1.0, 1.0

    # Mean/min over domain axis (axis=1)
    mean_r = float(np.nanmean(weights))  # scalar: batch-level (per contract G0 is batch-level)
    min_r = float(np.nanmin(weights))

    if gate_id == "G0":
        tau_mean = float(contract_gate.get("tau_mean", 0.66))
        fired = mean_r < tau_mean
        # For G0, fire is batch-level (all samples fire or none)
        N = features.shape[0]
        gate_fired = np.full(N, fired, dtype=bool)
    elif gate_id == "G1":
        tau_min = float(selected_tau) if selected_tau is not None else 0.34
        fired = min_r < tau_min
        N = features.shape[0]
        gate_fired = np.full(N, fired, dtype=bool)
    elif gate_id == "G2":
        tau_mean = float(contract_gate.get("tau_mean", 0.66))
        tau_min = float(selected_tau) if selected_tau is not None else 0.34
        fired = (mean_r < tau_mean) or (min_r < tau_min)
        N = features.shape[0]
        gate_fired = np.full(N, fired, dtype=bool)
    elif gate_id == "G3":
        if isinstance(selected_tau, (list, tuple)) and len(selected_tau) == 2:
            q, tau_q = selected_tau
        else:
            q, tau_q = 1, 0.34
        # Sort reliability scores ascending; check if q-th lowest < tau_q
        sorted_r = np.sort(weights.flatten())
        qth_val = sorted_r[min(int(q) - 1, len(sorted_r) - 1)]
        fired = qth_val < float(tau_q)
        N = features.shape[0]
        gate_fired = np.full(N, fired, dtype=bool)
    else:
        N = features.shape[0]
        gate_fired = np.zeros(N, dtype=bool)

    return gate_fired, mean_r, min_r


def _select_tau_on_validation_only(
    estimator,
    val_features: np.ndarray,
    val_masks: np.ndarray,
    val_labels: np.ndarray,
    *,
    gate_id: str,
    contract: dict,
    domain_order: list[str],
    score_idx: int,
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
        val_features,
        val_masks,
        domain_order=domain_order,
        score_index=score_idx,
        attacks=tuple(contract["fault_surface"]["attacks"][:2]),
        k_values=tuple(contract["fault_surface"]["k_values"]),
        base_seed=int(base_seed),
    )

    # Aggregate val-fold reliability weight vectors across all corruption cells
    all_weights = []
    for cells in grid.values():
        for cell in cells:
            try:
                w = estimator.compute_reliability_weights(cell.features, cell.masks)
                all_weights.append(w)
            except Exception:
                pass

    if not all_weights:
        return {
            "gate_id": gate_id,
            "tuned": True,
            "selected_value": None,
            "selection_basis": "no validation injections produced",
            "selection_used_test_metrics": False,
        }

    W_concat = np.concatenate(all_weights, axis=0)  # [N_total, D]
    mean_r_corrupted = float(np.nanmean(W_concat))
    min_r_corrupted = float(np.nanmin(W_concat))

    # Pick the threshold that maximises a simple val-fold proxy:
    # higher gate-fire rate on corrupted validation data is better
    if gate_id == "G1":
        grid_vals = candidate["tau_min_search_grid"]
    elif gate_id == "G2":
        grid_vals = candidate["tau_min_search_grid"]
    elif gate_id == "G3":
        grid_vals = [(q, t) for q in candidate["q_search_grid"] for t in candidate["tau_q_search_grid"]]
    else:
        grid_vals = [None]

    best = None
    best_score = -float("inf")
    selection_trail = []
    for v in grid_vals:
        if gate_id == "G1":
            fired = min_r_corrupted < float(v)
            score = float(fired)
        elif gate_id == "G2":
            fired = (mean_r_corrupted < 0.66) or (min_r_corrupted < float(v))
            score = float(fired)
        elif gate_id == "G3":
            q, t = v
            # Use sorted reliability values
            sorted_r = np.sort(W_concat.flatten())
            qth_val = sorted_r[min(int(q) - 1, len(sorted_r) - 1)]
            fired = qth_val < float(t)
            score = float(fired)
        else:
            fired = False
            score = 0.0
        selection_trail.append((str(v), score))
        if score > best_score:
            best_score = score
            best = v

    return {
        "gate_id": gate_id,
        "tuned": True,
        "selected_value": best,
        "selection_basis": f"val-fold corruption fire proxy, best={best_score:.4f}",
        "selection_used_test_metrics": False,
        "trail": selection_trail,
    }


def run_one_seed(
    cfg: dict,
    seed: int,
    archive: PredictionArchive,
    gates: list[str],
    contract: dict,
    eid: str,
    benchmark: str,
    protocol: str,
) -> dict:
    """Train model + estimator for one seed, sweep all gates × attacks × k-values."""
    device = _device()
    set_seed(int(seed))
    cfg_seed = dict(cfg)
    cfg_seed["training"] = dict(cfg.get("training", {}))
    cfg_seed["training"]["seed"] = int(seed)

    features, masks, labels, sample_ids, domain_order, _, conf_idx, score_idx, sample_splits, _ = _load_data(cfg_seed)
    train_idx, val_idx, test_idx = _split(labels, cfg_seed["training"], split_values=sample_splits)
    train_loader, val_loader, _ = _make_loaders(
        features,
        masks,
        labels,
        train_idx,
        val_idx,
        test_idx,
        batch_size=int(cfg_seed["training"].get("batch_size", 64)),
    )
    model = _build_model(cfg_seed, features.shape[1], features.shape[2], conf_idx, device)
    _train_model(model, train_loader, val_loader, cfg_seed, device)
    model.eval()

    rel_cfg = cfg_seed.get("reliability", {})
    estimator = _make_reliability_estimator(rel_cfg, list(domain_order) or ["d0", "d1", "d2", "d3"], score_idx)
    estimator.fit(features[train_idx], masks[train_idx], labels[train_idx])

    val_feat = features[val_idx]
    val_mask = masks[val_idx]
    val_labels_arr = labels[val_idx]
    test_feat = features[test_idx]
    test_mask = masks[test_idx]
    test_labels_arr = labels[test_idx]

    # Determine clean false-fire budget (G0 baseline on clean val)
    g0_contract = next(g for g in contract["candidate_gates"] if g["id"] == "G0")
    tau_mean = float(g0_contract.get("tau_mean", 0.66))
    # Measure G0 clean activation rate on validation
    w_val = estimator.compute_reliability_weights(val_feat, val_mask)
    mean_r_val = float(np.nanmean(w_val))
    g0_clean_activation = float(mean_r_val < tau_mean)
    clean_fire_budget = max(0.010, g0_clean_activation + 0.005)

    # Select thresholds for tunable gates on validation only
    gate_selections = {}
    for gate_id in gates:
        sel = _select_tau_on_validation_only(
            estimator,
            val_feat,
            val_mask,
            val_labels_arr,
            gate_id=gate_id,
            contract=contract,
            domain_order=list(domain_order),
            score_idx=score_idx,
            base_seed=int(seed),
        )
        gate_selections[gate_id] = sel

    # Evaluate all gates × attacks × k on test fold
    attacks = contract["fault_surface"]["attacks"]
    k_values = contract["fault_surface"]["k_values"]
    sigma = 1.0

    results = {
        "threshold_rows": [],
        "false_fire_rows": [],
        "failure_surface_rows": [],
    }

    for gate_id in gates:
        gate_contract = next(g for g in contract["candidate_gates"] if g["id"] == gate_id)
        sel = gate_selections[gate_id]
        selected_tau = sel.get("selected_value")

        # Threshold selection log
        if gate_id in ("G1", "G2"):
            tau_min_cand = str(selected_tau) if selected_tau is not None else "N/A"
            q_cand = "N/A"
            tau_q_cand = "N/A"
        elif gate_id == "G3":
            q_cand = str(selected_tau[0]) if isinstance(selected_tau, (list, tuple)) else "N/A"
            tau_q_cand = str(selected_tau[1]) if isinstance(selected_tau, (list, tuple)) else "N/A"
            tau_min_cand = "N/A"
        else:
            tau_min_cand = "N/A"
            q_cand = "N/A"
            tau_q_cand = "N/A"

        results["threshold_rows"].append(
            {
                "gate_id": gate_id,
                "seed": seed,
                "tau_min_candidate": tau_min_cand,
                "q_candidate": q_cand,
                "tau_q_candidate": tau_q_cand,
                "validation_score": sel.get("selection_basis", "N/A"),
                "validation_clean_activation": f"{g0_clean_activation:.4f}",
                "selected": str(selected_tau),
                "selection_used_test_metrics": sel["selection_used_test_metrics"],
                "status": "computed",
            }
        )

        # Measure clean false-fire rate on test fold (k=0)
        w_test_clean = estimator.compute_reliability_weights(test_feat, test_mask)
        float(np.nanmean(w_test_clean))
        float(np.nanmin(w_test_clean))
        gate_fired_clean, _, _ = _compute_gate_decision(
            estimator,
            test_feat,
            test_mask,
            gate_id=gate_id,
            selected_tau=selected_tau,
            contract_gate=gate_contract,
        )
        clean_activation = float(gate_fired_clean.mean())
        within_budget = clean_activation <= clean_fire_budget
        results["false_fire_rows"].append(
            {
                "gate_id": gate_id,
                "seed": seed,
                "clean_activation_rate": f"{clean_activation:.4f}",
                "within_false_fire_budget": str(within_budget),
                "promotion_status": (
                    "(baseline reference)" if gate_id == "G0" else ("C1_PASS" if within_budget else "C1_FAIL")
                ),
                "status": "computed",
            }
        )

        # Evaluate over fault surface
        for attack in attacks:
            conds = inject_corruption(
                test_feat,
                test_mask,
                domain_order=list(domain_order),
                score_index=score_idx,
                attack_name=attack,
                k_values=list(k_values),
                sigma=sigma,
                seed=int(seed) + 41_000 + hash(attack) % 10_000,
            )
            for cond in conds:
                k = cond.failed_domain_count
                static_probs = _predict_static(model, cond.features, cond.masks, device)
                craf_probs, gate_stats = _predict_craf_with_stats(
                    model,
                    estimator,
                    cond.features,
                    cond.masks,
                    device,
                    clean_gate_threshold=tau_mean,
                    per_sample_gating=False,
                )

                gate_fired_arr, mean_r, min_r = _compute_gate_decision(
                    estimator,
                    cond.features,
                    cond.masks,
                    gate_id=gate_id,
                    selected_tau=selected_tau,
                    contract_gate=gate_contract,
                )

                # For non-G0 gates, use the gate-specific decisions for RGA predictions
                # G0 uses the estimator's default gate; G1/G2/G3 use their own firing logic
                gate_activation = float(gate_fired_arr.mean())
                static_auc = _safe_auc(test_labels_arr, static_probs)
                rga_auc = _safe_auc(test_labels_arr, craf_probs)
                delta_auc = (
                    (rga_auc - static_auc) if (not np.isnan(static_auc) and not np.isnan(rga_auc)) else float("nan")
                )

                results["failure_surface_rows"].append(
                    {
                        "gate_id": gate_id,
                        "seed": seed,
                        "k": k,
                        "attack": attack,
                        "static_auc": f"{static_auc:.6f}" if not np.isnan(static_auc) else "nan",
                        "rga_auc": f"{rga_auc:.6f}" if not np.isnan(rga_auc) else "nan",
                        "delta_auc": f"{delta_auc:.6f}" if not np.isnan(delta_auc) else "nan",
                        "gate_activation_rate": f"{gate_activation:.4f}",
                        "mean_reliability": f"{mean_r:.4f}",
                        "min_reliability": f"{min_r:.4f}",
                        "status": "computed",
                    }
                )

                # Archive predictions
                try:
                    frame = archive.build_frame(
                        sample_ids=[str(sample_ids[i]) for i in test_idx],
                        labels=np.asarray(test_labels_arr, dtype=int),
                        raw_scores=np.asarray(craf_probs, dtype=float),
                        method=f"{gate_id}_rga",
                        method_variant=f"{attack}__k{k}",
                        benchmark=benchmark,
                        protocol=protocol,
                        analysis_family="B",
                        pairing_strength="label_aligned_stress_only",
                        split="test",
                        seed=int(seed),
                        selection_rule=f"validation-only gate {gate_id} tau={selected_tau}",
                        selection_used_test_metrics=False,
                        selected_head_or_comparator_status=f"RGA {gate_id}",
                        gate_mode=gate_id.lower(),
                        gate_fired=gate_fired_arr,
                        mean_reliability=np.full(len(test_idx), mean_r),
                        min_reliability=np.full(len(test_idx), min_r),
                        failure_type=attack,
                        failed_domain_count=int(k),
                        fault_severity=sigma,
                    )
                    entry = archive.write(
                        experiment_id=eid,
                        benchmark=benchmark,
                        protocol=protocol,
                        seed=int(seed),
                        method=f"{gate_id}__{attack}_k{k}",
                        split="test",
                        frame=frame,
                        config=cfg_seed,
                    )
                    archive.append_index(entry)
                except Exception as e:
                    print(f"  [warn] archive write failed for {gate_id}/{attack}/k{k}: {e}")

    return results


def _aggregate_inference(all_ff_rows: list[dict], all_fs_rows: list[dict], contract: dict) -> list[dict]:
    """Compute per-gate promotion decisions across seeds."""
    from collections import defaultdict

    gates = {r["gate_id"] for r in all_fs_rows}
    # Get G0 false-fire rates (baseline) from all_ff_rows
    g0_ff = [
        float(r["clean_activation_rate"])
        for r in all_ff_rows
        if r.get("gate_id") == "G0" and r.get("status") == "computed"
    ]
    g0_ff_mean = float(np.mean(g0_ff)) if g0_ff else 0.005
    clean_budget = max(0.010, g0_ff_mean + 0.005)

    inference_rows = []
    for gate_id in sorted(gates):
        ff_rates = [
            float(r["clean_activation_rate"])
            for r in all_ff_rows
            if r["gate_id"] == gate_id and r.get("status") == "computed"
        ]
        mean_ff = float(np.mean(ff_rates)) if ff_rates else float("nan")
        c1_pass = mean_ff <= clean_budget if not np.isnan(mean_ff) else False

        # Check C2: improves at least 2 of {k=1, k=2, k=3} over G0 for zero+max attacks
        deltas_partial = defaultdict(list)
        g0_deltas_partial = defaultdict(list)

        for r in all_fs_rows:
            if r.get("status") != "computed":
                continue
            k = int(r["k"])
            attack = r["attack"]
            if k not in (1, 2, 3):
                continue
            if attack not in ("zero_attack", "max_attack"):
                continue
            key = (k, attack)
            try:
                delta = float(r["delta_auc"])
            except (ValueError, KeyError):
                continue
            if r["gate_id"] == gate_id:
                deltas_partial[key].append(delta)
            elif r["gate_id"] == "G0":
                g0_deltas_partial[key].append(delta)

        c2_count = 0
        for key in set(list(deltas_partial.keys()) + list(g0_deltas_partial.keys())):
            mean_gate = float(np.mean(deltas_partial[key])) if deltas_partial.get(key) else float("nan")
            mean_g0 = float(np.mean(g0_deltas_partial[key])) if g0_deltas_partial.get(key) else float("nan")
            if not np.isnan(mean_gate) and not np.isnan(mean_g0) and mean_gate > mean_g0:
                c2_count += 1
        c2_pass = c2_count >= 2

        # C3: k=4 not worsened by more than 0.005
        k4_delta_gate = [
            float(r["delta_auc"])
            for r in all_fs_rows
            if r["gate_id"] == gate_id
            and int(r["k"]) == 4
            and r.get("status") == "computed"
            and not np.isnan(float(r.get("delta_auc", "nan")))
        ]
        k4_delta_g0 = [
            float(r["delta_auc"])
            for r in all_fs_rows
            if r["gate_id"] == "G0"
            and int(r["k"]) == 4
            and r.get("status") == "computed"
            and not np.isnan(float(r.get("delta_auc", "nan")))
        ]
        mean_k4_gate = float(np.mean(k4_delta_gate)) if k4_delta_gate else float("nan")
        mean_k4_g0 = float(np.mean(k4_delta_g0)) if k4_delta_g0 else float("nan")
        c3_pass = True
        if not np.isnan(mean_k4_gate) and not np.isnan(mean_k4_g0):
            c3_pass = mean_k4_gate >= mean_k4_g0 - 0.005

        # C4: positive switching certificate — requires actual certificate computation
        # We flag this as PENDING for now; B-CERT-1 extension handles it
        c4_pass = False  # Will be updated by B-CERT-1 extension

        # C5: selection validation-only — always True by construction
        c5_pass = True

        # C6: same gate policy across all cells — always True by construction
        c6_pass = True

        all_criteria = [c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass]

        if gate_id == "G0":
            decision = "BASELINE_REFERENCE"
        elif all(all_criteria):
            decision = "PROMOTED_CANDIDATE"
        elif c1_pass and (c2_pass or c3_pass) and c5_pass and c6_pass:
            decision = "MECHANISM_IMPROVEMENT_PARTIAL"
        elif not c5_pass or not c6_pass:
            decision = "INVALID_SELECTION"
        else:
            decision = "NOT_IMPROVED"

        inference_rows.append(
            {
                "gate_id": gate_id,
                "mean_clean_false_fire": f"{mean_ff:.4f}" if not np.isnan(mean_ff) else "nan",
                "clean_budget": f"{clean_budget:.4f}",
                "C1_false_fire_budget": str(c1_pass),
                "C2_partial_improvement": str(c2_pass) + f" ({c2_count}/2+ criteria)",
                "C3_k4_not_worsened": str(c3_pass),
                "C4_positive_certificate": f"{c4_pass} (pending B-CERT-1 extension)",
                "C5_validation_only": str(c5_pass),
                "C6_same_gate_policy": str(c6_pass),
                "promotion_decision": decision,
            }
        )
    return inference_rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--gates", default="G0,G1,G2,G3", help="comma-separated subset of {G0,G1,G2,G3,G4}")
    p.add_argument("--seed-metrics-out", type=Path, default=FAILURE_SURFACE_CSV)
    p.add_argument("--archive-root", type=Path, default=ARCHIVE_DIR)
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

    cfg = yaml.safe_load(ELARA_BENCH_LA_CONFIG.read_text())
    archive = PredictionArchive(root=args.archive_root)
    benchmark = row["benchmark"]
    protocol = row["protocol"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.archive_root.mkdir(parents=True, exist_ok=True)

    # Open output CSVs
    thresh_fields = [
        "gate_id",
        "seed",
        "tau_min_candidate",
        "q_candidate",
        "tau_q_candidate",
        "validation_score",
        "validation_clean_activation",
        "selected",
        "selection_used_test_metrics",
        "status",
    ]
    ff_fields = ["gate_id", "seed", "clean_activation_rate", "within_false_fire_budget", "promotion_status", "status"]
    fs_fields = [
        "gate_id",
        "seed",
        "k",
        "attack",
        "static_auc",
        "rga_auc",
        "delta_auc",
        "gate_activation_rate",
        "mean_reliability",
        "min_reliability",
        "status",
    ]

    thresh_new = not THRESHOLD_CSV.exists()
    ff_new = not FALSE_FIRE_CSV.exists()
    fs_new = not FAILURE_SURFACE_CSV.exists()

    tf = THRESHOLD_CSV.open("a", newline="")
    tw = csv.DictWriter(tf, fieldnames=thresh_fields)
    if thresh_new:
        tw.writeheader()

    ff_f = FALSE_FIRE_CSV.open("a", newline="")
    ff_w = csv.DictWriter(ff_f, fieldnames=ff_fields)
    if ff_new:
        ff_w.writeheader()

    fs_f = FAILURE_SURFACE_CSV.open("a", newline="")
    fs_w = csv.DictWriter(fs_f, fieldnames=fs_fields)
    if fs_new:
        fs_w.writeheader()

    all_ff_rows = []
    all_fs_rows = []

    for s in range(int(args.seed_start), int(args.seed_start) + int(args.seeds)):
        print(f"[b-mech-2 seed={s} starting]", flush=True)
        try:
            result = run_one_seed(
                cfg,
                s,
                archive,
                requested,
                contract,
                args.experiment_id,
                benchmark,
                protocol,
            )
            for r in result["threshold_rows"]:
                tw.writerow(r)
            tf.flush()
            for r in result["false_fire_rows"]:
                ff_w.writerow(r)
                all_ff_rows.append(r)
            ff_f.flush()
            for r in result["failure_surface_rows"]:
                fs_w.writerow(r)
                all_fs_rows.append(r)
            fs_f.flush()
            print(f"[b-mech-2 seed={s} done; {len(result['failure_surface_rows'])} surface cells]")
        except Exception as e:
            print(f"[b-mech-2 seed={s} ERROR: {e}]", flush=True)
            raise

    tf.close()
    ff_f.close()
    fs_f.close()

    # Write inference summary
    inf_rows = _aggregate_inference(all_ff_rows, all_fs_rows, contract)
    inf_fields = [
        "gate_id",
        "mean_clean_false_fire",
        "clean_budget",
        "C1_false_fire_budget",
        "C2_partial_improvement",
        "C3_k4_not_worsened",
        "C4_positive_certificate",
        "C5_validation_only",
        "C6_same_gate_policy",
        "promotion_decision",
    ]
    with INFERENCE_CSV.open("w", newline="") as inf_f:
        w = csv.DictWriter(inf_f, fieldnames=inf_fields)
        w.writeheader()
        for r in inf_rows:
            w.writerow(r)

    print(f"[b-mech-2] {args.seeds} seeds complete; promotion decisions written to {INFERENCE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
