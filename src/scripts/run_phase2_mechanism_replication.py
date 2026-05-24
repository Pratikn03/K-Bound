"""Phase 2.2B B-MECH-1 — primary B1/B2 coherent-collapse replication.

Runs ELARA-Bench-LA × {zero_attack, max_attack} × k=4 × G0 mean-gate
at τ=0.66 over 30 seeds. Archives per-sample static and CRAF (RGA)
predictions in the Phase-2.B 28-column parquet schema.

Validation-only selection invariants:
- Gate threshold τ=0.66 is LOCKED (Phase-2 contract);
- No test-fold reads inform gate selection;
- selection_used_test_metrics=False stamped per archived row.

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_mechanism_replication.py \\
      --experiment-id B-MECH-1 --seeds 30 --seed-start 42
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
from elara.family_b.corruption import inject_corruption  # noqa: E402

REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
ELARA_BENCH_LA_CONFIG = ROOT / "configs" / "attention_real_fusion.yaml"

# Locked Phase-2 contract for B-MECH-1.
LOCKED = {
    "benchmark": "ELARA-Bench-LA",
    "protocol": "k-of-D corruption k=4 mean-gate at locked tau=0.66 (B1+B2 endpoints)",
    "attacks": ("zero_attack", "max_attack"),
    "k_values": (4,),  # B-MECH-1 is coherent-collapse only
    "gate_mode": "mean",
    "tau_mean": 0.66,
    "sigma": 1.0,
    "pairing_strength": "label_aligned_stress_only",
}


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
    if eid != "B-MECH-1":
        raise SystemExit(f"this driver runs B-MECH-1 only; got {eid!r}")
    if row["analysis_family"] != "B":
        raise SystemExit(f"{eid}: registry analysis_family={row['analysis_family']!r}; refusing")
    if row["benchmark"] != LOCKED["benchmark"]:
        raise SystemExit(
            f"{eid}: registry benchmark={row['benchmark']!r}; expected {LOCKED['benchmark']!r}"
        )


def run_one_seed(cfg, seed, archive, eid, benchmark, protocol):
    device = _device()
    set_seed(int(seed))
    cfg_seed = dict(cfg)
    cfg_seed["training"] = dict(cfg.get("training", {}))
    cfg_seed["training"]["seed"] = int(seed)

    (features, masks, labels, sample_ids, domain_order, _, conf_idx,
     score_idx, sample_splits, _) = _load_data(cfg_seed)
    train_idx, val_idx, test_idx = _split(labels, cfg_seed["training"],
                                           split_values=sample_splits)
    train_loader, val_loader, _ = _make_loaders(
        features, masks, labels, train_idx, val_idx, test_idx,
        batch_size=int(cfg_seed["training"].get("batch_size", 64)),
    )
    model = _build_model(cfg_seed, features.shape[1], features.shape[2], conf_idx, device)
    _train_model(model, train_loader, val_loader, cfg_seed, device)
    model.eval()

    rel_cfg = cfg_seed.get("reliability", {})
    estimator = _make_reliability_estimator(rel_cfg, list(domain_order) or ["d0", "d1"], score_idx)
    estimator.fit(features[train_idx], masks[train_idx], labels[train_idx])
    # Lock gate mode + threshold per B-MECH-1 contract.
    estimator.gate_mode = LOCKED["gate_mode"]
    estimator.gate_threshold = LOCKED["tau_mean"]

    test_feat = features[test_idx]
    test_mask = masks[test_idx]
    test_labels = labels[test_idx]
    test_sids = [str(sample_ids[i]) for i in test_idx]

    # Build (attack, k=4) corrupted test conditions.
    out = []
    for attack in LOCKED["attacks"]:
        conds = inject_corruption(
            test_feat, test_mask,
            domain_order=list(domain_order), score_index=score_idx,
            attack_name=attack, k_values=list(LOCKED["k_values"]),
            sigma=LOCKED["sigma"], seed=int(seed) + 41_000,
        )
        # For k=4 there is exactly one subset (all D=4 domains)
        for cond in conds:
            static_probs = _predict_static(model, cond.features, cond.masks, device)
            craf_probs, gate_stats = _predict_craf_with_stats(
                model, estimator, cond.features, cond.masks, device,
                clean_gate_threshold=LOCKED["tau_mean"], per_sample_gating=False,
            )
            # Archive both methods under the same (cell, attack, k) slice
            for method, scores in (("static_attention", static_probs),
                                    ("rga_mean_gate_tau66", craf_probs)):
                frame = archive.build_frame(
                    sample_ids=test_sids,
                    labels=np.asarray(test_labels, dtype=int),
                    raw_scores=np.asarray(scores, dtype=float),
                    method=method,
                    method_variant=f"{attack}__k{cond.failed_domain_count}",
                    benchmark=benchmark, protocol=protocol,
                    analysis_family="B", pairing_strength=LOCKED["pairing_strength"],
                    split="test", seed=int(seed),
                    selection_rule=(
                        "validation-only: gate threshold tau=0.66 LOCKED by Phase-2 contract; "
                        "no test-fold reads inform gate selection"
                    ),
                    selection_used_test_metrics=False,
                    selected_head_or_comparator_status=(
                        "RGA G0 mean-gate" if method == "rga_mean_gate_tau66"
                        else "static reference"
                    ),
                    gate_mode=LOCKED["gate_mode"],
                    gate_fired=np.ones(len(test_sids), dtype=bool) if (
                        method == "rga_mean_gate_tau66" and gate_stats.get("adapted")
                    ) else np.zeros(len(test_sids), dtype=bool),
                    mean_reliability=np.full(len(test_sids), float(gate_stats.get("mean_reliability", 0.0))),
                    min_reliability=np.full(len(test_sids), float(gate_stats.get("min_reliability", 0.0))),
                    failure_type=attack,
                    failed_domain_count=int(cond.failed_domain_count),
                    fault_severity=LOCKED["sigma"],
                )
                entry = archive.write(
                    experiment_id=eid, benchmark=benchmark, protocol=protocol,
                    seed=int(seed), method=f"{method}__{attack}_k{cond.failed_domain_count}",
                    split="test", frame=frame, config=cfg_seed,
                )
                archive.append_index(entry)
            out.append({
                "seed": int(seed), "attack": attack,
                "k": int(cond.failed_domain_count),
                "static_test_auc": _safe_auc(test_labels, static_probs),
                "rga_test_auc": _safe_auc(test_labels, craf_probs),
                "adapted": bool(gate_stats.get("adapted", False)),
                "mean_reliability": float(gate_stats.get("mean_reliability", 0.0)),
                "min_reliability": float(gate_stats.get("min_reliability", 0.0)),
            })
    return out


def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--seed-metrics-out", type=Path,
                   default=ROOT / "experiments" / "phase2" / "mechanism"
                   / "family_b_primary_replication_seed_metrics.csv")
    p.add_argument("--archive-root", type=Path,
                   default=ROOT / "experiments" / "phase2" / "mechanism" / "b_mech_1_prediction_archives")
    args = p.parse_args()

    row = _registry_row(args.experiment_id)
    _validate(args.experiment_id, row)
    if int(args.seeds) <= 0:
        print(f"[b-mech-1 {args.experiment_id}] validation-only invocation; exiting OK")
        return 0

    cfg = yaml.safe_load(ELARA_BENCH_LA_CONFIG.read_text())
    archive = PredictionArchive(root=args.archive_root)
    args.seed_metrics_out.parent.mkdir(parents=True, exist_ok=True)

    fields = ["experiment_id", "seed", "attack", "k", "static_test_auc",
              "rga_test_auc", "adapted", "mean_reliability", "min_reliability"]
    new = not args.seed_metrics_out.exists()
    f = args.seed_metrics_out.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()

    benchmark = row["benchmark"]
    protocol = row["protocol"]
    for s in range(int(args.seed_start), int(args.seed_start) + int(args.seeds)):
        print(f"[b-mech-1 seed={s} starting]", flush=True)
        rows = run_one_seed(cfg, s, archive, args.experiment_id, benchmark, protocol)
        for r in rows:
            w.writerow({"experiment_id": args.experiment_id, **r})
        f.flush()
        print(f"[b-mech-1 seed={s} done; {len(rows)} conditions archived]", flush=True)
    f.close()
    print(f"[b-mech-1] {args.seeds} seeds complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
