"""Phase 2.2B.2 / Step 5 — B-CERT-1 v2.

Consumes the B-MECH-1 archive (now including the k=0 clean arm) and
emits the v2 certification CSVs with full risk-dominance terms
(q0, q1, Δ0, Δ1, π*) plus the existing paired-bootstrap fired-subset
certificate.

Outputs:
  experiments/phase2/certification/risk_dominance_terms_v2.csv
  experiments/phase2/certification/switching_certificates_v2.csv

Per-row mandatory boundary text:
  "retrospective evaluation certificate under defined stress protocol;
   NOT a production safety certificate or real-world deployment guarantee"
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from elara.certification import (  # noqa: E402
    estimate_risk_dominance, fired_subset_certificate,
)

ARCHIVE = ROOT / "experiments" / "phase2" / "mechanism" / "b_mech_1_prediction_archives"
CERT_DIR = ROOT / "experiments" / "phase2" / "certification"
BOUNDARY = (
    "retrospective evaluation certificate under defined stress protocol; "
    "NOT a production safety certificate or real-world deployment guarantee"
)


def _load_per_seed_method(cell_dir: Path, method_dir_name: str) -> dict[int, pd.DataFrame]:
    p = cell_dir / method_dir_name / "test"
    out = {}
    if not p.exists():
        return out
    for f in sorted(p.glob("seed_*.parquet")):
        if f.name.startswith("._"):
            continue
        seed = int(f.stem.replace("seed_", ""))
        out[seed] = pd.read_parquet(f)
    return out


def _ensemble(per_seed: dict[int, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seeds = sorted(per_seed)
    canonical_ids = per_seed[seeds[0]]["sample_id"].to_numpy()
    canonical_labels = per_seed[seeds[0]]["label"].to_numpy().astype(int)
    stack = []
    for s in seeds:
        df = per_seed[s]
        if not np.array_equal(df["sample_id"].to_numpy(), canonical_ids):
            raise ValueError(f"sample_id mismatch at seed {s}")
        stack.append(df["raw_score"].to_numpy().astype(float))
    return canonical_ids, canonical_labels, np.mean(np.stack(stack, axis=0), axis=0)


def main() -> int:
    cell_dirs = [d for d in ARCHIVE.iterdir()
                 if d.is_dir() and not d.name.startswith("._")]
    if not cell_dirs:
        print("no archive")
        return 1
    cell_dir = cell_dirs[0]
    methods = {d.name for d in cell_dir.iterdir() if d.is_dir() and not d.name.startswith("._")}

    clean_static_name = "static_attention__clean_k0"
    clean_rga_name = "rga_mean_gate_tau66__clean_k0"
    if clean_static_name not in methods or clean_rga_name not in methods:
        print(f"clean k=0 arm missing; run run_phase2_b_mech_1_clean_arm.py first")
        return 2

    clean_static_per_seed = _load_per_seed_method(cell_dir, clean_static_name)
    clean_rga_per_seed = _load_per_seed_method(cell_dir, clean_rga_name)
    if not clean_static_per_seed or not clean_rga_per_seed:
        print("clean arm parquet files missing")
        return 2
    ids_clean, lbl_clean, static_clean = _ensemble(clean_static_per_seed)
    _, _, rga_clean = _ensemble(clean_rga_per_seed)
    # Empirical fired-subset definition: where ensemble RGA differs from static
    clean_fired = np.abs(rga_clean - static_clean) > 1e-6

    scenarios = [
        ("zero_attack_k4", "static_attention__zero_attack_k4", "rga_mean_gate_tau66__zero_attack_k4", "B1"),
        ("max_attack_k4",  "static_attention__max_attack_k4",  "rga_mean_gate_tau66__max_attack_k4",  "B2"),
    ]

    rd_rows = []
    cert_rows = []
    for scenario_id, static_name, rga_name, endpoint_label in scenarios:
        if static_name not in methods or rga_name not in methods:
            print(f"scenario {scenario_id}: missing pair")
            continue
        static_per = _load_per_seed_method(cell_dir, static_name)
        rga_per = _load_per_seed_method(cell_dir, rga_name)
        ids_d, lbl_d, static_deg = _ensemble(static_per)
        _, _, rga_deg = _ensemble(rga_per)
        deg_fired = np.abs(rga_deg - static_deg) > 1e-6

        if not np.array_equal(ids_d, ids_clean):
            print(f"scenario {scenario_id}: sample_id mismatch vs clean arm; "
                   "computing risk-dominance via independent samples (n_clean, n_deg)")
            # The dataset uses the same test split with the same sample ids
            # for both clean and degraded predictions; mismatch here is unexpected.
        rd = estimate_risk_dominance(
            gate_id="G0_mean_tau66",
            scenario_id=scenario_id,
            clean_static_scores=static_clean,
            clean_gated_scores=rga_clean,
            clean_gate_fired=clean_fired,
            clean_labels=lbl_clean,
            degraded_static_scores=static_deg,
            degraded_gated_scores=rga_deg,
            degraded_gate_fired=deg_fired,
            degraded_labels=lbl_d,
            notes=BOUNDARY,
        )
        rd_rows.append({
            "gate_id": rd.gate_id,
            "scenario_id": rd.scenario_id,
            "endpoint": endpoint_label,
            "q0_clean_fire_rate": rd.q0,
            "q1_degraded_fire_rate": rd.q1,
            "delta_0_clean_cost": rd.delta_0,
            "delta_1_degraded_benefit": rd.delta_1,
            "pi_star_indifference_prevalence": rd.pi_star,
            "n_clean_samples": rd.n_clean_samples,
            "n_degraded_samples": rd.n_degraded_samples,
            "boundary_notice": BOUNDARY,
        })

        cert = fired_subset_certificate(
            gate_id="G0_mean_tau66",
            scenario_id=scenario_id,
            static_scores=static_deg,
            gated_scores=rga_deg,
            labels=lbl_d,
            gate_fired=deg_fired,
            alpha=0.05, n_iter=10000, seed=0,
            notes=BOUNDARY,
        )
        cert_rows.append({
            "gate_id": cert.gate_id,
            "scenario_id": cert.scenario_id,
            "endpoint": endpoint_label,
            "n_fired_samples": cert.n_fired_samples,
            "mean_paired_benefit": cert.mean_paired_benefit,
            "bootstrap_lcb": cert.bootstrap_lcb,
            "alpha": cert.alpha,
            "n_iter": cert.n_iter,
            "certified": cert.certified,
            "boundary_notice": BOUNDARY,
        })
        print(f"{endpoint_label} {scenario_id}: q0={rd.q0:.4f} q1={rd.q1:.4f} "
              f"Δ0={rd.delta_0:+.4f} Δ1={rd.delta_1:+.4f} π*={rd.pi_star:.4f} "
              f"cert_LCB={cert.bootstrap_lcb:+.4f} certified={cert.certified}")

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    with (CERT_DIR / "risk_dominance_terms_v2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rd_rows[0].keys()))
        w.writeheader()
        for r in rd_rows:
            w.writerow(r)
    with (CERT_DIR / "switching_certificates_v2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cert_rows[0].keys()))
        w.writeheader()
        for r in cert_rows:
            w.writerow(r)
    print(f"wrote {CERT_DIR / 'risk_dominance_terms_v2.csv'}")
    print(f"wrote {CERT_DIR / 'switching_certificates_v2.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
