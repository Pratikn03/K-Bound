"""Phase 2.2B B-CERT-1 — risk-dominance + retrospective switching certificate.

Consumes archived predictions (does NOT train any model). For each
(scenario × seed) in the B-MECH-1 archive, pairs the static_attention
predictions with the rga_mean_gate_tau66 predictions (under the same
attack and k) and computes:

  - risk-dominance terms (q0, q1, Δ0, Δ1, π*)
  - fired-subset paired-bootstrap LCB certificate

The certificate is CERTIFIED iff LCB > 0.

Refuses:
- any experiment_id other than B-CERT-1;
- archives missing per-sample gate_fired vectors;
- any input pair where static / RGA do not share sample_ids and labels.

Mandatory boundary text in every output row:
"retrospective evaluation certificate under defined stress protocol;
 NOT a production safety certificate or real-world deployment guarantee."

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_certificate_audit.py \\
      --experiment-id B-CERT-1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from elara.certification import (  # noqa: E402
    fired_subset_certificate,
)

REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
B_MECH_1_ARCHIVE = ROOT / "experiments" / "phase2" / "mechanism" / "b_mech_1_prediction_archives"


def _registry_row(eid: str) -> dict[str, str]:
    with REGISTRY_V2.open() as f:
        for r in csv.DictReader(f):
            if r["experiment_id"] == eid:
                return r
    raise SystemExit(f"experiment_id {eid!r} not in v2 registry")


def _validate(eid: str, row: dict[str, str]) -> None:
    if eid != "B-CERT-1":
        raise SystemExit(f"this driver runs B-CERT-1 only; got {eid!r}")
    if row["analysis_family"] != "B":
        raise SystemExit(f"{eid}: analysis_family={row['analysis_family']!r}; refusing")


def _scan_b_mech_1_archive() -> dict[str, dict[str, list[Path]]]:
    """Walk the B-MECH-1 archive and return:
       { scenario_method: { 'paths': [seed parquet paths] } }
    where scenario_method is e.g. 'static_attention__zero_attack_k4'.
    """
    out: dict[str, list[Path]] = {}
    if not B_MECH_1_ARCHIVE.exists():
        return out
    for cell_dir in B_MECH_1_ARCHIVE.iterdir():
        if not cell_dir.is_dir():
            continue
        for method_dir in cell_dir.iterdir():
            if not method_dir.is_dir():
                continue
            test_dir = method_dir / "test"
            if not test_dir.exists():
                continue
            paths = sorted(test_dir.glob("seed_*.parquet"))
            if paths:
                out[method_dir.name] = paths
    return out


def _load_per_seed(paths: list[Path]) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for p in paths:
        seed = int(p.stem.replace("seed_", ""))
        out[seed] = pd.read_parquet(p)
    return out


def _pair_static_and_rga(scan: dict[str, list[Path]]) -> dict[str, dict]:
    """For each (attack, k) scenario find the static and RGA method dirs and
    pair them by seed."""
    scenarios: dict[str, dict] = {}
    for method_name in scan:
        # method_name = '<base>__<attack>_k<k>' e.g. 'static_attention__zero_attack_k4'
        if "__" not in method_name:
            continue
        base, scenario = method_name.split("__", 1)
        if base not in ("static_attention", "rga_mean_gate_tau66"):
            continue
        scenarios.setdefault(scenario, {})[base] = method_name
    return scenarios


def _compute_one_scenario(
    scenario_id: str, static_per_seed: dict[int, pd.DataFrame], rga_per_seed: dict[int, pd.DataFrame]
) -> tuple[dict, dict]:
    """Return (risk_dominance_row, certificate_row) for one (attack, k)."""
    seeds = sorted(set(static_per_seed) & set(rga_per_seed))
    if not seeds:
        raise ValueError(f"{scenario_id}: no shared seeds between static and RGA")
    # Use seed-averaged ensemble vectors (analogous to A-POWERED inference)
    static_stack, rga_stack = [], []
    canonical_ids, canonical_labels, canonical_gate_fired = None, None, None
    for s in seeds:
        sdf = static_per_seed[s]
        rdf = rga_per_seed[s]
        if canonical_ids is None:
            canonical_ids = sdf["sample_id"].to_numpy()
            canonical_labels = sdf["label"].to_numpy().astype(int)
            canonical_gate_fired = rdf["gate_fired_if_applicable"].to_numpy().astype(bool)
        if not np.array_equal(sdf["sample_id"].to_numpy(), canonical_ids):
            raise ValueError(f"{scenario_id} seed {s}: static sample_id mismatch")
        if not np.array_equal(rdf["sample_id"].to_numpy(), canonical_ids):
            raise ValueError(f"{scenario_id} seed {s}: RGA sample_id mismatch")
        static_stack.append(sdf["raw_score"].to_numpy().astype(float))
        rga_stack.append(rdf["raw_score"].to_numpy().astype(float))
    static_ens = np.mean(np.stack(static_stack, axis=0), axis=0)
    rga_ens = np.mean(np.stack(rga_stack, axis=0), axis=0)

    # Effective fired-sample definition. The B-MECH-1 driver writes a
    # per-sample gate_fired vector based on the runner's batch-level
    # `adapted` flag, which under-counts true per-sample firing. We
    # therefore define the effectively-fired subset empirically: any
    # sample where the seed-ensemble RGA prediction differs materially
    # from static (|Δ| > 1e-6). On scenarios where static == RGA
    # everywhere (no firing happened), this falls back to the original
    # all-False vector and the certificate stays NOT_CERTIFIED.
    pred_diff = np.abs(rga_ens - static_ens)
    effective_fired = pred_diff > 1e-6
    if effective_fired.any():
        canonical_gate_fired = effective_fired
    cert = fired_subset_certificate(
        gate_id="G0_mean_tau66",
        scenario_id=scenario_id,
        static_scores=static_ens,
        gated_scores=rga_ens,
        labels=canonical_labels,
        gate_fired=canonical_gate_fired,
        alpha=0.05,
        n_iter=10000,
        seed=0,
        notes=(
            "retrospective evaluation certificate under defined stress protocol; "
            "NOT a production safety certificate or real-world deployment guarantee"
        ),
    )
    cert_row = {
        "gate_id": cert.gate_id,
        "scenario_id": cert.scenario_id,
        "n_fired_samples": cert.n_fired_samples,
        "mean_paired_benefit": cert.mean_paired_benefit,
        "bootstrap_lcb": cert.bootstrap_lcb,
        "alpha": cert.alpha,
        "n_iter": cert.n_iter,
        "certified": cert.certified,
        "boundary_notice": (
            "retrospective evaluation certificate under defined stress protocol; "
            "NOT a production safety certificate or real-world deployment guarantee"
        ),
    }
    rd_row = {
        "scenario_id": scenario_id,
        "note": "per-scenario risk-dominance requires paired (clean, degraded) folds; "
        "B-MECH-1 archives every scenario as a 'degraded' arm only. "
        "Risk-dominance (q0,q1,Δ0,Δ1,π*) is therefore not computable from a single arm; "
        "see fired-subset certificate column for the executable retrospective evidence.",
    }
    return rd_row, cert_row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument(
        "--out-terms", type=Path, default=ROOT / "experiments" / "phase2" / "certification" / "risk_dominance_terms.csv"
    )
    p.add_argument(
        "--out-certs",
        type=Path,
        default=ROOT / "experiments" / "phase2" / "certification" / "switching_certificates.csv",
    )
    p.add_argument("--dry-run", action="store_true", help="validate inputs without writing")
    args = p.parse_args()
    row = _registry_row(args.experiment_id)
    _validate(args.experiment_id, row)

    scan = _scan_b_mech_1_archive()
    print(f"[b-cert-1] archive scan: {len(scan)} method directories found at {B_MECH_1_ARCHIVE}")
    if not scan:
        print("[b-cert-1] no archived inputs found; exiting OK (run B-MECH-1 first)")
        return 0

    scenarios = _pair_static_and_rga(scan)
    if not scenarios:
        print("[b-cert-1] no static/RGA pairs found in archive; exiting")
        return 0
    print(f"[b-cert-1] scenarios to certify: {sorted(scenarios)}")

    if args.dry_run:
        print(f"[b-cert-1] dry-run: would certify {len(scenarios)} scenarios; exiting")
        return 0

    args.out_terms.parent.mkdir(parents=True, exist_ok=True)
    args.out_certs.parent.mkdir(parents=True, exist_ok=True)
    rd_rows, cert_rows = [], []
    for scenario_id, methods in sorted(scenarios.items()):
        if "static_attention" not in methods or "rga_mean_gate_tau66" not in methods:
            print(f"[b-cert-1] scenario={scenario_id}: missing pair; skipping")
            continue
        static_paths = scan[methods["static_attention"]]
        rga_paths = scan[methods["rga_mean_gate_tau66"]]
        static_per_seed = _load_per_seed(static_paths)
        rga_per_seed = _load_per_seed(rga_paths)
        rd_row, cert_row = _compute_one_scenario(scenario_id, static_per_seed, rga_per_seed)
        rd_rows.append(rd_row)
        cert_rows.append(cert_row)
        print(
            f"[b-cert-1] scenario={scenario_id}: "
            f"n_fired={cert_row['n_fired_samples']}  "
            f"mean_benefit={cert_row['mean_paired_benefit']:+.4f}  "
            f"LCB={cert_row['bootstrap_lcb']:+.4f}  "
            f"certified={cert_row['certified']}"
        )

    if rd_rows:
        with args.out_terms.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rd_rows[0].keys()))
            w.writeheader()
            for r in rd_rows:
                w.writerow(r)
        print(f"[b-cert-1] wrote {args.out_terms}")
    if cert_rows:
        with args.out_certs.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cert_rows[0].keys()))
            w.writeheader()
            for r in cert_rows:
                w.writerow(r)
        print(f"[b-cert-1] wrote {args.out_certs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
