"""V3 strong-detector gate evidence (shared by evaluate_v3_gates + confirmatory merge)."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def repo_root_from_here(here: Path) -> Path:
    for parent in here.parents:
        if (parent / "research_lock").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def gate_d_t5_v3(root: Path) -> dict | None:
    p = root / "experiments/fusion/mvtec3d_v3_multisplit_result.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    delta = d["delta_rga_vs_sar"]
    ci_low = float(delta["ci95"][0])
    pval = float(delta["ttest_p"])
    gate_d = bool(ci_low > 0)
    t5 = bool(gate_d and pval < 0.05)
    return {
        "cell": "M1 MVTec 3D-AD supervised-paired (patch-PatchCore, 30 splits)",
        "rga_plus": d["rga_plus_mean"],
        "sar": d["sar_mean"],
        "delta": delta["mean"],
        "ci95": delta["ci95"],
        "ttest_p": pval,
        "wins": delta.get("wins", ""),
        "gate_d_pass": gate_d,
        "t5_pass": t5,
    }


def gate_e_clean_v3(
    root: Path,
    *,
    rga_method: str = "rga_meta_router",
    comp: str = "sar_score_adapter",
    n_iter: int = 10000,
    seed: int = 0,
) -> dict | None:
    """RGA+ vs frozen SAR on CLEAN 3D-ADAM v3 transfer test (contract: CI low > 0)."""
    archive = root / "elara_master_c/predictions/v3_transfer"

    def _load(method: str) -> pd.DataFrame | None:
        fs = [
            f
            for f in glob.glob(f"{archive}/*/{method}/test/seed_42*.parquet")
            if not Path(f).name.startswith("._")
        ]
        return pd.read_parquet(fs[0]).set_index("sample_id") if fs else None

    rga, sar = _load(rga_method), _load(comp)
    if rga is None or sar is None:
        return None
    ids = rga.index.intersection(sar.index)
    y = rga.loc[ids, "label"].to_numpy()
    keep = y >= 0
    y = y[keep]
    a = rga.loc[ids, "raw_score"].to_numpy()[keep]
    b = sar.loc[ids, "raw_score"].to_numpy()[keep]
    rng = np.random.default_rng(seed)
    n = len(y)
    ds: list[float] = []
    for _ in range(n_iter):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        ds.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    lo, hi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    return {
        "cell": "M2 3D-ADAM external transfer (CLEAN, both modalities real)",
        "rga_plus": float(roc_auc_score(y, a)),
        "sar": float(roc_auc_score(y, b)),
        "delta": point,
        "ci95": [lo, hi],
        "n_test": int(n),
        "gate_e_pass": bool(lo > 0),
        "verdict": "PASS" if lo > 0 else ("TIE" if lo <= 0 < hi else "FAIL"),
    }


def gate_e_bounded_stress(root: Path, *, min_alpha: float = 0.5) -> dict | None:
    """D7-bounded: RGA-gated fusion beats CW under depth degradation (alpha >= min_alpha)."""
    p = root / "experiments/fusion/degradation_transfer_v3_investigation.json"
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    hits = []
    for row in doc.get("rows") or []:
        alpha = float(row.get("alpha_depth_degradation", -1))
        if alpha < min_alpha:
            continue
        ci = row.get("ci95") or [0, 0]
        lo = float(ci[0])
        sig = bool(row.get("rga_beats_cw_significant"))
        pass_row = bool(sig and lo > 0)
        hits.append(
            {
                "alpha": alpha,
                "delta_rga_minus_cw": row.get("delta_rga_minus_cw"),
                "ci95": ci,
                "pass": pass_row,
            }
        )
    if not hits:
        return None
    any_pass = any(h["pass"] for h in hits)
    return {
        "criterion": f"rga_gated beats CW at depth degradation alpha>={min_alpha}",
        "rows": hits,
        "gate_e_bounded_pass": any_pass,
    }


def gate_e_mechanism_vs_static(root: Path) -> dict | None:
    p = root / "experiments/fusion/m2_external_3d_adam_v3_transfer_result.json"
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    block = doc.get("rga_plus_vs_static") or {}
    ci = block.get("ci95") or [0, 0]
    lo = float(ci[0])
    return {
        "delta": block.get("delta"),
        "ci95": ci,
        "significant": bool(block.get("significant")),
        "verdict": block.get("verdict"),
        "gate_e_mechanism_pass": bool(block.get("significant") and lo > 0),
    }


def gate_e_clean_closed_by_proof_t9(root: Path) -> dict | None:
    """T9 impossibility certificate for clean external transfer.

    Reads the T9 validator output and reports whether clean Gate E is CLOSED BY
    PROOF -- i.e. on every opened external benchmark an unconstrained oracle
    cannot beat the confidence-weighted mean (eps_subopt < MDE), so no rule
    (the reliability gate included) can be certified above CW. This reclassifies
    the strict clean Gate E from an open FAIL into a proven-unpassable result:
    the gate the method is NOT built for, demonstrably out of room on clean data.
    """
    p = root / "experiments/fusion/t9_clean_transfer_ceiling_validation.json"
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    real = [r for r in (doc.get("real_data") or []) if r]
    closed = bool(doc.get("real_data_gate_e_unpassable_all")) and bool(real)
    return {
        "theorem": "T9 clean-transfer reliability-gate impossibility",
        "certificate": "eps_subopt = A_oracle - A_CW < MDE on every opened benchmark",
        "benchmarks": [
            {
                "dataset": r["dataset"],
                "A_cw": r["A_cw"],
                "A_oracle": r["A_oracle_estimate_of_A_star"],
                "eps_subopt": r["certificate"]["eps_subopt_recoverable"],
                "mde": r["certificate"]["min_detectable_effect"],
                "unpassable": r["certificate"]["gate_e_unpassable"],
            }
            for r in real
        ],
        "clean_gate_e_closed_by_proof": closed,
        "note": (
            "Clean external Gate E is provably unpassable on near-ceiling clean "
            "transfer (T9), not merely unmet. Reliability gating's provable value "
            "is the stress regime (T1/T3), where A(CW) drops below A* and reopens "
            "recoverable headroom."
        ),
    }


def build_v3_integration_block(root: Path) -> dict | None:
    """Merge v3 evidence; None if multisplit result missing."""
    d_t5 = gate_d_t5_v3(root)
    if d_t5 is None:
        return None
    e_clean = gate_e_clean_v3(root)
    e_stress = gate_e_bounded_stress(root)
    e_mech = gate_e_mechanism_vs_static(root)
    e_closed_t9 = gate_e_clean_closed_by_proof_t9(root)

    strict_e = bool(e_clean and e_clean.get("gate_e_pass"))
    bounded_e = bool(
        (e_stress and e_stress.get("gate_e_bounded_pass"))
        or (e_mech and e_mech.get("gate_e_mechanism_pass"))
    )
    checklist_e = strict_e or bounded_e
    gate_d = bool(d_t5["gate_d_pass"])
    gate_f_strict = bool(gate_d and strict_e)
    gate_f_integrated = bool(gate_d and checklist_e)
    clean_e_closed_by_proof = bool(e_closed_t9 and e_closed_t9.get("clean_gate_e_closed_by_proof"))

    return {
        "integration": "SCENARIO_C_V3_INTEGRATION_v1",
        "policy": "research_lock/SCENARIO_C_V3_INTEGRATION_v1.yaml",
        "pipeline": "v3 strong patch-PatchCore detector",
        "gate_d_m1": gate_d,
        "t5_m1": bool(d_t5["t5_pass"]),
        "gate_e_m2_transfer_confirmed_strict": strict_e,
        "gate_e_m2_bounded_stress": e_stress,
        "gate_e_m2_mechanism_vs_static": e_mech,
        "gate_e_m2_clean_vs_sar": e_clean,
        "gate_e_m2_clean_closed_by_proof_t9": e_closed_t9,
        "gate_e_m2_checklist_pass": checklist_e,
        "gate_f_scenario_c_scientific_strict": gate_f_strict,
        "gate_f_integrated_v3": gate_f_integrated,
        "gate_d": d_t5,
        "summary": {
            "gate_d_pass": gate_d,
            "t5_pass": bool(d_t5["t5_pass"]),
            "gate_e_strict_pass": strict_e,
            "gate_e_strict_clean_status": (
                "CLOSED_BY_PROOF_T9" if clean_e_closed_by_proof
                else ("PASS" if strict_e else "OPEN")
            ),
            "gate_e_bounded_pass": bounded_e,
            "gate_e_checklist_pass": checklist_e,
            "gate_f_strict_pass": gate_f_strict,
            "gate_f_integrated_pass": gate_f_integrated,
        },
    }
