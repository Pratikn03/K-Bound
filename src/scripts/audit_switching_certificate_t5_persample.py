"""Per-sample finite-sample switching-certificate audit (T5, Phase-2).

The per-SEED audit (audit_switching_certificate_t5.py) operates on n=5-30
seeds with an aggregate-AUROC surrogate whose range is R=2. At that n the
empirical-Bernstein range term 7 R ln(2/alpha) / (3(n-1)) is ~4, so the
closed-form bound is vacuous -- which honestly reveals that the n=5
bootstrap certificates are NOT finite-sample valid.

This script computes the certificate at the level the theorem actually
addresses: the per-SAMPLE fired subset, where n is in the hundreds-to-
thousands. There, both the paired bootstrap and the empirical-Bernstein
closed form become meaningful, and the closed form is deterministic and
streamable.

It reads the B-MECH-1 per-sample prediction archives (static vs the
mean-gate RGA path under zero_attack and max_attack k=4 on ELARA-Bench-LA),
restricts to the fired subset, and reports:
  - n_fired
  - mean paired benefit  X_i = |p_static - y| - |p_rga - y|
  - paired-bootstrap LCB
  - empirical-Bernstein LCB (closed form)
  - certified? under each rule

Writes:
  experiments/fusion/switching_certificate_t5_persample_audit.json
  docs/research/tables/switching_certificate_t5_persample.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from elara.certification.switching_certificate import (  # noqa: E402
    empirical_bernstein_lcb,
    paired_bootstrap_lcb,
)

ARCHIVE_INDEX = ROOT / "experiments/phase2/mechanism/b_mech_1_prediction_archives/PREDICTION_ARCHIVE_INDEX.csv"
TAU = 0.66   # documented mean-gate threshold


def _load_method_frames(idx: pd.DataFrame, method_token: str) -> dict[int, pd.DataFrame]:
    """Load per-seed parquet frames whose method column contains method_token."""
    out: dict[int, pd.DataFrame] = {}
    sub = idx[idx["method"].astype(str).str.contains(method_token, regex=False)]
    for _, r in sub.iterrows():
        p = ROOT / r["artifact_path"] if not Path(r["artifact_path"]).is_absolute() else Path(r["artifact_path"])
        if not p.exists() or p.name.startswith("._"):
            continue
        out[int(r["seed"])] = pd.read_parquet(p)
    return out


def _loss(y, p):
    return np.abs(np.asarray(p, dtype=float) - np.asarray(y, dtype=float))


def audit_condition(idx: pd.DataFrame, condition: str, alpha: float, seed: int = 0) -> dict | None:
    """Pool fired-subset paired benefits across all seeds for one attack condition."""
    static_token = f"static_attention__{condition}"
    rga_token = f"rga_mean_gate_tau66__{condition}"
    static_frames = _load_method_frames(idx, static_token)
    rga_frames = _load_method_frames(idx, rga_token)
    common = sorted(set(static_frames) & set(rga_frames))
    if not common:
        return None

    benefits: list[float] = []
    n_fired_total = 0
    n_total = 0
    for s in common:
        sf = static_frames[s].set_index("sample_id")
        rf = rga_frames[s].set_index("sample_id")
        ids = sf.index.intersection(rf.index)
        sf = sf.loc[ids]
        rf = rf.loc[ids]
        y = sf["label"].to_numpy()
        n_total += len(ids)
        # The archived gate_fired_if_applicable column is a constant-False
        # placeholder in the B-MECH-1 archives, but the per-sample
        # mean_reliability_if_applicable IS logged. We therefore reconstruct
        # the fired flag from the DOCUMENTED gate rule (fire iff mean
        # reliability < tau=0.66), applied to the logged reliability signal.
        fired_col = "gate_fired_if_applicable"
        rel_col = "mean_reliability_if_applicable"
        stored = rf[fired_col].fillna(False).to_numpy().astype(bool) if fired_col in rf.columns else np.zeros(len(ids), bool)
        if stored.any():
            fired = stored
        elif rel_col in rf.columns:
            fired = rf[rel_col].to_numpy() < TAU
        else:
            fired = np.zeros(len(ids), bool)
        if not fired.any():
            continue
        l_static = _loss(y[fired], sf["raw_score"].to_numpy()[fired])
        l_rga = _loss(y[fired], rf["raw_score"].to_numpy()[fired])
        benefits.extend((l_static - l_rga).tolist())
        n_fired_total += int(fired.sum())

    if not benefits:
        return {
            "condition": condition,
            "n_seeds": len(common),
            "n_fired": 0,
            "n_total": n_total,
            "note": "no fired samples in any seed",
        }

    boot_mean, boot_lcb = paired_bootstrap_lcb(benefits, alpha=alpha, n_iter=10_000, seed=seed)
    eb_mean, eb_lcb, eb_var = empirical_bernstein_lcb(benefits, alpha=alpha, benefit_range=2.0)
    return {
        "condition": condition,
        "n_seeds": len(common),
        "n_fired": int(n_fired_total),
        "n_total": int(n_total),
        "fire_rate": float(n_fired_total / max(n_total, 1)),
        "mean_paired_benefit": float(boot_mean),
        "bootstrap_lcb": float(boot_lcb),
        "empirical_bernstein_lcb": float(eb_lcb),
        "sample_variance": float(eb_var),
        "alpha": float(alpha),
        "certified_bootstrap": bool(boot_lcb > 0.0),
        "certified_eb": bool(eb_lcb > 0.0),
    }


def emit_tex(rows: list[dict], alpha: float) -> str:
    out = [
        "% Auto-generated by audit_switching_certificate_t5_persample.py",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Condition} & $n_{\mathrm{fired}}$ & "
        r"\textbf{Mean benefit} & \textbf{Boot. LCB} & "
        r"\textbf{EB LCB (closed-form)} & \textbf{Certified?} \\",
        r"\midrule",
    ]
    for r in rows:
        if r.get("n_fired", 0) == 0:
            out.append(rf"{r['condition'].replace('_', chr(92)+'_')} & 0 & -- & -- & -- & -- \\")
            continue
        cond = r["condition"].replace("_", r"\_")
        cert = r"\textbf{yes}" if r["certified_eb"] else "no"
        out.append(
            rf"{cond} & {r['n_fired']} & {r['mean_paired_benefit']:+.4f} & "
            rf"{r['bootstrap_lcb']:+.4f} & {r['empirical_bernstein_lcb']:+.4f} & {cert} \\"
        )
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        rf"% Empirical-Bernstein LCB (Maurer-Pontil 2009) at alpha={alpha:.2f}, range R=2.",
        r"% Per-sample fired subset (n in hundreds-thousands) -- the regime where the",
        r"% closed-form bound is tight. Deterministic, streamable, finite-sample valid.",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-json", type=Path,
                        default=ROOT / "experiments/fusion/switching_certificate_t5_persample_audit.json")
    parser.add_argument("--out-tex", type=Path,
                        default=ROOT / "docs/research/tables/switching_certificate_t5_persample.tex")
    args = parser.parse_args()

    if not ARCHIVE_INDEX.exists():
        print(f"-- missing archive index: {ARCHIVE_INDEX}")
        return 1
    idx = pd.read_csv(ARCHIVE_INDEX)

    rows = []
    for condition in ("zero_attack_k4", "max_attack_k4", "clean_k0"):
        row = audit_condition(idx, condition, alpha=args.alpha)
        if row is None:
            print(f"-- skipped {condition} (no archives)")
            continue
        rows.append(row)
        if row.get("n_fired", 0) == 0:
            print(f"{condition:<16s} n_fired=0  (gate never fired)")
        else:
            print(
                f"{condition:<16s} n_fired={row['n_fired']:>5d}  "
                f"mean={row['mean_paired_benefit']:+.4f}  "
                f"boot_LCB={row['bootstrap_lcb']:+.4f}  "
                f"EB_LCB={row['empirical_bernstein_lcb']:+.4f}  "
                f"{'CERT_EB' if row['certified_eb'] else 'no_EB'}"
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"alpha": args.alpha, "rows": rows}, indent=2))
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(emit_tex(rows, args.alpha))
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
