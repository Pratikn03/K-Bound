"""Build multimodal fusion inputs from PhysioNet BIDMC numerics (local raw data)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DOMAIN_ORDER = ["heart_rate", "oxygenation", "respiration", "shock_index"]
CASE_RE = re.compile(r"bidmc_(\d+)_Numerics\.csv$", re.IGNORECASE)


def _minmax(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    fit = values[fit_mask]
    fit = fit[np.isfinite(fit)]
    if fit.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.min(fit))
    hi = float(np.percentile(fit, 95))
    if hi - lo <= 1e-9:
        hi = lo + 1.0
    scaled = (values - lo) / (hi - lo)
    return np.clip(np.nan_to_num(scaled, nan=0.5), 0.0, 1.0).astype(np.float32)


def _load_cases(raw_root: Path) -> pd.DataFrame:
    numerics_dir = raw_root / "bidmc_csv"
    if not numerics_dir.is_dir():
        numerics_dir = raw_root / "bidmc"
    rows: list[dict] = []
    for path in sorted(numerics_dir.glob("bidmc_*_Numerics.csv")):
        m = CASE_RE.search(path.name)
        if not m:
            continue
        case_id = f"bidmc_{int(m.group(1)):02d}"
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        time_col = df.columns[0]
        for _, row in df.iterrows():
            t = float(row[time_col])
            hr = float(row.get("HR", np.nan))
            spo2 = float(row.get("SpO2", np.nan))
            resp = float(row.get("RESP", np.nan))
            pulse = float(row.get("PULSE", hr))
            if np.isnan(hr) or np.isnan(spo2) or np.isnan(resp):
                continue
            shock = hr / max(spo2, 1.0)
            critical = int(hr < 55 or spo2 < 92 or resp > 28)
            rows.append(
                {
                    "patient_key": case_id,
                    "timestamp": f"{case_id}_t{int(t)}",
                    "time_s": t,
                    "heart_rate": hr,
                    "oxygenation": spo2,
                    "respiration": resp,
                    "shock_index": shock,
                    "pulse": pulse,
                    "is_critical": critical,
                    "category": "bidmc",
                }
            )
    if not rows:
        raise FileNotFoundError(f"No bidmc_*_Numerics.csv under {numerics_dir}")
    return pd.DataFrame(rows)


def _patient_splits(patients: list[str], seed: int, val_frac: float, test_frac: float) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    ids = list(patients)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
    n_val = max(1, int(round((n - n_test) * val_frac))) if n - n_test >= 2 else 0
    out: dict[str, str] = {}
    for pid in ids[:n_test]:
        out[pid] = "test"
    for pid in ids[n_test : n_test + n_val]:
        out[pid] = "validation"
    for pid in ids[n_test + n_val :]:
        out[pid] = "train"
    return out


def build_bidmc_fusion_frame(
    raw_root: Path,
    *,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    embedding_dim: int = 2,
) -> tuple[pd.DataFrame, dict]:
    base = _load_cases(raw_root)
    split_map = _patient_splits(sorted(base["patient_key"].unique()), seed, val_fraction, test_fraction)
    base["fusion_split"] = base["patient_key"].map(split_map)
    train_mask = base["fusion_split"].to_numpy() == "train"

    domain_raw = {
        "heart_rate": base["heart_rate"].to_numpy(dtype=np.float32),
        "oxygenation": base["oxygenation"].to_numpy(dtype=np.float32),
        "respiration": base["respiration"].to_numpy(dtype=np.float32),
        "shock_index": base["shock_index"].to_numpy(dtype=np.float32),
    }
    domain_scores = {d: _minmax(v, train_mask) for d, v in domain_raw.items()}

    rows: list[dict] = []
    for idx in range(len(base)):
        incident_id = hashlib.md5(
            f"{base['patient_key'].iat[idx]}::{base['timestamp'].iat[idx]}".encode()
        ).hexdigest()[:16]
        for domain in DOMAIN_ORDER:
            score = float(domain_scores[domain][idx])
            rows.append(
                {
                    "incident_id": incident_id,
                    "patient_key": base["patient_key"].iat[idx],
                    "timestamp": base["timestamp"].iat[idx],
                    "category": "bidmc",
                    "fusion_split": base["fusion_split"].iat[idx],
                    "domain": domain,
                    "label": int(base["is_critical"].iat[idx]),
                    "score": score,
                    "confidence": float(np.clip(2.0 * abs(score - 0.5), 0.0, 1.0)),
                    "embedding_0": score,
                    "embedding_1": float(
                        domain_raw[domain][idx]
                        / (float(np.nanmax(domain_raw[domain][train_mask])) + 1e-6)
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    meta = {
        "benchmark_type": "naturally_paired_bidmc_clinical_fusion",
        "natural_pairing": True,
        "pairing_unit": "BIDMC case x time co-observed HR/SpO2/RESP",
        "raw_root": str(raw_root),
        "domain_order": DOMAIN_ORDER,
        "samples": int(frame["incident_id"].nunique()),
        "rows": int(len(frame)),
        "patients": int(frame["patient_key"].nunique()),
        "positive_fraction": float(frame.drop_duplicates("incident_id")["label"].mean()),
        "label_rule": "proxy critical: HR<55 or SpO2<92 or RESP>28",
    }
    return frame, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/healthcare"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/fusion/healthcare_bidmc_patient_stratified_fusion_inputs.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("experiments/fusion/healthcare_bidmc_patient_stratified_metadata.json"),
    )
    args = parser.parse_args()
    frame, meta = build_bidmc_fusion_frame(args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    meta["output"] = str(args.output)
    args.metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
