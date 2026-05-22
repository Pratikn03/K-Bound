"""Derive a noise-floor variant of an existing fusion CSV.

A senior-reviewer criticism of the VisA and LOCO-AD RGB+edge_proxy
benchmarks is that the edge_proxy domain is a deterministic function of
the same RGB image (Sobel-gradient filter), so the two domains are not
really independent. This script builds a counterfactual where the
edge_proxy domain's score + embedding are replaced by random-uniform
samples. Re-running the fusion benchmark on this noise-floor CSV
quantifies how much of the headline result comes from the edge_proxy
domain actually carrying signal versus the RGA+ router pulling on a
random feature.

If RGA+ headline survives noise-floor replacement (i.e. random
edge_proxy still beats baseline), the win is RGB-driven and the
edge_proxy criticism does not matter.

If RGA+ collapses to baseline under noise-floor replacement, the
edge_proxy was carrying real signal and is therefore not just a
deterministic re-encoding of the RGB image.

Either outcome is honest information for the discussion section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def derive_noise_floor(
    canonical_csv: Path,
    *,
    domain_to_randomise: str = "edge_proxy",
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(canonical_csv)
    if domain_to_randomise not in set(df["domain"].astype(str)):
        raise ValueError(
            f"Domain '{domain_to_randomise}' not present in {canonical_csv}. "
            f"Available: {sorted(set(df['domain'].astype(str)))}"
        )
    rng = np.random.default_rng(int(seed))
    out = df.copy()
    target_mask = out["domain"].astype(str) == domain_to_randomise
    n_rows = int(target_mask.sum())

    # Replace the score with uniform noise in [0, 1].
    out.loc[target_mask, "score"] = rng.uniform(0.0, 1.0, size=n_rows)
    # Replace confidence with a constant 0.5 (no information).
    out.loc[target_mask, "confidence"] = 0.5
    # Replace every embedding_* column with uniform noise.
    embedding_cols = [c for c in out.columns if c.startswith("embedding_")]
    for col in embedding_cols:
        out.loc[target_mask, col] = rng.uniform(0.0, 1.0, size=n_rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--domain-to-randomise", default="edge_proxy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    derived = derive_noise_floor(
        args.canonical_csv,
        domain_to_randomise=args.domain_to_randomise,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(f"Randomised domain: {args.domain_to_randomise}")
    samples = derived.drop_duplicates("sample_id")
    if "split" in samples.columns:
        print("Splits:", samples["split"].value_counts().to_dict())

    if args.metadata_output is not None:
        meta = {
            "derived_from": str(args.canonical_csv),
            "noise_floor_protocol": {
                "domain_to_randomise": args.domain_to_randomise,
                "score_distribution": "uniform[0, 1]",
                "confidence": 0.5,
                "embedding_distribution": "uniform[0, 1]",
                "rationale": (
                    "Defends against the 'edge_proxy is a deterministic function of "
                    "the RGB image' criticism. If RGA+ point-estimate survives the "
                    "replacement, the win is RGB-driven; if it collapses to the "
                    "baseline, the edge_proxy carried real (non-trivial) signal."
                ),
                "seed": int(args.seed),
            },
            "n_samples": int(len(samples)),
        }
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
