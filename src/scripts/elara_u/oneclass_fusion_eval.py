"""Canonical MVTec 3D-AD one-class fusion evaluation (Task L3.1).

Produces a leaderboard-comparable RGA / gated-CW number under the *one-class* protocol
(train + val normal-only, mixed test), per category, averaged -- so an ELARA row can sit
next to the published one-class leaderboard (M3DM 0.945 / AST 0.937 / PatchCore-3D 0.901)
instead of the supervised-paired table.

It reuses the EXISTING per-category detector scores in
``experiments/fusion/mvtec3d_score_cache/*.npz`` (built one-class by
gpu_build_mvtec3d_cache.py: memory bank = train/good only). The only change here is the
*fusion protocol*, not the detector: we re-partition each category's scored pool into a
one-class split -- a normal-only validation reference, and a mixed (good + defect) test --
then fuse with the parameter-free reliability-gated / gated-CW rule (no trained head;
one-class has no fusion-training positives) and the per-category baselines.

Fusion rule (parameter-free, cf. src/uais/fusion/attention/reliability_boosted_fusion.py
and the D23 gate in multimodal_reliability_test.py): the gate DEFAULTS to the
confidence-weighted mean (CW) and deviates only when one modality shows *differentially*
larger validation->test drift than the other (a degraded/unreliable channel), in which
case it drops that channel and CWs the rest. With both modalities reliable it keeps both
-> exactly CW. No test labels are used to fuse or to gate.

CPU-only and light (scores already exist). ``--dry-run`` evaluates a single category and
prints, without writing outputs or averaging the full panel. Honest expectation (L3.1):
gated-CW one-class will likely land below M3DM/AST, which use cross-modal patch
interactions ELARA does not -- the value is an honest leaderboard position, not SOTA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments/fusion/mvtec3d_score_cache"
OUT = ROOT / "experiments/elara_u/oneclass_fusion_results.json"
TEX = ROOT / "docs/research/tables/mvtec3d_oneclass_demarcation.tex"
RNG = 0
DRIFT_GAP = 0.15  # fixed, parameter-free: a modality is "failed" if its KS drift exceeds
#                   the most-reliable modality's by > DRIFT_GAP. No tuning on test labels.
DEGEN_STD = 0.02  # a modality whose validation scores are ~constant is degenerate.

# Published MVTec 3D-AD one-class image-AUROC (mean over categories), for demarcation only.
# Source: the values quoted in docs/research/phase3/LEVEL_3_PLAN_one_class_and_natural_degradation.md
# (M3DM, AST, PatchCore-3D). These are external reference numbers, NOT produced here.
PUBLISHED = {"M3DM": 0.945, "AST": 0.937, "PatchCore-3D": 0.901}


def _cw(S):
    """Confidence-weighted mean over the modality columns of S[n, k]."""
    if S.ndim == 1 or S.shape[1] == 0:
        return S.reshape(len(S), -1).mean(1)
    w = 2.0 * np.abs(S - 0.5)
    return (S * w).sum(1) / np.clip(w.sum(1), 1e-9, None)


def _ks(a, b):
    from scipy.stats import ks_2samp

    if len(a) < 2 or len(b) < 2:
        return 0.0
    return float(ks_2samp(a, b).statistic)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5


def oneclass_split(S, y, seed=RNG):
    """One-class partition of a scored pool: normal-only val reference + mixed test.
    Returns (Sval_ref[normal, k], Stest[mixed, k], ytest)."""
    rng = np.random.default_rng(seed)
    normals = np.where(y == 0)[0]
    anoms = np.where(y == 1)[0]
    rng.shuffle(normals)
    half = max(1, len(normals) // 2)
    val_ref = normals[:half]
    test = np.concatenate([normals[half:], anoms])
    return S[val_ref], S[test], y[test]


def fuse_methods(Sval_ref, Stest):
    """Return {method: per-sample test score} for the parameter-free fusion family."""
    k = Stest.shape[1]
    val_std = np.array([Sval_ref[:, m].std() for m in range(k)])
    val_ok = val_std >= DEGEN_STD  # degenerate-channel guard
    drift = np.array([_ks(Sval_ref[:, m], Stest[:, m]) for m in range(k)])
    # relative-drift gate: keep modalities whose drift is not differentially worse than
    # the most-reliable kept modality. With both reliable -> keep both -> CW (default).
    ok_idx = np.where(val_ok)[0]
    if len(ok_idx) == 0:
        keep = np.ones(k, bool)
    else:
        base = float(drift[ok_idx].min())
        keep = val_ok & (drift <= base + DRIFT_GAP)
        if not keep.any():
            keep = val_ok
    out = {
        "rgb_only": Stest[:, 0],
        # column 1 = the geometry modality (xyz surface-normals for MVTec-3D, contrast-
        # stretched depth for 3D-ADAM); labelled "depth_only" to match the plan's baseline.
        "depth_only": Stest[:, 1] if k > 1 else Stest[:, 0],
        "static_mean": Stest.mean(1),
        "cw": _cw(Stest),
        "no_reliability": _cw(Stest[:, val_ok]) if val_ok.any() else Stest.mean(1),
        "reliability_gate": _cw(Stest[:, keep]) if keep.any() else _cw(Stest),
    }
    return out, {"val_ok": val_ok.tolist(), "drift": np.round(drift, 4).tolist(),
                 "kept": keep.tolist()}


def eval_category(npz_path, seed=RNG):
    z = np.load(npz_path)
    S = np.vstack([np.asarray(z["Sval"], float), np.asarray(z["Stest"], float)])
    y = np.concatenate([np.asarray(z["yval"], int), np.asarray(z["ytest"], int)])
    if len(np.unique(y)) < 2 or S.shape[0] < 16:
        return None
    Sval_ref, Stest, ytest = oneclass_split(S, y, seed)
    if len(np.unique(ytest)) < 2:
        return None
    fused, diag = fuse_methods(Sval_ref, Stest)
    aucs = {m: round(_auc(ytest, s), 4) for m, s in fused.items()}
    return {"n_val_ref": int(len(Sval_ref)), "n_test": int(len(ytest)),
            "n_test_anom": int(ytest.sum()), "auroc": aucs, "gate": diag}


def _categories(cache):
    import glob
    import os

    return [f for f in sorted(glob.glob(str(Path(cache) / "*.npz")))
            if not os.path.basename(f).startswith("._")]


def write_demarcation_tex(mean_auroc, n_cat, tex_path):
    head = (mean_auroc.get("reliability_gate"), mean_auroc.get("cw"))
    lines = [
        "% Auto-generated by oneclass_fusion_eval.py -- MVTec 3D-AD one-class demarcation.",
        "%% ELARA rows are one-class image-AUROC over %d categories (parameter-free fusion)." % n_cat,
        "\\begin{table}[t]\\centering",
        "\\caption{MVTec 3D-AD one-class image-AUROC: ELARA parameter-free fusion vs the "
        "published one-class leaderboard. ELARA rows are \\emph{protocol-comparable} "
        "(one-class, image-AUROC) but use a \\emph{different method} (score-level fusion "
        "with no cross-modal patch interaction), so they are not a SOTA claim. ELARA "
        "numbers are on the cached, capped evaluation pool (single seed), an approximate "
        "leaderboard position rather than the full canonical test split.}",
        "\\label{tab:mvtec3d_oneclass}\\footnotesize",
        "\\begin{tabular}{llc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Type} & \\textbf{Image-AUROC} \\\\",
        "\\midrule",
        "M3DM~\\cite{m3dm} & published, cross-modal patch & %.3f \\\\" % PUBLISHED["M3DM"],
        "AST & published, cross-modal & %.3f \\\\" % PUBLISHED["AST"],
        "PatchCore-3D~\\cite{patchcore} & published, per-modality bank & %.3f \\\\" % PUBLISHED["PatchCore-3D"],
        "\\midrule",
        "ELARA reliability-gate (gated-CW) & ours, one-class, comparable & \\textbf{%.3f} \\\\"
        % (head[0] if head[0] is not None else float("nan")),
        "ELARA CW (parameter-free) & ours, one-class, comparable & %.3f \\\\"
        % (head[1] if head[1] is not None else float("nan")),
        "ELARA RGB-only & ours, single modality & %.3f \\\\" % mean_auroc.get("rgb_only", float("nan")),
        "ELARA geom-only (xyz/depth) & ours, single modality & %.3f \\\\" % mean_auroc.get("depth_only", float("nan")),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    Path(tex_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tex_path).write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="MVTec 3D-AD one-class fusion eval (L3.1).")
    ap.add_argument("--cache", default=str(CACHE), help="per-category one-class score cache dir")
    ap.add_argument("--out", default=str(OUT), help="output JSON (per-category + mean AUROC)")
    ap.add_argument("--tex", default=str(TEX), help="leaderboard-demarcation LaTeX table (new file)")
    ap.add_argument("--seed", type=int, default=RNG)
    ap.add_argument("--dry-run", action="store_true", help="evaluate ONE category, print, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N categories (0 = all)")
    args = ap.parse_args()

    files = _categories(args.cache)
    if not files:
        print(f"DATA NEEDED: no per-category caches under {args.cache}\n"
              "  Build them first with gpu_build_mvtec3d_cache.py (GPU), then re-run.")
        raise SystemExit(2)

    if args.dry_run:
        r = eval_category(files[0], args.seed)
        cat = Path(files[0]).stem
        print("=== DRY RUN (one category, CPU-only, nothing written) ===")
        print(f"category={cat}  {('skipped (degenerate split)' if r is None else r)}")
        return 0

    if args.limit:
        files = files[: args.limit]
    per_cat, methods = {}, None
    for f in files:
        r = eval_category(f, args.seed)
        if r is None:
            print(f"[{Path(f).stem}] skipped (degenerate split)")
            continue
        per_cat[Path(f).stem] = r
        methods = list(r["auroc"].keys())
        print(f"[{Path(f).stem}] one-class AUROC {r['auroc']}  kept={r['gate']['kept']}", flush=True)
    if not per_cat:
        print("no evaluable categories")
        raise SystemExit(2)

    mean_auroc = {m: round(float(np.mean([c["auroc"][m] for c in per_cat.values()])), 4) for m in methods}
    result = {
        "protocol": "MVTEC3D_ONECLASS_FUSION_L3_1",
        "fusion": "parameter-free reliability-gated / gated-CW (no trained head)",
        "n_categories": len(per_cat),
        "mean_image_auroc": mean_auroc,
        "published_reference_image_auroc": PUBLISHED,
        "comparability": "one-class image-AUROC is protocol-comparable to the published "
                         "leaderboard; method differs (no cross-modal patch interaction).",
        "scope_caveat": "ELARA numbers are computed on each category's CACHED evaluation "
                        "pool (built one-class by gpu_build_mvtec3d_cache.py, stratified and "
                        "capped ~130 samples/category), re-partitioned into a normal-only "
                        "val reference + mixed test -- this is a single-seed approximate "
                        "leaderboard position, NOT the full canonical MVTec-3D test split. "
                        "Average over several --seed values before quoting; report ties/losses.",
        "seed": int(args.seed),
        "per_category": per_cat,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    write_demarcation_tex(mean_auroc, len(per_cat), args.tex)

    print("\n=== MVTec 3D-AD ONE-CLASS (mean image-AUROC over %d categories) ===" % len(per_cat))
    for m in methods:
        print(f"  {m:18} {mean_auroc[m]:.3f}")
    print("  --- published reference ---")
    for m, v in PUBLISHED.items():
        print(f"  {m:18} {v:.3f}")
    print(f"wrote {args.out}\nwrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
