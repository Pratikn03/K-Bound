"""Multimodal reliability-gating experiment (RGB + 3D) -- the REAL test of the
pre-registered hypothesis from synthetic_multimodal_poc.py.

PRE-REGISTERED HYPOTHESIS (falsifiable; frozen before looking at real-data results):
  Under INDEPENDENT per-modality deployment degradation (one modality degrades while
  the other stays clean), reliability-gated fusion will:
    (H1) beat equal-weight fusion (mean)            with paired-bootstrap CI > 0
    (H2) beat stale validation-AUROC selection      with paired-bootstrap CI > 0
    (H3) beat its own no-test-time-reliability ablation (reliability features, not
         validation quality, carry the gain)         with paired-bootstrap CI > 0
  If any of H1-H3 fails on real multimodal data, the reliability-gating premise is
  reported NEGATIVE even where it structurally should hold, and ELARA-U is closed.

Why this can succeed where the 123-task single-modality benchmark failed: there, all
detectors share one input, so degradation is shared and no clean channel survives
(synthetic_multimodal_poc.py, SHARED regime). RGB and 3D fail independently, so a
clean channel survives (INDEPENDENT regime), which is the only condition under which
reliability gating helps.

GPU PATH (run on your box; see research_lock/MULTIMODAL_RELIABILITY_PROTOCOL_v1.md):
  per category -> PatchCore memory bank per modality (RGB, 3D) on NORMAL train ->
  per-modality val/test patch-anomaly scores -> apply per-modality input degradation
  at test only -> fuse -> evaluate the contract. Reuses
  uais.fusion.attention.{patchcore_patch, realiad_3d_detector, cross_modal_patchcore}.

CPU SMOKE PATH (`--smoke`): validates the entire degradation/fusion/contract logic on
synthetic per-modality scores with an independent-failure structure (no GPU/data).
No test labels are used by any fusion rule.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/elara_u/multimodal_reliability_result.json"
RNG = 0


# --------------------------------------------------------------------------- #
# core science (shared by GPU and smoke paths) -- identical reliability logic
# to synthetic_multimodal_poc.py and honest_benchmark.py
# --------------------------------------------------------------------------- #
def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5


def _ranknorm(S):
    return np.column_stack([rankdata(S[:, j]) / len(S) for j in range(S.shape[1])])


def reliability_weights(Stest, val_auc, use_testtime):
    """Per-modality weight: validation-quality x test-time consensus-agreement.
    Ablation (use_testtime=False) uses validation quality only. Unsupervised at test."""
    q = np.clip(val_auc - 0.5, 0, None)
    if not use_testtime:
        return q / q.sum() if q.sum() > 0 else np.ones(len(q)) / len(q)
    M = Stest.shape[1]; agree = np.zeros(M)
    for m in range(M):
        cs = [spearmanr(Stest[:, m], Stest[:, k]).correlation for k in range(M) if k != m]
        cs = [0.0 if np.isnan(c) else c for c in cs]
        agree[m] = max(0.0, float(np.mean(cs))) if cs else 1.0
    w = q * agree
    return w / w.sum() if w.sum() > 0 else np.ones(M) / M


def evaluate_fusion(Sval, yval, Stest, ytest):
    """All fusion rules + the pre-registered contrasts for ONE category."""
    val_auc = np.array([_auc(yval, Sval[:, m]) for m in range(Sval.shape[1])])
    Rt = _ranknorm(Stest)
    res = {
        "per_modality_auc": [float(_auc(ytest, Stest[:, m])) for m in range(Stest.shape[1])],
        "val_auc": [float(a) for a in val_auc],
        "mean": _auc(ytest, Rt.mean(1)),
        "val_select": _auc(ytest, Stest[:, int(np.argmax(val_auc))]),
        "rel_gate": _auc(ytest, Rt @ reliability_weights(Stest, val_auc, True)),
        "rel_gate_abl": _auc(ytest, Rt @ reliability_weights(Stest, val_auc, False)),
    }
    return res


def paired_ci(a, b, n_boot=10000):
    a, b = np.asarray(a), np.asarray(b); d = a - b; n = len(d)
    rng = np.random.default_rng(RNG)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "sig": bool(np.percentile(boot, 2.5) > 0)}


def run_contract(per_category):
    """per_category: {cat: evaluate_fusion(...) dict}. Returns the pre-registered verdict."""
    cats = list(per_category)
    col = lambda k: np.array([per_category[c][k] for c in cats])
    contrasts = {
        "H1_rel_gate_vs_mean": paired_ci(col("rel_gate"), col("mean")),
        "H2_rel_gate_vs_val_select": paired_ci(col("rel_gate"), col("val_select")),
        "H3_rel_gate_vs_ablation": paired_ci(col("rel_gate"), col("rel_gate_abl")),
    }
    passed = all(contrasts[h]["sig"] for h in contrasts)
    verdict = ("POSITIVE: reliability gating helps on real multimodal data under per-modality "
               "degradation (H1,H2,H3 all CI>0). ELARA-U has a real, scoped novel claim."
               if passed else
               "NEGATIVE: at least one of H1-H3 fails; reliability gating does not help even on "
               "multimodal data. Reliability premise closed.")
    return {"n_categories": len(cats), "mean_auc": {k: float(col(k).mean())
            for k in ["mean", "val_select", "rel_gate", "rel_gate_abl"]},
            "contrasts": contrasts, "contract_passed": passed, "verdict": verdict}


# --------------------------------------------------------------------------- #
# CPU smoke path: synthetic per-modality scores with INDEPENDENT failure
# --------------------------------------------------------------------------- #
def smoke(n_categories=10, n_seeds=6, n=600):
    """Positive control + design preview: 2 modalities, 2 detectors each (M=4 channels),
    one MODALITY (its 2 channels) fails independently at test. Statistical unit is
    (category x degradation-seed) -> adequate power, matching the recommended real
    design (MVTec-3D's 10 categories alone are too few to bootstrap). Validates the
    full degradation/fusion/contract logic without GPU/data."""
    rng = np.random.default_rng(RNG)
    modalities = [(0, 1), (2, 3)]; M = 4
    per = {}
    for c in range(n_categories):
        d = rng.uniform(1.4, 2.4, M)
        for s in range(n_seeds):
            yv = (rng.random(n) < 0.2).astype(int); yt = (rng.random(n) < 0.2).astype(int)
            Sval = np.column_stack([d[m] * yv + rng.standard_normal(n) for m in range(M)])
            gains = np.ones(M)
            for m in modalities[int(rng.integers(0, len(modalities)))]:
                gains[m] = 0.0                                       # one modality's channels fail together
            Stest = np.column_stack([d[m] * gains[m] * yt + rng.standard_normal(n) for m in range(M)])
            per[f"cat{c}_seed{s}"] = evaluate_fusion(Sval, yv, Stest, yt)
    return per


# --------------------------------------------------------------------------- #
# GPU path: real RGB+3D scoring (wired to existing detectors; runs on your box)
# --------------------------------------------------------------------------- #
def score_real(data_root: Path, category: str, degrade: dict | None):
    """Return (Sval[n_val,2], yval, Stest[n_test,2], ytest) for one category.

    Columns are [RGB_patchcore_score, 3D_patchcore_score]. `degrade`, if given, is
    applied to the TEST inputs of ONE modality only (independent failure), e.g.
    {"modality": "rgb", "kind": "blur", "severity": 3} or
    {"modality": "depth", "kind": "dropout", "severity": 0.5}. Validation stays clean.

    Implementation note (for the GPU run): build a normal-only PatchCore memory bank
    per modality, then score val/test images; apply input degradation to the chosen
    modality's TEST images before scoring. Reuse:
        uais.fusion.attention.realiad_3d_detector.score_one_class_patchcore  (RGB / depth-as-image)
        uais.fusion.attention.realiad_3d_detector.score_one_class_point_cloud (3D)
        uais.fusion.attention.realiad_3d_detector.{load_modality_image, pcd_to_geometry_image}
    and degrade_image()/degrade_depth() below for the per-modality corruptions.
    """
    raise NotImplementedError(
        "GPU path: wire to your MVTec-3D / Real-IAD data per the protocol. The fusion/"
        "contract logic (evaluate_fusion / run_contract) is shared with --smoke and is "
        "already validated. See research_lock/MULTIMODAL_RELIABILITY_PROTOCOL_v1.md.")


def degrade_image(img: np.ndarray, kind: str, severity: float) -> np.ndarray:
    """Per-modality RGB degradation (independent sensor failure). img: HxWx3 in [0,1]."""
    rng = np.random.default_rng(RNG)
    if kind == "blur":
        k = int(2 * severity + 1)
        pad = np.pad(img, ((k, k), (k, k), (0, 0)), mode="edge")
        out = np.zeros_like(img)
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                out += pad[k + dy:k + dy + img.shape[0], k + dx:k + dx + img.shape[1]]
        return out / ((2 * k + 1) ** 2)
    if kind == "noise":
        return np.clip(img + 0.15 * severity * rng.standard_normal(img.shape), 0, 1)
    if kind == "bright":
        return np.clip(img * (1 + 0.3 * severity), 0, 1)
    return img


def degrade_depth(depth: np.ndarray, kind: str, severity: float) -> np.ndarray:
    """Per-modality 3D/depth degradation (independent sensor failure). depth: HxW."""
    rng = np.random.default_rng(RNG)
    if kind == "dropout":
        mask = rng.random(depth.shape) < severity
        out = depth.copy(); out[mask] = 0.0; return out          # missing returns
    if kind == "noise":
        return depth + 0.1 * severity * rng.standard_normal(depth.shape)
    if kind == "quant":
        levels = max(2, int(16 / max(severity, 1)))
        return np.round(depth * levels) / levels
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU logic check on synthetic independent-failure scores")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--degrade-modality", choices=["rgb", "depth"], default="depth")
    ap.add_argument("--degrade-kind", default="dropout")
    ap.add_argument("--degrade-severity", type=float, default=0.5)
    args = ap.parse_args()

    if args.smoke or args.data_root is None:
        if not args.smoke:
            print("no --data-root given; running --smoke logic check (synthetic).")
        per = smoke()
        tag = "SMOKE(synthetic independent-failure)"
    else:
        deg = {"modality": args.degrade_modality, "kind": args.degrade_kind, "severity": args.degrade_severity}
        per = {c: score_real(Path(args.data_root), c, deg) for c in (args.categories or [])}
        tag = f"REAL({args.data_root}, degrade={deg})"

    result = {"mode": tag, **run_contract(per), "per_category": per}
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\n=== Multimodal reliability contract [{tag}] ===")
    m = result["mean_auc"]
    print(f"  mean={m['mean']:.3f}  val_select={m['val_select']:.3f}  "
          f"rel_gate={m['rel_gate']:.3f}  rel_gate_abl={m['rel_gate_abl']:.3f}")
    for h, v in result["contrasts"].items():
        print(f"  {h:26}{v['mean']:+.4f}  CI{[round(x,4) for x in v['ci95']]}  sig={v['sig']}")
    print(f"\nCONTRACT {'PASSED' if result['contract_passed'] else 'FAILED'}: {result['verdict']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
