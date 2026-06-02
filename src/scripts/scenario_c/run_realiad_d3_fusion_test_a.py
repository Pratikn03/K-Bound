"""Fusion test A (DEVELOPMENT) — does reliability-gated fusion beat the
confidence-weighted mean on Real-IAD-D3 natural degradation, using STRONG
per-modality detectors (deep PatchCore for RGB/PS, PCD-geometry for XYZ)?

Reliability signal is VALIDATION-ONLY: each modality's per-category validation
AUROC sets its fusion weight, so the gate trusts geometry (XYZ) on geometric-
defect categories and texture (RGB/PS) where XYZ is blind (scratches). No test
labels are used for any weight or selection; test labels are used only to score
the final AUROCs. The Real-IAD-D3 holdout is OPENED, so this is DEVELOPMENT
evidence and cannot set an official gate flag.

Comparisons (paired bootstrap, 95% CI) on the pooled test set:
  reliability-gated fusion  vs  confidence-weighted mean (CW)   [primary]
  reliability-gated fusion  vs  best single modality / equal mean [context]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.scenario_c.prepare_realiad_d3_headroom_inputs import _load_rows  # noqa: E402
from uais.fusion.attention.realiad_3d_detector import (  # noqa: E402
    load_modality_image,
    score_one_class_patchcore,
)

MODALITIES = ("rgb", "ps", "xyz")


def _zsig(raw: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu = float(np.mean(ref)); sd = float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _paired_bootstrap(y, cand, comp, n_iter=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y); deltas = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], cand[idx]) - roc_auc_score(y[idx], comp[idx]))
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {
        "delta": float(roc_auc_score(y, cand) - roc_auc_score(y, comp)),
        "ci95": [lo, hi], "ci_low_gt_0": bool(lo > 0),
    }


def _score_modality(zf, rows_split, modality, bank_imgs, coreset):
    imgs = [load_modality_image(zf, r.zip_member, modality) for r in rows_split]
    raw = score_one_class_patchcore(bank_imgs, imgs, coreset_size=coreset)
    return raw


def run(categories, coreset, max_train, max_val, max_test, out_path):
    report = {
        "protocol": "FUSION_TEST_A_development",
        "holdout_status": "OPENED_DEVELOPMENT_ONLY",
        "reliability_signal": "validation per-category per-modality AUROC (val labels only)",
        "categories": categories, "per_category": {}, "modalities": list(MODALITIES),
    }
    # accumulate pooled test arrays
    pool = {"y": [], "cw": [], "gated": [], "equal": [], **{m: [] for m in MODALITIES}}
    val_auc_record = {}

    for cat in categories:
        t0 = time.time()
        rows, _ = _load_rows(ROOT / "data/raw/realiad_d3", [cat])
        zf = zipfile.ZipFile(ROOT / "data/raw/realiad_d3/realiad_d3_raw" / f"{cat}.zip")
        # per-modality scores for val and test, aligned by sample_id
        val_scores, test_scores, val_auc = {}, {}, {}
        val_y = test_y = None
        ok = True
        for m in MODALITIES:
            mrows = [r for r in rows if r.modality == m]
            train = [r for r in mrows if r.split == "train" and r.label == 0][:max_train]
            vrows = [r for r in mrows if r.split == "validation"]
            trows = [r for r in mrows if r.split == "test"]
            # stratified caps (keep both classes)
            def strat(rs, cap):
                pos = [r for r in rs if r.label == 1][: cap // 2]
                neg = [r for r in rs if r.label == 0][: cap // 2]
                return pos + neg
            vrows = strat(vrows, max_val); trows = strat(trows, max_test)
            if len(train) < 10 or len(vrows) < 10 or len(trows) < 10:
                ok = False; break
            bank_imgs = [load_modality_image(zf, r.zip_member, m) for r in train]
            ref = score_one_class_patchcore(bank_imgs[: min(40, len(bank_imgs))], bank_imgs, coreset_size=coreset)
            v_raw = _score_modality(zf, vrows, m, bank_imgs, coreset)
            t_raw = _score_modality(zf, trows, m, bank_imgs, coreset)
            val_scores[m] = {r.sample_id: s for r, s in zip(vrows, _zsig(v_raw, ref))}
            test_scores[m] = {r.sample_id: s for r, s in zip(trows, _zsig(t_raw, ref))}
            val_lab = {r.sample_id: r.label for r in vrows}
            test_lab = {r.sample_id: r.label for r in trows}
            val_auc[m] = float(roc_auc_score([val_lab[i] for i in val_scores[m]],
                                             [val_scores[m][i] for i in val_scores[m]]))
            if val_y is None:
                val_y = val_lab; test_y = test_lab
        if not ok:
            report["per_category"][cat] = {"skipped": True}
            continue

        # align test samples present in all modalities
        common = set.intersection(*[set(test_scores[m]) for m in MODALITIES])
        common = sorted(common)
        y = np.array([test_y[i] for i in common], dtype=int)
        S = np.column_stack([[test_scores[m][i] for i in common] for m in MODALITIES])  # [N,3]
        # CW baseline: per-sample sharpness weights 2|s-0.5|
        wcw = 2.0 * np.abs(S - 0.5)
        cw = (S * wcw).sum(1) / np.clip(wcw.sum(1), 1e-9, None)
        # reliability-gated: validation per-modality AUROC weights (val labels only)
        rel_w = np.array([max(val_auc[m] - 0.5, 0.0) for m in MODALITIES], dtype=float)
        if rel_w.sum() < 1e-9:
            rel_w = np.ones(len(MODALITIES))
        rel_w = rel_w / rel_w.sum()
        gated = S @ rel_w
        equal = S.mean(1)

        val_auc_record[cat] = {m: round(val_auc[m], 4) for m in MODALITIES}
        report["per_category"][cat] = {
            "val_auc": val_auc_record[cat],
            "rel_weights": {m: round(float(w), 3) for m, w in zip(MODALITIES, rel_w)},
            "test_auc": {m: round(float(roc_auc_score(y, S[:, j])), 4) for j, m in enumerate(MODALITIES)},
            "test_auc_cw": round(float(roc_auc_score(y, cw)), 4),
            "test_auc_gated": round(float(roc_auc_score(y, gated)), 4),
            "test_auc_equal_mean": round(float(roc_auc_score(y, equal)), 4),
            "n_test": int(len(y)), "secs": round(time.time() - t0, 1),
        }
        print(f"[{cat}] valAUC rgb/ps/xyz={val_auc_record[cat]}  "
              f"test CW={roc_auc_score(y,cw):.4f} gated={roc_auc_score(y,gated):.4f}  ({time.time()-t0:.0f}s)",
              flush=True)
        pool["y"].append(y); pool["cw"].append(cw); pool["gated"].append(gated); pool["equal"].append(equal)
        for j, m in enumerate(MODALITIES):
            pool[m].append(S[:, j])

    # pooled comparison
    if pool["y"]:
        y = np.concatenate(pool["y"]); cw = np.concatenate(pool["cw"])
        gated = np.concatenate(pool["gated"]); equal = np.concatenate(pool["equal"])
        report["pooled"] = {
            "n_test": int(len(y)),
            "auc_cw": round(float(roc_auc_score(y, cw)), 4),
            "auc_gated": round(float(roc_auc_score(y, gated)), 4),
            "auc_equal_mean": round(float(roc_auc_score(y, equal)), 4),
            "auc_single": {m: round(float(roc_auc_score(y, np.concatenate(pool[m]))), 4) for m in MODALITIES},
            "gated_vs_cw": _paired_bootstrap(y, gated, cw),
            "gated_vs_equal_mean": _paired_bootstrap(y, gated, equal),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", nargs="*",
                    default=["knob_cap", "ferrite_bead", "fuse_holder",
                             "audio_jack_socket", "dc_power_connector", "ethernet_connector"])
    ap.add_argument("--coreset", type=int, default=4096)
    ap.add_argument("--max-train", type=int, default=70)
    ap.add_argument("--max-val", type=int, default=90)
    ap.add_argument("--max-test", type=int, default=130)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/fusion/realiad_d3_fusion_test_a.json")
    args = ap.parse_args()
    print(f"=== Fusion test A (development): {args.categories} ===", flush=True)
    rep = run(args.categories, args.coreset, args.max_train, args.max_val, args.max_test, args.out)
    p = rep.get("pooled")
    if p:
        print("\n=== POOLED (development) ===")
        print(f"  single: {p['auc_single']}")
        print(f"  CW={p['auc_cw']}  equal_mean={p['auc_equal_mean']}  GATED={p['auc_gated']}")
        gc = p["gated_vs_cw"]
        print(f"  GATED vs CW: delta={gc['delta']:+.4f}  CI95={[round(x,4) for x in gc['ci95']]}  "
              f"CI_low>0: {gc['ci_low_gt_0']}")
        print("  VERDICT:", "GATE BEATS CW (dev)" if gc["ci_low_gt_0"] else "tie/again no win (dev)")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
