#!/usr/bin/env python3
"""09 -- execute locked physical evidence and conformal gate ablations.

Fits each of the six pre-registered ablation variants on development splits (calibration-fit
and calibration-conformal) and evaluates them on the untouched held-out stream.
"""

import argparse
import os
import sys
import numpy as np

import _common as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from kbound_edge.evidence import EDGE_EVIDENCE_NAMES
from kbound_edge.benefit_estimator import EdgeBenefitEstimator
from kbound_edge.conformal import calibrate_conformal
from kbound_edge.policy import kga_decide
from kbound_edge.logging import read_jsonl
from kbound_edge import metrics as M

ABLATIONS = (
    "full_kga",
    "no_radius",
    "no_blur_brightness",
    "no_disagreement",
    "confidence_only",
    "entropy_only",
)

ABLATION_FEATURES = {
    "full_kga": list(EDGE_EVIDENCE_NAMES),
    "no_radius": list(EDGE_EVIDENCE_NAMES),
    "no_blur_brightness": [f for f in EDGE_EVIDENCE_NAMES if f not in ("pre_entropy", "post_entropy", "entropy_drop")],
    "no_disagreement": [f for f in EDGE_EVIDENCE_NAMES if f not in ("pbal_drop", "entropy_drop", "marginal_KL", "update_norm", "mean_js_div", "pred_flip_rate")],
    "confidence_only": ["pre_conf", "post_conf", "frac_highconf", "post_top2_margin"],
    "entropy_only": ["pre_entropy", "post_entropy", "entropy_drop"],
}


def subset_Z(Z, feature_names):
    indices = [EDGE_EVIDENCE_NAMES.index(f) for f in feature_names]
    return Z[:, indices]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_real_phone_v1.yaml")
    ap.add_argument("--calib-config", default="edge_calibration_v1.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    cal = C.load_config(args.calib_config)

    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"
    if not is_real:
        raise SystemExit("[09] Ablations script is only valid for physical real protocol mode.")

    # 1. Load calibration-fit and calibration-conformal data
    fit_data = np.load(C.resolve(cfg["paths"]["calibration_fit"]), allow_pickle=True)
    conf_data = np.load(C.resolve(cfg["paths"]["calibration_conformal"]), allow_pickle=True)

    Z_fit, B_fit = fit_data["Z"], fit_data["B"]
    Z_conf, B_conf = conf_data["Z"], conf_data["B"]

    # 2. Load held-out online log and true benefits
    log_path = C.resolve(cfg["paths"]["heldout_log"])
    if not os.path.exists(log_path):
        raise SystemExit(f"[09] Held-out log not found at: {log_path}. Run 06 first.")

    records = read_jsonl(log_path)
    if not records:
        raise SystemExit(f"[09] Held-out log is empty: {log_path}")

    # Build Z_held from records evidence dicts
    Z_held_list = []
    for r in records:
        ev = r["evidence"]
        Z_held_list.append([ev[f] for f in EDGE_EVIDENCE_NAMES])
    Z_held = np.asarray(Z_held_list)

    # Load true labels from held-out NPZs to compute true benefits B_held
    from kbound_edge.real_dataset import load_window
    windows_dir = C.resolve(cfg["paths"]["windows_dir"])
    split_dir = os.path.join(windows_dir, "heldout")
    files = sorted([f for f in os.listdir(split_dir) if not f.startswith(".") and f.endswith(".npz") and (f.startswith("S07_") or f.startswith("S08_"))])

    true_labels = []
    for fname in files:
        _, off_load = load_window(os.path.join(split_dir, fname))
        true_labels.append(off_load["labels"])

    B_held = []
    for labels, r in zip(true_labels, records):
        p0 = np.array(r["frozen_pred"])
        if "shadow_candidate_pred" in r:
            pa = np.array(r["shadow_candidate_pred"])
        else:
            pa = np.array(r["extra"]["shadow_candidate_pred"])

        # Build dummy outcomes to calculate candidate/frozen accuracies
        froz_acc = float((p0 == labels).mean())
        cand_acc = float((pa == labels).mean())
        B_held.append(cand_acc - froz_acc)
    B_held = np.asarray(B_held)

    # 3. Fit each ablation variant
    ablation_results = {}

    for variant in ABLATIONS:
        features = ABLATION_FEATURES[variant]
        Z_fit_sub = subset_Z(Z_fit, features)
        Z_conf_sub = subset_Z(Z_conf, features)
        Z_held_sub = subset_Z(Z_held, features)

        # Fit benefit estimator (HistGradientBoostingRegressor kwargs from calib config)
        est = EdgeBenefitEstimator(**cal["estimator"]).fit(Z_fit_sub, B_fit)

        if variant == "no_radius":
            eps = 0.0
        else:
            cr = calibrate_conformal(est, Z_conf_sub, B_conf, alpha=cfg["alpha"], conservative=cal["conservative"])
            eps = cr.eps

        # Predict on heldout and make decisions
        bhat_held = est.predict(Z_held_sub)
        decisions = [kga_decide(b, eps).decision for b in bhat_held]

        # Evaluate metrics
        metrics = M.evaluate(decisions, B_held)

        ablation_results[variant] = {
            "regret": metrics["mean_regret"],
            "false_adapt_uncond": metrics["false_adapt_uncond"],
            "adapt_rate": metrics["adapt_rate"],
            "abstain_rate": metrics["abstain_rate"],
            "eps": eps,
            "features_used": features
        }

    # 4. Save results to ablation_results.json
    results_dir = os.path.normpath(os.path.join(C.EDGE_ROOT, cfg["paths"]["results_dir"]))
    out_path = os.path.join(results_dir, "ablation_results.json")
    C.save_json(out_path, ablation_results)

    print(f"[09] Wrote ablation results to: {out_path}")
    print("\nAblation Results:")
    print("-" * 80)
    print(f"{'Variant':<28} | {'Regret':<8} | {'FA_u':<8} | {'Adapt':<8} | {'Abstain':<8} | {'Epsilon':<8}")
    print("-" * 80)
    for variant, m in ablation_results.items():
        print(f"{variant:<28} | {m['regret']:.4f}  | {m['false_adapt_uncond']:.4f}  | {m['adapt_rate']:.3f}   | {m['abstain_rate']:.3f}     | {m['eps']:.4f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
