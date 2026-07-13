#!/usr/bin/env python3
"""05 -- fit the benefit estimator + split-conformal radius (the KGA-edge artifact).

Supports both synthetic (default) and real_manifest protocols. In real_manifest
mode: fits the estimator on calibration_fit.npz and conformal radius on
calibration_conformal.npz. Selects confidence/entropy thresholds from
calibration-fit data. Saves kga_edge.joblib and metadata.
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

def select_conf_threshold(Z, B, alpha=0.10):
    post_conf = Z[:, 4]
    for thr in np.linspace(0.0, 1.0, 101):
        fa_rate = np.mean((post_conf >= thr) & (B <= 0.0))
        if fa_rate <= alpha:
            return float(thr)
    return 0.50

def select_entropy_threshold(Z, B, alpha=0.10):
    entropy_drop = Z[:, 7]
    for thr in np.linspace(0.0, 1.0, 101):
        fa_rate = np.mean((entropy_drop >= thr) & (B <= 0.0))
        if fa_rate <= alpha:
            return float(thr)
    return 0.05

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    ap.add_argument("--calib-config", default="edge_calibration_v1.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    cal = C.load_config(args.calib_config)

    from kbound_edge.benefit_estimator import EdgeBenefitEstimator
    from kbound_edge.conformal import calibrate_conformal, fit_real_certificate

    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"
    edge_dir = os.path.normpath(os.path.join(_HERE, ".."))

    if not is_real:
        # --- Synthetic Mode ---
        data = np.load(C.resolve(cfg["paths"]["calibration"]), allow_pickle=True)
        Z, B = data["Z"], data["B"]
        n = len(B)

        rng = np.random.default_rng(cal["split_seed"])
        idx = rng.permutation(n)
        cut = int(round(cal["fit_fraction"] * n))
        fit_idx, conf_idx = np.sort(idx[:cut]), np.sort(idx[cut:])
        if len(conf_idx) < 1:
            raise SystemExit("[05] conformal split is empty; lower fit_fraction or add conditions")

        est = EdgeBenefitEstimator(**cal["estimator"]).fit(Z[fit_idx], B[fit_idx])
        cr = calibrate_conformal(est, Z[conf_idx], B[conf_idx],
                                 alpha=cfg["alpha"], conservative=cal["conservative"])

        fit_mae = float(np.mean(np.abs(est.predict(Z[fit_idx]) - B[fit_idx])))
        conf_mae = float(np.mean(np.abs(est.predict(Z[conf_idx]) - B[conf_idx])))

        est_path = C.resolve(cfg["paths"]["kga_edge"])
        C.ensure_parent(est_path)
        est.save(est_path)

        # Save threshold values
        conf_tau = cal["policies"]["conf_tau"]
        entropy_tau = cal["policies"]["entropy_tau"]

        kga_meta_path = cfg["paths"].get("kga_edge_meta", "artifacts_synth/kga_edge_meta.json")
        C.save_json(C.resolve(kga_meta_path), {
            "eps": cr.eps,
            "alpha": cfg["alpha"],
            "method": cr.method,
            "n_fit": int(len(fit_idx)),
            "n_conformal": int(len(conf_idx)),
            "fit_idx": fit_idx.tolist(),
            "conformal_idx": conf_idx.tolist(),
            "fit_mae": fit_mae,
            "conformal_mae": conf_mae,
            "estimator": "HistGradientBoostingRegressor",
            "model_version": str(data["model_version"]),
            "policies": {
                "conf_tau": conf_tau,
                "entropy_tau": entropy_tau
            }
        })
        print(f"[05] fit={len(fit_idx)} conf={len(conf_idx)}  eps={cr.eps:.4f} ({cr.method}, alpha={cfg['alpha']})")
    else:
        # --- Real Manifest Mode ---
        fit_data = np.load(C.resolve(cfg["paths"]["calibration_fit"]), allow_pickle=True)
        conf_data = np.load(C.resolve(cfg["paths"]["calibration_conformal"]), allow_pickle=True)

        bundle = {
            "fit": {
                "Z": fit_data["Z"],
                "B": fit_data["B"],
                "sessions": sorted(list(set(fit_data["sessions"]))),
                "source_hashes": sorted(list(set(fit_data["source_hashes"])))
            },
            "conformal": {
                "Z": conf_data["Z"],
                "B": conf_data["B"],
                "sessions": sorted(list(set(conf_data["sessions"]))),
                "source_hashes": sorted(list(set(conf_data["source_hashes"])))
            }
        }

        # Fit and calibrate using helper function
        result = fit_real_certificate(bundle, estimator_kwargs=cal["estimator"], alpha=cfg["alpha"], conservative=cal["conservative"])

        fit_mae = float(np.mean(np.abs(result.estimator.predict(fit_data["Z"]) - fit_data["B"])))
        conf_mae = float(np.mean(np.abs(result.estimator.predict(conf_data["Z"]) - conf_data["B"])))

        # Select confidence and entropy thresholds from calibration-fit split
        conf_tau = select_conf_threshold(fit_data["Z"], fit_data["B"], alpha=cfg["alpha"])
        entropy_tau = select_entropy_threshold(fit_data["Z"], fit_data["B"], alpha=cfg["alpha"])

        # Save model and meta JSON
        est_path = C.resolve(cfg["paths"]["kga_edge"])
        C.ensure_parent(est_path)
        result.estimator.save(est_path)

        # Expose results directory
        results_dir = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["results_dir"]))
        summary_path = os.path.join(results_dir, "calibration_summary.json")

        kga_meta_path = cfg["paths"].get("kga_edge_meta", "artifacts_real/calibration/kga_edge_meta.json")
        C.save_json(C.resolve(kga_meta_path), {
            "eps": result.conformal_radius.eps,
            "alpha": cfg["alpha"],
            "method": result.conformal_radius.method,
            "n_fit": len(fit_data["B"]),
            "n_conformal": len(conf_data["B"]),
            "fit_sessions": bundle["fit"]["sessions"],
            "conformal_sessions": bundle["conformal"]["sessions"],
            "fit_mae": fit_mae,
            "conformal_mae": conf_mae,
            "estimator": "HistGradientBoostingRegressor",
            "model_version": str(fit_data["model_version"]),
            "policies": {
                "conf_tau": conf_tau,
                "entropy_tau": entropy_tau
            }
        })

        # Persist calibration provenance in results directory
        # "Write fit/conformal session IDs, clip hashes, feature schema hash, model hash, sample counts, alpha, epsilon, estimator parameters, MAE, and empirical conformal coverage to calibration_summary.json"

        # Calculate empirical conformal coverage
        # Residuals of conformal data <= eps
        conformal_residuals = np.abs(result.estimator.predict(conf_data["Z"]) - conf_data["B"])
        empirical_coverage = float(np.mean(conformal_residuals <= result.conformal_radius.eps))

        provenance = {
            "fit_sessions": bundle["fit"]["sessions"],
            "conformal_sessions": bundle["conformal"]["sessions"],
            "fit_clip_hashes": bundle["fit"]["source_hashes"],
            "conformal_clip_hashes": bundle["conformal"]["source_hashes"],
            "model_hash": str(fit_data["model_version"]),
            "n_fit": len(fit_data["B"]),
            "n_conformal": len(conf_data["B"]),
            "alpha": cfg["alpha"],
            "epsilon": result.conformal_radius.eps,
            "estimator_parameters": cal["estimator"],
            "fit_mae": fit_mae,
            "conformal_mae": conf_mae,
            "empirical_conformal_coverage": empirical_coverage,
            "policies": {
                "conf_tau": conf_tau,
                "entropy_tau": entropy_tau
            }
        }
        C.save_json(summary_path, provenance)
        print(f"[05] Wrote calibration summary to: {summary_path}")
        print(f"[05] fit={len(fit_data['B'])} conf={len(conf_data['B'])} eps={result.conformal_radius.eps:.4f} conf_tau={conf_tau:.3f} entropy_tau={entropy_tau:.3f}")

if __name__ == "__main__":
    main()
