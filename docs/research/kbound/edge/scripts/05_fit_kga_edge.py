#!/usr/bin/env python3
"""05 -- fit the benefit estimator + split-conformal radius (the KGA-edge artifact).

Split-conformal protocol (FIXED alpha=0.10):
  * fit the HistGradientBoostingRegressor on the calibration-FIT split ONLY;
  * compute residuals on the held-out calibration-CONFORMAL split ONLY;
  * eps = conservative order-statistic radius of those residuals.
Saves the fitted estimator (joblib) and a meta file with eps / alpha / splits.
"""

import argparse
import numpy as np

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    ap.add_argument("--calib-config", default="edge_calibration_v1.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    cal = C.load_config(args.calib_config)

    from kbound_edge.benefit_estimator import EdgeBenefitEstimator
    from kbound_edge.conformal import calibrate_conformal

    data = np.load(C.resolve(cfg["paths"]["calibration"]), allow_pickle=True)
    Z, B = data["Z"], data["B"]
    n = len(B)

    # deterministic fit / conformal split
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
    C.save_json(C.resolve(cfg["paths"]["kga_edge_meta"]), {
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
    })
    print(f"[05] fit={len(fit_idx)} conf={len(conf_idx)}  eps={cr.eps:.4f} ({cr.method}, alpha={cfg['alpha']})")
    print(f"[05] estimator MAE  fit={fit_mae:.4f}  conformal={conf_mae:.4f} -> {est_path}")


if __name__ == "__main__":
    main()
