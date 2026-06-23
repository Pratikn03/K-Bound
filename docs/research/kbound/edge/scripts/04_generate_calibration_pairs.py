#!/usr/bin/env python3
"""04 -- generate calibration (Z, B) pairs from synthetic conditions.

For each calibration condition: run the ONLINE chain (frozen pred -> episodic
Tent candidate -> label-free evidence Z) and, OFFLINE, measure the true benefit
B = acc(candidate) - acc(frozen) on that window using the held labels.  Z is
label-free (online); B uses labels but only here, offline.  Saves Z, B, and
per-condition metadata to an .npz for the calibration/conformal fit in 05.
"""

import argparse
import numpy as np

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    C.set_seed(cfg["seed"])

    from kbound_edge.model import predict_proba
    from kbound_edge.tent_adapter import EpisodicTentAdapter
    from kbound_edge.evidence import edge_evidence_vector, EDGE_EVIDENCE_NAMES
    from kbound_edge.dataset import build_conditions

    f0, version = C.load_f0(cfg)
    adapter = EpisodicTentAdapter(f0, lr=cfg["adapter"]["lr"], steps=cfg["adapter"]["steps"],
                                  device=cfg.get("device", "cpu"))

    conds = build_conditions(
        C.plan_tuples(cfg["calibration_plan"]),
        n_frames=cfg["window_size"],
        image_size=cfg["image_size"],
        seed=4000,
        n_classes=cfg["num_classes"],
        prefix="cal",
    )

    Z, B, regimes, divs, cond_ids = [], [], [], [], []
    for c in conds:
        x = c.tensor()
        p0 = predict_proba(f0, x)            # online: frozen
        res = adapter.adapt(x)               # online: isolated candidate
        pa = predict_proba(res.model, x)
        z = edge_evidence_vector(p0, pa, res.upd_norm)
        froz = float((p0.argmax(1) == c.labels).mean())   # OFFLINE label use
        cand = float((pa.argmax(1) == c.labels).mean())   # OFFLINE label use
        Z.append(z); B.append(cand - froz)
        regimes.append(c.regime); divs.append(c.diversity); cond_ids.append(c.cond_id)

    Z = np.asarray(Z); B = np.asarray(B)
    out = C.resolve(cfg["paths"]["calibration"])
    C.ensure_parent(out)
    np.savez_compressed(out, Z=Z, B=B, regimes=np.array(regimes), diversities=np.array(divs),
                        cond_ids=np.array(cond_ids), feature_names=np.array(list(EDGE_EVIDENCE_NAMES)),
                        model_version=version)
    print(f"[04] calibration pairs: Z={Z.shape} B in [{B.min():+.3f}, {B.max():+.3f}] "
          f"(mean {B.mean():+.3f}) -> {out}")
    pos = int((B > 0.05).sum()); neg = int((B < -0.05).sum()); neu = len(B) - pos - neg
    print(f"[04] benefit bands: positive={pos} near-zero={neu} negative={neg}")


if __name__ == "__main__":
    main()
