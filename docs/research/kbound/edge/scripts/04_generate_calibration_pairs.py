#!/usr/bin/env python3
"""04 -- generate calibration (Z, B) pairs from synthetic or real manifest.

In synthetic mode: reads cfg["calibration_plan"] and outputs cfg["paths"]["calibration"].
In real mode: reads validated windows from calibration_fit (S03+S04) and
calibration_conformal (S05+S06) and outputs separate fit and conformal pair NPZs.
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

def load_real_pairs(cfg, sessions, edge_dir, f0, adapter, version, bypass_gate=False):
    import os
    from kbound_edge.real_dataset import load_window, SESSION_SPLIT_MAP
    from kbound_edge.evidence import edge_evidence_vector
    from kbound_edge.model import predict_proba
    from kbound_edge.dataset import frames_to_tensor
    
    Z, B, s_ids, source_hashes = [], [], [], []
    windows_dir = os.path.normpath(os.path.join(edge_dir, cfg["paths"]["windows_dir"]))
    
    for s_id in sessions:
        split_name = SESSION_SPLIT_MAP[s_id]
        split_dir = os.path.join(windows_dir, split_name)
        if not os.path.exists(split_dir):
            print(f"[04] WARNING: Split directory not found: {split_dir}")
            continue
            
        count = 0
        for fname in sorted(os.listdir(split_dir)):
            if fname.startswith(".") or not fname.endswith(".npz"):
                continue
            if bypass_gate and count >= 10:
                break
            npz_path = os.path.join(split_dir, fname)
            payload, offline = load_window(npz_path)
            count += 1
            
            x = frames_to_tensor(payload["frames"], cfg["image_size"])
            p0 = predict_proba(f0, x)
            res = adapter.adapt(x)
            pa = predict_proba(res.model, x)
            
            z = edge_evidence_vector(p0, pa, res.upd_norm)
            
            labels = offline["labels"]
            froz = float((p0.argmax(1) == labels).mean())
            cand = float((pa.argmax(1) == labels).mean())
            
            Z.append(z)
            B.append(cand - froz)
            s_ids.append(s_id)
            source_hashes.extend([str(h) for h in payload["source_hashes"]])
            
    return np.asarray(Z), np.asarray(B), np.asarray(s_ids), np.asarray(source_hashes, dtype=object)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    ap.add_argument("--bypass-gate", action="store_true", help="fast mock calibration run")
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

    is_real = cfg.get("protocol", "edge_label_inspection_v1") == "edge_real_phone_v1"
    edge_dir = os.path.normpath(os.path.join(_HERE, ".."))

    if not is_real:
        # --- Synthetic Mode ---
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
            p0 = predict_proba(f0, x)
            res = adapter.adapt(x)
            pa = predict_proba(res.model, x)
            z = edge_evidence_vector(p0, pa, res.upd_norm)
            froz = float((p0.argmax(1) == c.labels).mean())
            cand = float((pa.argmax(1) == c.labels).mean())
            Z.append(z)
            B.append(cand - froz)
            regimes.append(c.regime)
            divs.append(c.diversity)
            cond_ids.append(c.cond_id)

        Z = np.asarray(Z)
        B = np.asarray(B)
        out = C.resolve(cfg["paths"]["calibration"])
        C.ensure_parent(out)
        np.savez_compressed(out, Z=Z, B=B, regimes=np.array(regimes), diversities=np.array(divs),
                            cond_ids=np.array(cond_ids), feature_names=np.array(list(EDGE_EVIDENCE_NAMES)),
                            model_version=version)
        print(f"[04] calibration pairs: Z={Z.shape} B in [{B.min():+.3f}, {B.max():+.3f}] (mean {B.mean():+.3f}) -> {out}")
    else:
        # --- Real Manifest Mode ---
        print("[04] Processing calibration-fit splits S03 and S04...")
        Z_fit, B_fit, s_fit, h_fit = load_real_pairs(cfg, ["S03", "S04"], edge_dir, f0, adapter, version, bypass_gate=args.bypass_gate)
        fit_out = C.resolve(cfg["paths"]["calibration_fit"])
        C.ensure_parent(fit_out)
        np.savez_compressed(fit_out, Z=Z_fit, B=B_fit, sessions=s_fit, source_hashes=h_fit, model_version=version)
        print(f"[04] calibration-fit pairs: Z={Z_fit.shape} B in [{B_fit.min():+.3f}, {B_fit.max():+.3f}] -> {fit_out}")
        
        print("[04] Processing calibration-conformal splits S05 and S06...")
        Z_conf, B_conf, s_conf, h_conf = load_real_pairs(cfg, ["S05", "S06"], edge_dir, f0, adapter, version, bypass_gate=args.bypass_gate)
        conf_out = C.resolve(cfg["paths"]["calibration_conformal"])
        C.ensure_parent(conf_out)
        np.savez_compressed(conf_out, Z=Z_conf, B=B_conf, sessions=s_conf, source_hashes=h_conf, model_version=version)
        print(f"[04] calibration-conformal pairs: Z={Z_conf.shape} B in [{B_conf.min():+.3f}, {B_conf.max():+.3f}] -> {conf_out}")

if __name__ == "__main__":
    main()
