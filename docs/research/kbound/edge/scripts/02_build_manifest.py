#!/usr/bin/env python3
"""02 -- build the run MANIFEST: the locked record of what will be processed.

Reads the source clip + protocol config and writes a manifest.json capturing the
classes, image/window size, alpha, the calibration / held-out condition plans and
the resolved artifact paths.  Downstream scripts and the report read this.
"""

import argparse
import numpy as np

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="edge_label_inspection_v1.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    clip_path = C.resolve(cfg["paths"]["source_clip"])
    try:
        clip = np.load(clip_path)
        n_frames = int(clip["frames"].shape[0])
        clip_shape = list(clip["frames"].shape)
    except FileNotFoundError:
        raise SystemExit(f"[02] source clip not found: {clip_path}. Run 01_capture_source.py first.")

    manifest = {
        "protocol": cfg["protocol"],
        "schema": "kbound-edge-manifest-v1",
        "classes": cfg["classes"],
        "num_classes": cfg["num_classes"],
        "image_size": cfg["image_size"],
        "window_size": cfg["window_size"],
        "alpha": cfg["alpha"],
        "seed": cfg["seed"],
        "source": {"kind": cfg["source"]["kind"], "n_frames": n_frames, "clip_shape": clip_shape},
        "calibration_plan": cfg["calibration_plan"],
        "heldout_plan": cfg["heldout_plan"],
        "paths": {k: C.resolve(v) for k, v in cfg["paths"].items()},
    }
    # config hash over the protocol-defining fields
    from kbound_edge.logging import config_hash

    manifest["config_hash"] = config_hash(C.clean_config(cfg))

    out = C.resolve(cfg["paths"]["manifest"])
    C.save_json(out, manifest)
    print(f"[02] manifest written -> {out}")
    print(f"[02] source frames={n_frames} window={cfg['window_size']} alpha={cfg['alpha']} "
          f"config_hash={manifest['config_hash']}")


if __name__ == "__main__":
    main()
