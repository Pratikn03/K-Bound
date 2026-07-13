#!/usr/bin/env python3
"""01 -- capture / generate the SOURCE clip used to train the frozen model f0.

Synthetic (source.kind=synthetic): generate a balanced, clean, LABELLED clip and
save it as an .npz (frames + labels).  This is OFFLINE data -- labels are allowed
here (training is offline); they never touch the online path.

Real (source.kind=opencv): record a clean clip from a camera / phone-as-camera or
a video file and save the frames.  Labelling a real source set is a manual step
(see README); this script just captures the frames.
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
    src = cfg["source"]
    out = C.resolve(cfg["paths"]["source_clip"])
    C.ensure_parent(out)

    if src["kind"] == "synthetic":
        from kbound_edge.dataset import make_training_clip

        frames, labels = make_training_clip(
            n_per_class=src["n_per_class"],
            image_size=cfg["image_size"],
            seed=src["seed"],
            n_classes=cfg["num_classes"],
        )
        arr = np.stack(frames).astype("uint8")
        np.savez_compressed(out, frames=arr, labels=labels, image_size=cfg["image_size"])
        print(f"[01] synthetic source clip: frames={arr.shape} classes={cfg['num_classes']} -> {out}")

    elif src["kind"] == "opencv":
        from kbound_edge.capture import open_opencv_source

        target = src.get("video_path") or src.get("camera_index", 0)
        cap = open_opencv_source(target)
        frames = []
        max_frames = int(src.get("n_per_class", 120)) * int(cfg["num_classes"])
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise SystemExit("[01] no frames captured from the OpenCV source")
        arr = np.stack(frames).astype("uint8")
        # Real source labelling is a manual step; save -1 placeholders.
        labels = np.full(len(arr), -1, dtype=int)
        np.savez_compressed(out, frames=arr, labels=labels, image_size=cfg["image_size"])
        print(f"[01] captured {arr.shape} frames from '{target}' -> {out}")
        print("[01] NOTE: real frames are UNLABELLED (labels=-1). Provide labels before 03 (see README).")
    else:
        raise SystemExit(f"[01] unknown source.kind={src['kind']!r}")


if __name__ == "__main__":
    main()
