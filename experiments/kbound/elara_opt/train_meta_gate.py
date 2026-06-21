"""train_meta_gate.py — train the ELARA-Opt meta gate (elara_meta) on DEV/source
shift tasks ONLY.

For each DEV task we (a) extract the label-free reliability features at the initial
state, (b) score every weight PRESET by its DEV-labeled benefit, and (c) make the
argmax preset's weights the regression target.  A tiny MLP is then fit to map
features -> preset weights.  Training-data IDs, seeds, and per-task choices are
saved alongside the checkpoint.  STRICT no-overlap: these synthetic DEV task IDs
are disjoint from the nine held-out evaluation datasets.

Run:  PYTHONPATH=.:packaging/kbound-tta/src ~/.venv_wilds/bin/python \
        experiments/kbound/elara_opt/train_meta_gate.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import ELARA_OPT_DEFAULTS
from ._compat import eval_frozen, predict_logits, balanced_acc
from .reliability import compute_features, to_vector, RELIABILITY_NAMES, FEATURE_DIM
from .gate import MetaGate
from .elara_opt import elara_opt_adapt
from .smoke_models import build_f0, synth_cell

_HERE = os.path.dirname(os.path.abspath(__file__))

# DEV / source shift tasks — explicit IDs, disjoint from the 9 eval datasets.
DEV_TASKS = [
    {"id": "dev_synth_shift_c10_s05", "num_classes": 10, "hw": 32, "in_ch": 3, "seed": 1001, "shift": 0.5},
    {"id": "dev_synth_shift_c10_s15", "num_classes": 10, "hw": 32, "in_ch": 3, "seed": 1002, "shift": 1.5},
    {"id": "dev_synth_shift_c10_s30", "num_classes": 10, "hw": 32, "in_ch": 3, "seed": 1003, "shift": 3.0},
    {"id": "dev_synth_shift_c02_s10", "num_classes": 2,  "hw": 32, "in_ch": 3, "seed": 1004, "shift": 1.0},
    {"id": "dev_synth_shift_c02_s25", "num_classes": 2,  "hw": 32, "in_ch": 3, "seed": 1005, "shift": 2.5},
    {"id": "dev_synth_shift_c05_s08", "num_classes": 5,  "hw": 32, "in_ch": 3, "seed": 1006, "shift": 0.8},
    {"id": "dev_synth_shift_c05_s20", "num_classes": 5,  "hw": 32, "in_ch": 3, "seed": 1007, "shift": 2.0},
    {"id": "dev_synth_shift_c20_s12", "num_classes": 20, "hw": 32, "in_ch": 3, "seed": 1008, "shift": 1.2},
    {"id": "dev_synth_shift_c20_s35", "num_classes": 20, "hw": 32, "in_ch": 3, "seed": 1009, "shift": 3.5},
    {"id": "dev_synth_shift_c65_s10", "num_classes": 65, "hw": 32, "in_ch": 3, "seed": 1010, "shift": 1.0},
    {"id": "dev_synth_shift_c65_s28", "num_classes": 65, "hw": 32, "in_ch": 3, "seed": 1011, "shift": 2.8},
    {"id": "dev_synth_shift_c10_s40", "num_classes": 10, "hw": 32, "in_ch": 3, "seed": 1012, "shift": 4.0},
]


def _benefit(f0, fa, eval_x, dev_y):
    a0, _, _ = eval_frozen(f0, eval_x, dev_y)
    preds = predict_logits(fa, eval_x, train_mode=True).argmax(axis=1)
    return float(balanced_acc(preds, dev_y) - a0)


def build_training_set(cfg, n=24, lr=1e-3):
    presets = cfg["meta"]["presets"]
    preset_names = list(presets.keys())
    feats_list, target_w, chosen = [], [], {}
    for t in DEV_TASKS:
        nc = t["num_classes"]
        f0 = build_f0(nc, t["in_ch"], seed=t["seed"])
        stream, eval_x, dev_y = synth_cell(nc, n, t["in_ch"], t["hw"], t["seed"], t["shift"])
        phi = compute_features(f0, f0, stream[0], nc)  # initial-state, label-free
        best_name, best_B = None, -1e9
        for name in preset_names:
            w = np.array(presets[name], dtype=np.float64)
            fa, _, _ = elara_opt_adapt(f0, stream, 1, lr, nc, fixed_weights=w,
                                       seed=t["seed"], collect_telemetry=False)
            B = _benefit(f0, fa, eval_x, dev_y)
            if B > best_B:
                best_B, best_name = B, name
        feats_list.append(to_vector(phi))
        target_w.append(np.array(presets[best_name], dtype=np.float64))
        chosen[t["id"]] = {"preset": best_name, "dev_benefit": round(best_B, 6)}
    return np.array(feats_list), np.array(target_w), chosen, preset_names


def train(cfg=None, epochs=400, seed=0):
    cfg = cfg or ELARA_OPT_DEFAULTS
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, W, chosen, preset_names = build_training_set(cfg)
    Xt = torch.tensor(X, dtype=torch.float32)
    Wt = torch.tensor(W, dtype=torch.float32)
    model = MetaGate(in_dim=FEATURE_DIM, hidden=int(cfg["meta"]["hidden"]))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    final = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        logp = F.log_softmax(model(Xt), dim=1)
        loss = -(Wt * logp).sum(1).mean()      # soft CE toward best-preset weights
        loss.backward()
        opt.step()
        final = float(loss.detach())

    ckpt_path = os.path.join(_HERE, cfg["meta"]["checkpoint"])
    ids_path = os.path.join(_HERE, cfg["meta"]["train_ids"])
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_dim": FEATURE_DIM,
                "hidden": int(cfg["meta"]["hidden"]), "config_version": cfg["version"],
                "feature_names": RELIABILITY_NAMES, "final_train_loss": final}, ckpt_path)
    with open(ids_path, "w") as fh:
        json.dump({
            "note": "ELARA-Opt meta gate trained ONLY on DEV/source synthetic shift "
                    "tasks; IDs are disjoint from the 9 held-out eval datasets "
                    "(cifar10c, imagenet_c, officehome, iwildcam, camelyon17, rxrx1, "
                    "imagenet_r, cifar101, fmow).",
            "task_ids": [t["id"] for t in DEV_TASKS],
            "seeds": [t["seed"] for t in DEV_TASKS],
            "presets": cfg["meta"]["presets"],
            "chosen_preset_per_task": chosen,
            "feature_names": RELIABILITY_NAMES,
            "train_seed": seed, "epochs": epochs, "final_train_loss": round(final, 6),
        }, fh, indent=2)
    return ckpt_path, ids_path, final


if __name__ == "__main__":
    cp, ip, loss = train()
    print(f"[train_meta_gate] saved checkpoint: {cp}")
    print(f"[train_meta_gate] saved train-ids:  {ip}")
    print(f"[train_meta_gate] final train loss: {loss:.6f}")
