"""
train_iwildcam_f0.py - train a PROPER WILDS-iWildCam source model f0 for K-Bound.

WHY: the finder's f0 was a ResNet18 *head-only* model trained for ~120-240 batches,
which sits at chance (id_val balanced-acc ~0.10 on a 182-class task).  K-Bound
help/harm structure measured on a chance model is an artifact.  This trains a real
ResNet-50 ERM source model (WILDS-standard backbone) by full fine-tuning from
ImageNet, with leakage-free model selection on id_val (in-distribution).  The OOD
splits val/test are NEVER seen here -> they stay clean for the TTA evaluation.

INTEGRITY: plain ERM (shuffle), standard recipe; nothing tuned to a downstream
K-Bound target.  Every epoch's id_val / val macro-F1 + accuracy is logged.  Best
checkpoint is selected by id_val macro-F1 (source-side selection).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.models as tvm
from sklearn.metrics import f1_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # partial extraction left some truncated JPEGs


class RobustDS(torch.utils.data.Dataset):
    """Skip the rare fully-corrupt JPEG (UnidentifiedImageError) by substituting the
    next decodable sample. Truncated files load via LOAD_TRUNCATED_IMAGES above."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        n = len(self.base)
        for k in range(16):
            try:
                return self.base[(i + k) % n]
            except Exception:
                continue
        raise RuntimeError("too many consecutive corrupt images")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_iwildcam_kbound as R  # reuse disk-filtered loader + transforms

NUM_CLASSES = 182


def build_model(device):
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m.to(device)


@torch.no_grad()
def evaluate(model, sub, device, n, seed, bs=64, workers=4):
    rng = np.random.default_rng(seed)
    picks = np.arange(len(sub))
    if len(picks) > n:
        picks = rng.choice(picks, n, replace=False)
    dl = DataLoader(Subset(sub, picks.tolist()), batch_size=bs, shuffle=False,
                    num_workers=workers, persistent_workers=False)
    model.eval()
    preds, ys = [], []
    for xb, yb, _ in dl:
        out = model(xb.to(device))
        preds.append(out.argmax(1).cpu().numpy()); ys.append(yb.numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys).astype(int)
    macro_f1 = float(f1_score(ys, preds, average="macro"))
    acc = float((preds == ys).mean())
    return {"n": int(len(ys)), "acc": acc, "macro_f1": macro_f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(R.REPO / "experiments/kbound/data/wilds"))
    p.add_argument("--out-dir", default=str(R.REPO / "experiments/kbound/results/iwildcam_f0_erm"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eval-n", type=int, default=3000)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"trainlog_seed{args.seed}.json"
    ckpt_best = out / f"f0_resnet50_erm_seed{args.seed}.pt"
    ckpt_last = out / f"f0_resnet50_erm_seed{args.seed}_last.pt"

    device = torch.device("mps")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[cfg] {vars(args)}", flush=True)

    ds, train_sub, y_tr, _ = R.get_iwildcam(args.data_root, "train", train_tf=True)
    _, idval_sub, _, _ = R.get_iwildcam(args.data_root, "id_val", train_tf=False)
    _, val_sub, _, _ = R.get_iwildcam(args.data_root, "val", train_tf=False)
    train_sub = RobustDS(train_sub); idval_sub = RobustDS(idval_sub); val_sub = RobustDS(val_sub)
    print(f"[data] train={len(train_sub)} id_val={len(idval_sub)} val={len(val_sub)} "
          f"train_classes={len(set(y_tr.tolist()))}", flush=True)

    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_sub) // args.bs
    total_steps = (args.max_steps or steps_per_epoch * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_sub, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        drop_last=True, persistent_workers=True, generator=g)

    log = {"config": vars(args), "epochs": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    best_f1 = -1.0
    gstep = 0
    t0 = time.time()
    done = False
    for ep in range(args.epochs):
        model.train()
        run_loss = []
        te = time.time()
        for xb, yb, _ in loader:
            xb = xb.to(device); yb = yb.to(device).long()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run_loss.append(float(loss.detach().cpu())); gstep += 1
            if gstep % 100 == 0:
                ips = 100 * args.bs / (time.time() - te); te = time.time()
                print(f"  ep{ep} step{gstep}/{total_steps} loss={np.mean(run_loss[-100:]):.3f} "
                      f"{ips:.0f}img/s lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if args.max_steps and gstep >= args.max_steps:
                done = True; break
        idv = evaluate(model, idval_sub, device, args.eval_n, 7, workers=args.workers)
        vv = evaluate(model, val_sub, device, args.eval_n, 13, workers=args.workers)
        rec = {"epoch": ep, "gstep": gstep, "train_loss": float(np.mean(run_loss)) if run_loss else None,
               "id_val": idv, "val": vv, "wall_min": round((time.time() - t0) / 60, 1)}
        log["epochs"].append(rec)
        print(f"[epoch {ep}] id_val_f1={idv['macro_f1']:.4f} acc={idv['acc']:.4f} | "
              f"val_f1={vv['macro_f1']:.4f} acc={vv['acc']:.4f} | wall={rec['wall_min']}min", flush=True)
        torch.save({"model": model.state_dict(), "backbone": "resnet50", "epoch": ep,
                    "id_val_macro_f1": idv["macro_f1"], "val_macro_f1": vv["macro_f1"],
                    "selection": "id_val_macro_f1"}, ckpt_last)
        if idv["macro_f1"] > best_f1:
            best_f1 = idv["macro_f1"]
            torch.save({"model": model.state_dict(), "backbone": "resnet50", "epoch": ep,
                        "id_val_macro_f1": idv["macro_f1"], "val_macro_f1": vv["macro_f1"],
                        "selection": "id_val_macro_f1"}, ckpt_best)
            log["best"] = rec
            print(f"  -> new best id_val_f1={best_f1:.4f}; saved {ckpt_best.name}", flush=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        if done:
            break
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    log["total_min"] = round((time.time() - t0) / 60, 1)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[DONE] best id_val_macro_f1={best_f1:.4f} -> {ckpt_best}", flush=True)


if __name__ == "__main__":
    main()
