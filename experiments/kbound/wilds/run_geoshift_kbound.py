"""
run_geoshift_kbound.py — K-Bound finder scan on WILDS FMoW or PovertyMap.

Protocol L pipeline: dev-screen on id_val -> full GPU on val/test -> analyze_F.
FMoW: geographic region shift within OOD splits (62 land-use classes, accuracy).
PovertyMap: country shift with train-quantile binned wealth (5 classes, accuracy proxy).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.models as tvm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis as an  # noqa: E402
import fmow_data as fd  # noqa: E402
import poverty_data as pd  # noqa: E402
import run_camelyon17_kbound as rc  # noqa: E402
import tta_methods as tm  # noqa: E402

BATCH_REGIMES = {"tiny": 8, "small": 16}
AGGR = {"mild": {"steps": 10, "lr": 1e-3}, "aggressive": {"steps": 30, "lr": 2.0e-3}}
DEFAULT_CANDIDATES = ["tent_online", "eata_online", "sar_online"]

DATASET_CFG = {
    "fmow": {
        "schema": "kbound_wilds_fmow_finder_v0.1",
        "dataset_tag": "wilds-fmow",
        "metric": "accuracy",
        "group_name": "region",
        "domain_prefix": "region",
    },
    "poverty": {
        "schema": "kbound_wilds_poverty_finder_v0.1",
        "dataset_tag": "wilds-poverty",
        "metric": "accuracy_binned_wealth",
        "group_name": "country",
        "domain_prefix": "country",
    },
}


def acc_metric(y_true, preds):
    return float((np.asarray(preds) == np.asarray(y_true)).mean())


def load_split(dataset: str, root: str, split: str, train_tf: bool = False):
    if dataset == "fmow":
        ds, sub, y, groups = fd.get_fmow(root, split, train_tf=train_tf)
        sub, y, groups = fd.filter_present_subset(ds, sub, y, groups)
        return ds, sub, y, groups, None
    if dataset == "poverty":
        ds, sub, y, groups, edges = pd.get_poverty(root, split)
        return ds, sub, y, groups, edges
    raise ValueError(dataset)


def make_model(dataset: str, backbone: str, num_classes: int, device: torch.device):
    if dataset == "poverty":
        return pd.make_poverty_resnet(backbone, device)
    if backbone == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    elif backbone == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(backbone)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device)


def set_trainable(model: nn.Module, mode: str):
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "head":
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif mode == "layer4_head":
        for p in model.layer4.parameters():
            p.requires_grad_(True)
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(mode)


def atomic_dump(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _cell_key(seed, grp, comp, regime, aggr):
    return (int(seed), int(grp), comp, regime, aggr)


def load_partial(partial_path: Path):
    if not partial_path.exists():
        return [], [], set()
    with partial_path.open() as f:
        d = json.load(f)
    records = d.get("records", [])
    conditions = d.get("conditions", [])
    done = {_cell_key(c["seed"], c["location"], c["comp"], c["regime"], c["aggr"]) for c in conditions}
    return records, conditions, done


def train_or_load_f0(args, num_classes: int, device: torch.device, out_dir: Path):
    ckpt = Path(args.ckpt) if args.ckpt else out_dir / f"f0_{args.backbone}_seed{args.train_seed}.pt"
    model = make_model(args.dataset, args.backbone, num_classes, device)
    if ckpt.exists() and not args.retrain:
        obj = torch.load(ckpt, map_location=device, weights_only=False)
        state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"[f0] loaded {ckpt}", flush=True)
        return model, str(ckpt), {"loaded": True}

    print(f"[f0] training {args.dataset} {args.backbone} ({args.trainable})", flush=True)
    _, train_sub, y_train, _, _ = load_split(args.dataset, args.data_root, "train", train_tf=True)
    g = torch.Generator().manual_seed(args.train_seed)
    sampler = None
    shuffle = True
    if args.balanced_train:
        counts = np.bincount(y_train, minlength=num_classes).astype(float)
        counts[counts == 0.0] = 1.0
        weights = torch.as_tensor(1.0 / counts[y_train], dtype=torch.double)
        n_samples = int((args.max_train_batches or max(1, len(train_sub) // args.train_bs)) * args.train_bs)
        sampler = WeightedRandomSampler(weights, num_samples=n_samples, replacement=True, generator=g)
        shuffle = False
    loader = DataLoader(
        train_sub, batch_size=args.train_bs, shuffle=shuffle, sampler=sampler,
        generator=g, num_workers=args.workers, pin_memory=False, drop_last=True,
    )
    set_trainable(model, args.trainable)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.train_lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    steps = 0
    losses = []
    for _ep in range(args.train_epochs):
        for xb, yb, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device).long()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if steps % max(args.log_every, 1) == 0:
                print(f"  [f0] step={steps} loss={np.mean(losses[-args.log_every:]):.4f}", flush=True)
            if args.max_train_batches and steps >= args.max_train_batches:
                break
        if args.max_train_batches and steps >= args.max_train_batches:
            break
    model.eval()
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": model.state_dict(),
        "dataset": args.dataset,
        "backbone": args.backbone,
        "steps": steps,
        "wall_sec": round(time.time() - t0, 1),
    }
    torch.save(meta, ckpt)
    print(f"[f0] saved {ckpt} steps={steps}", flush=True)
    return model, str(ckpt), {"loaded": False, **{k: v for k, v in meta.items() if k != "model"}}


@torch.no_grad()
def eval_subset(model, sub, device, n: int, seed: int, bs: int):
    rng = np.random.default_rng(seed)
    picks = np.arange(len(sub))
    if len(picks) > n:
        picks = rng.choice(picks, n, replace=False)
    loader = DataLoader(Subset(sub, picks.tolist()), batch_size=bs, shuffle=False, num_workers=0)
    preds, ys = [], []
    model.eval()
    for xb, yb, _ in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys).astype(int)
    return {"n": int(len(ys)), "acc": acc_metric(ys, preds), "balanced_acc": tm.balanced_acc(preds, ys)}


def select_groups(y, groups, max_groups: int, min_count: int):
    rows = []
    for g in sorted(set(groups.tolist())):
        pos = np.where(groups == g)[0]
        if len(pos) < min_count:
            continue
        rows.append((g, len(pos), len(np.unique(y[pos]))))
    rows.sort(key=lambda r: (r[2], r[1]), reverse=True)
    return rows[:max_groups]


def load_positions(sub, positions, device, rng, fallback_pool=None, tries=80):
    xs = []
    pool = np.asarray(fallback_pool if fallback_pool is not None else positions)
    for p in positions:
        try:
            x, _, _ = sub[int(p)]
            xs.append(x)
            continue
        except Exception:
            pass
        ok = False
        for q in rng.permutation(pool)[:tries]:
            try:
                x, _, _ = sub[int(q)]
                xs.append(x)
                ok = True
                break
            except Exception:
                continue
        if not ok:
            raise RuntimeError("no readable sample in group")
    return torch.stack(xs).to(device)


def build_condition(sub, y, groups, grp, comp, bs, n_eval, n_batches, rng, device):
    pos_all = np.where(groups == int(grp))[0]
    if len(pos_all) < bs * n_batches + 2:
        raise RuntimeError(f"group {grp} too small: n={len(pos_all)}")
    classes = np.unique(y[pos_all])
    per = max(1, n_eval // max(1, len(classes)))
    ev = []
    for c in classes:
        ci = pos_all[y[pos_all] == c]
        if len(ci):
            ev.append(rng.choice(ci, min(per, len(ci)), replace=False))
    ev = np.concatenate(ev) if ev else rng.choice(pos_all, min(n_eval, len(pos_all)), replace=False)
    if len(ev) > n_eval:
        ev = rng.choice(ev, n_eval, replace=False)
    rng.shuffle(ev)
    remain = np.setdiff1d(pos_all, ev)
    if len(remain) == 0:
        remain = pos_all
    n_stream = bs * n_batches
    if comp == "iid":
        s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    elif comp == "imbalanced":
        counts = Counter(y[remain].tolist())
        maj = counts.most_common(1)[0][0]
        mp = remain[y[remain] == maj]
        op = remain[y[remain] != maj]
        n_maj = int(round(0.85 * n_stream))
        if len(mp) and len(op):
            s = np.concatenate([
                rng.choice(mp, n_maj, replace=len(mp) < n_maj),
                rng.choice(op, n_stream - n_maj, replace=len(op) < (n_stream - n_maj)),
            ])
        else:
            s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    elif comp == "single_class":
        counts = Counter(y[remain].tolist())
        cls = counts.most_common(1)[0][0]
        pool = remain[y[remain] == cls]
        s = rng.choice(pool, n_stream, replace=len(pool) < n_stream)
    else:
        raise ValueError(comp)
    rng.shuffle(s)
    stream_x = load_positions(sub, s, device, rng, fallback_pool=remain)
    eval_x = load_positions(sub, ev, device, rng, fallback_pool=pos_all)
    eval_y = y[ev].astype(int)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    return stream, eval_x, eval_y


def parse_candidates(names):
    return [(n.split("_", 1)[0], n.split("_", 1)[1]) for n in names]


def run_scan(args, f0, num_classes: int, device: torch.device, out_dir: Path):
    cfg = DATASET_CFG[args.dataset]
    partial = out_dir / "_partial.json"
    records, conditions, done = ([], [], set())
    if args.resume:
        records, conditions, done = load_partial(partial)
    _, sub, y, groups, _ = load_split(args.dataset, args.data_root, args.split, train_tf=False)
    min_count = args.n_eval + max(BATCH_REGIMES[r] for r in args.batch_regimes) * args.n_batches
    grp_rows = select_groups(y, groups, args.max_groups, min_count)
    if not grp_rows:
        raise RuntimeError(f"no groups in split={args.split}")
    print(f"[{args.dataset} groups] " + ", ".join(f"{g}(n={n},cls={c})" for g, n, c in grp_rows), flush=True)
    candidates = parse_candidates(args.candidates)
    n_cells = len(args.seeds) * len(grp_rows) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness)
    t0 = time.time()
    ci = 0
    for seed in args.seeds:
        for grp, grp_n, grp_classes in grp_rows:
            for comp in args.compositions:
                for regime in args.batch_regimes:
                    bs = BATCH_REGIMES[regime]
                    for aggr in args.aggressiveness:
                        ci += 1
                        tag = f"s{seed}/{cfg['domain_prefix']}{grp}/{comp}/{regime}/{aggr}"
                        if _cell_key(seed, grp, comp, regime, aggr) in done:
                            print(f"  [{ci}/{n_cells}] {tag} SKIP", flush=True)
                            continue
                        cell_seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
                        rng = np.random.default_rng(cell_seed)
                        torch.manual_seed(cell_seed)
                        try:
                            stream, eval_x, eval_y = build_condition(
                                sub, y, groups, grp, comp, bs, args.n_eval, args.n_batches, rng, device
                            )
                            steps = args.steps_override or AGGR[aggr]["steps"]
                            lr = AGGR[aggr]["lr"]
                            a0_bacc, p0, p0_pos = tm.eval_frozen(f0, eval_x, eval_y, prob_mode="max", bs=args.eval_bs)
                            a0 = acc_metric(eval_y, p0)
                            stream_f0_pos = tm._predict_prob(
                                f0, torch.cat(stream, 0), train_mode=False, bs=args.eval_bs, mode="max"
                            )
                            preds_all = [p0]
                            aa_all = [a0]
                            cand_names = ["freeze_f0"]
                            best_pa, best_aa = p0_pos, float(a0)
                            for method, mode in candidates:
                                aa_bacc, z, upd, preds, pa_pos = tm.run_candidate(
                                    method, mode, f0, stream, eval_x, eval_y, num_classes,
                                    steps, lr, eval_bs=args.episodic_batch, prob_mode="max",
                                    episodic_steps=args.episodic_steps,
                                )
                                aa = acc_metric(eval_y, preds)
                                cand = f"{method}_{mode}"
                                records.append({
                                    "seed": int(seed),
                                    "domain": f"{cfg['domain_prefix']}{grp}",
                                    "location": int(grp),
                                    "location_n": int(grp_n),
                                    "location_classes": int(grp_classes),
                                    "split": args.split,
                                    "comp": comp,
                                    "regime": regime,
                                    "aggr": aggr,
                                    "method": method,
                                    "mode": mode,
                                    "candidate": cand,
                                    "metric": cfg["metric"],
                                    "a0": float(a0),
                                    "aa": float(aa),
                                    "B": float(aa - a0),
                                    "a0_bacc": float(a0_bacc),
                                    "aa_bacc": float(aa_bacc),
                                    "upd_norm": float(upd),
                                    "Z": [float(v) for v in z],
                                    "preds": [int(v) for v in preds],
                                    "regime_label": an.label_regime(float(aa - a0)),
                                })
                                preds_all.append(preds)
                                aa_all.append(float(aa))
                                cand_names.append(cand)
                                if float(aa) > best_aa:
                                    best_aa, best_pa = float(aa), pa_pos
                                tm.mps_free()
                                gc.collect()
                            route = an.multicandidate_route(np.stack(preds_all, 0), tau_star=args.tau_star, kappa=args.kappa)
                            realized = rc.route_realized(route, aa_all)
                            oracle = float(max(aa_all))
                            best_adapt = float(max(aa_all[1:]))
                            conditions.append({
                                "seed": int(seed),
                                "domain": f"{cfg['domain_prefix']}{grp}",
                                "location": int(grp),
                                "location_n": int(grp_n),
                                "split": args.split,
                                "comp": comp,
                                "regime": regime,
                                "aggr": aggr,
                                "cand_names": cand_names,
                                "aa_all": [float(v) for v in aa_all],
                                "a0": float(a0),
                                "oracle": oracle,
                                "best_adapt": best_adapt,
                                "route": route,
                                "realized": realized,
                                "regime_label": an.label_regime(best_adapt - a0),
                            })
                            print(f"  [{ci}/{n_cells}] {tag} a0={a0:.3f} best={best_adapt:.3f} oracle={oracle:.3f}", flush=True)
                        except Exception as e:
                            print(f"  [{ci}/{n_cells}] {tag} ERROR: {repr(e)[:160]}", flush=True)
                        finally:
                            atomic_dump({"progress": f"{ci}/{n_cells}", "records": records, "conditions": conditions}, partial)
                            gc.collect()
                            tm.mps_free()
    return records, conditions, {"target_groups": grp_rows, "wall_sec": time.time() - t0}


def summarize(records):
    if not records:
        return {"note": "no records"}
    B = np.array([r["B"] for r in records], float)
    det = an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {}
    if float(np.mean(B < 0)) < 0.10 and float(B.mean()) > 0:
        cls = "helpful-dominated"
    elif float(np.mean(B < 0)) > 0.60:
        cls = "harmful-dominated"
    else:
        cls = "mixed+detectable" if det.get("detectability_verdict") == "detectable" else "mixed+undetectable"
    return {
        "classification": cls,
        "n_records": len(records),
        "mean_B": float(B.mean()),
        "base_rate_harmful_B<0": float(np.mean(B < 0)),
        "detectability_verdict": det.get("detectability_verdict"),
        "best_single_feature_harm_AUC": det.get("best_single_feature_harm_AUC"),
    }


def build_manifest(args, num_classes, f0_ckpt, train_meta, eval_meta, records, conditions, meta):
    cfg = DATASET_CFG[args.dataset]
    sha = hashlib.sha256(json.dumps(vars(args), sort_keys=True, default=str).encode()).hexdigest()[:8]
    return {
        "schema": cfg["schema"],
        "dataset": cfg["dataset_tag"],
        "metric": cfg["metric"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"node": platform.node(), "python": platform.python_version(), "torch": torch.__version__},
        "config": vars(args),
        "config_sha8": sha,
        "f0_checkpoint": f0_ckpt,
        "f0_training": train_meta,
        "f0_quick_eval": eval_meta,
        "num_classes": num_classes,
        "evidence_names": tm.EVIDENCE_NAMES,
        "kbound_summary": summarize(records),
        "detectability": an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {},
        "records": records,
        "conditions": conditions,
        "data": {"target_groups": meta["target_groups"], "wall_sec": round(meta["wall_sec"], 1)},
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="K-Bound geoshift finder (FMoW / PovertyMap)")
    p.add_argument("--dataset", required=True, choices=["fmow", "poverty"])
    p.add_argument("--data-root", default=str(REPO / "experiments/kbound/data/wilds"))
    p.add_argument("--results-root", default=str(REPO / "experiments/kbound/results"))
    p.add_argument("--run-name", default="")
    p.add_argument("--ckpt", default="")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--trainable", choices=["head", "layer4_head", "full"], default="head")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--max-train-batches", type=int, default=200)
    p.add_argument("--train-bs", type=int, default=32)
    p.add_argument("--train-lr", type=float, default=1e-3)
    p.add_argument("--balanced-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--split", default="val", choices=["val", "test", "id_val", "id_test"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max-groups", type=int, default=6, dest="max_groups")
    p.add_argument("--compositions", nargs="+", default=["iid", "imbalanced", "single_class"])
    p.add_argument("--batch-regimes", nargs="+", default=["tiny", "small"], dest="batch_regimes")
    p.add_argument("--aggressiveness", nargs="+", default=["mild", "aggressive"])
    p.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    p.add_argument("--n-eval", type=int, default=64, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=2, dest="n_batches")
    p.add_argument("--eval-bs", type=int, default=64, dest="eval_bs")
    p.add_argument("--episodic-steps", type=int, default=3, dest="episodic_steps")
    p.add_argument("--episodic-batch", type=int, default=32, dest="episodic_batch")
    p.add_argument("--tau-star", type=float, default=0.52, dest="tau_star")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument("--steps-override", type=int, default=0, dest="steps_override")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--out", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    if not args.run_name:
        args.run_name = f"{args.dataset}_protocol_L_dev" if args.split == "id_val" else f"{args.dataset}_protocol_L_{args.split}"
    if args.smoke:
        args.run_name = f"{args.dataset}_smoke"
        args.max_train_batches = 6
        args.max_groups = 2
        args.n_eval = 16
        args.n_batches = 1
        args.seeds = [0]
        args.compositions = ["iid"]
        args.batch_regimes = ["tiny"]
        args.aggressiveness = ["mild"]
        args.candidates = ["tent_online"]
        args.steps_override = 2
    return args


def main(argv=None):
    args = parse_args(argv)
    num_classes = fd.NUM_CLASSES if args.dataset == "fmow" else pd.NUM_CLASSES
    out_dir = Path(args.results_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        _, _, y, groups, _ = load_split(args.dataset, args.data_root, args.split)
        min_count = args.n_eval + max(BATCH_REGIMES[r] for r in args.batch_regimes) * args.n_batches
        print(f"DRY RUN {args.dataset} split={args.split} groups={select_groups(y, groups, args.max_groups, min_count)}")
        return None
    device = tm.pick_device(args.device)
    f0, ckpt, train_meta = train_or_load_f0(args, num_classes, device, out_dir)
    _, id_sub, _, _, _ = load_split(args.dataset, args.data_root, "id_val")
    _, tgt_sub, _, _, _ = load_split(args.dataset, args.data_root, args.split)
    eval_meta = {
        "id_val": eval_subset(f0, id_sub, device, 128, args.train_seed + 3, args.eval_bs),
        args.split: eval_subset(f0, tgt_sub, device, 128, args.train_seed + 9, args.eval_bs),
    }
    records, conditions, meta = run_scan(args, f0, num_classes, device, out_dir)
    manifest = build_manifest(args, num_classes, ckpt, train_meta, eval_meta, records, conditions, meta)
    out = Path(args.out) if args.out else out_dir / f"result_{manifest['config_sha8']}.json"
    out.write_text(json.dumps(manifest, indent=2))
    s = manifest["kbound_summary"]
    print(f"\nDONE {args.dataset} classification={s.get('classification')} harm_AUC={s.get('best_single_feature_harm_AUC')}")
    print(f"manifest -> {out}")
    return str(out)


if __name__ == "__main__":
    main()
