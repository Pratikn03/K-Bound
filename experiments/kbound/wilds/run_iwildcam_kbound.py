"""
run_iwildcam_kbound.py - K-Bound finder scan on WILDS iWildCam.

Purpose: quickly test whether iWildCam has the natural mixed help/harm structure
K-Bound needs: some target locations where TTA helps, some where it hurts, and
label-free evidence that separates those cases.  This is a finder/preview runner,
not a locked paper protocol.
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
from sklearn.metrics import f1_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # partial extraction left some truncated JPEGs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.models as tvm
import torchvision.transforms as T

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis as an  # noqa: E402
import run_camelyon17_kbound as rc  # noqa: E402
import tta_methods as tm  # noqa: E402

NUM_CLASSES = 182
BATCH_REGIMES = {"tiny": 8, "small": 16}
AGGR = {"mild": {"steps": 10, "lr": 1e-3}, "aggressive": {"steps": 30, "lr": 2.0e-3}}
DEFAULT_CANDIDATES = ["tent_online", "eata_online", "sar_online"]
_PRESENT_CACHE: dict[str, set[str]] = {}


def macro_f1(y_true, preds):
    """WILDS-standard iWildCam metric: macro-averaged F1 over classes present."""
    return float(f1_score(np.asarray(y_true, int), np.asarray(preds, int), average="macro"))


def image_transform(train: bool):
    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_iwildcam(root: str, split: str, train_tf: bool = False):
    from wilds import get_dataset

    ds = get_dataset(dataset="iwildcam", download=False, root_dir=root)
    sub = ds.get_subset(split, transform=image_transform(train_tf))
    idx = np.asarray(sub.indices)
    # The local iWildCam archive was previously observed in a partially extracted
    # state.  Filter to files present on disk so preview runs fail closed instead
    # of crashing randomly inside WILDS/PIL.
    data_dir = Path(ds.data_dir) / "train"
    inp = ds._input_array
    present = present_jpgs(data_dir)
    keep = np.fromiter((str(inp[i]) in present for i in idx), dtype=bool, count=len(idx))
    if not bool(keep.all()):
        sub.indices = idx[keep]
        idx = np.asarray(sub.indices)
    y = ds.y_array[idx].numpy().astype(int)
    md = ds.metadata_array[idx].numpy()
    loc_i = ds.metadata_fields.index("location")
    locations = md[:, loc_i].astype(int)
    return ds, sub, y, locations


def present_jpgs(data_dir: Path):
    key = str(data_dir)
    if key in _PRESENT_CACHE:
        return _PRESENT_CACHE[key]
    cache = data_dir.parent / "_present_jpgs_cache.json"
    if cache.exists():
        try:
            with cache.open() as f:
                names = set(json.load(f)["names"])
            _PRESENT_CACHE[key] = names
            return names
        except Exception:
            pass
    names = {
        ent.name
        for ent in os.scandir(data_dir)
        if ent.is_file() and ent.name.endswith(".jpg") and not ent.name.startswith("._")
    }
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump({"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "count": len(names), "names": sorted(names)}, f)
    os.replace(tmp, cache)
    _PRESENT_CACHE[key] = names
    print(f"[iwildcam] present image cache: {len(names)} files -> {cache}", flush=True)
    return names


def make_model(backbone: str, device: torch.device):
    if backbone == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT
        model = tvm.resnet18(weights=weights)
    elif backbone == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT
        model = tvm.resnet50(weights=weights)
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


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
        raise ValueError(f"unknown trainable mode: {mode}")


def atomic_dump(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _iwc_cell_key(seed, loc, comp, regime, aggr):
    return (int(seed), int(loc), comp, regime, aggr)


def load_partial_iwc(partial_path: Path):
    if not partial_path.exists():
        return [], [], set()
    with partial_path.open() as f:
        d = json.load(f)
    records = d.get("records", [])
    conditions = d.get("conditions", [])
    done = {_iwc_cell_key(c["seed"], c["location"], c["comp"], c["regime"], c["aggr"])
            for c in conditions}
    return records, conditions, done


def train_or_load_f0(args, device: torch.device, out_dir: Path):
    ckpt = Path(args.ckpt) if args.ckpt else out_dir / f"f0_{args.backbone}_seed{args.train_seed}.pt"
    model = make_model(args.backbone, device)
    if ckpt.exists() and not args.retrain:
        obj = torch.load(ckpt, map_location=device, weights_only=False)
        state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"[f0] loaded {ckpt}", flush=True)
        return model, str(ckpt), {"loaded": True}

    print(f"[f0] training {args.backbone} ({args.trainable}) for iWildCam source preview", flush=True)
    _, train_sub, y_train, _ = get_iwildcam(args.data_root, "train", train_tf=True)
    g = torch.Generator().manual_seed(args.train_seed)
    sampler = None
    shuffle = True
    if args.balanced_train:
        counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
        counts[counts == 0.0] = 1.0
        weights = torch.as_tensor(1.0 / counts[y_train], dtype=torch.double)
        n_samples = int((args.max_train_batches or max(1, len(train_sub) // args.train_bs)) * args.train_bs)
        sampler = WeightedRandomSampler(weights, num_samples=n_samples, replacement=True, generator=g)
        shuffle = False
    loader = DataLoader(
        train_sub,
        batch_size=args.train_bs,
        shuffle=shuffle,
        sampler=sampler,
        generator=g,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
    )
    set_trainable(model, args.trainable)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.train_lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    steps = 0
    losses = []
    for ep in range(args.train_epochs):
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
        "backbone": args.backbone,
        "trainable": args.trainable,
        "balanced_train": bool(args.balanced_train),
        "steps": steps,
        "mean_loss_tail": float(np.mean(losses[-20:])) if losses else None,
        "wall_sec": round(time.time() - t0, 1),
    }
    torch.save(meta, ckpt)
    print(f"[f0] saved {ckpt} steps={steps} wall={meta['wall_sec']}s", flush=True)
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
        out = model(xb.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys).astype(int)
    return {"n": int(len(ys)), "acc": float((preds == ys).mean()), "balanced_acc": tm.balanced_acc(preds, ys)}


def select_locations(y, locations, max_locations: int, min_count: int):
    rows = []
    for loc in sorted(set(locations.tolist())):
        pos = np.where(locations == loc)[0]
        if len(pos) < min_count:
            continue
        n_classes = len(np.unique(y[pos]))
        rows.append((loc, len(pos), n_classes))
    rows.sort(key=lambda r: (r[2], r[1]), reverse=True)
    return rows[:max_locations]


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
            raise RuntimeError("no readable image found for sampled position")
    return torch.stack(xs).to(device)


def build_condition(sub, y, locations, loc, comp, bs, n_eval, n_batches, rng, device):
    pos_all = np.where(locations == int(loc))[0]
    if len(pos_all) < bs * n_batches + 2:
        raise RuntimeError(f"location {loc} too small: n={len(pos_all)}")
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
        raise ValueError(f"unknown composition: {comp}")
    rng.shuffle(s)

    stream_x = load_positions(sub, s, device, rng, fallback_pool=remain)
    eval_x = load_positions(sub, ev, device, rng, fallback_pool=pos_all)
    eval_y = y[ev].astype(int)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    return stream, eval_x, eval_y


def parse_candidates(names):
    out = []
    for name in names:
        if "_" not in name:
            raise ValueError(f"candidate must look like method_mode, got {name}")
        method, mode = name.split("_", 1)
        out.append((method, mode))
    return out


def run_scan(args, f0, device, out_dir: Path):
    partial = out_dir / "_partial.json"
    records, conditions, done = ([], [], set())
    if args.resume:
        records, conditions, done = load_partial_iwc(partial)
        if done:
            print(f"[resume] {len(done)} cells loaded from {partial}", flush=True)
    _, sub, y, locations = get_iwildcam(args.data_root, args.split, train_tf=False)
    min_count = args.n_eval + max(BATCH_REGIMES[r] for r in args.batch_regimes) * args.n_batches
    loc_rows = select_locations(y, locations, args.max_locations, min_count)
    if not loc_rows:
        raise RuntimeError(f"no target locations in split={args.split} with at least {min_count} samples")
    print("[target locations] " + ", ".join(f"{loc}(n={n},classes={c})" for loc, n, c in loc_rows), flush=True)
    candidates = parse_candidates(args.candidates)
    n_cells = len(args.seeds) * len(loc_rows) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness)
    t0 = time.time()
    ci = 0
    for seed in args.seeds:
        for loc, loc_n, loc_classes in loc_rows:
            for comp in args.compositions:
                for regime in args.batch_regimes:
                    bs = BATCH_REGIMES[regime]
                    for aggr in args.aggressiveness:
                        ci += 1
                        tag = f"s{seed}/loc{loc}/{comp}/{regime}/{aggr}"
                        if _iwc_cell_key(seed, loc, comp, regime, aggr) in done:
                            print(f"  [{ci}/{n_cells}] {tag} SKIP (resume)", flush=True)
                            continue
                        cell_seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
                        rng = np.random.default_rng(cell_seed)
                        torch.manual_seed(cell_seed)
                        try:
                            stream, eval_x, eval_y = build_condition(
                                sub, y, locations, loc, comp, bs, args.n_eval, args.n_batches, rng, device
                            )
                            steps = args.steps_override or AGGR[aggr]["steps"]
                            lr = AGGR[aggr]["lr"]
                            a0_bacc, p0, p0_pos = tm.eval_frozen(f0, eval_x, eval_y, prob_mode="max", bs=args.eval_bs)
                            a0 = macro_f1(eval_y, p0)            # headline metric = WILDS macro-F1
                            a0_acc = float((np.asarray(p0) == np.asarray(eval_y)).mean())
                            stream_f0_pos = tm._predict_prob(
                                f0, torch.cat(stream, 0), train_mode=False, bs=args.eval_bs, mode="max"
                            )
                            preds_all = [p0]
                            aa_all = [a0]
                            cand_names = ["freeze_f0"]
                            best_pa = p0_pos
                            best_aa = float(a0)
                            for method, mode in candidates:
                                aa_bacc, z, upd, preds, pa_pos = tm.run_candidate(
                                    method,
                                    mode,
                                    f0,
                                    stream,
                                    eval_x,
                                    eval_y,
                                    NUM_CLASSES,
                                    steps,
                                    lr,
                                    eval_bs=args.episodic_batch,
                                    prob_mode="max",
                                    episodic_steps=args.episodic_steps,
                                )
                                aa = macro_f1(eval_y, preds)     # headline metric = WILDS macro-F1
                                aa_acc = float((np.asarray(preds) == np.asarray(eval_y)).mean())
                                cand = f"{method}_{mode}"
                                B = float(aa - a0)
                                records.append({
                                    "seed": int(seed),
                                    "domain": f"loc{loc}",
                                    "location": int(loc),
                                    "location_n": int(loc_n),
                                    "location_classes": int(loc_classes),
                                    "split": args.split,
                                    "comp": comp,
                                    "regime": regime,
                                    "aggr": aggr,
                                    "method": method,
                                    "mode": mode,
                                    "candidate": cand,
                                    "metric": "macro_f1",
                                    "a0": float(a0),
                                    "aa": float(aa),
                                    "B": B,
                                    "a0_bacc": float(a0_bacc),
                                    "aa_bacc": float(aa_bacc),
                                    "a0_acc": float(a0_acc),
                                    "aa_acc": float(aa_acc),
                                    "upd_norm": float(upd),
                                    "Z": [float(v) for v in z],
                                    "preds": [int(v) for v in preds],
                                    "c0": [int(x) for x in (np.asarray(p0) == np.asarray(eval_y))],
                                    "ca": [int(x) for x in (np.asarray(preds) == np.asarray(eval_y))],
                                    "regime_label": an.label_regime(B),
                                })
                                preds_all.append(preds)
                                aa_all.append(float(aa))
                                cand_names.append(cand)
                                if float(aa) > best_aa:
                                    best_aa = float(aa)
                                    best_pa = pa_pos
                                tm.mps_free()
                                gc.collect()
                            route = an.multicandidate_route(np.stack(preds_all, 0), tau_star=args.tau_star, kappa=args.kappa)
                            realized = rc.route_realized(route, aa_all)
                            oracle = float(max(aa_all))
                            best_adapt = float(max(aa_all[1:]))
                            try:
                                route_c = an.smooth_drift_route(p0_pos, best_pa, stream_f0_pos, L=args.sd_L)
                                if route_c.get("implemented") and "bracket" in route_c:
                                    true_b = best_adapt - a0
                                    route_c["true_B_best"] = float(true_b)
                                    route_c["bracket_covers_trueB"] = bool(
                                        route_c["bracket"][0] <= true_b <= route_c["bracket"][1]
                                    )
                            except Exception as e:
                                route_c = {"decision": "ERROR", "implemented": False, "reason": repr(e)}
                            conditions.append({
                                "seed": int(seed),
                                "domain": f"loc{loc}",
                                "location": int(loc),
                                "location_n": int(loc_n),
                                "location_classes": int(loc_classes),
                                "split": args.split,
                                "comp": comp,
                                "regime": regime,
                                "aggr": aggr,
                                "cand_names": cand_names,
                                "aa_all": [float(v) for v in aa_all],
                                "a0": float(a0),
                                "a0_bacc": float(a0_bacc),
                                "metric": "macro_f1",
                                "oracle": oracle,
                                "best_adapt": best_adapt,
                                "true_best": cand_names[int(np.argmax(aa_all))],
                                "route": route,
                                "route_c": route_c,
                                "realized": realized,
                                "eval_y": [int(v) for v in eval_y],
                                "preds_frozen": [int(v) for v in p0],
                                "regime_label": an.label_regime(best_adapt - a0),
                            })
                            print(
                                f"  [{ci}/{n_cells}] {tag} a0={a0:.3f} best={best_adapt:.3f} "
                                f"oracle={oracle:.3f} route={route.get('decision')} "
                                f"tau={route.get('tau', float('nan')):.3f}",
                                flush=True,
                            )
                        except Exception as e:
                            print(f"  [{ci}/{n_cells}] {tag} ERROR: {repr(e)[:180]}", flush=True)
                        finally:
                            atomic_dump(
                                {
                                    "progress": f"{ci}/{n_cells}",
                                    "elapsed_sec": round(time.time() - t0, 1),
                                    "records": records,
                                    "conditions": conditions,
                                },
                                partial,
                            )
                            for name in ("stream", "eval_x", "stream_f0_pos", "preds_all"):
                                if name in locals():
                                    del locals()[name]
                            gc.collect()
                            tm.mps_free()
    return records, conditions, {"target_locations": loc_rows, "wall_sec": time.time() - t0}


def summarize(records, conditions):
    if not records:
        return {"note": "no records"}
    B = np.array([r["B"] for r in records], float)
    det = an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {"note": "need>=4"}
    if float(np.mean(B < 0)) < 0.10 and float(B.mean()) > 0:
        classification = "helpful-dominated"
    elif float(np.mean(B < 0)) > 0.60:
        classification = "harmful-dominated"
    else:
        classification = "mixed+detectable" if det.get("detectability_verdict") == "detectable" else "mixed+undetectable"
    return {
        "classification": classification,
        "n_records": int(len(records)),
        "n_conditions": int(len(conditions)),
        "mean_B": float(B.mean()),
        "base_rate_harmful_B<0": float(np.mean(B < 0)),
        "min_B": float(B.min()),
        "max_B": float(B.max()),
        "detectability_verdict": det.get("detectability_verdict"),
        "best_single_feature_harm_AUC": det.get("best_single_feature_harm_AUC"),
    }


def build_manifest(args, f0_ckpt, train_meta, eval_meta, records, conditions, meta):
    cfg = {k: getattr(args, k) for k in vars(args)}
    cfg["ckpt_resolved"] = f0_ckpt
    sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return {
        "schema": "kbound_wilds_iwildcam_finder_v0.2",
        "dataset": "wilds-iwildcam",
        "metric": "macro_f1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps": bool(torch.backends.mps.is_available()),
        },
        "config": cfg,
        "config_sha8": sha,
        "f0_checkpoint": f0_ckpt,
        "f0_training": train_meta,
        "f0_quick_eval": eval_meta,
        "num_classes": NUM_CLASSES,
        "evidence_names": tm.EVIDENCE_NAMES,
        "candidates": args.candidates,
        "data": {
            "data_root": args.data_root,
            "split": args.split,
            "target_locations": [
                {"location": int(loc), "n": int(n), "classes": int(c)} for loc, n, c in meta["target_locations"]
            ],
            "wall_sec": round(meta["wall_sec"], 1),
        },
        "baselines": {
            "always_freeze_mean_acc": float(np.mean([r["a0"] for r in records])) if records else None,
            "per_candidate_always_adapt_mean_acc": {
                c: float(np.mean([r["aa"] for r in records if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in records))
            },
            "per_condition_oracle_mean_acc": float(np.mean([c["oracle"] for c in conditions])) if conditions else None,
        },
        "routing_a_single_candidate": rc.aggregate_single_candidate(records),
        "routing_b_multicandidate": rc.aggregate_multicandidate(conditions),
        "routing_c_smooth_drift": rc.aggregate_smoothdrift(conditions),
        "detectability": an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {"note": "need>=4"},
        "kbound_summary": summarize(records, conditions),
        "records": records,
        "conditions": conditions,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Finder scan for K-Bound on WILDS iWildCam")
    p.add_argument("--data-root", default=str(REPO / "experiments/kbound/data/wilds"))
    p.add_argument("--results-root", default=str(REPO / "experiments/kbound/results"))
    p.add_argument("--run-name", default="iwildcam_kbound_finder_v1")
    p.add_argument("--ckpt", default="")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--trainable", choices=["head", "layer4_head", "full"], default="head")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--max-train-batches", type=int, default=120)
    p.add_argument("--train-bs", type=int, default=32)
    p.add_argument("--train-lr", type=float, default=1e-3)
    p.add_argument("--balanced-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--split", default="val", choices=["val", "test", "id_val", "id_test"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-locations", type=int, default=4)
    p.add_argument("--compositions", nargs="+", default=["iid", "single_class"])
    p.add_argument("--batch-regimes", nargs="+", default=["tiny"], dest="batch_regimes")
    p.add_argument("--aggressiveness", nargs="+", default=["mild"])
    p.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    p.add_argument("--n-eval", type=int, default=48, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=2, dest="n_batches")
    p.add_argument("--eval-bs", type=int, default=64, dest="eval_bs")
    p.add_argument("--episodic-steps", type=int, default=3, dest="episodic_steps")
    p.add_argument("--episodic-batch", type=int, default=32, dest="episodic_batch")
    p.add_argument("--tau-star", type=float, default=0.52, dest="tau_star")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument("--sd-L", type=float, default=0.6, dest="sd_L")
    p.add_argument("--steps-override", type=int, default=0, dest="steps_override")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--out", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip cells already in _partial.json (default: on)")
    args = p.parse_args(argv)
    if args.smoke:
        args.run_name = "iwildcam_kbound_smoke"
        args.max_train_batches = min(args.max_train_batches, 8)
        args.train_bs = min(args.train_bs, 16)
        args.max_locations = 2
        args.n_eval = 16
        args.n_batches = 1
        args.compositions = ["iid"]
        args.batch_regimes = ["tiny"]
        args.aggressiveness = ["mild"]
        args.candidates = ["tent_online"]
        args.steps_override = 2
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.results_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        _, _, y, locations = get_iwildcam(args.data_root, args.split, train_tf=False)
        min_count = args.n_eval + max(BATCH_REGIMES[r] for r in args.batch_regimes) * args.n_batches
        loc_rows = select_locations(y, locations, args.max_locations, min_count)
        print("DRY RUN iWildCam finder")
        print(f"split={args.split} locations={loc_rows}")
        print(f"conditions={len(args.seeds) * len(loc_rows) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness)}")
        print(f"candidate_records={len(args.candidates)} per condition")
        return None
    device = tm.pick_device(args.device)
    print(f"[iwildcam] classes={NUM_CLASSES} split={args.split} device={device}", flush=True)
    f0, f0_ckpt, train_meta = train_or_load_f0(args, device, out_dir)
    _, id_sub, _, _ = get_iwildcam(args.data_root, "id_val", train_tf=False)
    _, tgt_sub, _, _ = get_iwildcam(args.data_root, args.split, train_tf=False)
    eval_meta = {
        "id_val": eval_subset(f0, id_sub, device, min(256, args.n_eval * 4), args.train_seed + 7, args.eval_bs),
        args.split: eval_subset(f0, tgt_sub, device, min(256, args.n_eval * 4), args.train_seed + 13, args.eval_bs),
    }
    print(f"[f0 eval] {eval_meta}", flush=True)
    records, conditions, meta = run_scan(args, f0, device, out_dir)
    manifest = build_manifest(args, f0_ckpt, train_meta, eval_meta, records, conditions, meta)
    out = Path(args.out) if args.out else out_dir / f"result_{manifest['config_sha8']}.json"
    with out.open("w") as f:
        json.dump(manifest, f, indent=2)
    summary = manifest["kbound_summary"]
    mb = manifest["routing_b_multicandidate"]
    print("\n" + "=" * 72, flush=True)
    print(f"records={len(records)} conditions={len(conditions)} wall={meta['wall_sec']:.1f}s", flush=True)
    print(
        f"classification={summary.get('classification')} harmful={summary.get('base_rate_harmful_B<0'):.3f} "
        f"mean_B={summary.get('mean_B'):+.4f} B_range=[{summary.get('min_B'):+.4f},{summary.get('max_B'):+.4f}]",
        flush=True,
    )
    print(
        f"detectability={summary.get('detectability_verdict')} "
        f"best_harm_AUC={summary.get('best_single_feature_harm_AUC')}",
        flush=True,
    )
    print(f"multicand={mb}", flush=True)
    print(f"manifest -> {out}", flush=True)
    return str(out)


if __name__ == "__main__":
    main()
