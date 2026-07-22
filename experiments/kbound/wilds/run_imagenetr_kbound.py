"""
run_imagenetr_kbound.py - K-Bound TTA sweep on ImageNet-R (rendition shift).

Same protocol as the Camelyon17 debug run, generalized to multi-class:
  frozen f0   : torchvision ResNet-50 (IMAGENET1K_V2), logits masked to the 200
                ImageNet-R classes  (ImageNet-R is a robustness test set for ImageNet
                classifiers -> NO training; just restrict the 1000 logits to the 200).
  candidates  : {tent,eata,sar} x {online,episodic}   (reused from tta_methods)
  routing     : (a) single-cand KGA, (b) multi-cand tau-route, (c) smooth-drift
                surrogate (max-prob Brier view)        (reused from analysis)
  conditions  : composition x batch_regime x aggressiveness x seed  (no hospital/center
                axis here; the rendition shift is the single target domain).

Reuses the proven aggregation/manifest helpers from run_camelyon17_kbound.  INTEGRITY:
real runs only; honest helpful/harmful/mixed+/-detectable classification from measured B;
tau* calibrated + per-condition tau stored so the operating point is re-pickable.
"""
from __future__ import annotations
import os, sys, json, time, argparse, platform, hashlib
from os.path import join, dirname, abspath
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = dirname(abspath(__file__))
REPO = dirname(dirname(dirname(HERE)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tta_methods as tm            # noqa: E402
import analysis as an              # noqa: E402
import run_camelyon17_kbound as rc  # noqa: E402  (reuse aggregations: AGGR, CANDIDATES, aggregate_*, kbound_summary, route_realized)
import per_condition_serialize as pcs  # noqa: E402  (torch-free per-condition serializer)
import panel_capture as pc          # noqa: E402  (Wave-5: c_ij/n_D capture)
assert list(pcs.EVIDENCE_NAMES) == list(tm.EVIDENCE_NAMES), "EVIDENCE_NAMES drift"

NUM_CLASSES = 200
BATCH_REGIMES = {"large_iid": 200, "small": 16, "tiny": 8}
DIVERSE_BACKBONES = [
    "resnet101",
    "resnet152",
    "resnext101_32x8d",
    "efficientnet_b0",
    "efficientnet_b3",
    "convnext_tiny",
    "convnext_base",
    "vit_b_16",
    "swin_t",
    "swin_b",
]
import torchvision.transforms as T  # noqa: E402
TRANSFORM = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_img(path):
    return TRANSFORM(Image.open(path).convert("RGB"))


class MaskedImageNetModel(nn.Module):
    """ImageNet-1K model whose logits are restricted to the ImageNet-R classes."""
    def __init__(self, base, select_indices):
        super().__init__()
        self.base = base
        self.register_buffer("idx", torch.tensor(select_indices, dtype=torch.long))

    def forward(self, x):
        return self.base(x).index_select(1, self.idx)


def load_select_indices(class_index_path, imagenetr_dir, max_classes=0):
    with open(class_index_path) as fh:
        m = json.load(fh)
    # Support both common canonical ImageNet-1K index schemas:
    #   {"0": ["n01440764", "tench"], ...} and {"n01440764": 0, ...}.
    # The latter is shipped by RobustBench and avoids requiring a duplicate file.
    if m and all(str(k).startswith("n") for k in m):
        wnid2idx = {str(k): int(v) for k, v in m.items()}
    else:
        try:
            wnid2idx = {str(v[0]): int(k) for k, v in m.items()}
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"Unsupported ImageNet class-index schema in {class_index_path}"
            ) from exc
    wnids = sorted([d for d in os.listdir(imagenetr_dir)
                    if d.startswith("n") and os.path.isdir(join(imagenetr_dir, d))])
    if max_classes and max_classes > 0:
        wnids = wnids[:max_classes]
    missing = [w for w in wnids if w not in wnid2idx]
    if missing:
        raise ValueError(
            f"ImageNet class index {class_index_path} is missing "
            f"{len(missing)} dataset WNIDs (first: {missing[:5]})"
        )
    sel = [wnid2idx[w] for w in wnids]
    return wnids, sel


def make_f0(select_indices, device):
    model, _ = make_masked_backbone("resnet50", select_indices, device)
    return model


def make_masked_backbone(backbone, select_indices, device):
    import torchvision.models as M
    if backbone == "resnet50":
        weights = M.ResNet50_Weights.IMAGENET1K_V2
        base = M.resnet50(weights=weights)
    elif backbone == "resnet101":
        weights = M.ResNet101_Weights.DEFAULT
        base = M.resnet101(weights=weights)
    elif backbone == "resnet152":
        weights = M.ResNet152_Weights.DEFAULT
        base = M.resnet152(weights=weights)
    elif backbone == "resnext101_32x8d":
        weights = M.ResNeXt101_32X8D_Weights.DEFAULT
        base = M.resnext101_32x8d(weights=weights)
    elif backbone == "efficientnet_b0":
        weights = M.EfficientNet_B0_Weights.DEFAULT
        base = M.efficientnet_b0(weights=weights)
    elif backbone == "efficientnet_b3":
        weights = M.EfficientNet_B3_Weights.DEFAULT
        base = M.efficientnet_b3(weights=weights)
    elif backbone == "convnext_tiny":
        weights = M.ConvNeXt_Tiny_Weights.DEFAULT
        base = M.convnext_tiny(weights=weights)
    elif backbone == "convnext_base":
        weights = M.ConvNeXt_Base_Weights.DEFAULT
        base = M.convnext_base(weights=weights)
    elif backbone == "vit_b_16":
        weights = M.ViT_B_16_Weights.DEFAULT
        base = M.vit_b_16(weights=weights)
    elif backbone == "swin_t":
        weights = M.Swin_T_Weights.DEFAULT
        base = M.swin_t(weights=weights)
    elif backbone == "swin_b":
        weights = M.Swin_B_Weights.DEFAULT
        base = M.swin_b(weights=weights)
    else:
        raise ValueError(f"unknown backbone {backbone!r}")
    model = MaskedImageNetModel(base, select_indices).to(device)
    model.eval()
    return model, f"torchvision {backbone} {weights.__class__.__name__}.{weights.name}"


def build_index(imagenetr_dir, wnids):
    w2l = {w: i for i, w in enumerate(wnids)}
    items = []
    for w in wnids:
        d = join(imagenetr_dir, w)
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("._"):
                items.append((join(d, f), w2l[w]))
    return items


def build_condition(index, labels, comp, bs, n_eval, rng, device, n_batches=4, tries=15):
    """Class-balanced held-out eval + composition-controlled adaptation stream.
    single_class/imbalanced + tiny batches are the natural collapse-prone cells.
    Stream is label-free at use; eval keeps labels (resample within class on read error)."""
    N = len(index); pos_all = np.arange(N)
    classes = np.unique(labels)
    per = max(1, n_eval // len(classes))
    ev = []
    for c in classes:
        ci = pos_all[labels == c]
        if len(ci):
            ev.append(rng.choice(ci, min(per, len(ci)), replace=False))
    ev = np.concatenate(ev); rng.shuffle(ev)
    remain = np.setdiff1d(pos_all, ev)
    if len(remain) == 0:
        remain = pos_all
    n_stream = max(bs, bs * n_batches)
    if comp == "iid":
        s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    elif comp == "imbalanced":
        maj = int(rng.choice(classes))
        mp = np.intersect1d(pos_all[labels == maj], remain); op = np.setdiff1d(remain, mp)
        if len(mp) and len(op):
            nM = int(n_stream * 0.85)
            s = np.concatenate([rng.choice(mp, nM, replace=len(mp) < nM),
                                rng.choice(op, n_stream - nM, replace=len(op) < (n_stream - nM))])
        else:
            s = rng.choice(remain, n_stream, replace=len(remain) < n_stream)
    else:  # single_class label shift
        maj = int(rng.choice(classes))
        mp = np.intersect1d(pos_all[labels == maj], remain)
        pool = mp if len(mp) else remain
        s = rng.choice(pool, n_stream, replace=len(pool) < n_stream)
    rng.shuffle(s)

    def _load(positions, same_class=None):
        xs = []
        for p in positions:
            ok = False
            cand = pos_all[labels == same_class] if same_class is not None else pos_all
            order = [int(p)] + [int(q) for q in rng.permutation(cand)]
            for q in order[:tries]:
                try:
                    xs.append(load_img(index[int(q)][0])); ok = True; break
                except Exception:
                    continue
            if not ok:
                raise RuntimeError("no readable image")
        return torch.stack(xs).to(device)

    stream_x = _load(s)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    ex, ey = [], []
    for p in ev:
        c = int(labels[p])
        x1 = _load([p], same_class=c)
        ex.append(x1); ey.append(c)
    eval_x = torch.cat(ex, 0).to(device); eval_y = np.array(ey, dtype=int)
    return stream, eval_x, eval_y


def run_diverse_backbones(args, partial_path=None):
    """Protocol D: frozen, independent ImageNet-1K backbones as candidate panel.

    No ImageNet-R labels are used for model/candidate selection. Labels enter only
    after predictions are logged, for B/oracle evaluation and detectability audits.
    """
    t0 = time.time()
    device = tm.pick_device(args.device)
    wnids, sel = load_select_indices(args.class_index, args.imagenetr_dir, getattr(args, "max_classes", 0))
    index = build_index(args.imagenetr_dir, wnids)
    labels = np.array([l for _, l in index])
    num = len(wnids)
    print(f"[imagenet-r:D] classes={num} images={len(index)} device={device} panel=diverse_backbones")
    f0, f0_desc = make_masked_backbone(args.f0_backbone, sel, device)
    print(f"[imagenet-r:D] f0={args.f0_backbone} candidates={','.join(args.candidate_backbones)}")

    records, conditions = [], []
    # ---- OOM-resilience: resume from a prior partial (skip the heavy 10-backbone
    # inference for cells already completed, while still advancing the per-seed RNG in
    # lock-step so the not-yet-done cells are byte-identical to a fresh run). ----------
    done = set()
    if partial_path and getattr(args, "resume", True) and os.path.exists(partial_path):
        try:
            prev = json.load(open(partial_path))
            records = prev.get("records", []); conditions = prev.get("conditions", [])
            done = {(int(c["seed"]), c["comp"], c["regime"], c["aggr"]) for c in conditions}
            if done:
                print(f"[resume] {len(done)} cells loaded from {partial_path}", flush=True)
        except Exception as e:
            print(f"[resume] could not read partial ({e!r}); starting fresh")
            records, conditions, done = [], [], set()
    n_total = (len(args.seeds) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness))
    ci = 0
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        for comp in args.compositions:
            for regime in args.batch_regimes:
                bs = BATCH_REGIMES[regime]
                for aggr in args.aggressiveness:
                    ci += 1; tag = f"s{seed}/{comp}/{regime}/{aggr}"
                    try:
                        stream, eval_x, eval_y = build_condition(
                            index, labels, comp, bs, args.n_eval, rng, device, n_batches=args.n_batches)
                    except Exception as e:
                        print(f"  [{ci}/{n_total}] {tag} SKIP build: {e}"); continue
                    if (int(seed), comp, regime, aggr) in done:
                        # RNG already advanced via build_condition above -> not-yet-done
                        # cells stay byte-identical. Release tensors; skip heavy inference.
                        del stream, eval_x, eval_y; tm.mps_free()
                        print(f"  [{ci}/{n_total}] {tag} SKIP (resume)", flush=True)
                        continue
                    try:
                        a0, p0, p0_pos = tm.eval_frozen(
                            f0, eval_x, eval_y, prob_mode="max", bs=args.frozen_eval_batch)
                        stream_f0_pos = tm._predict_prob(
                            f0, torch.cat(stream, 0), train_mode=False,
                            bs=args.frozen_eval_batch, mode="max")
                        preds_all = [p0]; aa_all = [a0]; cand_names = ["freeze_f0"]
                        best_pa = p0_pos; best_aa_c = float(a0)
                        probe = stream[0]
                        for name in args.candidate_backbones:
                            cand = None
                            try:
                                # load candidate lazily: heavyweight backbones must not
                                # accumulate in unified memory on Apple-silicon MPS.
                                cand, _desc = make_masked_backbone(name, sel, device)
                                cand.eval()
                                aa, preds, pa_pos = tm.eval_frozen(
                                    cand, eval_x, eval_y, prob_mode="max", bs=args.frozen_eval_batch)
                                Z = tm.evidence_vector(f0, cand, probe, num, upd_norm=0.0)
                                B = float(aa - a0)
                                records.append(dict(seed=int(seed), domain="imagenet_r", comp=comp, regime=regime,
                                                    aggr=aggr, method="backbone", mode="frozen", candidate=name,
                                                    a0=float(a0), aa=float(aa), B=B, upd_norm=0.0,
                                                    Z=[float(z) for z in Z], regime_label=an.label_regime(B)))
                                preds_all.append(preds); aa_all.append(float(aa)); cand_names.append(name)
                                if float(aa) > best_aa_c:
                                    best_aa_c = float(aa); best_pa = pa_pos
                            finally:
                                if cand is not None:
                                    cand.to(torch.device("cpu"))
                                    del cand
                                tm.mps_free()
                        preds_mat = np.stack(preds_all, 0)
                        # Wave-5 (Gap B): panel agreements + n_D on the DIVERSE-BACKBONE
                        # panel (the decisive independently-trained candidate set).
                        try:
                            pc.attach_to_last(records, len(cand_names) - 1,
                                              pc.panel_fields(preds_mat))
                        except Exception as _e:
                            print(f"  panel_capture skipped: {_e!r}")
                        route = an.multicandidate_route(preds_mat, tau_star=args.tau_star, kappa=args.kappa)
                        realized = rc.route_realized(route, aa_all)
                        oracle = float(max(aa_all)); best_adapt = float(max(aa_all[1:]))
                        try:
                            route_c = an.smooth_drift_route(p0_pos, best_pa, stream_f0_pos, L=args.sd_L)
                            if route_c.get("implemented") and "bracket" in route_c:
                                trueB = best_adapt - a0
                                route_c["true_B_best"] = float(trueB)
                                route_c["bracket_covers_trueB"] = bool(route_c["bracket"][0] <= trueB <= route_c["bracket"][1])
                        except Exception as e:
                            route_c = {"decision": "ERROR", "implemented": False, "reason": repr(e)}
                        conditions.append(dict(seed=int(seed), domain="imagenet_r", comp=comp, regime=regime,
                                               aggr=aggr, cand_names=cand_names, aa_all=[float(a) for a in aa_all],
                                               a0=float(a0), oracle=oracle, best_adapt=best_adapt,
                                               true_best=cand_names[int(np.argmax(aa_all))], route=route,
                                               route_c=route_c, realized=realized,
                                               regime_label=an.label_regime(best_adapt - a0)))
                        print(f"  [{ci}/{n_total}] {tag} a0={a0:.3f} best_aa={best_adapt:.3f} "
                              f"oracle={oracle:.3f} route={route.get('decision')} "
                              f"tau={route.get('tau', float('nan')):.3f} sd_c={route_c.get('decision')}")
                    except Exception as e:
                        print(f"  [{ci}/{n_total}] {tag} ERROR: {repr(e)[:120]}")
                    if partial_path:
                        try:
                            json.dump({"progress": f"{ci}/{n_total}", "records": records, "conditions": conditions},
                                      open(partial_path, "w"))
                        except Exception:
                            pass
    f0.to(torch.device("cpu")); tm.mps_free()
    return records, conditions, {
        "n_images": len(index), "n_classes": len(wnids), "wall_sec": time.time() - t0,
        "panel": "diverse_backbones", "f0": f0_desc,
        "candidate_backbones": list(args.candidate_backbones),
        "candidate_names": list(args.candidate_backbones),
    }


def run(args, partial_path=None):
    if args.panel == "diverse_backbones":
        return run_diverse_backbones(args, partial_path=partial_path)
    t0 = time.time()
    device = tm.pick_device(args.device)
    wnids, sel = load_select_indices(args.class_index, args.imagenetr_dir, getattr(args, "max_classes", 0))
    index = build_index(args.imagenetr_dir, wnids)
    labels = np.array([l for _, l in index])
    num = len(wnids)
    print(f"[imagenet-r] classes={num} images={len(index)} device={device}")
    f0 = make_f0(sel, device)                              # fixed pretrained f0, reused across seeds
    records, conditions = [], []
    n_total = (len(args.seeds) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness))
    ci = 0
    # WIN_HUNT_v5: online-only candidate pool (the "continual" no-episodic-reset op-point) when
    # --online-only is set; default keeps all six online+episodic candidates (byte-identical).
    _cands = [(m, md) for (m, md) in rc.CANDIDATES
              if (not getattr(args, "online_only", False)) or md == "online"]
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        for comp in args.compositions:
            for regime in args.batch_regimes:
                bs = BATCH_REGIMES[regime]
                for aggr in args.aggressiveness:
                    steps = args.steps_override or rc.AGGR[aggr]["steps"]; lr = args.adapt_lr if getattr(args, "adapt_lr", None) is not None else rc.AGGR[aggr]["lr"]
                    ci += 1; tag = f"s{seed}/{comp}/{regime}/{aggr}"
                    try:
                        stream, eval_x, eval_y = build_condition(
                            index, labels, comp, bs, args.n_eval, rng, device, n_batches=args.n_batches)
                    except Exception as e:
                        print(f"  [{ci}/{n_total}] {tag} SKIP build: {e}"); continue
                    try:
                        a0, p0, p0_pos = tm.eval_frozen(f0, eval_x, eval_y, prob_mode="max")
                        stream_f0_pos = tm._predict_prob(f0, torch.cat(stream, 0), train_mode=False, bs=128, mode="max")
                        preds_all = [p0]; aa_all = [a0]; cand_names = ["freeze_f0"]
                        best_pa = p0_pos; best_aa_c = float(a0)
                        for (method, mode) in _cands:
                            aa, Z, upd, preds, pa_pos = tm.run_candidate(
                                method, mode, f0, stream, eval_x, eval_y, num,
                                steps, lr, eval_bs=args.episodic_batch, prob_mode="max",
                                episodic_steps=args.episodic_steps)
                            B = float(aa - a0)
                            records.append(dict(seed=int(seed), domain="imagenet_r", comp=comp, regime=regime,
                                                aggr=aggr, method=method, mode=mode, candidate=f"{method}_{mode}",
                                                a0=float(a0), aa=float(aa), B=B, upd_norm=float(upd),
                                                Z=[float(z) for z in Z], regime_label=an.label_regime(B)))
                            preds_all.append(preds); aa_all.append(float(aa)); cand_names.append(f"{method}_{mode}")
                            if float(aa) > best_aa_c:
                                best_aa_c = float(aa); best_pa = pa_pos
                            tm.mps_free()
                        preds_mat = np.stack(preds_all, 0)
                        # Wave-5 (Gap B): panel agreements + n_D for the tau' gate
                        try:
                            pc.attach_to_last(records, len(cand_names) - 1,
                                              pc.panel_fields(preds_mat))
                        except Exception as _e:
                            print(f"  panel_capture skipped: {_e!r}")
                        route = an.multicandidate_route(preds_mat, tau_star=args.tau_star, kappa=args.kappa)
                        realized = rc.route_realized(route, aa_all)
                        oracle = float(max(aa_all)); best_adapt = float(max(aa_all[1:]))
                        try:
                            route_c = an.smooth_drift_route(p0_pos, best_pa, stream_f0_pos, L=args.sd_L)
                            if route_c.get("implemented") and "bracket" in route_c:
                                trueB = best_adapt - a0
                                route_c["true_B_best"] = float(trueB)
                                route_c["bracket_covers_trueB"] = bool(route_c["bracket"][0] <= trueB <= route_c["bracket"][1])
                        except Exception as e:
                            route_c = {"decision": "ERROR", "implemented": False, "reason": repr(e)}
                        conditions.append(dict(seed=int(seed), domain="imagenet_r", comp=comp, regime=regime,
                                               aggr=aggr, cand_names=cand_names, aa_all=[float(a) for a in aa_all],
                                               a0=float(a0), oracle=oracle, best_adapt=best_adapt,
                                               true_best=cand_names[int(np.argmax(aa_all))], route=route,
                                               route_c=route_c, realized=realized,
                                               regime_label=an.label_regime(best_adapt - a0)))
                        print(f"  [{ci}/{n_total}] {tag} a0={a0:.3f} best_aa={best_adapt:.3f} "
                              f"oracle={oracle:.3f} route={route.get('decision')} "
                              f"tau={route.get('tau', float('nan')):.3f} sd_c={route_c.get('decision')}")
                    except Exception as e:
                        print(f"  [{ci}/{n_total}] {tag} ERROR: {repr(e)[:120]}")
                    if partial_path:
                        try:
                            json.dump({"progress": f"{ci}/{n_total}", "records": records, "conditions": conditions},
                                      open(partial_path, "w"))
                        except Exception:
                            pass
    return records, conditions, {"n_images": len(index), "n_classes": len(wnids), "wall_sec": time.time() - t0}


def build_manifest(args, records, conditions, meta):
    cfg = {k: getattr(args, k) for k in (
        "imagenetr_dir", "panel", "f0_backbone", "candidate_backbones",
        "seeds", "compositions", "batch_regimes", "aggressiveness",
        "n_eval", "n_batches", "tau_star", "kappa", "sd_L", "delta", "device",
        "steps_override", "max_classes", "episodic_steps", "episodic_batch",
        "frozen_eval_batch", "smoke",
        "adapt_lr", "online_only")}   # WIN_HUNT_v5 operating-point overrides enter the config hash
    sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    candidate_names = meta.get("candidate_names", [f"{m}_{md}" for (m, md) in rc.CANDIDATES])
    return {
        "schema": "kbound_imagenetr_v0.5", "dataset": "imagenet-r",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"platform": platform.platform(), "torch": torch.__version__,
                 "mps": bool(torch.backends.mps.is_available())},
        "config": cfg, "config_sha8": sha,
        "f0": meta.get("f0", "torchvision resnet50 IMAGENET1K_V2; 1000 logits masked to 200 ImageNet-R classes (frozen)"),
        "num_classes": meta["n_classes"], "candidates": candidate_names,
        "panel": getattr(args, "panel", "shared_tta"),
        "multiclass_caveat": ("multi-candidate tau-route (b) uses prediction agreement on a 200-class "
                              "label space; the binary-Y advantage recovery (Thm 1A) is heuristic here, "
                              "so per-condition tau is stored for re-pick. (a) KGA and (c) smooth-drift "
                              "(max-prob Brier) generalize directly."),
        "data": {"n_images": meta["n_images"], "n_classes": meta["n_classes"], "wall_sec": round(meta["wall_sec"], 1)},
        "baselines": {
            "always_freeze_mean_acc": float(np.mean([r["a0"] for r in records])) if records else None,
            "per_candidate_always_adapt_mean_acc": {
                c: float(np.mean([r["aa"] for r in records if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in records))},
            "per_condition_oracle_mean_acc": float(np.mean([c["oracle"] for c in conditions])) if conditions else None},
        "routing_a_single_candidate": rc.aggregate_single_candidate(records),
        "routing_b_multicandidate": rc.aggregate_multicandidate(conditions),
        "routing_c_smooth_drift": rc.aggregate_smoothdrift(conditions),
        "detectability": an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {"note": "need>=4"},
        "kbound_summary": rc.kbound_summary(records, conditions, delta=args.delta),
        "tau_distribution": sorted([float(c["route"]["tau"]) for c in conditions if c["route"].get("tau") is not None]),
        "records": records, "conditions": conditions,
    }


def parse_args(argv=None):
    DATA = join(REPO, "experiments/kbound/data")
    p = argparse.ArgumentParser(description="K-Bound TTA sweep on ImageNet-R")
    p.add_argument("--imagenetr-dir", default=join(DATA, "imagenet-r"), dest="imagenetr_dir")
    p.add_argument("--class-index", default=join(DATA, "imagenet_class_index.json"), dest="class_index")
    p.add_argument("--panel", default="shared_tta", choices=["shared_tta", "diverse_backbones"],
                   help="shared_tta reproduces the original six TTA candidates on one f0; "
                        "diverse_backbones runs Protocol D independent frozen backbones.")
    p.add_argument("--f0-backbone", default="resnet50", dest="f0_backbone",
                   choices=["resnet50"], help="Protocol D frozen anchor backbone")
    p.add_argument("--candidate-backbones", nargs="+", default=list(DIVERSE_BACKBONES),
                   choices=list(DIVERSE_BACKBONES),
                   dest="candidate_backbones")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--compositions", nargs="+", default=["iid", "imbalanced", "single_class"])
    p.add_argument("--batch-regimes", nargs="+", default=["large_iid", "small", "tiny"], dest="batch_regimes")
    p.add_argument("--aggressiveness", nargs="+", default=["mild", "aggressive"])
    p.add_argument("--n-eval", type=int, default=1000, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=4, dest="n_batches")
    p.add_argument("--frozen-eval-batch", type=int, default=32, dest="frozen_eval_batch",
                   help="prediction batch size for frozen ImageNet-R backbones; small default keeps Protocol D MPS-tractable")
    p.add_argument("--max-classes", type=int, default=0, dest="max_classes",
                   help="restrict to first N ImageNet-R classes (0=all 200)")
    p.add_argument("--episodic-steps", type=int, default=5, dest="episodic_steps",
                   help="adaptation steps per test-batch in episodic mode (MPS-tractable; "
                        "faithful to episodic TTA)")
    p.add_argument("--episodic-batch", type=int, default=64, dest="episodic_batch",
                   help="fixed eval-batch size for episodic resets")
    p.add_argument("--tau-star", type=float, default=0.52, dest="tau_star")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument("--sd-L", type=float, default=0.6, dest="sd_L")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--steps-override", type=int, default=0, dest="steps_override")
    p.add_argument("--run-name", default="imagenetr_kbound_debug_mps", dest="run_name")
    p.add_argument("--results-root", default="", dest="results_root",
                   help="dir to write results under (default: repo/experiments/kbound/results; "
                        "set to an INTERNAL path to avoid slow T9 I/O)")
    p.add_argument("--out", default="")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved Protocol D/shared-panel grid and exit before loading models")
    p.add_argument("--serialize-per-condition", action=argparse.BooleanOptionalAction,
                   default=True, dest="serialize_per_condition",
                   help="also write per_condition_imagenet-r_<method>_seed<S>.json files "
                        "(stress_grid_multiseed schema; default: on)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="OOM-resilience: skip cells already in _partial.json while keeping "
                        "the per-seed RNG in lock-step (default: on)")
    p.add_argument("--smoke", action="store_true")
    # ---- WIN_HUNT_v5 aggressive-regime wave operating-point overrides (opt-in; shared_tta panel) ----
    p.add_argument("--adapt-lr", type=float, default=None, dest="adapt_lr",
                   help="WIN_HUNT_v5: absolute adapter LR override for tent/eata/sar (rc.AGGR cell "
                        "lr ignored when set). DEFAULT None = per-cell lr (byte-identical). v5 sets "
                        "0.004 (= 4x the 1e-3 shared-baseline lr). No effect on diverse_backbones.")
    p.add_argument("--online-only", action="store_true", dest="online_only",
                   help="WIN_HUNT_v5: restrict the shared_tta candidate pool to online (no-episodic-"
                        "reset) adapters -- the 'continual' operating point. DEFAULT off (byte-identical).")
    a = p.parse_args(argv)
    if a.smoke:
        a.compositions = ["iid", "single_class"]; a.batch_regimes = ["tiny"]
        a.aggressiveness = ["mild"]; a.seeds = [0, 1]; a.n_eval = 40; a.n_batches = 2
        a.steps_override = 4; a.max_classes = 20
        if a.device == "auto":
            a.device = "cpu"
    return a


def main(argv=None):
    a = parse_args(argv)
    root = a.results_root or join(REPO, "experiments/kbound/results")
    out_dir = join(root, "imagenetr_kbound_smoke" if a.smoke else a.run_name)
    os.makedirs(out_dir, exist_ok=True)
    if a.dry_run:
        n_conditions = len(a.seeds) * len(a.compositions) * len(a.batch_regimes) * len(a.aggressiveness)
        _n_shared = len([1 for (m, md) in rc.CANDIDATES if (not getattr(a, "online_only", False)) or md == "online"])
        n_records = n_conditions * (len(a.candidate_backbones) if a.panel == "diverse_backbones" else _n_shared)
        print("DRY RUN ImageNet-R")
        print(f"  --panel {a.panel}")
        print(f"  run_name={a.run_name}")
        print(f"  imagenetr_dir={a.imagenetr_dir}")
        print(f"  seeds={' '.join(map(str, a.seeds))}")
        print(f"  grid={len(a.compositions)} compositions x {len(a.batch_regimes)} batch regimes x {len(a.aggressiveness)} aggressiveness")
        print(f"  conditions={n_conditions}")
        print(f"  records={n_records}")
        print(f"  candidate_backbones={','.join(a.candidate_backbones)}")
        print(f"  frozen_eval_batch={a.frozen_eval_batch}")
        print("  load_models=False")
        return ""
    partial = join(out_dir, "_partial.json")
    records, conditions, meta = run(a, partial_path=partial)
    man = build_manifest(a, records, conditions, meta)
    out = a.out or join(out_dir, f"result_{man['config_sha8']}.json")
    json.dump(man, open(out, "w"), indent=2)
    # ---- per-condition serialization (stress_grid_multiseed schema) ----------
    if getattr(a, "serialize_per_condition", True) and records:
        # diverse_backbones panel: each frozen backbone is the "method" axis
        # (records carry method="backbone", candidate=<backbone>); shared_tta panel
        # uses method in {tent,eata,sar}.
        m_field = "candidate" if a.panel == "diverse_backbones" else "method"
        methods = sorted({r[m_field] for r in records})
        seeds = [int(s) for s in a.seeds]
        ser = pcs.serialize_run(records, dataset="imagenet-r", out_dir=out_dir,
                                seeds=seeds, methods=methods, method_field=m_field)
        print(f"[serialize] wrote {len(ser['written'])} per-condition files "
              f"(panel={a.panel}, methods={methods}, seeds={seeds}, "
              f"kga_backend={ser['kga_backend']}) -> {out_dir}")
        if a.panel == "diverse_backbones" and conditions:
            pan = pcs.serialize_panel_run(
                records, conditions, dataset="imagenet-r", out_dir=out_dir,
                seeds=seeds, candidate_order=list(a.candidate_backbones))
            print(f"[serialize] wrote {len(pan['written'])} per-panel files "
                  f"({pan['n_conditions']} conditions) -> {out_dir}")
    ks = man["kbound_summary"]; mb = man["routing_b_multicandidate"]; rcd = man["routing_c_smooth_drift"]
    td = man["tau_distribution"]
    print("\n" + "=" * 70)
    print(f"records={len(records)} conditions={len(conditions)} wall={meta['wall_sec']:.1f}s")
    print(f"classification        : {ks['classification']}  base_harmful={ks['base_rate_harmful_B<0']:.3f} mean_B={ks['mean_B']:+.4f}")
    print(f"detectability         : {ks['detectability_verdict']} (best harm-AUC={ks['best_single_feature_harm_AUC']})")
    print(f"multicand route       : mean_tau={mb.get('mean_tau')} abstain={mb.get('abstention_rate')} breakdown={mb.get('routing_breakdown')}")
    print(f"tau range             : [{td[0]:.3f}..{td[-1]:.3f}] (tau*={a.tau_star})" if td else "tau range: n/a")
    print(f"smooth-drift (c)      : impl={rcd.get('implemented')} decisions={rcd.get('decision_counts')} bracket_cov={rcd.get('bracket_coverage_trueB')}")
    print(f"\nmanifest -> {out}")
    return out


if __name__ == "__main__":
    main()
