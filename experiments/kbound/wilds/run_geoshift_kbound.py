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
import run_integrity as ri  # noqa: E402
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
        image_dir = Path(ds.data_dir) / "images"
        official = np.asarray(sub.indices, dtype=int)
        missing = [
            str(image_dir / f"rgb_img_{int(index)}.png")
            for index in official
            if not (image_dir / f"rgb_img_{int(index)}.png").is_file()
        ]
        if missing:
            raise RuntimeError(
                f"FMoW split={split!r} is incomplete: {len(missing)}/{len(official)} official "
                f"images are missing (first: {missing[:5]}); restore the archive before running"
            )
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


def _cell_spec(dataset, split, model_seed, checkpoint_sha256, stream_seed,
               grp, comp, regime, aggr):
    return {
        "dataset": dataset,
        "split": split,
        "model_seed": int(model_seed),
        "checkpoint_sha256": checkpoint_sha256,
        "stream_seed": int(stream_seed),
        "location": int(grp),
        "composition": comp,
        "batch_regime": regime,
        "aggressiveness": aggr,
    }


def _cell_id(dataset, split, model_seed, checkpoint_sha256, stream_seed,
             grp, comp, regime, aggr):
    return ri.make_cell_id(**_cell_spec(
        dataset, split, model_seed, checkpoint_sha256, stream_seed,
        grp, comp, regime, aggr))


def _expected_cell_ids(args, grp_rows, checkpoint):
    return [
        _cell_id(args.dataset, args.split, args.train_seed, checkpoint["sha256"],
                 stream_seed, grp, comp, regime, aggr)
        for stream_seed in args.seeds
        for grp, _grp_n, _grp_classes in grp_rows
        for comp in args.compositions
        for regime in args.batch_regimes
        for aggr in args.aggressiveness
    ]


def _checkpoint_identity(path):
    absolute = str(Path(path).expanduser().resolve())
    if not Path(absolute).exists():
        raise FileNotFoundError(f"f0 checkpoint missing: {absolute}")
    return {
        "path": absolute,
        "sha256": ri.file_sha256(absolute),
        "tensor_sha256": checkpoint_tensor_sha256(absolute),
    }


def checkpoint_tensor_sha256(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {path}")
    digest = hashlib.sha256()
    count = 0
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        header = json.dumps({
            "name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape),
        }, sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
        count += 1
    if count == 0:
        raise ValueError(f"checkpoint contains no tensors: {path}")
    return digest.hexdigest()


def _scientific_config(args, checkpoint, grp_rows, num_classes, population):
    config = {k: getattr(args, k) for k in (
        "dataset", "split", "max_groups", "compositions", "batch_regimes",
        "aggressiveness", "candidates", "n_eval", "n_batches", "eval_bs",
        "episodic_steps", "episodic_batch", "tau_star", "kappa", "steps_override",
        "backbone", "trainable", "train_seed", "train_epochs", "max_train_batches",
        "train_bs", "train_lr", "balanced_train", "device", "workers", "smoke",
    )}
    config["data_root"] = str(Path(args.data_root).expanduser().resolve())
    config["stream_seeds"] = [int(seed) for seed in args.seeds]
    config["checkpoint"] = checkpoint
    config["model_identity"] = {
        "model_seed": int(args.train_seed),
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
    }
    config["inference_unit"] = {
        "unit": "stream_seed_on_one_fixed_model_checkpoint",
        "independent_model_ci_eligible": False,
        "legacy_seed_field_semantics": "stream_seed",
    }
    config["population"] = population
    config["target_groups"] = [
        {"location": int(g), "n": int(n), "n_classes": int(c)} for g, n, c in grp_rows
    ]
    config["route_b_contract"] = {
        "objective": DATASET_CFG[args.dataset]["metric"],
        "n_classes": int(num_classes),
        "eligibility": "UNSUPPORTED_MULTICLASS",
    }
    config["route_c_contract"] = rc.route_c_contract(
        DATASET_CFG[args.dataset]["metric"], num_classes
    )
    config["implementation_sha256"] = {
        "runner": ri.file_sha256(__file__),
        "tta_methods": ri.file_sha256(tm.__file__),
        "analysis": ri.file_sha256(an.__file__),
        "routing_aggregates": ri.file_sha256(rc.__file__),
        "fmow_data": ri.file_sha256(fd.__file__),
        "poverty_data": ri.file_sha256(pd.__file__),
    }
    return config


def _unsupported_route_b_summary(conditions, num_classes):
    return {
        "status": "UNSUPPORTED",
        "scorable": False,
        "reason": ("Route B is defined only for binary standard accuracy; this task's objective "
                   f"and/or {num_classes}-class label space is outside that contract."),
        "n_conditions": len(conditions),
        "n_scorable_conditions": 0,
    }


def _commit_cell(records, conditions, failures, cell_records, condition):
    """Commit a complete finite cell as one unit; never expose staged rows early."""
    cell_id = condition.get("cell_id")
    if not isinstance(cell_id, str) or any(r.get("cell_id") != cell_id for r in cell_records):
        raise ri.RunIntegrityError("staged cell records do not share the condition cell_id")
    if any(c.get("cell_id") == cell_id for c in conditions):
        raise ri.RunIntegrityError(f"duplicate completed cell_id: {cell_id}")
    payload = {"records": cell_records, "condition": condition}
    if not ri.finite_tree(payload):
        raise ValueError("cell produced NaN/Infinity and cannot be committed")
    records.extend(cell_records)
    conditions.append(condition)
    ri.clear_failure(failures, cell_id)


def train_or_load_f0(args, num_classes: int, device: torch.device, out_dir: Path):
    ckpt = Path(args.ckpt) if args.ckpt else out_dir / f"f0_{args.backbone}_seed{args.train_seed}.pt"
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed % (2 ** 31))
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
        "train_seed": int(args.train_seed),
        "trainable": args.trainable,
        "balanced_train": bool(args.balanced_train),
        "train_lr": float(args.train_lr),
        "steps": steps,
        "wall_sec": round(time.time() - t0, 1),
    }
    temporary = ckpt.with_suffix(ckpt.suffix + ".tmp")
    try:
        torch.save(meta, temporary)
        os.replace(temporary, ckpt)
    finally:
        if temporary.exists():
            temporary.unlink()
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


def load_positions(sub, positions, device):
    """Load exactly the requested subset-local positions or fail the cell."""
    xs, resolved = [], []
    for p in positions:
        try:
            x, _, _ = sub[int(p)]
        except Exception as exc:
            raise RuntimeError(
                f"unreadable requested geoshift subset position {int(p)}; "
                "sample substitution is forbidden"
            ) from exc
        xs.append(x)
        resolved.append(int(p))
    return torch.stack(xs).to(device), np.asarray(resolved, dtype=int)


def _dataset_sample_ids(sub, positions):
    """Map subset-local positions to stable underlying dataset sample IDs."""
    source = getattr(sub, "indices", None)
    if source is None:
        source = np.arange(len(sub))
    source = np.asarray(source)
    return [int(source[int(position)]) for position in np.asarray(positions, dtype=int)]


def _condition_positions(y, groups, grp, comp, bs, n_eval, n_batches, rng):
    """Return deterministic subset-local stream/evaluation positions."""

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
        raise RuntimeError("evaluation pool leaves no disjoint geoshift adaptation samples")
    n_stream = bs * n_batches
    if comp == "iid":
        if len(remain) < n_stream:
            raise RuntimeError(f"iid stream needs {n_stream} unique samples; only {len(remain)} remain")
        s = rng.choice(remain, n_stream, replace=False)
    elif comp == "imbalanced":
        counts = Counter(y[remain].tolist())
        maj = counts.most_common(1)[0][0]
        mp = remain[y[remain] == maj]
        op = remain[y[remain] != maj]
        n_maj = int(round(0.85 * n_stream))
        if len(mp) and len(op):
            if len(mp) < n_maj or len(op) < n_stream - n_maj:
                raise RuntimeError(
                    f"imbalanced stream needs {n_maj}/{n_stream - n_maj} unique majority/other "
                    f"samples; only {len(mp)}/{len(op)} remain"
                )
            s = np.concatenate([
                rng.choice(mp, n_maj, replace=False),
                rng.choice(op, n_stream - n_maj, replace=False),
            ])
        else:
            raise RuntimeError("imbalanced stream requires both majority and non-majority samples")
    elif comp == "single_class":
        counts = Counter(y[remain].tolist())
        cls = counts.most_common(1)[0][0]
        pool = remain[y[remain] == cls]
        if len(pool) < n_stream:
            raise RuntimeError(
                f"single-class stream needs {n_stream} unique class-{cls} samples; only {len(pool)} remain"
            )
        s = rng.choice(pool, n_stream, replace=False)
    else:
        raise ValueError(comp)
    rng.shuffle(s)
    if len(np.unique(s)) != len(s) or len(np.unique(ev)) != len(ev):
        raise RuntimeError("geoshift condition contains duplicate requested identities")
    if np.intersect1d(s, ev).size:
        raise RuntimeError("geoshift adaptation and evaluation identities overlap")
    return np.asarray(s, dtype=int), np.asarray(ev, dtype=int)


def _condition_sample_provenance(sub, stream_positions, eval_positions, condition_seed):
    stream_positions = np.asarray(stream_positions, dtype=int)
    eval_positions = np.asarray(eval_positions, dtype=int)
    return {
        "condition_seed": int(condition_seed),
        "stream_requested_positions": [int(v) for v in stream_positions],
        "stream_resolved_positions": [int(v) for v in stream_positions],
        "stream_requested_sample_ids": _dataset_sample_ids(sub, stream_positions),
        "stream_resolved_sample_ids": _dataset_sample_ids(sub, stream_positions),
        "eval_requested_positions": [int(v) for v in eval_positions],
        "eval_resolved_positions": [int(v) for v in eval_positions],
        "eval_requested_sample_ids": _dataset_sample_ids(sub, eval_positions),
        "eval_resolved_sample_ids": _dataset_sample_ids(sub, eval_positions),
        "stream_substitution_count": 0,
        "eval_substitution_count": 0,
        "requested_resolved_identity_equal": True,
        "stream_eval_disjoint": True,
        "stream_unique": True,
        "eval_unique": True,
        "stream_eval_overlap_count": 0,
    }


def build_condition(
    sub, y, groups, grp, comp, bs, n_eval, n_batches, rng, device,
    condition_seed=None,
):
    s, ev = _condition_positions(y, groups, grp, comp, bs, n_eval, n_batches, rng)
    stream_x, stream_resolved = load_positions(sub, s, device)
    eval_x, eval_resolved = load_positions(sub, ev, device)
    if not np.array_equal(s, stream_resolved) or not np.array_equal(ev, eval_resolved):
        raise RuntimeError("geoshift requested/resolved identity mismatch")
    eval_y = y[ev].astype(int)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    sample_ids = _condition_sample_provenance(
        sub, stream_resolved, eval_resolved,
        condition_seed if condition_seed is not None else 0,
    )
    return stream, eval_x, eval_y, sample_ids


def _close_score(actual, expected, *, atol=1e-12):
    try:
        return bool(np.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= atol)
    except (TypeError, ValueError):
        return False


def _validate_geoshift_completed_cell(
    condition,
    cell_records,
    *,
    args,
    checkpoint,
    grp_rows,
    sub,
    y,
    groups,
):
    """Validate scores, evidence, provenance, and identities for one saved cell."""

    if not isinstance(condition, dict):
        raise ri.RunIntegrityError("GeoShift resumed condition must be an object")
    try:
        stream_seed = int(condition["stream_seed"])
        location = int(condition["location"])
        comp = str(condition["comp"])
        regime = str(condition["regime"])
        aggr = str(condition["aggr"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ri.RunIntegrityError("GeoShift resumed condition has incomplete axes") from exc
    group_map = {int(group): (int(n), int(n_classes)) for group, n, n_classes in grp_rows}
    if (
        stream_seed not in [int(value) for value in args.seeds]
        or location not in group_map
        or comp not in args.compositions
        or regime not in args.batch_regimes
        or aggr not in args.aggressiveness
    ):
        raise ri.RunIntegrityError("GeoShift resumed condition is outside the configured grid")

    scientific_identity = _cell_spec(
        args.dataset,
        args.split,
        args.train_seed,
        checkpoint["sha256"],
        stream_seed,
        location,
        comp,
        regime,
        aggr,
    )
    cell_id = ri.make_cell_id(**scientific_identity)
    ri.validate_scientific_cell_identity(
        condition.get("cell_id"),
        condition.get("scientific_cell_identity"),
        context="GeoShift resumed condition",
    )
    if (
        condition.get("cell_id") != cell_id
        or condition.get("scientific_cell_identity") != scientific_identity
    ):
        raise ri.RunIntegrityError("GeoShift resumed scientific cell identity mismatch")
    sample_seed = ri.deterministic_seed(cell_id)
    cfg = DATASET_CFG[args.dataset]
    location_n, location_classes = group_map[location]
    expected_identity = {
        "seed": stream_seed,
        "stream_seed": stream_seed,
        "sampling_seed": sample_seed,
        "model_seed": int(args.train_seed),
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
        "inference_unit": "stream_seed_on_one_fixed_model_checkpoint",
        "independent_model_ci_eligible": False,
        "domain": f"{cfg['domain_prefix']}{location}",
        "location": location,
        "location_n": location_n,
        "location_classes": location_classes,
        "split": args.split,
        "comp": comp,
        "regime": regime,
        "aggr": aggr,
    }
    for field, expected in expected_identity.items():
        if condition.get(field) != expected:
            raise ri.RunIntegrityError(f"GeoShift resumed condition has mismatched {field}")

    stream_positions, eval_positions = _condition_positions(
        y,
        groups,
        location,
        comp,
        BATCH_REGIMES[regime],
        args.n_eval,
        args.n_batches,
        np.random.default_rng(sample_seed),
    )
    expected_provenance = _condition_sample_provenance(
        sub, stream_positions, eval_positions, sample_seed,
    )
    provenance = condition.get("sample_provenance")
    if provenance != expected_provenance:
        raise ri.RunIntegrityError(
            "GeoShift resumed sample provenance differs from deterministic selection"
        )
    try:
        eval_y = np.asarray(condition.get("eval_y"), dtype=int)
        frozen = np.asarray(condition.get("preds_frozen"), dtype=int)
    except (TypeError, ValueError) as exc:
        raise ri.RunIntegrityError(
            "GeoShift resumed evaluation labels/predictions are invalid"
        ) from exc
    expected_eval_y = np.asarray(y, dtype=int)[eval_positions]
    if (
        eval_y.ndim != 1
        or eval_y.size == 0
        or not np.array_equal(eval_y, expected_eval_y)
        or frozen.shape != eval_y.shape
    ):
        raise ri.RunIntegrityError(
            "GeoShift resumed evaluation labels/predictions are inconsistent"
        )
    a0 = acc_metric(eval_y, frozen)
    a0_bacc = tm.balanced_acc(frozen, eval_y)
    if not _close_score(condition.get("a0"), a0):
        raise ri.RunIntegrityError("GeoShift resumed frozen score is inconsistent")

    candidate_specs = [
        (method, mode, f"{method}_{mode}") for method, mode in parse_candidates(args.candidates)
    ]
    rows = {record.get("candidate"): record for record in cell_records}
    names = ["freeze_f0", *[candidate for _, _, candidate in candidate_specs]]
    if len(rows) != len(candidate_specs) or condition.get("cand_names") != names:
        raise ri.RunIntegrityError(
            "GeoShift resumed candidate transaction differs from configured candidates"
        )

    aa_all = [a0]
    for method, mode, candidate in candidate_specs:
        record = rows.get(candidate)
        if not isinstance(record, dict):
            raise ri.RunIntegrityError(f"GeoShift resumed cell is missing candidate {candidate}")
        if (
            record.get("cell_id") != cell_id
            or record.get("scientific_cell_identity") != scientific_identity
            or record.get("method") != method
            or record.get("mode") != mode
            or record.get("candidate") != candidate
            or record.get("metric") != cfg["metric"]
            or record.get("sample_provenance") != provenance
        ):
            raise ri.RunIntegrityError(
                f"GeoShift resumed candidate {candidate} has inconsistent identity/provenance"
            )
        for field, expected in expected_identity.items():
            if record.get(field) != expected:
                raise ri.RunIntegrityError(
                    f"GeoShift resumed candidate {candidate} has mismatched {field}"
                )
        ri.validate_evidence_record(
            record,
            tm.EVIDENCE_NAMES,
            expected_tta_protocol=tm.tta_protocol_contract(mode),
            context=f"GeoShift {cell_id}/{candidate}",
        )
        try:
            preds = np.asarray(record.get("preds"), dtype=int)
        except (TypeError, ValueError) as exc:
            raise ri.RunIntegrityError(
                f"GeoShift resumed candidate {candidate} predictions are invalid"
            ) from exc
        if preds.shape != eval_y.shape:
            raise ri.RunIntegrityError(
                f"GeoShift resumed candidate {candidate} prediction length mismatch"
            )
        aa = acc_metric(eval_y, preds)
        aa_bacc = tm.balanced_acc(preds, eval_y)
        if (
            not _close_score(record.get("a0"), a0)
            or not _close_score(record.get("aa"), aa)
            or not _close_score(record.get("B"), aa - a0)
            or not _close_score(record.get("a0_bacc"), a0_bacc)
            or not _close_score(record.get("aa_bacc"), aa_bacc)
            or record.get("regime_label") != an.label_regime(aa - a0)
        ):
            raise ri.RunIntegrityError(
                f"GeoShift resumed candidate {candidate} score semantics are inconsistent"
            )
        aa_all.append(aa)

    stored_scores = condition.get("aa_all")
    if not isinstance(stored_scores, list) or len(stored_scores) != len(aa_all) or any(
        not _close_score(actual, expected)
        for actual, expected in zip(stored_scores, aa_all)
    ):
        raise ri.RunIntegrityError("GeoShift resumed condition scores are inconsistent")
    if (
        not _close_score(condition.get("best_adapt"), max(aa_all[1:]))
        or not _close_score(condition.get("oracle"), max(aa_all))
        or condition.get("true_best") != names[int(np.argmax(aa_all))]
        or condition.get("regime_label") != an.label_regime(max(aa_all[1:]) - a0)
    ):
        raise ri.RunIntegrityError("GeoShift resumed condition summary is inconsistent")


def _validate_resume_semantics(
    records,
    conditions,
    *,
    args,
    checkpoint,
    grp_rows,
    sub,
    y,
    groups,
):
    records_by_cell = {}
    for record in records:
        records_by_cell.setdefault(record.get("cell_id"), []).append(record)
    for condition in conditions:
        _validate_geoshift_completed_cell(
            condition,
            records_by_cell.get(condition.get("cell_id"), []),
            args=args,
            checkpoint=checkpoint,
            grp_rows=grp_rows,
            sub=sub,
            y=y,
            groups=groups,
        )


def _population_identity(dataset_name, dataset, sub, y, groups):
    rows = []
    data_dir = Path(dataset.data_dir).expanduser().resolve()
    for position, official_index in enumerate(np.asarray(sub.indices, dtype=int)):
        if dataset_name == "fmow":
            path = data_dir / "images" / f"rgb_img_{int(official_index)}.png"
            sample_id = f"images/{path.name}"
        else:
            path = data_dir / "images" / f"landsat_poverty_img_{int(official_index)}.npz"
            sample_id = f"images/{path.name}"
        if not path.is_file():
            raise FileNotFoundError(f"geoshift population member is missing: {path}")
        rows.append({
            "official_index": int(official_index),
            "sample_id": sample_id,
            "label": int(y[position]),
            "group": int(groups[position]),
            "bytes": int(path.stat().st_size),
            "content_sha256": ri.file_sha256(path),
        })
    return {
        "sha256": ri.stable_sha256(rows),
        "n": int(len(rows)),
        "identity_fields": [
            "official_index", "sample_id", "label", "group", "bytes", "content_sha256",
        ],
    }


def parse_candidates(names):
    return [(n.split("_", 1)[0], n.split("_", 1)[1]) for n in names]


def run_scan(args, f0, num_classes: int, device: torch.device, out_dir: Path, checkpoint):
    cfg = DATASET_CFG[args.dataset]
    partial = out_dir / "_partial.json"
    dataset, sub, y, groups, _ = load_split(
        args.dataset, args.data_root, args.split, train_tf=False
    )
    min_count = args.n_eval + max(BATCH_REGIMES[r] for r in args.batch_regimes) * args.n_batches
    grp_rows = select_groups(y, groups, args.max_groups, min_count)
    if not grp_rows:
        raise RuntimeError(f"no groups in split={args.split}")
    population = _population_identity(args.dataset, dataset, sub, y, groups)
    expected = _expected_cell_ids(args, grp_rows, checkpoint)
    scientific_config = _scientific_config(args, checkpoint, grp_rows, num_classes, population)
    config_sha256 = ri.stable_sha256(scientific_config)

    def validate_partial_semantics(candidate_records, completed_conditions):
        _validate_resume_semantics(
            candidate_records,
            completed_conditions,
            args=args,
            checkpoint=checkpoint,
            grp_rows=grp_rows,
            sub=sub,
            y=y,
            groups=groups,
        )

    records, conditions, failures = ([], [], [])
    if args.resume:
        records, conditions, failures = ri.load_partial_state(
            partial,
            run_config_sha256=config_sha256,
            expected_cell_ids=expected,
            require_scientific_cell_identity=True,
            semantic_validator=validate_partial_semantics,
        )
    done = {c["cell_id"] for c in conditions}
    print(f"[{args.dataset} groups] " + ", ".join(f"{g}(n={n},cls={c})" for g, n, c in grp_rows), flush=True)
    candidates = parse_candidates(args.candidates)
    n_cells = len(args.seeds) * len(grp_rows) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness)
    t0 = time.time()
    ci = 0
    for stream_seed in args.seeds:
        for grp, grp_n, grp_classes in grp_rows:
            for comp in args.compositions:
                for regime in args.batch_regimes:
                    bs = BATCH_REGIMES[regime]
                    for aggr in args.aggressiveness:
                        ci += 1
                        tag = f"stream{stream_seed}/{cfg['domain_prefix']}{grp}/{comp}/{regime}/{aggr}"
                        cell_id = _cell_id(
                            args.dataset, args.split, args.train_seed, checkpoint["sha256"],
                            stream_seed, grp, comp, regime, aggr)
                        if cell_id in done:
                            print(f"  [{ci}/{n_cells}] {tag} SKIP", flush=True)
                            continue
                        cell_seed = ri.deterministic_seed(cell_id)
                        rng = np.random.default_rng(cell_seed)
                        torch.manual_seed(cell_seed)
                        np.random.seed(cell_seed % (2 ** 31))
                        cell_records = []
                        try:
                            stream, eval_x, eval_y, sample_ids = build_condition(
                                sub, y, groups, grp, comp, bs, args.n_eval, args.n_batches,
                                rng, device, condition_seed=cell_seed,
                            )
                            steps = args.steps_override or AGGR[aggr]["steps"]
                            lr = AGGR[aggr]["lr"]
                            a0_bacc, p0, _ = tm.eval_frozen(
                                f0, eval_x, eval_y, prob_mode="max", bs=args.eval_bs
                            )
                            a0 = acc_metric(eval_y, p0)
                            preds_all = [p0]
                            aa_all = [a0]
                            cand_names = ["freeze_f0"]
                            for method, mode in candidates:
                                aa_bacc, z, upd, preds, _ = tm.run_candidate(
                                    method, mode, f0, stream, eval_x, eval_y, num_classes,
                                    steps, lr, eval_bs=args.episodic_batch, prob_mode="max",
                                    episodic_steps=args.episodic_steps,
                                )
                                aa = acc_metric(eval_y, preds)
                                cand = f"{method}_{mode}"
                                cell_records.append({
                                    "cell_id": cell_id,
                                    "scientific_cell_identity": _cell_spec(
                                        args.dataset, args.split, args.train_seed,
                                        checkpoint["sha256"], stream_seed, grp,
                                        comp, regime, aggr,
                                    ),
                                    "seed": int(stream_seed),
                                    "stream_seed": int(stream_seed),
                                    "sampling_seed": int(cell_seed),
                                    "model_seed": int(args.train_seed),
                                    "checkpoint_sha256": checkpoint["sha256"],
                                    "checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
                                    "inference_unit": "stream_seed_on_one_fixed_model_checkpoint",
                                    "independent_model_ci_eligible": False,
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
                                    "tta_protocol": tm.tta_protocol_contract(mode),
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
                                    "sample_provenance": sample_ids,
                                    "regime_label": an.label_regime(float(aa - a0)),
                                })
                                preds_all.append(preds)
                                aa_all.append(float(aa))
                                cand_names.append(cand)
                                tm.mps_free()
                                gc.collect()
                            route = an.multicandidate_route(
                                np.stack(preds_all, 0), tau_star=args.tau_star, kappa=args.kappa,
                                objective=cfg["metric"], n_classes=num_classes,
                                anchor_above_chance=False,
                            )
                            realized = rc.route_realized(route, aa_all)
                            oracle = float(max(aa_all))
                            best_adapt = float(max(aa_all[1:]))
                            route_c = rc.unsupported_route_c(cfg["metric"], num_classes)
                            condition = {
                                "cell_id": cell_id,
                                "scientific_cell_identity": _cell_spec(
                                    args.dataset, args.split, args.train_seed,
                                    checkpoint["sha256"], stream_seed, grp,
                                    comp, regime, aggr,
                                ),
                                "seed": int(stream_seed),
                                "stream_seed": int(stream_seed),
                                "sampling_seed": int(cell_seed),
                                "model_seed": int(args.train_seed),
                                "checkpoint_sha256": checkpoint["sha256"],
                                "checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
                                "inference_unit": "stream_seed_on_one_fixed_model_checkpoint",
                                "independent_model_ci_eligible": False,
                                "domain": f"{cfg['domain_prefix']}{grp}",
                                "location": int(grp),
                                "location_n": int(grp_n),
                                "location_classes": int(grp_classes),
                                "split": args.split,
                                "comp": comp,
                                "regime": regime,
                                "aggr": aggr,
                                "cand_names": cand_names,
                                "aa_all": [float(v) for v in aa_all],
                                "a0": float(a0),
                                "oracle": oracle,
                                "best_adapt": best_adapt,
                                "true_best": cand_names[int(np.argmax(aa_all))],
                                "route": route,
                                "route_c": route_c,
                                "realized": realized,
                                "route_objective": {
                                    "metric": cfg["metric"], "n_classes": int(num_classes),
                                    "route_b_eligible": False,
                                    "reason": "binary-accuracy identity is inapplicable",
                                },
                                "sample_provenance": sample_ids,
                                "eval_y": [int(v) for v in eval_y],
                                "preds_frozen": [int(v) for v in p0],
                                "regime_label": an.label_regime(best_adapt - a0),
                            }
                            _validate_geoshift_completed_cell(
                                condition,
                                cell_records,
                                args=args,
                                checkpoint=checkpoint,
                                grp_rows=grp_rows,
                                sub=sub,
                                y=y,
                                groups=groups,
                            )
                            _commit_cell(records, conditions, failures, cell_records, condition)
                            done.add(cell_id)
                            print(f"  [{ci}/{n_cells}] {tag} a0={a0:.3f} best={best_adapt:.3f} oracle={oracle:.3f}", flush=True)
                        except Exception as e:
                            print(f"  [{ci}/{n_cells}] {tag} ERROR: {repr(e)[:160]}", flush=True)
                            ri.upsert_failure(failures, {
                                "cell_id": cell_id, "tag": tag, "error_type": type(e).__name__,
                                "error": str(e)[:500], "attempted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            })
                        finally:
                            document = ri.partial_document(
                                run_config_sha256=config_sha256, expected_cell_ids=expected,
                                records=records, conditions=conditions, failures=failures,
                                progress=f"{len(done)}/{n_cells}",
                                require_scientific_cell_identity=True,
                                semantic_validator=validate_partial_semantics,
                            )
                            document["elapsed_sec"] = round(time.time() - t0, 1)
                            ri.atomic_json_dump(document, partial)
                            gc.collect()
                            tm.mps_free()
    ledger = ri.build_ledger(expected, conditions, failures)
    return records, conditions, {
        "target_groups": grp_rows, "wall_sec": time.time() - t0,
        "completion_ledger": ledger, "scientific_config": scientific_config,
        "config_sha256": config_sha256, "population_sha256": population["sha256"],
    }


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
    sha = meta["config_sha256"]
    return {
        "schema": cfg["schema"],
        "dataset": cfg["dataset_tag"],
        "metric": cfg["metric"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"node": platform.node(), "python": platform.python_version(), "torch": torch.__version__},
        "config": meta["scientific_config"],
        "config_sha256": sha,
        "config_sha8": sha[:8],
        "completion_ledger": meta["completion_ledger"],
        "execution_complete": bool(meta["completion_ledger"]["execution_complete"]),
        "publication_eligible": False,
        "publication_eligibility_note": (
            "finder diagnostic on an opened target with stream seeds; completion is not "
            "held-out confirmatory eligibility"
        ),
        "claim_eligibility": {
            "raw_completed_records": bool(meta["completion_ledger"]["execution_complete"]),
            "route_a_single_candidate": False,
            "route_b_multicandidate": False,
            "route_c_smooth_drift": False,
            "independent_model_ci": False,
        },
        "model_identity": meta["scientific_config"]["model_identity"],
        "inference_unit": meta["scientific_config"]["inference_unit"],
        "f0_checkpoint": f0_ckpt,
        "f0_training": train_meta,
        "f0_quick_eval": eval_meta,
        "num_classes": num_classes,
        "evidence_names": tm.EVIDENCE_NAMES,
        "kbound_summary": summarize(records),
        "detectability": an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {},
        "routing_b_multicandidate": _unsupported_route_b_summary(conditions, num_classes),
        "routing_c_smooth_drift": rc.aggregate_smoothdrift(conditions),
        "records": records,
        "conditions": conditions,
        "data": {"target_groups": meta["target_groups"],
                 "population_sha256": meta["population_sha256"],
                 "wall_sec": round(meta["wall_sec"], 1)},
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
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="stream RNG seeds evaluated on one checkpoint; not independent model seeds")
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
    checkpoint = _checkpoint_identity(ckpt)
    records, conditions, meta = run_scan(args, f0, num_classes, device, out_dir, checkpoint)
    ledger = meta["completion_ledger"]
    if not ledger["execution_complete"]:
        print(f"[incomplete] completed={ledger['completed_cells']}/{ledger['expected_cells']} "
              f"failed={ledger['failed_cells']} missing={ledger['missing_cells']}; "
              "partial is NOT publication-eligible", flush=True)
        return None
    manifest = build_manifest(args, num_classes, ckpt, train_meta, eval_meta, records, conditions, meta)
    out = Path(args.out) if args.out else out_dir / f"result_{manifest['config_sha8']}.json"
    ri.atomic_json_dump(manifest, out)
    s = manifest["kbound_summary"]
    print(f"\nDONE {args.dataset} classification={s.get('classification')} harm_AUC={s.get('best_single_feature_harm_AUC')}")
    print(f"manifest -> {out}")
    return str(out)


if __name__ == "__main__":
    main()
