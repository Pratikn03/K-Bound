"""
run_rxrx1_kbound.py - K-Bound TTA sweep on WILDS RxRx1 (experimental-batch shift, 1139-class).

Models the ImageNet-R multi-class runner (run_imagenetr_kbound.py) on the REAL WILDS RxRx1 setup:
  frozen f0   : torchvision resnet50(num_classes=1139) loaded from the OFFICIAL WILDS ERM
                checkpoint (rxrx1_seed:0_epoch:best_model.pth; 'model.'-prefixed state_dict
                -> 0 missing/0 unexpected). NO training. In-dist acc ~35.9% (verified ~33% on n=256).
  data        : WILDS RxRx1 v1.0, OOD 'test' split (14 unseen experiments) = the single target
                domain. WILDS 'rxrx1' eval transform (ToTensor + per-image standardize) - REQUIRED
                to reproduce the checkpoint's accuracy.
  candidates  : {tent,eata,sar} x {online,episodic}        (reused VERBATIM from tta_methods)
  routing     : (a) single-cand KGA. Route B is UNSUPPORTED because its agreement
                identity is binary-only. Route C is UNSUPPORTED because its binary
                Brier-score bracket does not identify balanced-accuracy benefit.
  conditions  : composition x batch_regime x aggressiveness x stream seed (single fixed model).

SURVIVAL HARNESS (16 GB MPS, has OOM-killed heavy sweeps): per-condition memory hygiene
(del + gc.collect + torch.mps.empty_cache + torch.mps.synchronize), resume-from-_partial
(skip completed cells, APPEND, atomic flush, never truncate), per-cell deterministic rng so
resume is bit-identical, plus an external auto-restart supervisor (supervise_rxrx1.sh).

INTEGRITY: real runs only; honest helpful/harmful/mixed classification from measured B; invalid,
failed, or unsupported routes are never scored as freeze decisions.
"""
from __future__ import annotations
import os, sys, gc, re, time, argparse, platform, hashlib, json
import numpy as np
import torch
import torchvision.models as M
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tta_methods as tm            # noqa: E402
import analysis as an              # noqa: E402
import run_camelyon17_kbound as rc  # noqa: E402  (AGGR, CANDIDATES, aggregate_*, kbound_summary, route_realized)
import run_integrity as ri          # noqa: E402

NUM_CLASSES = 1139
BATCH_REGIMES = {"large_iid": 200, "small": 16, "tiny": 8}
DOMAIN = "rxrx1"
COMPLETION_RECEIPT_SCHEMA = "kbound_rxrx1_completion_receipt_v2"


def _standardize(x):
    mean = x.mean(dim=(1, 2)); std = x.std(dim=(1, 2)); std[std == 0.] = 1.
    return TF.normalize(x, mean, std)


def rxrx1_eval_transform():
    """WILDS 'rxrx1' eval transform: ToTensor then per-image per-channel standardize."""
    return transforms.Compose([transforms.ToTensor(), transforms.Lambda(_standardize)])


def ensure_rxrx1_patch():
    """wilds 2.0.0 RxRx1Dataset: the split_array in-place write needs a WRITABLE numpy array
    (numpy>=1.24 returns read-only .values -> 'assignment destination is read-only'). Ensure the
    installed source uses .values.copy() on that line. Idempotent (no-op once patched)."""
    try:
        import wilds.datasets.rxrx1_dataset as rd
        p = rd.__file__
        with open(p) as f:
            src = f.read()
        new = re.sub(r"(self\._split_dict\.get\)\.values)(?!\.copy)", r"\1.copy()", src)
        if new != src:
            with open(p, "w") as f:
                f.write(new)
            return "patched"
        return "already_patched"
    except Exception as e:
        return f"skip:{e!r}"


def load_f0(ckpt, device, num_classes=NUM_CLASSES):
    """Frozen f0 = torchvision resnet50(num_classes) <- official WILDS ERM checkpoint.
    Strips the 'model.' prefix; asserts 0 missing / 0 unexpected keys. NO training."""
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = obj["algorithm"] if isinstance(obj, dict) and "algorithm" in obj else obj
    new = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
    model = M.resnet50(weights=None, num_classes=num_classes)
    res = model.load_state_dict(new, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, \
        f"checkpoint key mismatch: missing={res.missing_keys[:4]} unexpected={res.unexpected_keys[:4]}"
    model.to(device).eval()
    return model


def load_rxrx1(data_root, split, device):
    """Build the exact official target subset; partial archives fail closed."""
    from wilds import get_dataset
    ds = get_dataset(dataset="rxrx1", download=False, root_dir=data_root)
    sub = ds.get_subset(split, transform=rxrx1_eval_transform())
    idx = np.asarray(sub.indices)
    data_dir = str(ds.data_dir)
    inp = ds._input_array
    keep = np.fromiter((os.path.exists(os.path.join(data_dir, str(inp[i]))) for i in idx),
                       dtype=bool, count=len(idx))
    n_total = int(len(idx)); n_present = int(keep.sum())
    if n_present != n_total:
        missing = [str(inp[i]) for i in idx[~keep][:5]]
        raise RuntimeError(
            f"RxRx1 split={split!r} is incomplete: {n_total - n_present}/{n_total} official "
            f"images are missing (first: {missing}); restore the archive before running"
        )
    y = ds.y_array[idx].numpy().astype(int)
    return ds, sub, y, n_present, n_total


def _load_x(sub, pos):
    x, _, _ = sub[int(pos)]
    return x


def _condition_positions(y, comp, bs, n_eval, rng, n_batches=4):
    """Return the deterministic, unique, disjoint stream/evaluation positions."""

    y = np.asarray(y, dtype=int)
    pos_all = np.arange(len(y))
    classes = np.unique(y)
    per = max(1, n_eval // max(1, len(classes)))
    ev = []
    for c in classes:
        ci = pos_all[y == c]
        if len(ci):
            ev.append(rng.choice(ci, min(per, len(ci)), replace=False))
    ev = np.concatenate(ev)
    if len(ev) > n_eval:
        ev = rng.choice(ev, n_eval, replace=False)
    rng.shuffle(ev)
    remain = np.setdiff1d(pos_all, ev)
    if len(remain) == 0:
        raise RuntimeError("evaluation pool leaves no disjoint RxRx1 adaptation samples")
    n_stream = max(bs, bs * n_batches)
    if comp == "iid":
        if len(remain) < n_stream:
            raise RuntimeError(f"iid stream needs {n_stream} unique samples; only {len(remain)} remain")
        stream = rng.choice(remain, n_stream, replace=False)
    elif comp == "imbalanced":
        maj = int(rng.choice(classes))
        majority = np.intersect1d(pos_all[y == maj], remain)
        other = np.setdiff1d(remain, majority)
        n_majority = int(n_stream * 0.85)
        if not len(majority) or not len(other):
            raise RuntimeError("imbalanced stream requires both majority and non-majority samples")
        if len(majority) < n_majority or len(other) < n_stream - n_majority:
            raise RuntimeError(
                f"imbalanced stream needs {n_majority}/{n_stream - n_majority} unique "
                f"majority/other samples; only {len(majority)}/{len(other)} remain"
            )
        stream = np.concatenate([
            rng.choice(majority, n_majority, replace=False),
            rng.choice(other, n_stream - n_majority, replace=False),
        ])
    elif comp == "single_class":
        maj = int(rng.choice(classes))
        majority = np.intersect1d(pos_all[y == maj], remain)
        if len(majority) < n_stream:
            raise RuntimeError(
                f"single-class stream needs {n_stream} unique class-{maj} samples; "
                f"only {len(majority)} remain"
            )
        stream = rng.choice(majority, n_stream, replace=False)
    else:
        raise ValueError(f"unknown composition: {comp}")
    rng.shuffle(stream)
    if len(np.unique(stream)) != len(stream) or len(np.unique(ev)) != len(ev):
        raise RuntimeError("RxRx1 condition contains duplicate requested identities")
    if np.intersect1d(stream, ev).size:
        raise RuntimeError("RxRx1 adaptation and evaluation identities overlap")
    return np.asarray(stream, dtype=int), np.asarray(ev, dtype=int)


def build_condition(sub, y, comp, bs, n_eval, rng, device, n_batches=4, tries=None,
                    return_ids=False):
    """Class-balanced (CAPPED at n_eval) held-out eval + composition-controlled adaptation stream.
    RxRx1 has 1139 classes; the per-class eval pool is capped at n_eval so eval_x stays MPS-tractable
    while the reported score remains explicitly balanced accuracy. No ordinary-accuracy parity is
    assumed. single_class / imbalanced + tiny batches are the natural collapse-prone cells; harm
    arises from the DATA, never tuned HPs."""
    if len(sub) != len(y):
        raise RuntimeError("RxRx1 subset and label population lengths differ")
    s, ev = _condition_positions(y, comp, bs, n_eval, rng, n_batches=n_batches)

    def _load_exact(positions):
        xs = []
        for position in positions:
            try:
                xs.append(_load_x(sub, int(position)))
            except Exception as exc:
                raise RuntimeError(
                    f"unreadable requested RxRx1 subset position {int(position)}; "
                    "sample substitution is forbidden"
                ) from exc
        return torch.stack(xs).to(device)

    stream_x = _load_exact(s)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    eval_x = _load_exact(ev)
    eval_y = y[ev].astype(int)
    if return_ids:
        return stream, eval_x, eval_y, {
            "stream_requested_positions": np.asarray(s, dtype=int),
            "stream_resolved_positions": np.asarray(s, dtype=int),
            "eval_requested_positions": np.asarray(ev, dtype=int),
            "eval_resolved_positions": np.asarray(ev, dtype=int),
        }
    return stream, eval_x, eval_y


# ----------------------------- survival helpers ------------------------------
def deep_free(device):
    """Per-condition memory hygiene for MPS: collect python garbage, then empty + sync MPS."""
    gc.collect()
    try:
        if getattr(device, "type", None) == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache(); torch.mps.synchronize()
    except Exception:
        pass


def _tag(seed, comp, regime, aggr):
    return f"stream{seed}/{comp}/{regime}/{aggr}"


def _cell_spec(run_config_sha256, split, model_seed, checkpoint_sha256,
               checkpoint_tensor_sha256, stream_seed, comp, regime, aggr):
    return {
        "dataset": "wilds-rxrx1",
        "run_config_sha256": str(run_config_sha256),
        "split": str(split),
        "model_seed": int(model_seed),
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint_tensor_sha256": str(checkpoint_tensor_sha256),
        "stream_seed": int(stream_seed),
        "domain": DOMAIN,
        "composition": comp,
        "batch_regime": regime,
        "aggressiveness": aggr,
    }


def _cell_id(run_config_sha256, split, model_seed, checkpoint_sha256,
             checkpoint_tensor_sha256, stream_seed, comp, regime, aggr):
    return ri.make_cell_id(**_cell_spec(
        run_config_sha256, split, model_seed, checkpoint_sha256,
        checkpoint_tensor_sha256, stream_seed, comp, regime, aggr))


def _expected_cell_ids(args, checkpoint, run_config_sha256):
    return [
        _cell_id(
            run_config_sha256, args.split, args.model_seed, checkpoint["sha256"],
            checkpoint["tensor_sha256"], stream_seed, comp, regime, aggr,
        )
        for stream_seed in args.seeds
        for comp in args.compositions
        for regime in args.batch_regimes
        for aggr in args.aggressiveness
    ]


def _checkpoint_identity(args):
    path = os.path.abspath(os.path.expanduser(args.ckpt))
    if not os.path.exists(path):
        raise FileNotFoundError(f"RxRx1 checkpoint missing: {path}")
    return {
        "path": path,
        "sha256": ri.file_sha256(path),
        "tensor_sha256": checkpoint_tensor_sha256(path),
        "model_seed": int(args.model_seed),
    }


def checkpoint_tensor_sha256(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["algorithm"] if isinstance(obj, dict) and "algorithm" in obj else obj
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


def _scientific_config(args, checkpoint, population):
    cfg = {k: getattr(args, k) for k in (
        "split", "model_seed", "compositions", "batch_regimes", "aggressiveness",
        "n_eval", "n_batches", "tau_star", "kappa", "sd_L", "delta", "device",
        "steps_override", "episodic_steps", "episodic_batch", "smoke",
        "adapt_lr", "online_only",
    )}
    cfg["data_root"] = os.path.abspath(os.path.expanduser(args.data_root))
    cfg["stream_seeds"] = [int(seed) for seed in args.seeds]
    cfg["checkpoint"] = checkpoint
    cfg["model_identity"] = {
        "model_seed": int(args.model_seed),
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
    }
    cfg["inference_unit"] = {
        "unit": "stream_seed_on_one_fixed_model_checkpoint",
        "independent_model_ci_eligible": False,
        "legacy_seed_field_semantics": "stream_seed",
    }
    cfg["population"] = population
    cfg["candidate_set"] = [
        f"{method}_{mode}" for method, mode in rc.CANDIDATES
        if (not getattr(args, "online_only", False)) or mode == "online"
    ]
    cfg["route_b_contract"] = {
        "objective": "balanced_accuracy",
        "n_classes": NUM_CLASSES,
        "eligibility": "UNSUPPORTED_MULTICLASS",
    }
    cfg["metric"] = "balanced_accuracy"
    cfg["route_c_contract"] = rc.route_c_contract("balanced_accuracy", NUM_CLASSES)
    cfg["implementation_sha256"] = {
        "runner": ri.file_sha256(__file__),
        "tta_methods": ri.file_sha256(tm.__file__),
        "analysis": ri.file_sha256(an.__file__),
        "routing_aggregates": ri.file_sha256(rc.__file__),
    }
    return cfg


def _rxrx1_population_identity(dataset, sub, y):
    data_dir = os.path.abspath(str(dataset.data_dir))
    rows = []
    for position, official_index in enumerate(np.asarray(sub.indices, dtype=int)):
        raw_id = str(dataset._input_array[int(official_index)])
        path = raw_id if os.path.isabs(raw_id) else os.path.join(data_dir, raw_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"RxRx1 population member is missing: {path}")
        rows.append({
            "official_index": int(official_index),
            "official_input_id": raw_id.replace("\\", "/"),
            "label": int(y[position]),
            "bytes": int(os.path.getsize(path)),
            "content_sha256": ri.file_sha256(path),
        })
    return {
        "sha256": ri.stable_sha256(rows),
        "n_present": int(len(rows)),
        "n_total": int(len(rows)),
        "identity_fields": [
            "official_index", "official_input_id", "label", "bytes", "content_sha256",
        ],
    }


def _condition_sample_provenance(sub, sample_ids, condition_seed):
    stream_requested = np.asarray(sample_ids["stream_requested_positions"], dtype=int)
    stream_resolved = np.asarray(sample_ids["stream_resolved_positions"], dtype=int)
    eval_requested = np.asarray(sample_ids["eval_requested_positions"], dtype=int)
    eval_resolved = np.asarray(sample_ids["eval_resolved_positions"], dtype=int)
    equal = (
        np.array_equal(stream_requested, stream_resolved)
        and np.array_equal(eval_requested, eval_resolved)
    )
    overlap = np.intersect1d(stream_resolved, eval_resolved)
    if not equal or overlap.size:
        raise RuntimeError("RxRx1 requested/resolved identity or disjointness invariant failed")
    official = np.asarray(sub.indices, dtype=int)
    return {
        "condition_seed": int(condition_seed),
        "sample_id_scheme": "WILDS official subset positions and official dataset row indices",
        "stream_n": int(len(stream_resolved)),
        "eval_n": int(len(eval_resolved)),
        "ordered_stream_requested_positions_sha256": ri.stable_sha256(stream_requested.tolist()),
        "ordered_stream_resolved_positions_sha256": ri.stable_sha256(stream_resolved.tolist()),
        "ordered_eval_requested_positions_sha256": ri.stable_sha256(eval_requested.tolist()),
        "ordered_eval_resolved_positions_sha256": ri.stable_sha256(eval_resolved.tolist()),
        "ordered_stream_official_ids_sha256": ri.stable_sha256(official[stream_resolved].tolist()),
        "ordered_eval_official_ids_sha256": ri.stable_sha256(official[eval_resolved].tolist()),
        "requested_resolved_identity_equal": bool(equal),
        "stream_eval_disjoint": bool(overlap.size == 0),
        "stream_unique": bool(len(np.unique(stream_resolved)) == len(stream_resolved)),
        "eval_unique": bool(len(np.unique(eval_resolved)) == len(eval_resolved)),
        "stream_eval_overlap_count": int(overlap.size),
    }


def _validate_rxrx1_state(
    records,
    conditions,
    *,
    args,
    checkpoint,
    run_config_sha256,
    sub,
    y,
):
    """Recompute every deterministic scientific invariant before reuse or write."""

    expected_candidates = [
        f"{method}_{mode}" for method, mode in rc.CANDIDATES
        if (not getattr(args, "online_only", False)) or mode == "online"
    ]
    by_cell = {}
    for record in records:
        if not isinstance(record, dict):
            raise ri.RunIntegrityError("RxRx1 candidate records must be objects")
        by_cell.setdefault(record.get("cell_id"), []).append(record)

    for condition in conditions:
        if not isinstance(condition, dict):
            raise ri.RunIntegrityError("RxRx1 conditions must be objects")
        cell_id = condition.get("cell_id")
        identity = condition.get("scientific_cell_identity")
        ri.validate_scientific_cell_identity(
            cell_id, identity, context="RxRx1 completed condition"
        )
        required_identity = {
            "run_config_sha256": str(run_config_sha256),
            "split": str(args.split),
            "model_seed": int(args.model_seed),
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
        }
        if any(identity.get(field) != value for field, value in required_identity.items()):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} is not bound to this run contract")
        expected_identity = _cell_spec(
            run_config_sha256,
            args.split,
            args.model_seed,
            checkpoint["sha256"],
            checkpoint["tensor_sha256"],
            identity.get("stream_seed"),
            identity.get("composition"),
            identity.get("batch_regime"),
            identity.get("aggressiveness"),
        )
        if identity != expected_identity or ri.make_cell_id(**expected_identity) != cell_id:
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} scientific identity mismatch")

        archived_identity = {
            "seed": int(expected_identity["stream_seed"]),
            "stream_seed": int(expected_identity["stream_seed"]),
            "model_seed": int(expected_identity["model_seed"]),
            "checkpoint_sha256": expected_identity["checkpoint_sha256"],
            "checkpoint_tensor_sha256": expected_identity["checkpoint_tensor_sha256"],
            "domain": DOMAIN,
            "comp": expected_identity["composition"],
            "regime": expected_identity["batch_regime"],
            "aggr": expected_identity["aggressiveness"],
            "inference_unit": "stream_seed_on_one_fixed_model_checkpoint",
            "independent_model_ci_eligible": False,
        }
        if any(condition.get(field) != value for field, value in archived_identity.items()):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} archived identity fields mismatch")

        stream_seed = int(expected_identity["stream_seed"])
        comp = expected_identity["composition"]
        regime = expected_identity["batch_regime"]
        aggr = expected_identity["aggressiveness"]
        if regime not in BATCH_REGIMES:
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} has unknown batch regime")
        cell_seed = ri.deterministic_seed(cell_id)
        expected_stream, expected_eval = _condition_positions(
            y,
            comp,
            BATCH_REGIMES[regime],
            args.n_eval,
            np.random.default_rng(cell_seed),
            n_batches=args.n_batches,
        )
        expected_provenance = _condition_sample_provenance(
            sub,
            {
                "stream_requested_positions": expected_stream,
                "stream_resolved_positions": expected_stream,
                "eval_requested_positions": expected_eval,
                "eval_resolved_positions": expected_eval,
            },
            cell_seed,
        )
        if condition.get("sample_provenance") != expected_provenance:
            raise ri.RunIntegrityError(
                f"RxRx1 cell {cell_id} sample provenance is not deterministically reproducible"
            )

        eval_y = np.asarray(condition.get("eval_y"), dtype=int)
        frozen = np.asarray(condition.get("preds_frozen"), dtype=int)
        if eval_y.ndim != 1 or eval_y.size != len(expected_eval) or frozen.shape != eval_y.shape:
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} has invalid evaluation truth/predictions")
        if not np.array_equal(eval_y, np.asarray(y, dtype=int)[expected_eval]):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} evaluation labels do not match sampled rows")
        if np.any(frozen < 0) or np.any(frozen >= NUM_CLASSES):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} frozen predictions are out of range")
        a0 = float(tm.balanced_acc(frozen, eval_y))
        if not np.isclose(float(condition.get("a0")), a0, rtol=0.0, atol=1e-12):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} frozen score is inconsistent")

        cell_records = by_cell.get(cell_id, [])
        if [row.get("candidate") for row in cell_records] != expected_candidates:
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate transaction is incomplete or reordered")
        predictions = [frozen]
        scores = [a0]
        for candidate, record in zip(expected_candidates, cell_records):
            if record.get("scientific_cell_identity") != expected_identity:
                raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate identity mismatch")
            if any(record.get(field) != value for field, value in archived_identity.items()):
                raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate archived identity mismatch")
            if record.get("sample_provenance") != expected_provenance:
                raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate provenance mismatch")
            method, mode = candidate.split("_", 1)
            if record.get("method") != method or record.get("mode") != mode:
                raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate method/mode mismatch")
            ri.validate_evidence_record(
                record,
                tm.EVIDENCE_NAMES,
                expected_tta_protocol=tm.tta_protocol_contract(mode),
                context=f"RxRx1 {cell_id}/{candidate}",
            )
            preds = np.asarray(record.get("preds"), dtype=int)
            if preds.shape != eval_y.shape or np.any(preds < 0) or np.any(preds >= NUM_CLASSES):
                raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate predictions are invalid")
            aa = float(tm.balanced_acc(preds, eval_y))
            if not np.isclose(float(record.get("a0")), a0, rtol=0.0, atol=1e-12):
                raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate frozen score mismatch")
            if not np.isclose(float(record.get("aa")), aa, rtol=0.0, atol=1e-12):
                raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate score mismatch")
            if not np.isclose(float(record.get("B")), aa - a0, rtol=0.0, atol=1e-12):
                raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate benefit mismatch")
            if record.get("regime_label") != an.label_regime(aa - a0):
                raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} candidate regime label mismatch")
            predictions.append(preds)
            scores.append(aa)

        names = ["freeze_f0", *expected_candidates]
        if condition.get("cand_names") != names or len(condition.get("aa_all", [])) != len(scores):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} condition candidate summary mismatch")
        if any(
            not np.isclose(float(actual), expected, rtol=0.0, atol=1e-12)
            for actual, expected in zip(condition["aa_all"], scores)
        ):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} condition score vector mismatch")
        if not np.isclose(float(condition.get("oracle")), max(scores), rtol=0.0, atol=1e-12):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} oracle score mismatch")
        if not np.isclose(float(condition.get("best_adapt")), max(scores[1:]), rtol=0.0, atol=1e-12):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} best-adapt score mismatch")
        if condition.get("true_best") != names[int(np.argmax(scores))]:
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} true-best label mismatch")
        expected_route = an.multicandidate_route(
            np.stack(predictions, 0),
            tau_star=args.tau_star,
            kappa=args.kappa,
            objective="balanced_accuracy",
            n_classes=NUM_CLASSES,
            anchor_above_chance=False,
        )
        _validate_unsupported_route_b(expected_route)
        if condition.get("route") != expected_route or condition.get("realized") is not None:
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} Route-B payload is inconsistent")
        if condition.get("route_objective") != {
            "metric": "balanced_accuracy",
            "n_classes": NUM_CLASSES,
            "route_b_eligible": False,
            "reason": "binary-accuracy identity is inapplicable",
        }:
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} Route-B objective contract mismatch")
        rc.validate_unsupported_route_c(
            condition.get("route_c"), "balanced_accuracy", NUM_CLASSES
        )
        if condition.get("regime_label") != an.label_regime(max(scores[1:]) - a0):
            raise ri.RunIntegrityError(f"RxRx1 cell {cell_id} condition regime label mismatch")


def _route_b_summary(conditions):
    scorable = [c for c in conditions if c.get("route", {}).get("scorable") is True]
    if not scorable:
        return {
            "status": "UNSUPPORTED",
            "scorable": False,
            "reason": (
                "Route B identifies binary ordinary accuracy, not balanced accuracy on "
                "RxRx1's 1139-class evaluation pool."
            ),
            "n_conditions": len(conditions),
            "n_scorable_conditions": 0,
        }
    return rc.aggregate_multicandidate(scorable)


def _validate_unsupported_route_b(route):
    if not isinstance(route, dict):
        raise ri.RunIntegrityError("RxRx1 Route B returned a non-object payload")
    if route.get("decision") == "ERROR" or route.get("status") == "ERROR":
        raise ri.RunIntegrityError(
            f"RxRx1 Route B failed: {route.get('reason', 'missing reason')}"
        )
    if not (
        route.get("decision") == "ABSTAIN"
        and route.get("status") == "UNSUPPORTED"
        and route.get("scorable") is False
    ):
        raise ri.RunIntegrityError(
            "RxRx1 balanced-accuracy Route B must be unsupported and unscorable"
        )


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


# --------------------------------- the sweep ---------------------------------
def run(args, out_dir):
    t0 = time.time()
    device = tm.pick_device(args.device)
    partial = os.path.join(out_dir, "_partial.json")
    checkpoint = _checkpoint_identity(args)
    ds, sub, y, n_present, n_total = load_rxrx1(args.data_root, args.split, device)
    population = _rxrx1_population_identity(ds, sub, y)
    scientific_config = _scientific_config(args, checkpoint, population)
    config_sha256 = ri.stable_sha256(scientific_config)
    expected = _expected_cell_ids(args, checkpoint, config_sha256)

    def validate_partial_semantics(candidate_records, completed_conditions):
        _validate_rxrx1_state(
            candidate_records,
            completed_conditions,
            args=args,
            checkpoint=checkpoint,
            run_config_sha256=config_sha256,
            sub=sub,
            y=y,
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
    print(f"[rxrx1] split={args.split} present={n_present}/{n_total} "
          f"classes={len(np.unique(y))} device={device}", flush=True)
    print(f"[resume] {len(done)} completed cells loaded, {len(records)} records carried", flush=True)
    f0 = load_f0(args.ckpt, device)
    n_cells = (len(args.seeds) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness))
    ci = 0
    # WIN_HUNT_v5: online-only candidate pool (the "continual" no-episodic-reset op-point) when
    # --online-only is set; default keeps all six online+episodic candidates (byte-identical).
    _cands = [(m, md) for (m, md) in rc.CANDIDATES
              if (not getattr(args, "online_only", False)) or md == "online"]
    for stream_seed in args.seeds:
        for comp in args.compositions:
            for regime in args.batch_regimes:
                for aggr in args.aggressiveness:
                    ci += 1
                    tag = _tag(stream_seed, comp, regime, aggr)
                    scientific_cell_identity = _cell_spec(
                        config_sha256,
                        args.split,
                        args.model_seed,
                        checkpoint["sha256"],
                        checkpoint["tensor_sha256"],
                        stream_seed,
                        comp,
                        regime,
                        aggr,
                    )
                    cell_id = ri.make_cell_id(**scientific_cell_identity)
                    if cell_id in done:
                        print(f"  [{ci}/{n_cells}] {tag} SKIP (already done)", flush=True)
                        continue
                    cell_seed = ri.deterministic_seed(cell_id)
                    torch.manual_seed(cell_seed); np.random.seed(cell_seed % (2 ** 31))
                    rng = np.random.default_rng(cell_seed)         # per-cell -> resume is bit-identical
                    bs = BATCH_REGIMES[regime]
                    steps = args.steps_override or rc.AGGR[aggr]["steps"]; lr = args.adapt_lr if getattr(args, "adapt_lr", None) is not None else rc.AGGR[aggr]["lr"]
                    cell_records = []
                    try:
                        stream, eval_x, eval_y, sample_ids = build_condition(
                            sub, y, comp, bs, args.n_eval, rng, device,
                            n_batches=args.n_batches, return_ids=True,
                        )
                        sample_provenance = _condition_sample_provenance(
                            sub, sample_ids, cell_seed
                        )
                        a0, p0, _ = tm.eval_frozen(f0, eval_x, eval_y, prob_mode="max")
                        preds_all = [p0]; aa_all = [a0]; cand_names = ["freeze_f0"]
                        for (method, mode) in _cands:
                            aa, Z, upd, preds, _ = tm.run_candidate(
                                method, mode, f0, stream, eval_x, eval_y, NUM_CLASSES, steps, lr,
                                eval_bs=args.episodic_batch, prob_mode="max", episodic_steps=args.episodic_steps)
                            cell_records.append(dict(cell_id=cell_id,
                                                scientific_cell_identity=scientific_cell_identity,
                                                seed=int(stream_seed),
                                                stream_seed=int(stream_seed), model_seed=int(args.model_seed),
                                                checkpoint_sha256=checkpoint["sha256"],
                                                checkpoint_tensor_sha256=checkpoint["tensor_sha256"],
                                                inference_unit="stream_seed_on_one_fixed_model_checkpoint",
                                                independent_model_ci_eligible=False, domain=DOMAIN,
                                                comp=comp, regime=regime, aggr=aggr,
                                                method=method, mode=mode, candidate=f"{method}_{mode}",
                                                tta_protocol=tm.tta_protocol_contract(mode),
                                                metric="balanced_accuracy",
                                                a0=float(a0), aa=float(aa), B=float(aa - a0), upd_norm=float(upd),
                                                Z=[float(z) for z in Z],
                                                preds=[int(value) for value in preds],
                                                sample_provenance=sample_provenance,
                                                regime_label=an.label_regime(float(aa - a0))))
                            preds_all.append(preds); aa_all.append(float(aa)); cand_names.append(f"{method}_{mode}")
                            tm.mps_free(); gc.collect()
                        preds_mat = np.stack(preds_all, 0)
                        route = an.multicandidate_route(
                            preds_mat, tau_star=args.tau_star, kappa=args.kappa,
                            objective="balanced_accuracy", n_classes=NUM_CLASSES,
                            anchor_above_chance=False,
                        )
                        _validate_unsupported_route_b(route)
                        realized = rc.route_realized(route, aa_all)
                        oracle = float(max(aa_all)); best_adapt = float(max(aa_all[1:]))
                        route_c = rc.unsupported_route_c("balanced_accuracy", NUM_CLASSES)
                        condition = dict(cell_id=cell_id,
                                               scientific_cell_identity=scientific_cell_identity,
                                               seed=int(stream_seed),
                                               stream_seed=int(stream_seed), model_seed=int(args.model_seed),
                                               checkpoint_sha256=checkpoint["sha256"],
                                               checkpoint_tensor_sha256=checkpoint["tensor_sha256"],
                                               inference_unit="stream_seed_on_one_fixed_model_checkpoint",
                                               independent_model_ci_eligible=False, domain=DOMAIN,
                                               comp=comp, regime=regime, aggr=aggr,
                                               cand_names=cand_names, aa_all=[float(a) for a in aa_all], a0=float(a0),
                                               oracle=oracle, best_adapt=best_adapt,
                                               true_best=cand_names[int(np.argmax(aa_all))], route=route,
                                               route_c=route_c, realized=realized,
                                               route_objective={"metric": "balanced_accuracy", "n_classes": NUM_CLASSES,
                                                                "route_b_eligible": False,
                                                                "reason": "binary-accuracy identity is inapplicable"},
                                               sample_provenance=sample_provenance,
                                               eval_y=[int(value) for value in eval_y],
                                               preds_frozen=[int(value) for value in p0],
                                               regime_label=an.label_regime(best_adapt - a0))
                        _commit_cell(records, conditions, failures, cell_records, condition)
                        done.add(cell_id)
                        print(f"  [{ci}/{n_cells}] {tag} a0={a0:.3f} best_aa={best_adapt:.3f} "
                              f"oracle={oracle:.3f} route={route.get('decision')} "
                              f"tau={route.get('tau', float('nan')):.3f} sd_c={route_c.get('decision')}", flush=True)
                    except Exception as e:
                        print(f"  [{ci}/{n_cells}] {tag} ERROR: {repr(e)[:140]}", flush=True)
                        ri.upsert_failure(failures, {
                            "cell_id": cell_id, "tag": tag, "error_type": type(e).__name__,
                            "error": str(e)[:500], "attempted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        })
                    # ---- survival: validate then atomically flush; write/semantic failures are fatal ----
                    document = ri.partial_document(
                        run_config_sha256=config_sha256,
                        expected_cell_ids=expected,
                        records=records,
                        conditions=conditions,
                        failures=failures,
                        progress=f"{len(done)}/{n_cells}",
                        require_scientific_cell_identity=True,
                        semantic_validator=validate_partial_semantics,
                    )
                    document["elapsed_sec"] = round(time.time() - t0, 1)
                    ri.atomic_json_dump(document, partial)
                    try:
                        del stream, eval_x
                    except Exception:
                        pass
                    try:
                        del preds_all, preds_mat
                    except Exception:
                        pass
                    deep_free(device)
    del f0; deep_free(device)
    ledger = ri.build_ledger(expected, conditions, failures)
    all_done = ledger["execution_complete"]
    return records, conditions, {"n_present": n_present, "n_total": n_total,
                                 "n_classes": int(len(np.unique(y))), "split": args.split,
                                 "wall_sec": time.time() - t0, "all_done": all_done,
                                 "n_cells_done": len(done), "n_cells_total": n_cells,
                                 "ledger": ledger, "scientific_config": scientific_config,
                                 "config_sha256": config_sha256,
                                 "population_sha256": population["sha256"]}


def build_manifest(args, records, conditions, meta):
    cfg = meta["scientific_config"]
    sha = meta["config_sha256"]
    return {
        "schema": "kbound_rxrx1_v0.6", "dataset": "wilds-rxrx1", "metric": "balanced_accuracy",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"node": platform.node(), "platform": platform.platform(),
                 "python": platform.python_version(), "torch": torch.__version__,
                 "mps": bool(torch.backends.mps.is_available())},
        "config": cfg, "config_sha256": sha, "config_sha8": sha[:8],
        "completion_ledger": meta["ledger"],
        "execution_complete": bool(meta["ledger"]["execution_complete"]),
        "publication_eligible": False,
        "publication_eligibility_note": (
            "diagnostic opened-target run with stream-seed pseudo-replications; no locked "
            "independent-model held-out confirmation"
        ),
        "claim_eligibility": {
            "raw_completed_records": bool(meta["ledger"]["execution_complete"]),
            "route_a_single_candidate": False,
            "route_b_multicandidate": False,
            "route_c_smooth_drift": False,
            "independent_model_ci": False,
        },
        "model_identity": cfg["model_identity"],
        "inference_unit": cfg["inference_unit"],
        "f0": ("torchvision resnet50(num_classes=1139) <- official WILDS RxRx1 ERM checkpoint "
               "(rxrx1_seed:0_epoch:best_model.pth, 'model.'-stripped, 0 missing/0 unexpected); frozen; "
               "WILDS rxrx1 eval transform (ToTensor + per-image standardize); in-dist acc ~35.9%"),
        "num_classes": NUM_CLASSES,
        "candidates": [f"{m}_{md}" for (m, md) in rc.CANDIDATES
                       if (not getattr(args, "online_only", False)) or md == "online"],
        "domain": f"{DOMAIN} (OOD '{args.split}' split = 14 unseen experiments)",
        "metric_contract": {
            "name": "balanced_accuracy",
            "definition": "mean per-class recall over labels present in each evaluation pool",
            "ordinary_accuracy_alias_allowed": False,
        },
        "route_b_eligibility": (
            "UNSUPPORTED: the Route-B agreement identity targets binary ordinary accuracy, "
            "whereas RxRx1 is evaluated by balanced accuracy over a 1139-class label space. "
            "No Route-B decision, realized score, regret, or beats-both claim is computed."
        ),
        "eval_pool_note": (
            "the evaluation pool is class-balanced then capped at n_eval; the runner reports "
            "balanced accuracy directly and does not assume equality with ordinary accuracy"
        ),
        "data": {"data_root": args.data_root, "split": args.split, "n_present": meta["n_present"],
                 "n_total": meta["n_total"], "n_dropped_disk_filter": meta["n_total"] - meta["n_present"],
                 "n_classes": meta["n_classes"], "population_sha256": meta["population_sha256"],
                 "wall_sec": round(meta["wall_sec"], 1),
                 "cells_done": meta["n_cells_done"], "cells_total": meta["n_cells_total"]},
        "baselines": {
            "always_freeze_mean_balanced_accuracy": (
                float(np.mean([r["a0"] for r in records])) if records else None
            ),
            "per_candidate_always_adapt_mean_balanced_accuracy": {
                c: float(np.mean([r["aa"] for r in records if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in records))},
            "per_condition_oracle_mean_balanced_accuracy": (
                float(np.mean([c["oracle"] for c in conditions])) if conditions else None
            )},
        "routing_a_single_candidate": rc.relabel_balanced_accuracy_fields(
            rc.aggregate_single_candidate(records)
        ),
        "routing_b_multicandidate": _route_b_summary(conditions),
        "routing_c_smooth_drift": rc.aggregate_smoothdrift(conditions),
        "detectability": an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {"note": "need>=4"},
        "kbound_summary": rc.kbound_summary(records, conditions, delta=args.delta),
        "tau_distribution": sorted([float(c["route"]["tau"]) for c in conditions if c["route"].get("tau") is not None]),
        "records": records, "conditions": conditions,
    }


def _completion_paths(args, config_sha256):
    """Resolve the exact operational run/result identity bound by ``.done``."""

    run_name = "rxrx1_kbound_smoke" if args.smoke else str(args.run_name)
    if not run_name or os.path.basename(run_name) != run_name:
        raise ri.RunIntegrityError("RxRx1 run name must be one safe path component")
    run_dir = os.path.abspath(os.path.join(os.path.expanduser(args.results_root), run_name))
    result_path = os.path.abspath(
        os.path.expanduser(
            args.out or os.path.join(run_dir, f"result_{str(config_sha256)[:8]}.json")
        )
    )
    return run_name, run_dir, result_path


def _expected_completion_context(args):
    """Recompute the current scientific configuration before trusting old state.

    Receipt verification deliberately performs the same checkpoint and ordered
    population hashing used by a real run.  This is more expensive than trusting
    metadata stored beside ``.done``, but it prevents a receipt from an older code,
    checkpoint, dataset population, or CLI configuration from stopping a new run.
    """

    checkpoint = _checkpoint_identity(args)
    dataset, subset, labels, _, _ = load_rxrx1(args.data_root, args.split, None)
    population = _rxrx1_population_identity(dataset, subset, labels)
    scientific_config = _scientific_config(args, checkpoint, population)
    config_sha256 = ri.stable_sha256(scientific_config)
    run_name, run_dir, result_path = _completion_paths(args, config_sha256)
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "result_path": result_path,
        "config_sha256": config_sha256,
    }


def _completion_receipt(result_path, manifest, *, run_name, run_dir):
    ledger = manifest.get("completion_ledger")
    if manifest.get("execution_complete") is not True or not isinstance(ledger, dict):
        raise ri.RunIntegrityError("refusing to issue a completion receipt for an incomplete result")
    if ledger.get("execution_complete") is not True:
        raise ri.RunIntegrityError("completion ledger is not complete")
    config = manifest.get("config")
    config_sha256 = manifest.get("config_sha256")
    if not isinstance(config, dict) or ri.stable_sha256(config) != config_sha256:
        raise ri.RunIntegrityError("RxRx1 result scientific configuration hash mismatch")
    resolved = os.path.abspath(os.path.expanduser(str(result_path)))
    resolved_run_dir = os.path.abspath(os.path.expanduser(str(run_dir)))
    run_name = str(run_name)
    if (
        not run_name
        or os.path.basename(run_name) != run_name
        or os.path.basename(resolved_run_dir) != run_name
    ):
        raise ri.RunIntegrityError("RxRx1 completion receipt has an invalid run identity")
    return {
        "schema": COMPLETION_RECEIPT_SCHEMA,
        "run_name": run_name,
        "run_dir": resolved_run_dir,
        "result_path": resolved,
        "result_sha256": ri.file_sha256(resolved),
        "result_schema": manifest.get("schema"),
        "config_sha256": config_sha256,
        "completion_ledger_sha256": ri.stable_sha256(ledger),
        "execution_complete": True,
    }


def validate_completion_receipt(
    receipt_path,
    *,
    expected_run_name,
    expected_run_dir,
    expected_result_path,
    expected_config_sha256,
):
    """Validate ``.done`` against an independently recomputed run context."""

    receipt_path = os.path.abspath(os.path.expanduser(str(receipt_path)))
    expected_run_name = str(expected_run_name)
    expected_run_dir = os.path.abspath(os.path.expanduser(str(expected_run_dir)))
    expected_result_path = os.path.abspath(os.path.expanduser(str(expected_result_path)))
    if (
        not expected_run_name
        or os.path.basename(expected_run_name) != expected_run_name
        or os.path.basename(expected_run_dir) != expected_run_name
    ):
        raise ri.RunIntegrityError("invalid expected RxRx1 run identity")
    if receipt_path != os.path.join(expected_run_dir, ".done"):
        raise ri.RunIntegrityError("RxRx1 completion receipt is outside the expected run directory")
    receipt = ri.strict_json_load(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("schema") != COMPLETION_RECEIPT_SCHEMA:
        raise ri.RunIntegrityError(f"unrecognized RxRx1 completion receipt: {receipt_path}")
    if (
        receipt.get("run_name") != expected_run_name
        or receipt.get("run_dir") != expected_run_dir
        or receipt.get("config_sha256") != expected_config_sha256
    ):
        raise ri.RunIntegrityError("RxRx1 completion receipt does not match the expected run context")
    result_path = receipt.get("result_path")
    if not isinstance(result_path, str) or not os.path.isabs(result_path):
        raise ri.RunIntegrityError("RxRx1 completion receipt has no absolute result path")
    if result_path != expected_result_path:
        raise ri.RunIntegrityError("RxRx1 completion receipt points to an unexpected result path")
    if not os.path.isfile(result_path):
        raise ri.RunIntegrityError(f"RxRx1 completion result is missing: {result_path}")
    if receipt.get("result_sha256") != ri.file_sha256(result_path):
        raise ri.RunIntegrityError("RxRx1 completion result hash mismatch")
    manifest = ri.strict_json_load(result_path)
    if not isinstance(manifest, dict):
        raise ri.RunIntegrityError("RxRx1 completion result must be a JSON object")
    ledger = manifest.get("completion_ledger")
    if (
        manifest.get("execution_complete") is not True
        or not isinstance(ledger, dict)
        or ledger.get("execution_complete") is not True
    ):
        raise ri.RunIntegrityError("RxRx1 completion receipt points to an incomplete result")
    config = manifest.get("config")
    if (
        manifest.get("config_sha256") != expected_config_sha256
        or not isinstance(config, dict)
        or ri.stable_sha256(config) != expected_config_sha256
    ):
        raise ri.RunIntegrityError("RxRx1 completion result does not match the expected scientific configuration")
    expected = _completion_receipt(
        result_path,
        manifest,
        run_name=expected_run_name,
        run_dir=expected_run_dir,
    )
    if receipt != expected:
        raise ri.RunIntegrityError("RxRx1 completion receipt metadata mismatch")
    return result_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="K-Bound TTA sweep on WILDS RxRx1 (1139-class, MPS survival)")
    p.add_argument("--data-root", default=os.path.expanduser("~/kbound_rxrx1_data"), dest="data_root",
                   help="dir containing rxrx1_v1.0 (INTERNAL copy; T9 exFAT reads stall MPS)")
    p.add_argument("--ckpt", default=os.path.expanduser("~/kbound_rxrx1_ckpt/rxrx1_seed:0_epoch:best_model.pth"))
    p.add_argument("--model-seed", type=int, default=0, dest="model_seed",
                   help="training seed of the one fixed RxRx1 checkpoint (default: official seed 0)")
    p.add_argument("--split", default="test", help="OOD target domain (default 'test' = 14 unseen experiments)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="stream RNG seeds evaluated on the same fixed model; not independent model seeds")
    p.add_argument("--compositions", nargs="+", default=["iid", "imbalanced", "single_class"])
    p.add_argument("--batch-regimes", nargs="+", default=["small", "tiny"], dest="batch_regimes",
                   help="LIGHT recipe: large_iid DROPPED (it OOM-killed heavy sweeps)")
    p.add_argument("--aggressiveness", nargs="+", default=["mild", "aggressive"])
    p.add_argument("--n-eval", type=int, default=256, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=4, dest="n_batches")
    p.add_argument("--episodic-steps", type=int, default=5, dest="episodic_steps")
    p.add_argument("--episodic-batch", type=int, default=64, dest="episodic_batch")
    p.add_argument("--tau-star", type=float, default=0.52, dest="tau_star")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument("--sd-L", type=float, default=0.6, dest="sd_L")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--steps-override", type=int, default=0, dest="steps_override")
    p.add_argument("--results-root", default=os.path.expanduser("~/kbound_rxrx1_results"), dest="results_root")
    p.add_argument("--run-name", default="rxrx1_kbound_light_mps_internal", dest="run_name")
    p.add_argument("--out", default="")
    p.add_argument(
        "--verify-completion",
        default="",
        dest="verify_completion",
        help=(
            "recompute the current checkpoint/data/config identity, validate a bound .done "
            "completion receipt against it, and exit"
        ),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    # ---- WIN_HUNT_v5 aggressive-regime wave operating-point overrides (opt-in) ----
    p.add_argument("--adapt-lr", type=float, default=None, dest="adapt_lr",
                   help="WIN_HUNT_v5: absolute adapter LR override for tent/eata/sar (rc.AGGR cell "
                        "lr ignored when set). DEFAULT None = per-cell lr (byte-identical). v5 sets "
                        "0.004 (= 4x the 1e-3 shared-baseline lr).")
    p.add_argument("--online-only", action="store_true", dest="online_only",
                   help="WIN_HUNT_v5: restrict the candidate pool to online (no-episodic-reset) "
                        "adapters -- the 'continual' operating point. DEFAULT off (byte-identical).")
    a = p.parse_args(argv)
    if a.smoke:
        a.compositions = ["iid", "single_class"]; a.batch_regimes = ["tiny"]; a.aggressiveness = ["mild"]
        a.seeds = [0, 1]; a.n_eval = 32; a.n_batches = 2; a.steps_override = 4
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.verify_completion:
        print("[patch]", ensure_rxrx1_patch(), flush=True)
        expected = _expected_completion_context(a)
        result = validate_completion_receipt(
            a.verify_completion,
            expected_run_name=expected["run_name"],
            expected_run_dir=expected["run_dir"],
            expected_result_path=expected["result_path"],
            expected_config_sha256=expected["config_sha256"],
        )
        print(f"[completion receipt valid] {result}", flush=True)
        return result
    print("[patch]", ensure_rxrx1_patch(), flush=True)
    effective_run_name, out_dir, _ = _completion_paths(a, "pending")
    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, ".done")
    if os.path.exists(done_path):
        os.unlink(done_path)
        print(f"[completion] removed pre-existing receipt before run: {done_path}", flush=True)
    records, conditions, meta = run(a, out_dir)
    if not meta["all_done"]:
        ledger = meta["ledger"]
        raise RuntimeError(
            f"incomplete RxRx1 run: {meta['n_cells_done']}/{meta['n_cells_total']} cells done; "
            f"failed={ledger['failed_cells']} missing={ledger['missing_cells']}; "
            "partial is not publication-eligible and the process must exit nonzero"
        )
    man = build_manifest(a, records, conditions, meta)
    effective_run_name, expected_out_dir, out = _completion_paths(a, meta["config_sha256"])
    if expected_out_dir != os.path.abspath(out_dir):
        raise ri.RunIntegrityError("RxRx1 output directory changed during execution")
    ri.atomic_json_dump(man, out)
    if ri.strict_json_load(out) != man:
        raise ri.RunIntegrityError("RxRx1 result round-trip verification failed")
    ri.atomic_json_dump(
        _completion_receipt(
            out,
            man,
            run_name=effective_run_name,
            run_dir=out_dir,
        ),
        done_path,
    )
    validate_completion_receipt(
        done_path,
        expected_run_name=effective_run_name,
        expected_run_dir=out_dir,
        expected_result_path=out,
        expected_config_sha256=meta["config_sha256"],
    )
    ks = man["kbound_summary"]; mb = man["routing_b_multicandidate"]; td = man["tau_distribution"]
    print("\n" + "=" * 70, flush=True)
    print(f"records={len(records)} conditions={len(conditions)} wall={meta['wall_sec']:.1f}s", flush=True)
    print(f"classification : {ks['classification']}  base_harmful={ks['base_rate_harmful_B<0']:.3f}  mean_B={ks['mean_B']:+.4f}", flush=True)
    print(f"detectability  : {ks['detectability_verdict']} (best harm-AUC={ks['best_single_feature_harm_AUC']})", flush=True)
    print(f"multicand route: mean_tau={mb.get('mean_tau')} abstain={mb.get('abstention_rate')} breakdown={mb.get('routing_breakdown')}", flush=True)
    print((f"tau range      : [{td[0]:.3f}..{td[-1]:.3f}] (tau*={a.tau_star})") if td else "tau range: n/a", flush=True)
    print(f"manifest -> {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
