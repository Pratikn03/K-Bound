"""
run_imagenetr_kbound.py - K-Bound TTA sweep on ImageNet-R (rendition shift).

Same protocol as the Camelyon17 debug run, generalized to multi-class:
  frozen f0   : torchvision ResNet-50 (IMAGENET1K_V2), logits masked to the 200
                ImageNet-R classes  (ImageNet-R is a robustness test set for ImageNet
                classifiers -> NO training; just restrict the 1000 logits to the 200).
  candidates  : {tent,eata,sar} x {online,episodic}   (reused from tta_methods)
  routing     : (a) single-cand KGA; Route B is unsupported for the multiclass
                objective; Route C is unsupported because its binary Brier-score
                bracket does not identify balanced-accuracy benefit.
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
import run_integrity as ri          # noqa: E402  (strict resume/completeness/publication contract)
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
    try:
        with Image.open(path) as image:
            return TRANSFORM(image.convert("RGB"))
    except Exception as exc:
        raise RuntimeError(
            f"unreadable requested ImageNet-R image {path!r}; sample substitution is forbidden"
        ) from exc


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
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("._"):
                items.append((join(d, f), w2l[w]))
    return items


def _condition_positions(labels, comp, bs, n_eval, rng, n_batches=4):
    labels = np.asarray(labels, dtype=int)
    N = len(labels); pos_all = np.arange(N)
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
        raise RuntimeError("evaluation pool leaves no disjoint ImageNet-R adaptation samples")
    n_stream = max(bs, bs * n_batches)
    if comp == "iid":
        if len(remain) < n_stream:
            raise RuntimeError(f"iid stream needs {n_stream} unique samples; only {len(remain)} remain")
        s = rng.choice(remain, n_stream, replace=False)
    elif comp == "imbalanced":
        maj = int(rng.choice(classes))
        mp = np.intersect1d(pos_all[labels == maj], remain); op = np.setdiff1d(remain, mp)
        if len(mp) and len(op):
            nM = int(n_stream * 0.85)
            if len(mp) < nM or len(op) < n_stream - nM:
                raise RuntimeError(
                    f"imbalanced stream needs {nM}/{n_stream - nM} unique majority/other "
                    f"samples; only {len(mp)}/{len(op)} remain"
                )
            s = np.concatenate([rng.choice(mp, nM, replace=False),
                                rng.choice(op, n_stream - nM, replace=False)])
        else:
            raise RuntimeError("imbalanced stream requires both majority and non-majority samples")
    else:  # single_class label shift
        maj = int(rng.choice(classes))
        mp = np.intersect1d(pos_all[labels == maj], remain)
        if len(mp) < n_stream:
            raise RuntimeError(
                f"single-class stream needs {n_stream} unique class-{maj} samples; only {len(mp)} remain"
            )
        s = rng.choice(mp, n_stream, replace=False)
    rng.shuffle(s)
    if len(np.unique(s)) != len(s) or len(np.unique(ev)) != len(ev):
        raise RuntimeError("ImageNet-R condition contains duplicate requested identities")
    if np.intersect1d(s, ev).size:
        raise RuntimeError("ImageNet-R adaptation and evaluation identities overlap")
    return np.asarray(s, dtype=int), np.asarray(ev, dtype=int)


def build_condition(index, labels, comp, bs, n_eval, rng, device, n_batches=4, tries=None,
                    return_ids=False):
    """Class-balanced held-out eval + composition-controlled adaptation stream.
    single_class/imbalanced + tiny batches are the natural collapse-prone cells.
    Stream is label-free at use; selected identities are loaded exactly or the
    whole cell fails.  Substitution, duplicate identities, and stream/eval
    overlap are forbidden."""
    s, ev = _condition_positions(labels, comp, bs, n_eval, rng, n_batches=n_batches)

    def _load_exact(positions):
        import concurrent.futures
        paths = [index[int(position)][0] for position in positions]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            xs = list(executor.map(load_img, paths))
        return torch.stack(xs).to(device)

    stream_x = _load_exact(s)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    eval_x = _load_exact(ev)
    eval_y = labels[ev].astype(int)
    if return_ids:
        return stream, eval_x, eval_y, {
            "stream_requested_positions": np.asarray(s, dtype=int),
            "stream_resolved_positions": np.asarray(s, dtype=int),
            "eval_requested_positions": np.asarray(ev, dtype=int),
            "eval_resolved_positions": np.asarray(ev, dtype=int),
        }
    return stream, eval_x, eval_y


def _cell_spec(seed, comp, regime, aggr):
    return {
        "dataset": "imagenet-r",
        "model_seed": 0,
        "model_replication": "fixed_torchvision_pretrained_weights",
        "stream_seed": int(seed),
        "composition": comp,
        "batch_regime": regime,
        "aggressiveness": aggr,
    }


def _cell_id(seed, comp, regime, aggr):
    return ri.make_cell_id(**_cell_spec(seed, comp, regime, aggr))


def _expected_cell_ids(args):
    return [
        _cell_id(seed, comp, regime, aggr)
        for seed in args.seeds
        for comp in args.compositions
        for regime in args.batch_regimes
        for aggr in args.aggressiveness
    ]


def _population_identity(index, imagenetr_dir):
    root = os.path.abspath(imagenetr_dir)
    rows = []
    for path, label in index:
        absolute = os.path.abspath(path)
        relative = os.path.relpath(absolute, root)
        size = os.path.getsize(absolute)
        content_sha256 = ri.file_sha256(absolute)
        rows.append({
            "path": relative,
            "label": int(label),
            "bytes": size,
            "content_sha256": content_sha256,
        })
    return {
        "sha256": ri.stable_sha256(rows),
        "n_images": len(rows),
        "identity_fields": ["path", "label", "bytes", "content_sha256"],
        "order": "class-WNID then filename lexical order",
    }


def _model_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_identity(backbone, description, model):
    tensor_sha256 = _model_state_sha256(model)
    return {
        "backbone": str(backbone),
        "description": str(description),
        "tensor_sha256": tensor_sha256,
        # Retain the historical name as an explicit alias for old audit tooling.
        "state_sha256": tensor_sha256,
    }


def _collect_candidate_identities(backbones, select_indices, device):
    names = list(backbones)
    if len(names) != len(set(names)):
        raise ValueError("candidate backbone inventory contains duplicates")
    identities = {}
    for name in names:
        candidate = None
        try:
            candidate, description = make_masked_backbone(name, select_indices, device)
            identities[name] = _model_identity(name, description, candidate)
        finally:
            if candidate is not None:
                candidate.to(torch.device("cpu"))
                del candidate
            tm.mps_free()
    hashes = [identities[name]["tensor_sha256"] for name in names]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("candidate backbones do not have unique tensor-state identities")
    return identities


def _validate_diverse_resume_model_identities(records, conditions, candidate_identities):
    """Bind every resumed diverse-panel row to the tensor identities in config."""
    expected = dict(candidate_identities)
    if not expected:
        raise ri.RunIntegrityError("diverse-backbone resume lacks candidate tensor identities")
    for index, record in enumerate(records):
        candidate = record.get("candidate")
        if candidate not in expected:
            raise ri.RunIntegrityError(
                f"resumed diverse record {index} has an unknown candidate backbone"
            )
        if (
            record.get("candidate_model_identity") != expected[candidate]
            or record.get("candidate_tensor_sha256") != expected[candidate]["tensor_sha256"]
        ):
            raise ri.RunIntegrityError(
                f"resumed candidate {candidate!r} is not bound to its configured tensor identity"
            )
    for index, condition in enumerate(conditions):
        if condition.get("candidate_model_identities") != expected:
            raise ri.RunIntegrityError(
                f"resumed diverse condition {index} has a mismatched candidate identity inventory"
            )


def _scientific_config(
    args,
    *,
    resolved_device,
    population_identity,
    f0_identity,
    candidate_identities=None,
):
    fields = (
        "imagenetr_dir", "panel", "f0_backbone", "candidate_backbones",
        "seeds", "compositions", "batch_regimes", "aggressiveness",
        "n_eval", "n_batches", "tau_star", "kappa", "sd_L", "delta",
        "steps_override", "max_classes", "episodic_steps", "episodic_batch",
        "frozen_eval_batch", "smoke", "adapt_lr", "online_only",
    )
    config = {key: getattr(args, key) for key in fields}
    config["imagenetr_dir"] = os.path.abspath(os.path.expanduser(config["imagenetr_dir"]))
    config["class_index"] = os.path.abspath(os.path.expanduser(args.class_index))
    config["class_index_sha256"] = ri.file_sha256(config["class_index"])
    config["resolved_device"] = str(resolved_device)
    config["metric"] = "balanced_accuracy"
    config["route_b_task_status"] = "unsupported_multiclass"
    config["route_c_contract"] = rc.route_c_contract("balanced_accuracy", NUM_CLASSES)
    config["population_manifest"] = population_identity
    config["f0_artifact"] = f0_identity
    candidate_identities = candidate_identities or {}
    if config["panel"] == "diverse_backbones":
        expected_candidates = list(config["candidate_backbones"])
        if (
            len(expected_candidates) != len(set(expected_candidates))
            or set(candidate_identities) != set(expected_candidates)
        ):
            raise ValueError(
                "diverse-backbone scientific config requires one exact tensor identity per candidate"
            )
        hashes = []
        for name in expected_candidates:
            identity = candidate_identities[name]
            tensor_sha256 = identity.get("tensor_sha256") if isinstance(identity, dict) else None
            if (
                not isinstance(tensor_sha256, str)
                or len(tensor_sha256) != 64
                or any(character not in "0123456789abcdef" for character in tensor_sha256.lower())
            ):
                raise ValueError(f"candidate {name!r} lacks a valid tensor-state SHA-256")
            if identity.get("backbone") != name:
                raise ValueError(f"candidate {name!r} identity is bound to another backbone")
            hashes.append(tensor_sha256.lower())
        if len(hashes) != len(set(hashes)):
            raise ValueError("candidate tensor-state identities must be unique")
        config["candidate_model_artifacts"] = {
            name: candidate_identities[name] for name in expected_candidates
        }
        config["candidate_tta_protocols"] = {}
    else:
        if candidate_identities:
            raise ValueError("shared-TTA config cannot carry unrelated candidate-backbone identities")
        config["candidate_model_artifacts"] = {}
        shared_candidates = [
            (method, mode)
            for method, mode in rc.CANDIDATES
            if (not config["online_only"]) or mode == "online"
        ]
        config["candidate_tta_protocols"] = {
            f"{method}_{mode}": tm.tta_protocol_contract(mode)
            for method, mode in shared_candidates
        }
    config["implementation_sha256"] = {
        "runner": ri.file_sha256(__file__),
        "tta_methods": ri.file_sha256(tm.__file__),
        "analysis": ri.file_sha256(an.__file__),
        "routing_aggregates": ri.file_sha256(rc.__file__),
        "per_condition_serialize": ri.file_sha256(pcs.__file__),
        "panel_capture": ri.file_sha256(pc.__file__),
    }
    config["seed_semantics"] = {
        "model_seed": 0,
        "model_replications": 1,
        "args_seeds_role": "stream_seed",
        "independent_model_ci_eligible": False,
    }
    return config


def _condition_sample_provenance(index, sample_ids, condition_seed):
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
        raise RuntimeError("ImageNet-R requested/resolved identity or disjointness invariant failed")
    return {
        "condition_seed": int(condition_seed),
        "stream_n": int(len(stream_resolved)),
        "eval_n": int(len(eval_resolved)),
        "ordered_stream_requested_positions_sha256": ri.stable_sha256(stream_requested.tolist()),
        "ordered_stream_resolved_positions_sha256": ri.stable_sha256(stream_resolved.tolist()),
        "ordered_eval_requested_positions_sha256": ri.stable_sha256(eval_requested.tolist()),
        "ordered_eval_resolved_positions_sha256": ri.stable_sha256(eval_resolved.tolist()),
        "ordered_stream_sample_ids_sha256": ri.stable_sha256([
            os.path.abspath(index[int(position)][0]) for position in stream_resolved
        ]),
        "ordered_eval_sample_ids_sha256": ri.stable_sha256([
            os.path.abspath(index[int(position)][0]) for position in eval_resolved
        ]),
        "requested_resolved_identity_equal": bool(equal),
        "stream_eval_disjoint": bool(overlap.size == 0),
        "stream_unique": bool(len(np.unique(stream_resolved)) == len(stream_resolved)),
        "eval_unique": bool(len(np.unique(eval_resolved)) == len(eval_resolved)),
        "stream_eval_overlap_count": int(overlap.size),
    }


def _close_score(actual, expected, *, atol=1e-12):
    try:
        return bool(np.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= atol)
    except (TypeError, ValueError):
        return False


def _validate_imagenetr_completed_cell(
    condition,
    cell_records,
    *,
    args,
    index,
    labels,
    f0_identity,
    candidate_identities,
):
    """Validate a completed ImageNet-R cell from current scientific context."""

    if not isinstance(condition, dict):
        raise ri.RunIntegrityError("ImageNet-R resumed condition must be an object")
    try:
        seed = int(condition["stream_seed"])
        comp = str(condition["comp"])
        regime = str(condition["regime"])
        aggr = str(condition["aggr"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ri.RunIntegrityError("ImageNet-R resumed condition has incomplete axes") from exc
    if (
        seed not in [int(value) for value in args.seeds]
        or comp not in args.compositions
        or regime not in args.batch_regimes
        or aggr not in args.aggressiveness
    ):
        raise ri.RunIntegrityError("ImageNet-R resumed condition is outside the configured grid")

    scientific_identity = _cell_spec(seed, comp, regime, aggr)
    cell_id = _cell_id(seed, comp, regime, aggr)
    ri.validate_scientific_cell_identity(
        condition.get("cell_id"),
        condition.get("scientific_cell_identity"),
        context="ImageNet-R resumed condition",
    )
    if (
        condition.get("cell_id") != cell_id
        or condition.get("scientific_cell_identity") != scientific_identity
    ):
        raise ri.RunIntegrityError("ImageNet-R resumed scientific cell identity mismatch")
    sample_seed = ri.deterministic_seed(cell_id)
    expected_identity = {
        "model_seed": 0,
        "stream_seed": seed,
        "sampling_seed": sample_seed,
        "seed": seed,
        "domain": "imagenet_r",
        "comp": comp,
        "regime": regime,
        "aggr": aggr,
        "f0_model_identity": f0_identity,
    }
    for field, expected in expected_identity.items():
        if condition.get(field) != expected:
            raise ri.RunIntegrityError(
                f"ImageNet-R resumed condition has mismatched {field}"
            )

    stream_positions, eval_positions = _condition_positions(
        labels,
        comp,
        BATCH_REGIMES[regime],
        args.n_eval,
        np.random.default_rng(sample_seed),
        n_batches=args.n_batches,
    )
    sample_ids = {
        "stream_requested_positions": stream_positions,
        "stream_resolved_positions": stream_positions,
        "eval_requested_positions": eval_positions,
        "eval_resolved_positions": eval_positions,
    }
    expected_provenance = _condition_sample_provenance(index, sample_ids, sample_seed)
    provenance = condition.get("sample_provenance")
    if provenance != expected_provenance:
        raise ri.RunIntegrityError(
            "ImageNet-R resumed sample provenance differs from deterministic selection"
        )
    try:
        eval_y = np.asarray(condition.get("eval_y"), dtype=int)
        frozen = np.asarray(condition.get("preds_frozen"), dtype=int)
    except (TypeError, ValueError) as exc:
        raise ri.RunIntegrityError(
            "ImageNet-R resumed evaluation labels/predictions are invalid"
        ) from exc
    expected_eval_y = np.asarray(labels, dtype=int)[eval_positions]
    if (
        eval_y.ndim != 1
        or eval_y.size == 0
        or not np.array_equal(eval_y, expected_eval_y)
        or frozen.shape != eval_y.shape
    ):
        raise ri.RunIntegrityError(
            "ImageNet-R resumed evaluation labels/predictions are inconsistent"
        )
    a0 = tm.balanced_acc(frozen, eval_y)
    if not _close_score(condition.get("a0"), a0):
        raise ri.RunIntegrityError("ImageNet-R resumed frozen score is inconsistent")

    if args.panel == "diverse_backbones":
        candidate_specs = [("backbone", "frozen", name) for name in args.candidate_backbones]
        if condition.get("candidate_model_identities") != candidate_identities:
            raise ri.RunIntegrityError(
                "ImageNet-R resumed diverse candidate identity inventory mismatch"
            )
    else:
        candidate_specs = [
            (method, mode, f"{method}_{mode}")
            for method, mode in rc.CANDIDATES
            if (not getattr(args, "online_only", False)) or mode == "online"
        ]
        if candidate_identities:
            raise ri.RunIntegrityError(
                "ImageNet-R shared-TTA resume has unrelated backbone identities"
            )
    rows = {record.get("candidate"): record for record in cell_records}
    names = ["freeze_f0", *[candidate for _, _, candidate in candidate_specs]]
    if len(rows) != len(candidate_specs) or condition.get("cand_names") != names:
        raise ri.RunIntegrityError(
            "ImageNet-R resumed candidate transaction differs from configured candidates"
        )

    aa_all = [a0]
    for method, mode, candidate in candidate_specs:
        record = rows.get(candidate)
        if not isinstance(record, dict):
            raise ri.RunIntegrityError(
                f"ImageNet-R resumed cell is missing candidate {candidate}"
            )
        if (
            record.get("cell_id") != cell_id
            or record.get("scientific_cell_identity") != scientific_identity
            or record.get("method") != method
            or record.get("mode") != mode
            or record.get("candidate") != candidate
            or record.get("metric") != "balanced_accuracy"
            or record.get("sample_provenance") != provenance
        ):
            raise ri.RunIntegrityError(
                f"ImageNet-R resumed candidate {candidate} has inconsistent identity/provenance"
            )
        for field, expected in expected_identity.items():
            if record.get(field) != expected:
                raise ri.RunIntegrityError(
                    f"ImageNet-R resumed candidate {candidate} has mismatched {field}"
                )

        expected_protocol = None
        if args.panel == "diverse_backbones":
            expected_model_identity = candidate_identities.get(candidate)
            if (
                not isinstance(expected_model_identity, dict)
                or record.get("candidate_model_identity") != expected_model_identity
                or record.get("candidate_tensor_sha256") != expected_model_identity.get("tensor_sha256")
                or record.get("candidate_artifact") != expected_model_identity.get("description")
                or not _close_score(record.get("upd_norm"), 0.0)
            ):
                raise ri.RunIntegrityError(
                    f"ImageNet-R resumed backbone {candidate} has mismatched tensor identity"
                )
        else:
            expected_protocol = tm.tta_protocol_contract(mode)
        ri.validate_evidence_record(
            record,
            tm.EVIDENCE_NAMES,
            expected_tta_protocol=expected_protocol,
            context=f"ImageNet-R {cell_id}/{candidate}",
        )

        try:
            preds = np.asarray(record.get("preds"), dtype=int)
        except (TypeError, ValueError) as exc:
            raise ri.RunIntegrityError(
                f"ImageNet-R resumed candidate {candidate} predictions are invalid"
            ) from exc
        if preds.shape != eval_y.shape:
            raise ri.RunIntegrityError(
                f"ImageNet-R resumed candidate {candidate} prediction length mismatch"
            )
        aa = tm.balanced_acc(preds, eval_y)
        if (
            not _close_score(record.get("a0"), a0)
            or not _close_score(record.get("aa"), aa)
            or not _close_score(record.get("B"), aa - a0)
            or record.get("regime_label") != an.label_regime(aa - a0)
        ):
            raise ri.RunIntegrityError(
                f"ImageNet-R resumed candidate {candidate} score semantics are inconsistent"
            )
        aa_all.append(aa)

    stored_scores = condition.get("aa_all")
    if not isinstance(stored_scores, list) or len(stored_scores) != len(aa_all) or any(
        not _close_score(actual, expected)
        for actual, expected in zip(stored_scores, aa_all)
    ):
        raise ri.RunIntegrityError("ImageNet-R resumed condition scores are inconsistent")
    if (
        not _close_score(condition.get("best_adapt"), max(aa_all[1:]))
        or not _close_score(condition.get("oracle"), max(aa_all))
        or condition.get("true_best") != names[int(np.argmax(aa_all))]
        or condition.get("regime_label") != an.label_regime(max(aa_all[1:]) - a0)
    ):
        raise ri.RunIntegrityError("ImageNet-R resumed condition summary is inconsistent")


def _validate_resume_semantics(
    records,
    conditions,
    *,
    args,
    index,
    labels,
    f0_identity,
    candidate_identities,
):
    records_by_cell = {}
    for record in records:
        records_by_cell.setdefault(record.get("cell_id"), []).append(record)
    for condition in conditions:
        _validate_imagenetr_completed_cell(
            condition,
            records_by_cell.get(condition.get("cell_id"), []),
            args=args,
            index=index,
            labels=labels,
            f0_identity=f0_identity,
            candidate_identities=candidate_identities,
        )


def _flush_partial(partial_path, *, run_config_sha256, expected_cell_ids,
                   records, conditions, failures, progress, semantic_validator):
    if not partial_path:
        return
    payload = ri.partial_document(
        run_config_sha256=run_config_sha256,
        expected_cell_ids=expected_cell_ids,
        records=records,
        conditions=conditions,
        failures=failures,
        progress=progress,
        require_scientific_cell_identity=True,
        semantic_validator=semantic_validator,
    )
    ri.atomic_json_dump(payload, partial_path)


def _execute_diverse_cell(
    *, args, f0, index, labels, num_classes, select_indices, device,
    seed, comp, regime, aggr, bs, cell_id, sample_seed, f0_identity,
    candidate_identities,
):
    rng = np.random.default_rng(sample_seed)
    torch.manual_seed(sample_seed)
    stream, eval_x, eval_y, sample_ids = build_condition(
        index, labels, comp, bs, args.n_eval, rng, device, n_batches=args.n_batches,
        return_ids=True,
    )
    sample_provenance = _condition_sample_provenance(index, sample_ids, sample_seed)
    a0, p0, _ = tm.eval_frozen(
        f0, eval_x, eval_y, prob_mode="max", bs=args.frozen_eval_batch,
    )
    cell_records = []
    preds_all = [p0]
    aa_all = [a0]
    cand_names = ["freeze_f0"]
    probe = stream[0]
    for name in args.candidate_backbones:
        candidate = None
        try:
            # load candidate lazily so heavyweight backbones never accumulate in
            # unified memory during the diverse-panel sweep.
            candidate, description = make_masked_backbone(name, select_indices, device)
            candidate.eval()
            actual_identity = _model_identity(name, description, candidate)
            expected_identity = candidate_identities.get(name)
            if actual_identity != expected_identity:
                raise RuntimeError(
                    f"candidate backbone tensor identity changed after run configuration lock: {name}"
                )
            aa, preds, _ = tm.eval_frozen(
                candidate, eval_x, eval_y, prob_mode="max", bs=args.frozen_eval_batch,
            )
            evidence = tm.evidence_vector(f0, candidate, probe, num_classes, upd_norm=0.0)
            benefit = float(aa - a0)
            cell_records.append({
                "cell_id": cell_id,
                "scientific_cell_identity": _cell_spec(seed, comp, regime, aggr),
                "model_seed": 0,
                "stream_seed": int(seed),
                "sampling_seed": int(sample_seed),
                "seed": int(seed),
                "domain": "imagenet_r",
                "comp": comp,
                "regime": regime,
                "aggr": aggr,
                "method": "backbone",
                "mode": "frozen",
                "candidate": name,
                "candidate_artifact": description,
                "candidate_model_identity": actual_identity,
                "candidate_tensor_sha256": actual_identity["tensor_sha256"],
                "f0_model_identity": f0_identity,
                "metric": "balanced_accuracy",
                "a0": float(a0),
                "aa": float(aa),
                "B": benefit,
                "upd_norm": 0.0,
                "Z": [float(value) for value in evidence],
                "regime_label": an.label_regime(benefit),
                "sample_provenance": sample_provenance,
                "preds": [int(value) for value in preds],
            })
            preds_all.append(preds)
            aa_all.append(float(aa))
            cand_names.append(name)
        finally:
            if candidate is not None:
                candidate.to(torch.device("cpu"))
                del candidate
            tm.mps_free()

    predictions = np.stack(preds_all, 0)
    pc.attach_to_last(cell_records, len(cand_names) - 1, pc.panel_fields(predictions))
    route = an.multicandidate_route(
        predictions,
        tau_star=args.tau_star,
        kappa=args.kappa,
        task_type="multiclass_classification",
        n_classes=num_classes,
        objective="balanced_accuracy",
        anchor_above_chance=False,
    )
    realized = rc.route_realized(route, aa_all)
    oracle = float(max(aa_all))
    best_adapt = float(max(aa_all[1:]))
    route_c = rc.unsupported_route_c("balanced_accuracy", num_classes)
    return cell_records, {
        "cell_id": cell_id,
        "scientific_cell_identity": _cell_spec(seed, comp, regime, aggr),
        "model_seed": 0,
        "stream_seed": int(seed),
        "sampling_seed": int(sample_seed),
        "f0_model_identity": f0_identity,
        "seed": int(seed),
        "domain": "imagenet_r",
        "comp": comp,
        "regime": regime,
        "aggr": aggr,
        "cand_names": cand_names,
        "aa_all": [float(value) for value in aa_all],
        "a0": float(a0),
        "oracle": oracle,
        "best_adapt": best_adapt,
        "true_best": cand_names[int(np.argmax(aa_all))],
        "route": route,
        "route_objective": {
            "metric": "balanced_accuracy",
            "n_classes": int(num_classes),
            "route_b_eligible": False,
        },
        "route_c": route_c,
        "realized": realized,
        "route_scorable": realized is not None,
        "regime_label": an.label_regime(best_adapt - a0),
        "sample_provenance": sample_provenance,
        "eval_y": [int(value) for value in eval_y],
        "preds_frozen": [int(value) for value in p0],
        "candidate_model_identities": {
            name: candidate_identities[name] for name in args.candidate_backbones
        },
    }


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

    population_identity = _population_identity(index, args.imagenetr_dir)
    f0_identity = _model_identity(args.f0_backbone, f0_desc, f0)
    candidate_identities = _collect_candidate_identities(
        args.candidate_backbones, sel, device
    )
    scientific_config = _scientific_config(
        args,
        resolved_device=device,
        population_identity=population_identity,
        f0_identity=f0_identity,
        candidate_identities=candidate_identities,
    )
    run_config_sha256 = ri.stable_sha256(scientific_config)
    expected_cell_ids = _expected_cell_ids(args)

    def validate_partial_semantics(candidate_records, completed_conditions):
        _validate_diverse_resume_model_identities(
            candidate_records, completed_conditions, candidate_identities
        )
        _validate_resume_semantics(
            candidate_records,
            completed_conditions,
            args=args,
            index=index,
            labels=labels,
            f0_identity=f0_identity,
            candidate_identities=candidate_identities,
        )

    records, conditions, failures = [], [], []
    if partial_path and getattr(args, "resume", True):
        records, conditions, failures = ri.load_partial_state(
            partial_path,
            run_config_sha256=run_config_sha256,
            expected_cell_ids=expected_cell_ids,
            require_scientific_cell_identity=True,
            semantic_validator=validate_partial_semantics,
        )
        if conditions or failures:
            print(
                f"[resume] completed={len(conditions)} prior_failures={len(failures)} "
                f"from {partial_path}",
                flush=True,
            )
    done = {condition["cell_id"] for condition in conditions}
    n_total = len(expected_cell_ids)
    ci = 0
    for seed in args.seeds:
        for comp in args.compositions:
            for regime in args.batch_regimes:
                bs = BATCH_REGIMES[regime]
                for aggr in args.aggressiveness:
                    ci += 1
                    tag = f"s{seed}/{comp}/{regime}/{aggr}"
                    cell_id = _cell_id(seed, comp, regime, aggr)
                    if cell_id in done:
                        print(f"  [{ci}/{n_total}] {tag} SKIP (resume)", flush=True)
                        continue
                    sample_seed = ri.deterministic_seed(cell_id)
                    try:
                        cell_records, condition = _execute_diverse_cell(
                            args=args,
                            f0=f0,
                            index=index,
                            labels=labels,
                            num_classes=num,
                            select_indices=sel,
                            device=device,
                            seed=seed,
                            comp=comp,
                            regime=regime,
                            aggr=aggr,
                            bs=bs,
                            cell_id=cell_id,
                            sample_seed=sample_seed,
                            f0_identity=f0_identity,
                            candidate_identities=candidate_identities,
                        )
                        _validate_imagenetr_completed_cell(
                            condition,
                            cell_records,
                            args=args,
                            index=index,
                            labels=labels,
                            f0_identity=f0_identity,
                            candidate_identities=candidate_identities,
                        )
                    except Exception as exc:
                        ri.upsert_failure(failures, {
                            "cell_id": cell_id,
                            **_cell_spec(seed, comp, regime, aggr),
                            "sampling_seed": int(sample_seed),
                            "stage": "cell_execution",
                            "error_type": type(exc).__name__,
                            "error": repr(exc),
                        })
                        print(f"  [{ci}/{n_total}] {tag} ERROR: {repr(exc)[:160]}", flush=True)
                        _flush_partial(
                            partial_path,
                            run_config_sha256=run_config_sha256,
                            expected_cell_ids=expected_cell_ids,
                            records=records,
                            conditions=conditions,
                            failures=failures,
                            progress=f"{ci}/{n_total}",
                            semantic_validator=validate_partial_semantics,
                        )
                        continue
                    records.extend(cell_records)
                    conditions.append(condition)
                    done.add(cell_id)
                    ri.clear_failure(failures, cell_id)
                    route = condition["route"]
                    print(
                        f"  [{ci}/{n_total}] {tag} a0={condition['a0']:.3f} "
                        f"best_aa={condition['best_adapt']:.3f} oracle={condition['oracle']:.3f} "
                        f"route={route.get('decision')} status={route.get('status')} "
                        f"sd_c={condition['route_c'].get('decision')}",
                        flush=True,
                    )
                    _flush_partial(
                        partial_path,
                        run_config_sha256=run_config_sha256,
                        expected_cell_ids=expected_cell_ids,
                        records=records,
                        conditions=conditions,
                        failures=failures,
                        progress=f"{ci}/{n_total}",
                        semantic_validator=validate_partial_semantics,
                    )
    f0.to(torch.device("cpu")); tm.mps_free()
    return records, conditions, {
        "n_images": len(index), "n_classes": len(wnids), "wall_sec": time.time() - t0,
        "panel": "diverse_backbones", "f0": f0_desc,
        "candidate_backbones": list(args.candidate_backbones),
        "candidate_names": list(args.candidate_backbones),
        "scientific_config": scientific_config,
        "run_config_sha256": run_config_sha256,
        "expected_cell_ids": expected_cell_ids,
        "failures": failures,
        "ledger": ri.build_ledger(expected_cell_ids, conditions, failures),
        "population_identity": population_identity,
        "f0_identity": f0_identity,
        "candidate_identities": candidate_identities,
    }


def _execute_shared_cell(
    *, args, f0, index, labels, num_classes, device, candidates,
    seed, comp, regime, aggr, bs, cell_id, sample_seed, f0_identity,
):
    rng = np.random.default_rng(sample_seed)
    torch.manual_seed(sample_seed)
    stream, eval_x, eval_y, sample_ids = build_condition(
        index, labels, comp, bs, args.n_eval, rng, device, n_batches=args.n_batches,
        return_ids=True,
    )
    sample_provenance = _condition_sample_provenance(index, sample_ids, sample_seed)
    a0, p0, _ = tm.eval_frozen(f0, eval_x, eval_y, prob_mode="max")
    cell_records = []
    preds_all = [p0]
    aa_all = [a0]
    cand_names = ["freeze_f0"]
    steps = args.steps_override or rc.AGGR[aggr]["steps"]
    lr = args.adapt_lr if getattr(args, "adapt_lr", None) is not None else rc.AGGR[aggr]["lr"]
    for method, mode in candidates:
        aa, evidence, update_norm, preds, _ = tm.run_candidate(
            method, mode, f0, stream, eval_x, eval_y, num_classes,
            steps, lr, eval_bs=args.episodic_batch, prob_mode="max",
            episodic_steps=args.episodic_steps,
        )
        benefit = float(aa - a0)
        cell_records.append({
            "cell_id": cell_id,
            "scientific_cell_identity": _cell_spec(seed, comp, regime, aggr),
            "model_seed": 0,
            "stream_seed": int(seed),
            "sampling_seed": int(sample_seed),
            "seed": int(seed),
            "domain": "imagenet_r",
            "comp": comp,
            "regime": regime,
            "aggr": aggr,
            "method": method,
            "mode": mode,
            "candidate": f"{method}_{mode}",
            "f0_model_identity": f0_identity,
            "tta_protocol": tm.tta_protocol_contract(mode),
            "metric": "balanced_accuracy",
            "a0": float(a0),
            "aa": float(aa),
            "B": benefit,
            "upd_norm": float(update_norm),
            "Z": [float(value) for value in evidence],
            "regime_label": an.label_regime(benefit),
            "sample_provenance": sample_provenance,
            "preds": [int(value) for value in preds],
        })
        preds_all.append(preds)
        aa_all.append(float(aa))
        cand_names.append(f"{method}_{mode}")
        tm.mps_free()

    predictions = np.stack(preds_all, 0)
    pc.attach_to_last(cell_records, len(candidates), pc.panel_fields(predictions))
    route = an.multicandidate_route(
        predictions,
        tau_star=args.tau_star,
        kappa=args.kappa,
        task_type="multiclass_classification",
        n_classes=num_classes,
        objective="balanced_accuracy",
        anchor_above_chance=False,
    )
    realized = rc.route_realized(route, aa_all)
    oracle = float(max(aa_all))
    best_adapt = float(max(aa_all[1:]))
    route_c = rc.unsupported_route_c("balanced_accuracy", num_classes)
    return cell_records, {
        "cell_id": cell_id,
        "scientific_cell_identity": _cell_spec(seed, comp, regime, aggr),
        "model_seed": 0,
        "stream_seed": int(seed),
        "sampling_seed": int(sample_seed),
        "f0_model_identity": f0_identity,
        "seed": int(seed),
        "domain": "imagenet_r",
        "comp": comp,
        "regime": regime,
        "aggr": aggr,
        "cand_names": cand_names,
        "aa_all": [float(value) for value in aa_all],
        "a0": float(a0),
        "oracle": oracle,
        "best_adapt": best_adapt,
        "true_best": cand_names[int(np.argmax(aa_all))],
        "route": route,
        "route_objective": {
            "metric": "balanced_accuracy",
            "n_classes": int(num_classes),
            "route_b_eligible": False,
        },
        "route_c": route_c,
        "realized": realized,
        "route_scorable": realized is not None,
        "regime_label": an.label_regime(best_adapt - a0),
        "sample_provenance": sample_provenance,
        "eval_y": [int(value) for value in eval_y],
        "preds_frozen": [int(value) for value in p0],
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
    population_identity = _population_identity(index, args.imagenetr_dir)
    f0_identity = {
        "description": "torchvision resnet50 IMAGENET1K_V2",
        "state_sha256": _model_state_sha256(f0),
    }
    scientific_config = _scientific_config(
        args,
        resolved_device=device,
        population_identity=population_identity,
        f0_identity=f0_identity,
        candidate_identities={},
    )
    run_config_sha256 = ri.stable_sha256(scientific_config)
    expected_cell_ids = _expected_cell_ids(args)

    def validate_partial_semantics(candidate_records, completed_conditions):
        _validate_resume_semantics(
            candidate_records,
            completed_conditions,
            args=args,
            index=index,
            labels=labels,
            f0_identity=f0_identity,
            candidate_identities={},
        )

    records, conditions, failures = [], [], []
    if partial_path and getattr(args, "resume", True):
        records, conditions, failures = ri.load_partial_state(
            partial_path,
            run_config_sha256=run_config_sha256,
            expected_cell_ids=expected_cell_ids,
            require_scientific_cell_identity=True,
            semantic_validator=validate_partial_semantics,
        )
        if conditions or failures:
            print(
                f"[resume] completed={len(conditions)} prior_failures={len(failures)} "
                f"from {partial_path}",
                flush=True,
            )
    done = {condition["cell_id"] for condition in conditions}
    n_total = len(expected_cell_ids)
    ci = 0
    # WIN_HUNT_v5: online-only candidate pool (the "continual" no-episodic-reset op-point) when
    # --online-only is set; default keeps all six online+episodic candidates (byte-identical).
    _cands = [(m, md) for (m, md) in rc.CANDIDATES
              if (not getattr(args, "online_only", False)) or md == "online"]
    for seed in args.seeds:
        for comp in args.compositions:
            for regime in args.batch_regimes:
                bs = BATCH_REGIMES[regime]
                for aggr in args.aggressiveness:
                    ci += 1
                    tag = f"s{seed}/{comp}/{regime}/{aggr}"
                    cell_id = _cell_id(seed, comp, regime, aggr)
                    if cell_id in done:
                        print(f"  [{ci}/{n_total}] {tag} SKIP (resume)", flush=True)
                        continue
                    sample_seed = ri.deterministic_seed(cell_id)
                    try:
                        cell_records, condition = _execute_shared_cell(
                            args=args,
                            f0=f0,
                            index=index,
                            labels=labels,
                            num_classes=num,
                            device=device,
                            candidates=_cands,
                            seed=seed,
                            comp=comp,
                            regime=regime,
                            aggr=aggr,
                            bs=bs,
                            cell_id=cell_id,
                            sample_seed=sample_seed,
                            f0_identity=f0_identity,
                        )
                        _validate_imagenetr_completed_cell(
                            condition,
                            cell_records,
                            args=args,
                            index=index,
                            labels=labels,
                            f0_identity=f0_identity,
                            candidate_identities={},
                        )
                    except Exception as exc:
                        ri.upsert_failure(failures, {
                            "cell_id": cell_id,
                            **_cell_spec(seed, comp, regime, aggr),
                            "sampling_seed": int(sample_seed),
                            "stage": "cell_execution",
                            "error_type": type(exc).__name__,
                            "error": repr(exc),
                        })
                        print(f"  [{ci}/{n_total}] {tag} ERROR: {repr(exc)[:160]}", flush=True)
                        _flush_partial(
                            partial_path,
                            run_config_sha256=run_config_sha256,
                            expected_cell_ids=expected_cell_ids,
                            records=records,
                            conditions=conditions,
                            failures=failures,
                            progress=f"{ci}/{n_total}",
                            semantic_validator=validate_partial_semantics,
                        )
                        continue
                    records.extend(cell_records)
                    conditions.append(condition)
                    done.add(cell_id)
                    ri.clear_failure(failures, cell_id)
                    route = condition["route"]
                    print(
                        f"  [{ci}/{n_total}] {tag} a0={condition['a0']:.3f} "
                        f"best_aa={condition['best_adapt']:.3f} oracle={condition['oracle']:.3f} "
                        f"route={route.get('decision')} status={route.get('status')} "
                        f"sd_c={condition['route_c'].get('decision')}",
                        flush=True,
                    )
                    _flush_partial(
                        partial_path,
                        run_config_sha256=run_config_sha256,
                        expected_cell_ids=expected_cell_ids,
                        records=records,
                        conditions=conditions,
                        failures=failures,
                        progress=f"{ci}/{n_total}",
                        semantic_validator=validate_partial_semantics,
                    )
    return records, conditions, {
        "n_images": len(index),
        "n_classes": len(wnids),
        "wall_sec": time.time() - t0,
        "panel": "shared_tta",
        "f0": "torchvision resnet50 IMAGENET1K_V2",
        "candidate_names": [f"{method}_{mode}" for method, mode in _cands],
        "scientific_config": scientific_config,
        "run_config_sha256": run_config_sha256,
        "expected_cell_ids": expected_cell_ids,
        "failures": failures,
        "ledger": ri.build_ledger(expected_cell_ids, conditions, failures),
        "population_identity": population_identity,
        "f0_identity": f0_identity,
    }


def build_manifest(args, records, conditions, meta):
    cfg = meta["scientific_config"]
    config_sha256 = meta["run_config_sha256"]
    complete = bool(meta["ledger"]["execution_complete"])
    candidate_names = meta.get("candidate_names", [f"{m}_{md}" for (m, md) in rc.CANDIDATES])
    not_computed = {
        "status": "NOT_COMPUTED_INCOMPLETE_RUN",
        "scorable": False,
        "note": "aggregate withheld because the expected/completed/failed ledger is incomplete",
    }
    routing_a = (
        rc.relabel_balanced_accuracy_fields(rc.aggregate_single_candidate(records))
        if complete else not_computed
    )
    routing_b = rc.aggregate_multicandidate(conditions) if complete else not_computed
    routing_c = rc.aggregate_smoothdrift(conditions) if complete else not_computed
    detectability = (
        an.detectability_analysis(records, tm.EVIDENCE_NAMES)
        if complete and len(records) >= 4
        else ({"note": "need>=4"} if complete else not_computed)
    )
    summary = rc.kbound_summary(records, conditions, delta=args.delta) if complete else not_computed
    baselines = {
        "always_freeze_mean_balanced_accuracy": (
            float(np.mean([r["a0"] for r in records])) if records else None
        ),
        "per_candidate_always_adapt_mean_balanced_accuracy": {
            candidate: float(np.mean([r["aa"] for r in records if r["candidate"] == candidate]))
            for candidate in sorted(set(r["candidate"] for r in records))
        },
        "per_condition_oracle_mean_balanced_accuracy": (
            float(np.mean([condition["oracle"] for condition in conditions])) if conditions else None
        ),
    } if complete else not_computed
    manifest = {
        "schema": "kbound_imagenetr_v0.7", "dataset": "imagenet-r",
        "metric": "balanced_accuracy",
        "metric_contract": {
            "name": "balanced_accuracy",
            "definition": "mean per-class recall over labels present in each evaluation pool",
            "ordinary_accuracy_alias_allowed": False,
        },
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"platform": platform.platform(), "torch": torch.__version__,
                 "mps": bool(torch.backends.mps.is_available())},
        "config": cfg,
        "config_sha256": config_sha256,
        "config_sha8": config_sha256[:8],
        "run_ledger": meta["ledger"],
        "execution_complete": complete,
        "publication_eligible": False,
        "publication_eligibility_note": (
            "diagnostic run on an opened target; no disjoint validation-locked confirmation"
        ),
        "seed_semantics": cfg["seed_semantics"],
        "model_artifact": meta["f0_identity"],
        "candidate_model_artifacts": meta.get("candidate_identities", {}),
        "claim_eligibility": {
            "raw_completed_records": complete,
            "route_a_single_candidate": False,
            "route_b_multicandidate": False,
            "route_c_smooth_drift": False,
        },
        "failures": meta["failures"],
        "f0": meta.get("f0", "torchvision resnet50 IMAGENET1K_V2; 1000 logits masked to 200 ImageNet-R classes (frozen)"),
        "num_classes": meta["n_classes"], "candidates": candidate_names,
        "panel": getattr(args, "panel", "shared_tta"),
        "multiclass_caveat": ("Route B is unsupported and unscored on the 200-class label space because "
                              "its correctness-agreement identity is binary-only. Stored agreements are "
                              "diagnostic data, not a routing result."),
        "data": {"n_images": meta["n_images"], "n_classes": meta["n_classes"],
                 "population_manifest": meta["population_identity"],
                 "wall_sec": round(meta["wall_sec"], 1)},
        "baselines": baselines,
        "routing_a_single_candidate": routing_a,
        "routing_b_multicandidate": routing_b,
        "routing_c_smooth_drift": routing_c,
        "detectability": detectability,
        "kbound_summary": summary,
        "tau_distribution": sorted([float(c["route"]["tau"]) for c in conditions if c["route"].get("tau") is not None]),
        "records": records, "conditions": conditions,
    }
    return rc.relabel_balanced_accuracy_fields(manifest)


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
    complete = man["execution_complete"]
    prefix = "diagnostic" if complete else "incomplete"
    out = a.out or join(out_dir, f"{prefix}_{man['config_sha8']}.json")
    ri.atomic_json_dump(man, out)
    if not complete:
        print(f"\nmanifest -> {out}")
        raise RuntimeError(f"run incomplete: {man['run_ledger']}; wrote non-promotable artifact {out}")
    # ---- per-condition serialization (stress_grid_multiseed schema) ----------
    if complete and getattr(a, "serialize_per_condition", True) and records:
        # diverse_backbones panel: each frozen backbone is the "method" axis
        # (records carry method="backbone", candidate=<backbone>); shared_tta panel
        # uses method in {tent,eata,sar}.
        m_field = "candidate" if a.panel == "diverse_backbones" else "method"
        methods = sorted({r[m_field] for r in records})
        seeds = [int(s) for s in a.seeds]
        seed_metadata = {
            int(seed): {
                "seed_role": "stream_seed",
                "stream_seed": int(seed),
                "model_seed": 0,
                "checkpoint_sha256": meta["f0_identity"]["state_sha256"],
                "independent_model_ci_eligible": False,
            }
            for seed in seeds
        }
        ser = pcs.serialize_run(records, dataset="imagenet-r", out_dir=out_dir,
                                seeds=seeds, methods=methods, method_field=m_field,
                                seed_metadata=seed_metadata)
        print(f"[serialize] wrote {len(ser['written'])} per-condition files "
              f"(panel={a.panel}, methods={methods}, seeds={seeds}, "
              f"kga_backend={ser['kga_backend']}) -> {out_dir}")
        if a.panel == "diverse_backbones":
            print(
                "[serialize] multicandidate panel artifact withheld: Route B is binary-only; "
                "agreement fields remain available in the completed raw manifest"
            )
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
    try:
        os.unlink(partial)
    except FileNotFoundError:
        pass
    return out


if __name__ == "__main__":
    main()
