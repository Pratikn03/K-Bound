"""
run_camelyon17_kbound.py - K-Bound natural-shift pipeline on WILDS Camelyon17.

PROTOCOL
  frozen f0       : standard WILDS DenseNet-121 (loaded from results/wilds/f0_seed{S}.pt)
  candidates (6)  : {tent,eata,sar} x {online,episodic}
  baselines       : always-adapt (per candidate), always-freeze, per-condition oracle
  routing         : (a) single-candidate KGA certificate  [analysis.decide_kga]
                    (b) multi-candidate tau-residual        [Theorem 1A; analysis.multicandidate_route
                        -> reuses theory_validation/val_multicandidate_residual.py]
                    (c) smooth-drift                        [Theorem 1B; TODO STUB]
  metrics         : mean acc, regret-to-oracle, false-adapt rate, coverage, abstention,
                    per-condition routing breakdown
  detectability   : per-condition label-free Z vs TRUE benefit sign  [analysis.detectability_analysis]

CONDITION = (domain, composition, batch_regime) x (aggressiveness) x seed.
INTEGRITY: every cell is run for real; labels are used ONLY for B/oracle/detectability
eval; the routers see only Z (a) or label-free agreements (b).  Every reported number
traces to records[] / conditions[] in the output JSON manifest.  Run for real; never
fabricate; report null/negative results as-is.
"""
from __future__ import annotations
import os, sys, time, argparse, platform, hashlib, json
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tta_methods as tm        # noqa: E402
import analysis as an           # noqa: E402
import cam_data as cd           # noqa: E402
import per_condition_serialize as pcs  # noqa: E402  (torch-free per-condition serializer)
import panel_capture as pc      # noqa: E402  (Wave-5: c_ij/n_D + ev2 evidence capture)
import run_integrity as ri      # noqa: E402  (strict resume/completeness/publication contract)
# integrity: the duplicated EVIDENCE_NAMES in the serializer must stay in lock-step
assert list(pcs.EVIDENCE_NAMES) == list(tm.EVIDENCE_NAMES), "EVIDENCE_NAMES drift"

CANDIDATES = [("tent", "online"), ("tent", "episodic"),
              ("eata", "online"), ("eata", "episodic"),
              ("sar", "online"), ("sar", "episodic")]
AGGR = {"mild": dict(steps=10, lr=1e-3), "aggressive": dict(steps=50, lr=2.5e-3)}
NUM_CLASSES = 2


def make_model(device):
    import torchvision.models as tv
    m = tv.densenet121(weights=None)            # arch only; weights come from the f0 ckpt
    m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    return m.to(device)


def load_f0(ckpt, device):
    m = make_model(device)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    return m


def route_realized(route, aa_all):
    """Realized score for a valid route; invalid/ERROR states are never scored."""
    if route.get("status") != "OK" or route.get("scorable") is not True:
        return None
    decision = route.get("decision")
    if decision == "ADAPT":
        choice = route.get("choice")
        if not isinstance(choice, int) or not 0 < choice < len(aa_all):
            return None
        return float(aa_all[choice])
    if decision in {"FREEZE", "ABSTAIN"}:
        return float(aa_all[0])
    return None


def route_c_contract(objective, n_classes):
    """Return the fail-closed Route-C scientific contract for these runners.

    ``analysis.smooth_drift_route`` bounds a change in binary Brier score for a
    fixed positive-class probability and a fixed candidate.  Classification
    accuracy, balanced accuracy, and macro-F1 are different estimands.  A
    max-class confidence is also not a positive-class probability, and choosing
    the candidate with target labels would make the route target-selected.
    """

    return {
        "status": "UNSUPPORTED",
        "reported_objective": str(objective),
        "n_classes": int(n_classes),
        "available_theorem_objective": "binary_brier_score_benefit",
        "candidate_requirement": "fixed_before_target_labels",
        "target_label_selection_used": False,
        "reason": (
            "Route C is a binary Brier-score bracket for a fixed candidate and fixed "
            "positive class; it cannot be interpreted as benefit in this runner's "
            f"{objective} objective, and target-label candidate selection is forbidden."
        ),
    }


def unsupported_route_c(objective, n_classes):
    """Materialize an explicitly unscorable Route-C result."""

    return {
        "decision": "ABSTAIN",
        "implemented": False,
        "scorable": False,
        **route_c_contract(objective, n_classes),
    }


def validate_unsupported_route_c(route, objective, n_classes):
    """Reject resumed Route-C outputs from the retired mismatched implementation."""

    expected = unsupported_route_c(objective, n_classes)
    required = {
        "decision": "ABSTAIN",
        "status": "UNSUPPORTED",
        "implemented": False,
        "scorable": False,
        "reported_objective": expected["reported_objective"],
        "n_classes": expected["n_classes"],
        "available_theorem_objective": expected["available_theorem_objective"],
        "target_label_selection_used": False,
    }
    if not isinstance(route, dict) or any(route.get(key) != value for key, value in required.items()):
        raise RuntimeError(
            "Route C must remain the explicit UNSUPPORTED binary-Brier/objective-mismatch contract"
        )
    forbidden = {"bracket", "true_B_best", "bracket_covers_trueB", "choice"}
    if forbidden & set(route):
        raise RuntimeError("unsupported Route C contains retired scored/bracket fields")
    return route


def relabel_balanced_accuracy_fields(value):
    """Replace legacy aggregate key names that imply ordinary accuracy."""

    aliases = {
        "mean_acc": "mean_balanced_accuracy",
        "always_freeze_mean_acc": "always_freeze_mean_balanced_accuracy",
        "per_candidate_always_adapt_mean_acc": "per_candidate_always_adapt_mean_balanced_accuracy",
        "per_condition_oracle_mean_acc": "per_condition_oracle_mean_balanced_accuracy",
    }
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            renamed = aliases.get(key, key)
            if renamed in out:
                raise ValueError(f"balanced-accuracy relabel would overwrite key {renamed!r}")
            out[renamed] = relabel_balanced_accuracy_fields(item)
        return out
    if isinstance(value, list):
        return [relabel_balanced_accuracy_fields(item) for item in value]
    return value


def route_b_metric_contract(eval_y, preds_all, reported_scores, *, atol=1e-12):
    """Verify when the binary agreement route targets the reported score.

    Route B identifies ordinary binary accuracy.  Camelyon reports balanced
    accuracy, which is numerically identical only because ``build_condition``
    constructs an evaluation pool with the same number of examples from each
    class.  Make that equivalence an explicit, fail-closed contract instead of
    silently labelling balanced accuracy as accuracy.

    Labels are used here only for an offline design/metric parity assertion;
    ``multicandidate_route`` still receives predictions alone.
    """

    labels = np.asarray(eval_y)
    predictions = np.asarray(preds_all)
    scores = np.asarray(reported_scores, dtype=float)
    if labels.ndim != 1 or predictions.ndim != 2 or predictions.shape[1] != len(labels):
        raise ValueError("Route-B metric contract requires (M,N) predictions aligned to N labels")
    if scores.shape != (predictions.shape[0],) or not np.isfinite(scores).all():
        raise ValueError("reported Route-B scores must be one finite value per predictor")

    classes, counts = np.unique(labels, return_counts=True)
    counts_by_class = {str(value): int(count) for value, count in zip(classes.tolist(), counts.tolist())}
    exactly_balanced_binary = bool(len(classes) == 2 and counts[0] == counts[1])
    contract = {
        "reported_metric": "balanced_accuracy",
        "identified_metric": "ordinary_accuracy",
        "class_counts": counts_by_class,
        "exactly_balanced_binary_evaluation_pool": exactly_balanced_binary,
        "route_objective": "accuracy" if exactly_balanced_binary else "balanced_accuracy",
        "route_b_metric_eligible": False,
        "verification_scope": "offline metric-parity assertion; router receives predictions only",
    }
    if not exactly_balanced_binary:
        contract["reason"] = (
            "ordinary accuracy is not guaranteed to equal balanced accuracy on this evaluation pool"
        )
        return contract

    ordinary = np.mean(predictions == labels[None, :], axis=1)
    if not np.allclose(ordinary, scores, rtol=0.0, atol=float(atol)):
        raise ValueError(
            "balanced-pool metric parity failed: reported balanced accuracy differs from ordinary accuracy"
        )
    contract.update({
        "route_b_metric_eligible": True,
        "metric_parity_verified": True,
        "reason": "equal per-class counts make balanced accuracy identical to ordinary accuracy",
    })
    return contract


def _cell_spec(seed, dom, comp, regime, aggr):
    return {
        "dataset": "camelyon17",
        "model_seed": int(seed),
        "domain": dom,
        "composition": comp,
        "batch_regime": regime,
        "aggressiveness": aggr,
    }


def _cell_id(seed, dom, comp, regime, aggr):
    return ri.make_cell_id(**_cell_spec(seed, dom, comp, regime, aggr))


def _expected_cell_ids(args):
    return [
        _cell_id(seed, dom, comp, regime, aggr)
        for seed in args.seeds
        for dom in args.domains
        for comp in args.compositions
        for regime in args.batch_regimes
        for aggr in args.aggressiveness
    ]


def _checkpoint_identities(args):
    identities = {}
    for seed in args.seeds:
        path = os.path.abspath(args.f0_template.format(seed=seed))
        if not os.path.exists(path):
            raise FileNotFoundError(f"f0 checkpoint missing: {path}")
        identities[str(int(seed))] = {
            "path": path,
            "sha256": ri.file_sha256(path),
            "tensor_sha256": checkpoint_tensor_sha256(path),
        }
    return identities


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


def _scientific_config(args, checkpoints, *, population_identity, resolved_device):
    config = {k: getattr(args, k) for k in (
        "data_root", "seeds", "domains", "compositions", "batch_regimes",
        "aggressiveness", "n_eval", "n_batches", "tau_star", "kappa", "device",
        "steps_override", "delta", "sd_L", "evidence_panel", "smoke",
        "adapt_lr", "online_only", "anchor_above_chance",
    )}
    config["data_root"] = os.path.abspath(os.path.expanduser(config["data_root"]))
    config["f0_checkpoints"] = checkpoints
    config["population_manifest"] = population_identity
    config["resolved_device"] = str(resolved_device)
    config["candidate_set"] = [
        f"{method}_{mode}" for method, mode in CANDIDATES
        if (not getattr(args, "online_only", False)) or mode == "online"
    ]
    config["metric"] = "balanced_accuracy"
    config["route_c_contract"] = route_c_contract("balanced_accuracy", 2)
    config["implementation_sha256"] = {
        "runner": ri.file_sha256(__file__),
        "cam_data": ri.file_sha256(cd.__file__),
        "tta_methods": ri.file_sha256(tm.__file__),
        "analysis": ri.file_sha256(an.__file__),
        "per_condition_serialize": ri.file_sha256(pcs.__file__),
        "panel_capture": ri.file_sha256(pc.__file__),
    }
    return config


def _population_identity(dom_cache, dataset):
    input_array = getattr(dataset, "_input_array", None)
    full_labels = getattr(dataset, "y_array", None)
    data_dir = os.path.abspath(getattr(dataset, "data_dir", ""))
    population = {}
    for domain, (subset, labels) in sorted(dom_cache.items()):
        rows = []
        for position, global_index in enumerate(np.asarray(subset.indices)):
            index = int(global_index)
            raw_id = str(input_array[index]) if input_array is not None else str(index)
            candidate = raw_id if os.path.isabs(raw_id) else os.path.join(data_dir, raw_id)
            if not os.path.isfile(candidate):
                raise FileNotFoundError(
                    f"Camelyon17 population member is missing: {candidate}"
                )
            size = os.path.getsize(candidate)
            label = int(full_labels[index]) if full_labels is not None else int(labels[position])
            rows.append({
                "official_index": index,
                "official_input_id": raw_id.replace("\\", "/"),
                "label": label,
                "bytes": size,
                "content_sha256": ri.file_sha256(candidate),
            })
        population[domain] = rows
    return {
        "sha256": ri.stable_sha256(population),
        "identity_fields": [
            "official_index", "official_input_id", "label", "bytes", "content_sha256",
        ],
        "counts_by_domain": {domain: len(rows) for domain, rows in population.items()},
    }


def _complete_domain(dataset, transform, keep_full, domain):
    subset = dataset.get_subset(domain, transform=transform)
    official = np.asarray(subset.indices, dtype=int)
    if keep_full is not None and not bool(np.asarray(keep_full)[official].all()):
        missing = official[~np.asarray(keep_full)[official]]
        raise RuntimeError(
            f"Camelyon17 domain={domain!r} is incomplete: {len(missing)}/{len(official)} "
            f"official patches are missing (first official indices: {missing[:5].tolist()}); "
            "restore the archive before running"
        )
    labels = dataset.y_array[official].numpy().astype(int)
    return subset, labels


def _condition_positions(labels, comp, bs, n_eval, rng, n_batches=4):
    """Return the deterministic stream/evaluation positions for one cell."""

    labels = np.asarray(labels, dtype=int)
    pos_all = np.arange(len(labels), dtype=int)
    classes = np.asarray([value for value in cd.CLASSES if np.any(labels == value)], dtype=int)
    if len(classes) < 2:
        classes = np.unique(labels)
    per = max(1, n_eval // max(1, len(classes)))
    eval_positions = np.concatenate([
        rng.choice(pos_all[labels == value], min(per, int(np.sum(labels == value))), replace=False)
        for value in classes if np.any(labels == value)
    ])
    rng.shuffle(eval_positions)
    remaining = np.setdiff1d(pos_all, eval_positions)
    if len(remaining) == 0:
        raise RuntimeError("evaluation pool leaves no disjoint Camelyon17 adaptation samples")
    n_stream = max(bs, bs * n_batches)
    if comp == "iid":
        if len(remaining) < n_stream:
            raise RuntimeError(f"iid stream needs {n_stream} unique samples; only {len(remaining)} remain")
        stream_positions = rng.choice(remaining, n_stream, replace=False)
    elif comp == "imbalanced":
        majority = int(rng.choice(classes))
        majority_pool = np.intersect1d(pos_all[labels == majority], remaining)
        other_pool = np.setdiff1d(remaining, majority_pool)
        n_majority = int(n_stream * 0.85)
        if len(majority_pool) < n_majority or len(other_pool) < n_stream - n_majority:
            raise RuntimeError(
                f"imbalanced stream needs {n_majority}/{n_stream - n_majority} unique "
                f"majority/other samples; only {len(majority_pool)}/{len(other_pool)} remain"
            )
        stream_positions = np.concatenate([
            rng.choice(majority_pool, n_majority, replace=False),
            rng.choice(other_pool, n_stream - n_majority, replace=False),
        ])
    elif comp == "single_class":
        majority = int(rng.choice(classes))
        pool = np.intersect1d(pos_all[labels == majority], remaining)
        if len(pool) < n_stream:
            raise RuntimeError(
                f"single-class stream needs {n_stream} unique class-{majority} samples; "
                f"only {len(pool)} remain"
            )
        stream_positions = rng.choice(pool, n_stream, replace=False)
    else:
        raise ValueError(f"unknown composition: {comp}")
    rng.shuffle(stream_positions)
    if len(np.unique(stream_positions)) != len(stream_positions) or len(
        np.unique(eval_positions)
    ) != len(eval_positions):
        raise RuntimeError("Camelyon17 condition contains duplicate requested identities")
    if np.intersect1d(stream_positions, eval_positions).size:
        raise RuntimeError("Camelyon17 adaptation and evaluation identities overlap")

    return np.asarray(stream_positions, dtype=int), np.asarray(eval_positions, dtype=int)


def _condition_sample_provenance(sub, stream_positions, eval_positions, condition_seed):
    stream_positions = np.asarray(stream_positions, dtype=int)
    eval_positions = np.asarray(eval_positions, dtype=int)
    official = np.asarray(sub.indices, dtype=int)
    overlap = np.intersect1d(stream_positions, eval_positions)
    return {
        "condition_seed": int(condition_seed),
        "stream_n": int(len(stream_positions)),
        "eval_n": int(len(eval_positions)),
        "ordered_stream_requested_positions_sha256": ri.stable_sha256(stream_positions.tolist()),
        "ordered_stream_resolved_positions_sha256": ri.stable_sha256(stream_positions.tolist()),
        "ordered_eval_requested_positions_sha256": ri.stable_sha256(eval_positions.tolist()),
        "ordered_eval_resolved_positions_sha256": ri.stable_sha256(eval_positions.tolist()),
        "ordered_stream_official_ids_sha256": ri.stable_sha256(official[stream_positions].tolist()),
        "ordered_eval_official_ids_sha256": ri.stable_sha256(official[eval_positions].tolist()),
        "requested_resolved_identity_equal": True,
        "stream_eval_disjoint": bool(overlap.size == 0),
        "stream_unique": bool(len(np.unique(stream_positions)) == len(stream_positions)),
        "eval_unique": bool(len(np.unique(eval_positions)) == len(eval_positions)),
        "stream_eval_overlap_count": int(overlap.size),
    }


def _build_condition_exact(
    sub, labels, comp, bs, n_eval, rng, device, n_batches=4, condition_seed=None,
):
    labels = np.asarray(labels, dtype=int)
    stream_positions, eval_positions = _condition_positions(
        labels, comp, bs, n_eval, rng, n_batches=n_batches,
    )

    def load_exact(positions):
        tensors = []
        for position in positions:
            try:
                tensor, actual_label, _ = sub[int(position)]
            except Exception as exc:
                raise RuntimeError(
                    f"unreadable requested Camelyon17 subset position {int(position)}; "
                    "sample substitution is forbidden"
                ) from exc
            if int(actual_label) != int(labels[int(position)]):
                raise RuntimeError(
                    f"Camelyon17 loaded label mismatch at subset position {int(position)}"
                )
            tensors.append(tensor)
        return torch.stack(tensors).to(device)

    stream_x = load_exact(stream_positions)
    eval_x = load_exact(eval_positions)
    stream = [stream_x[index:index + bs] for index in range(0, len(stream_x), bs)]
    provenance = _condition_sample_provenance(
        sub, stream_positions, eval_positions,
        condition_seed if condition_seed is not None else 0,
    )
    return stream, eval_x, labels[eval_positions].astype(int), provenance


def _close_score(actual, expected, *, atol=1e-12):
    try:
        return bool(np.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= atol)
    except (TypeError, ValueError):
        return False


def _validate_camelyon_completed_cell(
    condition, cell_records, *, args, checkpoints, dom_cache,
):
    """Recompute every deterministic resume invariant available to this runner."""

    if not isinstance(condition, dict):
        raise ri.RunIntegrityError("Camelyon17 resumed condition must be an object")
    try:
        seed = int(condition["model_seed"])
        domain = str(condition["domain"])
        comp = str(condition["comp"])
        regime = str(condition["regime"])
        aggr = str(condition["aggr"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ri.RunIntegrityError("Camelyon17 resumed condition has incomplete axes") from exc
    if (
        seed not in [int(value) for value in args.seeds]
        or domain not in args.domains
        or comp not in args.compositions
        or regime not in args.batch_regimes
        or aggr not in args.aggressiveness
    ):
        raise ri.RunIntegrityError("Camelyon17 resumed condition is outside the configured grid")

    scientific_identity = _cell_spec(seed, domain, comp, regime, aggr)
    cell_id = _cell_id(seed, domain, comp, regime, aggr)
    ri.validate_scientific_cell_identity(
        condition.get("cell_id"),
        condition.get("scientific_cell_identity"),
        context="Camelyon17 resumed condition",
    )
    if (
        condition.get("cell_id") != cell_id
        or condition.get("scientific_cell_identity") != scientific_identity
    ):
        raise ri.RunIntegrityError("Camelyon17 resumed scientific cell identity mismatch")

    checkpoint = checkpoints.get(str(seed))
    if not isinstance(checkpoint, dict):
        raise ri.RunIntegrityError("Camelyon17 resumed model seed lacks a current checkpoint")
    sample_seed = ri.deterministic_seed(cell_id)
    expected_identity = {
        "model_seed": seed,
        "seed": seed,
        "stream_seed": sample_seed,
        "domain": domain,
        "comp": comp,
        "regime": regime,
        "aggr": aggr,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
    }
    for field, expected in expected_identity.items():
        if condition.get(field) != expected:
            raise ri.RunIntegrityError(
                f"Camelyon17 resumed condition has mismatched {field}"
            )

    sub, labels = dom_cache[domain]
    stream_positions, eval_positions = _condition_positions(
        labels,
        comp,
        cd.BATCH_REGIMES[regime],
        args.n_eval,
        np.random.default_rng(sample_seed),
        n_batches=args.n_batches,
    )
    expected_provenance = _condition_sample_provenance(
        sub, stream_positions, eval_positions, sample_seed,
    )
    provenance = condition.get("sample_provenance")
    if provenance != expected_provenance:
        raise ri.RunIntegrityError(
            "Camelyon17 resumed sample provenance differs from deterministic selection"
        )
    eval_y = np.asarray(condition.get("eval_y"), dtype=int)
    expected_eval_y = np.asarray(labels, dtype=int)[eval_positions]
    frozen = np.asarray(condition.get("preds_frozen"), dtype=int)
    if (
        eval_y.ndim != 1
        or not np.array_equal(eval_y, expected_eval_y)
        or frozen.shape != eval_y.shape
        or eval_y.size == 0
    ):
        raise ri.RunIntegrityError(
            "Camelyon17 resumed evaluation labels/predictions are inconsistent"
        )
    a0 = tm.balanced_acc(frozen, eval_y)
    if not _close_score(condition.get("a0"), a0):
        raise ri.RunIntegrityError("Camelyon17 resumed frozen score is inconsistent")

    candidate_specs = [
        (method, mode, f"{method}_{mode}")
        for method, mode in CANDIDATES
        if (not getattr(args, "online_only", False)) or mode == "online"
    ]
    rows = {record.get("candidate"): record for record in cell_records}
    expected_names = ["freeze_f0", *[name for _, _, name in candidate_specs]]
    if len(rows) != len(candidate_specs) or condition.get("cand_names") != expected_names:
        raise ri.RunIntegrityError(
            "Camelyon17 resumed candidate transaction differs from configured candidates"
        )

    aa_all = [a0]
    for method, mode, candidate in candidate_specs:
        record = rows.get(candidate)
        if not isinstance(record, dict):
            raise ri.RunIntegrityError(
                f"Camelyon17 resumed cell is missing candidate {candidate}"
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
                f"Camelyon17 resumed candidate {candidate} has inconsistent identity/provenance"
            )
        for field, expected in expected_identity.items():
            if record.get(field) != expected:
                raise ri.RunIntegrityError(
                    f"Camelyon17 resumed candidate {candidate} has mismatched {field}"
                )

        evidence_names = list(tm.EVIDENCE_NAMES)
        if args.evidence_panel == "rich":
            evidence_names += list(tm.RICH_EVIDENCE_NAMES)
        ri.validate_evidence_record(
            record,
            evidence_names,
            expected_tta_protocol=tm.tta_protocol_contract(mode),
            context=f"Camelyon17 {cell_id}/{candidate}",
        )
        z = record["Z"]
        if record.get("evidence_panel") != args.evidence_panel or record.get("Z_base") != z[:len(tm.EVIDENCE_NAMES)]:
            raise ri.RunIntegrityError(
                f"Camelyon17 resumed candidate {candidate} has inconsistent base evidence"
            )
        if args.evidence_panel == "rich" and record.get("Z_rich") != z[len(tm.EVIDENCE_NAMES):]:
            raise ri.RunIntegrityError(
                f"Camelyon17 resumed candidate {candidate} has inconsistent rich evidence"
            )

        preds = np.asarray(record.get("preds"), dtype=int)
        if preds.shape != eval_y.shape:
            raise ri.RunIntegrityError(
                f"Camelyon17 resumed candidate {candidate} prediction length mismatch"
            )
        aa = tm.balanced_acc(preds, eval_y)
        if (
            not _close_score(record.get("a0"), a0)
            or not _close_score(record.get("aa"), aa)
            or not _close_score(record.get("B"), aa - a0)
            or record.get("regime_label") != an.label_regime(aa - a0)
        ):
            raise ri.RunIntegrityError(
                f"Camelyon17 resumed candidate {candidate} score semantics are inconsistent"
            )
        aa_all.append(aa)

    stored_scores = condition.get("aa_all")
    if not isinstance(stored_scores, list) or len(stored_scores) != len(aa_all) or any(
        not _close_score(actual, expected)
        for actual, expected in zip(stored_scores, aa_all)
    ):
        raise ri.RunIntegrityError("Camelyon17 resumed condition scores are inconsistent")
    if (
        not _close_score(condition.get("best_adapt"), max(aa_all[1:]))
        or not _close_score(condition.get("oracle"), max(aa_all))
        or condition.get("true_best") != expected_names[int(np.argmax(aa_all))]
        or condition.get("regime_label") != an.label_regime(max(aa_all[1:]) - a0)
    ):
        raise ri.RunIntegrityError("Camelyon17 resumed condition summary is inconsistent")


def _validate_resume_semantics(records, conditions, *, args, checkpoints, dom_cache):
    records_by_cell = {}
    for record in records:
        records_by_cell.setdefault(record.get("cell_id"), []).append(record)
    for condition in conditions:
        _validate_camelyon_completed_cell(
            condition,
            records_by_cell.get(condition.get("cell_id"), []),
            args=args,
            checkpoints=checkpoints,
            dom_cache=dom_cache,
        )


def _execute_cell(
    *, args, f0, sub, labels, seed, domain, comp, regime, aggr, bs, device,
    candidates, logits_src, y_src, cell_id, sample_seed, checkpoint_identity,
):
    """Execute one cell transactionally and return rows only after full success."""

    rng = np.random.default_rng(sample_seed)
    torch.manual_seed(sample_seed)
    stream, eval_x, eval_y, sample_provenance = _build_condition_exact(
        sub, labels, comp, bs, args.n_eval, rng, device,
        n_batches=args.n_batches, condition_seed=sample_seed,
    )
    steps = args.steps_override or AGGR[aggr]["steps"]
    lr = args.adapt_lr if getattr(args, "adapt_lr", None) is not None else AGGR[aggr]["lr"]
    a0, p0, _ = tm.eval_frozen(f0, eval_x, eval_y)
    logits_f0_eval = None
    bn_kl = 0.0
    rich_note = None
    if args.evidence_panel == "rich":
        logits_f0_eval = tm.predict_logits(f0, eval_x, train_mode=False, bs=256)
        rm, rv = tm.bn_running_stats(f0)
        bm, bv = tm.bn_batch_stats(f0, stream[0])
        if rm is None or bm is None:
            rich_note = "BN stats unavailable; bn_kl set to 0"
        bn_kl = tm.bn_stat_kl_drift(rm, rv, bm, bv)
    cell_records = []
    preds_all = [p0]
    aa_all = [a0]
    cand_names = ["freeze_f0"]
    for method, mode in candidates:
        cand_out = tm.run_candidate(
            method, mode, f0, stream, eval_x, eval_y, NUM_CLASSES,
            steps, lr, eval_bs=min(bs, 64),
            return_details=(args.evidence_panel == "rich"),
        )
        if args.evidence_panel == "rich":
            aa, Z_base, upd, preds, pa_pos, details = cand_out
            rich = tm.rich_evidence_vector(
                logits_f0_eval, details["logits_eval"], logits_src, y_src, bn_kl,
            )
            Z = list(Z_base) + [float(value) for value in rich]
        else:
            aa, Z, upd, preds, pa_pos = cand_out
            Z_base = Z
            rich = None
        benefit = float(aa - a0)
        record = {
            "cell_id": cell_id,
            "scientific_cell_identity": _cell_spec(seed, domain, comp, regime, aggr),
            "model_seed": int(seed),
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "checkpoint_tensor_sha256": checkpoint_identity["tensor_sha256"],
            "stream_seed": int(sample_seed),
            "seed": int(seed),
            "domain": domain,
            "comp": comp,
            "regime": regime,
            "aggr": aggr,
            "method": method,
            "mode": mode,
            "tta_protocol": tm.tta_protocol_contract(mode),
            "candidate": f"{method}_{mode}",
            "metric": "balanced_accuracy",
            "a0": float(a0),
            "aa": float(aa),
            "B": benefit,
            "upd_norm": float(upd),
            "Z": [float(value) for value in Z],
            "Z_base": [float(value) for value in Z_base],
            "evidence_panel": args.evidence_panel,
            "regime_label": an.label_regime(benefit),
            "sample_provenance": sample_provenance,
            "preds": [int(value) for value in preds],
        }
        if args.evidence_panel == "rich":
            record.update({
                "Z_rich": [float(value) for value in rich],
                "bn_kl": float(bn_kl),
                "rich_note": rich_note,
            })
            try:
                record["Z_ev2"] = pc.ev2_vector(details["logits_eval"])
                record["Z_ev2_names"] = list(pc.EV2_NAMES)
            except Exception as exc:
                record["Z_ev2"] = None
                record["Z_ev2_note"] = repr(exc)
        cell_records.append(record)
        preds_all.append(preds)
        aa_all.append(float(aa))
        cand_names.append(f"{method}_{mode}")
        tm.mps_free()

    preds_mat = np.stack(preds_all, 0)
    pc.attach_to_last(cell_records, len(candidates), pc.panel_fields(preds_mat))
    route_metric_contract = route_b_metric_contract(eval_y, preds_mat, aa_all)
    route = an.multicandidate_route(
        preds_mat,
        tau_star=args.tau_star,
        kappa=args.kappa,
        task_type="binary_classification",
        n_classes=2,
        objective=route_metric_contract["route_objective"],
        anchor_above_chance=bool(args.anchor_above_chance),
    )
    route["metric_contract"] = route_metric_contract
    realized = route_realized(route, aa_all)
    oracle = float(max(aa_all))
    best_adapt = float(max(aa_all[1:]))
    route_c = unsupported_route_c("balanced_accuracy", 2)
    condition = {
        "cell_id": cell_id,
        "scientific_cell_identity": _cell_spec(seed, domain, comp, regime, aggr),
        "model_seed": int(seed),
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_tensor_sha256": checkpoint_identity["tensor_sha256"],
        "stream_seed": int(sample_seed),
        "seed": int(seed),
        "domain": domain,
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
        "route_metric_contract": route_metric_contract,
        "route_c": route_c,
        "realized": realized,
        "route_scorable": realized is not None,
        "regime_label": an.label_regime(best_adapt - a0),
        "sample_provenance": sample_provenance,
        "eval_y": [int(value) for value in eval_y],
        "preds_frozen": [int(value) for value in p0],
    }
    return cell_records, condition


def run(args, partial_path=None):
    t_start = time.time()
    device = tm.pick_device(args.device)
    print(f"[device] {device}")
    checkpoints = _checkpoint_identities(args)
    expected_cell_ids = _expected_cell_ids(args)

    ds, transform, keep_full, n_present, n_total = cd.load_camelyon(args.data_root)
    drop = n_total - n_present
    print(f"[disk-filter] {n_present}/{n_total} patches present "
          f"({drop} dropped, {100.0*drop/max(n_total,1):.2f}%)")
    dom_cache = {
        domain: _complete_domain(ds, transform, keep_full, domain) for domain in args.domains
    }
    source_cache = (
        _complete_domain(ds, transform, keep_full, "train")
        if args.evidence_panel == "rich" else None
    )
    identity_cache = dict(dom_cache)
    if source_cache is not None:
        identity_cache["source_evidence_train"] = source_cache
    population_identity = _population_identity(identity_cache, ds)
    scientific_config = _scientific_config(
        args,
        checkpoints,
        population_identity=population_identity,
        resolved_device=device,
    )
    run_config_sha256 = ri.stable_sha256(scientific_config)

    def validate_partial_semantics(candidate_records, completed_conditions):
        _validate_resume_semantics(
            candidate_records,
            completed_conditions,
            args=args,
            checkpoints=checkpoints,
            dom_cache=dom_cache,
        )

    for domain in args.domains:
        print(f"[domain] {domain}: {len(dom_cache[domain][0])} present patches")
    if source_cache is not None:
        print(f"[source] train: {len(source_cache[0])} present patches for source-calibrated rich evidence")

    records, conditions, failures = [], [], []
    if partial_path and args.resume:
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
    n_cells_total = len(expected_cell_ids)
    candidates = [
        (method, mode) for method, mode in CANDIDATES
        if (not getattr(args, "online_only", False)) or mode == "online"
    ]

    def flush(progress):
        if partial_path:
            payload = ri.partial_document(
                run_config_sha256=run_config_sha256,
                expected_cell_ids=expected_cell_ids,
                records=records,
                conditions=conditions,
                failures=failures,
                progress=progress,
                require_scientific_cell_identity=True,
                semantic_validator=validate_partial_semantics,
            )
            ri.atomic_json_dump(payload, partial_path)

    cell_i = 0
    for seed in args.seeds:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        f0 = load_f0(checkpoints[str(int(seed))]["path"], device)
        logits_src = y_src = None
        if source_cache is not None:
            source_seed = ri.deterministic_seed(
                ri.make_cell_id(dataset="camelyon17", model_seed=int(seed), role="source_evidence")
            )
            source_rng = np.random.default_rng(source_seed)
            _, source_x, y_src, _source_provenance = _build_condition_exact(
                source_cache[0], source_cache[1], "iid", 64,
                min(int(args.n_eval), 1024), source_rng, device, n_batches=1,
            )
            logits_src = tm.predict_logits(f0, source_x, train_mode=False, bs=256)
        for domain in args.domains:
            sub, labels = dom_cache[domain]
            for comp in args.compositions:
                for regime in args.batch_regimes:
                    bs = cd.BATCH_REGIMES[regime]
                    for aggr in args.aggressiveness:
                        cell_i += 1
                        cell_id = _cell_id(seed, domain, comp, regime, aggr)
                        tag = f"s{seed}/{domain}/{comp}/{regime}/{aggr}"
                        if cell_id in done:
                            print(f"  [{cell_i}/{n_cells_total}] {tag} SKIP (resume)", flush=True)
                            continue
                        sample_seed = ri.deterministic_seed(cell_id)
                        try:
                            cell_records, condition = _execute_cell(
                                args=args,
                                f0=f0,
                                sub=sub,
                                labels=labels,
                                seed=seed,
                                domain=domain,
                                comp=comp,
                                regime=regime,
                                aggr=aggr,
                                bs=bs,
                                device=device,
                                candidates=candidates,
                                logits_src=logits_src,
                                y_src=y_src,
                                cell_id=cell_id,
                                sample_seed=sample_seed,
                                checkpoint_identity=checkpoints[str(int(seed))],
                            )
                            _validate_camelyon_completed_cell(
                                condition,
                                cell_records,
                                args=args,
                                checkpoints=checkpoints,
                                dom_cache=dom_cache,
                            )
                        except Exception as exc:
                            ri.upsert_failure(failures, {
                                "cell_id": cell_id,
                                **_cell_spec(seed, domain, comp, regime, aggr),
                                "stream_seed": int(sample_seed),
                                "stage": "cell_execution",
                                "error_type": type(exc).__name__,
                                "error": repr(exc),
                            })
                            print(f"  [{cell_i}/{n_cells_total}] {tag} ERROR: {repr(exc)[:160]}", flush=True)
                            flush(f"{cell_i}/{n_cells_total}")
                            continue
                        records.extend(cell_records)
                        conditions.append(condition)
                        done.add(cell_id)
                        ri.clear_failure(failures, cell_id)
                        route = condition["route"]
                        tau = route.get("tau")
                        tau_text = f"{float(tau):.4f}" if isinstance(tau, (int, float)) else "n/a"
                        print(
                            f"  [{cell_i}/{n_cells_total}] {tag} a0={condition['a0']:.3f} "
                            f"best_aa={condition['best_adapt']:.3f} oracle={condition['oracle']:.3f} "
                            f"route={route.get('decision')} status={route.get('status')} tau={tau_text} "
                            f"sd_c={condition['route_c'].get('decision')}",
                            flush=True,
                        )
                        flush(f"{cell_i}/{n_cells_total}")
        del f0
        tm.mps_free()

    ledger = ri.build_ledger(expected_cell_ids, conditions, failures)
    return records, conditions, {
        "n_present": n_present,
        "n_total": n_total,
        "wall_sec": time.time() - t_start,
        "scientific_config": scientific_config,
        "run_config_sha256": run_config_sha256,
        "checkpoint_identities": checkpoints,
        "population_identity": population_identity,
        "expected_cell_ids": expected_cell_ids,
        "failures": failures,
        "ledger": ledger,
    }


def aggregate_single_candidate(records):
    """(a) per-candidate single-candidate KGA certificate + policy metrics."""
    out = {}
    cands = sorted(set(r["candidate"] for r in records))
    for c in cands:
        rs = [r for r in records if r["candidate"] == c]
        a0 = np.array([r["a0"] for r in rs]); aa = np.array([r["aa"] for r in rs])
        B = aa - a0; Z = np.array([r["Z"] for r in rs])
        entry = {"n_cells": len(rs), "mean_B": float(B.mean()),
                 "base_rate_harmful_B<0": float(np.mean(B < 0)),
                 "mean_acc": {"always_adapt": float(aa.mean()), "always_freeze": float(a0.mean())}}
        if len(rs) >= 2 and len(np.unique(B)) > 1:
            sample_ids = [
                str(
                    record.get("cell_id")
                    or record.get("_cell_key")
                    or (
                        record.get("seed"), record.get("domain"), record.get("split"),
                        record.get("comp"), record.get("regime"), record.get("aggr"),
                        record.get("mode"), record.get("candidate"),
                    )
                )
                for record in rs
            ]
            Bhat, eps, dec = an.decide_kga(Z, B, sample_ids=sample_ids)
            pm = an.policy_metrics(dec, a0, aa, B)
            # fix-queue item 4: eps is one radius per cell (the scored cell is
            # excluded from its own calibration pool), so the scalar that used to
            # be written here no longer exists. Record the summary, labelled.
            eps = np.asarray(eps, float)
            finite_eps = eps[np.isfinite(eps)]
            feasible = bool(len(eps) and len(finite_eps) == len(eps))
            pm["radius_status"] = "FEASIBLE" if feasible else "INFEASIBLE_EXACT_RANK"
            pm["radius_feasible"] = feasible
            pm["eps_conformal"] = float(np.mean(finite_eps)) if feasible else None
            pm["eps_conformal_is"] = "mean of label-disjoint cross-fitted split-conformal radii"
            pm["claim_scope"] = "development/opened-target diagnostic; not held-out confirmation"
            pm["eps_conformal_min"] = float(np.min(finite_eps)) if feasible else None
            pm["eps_conformal_max"] = float(np.max(finite_eps)) if feasible else None
            entry["kga"] = pm
        else:
            entry["kga"] = {"note": "need >=2 cells with B variation for the cross-cell certificate"}
        out[c] = entry
    return out


def aggregate_multicandidate(conditions, alpha=0.10):
    """(b) multi-candidate tau-route metrics + routing breakdown.
    beats_both REQUIRES the pre-registered false-adapt budget FA<=alpha, not regret alone."""
    if not conditions:
        return {"status": "NO_CONDITIONS", "scorable": False, "note": "no conditions"}
    invalid = [
        condition for condition in conditions
        if condition.get("route", {}).get("status") != "OK"
        or condition.get("route", {}).get("scorable") is not True
        or not isinstance(condition.get("realized"), (int, float))
        or not np.isfinite(float(condition["realized"]))
    ]
    if invalid:
        reasons = {}
        for condition in invalid:
            route = condition.get("route") or {}
            key = f"{route.get('status', 'MISSING')}:{route.get('decision', 'MISSING')}"
            reasons[key] = reasons.get(key, 0) + 1
        return {
            "status": "UNSCORABLE_ROUTE_CELLS",
            "scorable": False,
            "n_conditions": len(conditions),
            "n_scorable_conditions": len(conditions) - len(invalid),
            "n_unscorable_conditions": len(invalid),
            "unscorable_reasons": reasons,
            "beats_both_regret_only": None,
            "beats_both": None,
            "note": "No aggregate route score is reported when any expected route cell is invalid.",
        }
    a0 = np.array([c["a0"] for c in conditions])
    oracle = np.array([c["oracle"] for c in conditions])
    realized = np.array([c["realized"] for c in conditions])
    dec = np.array([c["route"].get("decision", "ERROR") for c in conditions])
    tau = np.array([c["route"].get("tau", np.nan) for c in conditions], float)
    adapt = dec == "ADAPT"
    # fixed best always-adapt candidate (max mean aa across conditions)
    names = conditions[0]["cand_names"][1:]
    aa_mat = np.array([c["aa_all"][1:] for c in conditions])     # (cells, K)
    fixed_idx = int(np.argmax(aa_mat.mean(0)))
    fixed_aa = aa_mat[:, fixed_idx]
    breakdown = {}
    for c in conditions:
        ch = c["route"].get("choice")
        nm = c["cand_names"][ch] if ch is not None else c["route"].get("decision", "ERROR")
        breakdown[nm] = breakdown.get(nm, 0) + 1
    by_regime = {}
    for c in conditions:
        g = c["regime_label"]; d = c["route"].get("decision", "ERROR")
        by_regime.setdefault(g, {}).setdefault(d, 0)
        by_regime[g][d] += 1
    by_domain = {}
    for c in conditions:
        d0 = c["domain"]; d = c["route"].get("decision", "ERROR")
        by_domain.setdefault(d0, {}).setdefault(d, 0)
        by_domain[d0][d] += 1
    false_adapt_mask = adapt & (realized <= a0 + 1e-12)
    fa_u = float(np.mean(false_adapt_mask))
    fa_c = float(np.mean((realized <= a0 + 1e-12)[adapt])) if adapt.any() else None
    return {
        "status": "OK",
        "scorable": True,
        "n_conditions": len(conditions),
        "mean_acc": {"router": float(realized.mean()), "always_freeze": float(a0.mean()),
                     "best_fixed_always_adapt": float(fixed_aa.mean()),
                     "per_condition_oracle": float(oracle.mean())},
        "regret_vs_oracle": {"router": float((oracle - realized).mean()),
                             "always_freeze": float((oracle - a0).mean()),
                             "best_fixed_always_adapt": float((oracle - fixed_aa).mean())},
        "coverage": float(np.mean((dec == "ADAPT") | (dec == "FREEZE"))),
        "abstention_rate": float(np.mean(dec == "ABSTAIN")),
        "false_adapt_unconditional": fa_u,
        "false_adapt_conditional": fa_c,
        "false_adapt_definition": "ADAPT and realized benefit <= 0; unconditional rate gates the certificate",
        "mean_tau": float(np.nanmean(tau)) if np.isfinite(tau).any() else None,
        "gate_pass_rate": float(np.mean([bool(c["route"].get("gate_pass", False)) for c in conditions])),
        "fixed_best_candidate": names[fixed_idx],
        "routing_breakdown": breakdown,
        "decisions_by_regime": by_regime,
        "decisions_by_domain": by_domain,
        "alpha_false_adapt_budget": float(alpha),
        "beats_both_regret_only": bool((oracle - realized).mean() < (oracle - a0).mean() - 1e-9 and
                                       (oracle - realized).mean() < (oracle - fixed_aa).mean() - 1e-9),
        "beats_both": bool((oracle - realized).mean() < (oracle - a0).mean() - 1e-9 and
                           (oracle - realized).mean() < (oracle - fixed_aa).mean() - 1e-9 and
                           adapt.any() and fa_u <= alpha),
    }


def kbound_summary(records, conditions, delta=0.05, evidence_names=None):
    """Camelyon17 regime classification + K-Bound gamma_S / gamma_T / tau (debug proxies)."""
    if not records:
        return {
            "classification": "not_available",
            "base_rate_harmful_B<0": None,
            "mean_B": None,
            "detectability_verdict": "not_available",
            "best_single_feature_harm_AUC": None,
            "gamma_S_proxy_indist_advantage": None,
            "gamma_T_proxy_oracle_advantage": None,
            "delta": delta,
            "abs_gammaT_minus_gammaS": None,
            "gamma_gap_within_delta": None,
            "real_data_multicandidate_tau_mean": None,
            "_note": "no completed candidate records",
        }
    B = np.array([r["B"] for r in records])
    base_h = float(np.mean(B < 0)); meanB = float(B.mean())
    det = an.detectability_analysis(records, evidence_names or tm.EVIDENCE_NAMES)
    verdict = det.get("detectability_verdict", "n/a")
    if base_h < 0.10 and meanB > 0:
        klass = "helpful-dominated"
    elif base_h > 0.60:
        klass = "harmful-dominated"
    else:
        klass = "mixed+detectable" if verdict == "detectable" else "mixed+undetectable"
    id_conds = [c for c in conditions if c["domain"] == "id_val"]
    tg_conds = [c for c in conditions if c["domain"] in ("test", "val")]
    gamma_S = float(np.mean([c["best_adapt"] - c["a0"] for c in id_conds])) if id_conds else None
    gamma_T = float(np.mean([c["oracle"] - c["a0"] for c in tg_conds])) if tg_conds else None
    taus = [
        c["route"].get("tau") for c in conditions
        if c["route"].get("status") == "OK"
        and c["route"].get("scorable") is True
        and c["route"].get("tau") is not None
    ]
    return {
        "classification": klass, "base_rate_harmful_B<0": base_h, "mean_B": meanB,
        "detectability_verdict": verdict,
        "best_single_feature_harm_AUC": det.get("best_single_feature_harm_AUC"),
        "gamma_S_proxy_indist_advantage": gamma_S,
        "gamma_T_proxy_oracle_advantage": gamma_T,
        "delta": delta,
        "abs_gammaT_minus_gammaS": (abs(gamma_T - gamma_S) if (gamma_S is not None and gamma_T is not None) else None),
        "gamma_gap_within_delta": (bool(abs(gamma_T - gamma_S) <= delta)
                                   if (gamma_S is not None and gamma_T is not None) else None),
        "real_data_multicandidate_tau_mean": (float(np.mean(taus)) if taus else None),
        "_note": "gamma_S/gamma_T are operational debug-scale proxies (in-dist vs OOD adaptation "
                 "advantage); final defs follow the paper. Classification honors the integrity policy: "
                 "reported from measured B, never tuned to a target.",
    }


def aggregate_smoothdrift(conditions):
    """Return a non-promotional summary; runner Route C is intentionally disabled."""

    routes = [condition.get("route_c") for condition in conditions if condition.get("route_c")]
    objectives = sorted({
        str(route.get("reported_objective"))
        for route in routes
        if route.get("reported_objective") is not None
    })
    return {
        "status": "UNSUPPORTED",
        "implemented": False,
        "scorable": False,
        "n_conditions": len(conditions),
        "reported_objectives": objectives,
        "available_theorem_objective": "binary_brier_score_benefit",
        "target_label_selection_used": False,
        "reason": (
            routes[0]["reason"] if routes else
            "Route C is not executed by natural-shift classification runners."
        ),
        "note": "No Route-C bracket, decision score, or promotional claim is produced.",
    }


REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # the repository root


def build_manifest(args, records, conditions, meta):
    cfg = meta["scientific_config"]
    cfg_sha256 = meta["run_config_sha256"]
    complete = bool(meta["ledger"]["execution_complete"])
    evidence_names = list(tm.EVIDENCE_NAMES)
    if args.evidence_panel == "rich":
        evidence_names += list(tm.RICH_EVIDENCE_NAMES)
    not_computed = {
        "status": "NOT_COMPUTED_INCOMPLETE_RUN",
        "scorable": False,
        "note": "aggregate withheld because the expected/completed/failed ledger is incomplete",
    }
    routing_a = aggregate_single_candidate(records) if complete else not_computed
    routing_b = aggregate_multicandidate(conditions) if complete else not_computed
    routing_c = aggregate_smoothdrift(conditions) if complete else not_computed
    detectability = (
        an.detectability_analysis(records, evidence_names)
        if complete and len(records) >= 4
        else ({"note": "need >=4 records"} if complete else not_computed)
    )
    summary = (
        kbound_summary(records, conditions, delta=args.delta, evidence_names=evidence_names)
        if complete else not_computed
    )
    baselines = {
        "always_freeze_mean_acc": float(np.mean([r["a0"] for r in records])) if records else None,
        "per_candidate_always_adapt_mean_acc": {
            candidate: float(np.mean([r["aa"] for r in records if r["candidate"] == candidate]))
            for candidate in sorted(set(r["candidate"] for r in records))
        },
        "per_condition_oracle_mean_acc": (
            float(np.mean([condition["oracle"] for condition in conditions])) if conditions else None
        ),
    } if complete else not_computed
    return {
        "schema": "kbound_wilds_camelyon17_v0.8",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"node": platform.node(), "platform": platform.platform(),
                 "python": platform.python_version(), "torch": torch.__version__,
                 "mps": bool(torch.backends.mps.is_available())},
        "config": cfg,
        "config_sha256": cfg_sha256,
        "config_sha8": cfg_sha256[:8],
        "run_ledger": meta["ledger"],
        "execution_complete": complete,
        "publication_eligible": False,
        "publication_eligibility_note": (
            "diagnostic run on an opened target; Route-A fits/calibrates on target labels and "
            "no validation-locked held-out confirmation is present"
        ),
        "claim_eligibility": {
            "raw_completed_records": complete,
            "route_a_single_candidate": False,
            "route_b_multicandidate": False,
            "route_c_smooth_drift": False,
        },
        "failures": meta["failures"],
        "evidence_panel": args.evidence_panel,
        "evidence_names": evidence_names,
        "f0_checkpoints": meta["checkpoint_identities"],
        "data": {"data_root": args.data_root, "n_present": meta["n_present"],
                 "n_total": meta["n_total"], "n_dropped_disk_filter": meta["n_total"] - meta["n_present"],
                 "population_manifest": meta["population_identity"],
                 "wall_sec": round(meta["wall_sec"], 1)},
        "candidates": [f"{m}_{md}" for (m, md) in CANDIDATES
                       if (not getattr(args, "online_only", False)) or md == "online"],
        "baselines": baselines,
        "routing_a_single_candidate": routing_a,
        "routing_b_multicandidate": routing_b,
        "routing_c_smooth_drift": routing_c,
        "detectability": detectability,
        "kbound_summary": summary,
        "records": records,
        "conditions": conditions,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="K-Bound natural-shift pipeline on WILDS Camelyon17")
    p.add_argument("--data-root", default=os.path.expanduser("~/kbound_cam/wilds"),
                   help="dir containing camelyon17_v1.0 (internal copy for speed)")
    p.add_argument("--f0-template",
                   default=os.path.join(REPO, "experiments/kbound/results/wilds/f0_seed{seed}.pt"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--domains", nargs="+", default=list(cd.DOMAINS))
    p.add_argument("--compositions", nargs="+", default=list(cd.COMPOSITIONS))
    p.add_argument("--batch-regimes", nargs="+", default=["small"], dest="batch_regimes")
    p.add_argument("--aggressiveness", nargs="+", default=["mild", "aggressive"])
    p.add_argument("--n-eval", type=int, default=256, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=4, dest="n_batches")
    p.add_argument("--tau-star", type=float, default=0.08, dest="tau_star")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--sd-L", type=float, default=0.6, dest="sd_L",
                   help="Theorem-1B drift-smoothness modulus L (variant c)")
    p.add_argument("--evidence-panel", choices=["base", "rich"], default="base",
                   dest="evidence_panel",
                   help="base = legacy 11-dim Z; rich = append Protocol-F drift-aware evidence")
    p.add_argument(
        "--anchor-above-chance",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="anchor_above_chance",
        help=(
            "Declare that an independent source/development evaluation established the frozen "
            "anchor is above binary chance. Route B fails closed unless this premise is explicit."
        ),
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--steps-override", type=int, default=0, dest="steps_override")
    p.add_argument("--out", default="")
    p.add_argument("--run-name", default="wilds_kbound", dest="run_name",
                   help="results subdir name under experiments/kbound/results/")
    p.add_argument("--smoke", action="store_true", help="tiny CPU end-to-end smoke")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip cells already in _partial.json (default: on)")
    p.add_argument("--serialize-per-condition", action=argparse.BooleanOptionalAction,
                   default=True, dest="serialize_per_condition",
                   help="also write per_condition_camelyon17_<method>_seed<S>.json files "
                        "(stress_grid_multiseed schema; default: on)")
    # ---- WIN_HUNT_v5 aggressive-regime wave operating-point overrides (opt-in) ----
    p.add_argument("--adapt-lr", type=float, default=None, dest="adapt_lr",
                   help="WIN_HUNT_v5: absolute adapter LR override for tent/eata/sar (the AGGR "
                        "cell lr is ignored when set). DEFAULT None = per-cell lr (byte-identical). "
                        "The v5 aggressive wave sets 0.004 (= 4x the 1e-3 shared-baseline lr).")
    p.add_argument("--online-only", action="store_true", dest="online_only",
                   help="WIN_HUNT_v5: restrict the candidate pool to the online (no-episodic-reset) "
                        "adapters -- the 'continual' operating point. DEFAULT off (all six "
                        "online+episodic candidates, byte-identical to prior runs).")
    args = p.parse_args(argv)
    if args.smoke:
        args.domains = ["test"]; args.compositions = ["iid", "single_class"]
        args.batch_regimes = ["tiny"]; args.aggressiveness = ["mild"]
        args.seeds = [0, 1]; args.n_eval = 32; args.n_batches = 2
        args.steps_override = 4
        if args.device == "auto":
            args.device = "cpu"
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(REPO, "experiments/kbound/results",
                           "wilds_kbound_smoke" if args.smoke else args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    partial = os.path.join(out_dir, "_partial.json")
    records, conditions, meta = run(args, partial_path=partial)
    manifest = build_manifest(args, records, conditions, meta)
    complete = manifest["execution_complete"]
    prefix = "diagnostic" if complete else "incomplete"
    out = args.out or os.path.join(out_dir, f"{prefix}_{manifest['config_sha8']}.json")
    ri.atomic_json_dump(manifest, out)
    if not complete:
        print(f"\nmanifest -> {out}")
        raise RuntimeError(
            f"run incomplete: {manifest['run_ledger']}; wrote non-promotable artifact {out}"
        )
    # ---- per-condition serialization (stress_grid_multiseed schema) ----------
    if complete and getattr(args, "serialize_per_condition", True) and records:
        methods = sorted({r["method"] for r in records})       # tent, eata, sar
        seeds = [int(s) for s in args.seeds]
        ser = pcs.serialize_run(records, dataset="camelyon17", out_dir=out_dir,
                                seeds=seeds, methods=methods)
        print(f"[serialize] wrote {len(ser['written'])} per-condition files "
              f"(methods={methods}, seeds={seeds}, kga_backend={ser['kga_backend']}) -> {out_dir}")
    ks = manifest["kbound_summary"]; mb = manifest["routing_b_multicandidate"]
    print("\n" + "=" * 70)
    print(f"records={len(records)}  conditions={len(conditions)}  wall={meta['wall_sec']:.1f}s")
    print(f"classification        : {ks['classification']}")
    if ks["mean_B"] is not None:
        print(f"base_rate_harmful B<0 : {ks['base_rate_harmful_B<0']:.3f}   mean_B={ks['mean_B']:+.4f}")
    print(f"detectability verdict : {ks['detectability_verdict']} "
          f"(best harm-AUC={ks['best_single_feature_harm_AUC']})")
    print(f"multicand route       : mean_tau={mb.get('mean_tau')}  "
          f"abstain={mb.get('abstention_rate')}  breakdown={mb.get('routing_breakdown')}")
    print(f"gamma_S={ks['gamma_S_proxy_indist_advantage']}  gamma_T={ks['gamma_T_proxy_oracle_advantage']}  "
          f"|dT-dS|<=delta: {ks['gamma_gap_within_delta']}")
    rc = manifest["routing_c_smooth_drift"]
    print(f"smooth-drift (c)      : implemented={rc.get('implemented')} "
          f"decisions={rc.get('decision_counts')} bracket_cov={rc.get('bracket_coverage_trueB')}")
    print(f"\nmanifest -> {out}")
    try:
        os.unlink(partial)
    except FileNotFoundError:
        pass
    return out


if __name__ == "__main__":
    main()
