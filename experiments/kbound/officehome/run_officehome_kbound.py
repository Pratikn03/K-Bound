"""
run_officehome_kbound.py - K-Bound natural-shift sweep on Office-Home (covariate shift).

PROTOCOL
  frozen f0     : ResNet-50 (ImageNet V2) fine-tuned on Real_World train.
  candidates(14): {tent,eata,sar} x {online,episodic} x {mild,aggressive}  (12, faithful, BN-affine update)
                  + labelshift  (MLLS target-prior logit adjustment, NO param update)
                  + conservative (damped+clipped prior correction, NO param update)
  role=source       -> Real_World/val            (in-distribution; certificate + tau* calibration)
  role=target_val   -> {Art,Clipart,Product}/val (DEV regime scan)
  role=target_test  -> {Art,Clipart,Product}/test(HELD-OUT; evaluated once)
  CONDITION = (domain, split, composition, batch_regime) x seed.
  B = acc(adapted) - acc(frozen) on a class-BALANCED held-out eval pool.
INTEGRITY: every cell run for real; labels only for B/oracle/eval; routers see only Z /
label-free agreements; every number traces to records[]/conditions[] in the manifest.
Resumable: completed (seed,domain,split,comp,regime) cells are skipped via _partial.json.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, platform, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

HERE = os.path.dirname(os.path.abspath(__file__))
WILDS = os.path.join(os.path.dirname(HERE), "wilds")
for p in (HERE, WILDS):
    if p not in sys.path:
        sys.path.insert(0, p)
import tta_methods as tm          # noqa: E402
import analysis as an            # noqa: E402
import run_camelyon17_kbound as rc  # noqa: E402  (route_realized + aggregate_* )
import run_integrity as ri          # noqa: E402  (strict JSON + finite payload guards)
import oh_data as ohd            # noqa: E402
import oh_candidates as ohc      # noqa: E402

NUM_CLASSES = ohd.NUM_CLASSES
BATCH_REGIMES = {"tiny": 8, "small": 16}
AGGR = {"mild": dict(steps=10, lr=1e-3), "aggressive": dict(steps=30, lr=2.5e-3)}
GRAD_METHODS = ("tent", "eata", "sar")
GRAD_MODES = ("online", "episodic")
GRAD_AGGR = ("mild", "aggressive")
PRIOR_CANDS = ("labelshift", "conservative")
RESUME_CONTRACT_SCHEMA = "kbound_officehome_resume_contract_v2"
ROLE_DOMS = {
    "source": [(ohd.SOURCE, "val")],
    "target_val": [(d, "val") for d in ohd.TARGETS],
    "target_test": [(d, "test") for d in ohd.TARGETS],
}


def default_candidates():
    out = [f"{m}_{mo}_{a}" for m in GRAD_METHODS for mo in GRAD_MODES for a in GRAD_AGGR]
    return out + list(PRIOR_CANDS)


def parse_cand(name):
    if name in PRIOR_CANDS:
        return ("prior", name, None, None)
    method, mode, aggr = name.split("_")
    return ("grad", method, mode, aggr)


def load_f0(ckpt, device):
    m = tvm.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_tensor_sha256(path):
    """Hash checkpoint tensor content independently of torch serialization metadata."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {path}")
    digest = hashlib.sha256()
    tensor_count = 0
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {"name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        tensor_count += 1
    if tensor_count == 0:
        raise ValueError(f"checkpoint contains no tensors: {path}")
    return digest.hexdigest()


def _json_sha256(value):
    return ri.stable_sha256(value)


def _sequence_sha256(values):
    return _json_sha256(list(values))


def _officehome_population_manifest(splits, included_domain_splits=None):
    populations = {}
    included = (
        sorted(set((str(domain), str(split)) for domain, split in included_domain_splits))
        if included_domain_splits is not None
        else [
            (domain, split)
            for domain in sorted(splits["splits"])
            for split in sorted(splits["splits"][domain])
        ]
    )
    for domain, split in included:
        if domain not in splits["splits"] or split not in splits["splits"][domain]:
            raise KeyError(f"OfficeHome split manifest lacks requested {domain}/{split}")
        items = splits["splits"][domain][split]
        rows = []
        for path, label in items:
            absolute = Path(path).expanduser().resolve()
            if not absolute.is_file():
                raise FileNotFoundError(
                    f"OfficeHome population member is missing: {absolute}"
                )
            rows.append({
                "sample_id": str(path),
                "label": int(label),
                "bytes": int(absolute.stat().st_size),
                "content_sha256": file_sha256(absolute),
            })
        populations[f"{domain}/{split}"] = {
            "n": int(len(items)),
            "ordered_sample_label_sha256": _json_sha256(
                [[str(path), int(label)] for path, label in items]
            ),
            "ordered_content_manifest_sha256": _json_sha256(rows),
        }
    return {
        "sha256": _json_sha256(populations),
        "identity_fields": ["sample_id", "label", "bytes", "content_sha256"],
        "splits": populations,
    }


def build_resume_contract(args, split_manifest_sha256, checkpoint_file_sha256,
                          checkpoint_tensor_hash, population_manifest,
                          resolved_device=None):
    """Return the immutable scientific identity required to reuse a partial run."""
    if not isinstance(population_manifest, dict) or not population_manifest.get("sha256"):
        raise ValueError("OfficeHome resume contract requires a verified content population manifest")
    resolved_candidates = []
    for name in args.candidates:
        kind, method, mode, aggressiveness = parse_cand(name)
        row = {"name": name, "kind": kind}
        if kind == "grad":
            row.update({
                "method": method,
                "mode": mode,
                "aggressiveness": aggressiveness,
                "steps": int(args.steps_override or AGGR[aggressiveness]["steps"]),
                "lr": float(
                    args.adapt_lr if args.adapt_lr is not None else AGGR[aggressiveness]["lr"]
                ),
            })
        else:
            row["prior_method"] = method
        resolved_candidates.append(row)
    payload = {
        "dataset": "office-home",
        "implementation_sha256": {
            "runner": file_sha256(__file__),
            "tta_methods": file_sha256(tm.__file__),
            "analysis": file_sha256(an.__file__),
            "routing_aggregates": file_sha256(rc.__file__),
            "officehome_data": file_sha256(ohd.__file__),
            "officehome_candidates": file_sha256(ohc.__file__),
        },
        "role": args.role,
        "domain_splits": [[domain, split] for domain, split in ROLE_DOMS[args.role]],
        "candidate_set_ordered": list(args.candidates),
        "candidate_settings_ordered": resolved_candidates,
        "route_b_contract": {
            "objective": "accuracy",
            "n_classes": int(NUM_CLASSES),
            "anchor_above_chance": False,
            "eligibility": "UNSUPPORTED_MULTICLASS",
        },
        "split_manifest": {
            "path": str(Path(args.splits).expanduser().resolve()),
            "sha256": str(split_manifest_sha256),
        },
        "population_manifest": population_manifest,
        "checkpoint": {
            "path": str(Path(args.ckpt).expanduser().resolve()),
            "file_sha256": str(checkpoint_file_sha256),
            "tensor_sha256": str(checkpoint_tensor_hash),
        },
        "seed_semantics": {
            "model_seed": int(args.model_seed),
            "stream_seeds_ordered": [int(seed) for seed in args.seeds],
            "condition_seed_rule": "uint32(sha256('{role}/s{stream_seed}/{domain}/{split}/{composition}/{batch_regime}')[:8]); model-seed invariant",
            "source_reference_seed_rule": "100000 + stream_seed",
        },
        "scientific_config": {
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "num_classes": int(NUM_CLASSES),
            "compositions_ordered": list(args.compositions),
            "batch_regimes_ordered": list(args.batch_regimes),
            "batch_sizes": {name: int(BATCH_REGIMES[name]) for name in args.batch_regimes},
            "n_eval": int(args.n_eval),
            "n_batches": int(args.n_batches),
            "episodic_steps": int(args.episodic_steps),
            "episodic_batch": int(args.episodic_batch),
            "tau_star": float(args.tau_star),
            "kappa": float(args.kappa),
            "route_min_disagreement": 8,
            "smooth_drift_L": float(args.sd_L),
            "route_c_contract": rc.route_c_contract("accuracy", NUM_CLASSES),
            "steps_override": int(args.steps_override),
            "adapt_lr": None if args.adapt_lr is None else float(args.adapt_lr),
            "device_requested": str(args.device),
            "device_resolved": str(resolved_device if resolved_device is not None else args.device),
            "torch_version": str(torch.__version__),
            "skip_ordered": list(args.skip),
            "source_reference_n": 512,
            "source_reference_batch_size": 128,
            "frozen_eval_batch_size": 256,
        },
    }
    return {
        "schema": RESUME_CONTRACT_SCHEMA,
        "sha256": _json_sha256(payload),
        "payload": payload,
    }


def validate_resume_contract(partial_doc, expected, partial_path):
    actual = partial_doc.get("resume_contract")
    if actual is None:
        raise RuntimeError(
            f"refusing legacy resume without {RESUME_CONTRACT_SCHEMA}: {partial_path}; "
            "use --fresh or a new --run-name"
        )
    if actual != expected:
        actual_payload = actual.get("payload", {}) if isinstance(actual, dict) else {}
        expected_payload = expected["payload"]
        fields = sorted(
            key for key in set(actual_payload) | set(expected_payload)
            if actual_payload.get(key) != expected_payload.get(key)
        )
        raise RuntimeError(
            f"refusing mismatched resume at {partial_path}; scientific contract fields differ: "
            f"{fields or ['schema_or_hash']}"
        )


def _officehome_cell_key(contract_sha256, model_seed, seed, role, domain, split, comp, regime):
    return (
        str(contract_sha256), int(model_seed), int(seed), str(role), str(domain),
        str(split), str(comp), str(regime),
    )


def _officehome_scientific_cell_identity(key):
    if len(key) != 8:
        raise ri.RunIntegrityError("OfficeHome scientific cell key must have eight fields")
    contract_sha, model_seed, stream_seed, role, domain, split, comp, regime = key
    return {
        "dataset": "office-home",
        "resume_contract_sha256": str(contract_sha),
        "model_seed": int(model_seed),
        "stream_seed": int(stream_seed),
        "role": str(role),
        "domain": str(domain),
        "split": str(split),
        "composition": str(comp),
        "batch_regime": str(regime),
    }


def _officehome_cell_id(key):
    return ri.make_cell_id(**_officehome_scientific_cell_identity(key))


def _key_token(key):
    return json.dumps(list(key), separators=(",", ":"), default=str)


def _ledger(expected_keys, done, failures, failure_history):
    expected = list(expected_keys)
    expected_set = set(expected)
    completed = sorted(expected_set & set(done), key=_key_token)
    pending = sorted(expected_set - set(done) - set(failures), key=_key_token)
    failed_rows = [failures[key] for key in sorted(failures, key=_key_token)]
    complete = len(completed) == len(expected_set) and not failed_rows
    return {
        "status": "complete" if complete else "incomplete",
        "expected": int(len(expected_set)),
        "completed": int(len(completed)),
        "failed": int(len(failed_rows)),
        "pending": int(len(pending)),
        "expected_keys_sha256": _json_sha256([list(key) for key in sorted(expected_set, key=_key_token)]),
        "completed_keys": [list(key) for key in completed],
        "failed_cells": failed_rows,
        "pending_keys": [list(key) for key in pending],
        "failure_history": list(failure_history),
    }


def _close_float(actual, expected, *, atol=1e-12):
    try:
        return bool(np.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= atol)
    except (TypeError, ValueError):
        return False


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_exact_sample_provenance(
    provenance,
    partial_path,
    *,
    expected_condition_seed,
    expected_stream_n,
    expected_eval_n,
):
    if not isinstance(provenance, dict):
        raise RuntimeError(f"partial condition is missing deterministic sample provenance: {partial_path}")
    pairs = (
        (
            "ordered_stream_requested_split_positions_sha256",
            "ordered_stream_resolved_split_positions_sha256",
        ),
        (
            "ordered_eval_requested_split_positions_sha256",
            "ordered_eval_resolved_split_positions_sha256",
        ),
    )
    if any(not _is_sha256(provenance.get(left)) or provenance.get(left) != provenance.get(right)
           for left, right in pairs):
        raise RuntimeError(f"partial sample requested/resolved identity mismatch: {partial_path}")
    if provenance.get("ordered_stream_split_positions_sha256") != provenance.get(
        "ordered_stream_requested_split_positions_sha256"
    ) or provenance.get("ordered_eval_split_positions_sha256") != provenance.get(
        "ordered_eval_requested_split_positions_sha256"
    ):
        raise RuntimeError(f"partial sample canonical/requested position hashes differ: {partial_path}")
    for field in (
        "ordered_stream_sample_ids_sha256",
        "ordered_eval_sample_ids_sha256",
    ):
        if not _is_sha256(provenance.get(field)):
            raise RuntimeError(f"partial sample provenance field {field!r} is invalid: {partial_path}")
    if any(provenance.get(field) is not True for field in (
        "requested_resolved_identity_equal", "stream_eval_disjoint", "stream_unique", "eval_unique",
    )) or provenance.get("stream_eval_overlap_count") != 0:
        raise RuntimeError(f"partial sample identity/disjointness assertion is invalid: {partial_path}")
    if provenance.get("condition_seed") != int(expected_condition_seed):
        raise RuntimeError(f"partial sample condition seed is inconsistent: {partial_path}")
    if provenance.get("stream_n") != int(expected_stream_n):
        raise RuntimeError(f"partial sample stream size is inconsistent: {partial_path}")
    if provenance.get("eval_n") != int(expected_eval_n):
        raise RuntimeError(f"partial sample evaluation size is inconsistent: {partial_path}")


def _validate_source_reference_provenance(provenance, expected_contract, stream_seed, partial_path):
    if not isinstance(provenance, dict):
        raise RuntimeError(f"partial condition is missing source-reference provenance: {partial_path}")
    scientific = expected_contract.get("payload", {}).get("scientific_config", {})
    populations = expected_contract.get("payload", {}).get("population_manifest", {}).get("splits", {})
    source_population = populations.get(f"{ohd.SOURCE}/train", {})
    expected_n = min(int(scientific["source_reference_n"]), int(source_population["n"]))
    if provenance.get("sampling_seed") != 100000 + int(stream_seed):
        raise RuntimeError(f"partial source-reference seed is inconsistent: {partial_path}")
    if provenance.get("n") != expected_n:
        raise RuntimeError(f"partial source-reference sample count is inconsistent: {partial_path}")
    for field in ("ordered_split_positions_sha256", "ordered_sample_ids_sha256"):
        if not _is_sha256(provenance.get(field)):
            raise RuntimeError(f"partial source-reference field {field!r} is invalid: {partial_path}")


def _officehome_sample_provenance(paths, stream_ids, eval_ids, condition_seed):
    """Build the exact identity receipt shared by execution and resume validation."""

    stream_ids = np.asarray(stream_ids, dtype=int)
    eval_ids = np.asarray(eval_ids, dtype=int)
    overlap = np.intersect1d(stream_ids, eval_ids)
    return {
        "condition_seed": int(condition_seed),
        "sample_id_scheme": "ordered paths in locked OfficeHome split manifest",
        "stream_n": int(len(stream_ids)),
        "eval_n": int(len(eval_ids)),
        "ordered_stream_split_positions_sha256": _sequence_sha256(stream_ids.tolist()),
        "ordered_eval_split_positions_sha256": _sequence_sha256(eval_ids.tolist()),
        "ordered_stream_sample_ids_sha256": _sequence_sha256(
            [str(paths[int(position)]) for position in stream_ids]
        ),
        "ordered_eval_sample_ids_sha256": _sequence_sha256(
            [str(paths[int(position)]) for position in eval_ids]
        ),
        "ordered_stream_requested_split_positions_sha256": _sequence_sha256(
            stream_ids.tolist()
        ),
        "ordered_stream_resolved_split_positions_sha256": _sequence_sha256(
            stream_ids.tolist()
        ),
        "ordered_eval_requested_split_positions_sha256": _sequence_sha256(eval_ids.tolist()),
        "ordered_eval_resolved_split_positions_sha256": _sequence_sha256(eval_ids.tolist()),
        "requested_resolved_identity_equal": True,
        "stream_eval_disjoint": bool(overlap.size == 0),
        "stream_unique": bool(len(np.unique(stream_ids)) == len(stream_ids)),
        "eval_unique": bool(len(np.unique(eval_ids)) == len(eval_ids)),
        "stream_eval_overlap_count": int(overlap.size),
    }


def _officehome_source_reference_provenance(splits, stream_seed, n):
    paths, _labels = ohd.split_items(splits, ohd.SOURCE, "train")
    rng = np.random.default_rng(100000 + int(stream_seed))
    positions = rng.choice(len(paths), min(int(n), len(paths)), replace=False)
    return {
        "sampling_seed": 100000 + int(stream_seed),
        "n": int(len(positions)),
        "ordered_split_positions_sha256": _sequence_sha256(
            [int(position) for position in positions]
        ),
        "ordered_sample_ids_sha256": _sequence_sha256(
            [str(paths[int(position)]) for position in positions]
        ),
    }


def _expected_officehome_resume_samples(key, expected_contract, splits):
    """Recompute a cell's sample receipt and labels from the live split index."""

    if not isinstance(splits, dict):
        raise ri.RunIntegrityError(
            "OfficeHome resume validation requires the current split/index document"
        )
    _contract_sha, _model_seed, stream_seed, role, domain, split, comp, regime = key
    scientific = expected_contract.get("payload", {}).get("scientific_config", {})
    try:
        batch_size = int(scientific["batch_sizes"][regime])
        n_eval = int(scientific["n_eval"])
        n_batches = int(scientific["n_batches"])
        source_n = int(scientific["source_reference_n"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ri.RunIntegrityError("OfficeHome resume contract lacks sampling parameters") from exc
    paths, labels = ohd.split_items(splits, domain, split)
    sampling_tag = f"{role}/s{stream_seed}/{domain}/{split}/{comp}/{regime}"
    condition_seed = int(hashlib.sha256(sampling_tag.encode()).hexdigest()[:8], 16)
    stream_ids, eval_ids = _officehome_condition_indices(
        labels,
        comp,
        batch_size,
        n_eval,
        n_batches,
        np.random.default_rng(condition_seed),
    )
    return {
        "sample_provenance": _officehome_sample_provenance(
            paths, stream_ids, eval_ids, condition_seed
        ),
        "eval_y": np.asarray(labels, dtype=int)[eval_ids],
        "source_reference_provenance": _officehome_source_reference_provenance(
            splits, stream_seed, source_n
        ),
    }


def _officehome_record_inventory(records, conditions):
    by_key = {}
    for record in records:
        key = tuple(record.get("_cell_key", ()))
        by_key.setdefault(key, []).append(record)
    condition_keys = {tuple(condition.get("_key", ())) for condition in conditions}
    orphan_keys = set(by_key) - condition_keys
    if orphan_keys:
        raise ri.RunIntegrityError("OfficeHome inventory contains orphan candidate records")
    inventory = {}
    for condition in conditions:
        key = tuple(condition.get("_key", ()))
        cell_id = condition.get("cell_id")
        if not key or not isinstance(cell_id, str):
            raise ri.RunIntegrityError("OfficeHome inventory requires cell key and cell_id")
        if cell_id in inventory:
            raise ri.RunIntegrityError("OfficeHome inventory contains duplicate cell_id values")
        rows = by_key.get(key, [])
        inventory[cell_id] = {
            "key": list(key),
            "candidate_count": len(rows),
            "candidates": [row.get("candidate") for row in rows],
            "condition_sha256": ri.stable_sha256(condition),
            "records_sha256": ri.stable_sha256(rows),
        }
    return inventory


def _validate_officehome_completed_cell(condition, cell_records, expected_contract,
                                        expected_candidates, partial_path, splits=None):
    key = tuple(condition["_key"])
    if len(key) != 8:
        raise RuntimeError(f"partial OfficeHome cell key has the wrong shape: {partial_path}")
    contract_sha, model_seed, stream_seed, role, domain, split, comp, regime = key
    scientific_identity = _officehome_scientific_cell_identity(key)
    cell_id = _officehome_cell_id(key)
    checkpoint = expected_contract.get("payload", {}).get("checkpoint", {})
    expected_identity = {
        "resume_contract_sha256": contract_sha,
        "checkpoint_tensor_sha256": checkpoint.get("tensor_sha256"),
        "model_seed": model_seed,
        "seed": stream_seed,
        "role": role,
        "domain": domain,
        "split": split,
        "comp": comp,
        "regime": regime,
    }
    if contract_sha != expected_contract.get("sha256"):
        raise RuntimeError(f"partial cell is not bound to the resume contract: {partial_path}")
    ri.validate_scientific_cell_identity(
        condition.get("cell_id"),
        condition.get("scientific_cell_identity"),
        context="OfficeHome completed condition",
    )
    if condition.get("cell_id") != cell_id or condition.get("scientific_cell_identity") != scientific_identity:
        raise RuntimeError(f"partial cell scientific identity is inconsistent: {partial_path}")
    for field, expected in expected_identity.items():
        if condition.get(field) != expected:
            raise RuntimeError(f"partial condition identity field {field!r} mismatches: {partial_path}")
    if len(cell_records) != len(expected_candidates):
        raise RuntimeError(f"partial completed cell has an incomplete candidate transaction: {partial_path}")
    eval_y = np.asarray(condition.get("eval_y"), dtype=int)
    frozen = np.asarray(condition.get("preds_frozen"), dtype=int)
    if eval_y.ndim != 1 or frozen.shape != eval_y.shape or eval_y.size == 0:
        raise RuntimeError(f"partial condition has invalid evaluation labels/predictions: {partial_path}")
    sampling_tag = f"{role}/s{stream_seed}/{domain}/{split}/{comp}/{regime}"
    condition_seed = int(hashlib.sha256(sampling_tag.encode()).hexdigest()[:8], 16)
    scientific = expected_contract.get("payload", {}).get("scientific_config", {})
    expected_stream_n = int(scientific["batch_sizes"][regime]) * int(scientific["n_batches"])
    provenance = condition.get("sample_provenance")
    _validate_exact_sample_provenance(
        provenance,
        partial_path,
        expected_condition_seed=condition_seed,
        expected_stream_n=expected_stream_n,
        expected_eval_n=eval_y.size,
    )
    if eval_y.size > int(scientific["n_eval"]):
        raise RuntimeError(f"partial condition exceeds the configured evaluation cap: {partial_path}")
    _validate_source_reference_provenance(
        condition.get("source_reference_provenance"), expected_contract, stream_seed, partial_path
    )
    if splits is not None:
        expected_samples = _expected_officehome_resume_samples(key, expected_contract, splits)
        if provenance != expected_samples["sample_provenance"]:
            raise RuntimeError(
                f"partial deterministic sample provenance differs from the current "
                f"OfficeHome split/index: {partial_path}"
            )
        if not np.array_equal(eval_y, expected_samples["eval_y"]):
            raise RuntimeError(
                f"partial evaluation labels differ from the current OfficeHome split/index: "
                f"{partial_path}"
            )
        if condition.get("source_reference_provenance") != expected_samples[
            "source_reference_provenance"
        ]:
            raise RuntimeError(
                f"partial source-reference identities differ from the current OfficeHome "
                f"split/index: {partial_path}"
            )
    a0 = _acc(frozen, eval_y)
    if not _close_float(condition.get("a0"), a0):
        raise RuntimeError(f"partial condition frozen score is inconsistent: {partial_path}")

    aa_all = [a0]
    predictions = [frozen]
    for candidate, record in zip(expected_candidates, cell_records):
        if tuple(record.get("_cell_key", ())) != key or record.get("candidate") != candidate:
            raise RuntimeError(f"partial candidate transaction order/identity mismatch: {partial_path}")
        for field, expected in expected_identity.items():
            if record.get(field) != expected:
                raise RuntimeError(f"partial candidate identity field {field!r} mismatches: {partial_path}")
        if record.get("cell_id") != cell_id or record.get("scientific_cell_identity") != scientific_identity:
            raise RuntimeError(f"partial candidate scientific cell identity mismatches: {partial_path}")
        if record.get("sample_provenance") != provenance:
            raise RuntimeError(f"partial candidate sample provenance differs from condition: {partial_path}")
        kind, _method, mode, _aggressiveness = parse_cand(candidate)
        expected_protocol = (
            tm.tta_protocol_contract(mode)
            if kind == "grad"
            else tm.stream_prior_protocol_contract(candidate)
        )
        ri.validate_evidence_record(
            record,
            ohc.EVIDENCE_NAMES_OH,
            expected_tta_protocol=expected_protocol,
            context=f"OfficeHome {cell_id}/{candidate}",
        )
        if kind == "prior" and not _close_float(record.get("upd_norm"), 0.0):
            raise RuntimeError(f"partial prior candidate has nonzero update norm: {partial_path}")
        preds = np.asarray(record.get("preds"), dtype=int)
        if preds.shape != eval_y.shape:
            raise RuntimeError(f"partial candidate prediction length mismatch: {partial_path}")
        aa = _acc(preds, eval_y)
        if record.get("metric") != "accuracy" or not _close_float(record.get("a0"), a0):
            raise RuntimeError(f"partial candidate metric/frozen score mismatch: {partial_path}")
        if not _close_float(record.get("aa"), aa) or not _close_float(record.get("B"), aa - a0):
            raise RuntimeError(f"partial candidate score arithmetic is inconsistent: {partial_path}")
        if record.get("regime_label") != an.label_regime(aa - a0):
            raise RuntimeError(f"partial candidate regime label is inconsistent: {partial_path}")
        aa_all.append(aa)
        predictions.append(preds)

    names = ["freeze_f0", *expected_candidates]
    if condition.get("cand_names") != names:
        raise RuntimeError(f"partial condition has a mismatched candidate set: {partial_path}")
    if len(condition.get("aa_all", [])) != len(aa_all) or any(
        not _close_float(actual, expected)
        for actual, expected in zip(condition.get("aa_all", []), aa_all)
    ):
        raise RuntimeError(f"partial condition score vector is inconsistent: {partial_path}")
    oracle = max(aa_all)
    best_adapt = max(aa_all[1:])
    if not _close_float(condition.get("oracle"), oracle):
        raise RuntimeError(f"partial condition oracle is inconsistent: {partial_path}")
    if not _close_float(condition.get("best_adapt"), best_adapt):
        raise RuntimeError(f"partial condition best-adapt score is inconsistent: {partial_path}")
    if condition.get("true_best") != names[int(np.argmax(aa_all))]:
        raise RuntimeError(f"partial condition true-best label is inconsistent: {partial_path}")
    if condition.get("regime_label") != an.label_regime(best_adapt - a0):
        raise RuntimeError(f"partial condition regime label is inconsistent: {partial_path}")

    expected_route = an.multicandidate_route(
        np.stack(predictions, 0),
        tau_star=float(scientific["tau_star"]),
        kappa=float(scientific["kappa"]),
        objective="accuracy",
        n_classes=NUM_CLASSES,
        anchor_above_chance=False,
    )
    if condition.get("route") != expected_route:
        raise RuntimeError(f"partial Route-B truth/status payload is inconsistent: {partial_path}")
    if condition.get("route_b_eligible") is not False or condition.get("realized") is not None:
        raise RuntimeError(f"partial unsupported Route-B was treated as scorable: {partial_path}")
    if condition.get("route_objective") != {
        "metric": "accuracy", "n_classes": NUM_CLASSES, "anchor_above_chance": False,
    }:
        raise RuntimeError(f"partial Route-B objective contract is inconsistent: {partial_path}")
    rc.validate_unsupported_route_c(condition.get("route_c"), "accuracy", NUM_CLASSES)


def _load_partial(partial_path, expected_contract, expected_keys, candidates, splits=None):
    if not os.path.exists(partial_path):
        return [], [], set(), {}, []
    doc = ri.strict_json_load(partial_path)
    if not ri.finite_tree(doc):
        raise ri.RunIntegrityError(f"partial contains NaN/Infinity: {partial_path}")
    if not isinstance(doc, dict):
        raise ri.RunIntegrityError(f"partial must be a JSON object: {partial_path}")
    if doc.get("schema") != "kbound_officehome_partial_v3":
        raise ri.RunIntegrityError(
            f"refusing legacy OfficeHome partial without bound record inventory: {partial_path}"
        )
    validate_resume_contract(doc, expected_contract, partial_path)
    records = doc.get("records", [])
    conditions = doc.get("conditions", [])
    if not isinstance(records, list) or not isinstance(conditions, list):
        raise ri.RunIntegrityError(f"partial records/conditions must be lists: {partial_path}")
    if conditions and not isinstance(splits, dict):
        raise ri.RunIntegrityError(
            "OfficeHome resume validation requires the current split/index document"
        )
    expected_set = set(expected_keys)
    keys = [tuple(condition.get("_key", ())) for condition in conditions]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise RuntimeError(f"partial has missing or duplicate completed cell keys: {partial_path}")
    if not set(keys).issubset(expected_set):
        raise RuntimeError(f"partial contains cells outside the current expected grid: {partial_path}")
    expected_candidates = list(candidates)
    by_key = {}
    for record in records:
        key = tuple(record.get("_cell_key", ()))
        if key not in set(keys):
            raise RuntimeError(f"partial contains an orphan candidate record: {partial_path}")
        by_key.setdefault(key, []).append(record)
    for key in keys:
        _validate_officehome_completed_cell(
            next(condition for condition in conditions if tuple(condition["_key"]) == key),
            by_key.get(key, []), expected_contract, expected_candidates, partial_path,
            splits=splits,
        )
    ledger_doc = doc.get("ledger", {})
    failures = {
        tuple(row["key"]): row for row in ledger_doc.get("failed_cells", []) if row.get("key")
    }
    if not set(failures).issubset(expected_set):
        raise RuntimeError(f"partial failure ledger contains cells outside the expected grid: {partial_path}")
    if set(failures) & set(keys):
        raise RuntimeError(f"partial marks the same cell completed and failed: {partial_path}")
    failure_history = list(ledger_doc.get("failure_history", []))
    completed_set = set(keys)
    rebuilt_ledger = _ledger(expected_keys, completed_set, failures, failure_history)
    if ledger_doc != rebuilt_ledger:
        raise RuntimeError(f"partial completion ledger is inconsistent: {partial_path}")
    inventory = _officehome_record_inventory(records, conditions)
    if doc.get("record_inventory") != inventory:
        raise RuntimeError(f"partial record inventory commitment is inconsistent: {partial_path}")
    return records, conditions, set(keys), failures, failure_history


def _partial_payload(contract, expected_keys, records, conditions, done, failures,
                     failure_history, elapsed_sec):
    expected_set = set(expected_keys)
    completed_set = {tuple(condition.get("_key", ())) for condition in conditions}
    if completed_set != set(done):
        raise ri.RunIntegrityError("OfficeHome done set does not equal completed condition keys")
    if not completed_set.issubset(expected_set) or not set(failures).issubset(expected_set):
        raise ri.RunIntegrityError("OfficeHome partial contains state outside the expected grid")
    if completed_set & set(failures):
        raise ri.RunIntegrityError("OfficeHome cell cannot be completed and failed")
    expected_candidates = list(contract.get("payload", {}).get("candidate_set_ordered", []))
    by_key = {}
    for record in records:
        by_key.setdefault(tuple(record.get("_cell_key", ())), []).append(record)
    for condition in conditions:
        key = tuple(condition.get("_key", ()))
        _validate_officehome_completed_cell(
            condition,
            by_key.get(key, []),
            contract,
            expected_candidates,
            "in-memory OfficeHome partial",
        )
    return {
        "schema": "kbound_officehome_partial_v3",
        "resume_contract": contract,
        "progress": f"{len(done)}/{len(set(expected_keys))}",
        "elapsed_sec": round(float(elapsed_sec), 1),
        "ledger": _ledger(expected_keys, done, failures, failure_history),
        "record_inventory": _officehome_record_inventory(records, conditions),
        "records": records,
        "conditions": conditions,
    }


def _require_route(route, name, require_choice=False):
    if not isinstance(route, dict):
        raise RuntimeError(f"{name} returned non-dict result: {type(route).__name__}")
    decision = str(route.get("decision", "ERROR")).upper()
    if decision == "ERROR":
        raise RuntimeError(f"{name} failed: {route.get('reason', 'missing decision')}")
    if decision not in {"ADAPT", "FREEZE", "ABSTAIN"}:
        raise RuntimeError(f"{name} returned unknown decision: {route.get('decision')!r}")
    route["decision"] = decision
    if require_choice and decision == "ADAPT" and not isinstance(route.get("choice"), (int, np.integer)):
        raise RuntimeError(f"{name} returned ADAPT without an integer candidate choice")


def _officehome_route_b(preds_all, aa_all, tau_star, kappa):
    """Archive Route B as explicitly unsupported for 65-class OfficeHome accuracy."""
    route = an.multicandidate_route(
        preds_all,
        tau_star=tau_star,
        kappa=kappa,
        objective="accuracy",
        n_classes=NUM_CLASSES,
        anchor_above_chance=False,
    )
    _require_route(route, "multicandidate route", require_choice=True)
    eligible = bool(route.get("scorable") is True)
    realized = rc.route_realized(route, aa_all) if eligible else None
    return route, realized, eligible


def _route_b_summary(conditions):
    eligible = [
        condition for condition in conditions
        if condition.get("route_b_eligible") is True and condition.get("realized") is not None
    ]
    if not eligible:
        return {
            "status": "UNSUPPORTED",
            "scorable": False,
            "reason": "Route B is defined only for binary accuracy; OfficeHome has 65 classes.",
            "n_conditions": int(len(conditions)),
            "n_scorable_conditions": 0,
        }
    return rc.aggregate_multicandidate(eligible)


def _commit_cell(records, conditions, cand_recs, condition):
    key = tuple(condition.get("_key", ()))
    if not key or any(tuple(record.get("_cell_key", ())) != key for record in cand_recs):
        raise ri.RunIntegrityError("staged OfficeHome records do not share the condition key")
    if any(tuple(row.get("_key", ())) == key for row in conditions):
        raise ri.RunIntegrityError(f"duplicate completed OfficeHome cell: {list(key)}")
    staged = {"records": cand_recs, "condition": condition}
    if not ri.finite_tree(staged):
        raise ValueError("OfficeHome cell produced NaN/Infinity and cannot be committed")
    records.extend(cand_recs)
    conditions.append(condition)


def _officehome_condition_indices(y, comp, bs, n_eval, n_batches, rng):
    """Mirror oh_data.build_condition sampling so exact sample hashes can be archived."""
    y = np.asarray(y, int)
    pos_all = np.arange(len(y))
    classes = np.unique(y)
    per = max(1, n_eval // max(1, len(classes)))
    ev = [rng.choice(pos_all[y == cls], min(per, int(np.sum(y == cls))), replace=False)
          for cls in classes if np.any(y == cls)]
    ev = np.concatenate(ev)
    if len(ev) > n_eval:
        ev = rng.choice(ev, n_eval, replace=False)
    rng.shuffle(ev)
    remain = np.setdiff1d(pos_all, ev)
    if len(remain) == 0:
        raise RuntimeError("evaluation pool leaves no disjoint OfficeHome adaptation samples")
    n_stream = max(bs, bs * n_batches)
    if comp == "iid":
        if len(remain) < n_stream:
            raise RuntimeError(f"iid stream needs {n_stream} unique samples; only {len(remain)} remain")
        stream = rng.choice(remain, n_stream, replace=False)
    elif comp == "imbalanced":
        counts = Counter(y[remain].tolist())
        majority = counts.most_common(1)[0][0]
        majority_pool = remain[y[remain] == majority]
        other_pool = remain[y[remain] != majority]
        n_majority = int(round(0.85 * n_stream))
        if len(majority_pool) and len(other_pool):
            if len(majority_pool) < n_majority or len(other_pool) < n_stream - n_majority:
                raise RuntimeError(
                    f"imbalanced stream needs {n_majority}/{n_stream - n_majority} unique "
                    f"majority/other samples; only {len(majority_pool)}/{len(other_pool)} remain"
                )
            stream = np.concatenate([
                rng.choice(majority_pool, n_majority, replace=False),
                rng.choice(other_pool, n_stream - n_majority, replace=False),
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
        stream = rng.choice(pool, n_stream, replace=False)
    else:
        raise ValueError(f"unknown composition: {comp}")
    rng.shuffle(stream)
    if len(np.unique(stream)) != len(stream) or len(np.unique(ev)) != len(ev):
        raise RuntimeError("OfficeHome condition contains duplicate requested identities")
    if np.intersect1d(stream, ev).size:
        raise RuntimeError("OfficeHome adaptation and evaluation identities overlap")
    return np.asarray(stream, int), np.asarray(ev, int)


def _load_officehome_positions(paths, labels, positions, device):
    xs = []
    positions = np.asarray(positions, dtype=int)
    for position in positions:
        path = paths[int(position)]
        try:
            xs.append(ohd._load(path, ohd.eval_transform()))
        except Exception as exc:
            raise RuntimeError(
                f"unreadable requested OfficeHome sample {path!r} at split position "
                f"{int(position)}; sample substitution is forbidden"
            ) from exc
    return torch.stack(xs).to(device), np.asarray(labels, dtype=int)[positions]


def _acc(preds, y):
    return float((np.asarray(preds, int) == np.asarray(y, int)).mean())


def atomic_dump(obj, path):
    ri.atomic_json_dump(obj, path)


def source_reference(splits, f0, device, seed, n=512, bs=128):
    """Logits + labels on an in-distribution source (Real_World train) sample for ATC/energy."""
    paths, y = ohd.split_items(splits, ohd.SOURCE, "train")
    rng = np.random.default_rng(100000 + seed)
    idx = rng.choice(len(paths), min(n, len(paths)), replace=False)
    xs = torch.stack([ohd._load(paths[int(i)], ohd.eval_transform()) for i in idx]).to(device)
    logits = tm.predict_logits(f0, xs, train_mode=False, bs=bs)
    del xs; tm.mps_free()
    provenance = _officehome_source_reference_provenance(splits, seed, n)
    if provenance["ordered_split_positions_sha256"] != _sequence_sha256(
        [int(position) for position in idx]
    ):
        raise RuntimeError("OfficeHome source-reference sampling implementation drift")
    return logits, y[idx].astype(int), provenance


def run(args):
    if args.role == "target_test":
        raise RuntimeError(
            "OfficeHome target_test is disabled before data/model access: no externally verified "
            "pre-opening decision lock and one-shot scoring receipt exist for this target. Use "
            "--role target_val for diagnostic development, or register and independently seal a "
            "new unopened target protocol before enabling target_test."
        )
    device = tm.pick_device(args.device)
    out_dir = os.path.join(args.results_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    partial = os.path.join(out_dir, "_partial.json")
    splits = ohd.load_or_make_splits(args.data_root, args.splits)
    dom_splits = ROLE_DOMS[args.role]
    population_manifest = _officehome_population_manifest(
        splits, [*dom_splits, (ohd.SOURCE, "train")]
    )
    split_manifest_sha = file_sha256(args.splits)
    checkpoint_file_sha = file_sha256(args.ckpt)
    checkpoint_tensor_sha = checkpoint_tensor_sha256(args.ckpt)
    contract = build_resume_contract(
        args, split_manifest_sha, checkpoint_file_sha, checkpoint_tensor_sha,
        population_manifest, device,
    )
    cands = list(args.candidates)
    if not cands or len(cands) != len(set(cands)):
        raise ValueError("candidate set must be non-empty and contain no duplicates")
    for candidate in cands:
        parse_cand(candidate)
    expected_keys = [
        _officehome_cell_key(
            contract["sha256"], args.model_seed, seed, args.role, domain, split, comp, regime
        )
        for seed in args.seeds
        for domain, split in dom_splits
        for comp in args.compositions
        for regime in args.batch_regimes
    ]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("duplicate seeds/compositions/batch regimes create duplicate expected cells")
    n_cells = len(expected_keys)
    records, conditions, done, failures, failure_history = [], [], set(), {}, []
    if os.path.exists(partial) and not args.fresh:
        records, conditions, done, failures, failure_history = _load_partial(
            partial, contract, expected_keys, cands, splits=splits
        )
        print(f"[resume] {len(done)} cells already done", flush=True)

    # Resume identity is checked before loading data into the model/evaluation path.
    sprior = ohd.source_prior(splits)
    f0 = load_f0(args.ckpt, device)
    print(f"[run] role={args.role} cells={n_cells} candidates={len(cands)} device={device}", flush=True)
    t0 = time.time()
    ci = 0
    source_reference_samples = {
        str(int(condition["seed"])): condition["source_reference_provenance"]
        for condition in conditions
    }

    def dump_partial():
        atomic_dump(
            _partial_payload(
                contract, expected_keys, records, conditions, done, failures,
                failure_history, time.time() - t0,
            ),
            partial,
        )

    # Establish/replace the contract-bound artifact before any expensive cell work.
    dump_partial()

    for seed in args.seeds:
        seed_keys = [key for key in expected_keys if key[2] == int(seed)]
        if seed_keys and all(key in done for key in seed_keys):
            if str(int(seed)) not in source_reference_samples:
                raise RuntimeError(f"completed seed {seed} lacks source-reference provenance")
            ci += len(seed_keys)
            continue
        try:
            logits_src, y_src, source_provenance = source_reference(splits, f0, device, seed)
            source_reference_samples[str(int(seed))] = source_provenance
        except Exception as exc:
            for key in expected_keys:
                if key[2] != int(seed) or key in done:
                    continue
                failure = {
                    "key": list(key),
                    "tag": f"m{args.model_seed}/s{seed}",
                    "stage": "source_reference",
                    "error": repr(exc)[:1000],
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                failures[key] = failure
                failure_history.append(failure)
            dump_partial()
            print(f"  [seed {seed}] source reference ERROR: {repr(exc)[:200]}", flush=True)
            continue
        for (dom, split) in dom_splits:
            paths, y = ohd.split_items(splits, dom, split)
            for comp in args.compositions:
                for regime in args.batch_regimes:
                    ci += 1
                    key = _officehome_cell_key(
                        contract["sha256"], args.model_seed, seed, args.role,
                        dom, split, comp, regime,
                    )
                    scientific_cell_identity = _officehome_scientific_cell_identity(key)
                    cell_id = _officehome_cell_id(key)
                    if key in done:
                        continue
                    bs = BATCH_REGIMES[regime]
                    tag = f"m{args.model_seed}/s{seed}/{dom}/{split}/{comp}/{regime}"
                    if tag in set(args.skip):
                        failure = {
                            "key": list(key), "tag": tag, "stage": "explicit_skip",
                            "error": "cell omitted by --skip; incomplete runs are not publishable",
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        failures[key] = failure
                        failure_history.append(failure)
                        dump_partial()
                        print(f"  [{ci}/{n_cells}] {tag} FAILED (--skip)", flush=True)
                        continue
                    sampling_tag = f"{args.role}/s{seed}/{dom}/{split}/{comp}/{regime}"
                    cseed = int(hashlib.sha256(sampling_tag.encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(cseed)
                    id_rng = np.random.default_rng(cseed)
                    torch.manual_seed(cseed)
                    stage = "sample_condition"
                    try:
                        stream_ids, eval_ids = _officehome_condition_indices(
                            y, comp, bs, args.n_eval, args.n_batches, id_rng
                        )
                        stream_x, _stream_y = _load_officehome_positions(
                            paths, y, stream_ids, device
                        )
                        eval_x, eval_y = _load_officehome_positions(paths, y, eval_ids, device)
                        stream = [
                            stream_x[i:i + bs] for i in range(0, len(stream_x), bs)
                        ]
                        overlap = np.intersect1d(stream_ids, eval_ids)
                        if overlap.size:
                            raise RuntimeError("OfficeHome adaptation/evaluation identity overlap")
                        sample_provenance = _officehome_sample_provenance(
                            paths, stream_ids, eval_ids, cseed
                        )
                        stage = "frozen_evaluation"
                        logits_f0_eval = tm.predict_logits(f0, eval_x, train_mode=False, bs=256)
                        p0_eval = ohc._softmax_np(logits_f0_eval)
                        p0_preds = p0_eval.argmax(1)
                        a0 = _acc(p0_preds, eval_y)
                        rm, rv = tm.bn_running_stats(f0)
                        bm, bv = tm.bn_batch_stats(f0, stream[0])
                        bn_kl = tm.bn_stat_kl_drift(rm, rv, bm, bv)

                        preds_all = [p0_preds]; aa_all = [a0]; cand_names = ["freeze_f0"]
                        Zs = []; cand_recs = []
                        for name in cands:
                            stage = f"candidate:{name}"
                            kind, a, b, agg = parse_cand(name)
                            if kind == "grad":
                                steps = args.steps_override or AGGR[agg]["steps"]
                                lr = args.adapt_lr if getattr(args, "adapt_lr", None) is not None else AGGR[agg]["lr"]
                                aa_b, _z, upd, preds, _, det = tm.run_candidate(
                                    a, b, f0, stream, eval_x, eval_y, NUM_CLASSES, steps, lr,
                                    eval_bs=args.episodic_batch, prob_mode="max",
                                    episodic_steps=args.episodic_steps, return_details=True)
                                pa_eval = ohc._softmax_np(det["logits_eval"])
                                tta_protocol = tm.tta_protocol_contract(b)
                            else:
                                preds, pa_eval, _adj, upd = ohc.run_prior_candidate(
                                    a, f0, stream, eval_x, NUM_CLASSES, sprior, eval_bs=args.episodic_batch)
                                tta_protocol = tm.stream_prior_protocol_contract(a)
                            aa = _acc(preds, eval_y)
                            B = float(aa - a0)
                            Z = ohc.full_evidence(p0_eval, pa_eval, logits_f0_eval, logits_src, y_src,
                                                  bn_kl, upd, NUM_CLASSES, sprior)
                            Zs.append(Z)
                            cand_recs.append(dict(
                                _cell_key=list(key), cell_id=cell_id,
                                scientific_cell_identity=scientific_cell_identity,
                                resume_contract_sha256=contract["sha256"],
                                checkpoint_tensor_sha256=checkpoint_tensor_sha,
                                model_seed=int(args.model_seed), seed=int(seed), role=args.role,
                                domain=dom, split=split, comp=comp, regime=regime,
                                candidate=name, tta_protocol=tta_protocol,
                                metric="accuracy", a0=float(a0), aa=float(aa), B=B,
                                aa_bacc=float(aa_b) if kind == "grad" else float(tm.balanced_acc(preds, eval_y)),
                                upd_norm=float(upd), preds=[int(v) for v in preds], sample_provenance=sample_provenance,
                                regime_label=an.label_regime(B)))
                            preds_all.append(preds); aa_all.append(float(aa)); cand_names.append(name)
                            tm.mps_free(); gc.collect()

                        # 2nd pass: cross-candidate disagreement (label-free) -> append to Z
                        adapted = np.stack(preds_all[1:], 0)  # (K, N)
                        K = adapted.shape[0]
                        for i in range(K):
                            dis = [float((adapted[i] != adapted[j]).mean()) for j in range(K) if j != i]
                            z = list(Zs[i]) + [float(np.mean(dis)) if dis else 0.0]
                            cand_recs[i]["Z"] = [float(v) for v in z]

                        stage = "route_b"
                        route, realized, route_b_eligible = _officehome_route_b(
                            np.stack(preds_all, 0), aa_all, args.tau_star, args.kappa
                        )
                        if route_b_eligible and route.get("decision") == "ADAPT" and not (
                            1 <= int(route["choice"]) < len(aa_all)
                        ):
                            raise RuntimeError("multicandidate route choice is out of range")
                        oracle = float(max(aa_all)); best_adapt = float(max(aa_all[1:]))
                        stage = "route_c"
                        route_c = rc.unsupported_route_c("accuracy", NUM_CLASSES)
                        condition = dict(
                            _key=list(key), cell_id=cell_id,
                            scientific_cell_identity=scientific_cell_identity,
                            checkpoint_tensor_sha256=checkpoint_tensor_sha,
                            model_seed=int(args.model_seed), seed=int(seed), role=args.role,
                            domain=dom, split=split, comp=comp, regime=regime,
                            resume_contract_sha256=contract["sha256"], sample_provenance=sample_provenance,
                            source_reference_provenance=source_provenance,
                            cand_names=cand_names, aa_all=[float(v) for v in aa_all], a0=float(a0),
                            metric="accuracy", oracle=oracle, best_adapt=best_adapt,
                            true_best=cand_names[int(np.argmax(aa_all))], route=route, route_c=route_c,
                            realized=realized, route_b_eligible=route_b_eligible,
                            route_objective={"metric": "accuracy", "n_classes": NUM_CLASSES,
                                             "anchor_above_chance": False},
                            eval_y=[int(v) for v in eval_y],
                            preds_frozen=[int(v) for v in p0_preds],
                            regime_label=an.label_regime(best_adapt - a0))
                        stage = "commit"
                        _commit_cell(records, conditions, cand_recs, condition)
                        done.add(key)
                        failures.pop(key, None)
                        print(f"  [{ci}/{n_cells}] {tag} a0={a0:.3f} best_adapt={best_adapt:.3f} "
                              f"oracle={oracle:.3f} route={route.get('decision')} "
                              f"tau={route.get('tau', float('nan')):.3f} ({time.time()-t0:.0f}s)", flush=True)
                    except Exception as exc:
                        failure = {
                            "key": list(key), "tag": tag, "stage": stage,
                            "error": repr(exc)[:1000],
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        failures[key] = failure
                        failure_history.append(failure)
                        print(f"  [{ci}/{n_cells}] {tag} ERROR[{stage}]: {repr(exc)[:200]}", flush=True)
                    finally:
                        dump_partial()
                        for nm in ("stream", "eval_x"):
                            if nm in locals():
                                del locals()[nm]
                        gc.collect(); tm.mps_free()
    key_order = {key: index for index, key in enumerate(expected_keys)}
    candidate_order = {name: index for index, name in enumerate(cands)}
    conditions.sort(key=lambda row: key_order[tuple(row["_key"])])
    records.sort(key=lambda row: (
        key_order[tuple(row["_cell_key"])], candidate_order[row["candidate"]]
    ))
    ledger = _ledger(expected_keys, done, failures, failure_history)
    dump_partial()
    meta = {
        "wall_sec": time.time() - t0,
        "n_cells": n_cells,
        "ledger": ledger,
        "resume_contract": contract,
        "split_manifest": {
            "path": str(Path(args.splits).expanduser().resolve()),
            "sha256": split_manifest_sha,
        },
        "population_manifest": population_manifest,
        "source_reference_samples": source_reference_samples,
    }
    if ledger["status"] != "complete":
        raise RuntimeError(
            f"OfficeHome run incomplete: completed={ledger['completed']}/{ledger['expected']} "
            f"failed={ledger['failed']} pending={ledger['pending']}; partial ledger: {partial}"
        )
    return records, conditions, meta


def build_manifest(args, records, conditions, meta):
    ledger = meta.get("ledger", {})
    if ledger.get("status") != "complete":
        raise RuntimeError("refusing to build an OfficeHome result manifest from an incomplete ledger")
    if len(conditions) != ledger["expected"]:
        raise RuntimeError("completed OfficeHome condition count does not match the expected ledger")
    cfg = {k: getattr(args, k) for k in vars(args)}
    contract = meta["resume_contract"]
    sha = contract["sha256"][:8]
    ev_names = list(ohc.EVIDENCE_NAMES_OH)
    routing_a = rc.aggregate_single_candidate(records)
    routing_b = _route_b_summary(conditions)
    manifest = {
        "schema": "kbound_officehome_v4",
        "dataset": "office-home", "role": args.role, "metric": "accuracy",
        "execution_complete": True,
        "publication_eligible": False,
        "publication_eligibility_note": (
            "diagnostic opened-target run; within-partition Route-A labels and stream seeds "
            "cannot establish held-out independent-model confirmation"
        ),
        "claim_eligibility": {
            "raw_completed_records": True,
            "route_a_single_candidate": False,
            "route_b_multicandidate": False,
            "route_c_smooth_drift": False,
            "independent_model_ci": False,
        },
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"node": platform.node(), "platform": platform.platform(),
                 "python": platform.python_version(), "torch": torch.__version__,
                 "mps": bool(torch.backends.mps.is_available())},
        "config": cfg, "config_sha8": sha, "num_classes": NUM_CLASSES,
        "resume_contract": contract,
        "completion_ledger": ledger,
        "evidence_names": ev_names, "candidates": args.candidates,
        "f0_checkpoint": args.ckpt,
        "f0_checkpoint_sha256": contract["payload"]["checkpoint"]["file_sha256"],
        "f0_checkpoint_tensor_sha256": contract["payload"]["checkpoint"]["tensor_sha256"],
        "model_seed": int(args.model_seed),
        "split_manifest": meta["split_manifest"],
        "population_manifest": meta["population_manifest"],
        "source_reference_samples": meta["source_reference_samples"],
        "baselines": {
            "always_freeze_mean_acc": float(np.mean([r["a0"] for r in records])) if records else None,
            "per_candidate_always_adapt_mean_acc": {
                c: float(np.mean([r["aa"] for r in records if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in records))},
            "per_condition_oracle_mean_acc": float(np.mean([c["oracle"] for c in conditions])) if conditions else None,
        },
        "routing_a_single_candidate": routing_a,
        "routing_b_multicandidate": routing_b,
        "routing_c_smooth_drift": rc.aggregate_smoothdrift(conditions),
        "detectability": an.detectability_analysis(records, ev_names) if len(records) >= 4 else {"note": "need>=4"},
        "wall_sec": round(meta["wall_sec"], 1),
        "records": records, "conditions": conditions,
    }
    if not ri.finite_tree(manifest):
        raise ValueError("OfficeHome manifest contains NaN/Infinity and cannot be published")
    return manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(os.path.expanduser("~/kbound_officehome")))
    p.add_argument("--results-root", default=os.path.join(os.path.dirname(HERE), "results"))
    p.add_argument("--splits", default=os.path.join(os.path.dirname(HERE), "results/officehome_kbound/splits.json"))
    p.add_argument("--ckpt", default=os.path.join(os.path.dirname(HERE), "results/officehome_f0/f0_resnet50_rw_seed0.pt"))
    p.add_argument("--model-seed", type=int, default=0, dest="model_seed",
                   help="independent source-checkpoint seed; distinct from stream --seeds")
    p.add_argument("--run-name", default="officehome_kbound_run")
    p.add_argument("--role", choices=list(ROLE_DOMS), required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--compositions", nargs="+", default=["iid", "imbalanced", "single_class"])
    p.add_argument("--batch-regimes", nargs="+", default=["tiny", "small"], dest="batch_regimes")
    p.add_argument("--candidates", nargs="+", default=default_candidates())
    p.add_argument("--n-eval", type=int, default=325, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=3, dest="n_batches")
    p.add_argument("--episodic-steps", type=int, default=5, dest="episodic_steps")
    p.add_argument("--episodic-batch", type=int, default=64, dest="episodic_batch")
    p.add_argument("--tau-star", type=float, default=0.52, dest="tau_star")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument("--sd-L", type=float, default=0.6, dest="sd_L")
    p.add_argument("--steps-override", type=int, default=0, dest="steps_override")
    p.add_argument("--device", default="mps")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--skip", nargs="*", default=[], help="condition tags s{seed}/{dom}/{split}/{comp}/{regime} to skip (e.g. MPS-wedging cells)")
    p.add_argument("--smoke", action="store_true")
    # ---- WIN_HUNT_v5: absolute adapter LR override (enters config hash via vars(args)) ----
    p.add_argument("--adapt-lr", type=float, default=None, dest="adapt_lr",
                   help="WIN_HUNT_v5: absolute adapter LR override for tent/eata/sar grad candidates "
                        "(AGGR cell lr ignored when set). DEFAULT None = per-cell lr (byte-identical). "
                        "v5 aggressive wave sets 0.004 (= 4x the 1e-3 shared baseline). The 'continual' "
                        "no-reset op-point is selected via --candidates tent_online_aggressive "
                        "eata_online_aggressive sar_online_aggressive.")
    args = p.parse_args(argv)
    if args.smoke:
        args.seeds = [0]; args.compositions = ["iid", "single_class"]; args.batch_regimes = ["tiny"]
        args.n_eval = 65; args.n_batches = 2; args.episodic_steps = 2
    return args


def main(argv=None):
    args = parse_args(argv)
    records, conditions, meta = run(args)
    manifest = build_manifest(args, records, conditions, meta)
    out_dir = os.path.join(args.results_root, args.run_name)
    out = os.path.join(out_dir, f"result_{args.role}_{manifest['config_sha8']}.json")
    atomic_dump(manifest, out)
    print(f"\n{'='*72}\nrole={args.role} records={len(records)} conditions={len(conditions)} "
          f"wall={meta['wall_sec']:.0f}s\nmanifest -> {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
