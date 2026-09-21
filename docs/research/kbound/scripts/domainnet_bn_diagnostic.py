"""Approved four-state, inference-only diagnostic on 512 opened painting images.

New outputs only. No optimizer, adaptation, calibration, recipe search or
automatic promotion. The run command requires a hash-bound independent review.
"""

import argparse
import hashlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

from domainnet_feasibility_contract import digest, file_identity, read_bytes, read_json, write_json

ARMS = ("source", "adapted", "adapted_weights_source_bn", "source_weights_adapted_bn")
V4_SHA = "b48c582d22b5e69eba9c83c3a16004044fd2b1931481a6e9a334284353f9c66d"
COMPLETION_SHA = "241ee97a637bf94af60eab2212a8169b3c858181d2b5de3c441848da52d9bb0c"
INVENTORY_SHA = "deda2780d77590ddd21d835166b9f4a1accc03f402429713d9b2cd12ebe52faa"
BUDGET = {"wall_seconds": 600, "rss_bytes": 4 * 1024**3, "disk_bytes": 128 * 1024**2}


def _check_state(model, state):
    import torch

    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("state keys differ")
    for k, v in state.items():
        if (
            type(v) is not torch.Tensor
            or v.device.type != "cpu"
            or v.shape != expected[k].shape
            or v.dtype != expected[k].dtype
            or not torch.isfinite(v).all()
        ):
            raise ValueError("invalid state tensor: " + k)
        if k.endswith(("running_var", "num_batches_tracked")) and (v < 0).any():
            raise ValueError("negative BN state")


def counterfactual_state(model, parameters_from, statistics_from):
    """Swap real BN buffers only; affine parameters remain with learned weights."""
    import torch

    _check_state(model, parameters_from)
    _check_state(model, statistics_from)
    keys = set()
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                if getattr(module, name) is not None:
                    keys.add((prefix + "." if prefix else "") + name)
    if not keys:
        raise ValueError("model contains no tracked BN statistics")
    return {k: (statistics_from if k in keys else parameters_from)[k].clone() for k in parameters_from}


def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(json.dumps([k, str(v.dtype), list(v.shape)]).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def predict_four(model, source, adapted, batch_factory, count):
    import torch

    pairs = ((source, source), (adapted, adapted), (adapted, source), (source, adapted))
    result = {}
    original = (state_hash(source), state_hash(adapted))
    for arm, (parameters, statistics) in zip(ARMS, pairs):
        state = counterfactual_state(model, parameters, statistics)
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        before, rng = state_hash(model.state_dict()), torch.get_rng_state().clone()
        predictions = []
        with torch.inference_mode():
            for batch in batch_factory():
                if batch.device.type != "cpu" or not 0 < len(batch) <= 4:
                    raise ValueError("CPU batch4 ceiling required")
                logits = model(batch)
                if logits.ndim != 2 or len(logits) != len(batch) or not torch.isfinite(logits).all():
                    raise ValueError("nonfinite or malformed model output")
                predictions.extend(logits.argmax(1).tolist())
        if state_hash(model.state_dict()) != before:
            raise ValueError("inference state changed")
        if not torch.equal(rng, torch.get_rng_state()):
            raise ValueError("inference consumed Torch RNG")
        if len(predictions) != count:
            raise ValueError("incomplete prediction count")
        if (state_hash(source), state_hash(adapted)) != original:
            raise ValueError("input states changed")
        result[arm] = {"predictions": predictions, "state_sha256": before}
        print("ARM_COMPLETE", arm, count, flush=True)
    return result


def select_entries(inventory, ids):
    if len(ids) != 512 or len(set(ids)) != 512:
        raise ValueError("exactly 512 unique approved IDs required")
    matches = [e for e in inventory["entries"] if e["image_id"] in set(ids)]
    if len(matches) != 512 or any(e["episode"] != 0 or e["role"] != "evaluation" for e in matches):
        raise ValueError("wrong or missing episode/evaluation image")
    by_id = {e["image_id"]: e for e in matches}
    if len(by_id) != 512:
        raise ValueError("duplicate inventory entries")
    return [by_id[name] for name in ids]


def score_bound(path, sha, truth_loader):
    """Verify persisted predictions before accessing this diagnostic's truth."""
    p = read_json(path, sha)
    ids, arms = p["sample_ids"], p["arms"]
    if len(set(ids)) != len(ids) or not ids or set(arms) != set(ARMS):
        raise ValueError("malformed scoring packet")
    for a in arms.values():
        if len(a["predictions"]) != len(ids) or any(type(x) is not int or not 0 <= x < 126 for x in a["predictions"]):
            raise ValueError("invalid scoring predictions")
    truth = truth_loader()
    if any(name not in truth or type(truth[name]) is not int or not 0 <= truth[name] < 126 for name in ids):
        raise ValueError("missing/invalid truth")
    frozen = [int(v == truth[k]) for k, v in zip(ids, arms["source"]["predictions"])]
    scores = {}
    for arm, row in arms.items():
        correct = [int(v == truth[k]) for k, v in zip(ids, row["predictions"])]
        delta = [a - b for a, b in zip(correct, frozen)]
        counts = Counter(row["predictions"])
        scores[arm] = {
            "n": len(ids),
            "correct": sum(correct),
            "accuracy": sum(correct) / len(ids),
            "benefit_vs_source": sum(delta) / len(ids),
            "improvements": delta.count(1),
            "degradations": delta.count(-1),
            "ties": delta.count(0),
            "predicted_classes": len(counts),
            "largest_class_share": max(counts.values()) / len(ids),
        }
    return scores


def original_inputs(manifest_path, run_root):
    from domainnet_feasibility_images import validate_inventory

    base = read_json(manifest_path, V4_SHA)
    root = Path(run_root)
    if str(root) != base["output_root"]:
        raise ValueError("original output root mismatch")
    for name, bound in base["code"].items():
        if file_identity(Path(__file__).parent / name) != bound:
            raise ValueError("accepted original implementation changed: " + name)
    completed = read_json(root / "collection-complete.json", COMPLETION_SHA)
    inventory = read_json(root / "images/inventory.json", INVENTORY_SHA)
    validate_inventory(base, inventory)
    if Path(inventory["root"]) != root / "images":
        raise ValueError("staged root mismatch")
    episode = completed["episodes"][0]
    if episode["episode"] != 0:
        raise ValueError("wrong original episode")
    files = {x["path"]: {"sha256": x["sha256"], "bytes": x["bytes"]} for x in episode["files"]}
    packet = read_json(root / "episode-0/predictions.json", files["episode-0/predictions.json"]["sha256"])
    if len(packet["sample_ids"]) != 4096 or not packet["reload_predictions_identical"]:
        raise ValueError("original predictions not complete")
    ids = packet["sample_ids"][:512]
    entries = select_entries(inventory, ids)
    return base, inventory, packet, entries, files


def prepare(original_manifest, root, out, destination):
    base, _, packet, entries, _ = original_inputs(original_manifest, root)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("new output directory must be absent")
    code = {**base["code"], Path(__file__).name: file_identity(__file__)}
    m = {
        "schema": "painting-bn-diagnostic-v1",
        "original_manifest": str(Path(original_manifest).absolute()),
        "original_root": str(Path(root).absolute()),
        "output_root": str(destination),
        "code": code,
        "runtime": base["runtime"],
        "budget": BUDGET,
        "sample_ids": packet["sample_ids"][:512],
        "entries_digest": digest(entries),
        "arms": list(ARMS),
        "episode": 0,
        "batch_size": 4,
        "torch_threads": 1,
        "prospective": False,
        "training": False,
        "kga_routing_evaluated": False,
        "purpose": "exploratory saved-state mechanism diagnostic; no automatic recipe selection",
    }
    write_json(out, m)
    return file_identity(out)


def authorize(manifest_path, manifest_sha, review_path, review_sha):
    m, review = read_json(manifest_path, manifest_sha), read_json(review_path, review_sha)
    if (
        review.get("schema") != "painting-bn-diagnostic-review-v1"
        or review.get("accepted_for_launch") is not True
        or review.get("manifest_sha256") != manifest_sha
        or review.get("code") != m["code"]
        or not review.get("reviewer")
    ):
        raise ValueError("independent diagnostic acceptance required")
    if (
        m["schema"] != "painting-bn-diagnostic-v1"
        or m["budget"] != BUDGET
        or m["arms"] != list(ARMS)
        or m["episode"] != 0
        or m["batch_size"] != 4
        or m["torch_threads"] != 1
        or m["prospective"] is not False
        or m["training"] is not False
        or m["kga_routing_evaluated"] is not False
    ):
        raise ValueError("unapproved diagnostic settings")
    base, inventory, packet, entries, files = original_inputs(m["original_manifest"], m["original_root"])
    expected_code = {**base["code"], Path(__file__).name: file_identity(__file__)}
    if (
        m["code"] != expected_code
        or m["sample_ids"] != packet["sample_ids"][:512]
        or m["entries_digest"] != digest(entries)
    ):
        raise ValueError("diagnostic input or code identity changed")
    from domainnet_feasibility_runner import validate_runtime

    validate_runtime(m["runtime"])
    return m, base, inventory, packet, entries, files


def check_supervisor(manifest, args):
    """A worker must belong to the supervising invocation, not a standalone CLI."""
    receipt = json.loads(read_bytes(Path(manifest["output_root"]) / "authorization.json"))
    if (
        receipt.get("supervisor_pid") != os.getppid()
        or receipt.get("manifest_sha256") != args.manifest_sha
        or receipt.get("review_sha256") != args.review_sha
        or receipt.get("code") != manifest["code"]
    ):
        raise ValueError("worker requires matching live supervisor authorization")


def worker(args):
    m, base, inventory, old, entries, files = authorize(args.manifest, args.manifest_sha, args.review, args.review_sha)
    check_supervisor(m, args)
    import torch
    from domainnet_feasibility_images import ImageView
    from domainnet_feasibility_runner import inference_batches, warning_receipt
    from domainnet_reference_source import load_clipart2020, read_verified_checkpoint

    torch.set_num_threads(1)
    output = Path(m["output_root"])
    warnings = []
    try:
        with warning_receipt(warnings):
            from domainnet_reference_adapter import load_reference

            model, source_receipt = load_clipart2020(base["checkpoint_path"])
            source = {k: v.clone() for k, v in model.state_dict().items()}
            bound = files["episode-0/inference.pt"]
            buf = read_verified_checkpoint(
                Path(m["original_root"]) / "episode-0/inference.pt", bound["sha256"], bound["bytes"]
            )
            saved = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=True)
            adapted = saved["classifier"]
            del buf, saved
            transform = load_reference(base["reference_root"]).get_augmentation("test")
            view = ImageView({"root": inventory["root"], "entries": entries}, 0, "evaluation")
            try:
                arms = predict_four(model, source, adapted, lambda: inference_batches(view, transform), 512)
            finally:
                view.close()
            if (
                arms["source"]["predictions"] != old["frozen"][:512]
                or arms["adapted"]["predictions"] != old["candidate_direct"][:512]
            ):
                raise ValueError("baseline saved-output replay mismatch")
            p = {
                "sample_ids": m["sample_ids"],
                "arms": arms,
                "manifest_sha256": args.manifest_sha,
                "entries_digest": m["entries_digest"],
                "source_receipt": source_receipt,
            }
            write_json(output / "predictions.json", p)
            identity = file_identity(output / "predictions.json")
            write_json(output / "predictions-binding.json", identity)
            # All four predictions are now closed before the first truth read.
            from domainnet_feasibility_score import parse_authenticated_truth

            def truth():
                data = read_bytes(base["source_list"]["path"])
                if len(data) != base["source_list"]["bytes"]:
                    raise ValueError("truth size changed")
                return parse_authenticated_truth(data)

            scores = score_bound(output / "predictions.json", identity["sha256"], truth)
            write_json(
                output / "results.json",
                {
                    "schema": "painting-bn-diagnostic-result-v1",
                    "status": "COMPLETED_DIAGNOSTIC_NOT_RECIPE_SELECTION",
                    "scores": scores,
                    "prediction_identity": identity,
                    "manifest_sha256": args.manifest_sha,
                    "prospective": False,
                    "training": False,
                    "kga_routing_evaluated": False,
                    "scope": "first512 opened episode0 images; one checkpoint; no confidence/coverage claim",
                    "baseline_replay_identical": True,
                    "warnings": warnings,
                },
            )
    finally:
        write_json(output / "warnings.json", warnings)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("original-manifest", "original-root", "out", "output-root"):
        prep.add_argument("--" + name, required=True, type=Path)
    for name in ("run", "_worker"):
        command = sub.add_parser(name)
        for arg in ("manifest", "review"):
            command.add_argument("--" + arg, required=True, type=Path)
            command.add_argument("--" + arg + "-sha", required=True)
    args = p.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.original_manifest, args.original_root, args.out, args.output_root)))
        return 0
    if args.command == "_worker":
        worker(args)
        return 0
    m, *_ = authorize(args.manifest, args.manifest_sha, args.review, args.review_sha)
    from domainnet_feasibility_images import fresh_directory
    from domainnet_feasibility_runner import run_bounded

    root = fresh_directory(m["output_root"])
    write_json(
        root / "authorization.json",
        {
            "manifest_sha256": args.manifest_sha,
            "review_sha256": args.review_sha,
            "code": m["code"],
            "runtime": m["runtime"],
            "supervisor_pid": os.getpid(),
        },
    )
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).absolute()),
        "_worker",
        "--manifest",
        str(args.manifest.absolute()),
        "--manifest-sha",
        args.manifest_sha,
        "--review",
        str(args.review.absolute()),
        "--review-sha",
        args.review_sha,
    ]
    receipt = run_bounded(command, root, root, m["budget"])
    write_json(root / "process.json", receipt)
    if receipt["status"] != "PASS":
        write_json(root / "failure.json", receipt)
        print(json.dumps(receipt))
        return 1
    identities = {
        name: file_identity(root / name)
        for name in (
            "predictions.json",
            "predictions-binding.json",
            "results.json",
            "warnings.json",
            "worker.log",
            "process.json",
        )
    }
    write_json(root / "complete.json", {"status": "COMPLETED_NOT_INDEPENDENTLY_VERIFIED", "files": identities})
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
