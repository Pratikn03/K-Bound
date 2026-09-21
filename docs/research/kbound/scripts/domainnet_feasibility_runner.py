"""Fixed, approved painting development executor. No KGA or target-label fitting.

Collection and scoring run in different processes. Only the exact manifest,
code/runtime identities and independent review may authorize real payload IO.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import os
import platform
import signal
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

from domainnet_feasibility_contract import (
    APPROVAL_SHA,
    BUDGET,
    CLASS_MAP_SHA256,
    LIST_SHA,
    PILOT_SHA,
    PROPOSAL_SHA,
    SCOPE,
    build_manifest,
    digest,
    epoch_orders,
    file_identity,
    implementation,
    read_approval,
    read_bytes,
    read_json,
    validate_manifest,
    write_bytes,
    write_json,
)
from domainnet_feasibility_images import (
    ImageView,
    directory_metadata,
    fresh_directory,
    stage_selected,
    validate_inventory,
)
from domainnet_pilot_images import authenticated_archive
from domainnet_reference_source import RELEASE_BYTES, RELEASE_SHA256, read_verified_checkpoint

KNOWN_WARNING = "`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`."


@contextmanager
def warning_receipt(records):
    """Retain known API deprecation; all other warnings abort, visibly."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")

        def emit(message, category, filename, lineno, file=None, line=None):
            known = category is FutureWarning and str(message) == KNOWN_WARNING
            records.append(
                {
                    "category": category.__name__,
                    "message": str(message),
                    "filename": str(filename),
                    "line": lineno,
                    "disposition": "retained_known_deprecation" if known else "STOP_WARNING",
                }
            )
            print(warnings.formatwarning(message, category, filename, lineno), file=sys.stderr, flush=True)
            if not known:
                raise category(str(message))

        warnings.showwarning = emit
        yield


def runtime_identity():
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "pillow", "PyYAML", "psutil")
        },
        "platform": platform.platform(),
        "device": "cpu",
    }


def validate_runtime(expected):
    actual = runtime_identity()
    if (
        actual != expected
        or not actual["python"].startswith("3.12.")
        or actual["packages"]["torch"] != "2.8.0"
        or actual["packages"]["torchvision"] != "0.23.0"
    ):
        raise ValueError("approved experiment runtime mismatch; no release-runtime promotion")
    return actual


def validate_review(review, manifest_sha, manifest):
    if (
        review.get("schema") != "painting-feasibility-independent-review-v1"
        or review.get("accepted_for_launch") is not True
        or not review.get("reviewer_id")
        or review.get("manifest_sha256") != manifest_sha
        or review.get("code") != manifest["code"]
        or review.get("sampling_interpretation") != "conditional_design_diagnostic_not_prospective"
        or review.get("outcome_exclusion_reviewed") is not True
    ):
        raise ValueError("independent prelaunch acceptance missing or stale")


def _execute_schedule(session, bank_batches, epoch_factory, infer, state_identity, progress):
    """Testable complete schedule. Never accepts or opens an outcome store."""
    from domainnet_reference_adapter import validate_loss_summary

    session.initialize_bank(bank_batches)
    for epoch in range(15):
        loss = validate_loss_summary(session.train_epoch(epoch_factory(epoch), epoch), 512)
        steps = (epoch + 1) * 512
        if session.completed_steps != steps or session.model.queue_ptr != steps * 4 % 16384:
            raise ValueError("native update/queue count mismatch")
        progress(
            {
                "epoch": epoch,
                "completed_steps": steps,
                "queue_ptr": session.model.queue_ptr,
                "initial_queue_fully_replaced": steps * 4 >= 16384,
                "loss": loss,
            }
        )
    # Normalize training flags before hashing; predict may only read state.
    if hasattr(session.model, "eval"):
        session.model.eval()
    before = state_identity()
    direct = infer(False)
    refined = infer(True)
    if state_identity() != before:
        raise ValueError("evaluation changed candidate/normalization/bank state")
    return direct, refined


def inference_batches(view, transform):
    import torch

    for start in range(0, len(view.ids), 4):
        rows = []
        for name in view.ids[start : start + 4]:
            with view.image(name) as image:
                rows.append(transform(image))
        yield torch.stack(rows)


class AdaptationEpoch:
    def __init__(self, view, augmentation, order):
        if len(view.ids) != 2048 or sorted(order) != list(range(2048)):
            raise ValueError("incomplete adaptation epoch")
        self.view, self.augmentation, self.order = view, augmentation, order

    def __len__(self):
        return 512

    def __iter__(self):
        import torch

        for start in range(0, 2048, 4):
            indices = self.order[start : start + 4]
            augmented = []
            for idx in indices:
                with self.view.image(self.view.ids[idx]) as image:
                    augmented.append(self.augmentation(image))
            yield [torch.stack([item[v] for item in augmented]) for v in range(3)], torch.tensor(indices)


def state_digest(session):
    from domainnet_reference_adapter import _state_digest

    return _state_digest(
        {
            "model": session.model.state_dict(),
            "banks": session.banks,
            "optimizer": session.optimizer.state_dict(),
            "queue_ptr": session.model.queue_ptr,
            "steps": session.completed_steps,
        }
    )


def export_inference(session, path):
    """Tensor-only final classifier/bank, not a resume or selected checkpoint."""
    import torch
    from domainnet_reference_adapter import _cpu_copy

    packet = {"classifier": _cpu_copy(session.model.src_model.state_dict()), "banks": _cpu_copy(session.banks)}
    buffer = io.BytesIO()
    torch.save(packet, buffer)
    data = buffer.getvalue()
    if len(data) > 128 * 1024**2:
        raise ValueError("final inference artifact exceeds bound")
    write_bytes(path, data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify_reloaded(session, artifact, identity, view, transform, direct, refined):
    import torch
    from domainnet_reference_adapter import _refresh_weightnorm, _state_digest
    from domainnet_reference_source import build_reference_classifier

    data = read_verified_checkpoint(artifact, identity["sha256"], identity["bytes"])
    saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    fresh = build_reference_classifier()
    fresh.load_state_dict(saved["classifier"], strict=True)
    _refresh_weightnorm(fresh)
    fresh.eval()
    before = _state_digest(saved)
    got, got_refined = [], []
    with torch.no_grad():
        for batch in inference_batches(view, transform):
            features, logits = fresh(batch, return_feats=True)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite reloaded inference")
            got.extend(logits.argmax(1).tolist())
            labels, _, _ = session.ref.refine_predictions(features, logits.softmax(1), saved["banks"], session.args)
            got_refined.extend(labels.tolist())
    if got != direct or got_refined != refined or _state_digest(saved) != before:
        raise ValueError("final serialized inference differs or mutates")


def collect_episode(manifest, inventory, episode_id, output):
    import torch
    from domainnet_reference_adapter import _plain_args, build_session
    from domainnet_reference_source import load_clipart2020

    validate_inventory(manifest, inventory)
    validate_runtime(manifest["runtime"])
    if implementation() != manifest["code"]:
        raise ValueError("collection code changed")
    if type(episode_id) is not int or episode_id not in range(3):
        raise ValueError("invalid episode")
    episode = manifest["episodes"][episode_id]
    output = Path(output)
    notices = []
    adapt = ImageView(inventory, episode_id, "adaptation")
    evaluation = ImageView(inventory, episode_id, "evaluation")
    if adapt.ids != episode["adaptation_ids"] or evaluation.ids != episode["evaluation_ids"]:
        raise ValueError("role view order mismatch")
    start = time.monotonic()
    epoch_start = start
    epoch_receipts = []
    try:
        with warning_receipt(notices):
            config = {
                "reference_root": manifest["reference_root"],
                "epochs": 15,
                "steps_per_epoch": 512,
                "seed": episode["stream_seed"],
            }
            session = build_session(manifest["checkpoint_path"], config, "cpu")
            if _plain_args(session.args) != manifest["resolved_args"]:
                raise ValueError("resolved native recipe changed")
            transform = session.ref.get_augmentation("test")
            augmentation = session.ref.get_augmentation_versions(session.args)
            orders = epoch_orders(episode["stream_seed"])

            def progress(item):
                nonlocal epoch_start
                now = time.monotonic()
                item.update(
                    {
                        "episode": episode_id,
                        "stream_seed": episode["stream_seed"],
                        "manifest_digest": digest(manifest),
                        "inventory_digest": digest(inventory),
                        "order_sha256": digest(orders[item["epoch"]]),
                        "epoch_seconds": now - epoch_start,
                        "elapsed_seconds": now - start,
                        "candidate_state_sha256": state_digest(session),
                    }
                )
                path = output / f"epoch-{item['epoch']:02}.json"
                write_json(path, item)
                epoch_receipts.append(file_identity(path))
                print(json.dumps(item), flush=True)
                epoch_start = time.monotonic()

            def infer(refine):
                # Both role views have a local LRU. Release the training view
                # before even the first evaluation image can populate its LRU.
                adapt.close()
                return session.predict(inference_batches(evaluation, transform), refine=refine).tolist()

            direct, refined = _execute_schedule(
                session,
                inference_batches(adapt, transform),
                lambda e: AdaptationEpoch(adapt, augmentation, orders[e]),
                infer,
                lambda: state_digest(session),
                progress,
            )
            adapt.close()
            artifact = output / "inference.pt"
            artifact_identity = export_inference(session, artifact)
            verify_reloaded(session, artifact, artifact_identity, evaluation, transform, direct, refined)
            baseline, source_receipt = load_clipart2020(manifest["checkpoint_path"])
            baseline.eval()
            frozen = []
            with torch.no_grad():
                for batch in inference_batches(evaluation, transform):
                    logits = baseline(batch)
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError("nonfinite source inference")
                    frozen.extend(logits.argmax(1).tolist())
            result = {
                "schema": "painting-feasibility-predictions-v2",
                "status": "COLLECTED_NOT_SCORED",
                "episode": episode_id,
                "stream_seed": episode["stream_seed"],
                "manifest_digest": digest(manifest),
                "inventory_digest": digest(inventory),
                "code": manifest["code"],
                "runtime": manifest["runtime"],
                "scope": SCOPE,
                "sample_ids": evaluation.ids,
                "adaptation_ids": adapt.ids,
                "frozen": frozen,
                "candidate_direct": direct,
                "candidate_refined": refined,
                "completed_steps": session.completed_steps,
                "queue_ptr": session.model.queue_ptr,
                "candidate_state_sha256": state_digest(session),
                "source_receipt": source_receipt,
                "source_provenance": session.provenance,
                "inference_artifact": artifact_identity,
                "reload_predictions_identical": True,
                "warnings": notices,
                "epoch_receipts": epoch_receipts,
                "wall_seconds": time.monotonic() - start,
            }
            validate_collection(result, manifest, inventory, episode_id)
            return result
    finally:
        adapt.close()
        evaluation.close()
        write_json(output / "warnings.json", notices)


def validate_collection(packet, manifest, inventory, episode_id):
    from domainnet_feasibility_score import validate_predictions

    e = manifest["episodes"][episode_id]
    if (
        packet.get("schema") != "painting-feasibility-predictions-v2"
        or packet.get("status") != "COLLECTED_NOT_SCORED"
        or packet.get("episode") != episode_id
        or packet.get("stream_seed") != e["stream_seed"]
        or packet.get("manifest_digest") != digest(manifest)
        or packet.get("inventory_digest") != digest(inventory)
        or packet.get("code") != manifest["code"]
        or packet.get("runtime") != manifest["runtime"]
        or packet.get("completed_steps") != 7680
        or packet.get("queue_ptr") != 14336
        or packet.get("sample_ids") != e["evaluation_ids"]
        or packet.get("adaptation_ids") != e["adaptation_ids"]
        or packet.get("scope") != SCOPE
        or packet.get("reload_predictions_identical") is not True
        or packet.get("source_receipt", {}).get("release_sha256") != RELEASE_SHA256
        or not isinstance(packet.get("epoch_receipts"), list)
        or len(packet["epoch_receipts"]) != 15
    ):
        raise ValueError("incomplete/unbound prediction packet")
    validate_predictions(packet)
    return packet


def prepare_manifest(repo, output):
    """Only authenticated metadata/code reads; never decompress ZIP members."""
    from domainnet_reference_adapter import _plain_args, load_reference, reference_args

    repo = Path(repo)
    records = repo / "experiments/kbound/results/natural_calibration_value_v1"
    approval = read_approval(records / "PAINTING_FEASIBILITY_APPROVAL_V1.json")
    proposal = read_json(records / "PAINTING_FEASIBILITY_METADATA_V1.json", PROPOSAL_SHA)
    pilot = read_json(records / "DOMAINNET_PILOT_PROPOSAL_V1.json", PILOT_SHA)
    plan = read_bytes(repo / approval["plan"]).decode()
    start = plan.index(approval["design_start_heading"])
    end = plan.index(approval["design_end_heading_exclusive"], start)
    if hashlib.sha256(plan[start:end].encode()).hexdigest() != approval["approved_design_section_sha256"]:
        raise ValueError("approved design text changed")
    if proposal["fresh_output_proposed"] != approval["approved_scope"]["output"]:
        raise ValueError("output differs from approved destination")
    if Path(proposal["fresh_output_proposed"]).exists():
        raise ValueError("approved output already exists; no automatic retry")
    source_list = {**file_identity(pilot["reference"]["list_path"]), "path": pilot["reference"]["list_path"]}
    source = read_bytes(source_list["path"])
    if source_list["sha256"] != LIST_SHA:
        raise ValueError("list authority changed")
    # Class identity is checked here for provenance, never used in selection.
    mapping = {}
    for line in source.decode().splitlines():
        name, label = line.split()
        category = name.split("/")[1]
        label = int(label)
        if category in mapping and mapping[category] != label:
            raise ValueError("class map inconsistent")
        mapping[category] = label
    if digest(mapping) != CLASS_MAP_SHA256:
        raise ValueError("class map identity mismatch")
    read_verified_checkpoint(pilot["checkpoint_path"], RELEASE_SHA256, RELEASE_BYTES)
    with authenticated_archive(pilot["archive"]) as archive:
        directory = {
            name: {"crc32": info.CRC, "encoded_bytes": info.file_size, "compressed_bytes": info.compress_size}
            for name, info in directory_metadata(archive).items()
        }
    args, _ = reference_args(pilot["reference"]["root"], epochs=15, steps_per_epoch=512)
    load_reference(pilot["reference"]["root"], "cpu")
    prior_path = (
        Path(pilot["archive"]["path"]).parent.parent / "kbound-natural-painting-capacity-v1/collect/predictions.json"
    )
    prior_identity = {**file_identity(prior_path), "path": str(prior_path)}
    prior = read_json(prior_path, prior_identity["sha256"])
    if (
        prior["proposal_sha256"] != PILOT_SHA
        or prior["status"] != "COLLECTED_NOT_SCORED"
        or prior["completed_updates"] != 4
    ):
        raise ValueError("prior inventory not from completed approved pilot")
    runtime = runtime_identity()
    validate_runtime(runtime)
    m = build_manifest(
        source,
        {
            "pilot": pilot,
            "proposal": proposal,
            "directory": directory,
            "source_list": source_list,
            "code": implementation(),
            "resolved_args": _plain_args(args),
            "runtime": runtime,
            "prior_inventory": prior_identity,
        },
    )
    write_json(output, m)
    return file_identity(output)


def disk_bytes(root):
    import stat

    from task3_image_panel import _open_directory_chain

    root_descriptor = _open_directory_chain(root)
    original = os.fstat(root_descriptor)
    os.close(root_descriptor)
    total = 0
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            info = (Path(folder) / name).lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("symlink in fresh output")
            if name in names:
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("nonregular output")
                total += info.st_size
    current = os.stat(root, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
        raise ValueError("output root changed while accounting")
    return total


def run_bounded(command, log_root, total_root, budget):
    """Private process group; sampled ceilings, no attachment to existing jobs."""
    import psutil
    from task3_image_panel import _open_directory_chain

    log_root, total_root = Path(log_root), Path(total_root)
    start = time.monotonic()
    peak = size = 0
    last_disk = -100.0
    proc = None
    status = "FAILED"
    # Keep the opened directory capabilities until the child is reaped. Parent
    # symlinks are rejected before a log write or subprocess can occur.
    total_descriptor = _open_directory_chain(total_root)
    try:
        log_descriptor = _open_directory_chain(log_root)
    except BaseException:
        os.close(total_descriptor)
        raise
    total_identity = os.fstat(total_descriptor)
    log_identity = os.fstat(log_descriptor)

    def verify_directories():
        for path, expected in ((total_root, total_identity), (log_root, log_identity)):
            current_fd = _open_directory_chain(path)
            try:
                current = os.fstat(current_fd)
                if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
                    raise ValueError("supervised directory identity changed")
            finally:
                os.close(current_fd)

    try:
        verify_directories()
        fd = os.open("worker.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=log_descriptor)
        with os.fdopen(fd, "wb") as log:
            env = dict(
                os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1"
            )
            verify_directories()
            proc = subprocess.Popen(
                command, cwd=log_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            observed = psutil.Process(proc.pid)
            while True:
                verify_directories()
                elapsed = time.monotonic() - start
                try:
                    peak = max(
                        peak,
                        sum(
                            p.memory_info().rss
                            for p in [observed] + observed.children(recursive=True)
                            if p.is_running()
                        ),
                    )
                except psutil.NoSuchProcess:
                    pass
                # Do not walk 18k staged files every 50ms. Disk is checked at
                # stage boundaries and every 10s; RSS/wall remain sampled .1s.
                if elapsed - last_disk >= 10 or proc.poll() is not None or size == 0:
                    size = disk_bytes(total_root)
                    last_disk = elapsed
                if elapsed > budget["wall_seconds"]:
                    status = "STOP_WALL"
                    break
                if peak > budget["rss_bytes"]:
                    status = "STOP_RSS"
                    break
                if size > budget["disk_bytes"]:
                    status = "STOP_DISK"
                    break
                if proc.poll() is not None:
                    status = "PASS" if proc.returncode == 0 else "PROCESS_FAILED"
                    break
                time.sleep(0.1)
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        os.close(log_descriptor)
        os.close(total_descriptor)
    return {
        "status": status,
        "returncode": proc.returncode,
        "child_reaped": True,
        "wall_seconds": time.monotonic() - start,
        "sampled_peak_rss_bytes": peak,
        "output_bytes": size,
        "ceiling_qualification": "sampled RSS/wall .1s, disk 10s; transient overshoot possible",
    }


def resource_probe(manifest):
    """Full source + 2048 synthetic-image bank; no real image or truth reads."""
    import torch
    from domainnet_reference_adapter import build_session

    notices = []
    with warning_receipt(notices):
        config = {"reference_root": manifest["reference_root"], "epochs": 2, "steps_per_epoch": 4, "seed": 2020}
        session = build_session(manifest["checkpoint_path"], config, "cpu")
        generator = torch.Generator().manual_seed(9817)
        batch = torch.randn(4, 3, 224, 224, generator=generator)
        start = time.monotonic()
        session.initialize_bank(batch for _ in range(512))
        bank_seconds = time.monotonic() - start
        updates = [([batch.clone() for _ in range(3)], torch.arange(i * 4, i * 4 + 4)) for i in range(4)]
        start = time.monotonic()
        for epoch in range(2):
            session.train_epoch(updates, epoch)
        step_seconds = (time.monotonic() - start) / 8
        start = time.monotonic()
        session.predict((batch for _ in range(16)), refine=True)
        per_image = (time.monotonic() - start) / 64
    estimate = step_seconds * 23040 + bank_seconds * 3 + per_image * 4096 * 3 * 5 + 1800
    return {
        "schema": "painting-synthetic-resource-probe-v1",
        "synthetic_only": True,
        "bank_images": 2048,
        "synthetic_updates": 8,
        "bank_seconds": bank_seconds,
        "update_seconds": step_seconds,
        "refined_inference_seconds_per_image": per_image,
        "projected_total_seconds": estimate,
        "fixed_staging_overhead_allowance_seconds": 1800,
        "within_wall_budget": estimate <= BUDGET["wall_seconds"],
        "warnings": notices,
        "manifest_digest": digest(manifest),
        "code": manifest["code"],
        "runtime": manifest["runtime"],
        "qualification": "projection, not measured full-study completion time; synthetic update schedule only for costing",
    }


def _episode_files(episode):
    prefix = f"episode-{episode}"
    return [f"{prefix}/epoch-{j:02}.json" for j in range(15)] + [
        f"{prefix}/predictions.json",
        f"{prefix}/inference.pt",
        f"{prefix}/worker.log",
        f"{prefix}/warnings.json",
        f"{prefix}-process.json",
    ]


def _validate_epoch(record, manifest, inventory, episode, index):
    import re

    from domainnet_reference_adapter import validate_loss_summary

    steps = (index + 1) * 512
    expected = {
        "episode": episode,
        "epoch": index,
        "stream_seed": manifest["episodes"][episode]["stream_seed"],
        "manifest_digest": digest(manifest),
        "inventory_digest": digest(inventory),
        "completed_steps": steps,
        "queue_ptr": steps * 4 % 16384,
        "initial_queue_fully_replaced": steps >= 4096,
        "order_sha256": manifest["episodes"][episode]["epoch_order_sha256"][index],
    }
    if (
        not isinstance(record, dict)
        or any(type(record.get(k)) is not type(v) or record[k] != v for k, v in expected.items())
        or not re.fullmatch("[0-9a-f]{64}", str(record.get("candidate_state_sha256", "")))
    ):
        raise ValueError("unbound or out-of-order epoch evidence")
    for key in ("epoch_seconds", "elapsed_seconds"):
        value = record.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid epoch timing")
    validate_loss_summary(record.get("loss"), 512)


def _verify_episode_completion(bound, manifest, inventory, root, episode):
    """Read fixed relative paths only; externally bound hashes precede truth."""
    expected_paths = _episode_files(episode)
    if (
        not isinstance(bound, dict)
        or type(bound.get("episode")) is not int
        or bound["episode"] != episode
        or not isinstance(bound.get("files"), list)
        or [x.get("path") for x in bound["files"]] != expected_paths
    ):
        raise ValueError("incomplete or reordered episode completion evidence")
    values = {}
    identities = {}
    for entry in bound["files"]:
        name = entry["path"]
        path = Path(root) / name
        identity = {k: entry[k] for k in ("sha256", "bytes")}
        data = (
            read_verified_checkpoint(path, identity["sha256"], identity["bytes"])
            if name.endswith(".pt")
            else read_bytes(path)
        )
        if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
            raise ValueError("closed evidence identity mismatch")
        identities[name] = identity
        if name.endswith(".json"):
            values[name] = json.loads(data)
    prefix = f"episode-{episode}"
    packet = validate_collection(values[f"{prefix}/predictions.json"], manifest, inventory, episode)
    if packet["inference_artifact"] != identities[f"{prefix}/inference.pt"]:
        raise ValueError("inference evidence differs from packet")
    previous_elapsed = 0
    for index in range(15):
        name = f"{prefix}/epoch-{index:02}.json"
        if identities[name] != packet["epoch_receipts"][index]:
            raise ValueError("epoch differs from collection binding")
        record = values[name]
        _validate_epoch(record, manifest, inventory, episode, index)
        if record["elapsed_seconds"] < previous_elapsed:
            raise ValueError("epoch time reversed")
        previous_elapsed = record["elapsed_seconds"]
    if values[f"{prefix}/warnings.json"] != packet["warnings"]:
        raise ValueError("warning evidence differs from packet")
    process = values[f"{prefix}-process.json"]
    if process.get("status") != "PASS" or process.get("returncode") != 0 or process.get("child_reaped") is not True:
        raise ValueError("collection process did not complete")
    return packet


def bind_episode_completion(manifest, inventory, root, episode):
    """Supervisor calls this only after worker exit and process receipt write."""
    packet_path = Path(root) / f"episode-{episode}/predictions.json"
    packet = read_json(packet_path, file_identity(packet_path)["sha256"])
    bound = {
        "episode": episode,
        "files": [
            {
                "path": name,
                **(packet["inference_artifact"] if name.endswith(".pt") else file_identity(Path(root) / name)),
            }
            for name in _episode_files(episode)
        ],
    }
    _verify_episode_completion(bound, manifest, inventory, root, episode)
    return bound


def score_complete(manifest, inventory, completion, root, truth_loader):
    """Authenticate all final packets/artifacts BEFORE calling label authority."""
    from domainnet_feasibility_score import score_episode, summarize_episodes

    if (
        completion.get("schema") != "painting-feasibility-completion-v2"
        or completion.get("status") != "ALL_THREE_COLLECTED_NOT_SCORED"
        or completion.get("manifest_digest") != digest(manifest)
        or completion.get("inventory_digest") != digest(inventory)
        or not isinstance(completion.get("episodes"), list)
        or len(completion["episodes"]) != 3
    ):
        raise ValueError("incomplete collection; truth remains unopened")
    packets = [
        _verify_episode_completion(bound, manifest, inventory, root, e)
        for e, bound in enumerate(completion["episodes"])
    ]
    truth = truth_loader()
    report = summarize_episodes([score_episode(p, truth) for p in packets])
    report.update(
        {
            "collection_digest": digest(completion),
            "manifest_digest": digest(manifest),
            "warnings": [p["warnings"] for p in packets],
        }
    )
    return report


def validate_stage_inputs(stage, inputs, manifest_digest):
    fields = {"manifest_digest"}
    if stage in ("collect", "score"):
        fields.add("inventory_sha256")
    if stage == "score":
        fields.add("completion_sha256")
    if (
        stage not in ("probe", "stage", "collect", "score")
        or set(inputs) != fields
        or inputs.get("manifest_digest") != manifest_digest
    ):
        raise ValueError("stage input authority missing/mismatched")
    for name in fields - {"manifest_digest"}:
        import re

        if not re.fullmatch("[0-9a-f]{64}", inputs[name]):
            raise ValueError("invalid stage input digest")


def _worker(manifest, root, stage, episode, inputs):
    """Called only by a hash-bound supervised process after CLI checks."""
    import resource

    import torch

    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2, 128 * 1024**2))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.basicConfig(level=logging.INFO)
    validate_stage_inputs(stage, inputs, digest(manifest))
    if stage == "probe":
        write_json(root / "probe/resource.json", resource_probe(manifest))
        return
    if stage == "stage":
        notices = []
        try:
            with warning_receipt(notices):
                stage_selected(manifest, root / "images")
        finally:
            write_json(root / "stage/warnings.json", notices)
        return
    inventory = read_json(root / "images/inventory.json", inputs["inventory_sha256"])
    validate_inventory(manifest, inventory)
    if stage == "collect":
        dest = root / f"episode-{episode}"
        write_json(dest / "predictions.json", collect_episode(manifest, inventory, episode, dest))
        return
    if stage == "score":
        from domainnet_feasibility_score import parse_authenticated_truth

        completion = read_json(root / "collection-complete.json", inputs["completion_sha256"])

        # No earlier worker opens this class-to-outcome parser.
        def truth():
            source = manifest["source_list"]
            return parse_authenticated_truth(
                read_verified_checkpoint(source["path"], source["sha256"], source["bytes"])
            )

        write_json(root / "score/report.json", score_complete(manifest, inventory, completion, root, truth))
        return
    raise ValueError("unknown worker stage")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repo", required=True)
    prep.add_argument("--out", required=True)
    for mode in ("run", "_worker"):
        p = sub.add_parser(mode)
        p.add_argument("--manifest", required=True)
        p.add_argument("--sha256", required=True)
        p.add_argument("--review", required=True)
        p.add_argument("--review-sha256", required=True)
        if mode == "_worker":
            p.add_argument("--authorization-sha256", required=True)
            p.add_argument("--stage", required=True)
            p.add_argument("--inputs-sha256", required=True)
            p.add_argument("--episode", type=int)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        print(json.dumps(prepare_manifest(args.repo, args.out)))
        return 0
    manifest = validate_manifest(read_json(args.manifest, args.sha256))
    if manifest["code"] != implementation():
        raise ValueError("manifest implementation changed")
    validate_runtime(manifest["runtime"])
    review = read_json(args.review, args.review_sha256)
    validate_review(review, args.sha256, manifest)
    root = Path(manifest["output_root"])
    if args.mode == "_worker":
        auth = read_json(root / "authorization.json", args.authorization_sha256)
        if (
            auth["supervisor_pid"] != os.getppid()
            or auth["manifest_sha256"] != args.sha256
            or auth["review_sha256"] != args.review_sha256
        ):
            raise ValueError("worker not owned by authorized supervisor")
        stage_name = f"episode-{args.episode}" if args.stage == "collect" else args.stage
        if stage_name not in ("probe", "stage", "score", "episode-0", "episode-1", "episode-2"):
            raise ValueError("unknown fixed stage")
        inputs = read_json(root / f"{stage_name}-inputs.json", args.inputs_sha256)
        _worker(manifest, root, args.stage, args.episode, inputs)
        return 0
    import shutil

    if shutil.disk_usage(root.parent).free < BUDGET["disk_bytes"] + 2 * 1024**3:
        raise ValueError("insufficient output disk headroom")
    import psutil

    if psutil.virtual_memory().available < BUDGET["rss_bytes"]:
        raise ValueError("insufficient RAM headroom")
    fresh_directory(root)
    start = time.monotonic()
    auth = {
        "supervisor_pid": os.getpid(),
        "manifest_sha256": args.sha256,
        "review_sha256": args.review_sha256,
        "approval_sha256": APPROVAL_SHA,
        "code": manifest["code"],
        "scope": SCOPE,
    }
    write_json(root / "authorization.json", auth)
    auth_sha = file_identity(root / "authorization.json")["sha256"]
    common = [
        sys.executable,
        str(Path(__file__).absolute()),
        "_worker",
        "--manifest",
        str(Path(args.manifest).absolute()),
        "--sha256",
        args.sha256,
        "--review",
        str(Path(args.review).absolute()),
        "--review-sha256",
        args.review_sha256,
        "--authorization-sha256",
        auth_sha,
    ]
    stage_name = "preflight"
    inventory_sha = completion_sha = None
    packets = []
    try:
        for stage, episode in (
            [("probe", None), ("stage", None)] + [("collect", e) for e in range(3)] + [("score", None)]
        ):
            stage_name = f"episode-{episode}" if stage == "collect" else stage
            directory = fresh_directory(root / stage_name)
            inputs = {"manifest_digest": digest(manifest)}
            if stage in ("collect", "score"):
                inputs["inventory_sha256"] = inventory_sha
            if stage == "score":
                inputs["completion_sha256"] = completion_sha
            validate_stage_inputs(stage, inputs, digest(manifest))
            input_path = root / f"{stage_name}-inputs.json"
            write_json(input_path, inputs)
            command = common + ["--stage", stage, "--inputs-sha256", file_identity(input_path)["sha256"]]
            if episode is not None:
                command += ["--episode", str(episode)]
            budget = {**BUDGET, "wall_seconds": BUDGET["wall_seconds"] - (time.monotonic() - start)}
            result = run_bounded(command, directory, root, budget)
            write_json(root / f"{stage_name}-process.json", result)
            if result["status"] != "PASS":
                raise ValueError(stage_name + ": " + result["status"])
            if stage == "probe":
                report = read_json(directory / "resource.json", file_identity(directory / "resource.json")["sha256"])
                if not report["within_wall_budget"]:
                    raise ValueError("expanded resource projection exceeds approved 12h")
            if stage == "stage":
                inventory_sha = file_identity(root / "images/inventory.json")["sha256"]
                validate_inventory(manifest, read_json(root / "images/inventory.json", inventory_sha))
            if stage == "collect":
                bound = file_identity(directory / "predictions.json")
                inventory = read_json(root / "images/inventory.json", inventory_sha)
                validate_collection(
                    read_json(directory / "predictions.json", bound["sha256"]), manifest, inventory, episode
                )
                packets.append(bind_episode_completion(manifest, inventory, root, episode))
            if stage == "collect" and episode == 2:
                inventory = read_json(root / "images/inventory.json", inventory_sha)
                completion = {
                    "schema": "painting-feasibility-completion-v2",
                    "status": "ALL_THREE_COLLECTED_NOT_SCORED",
                    "manifest_digest": digest(manifest),
                    "inventory_digest": digest(inventory),
                    "episodes": packets,
                }
                write_json(root / "collection-complete.json", completion)
                completion_sha = file_identity(root / "collection-complete.json")["sha256"]
        write_json(
            root / "status.json",
            {
                "status": "DEVELOPMENT_FEASIBILITY_COMPLETE",
                "scope": SCOPE,
                "wall_seconds": time.monotonic() - start,
                "report": file_identity(root / "score/report.json"),
            },
        )
    except BaseException as exc:
        write_json(
            root / "failure.json",
            {
                "status": "STOPPED",
                "stage": stage_name,
                "reason": str(exc),
                "automatic_retry": False,
                "wall_seconds": time.monotonic() - start,
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
