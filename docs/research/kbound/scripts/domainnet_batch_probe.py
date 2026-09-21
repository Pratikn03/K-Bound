"""One reviewed, synthetic-only full-ResNet-50 batch-128 resource attempt.

No dataset IO, prediction scoring, saved-model export, retry or recipe search.
Random three-view inputs and a random feature bank measure update geometry only.
"""

import argparse
import hashlib
import json
import os
import stat
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

from domainnet_feasibility_contract import file_identity, read_bytes, read_json, write_json

V4_SHA = "b48c582d22b5e69eba9c83c3a16004044fd2b1931481a6e9a334284353f9c66d"
BUDGET = {"wall_seconds": 120, "rss_bytes": 4 * 1024**3, "disk_bytes": 128 * 1024**2}
SETTINGS = {
    "batch_size": 128,
    "views": 3,
    "spatial_size": 224,
    "device": "cpu",
    "torch_threads": 1,
    "loader_workers": 0,
    "updates": 1,
    "epochs": 1,
    "bank_entries": 2048,
    "input_seed": 9817,
    "session_seed": 2020,
    "synthetic_only": True,
}
QUALIFICATION = (
    "Synthetic random inputs and feature/probability bank; one-step costing schedule. "
    "Not real adaptation, native bank initialization, augmentation, accuracy, convergence, "
    "full-protocol reproduction, a full-run time estimate or hardware-wide impossibility proof."
)


def validate_settings(settings, budget):
    if settings != SETTINGS or budget != BUDGET:
        raise ValueError("unapproved synthetic probe settings")


def input_authority(original_manifest):
    """Read only the pinned V4 metadata/code, not its dataset members or outcomes."""
    base = read_json(original_manifest, V4_SHA)
    for name, identity in base["code"].items():
        if file_identity(Path(__file__).parent / name) != identity:
            raise ValueError("accepted original implementation changed: " + name)
    return base


def prepare(original_manifest, manifest_path, output_root):
    base = input_authority(original_manifest)
    root = Path(output_root).absolute()
    if root.exists() or root.is_symlink():
        raise ValueError("probe output must be fresh")
    m = {
        "schema": "domainnet-batch-probe-v1",
        "original_manifest": str(Path(original_manifest).absolute()),
        "output_root": str(root),
        "settings": SETTINGS,
        "budget": BUDGET,
        "code": {**base["code"], Path(__file__).name: file_identity(__file__)},
        "runtime": base["runtime"],
        "checkpoint_path": base["checkpoint_path"],
        "reference_root": base["reference_root"],
        "qualification": QUALIFICATION,
    }
    write_json(manifest_path, m)
    return file_identity(manifest_path)


def validate_review(review, sha, manifest):
    if (
        review.get("schema") != "domainnet-batch-probe-review-v1"
        or review.get("accepted_for_launch") is not True
        or review.get("manifest_sha256") != sha
        or review.get("code") != manifest["code"]
        or not review.get("reviewer")
    ):
        raise ValueError("independent probe review missing or stale")


def authorize(manifest_path, manifest_sha, review_path, review_sha):
    m = read_json(manifest_path, manifest_sha)
    validate_review(read_json(review_path, review_sha), manifest_sha, m)
    validate_settings(m["settings"], m["budget"])
    base = input_authority(m["original_manifest"])
    if (
        m["schema"] != "domainnet-batch-probe-v1"
        or m["qualification"] != QUALIFICATION
        or m["code"] != {**base["code"], Path(__file__).name: file_identity(__file__)}
        or any(m[k] != base[k] for k in ("runtime", "reference_root", "checkpoint_path"))
    ):
        raise ValueError("probe source/runtime/code identity changed")
    from domainnet_feasibility_runner import validate_runtime

    validate_runtime(m["runtime"])
    return m


def check_supervisor(manifest, manifest_sha, review_sha):
    auth = json.loads(read_bytes(Path(manifest["output_root"]) / "authorization.json"))
    if (
        auth.get("supervisor_pid") != os.getppid()
        or auth.get("manifest_sha256") != manifest_sha
        or auth.get("review_sha256") != review_sha
        or auth.get("code") != manifest["code"]
    ):
        raise ValueError("matching live supervisor required")


@contextmanager
def retained_warnings(root):
    """Persist warnings immediately, including before a watchdog kills the child."""
    index = 0
    with warnings.catch_warnings():
        warnings.simplefilter("always")

        def emit(message, category, filename, lineno, file=None, line=None):
            nonlocal index
            write_json(
                Path(root) / f"warning-{index:03}.json",
                {"message": str(message), "category": category.__name__, "filename": str(filename), "line": lineno},
            )
            index += 1
            print(warnings.formatwarning(message, category, filename, lineno), file=sys.stderr, flush=True)

        warnings.showwarning = emit
        yield


def bn_counters(model):
    return {name: int(tensor) for name, tensor in model.state_dict().items() if name.endswith("num_batches_tracked")}


def validate_measurement(record, batch_size):
    from domainnet_reference_adapter import validate_loss_summary

    validate_loss_summary(record["loss"], 1)
    if (
        record["updates"] != 1
        or record["queue_ptr"] != batch_size
        or record["optimizer_state_entries"] <= 0
        or not record["bn_counter_deltas"]
        or set(record["bn_counter_deltas"].values()) != {2}
    ):
        raise ValueError("incomplete native synthetic update")


def execute_synthetic(session, views):
    """One native update; callers provide only in-memory unlabeled tensors."""
    import torch

    before = bn_counters(session.model)
    start = time.monotonic()
    loss = session.train_epoch([(views, torch.arange(len(views[0])))], 0)
    elapsed = time.monotonic() - start
    after = bn_counters(session.model)
    if set(after) != set(before):
        raise ValueError("normalization state keys changed")
    record = {
        "updates": session.completed_steps,
        "queue_ptr": session.model.queue_ptr,
        "loss": loss,
        "update_seconds": elapsed,
        "optimizer_state_entries": len(session.optimizer.state),
        "bn_counter_deltas": {k: after[k] - before[k] for k in before},
    }
    validate_measurement(record, len(views[0]))
    return record


def worker(args):
    m = authorize(args.manifest, args.manifest_sha, args.review, args.review_sha)
    check_supervisor(m, args.manifest_sha, args.review_sha)
    root = Path(m["output_root"])
    started = time.monotonic()

    def phase(index, name):
        write_json(
            root / f"phase-{index:02}.json", {"phase": name, "seconds_since_worker_auth": time.monotonic() - started}
        )
        print(name, flush=True)

    with retained_warnings(root):
        import torch
        from domainnet_reference_adapter import build_session
        from torch.nn import functional as F

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        phase(0, "authenticated_source_construction_started")
        session = build_session(
            m["checkpoint_path"],
            {"reference_root": m["reference_root"], "epochs": 1, "steps_per_epoch": 1, "seed": 2020},
            "cpu",
        )
        write_json(root / "source.json", session.provenance)
        phase(1, "full_source_and_teacher_ready")
        gen = torch.Generator(device="cpu").manual_seed(9817)
        # NOT initialized from any source inference, images, predictions or labels.
        session.banks = {
            "features": F.normalize(torch.randn(2048, session.model.src_model.output_dim, generator=gen), dim=1),
            "probs": F.softmax(torch.randn(2048, session.model.src_model.num_classes, generator=gen), dim=1),
            "ptr": 0,
        }
        views = [torch.randn(128, 3, 224, 224, generator=gen) for _ in range(3)]
        phase(2, "synthetic_inputs_ready_update_started")
        result = execute_synthetic(session, views)
        result.update(
            schema="domainnet-batch-resource-measurement-v1",
            settings=m["settings"],
            manifest_sha256=args.manifest_sha,
            code=m["code"],
            qualification=QUALIFICATION,
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
        )
        write_json(root / "measurement.json", result)
        phase(3, "one_synthetic_update_complete")


def close_attempt(root, receipt):
    root = Path(root)
    write_json(root / "process.json", receipt)
    if receipt["status"] != "PASS":
        identities = {path.name: closed_identity(path) for path in sorted(root.iterdir())}
        write_json(
            root / "failure.json",
            {
                "resource_receipt": receipt,
                "retry_authorized": False,
                "files": identities,
                "qualification": QUALIFICATION,
            },
        )
        return False
    measurement = json.loads(read_bytes(root / "measurement.json"))
    validate_measurement(measurement, 128)
    names = ["authorization.json", "source.json", "measurement.json", "process.json", "worker.log"]
    names += [f"phase-{i:02}.json" for i in range(4)]
    names += [p.name for p in sorted(root.glob("warning-*.json"))]
    identities = {name: closed_identity(root / name) for name in names}
    write_json(
        root / "complete.json",
        {"status": "SYNTHETIC_RESOURCE_PROBE_COMPLETE", "files": identities, "qualification": QUALIFICATION},
    )
    return True


def closed_identity(path):
    """Hash closed output, permitting an empty log after an early stop."""
    from task3_image_panel import _open_directory_chain

    path = Path(path).absolute()
    directory = _open_directory_chain(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or not 0 <= before.st_size <= BUDGET["disk_bytes"]
                or getattr(before, "st_flags", 0) & 0x40000000
            ):
                raise ValueError("resident bounded output file required")
            digest, count = hashlib.sha256(), 0
            while chunk := handle.read(1024**2):
                count += len(chunk)
                if count > BUDGET["disk_bytes"]:
                    raise ValueError("output grew beyond bound")
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if count != before.st_size or any(getattr(before, f) != getattr(after, f) for f in fields):
                raise ValueError("output changed during authentication")
            return {"sha256": digest.hexdigest(), "bytes": count}
    finally:
        os.close(directory)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for field in ("original-manifest", "out", "output-root"):
        prep.add_argument("--" + field, type=Path, required=True)
    for cmd in ("run", "_worker"):
        parser = sub.add_parser(cmd)
        for field in ("manifest", "review"):
            parser.add_argument("--" + field, type=Path, required=True)
            parser.add_argument("--" + field + "-sha", required=True)
    args = p.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.original_manifest, args.out, args.output_root)))
        return 0
    if args.command == "_worker":
        worker(args)
        return 0
    m = authorize(args.manifest, args.manifest_sha, args.review, args.review_sha)
    from domainnet_feasibility_images import fresh_directory
    from domainnet_feasibility_runner import run_bounded

    root = fresh_directory(m["output_root"])
    write_json(
        root / "authorization.json",
        {
            "supervisor_pid": os.getpid(),
            "manifest_sha256": args.manifest_sha,
            "review_sha256": args.review_sha,
            "code": m["code"],
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
    success = close_attempt(root, receipt)
    print(json.dumps(receipt))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
