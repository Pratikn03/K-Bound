#!/usr/bin/env python3
"""SHA-bound terminal interface for common-panel decisions and separate scoring.

Each input is a separate JSON file with an explicit expected SHA256 of its raw
bytes. ``decide`` has no score-outcome argument and never fits from that file.
``score`` consumes a decision packet bound by a separately supplied expected byte
digest; it never fits a predictor. Caller-supplied hashes are integrity bindings,
not independent authentication or proof of prospective sealing chronology.

Both commands require a fresh output directory under an existing parent. Files
are exclusively created, synchronized, and made read-only, then the directory
is made read-only. This prevents accidental overwrites, not privileged tampering.
The deterministic decision packet is separate from nondeterministic timing and
provenance receipts. No native estimator, image/backbone inference, or benchmark runs here.
"""
from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import sys
import time


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _directory(path):
    """Resolve every directory component without following symlinks."""
    path = Path(os.path.abspath(path))
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _nonfinite(value):
    raise ValueError("non-finite JSON numeric token")


def _read_bound(path, expected_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected SHA256 must be 64 lowercase hexadecimal characters")
    path = Path(os.path.abspath(path))
    parent = _directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if getattr(before, "st_flags", 0) & getattr(stat, "SF_DATALESS", 0x40000000):
                raise ValueError("dataless input must already be resident; refusing content read")
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 64 * 1024**2:
                raise ValueError("input must be a nonempty regular JSON file of at most 64 MiB")
            data = handle.read(64 * 1024**2 + 1)
            after = os.fstat(handle.fileno())
            signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if signature(before) != signature(after) or len(data) != before.st_size:
                raise ValueError("input changed while being read")
    finally:
        os.close(parent)
    digest = _sha(data)
    if digest != expected_sha256:
        raise ValueError(f"input byte SHA256 mismatch: {path.name}")
    parsed = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    return parsed, {"path": str(path), "sha256_bytes": digest, "size_bytes": len(data)}


def _expected_methods(value):
    if value == "none":
        return []
    methods = value.split(",")
    if (not methods or len(methods) != len(set(methods))
            or set(methods) - {"AETTA", "Baek_ALine"}):
        raise ValueError("expected methods must be AETTA, Baek_ALine, AETTA,Baek_ALine, or explicit none")
    return sorted(methods)


def _check_methods(scoring, manifest, estimates, expected):
    # The scorer validates complete per-cell identities, values and native/port
    # labels before any fitting. Presence of a method name alone is insufficient.
    native = scoring._native(scoring._manifest(manifest), estimates)
    if set(native) != set(expected):
        raise ValueError(f"native method set mismatch: expected {expected}, received {sorted(native)}")


def _fit_with_timing(scoring, values):
    """Observe the actual sklearn fit call, without replacing it or its settings."""
    profiler = cProfile.Profile()
    started = time.perf_counter()
    gate = profiler.runcall(scoring.fit_gate, values["manifest"], values["development-features"],
                           values["fit-outcomes"], values["calibration-outcomes"])
    elapsed = time.perf_counter() - started
    fit_code = inspect.unwrap(scoring.GradientBoostingRegressor.fit).__code__
    entries = [entry for entry in profiler.getstats() if entry.code is fit_code]
    if len(entries) != 1 or entries[0].callcount != 1:
        raise RuntimeError("unable to isolate exactly one sklearn predictor fit call")
    return gate, {"predictor_fit_seconds": entries[0].totaltime,
                  "predictor_fit_call_count": entries[0].callcount,
                  "fit_and_calibration_seconds": elapsed,
                  "predictor_fit_scope": "cProfile cumulative time of unwrapped sklearn GBR.fit; excludes calibration and score prediction; profiling overhead is present"}


def _preflight_output(path):
    path = Path(os.path.abspath(path))
    parent = _directory(path.parent)
    try:
        try:
            os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ValueError("output directory must be fresh; existing paths and symlinks are rejected")
    finally:
        os.close(parent)


def _write_file(fd, name, data):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), 0o444)


def _check_output_identity(path, parent, fd):
    """The caller-visible path must still name the retained output descriptor."""
    def identity(value):
        return value.st_dev, value.st_ino

    try:
        visible_parent = _directory(path.parent)
        try:
            if identity(os.fstat(visible_parent)) != identity(os.fstat(parent)):
                raise ValueError("output parent path identity changed during write")
            visible = os.stat(path.name, dir_fd=visible_parent, follow_symlinks=False)
            if not stat.S_ISDIR(visible.st_mode) or identity(visible) != identity(os.fstat(fd)):
                raise ValueError("output directory path identity changed during write")
        finally:
            os.close(visible_parent)
    except OSError as exc:
        raise ValueError("output path chain identity changed during write") from exc


def _write_output(path, artifact_name, artifact, receipt, started):
    path = Path(os.path.abspath(path))
    parent = _directory(path.parent)
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
        fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            data = _encode(artifact)
            _write_file(fd, artifact_name, data)
            _check_output_identity(path, parent, fd)
            receipt["outputs"] = {artifact_name: {"sha256_bytes": _sha(data), "size_bytes": len(data)}}
            receipt["timing"]["total_seconds"] = time.perf_counter() - started
            receipt["timing"]["total_scope"] = (
                "command entry through input validation, imports, computation and primary artifact sync; "
                "excludes interpreter startup, receipt write, final directory sealing and stdout")
            receipt["completed_at_utc_before_receipt_write"] = datetime.now(timezone.utc).isoformat()
            _write_file(fd, "receipt.json", _encode(receipt))
            os.fchmod(fd, 0o555)
            os.fsync(fd)
            _check_output_identity(path, parent, fd)
        finally:
            os.close(fd)
    finally:
        os.close(parent)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, keys in (
        ("decide", ("manifest", "development-features", "fit-outcomes", "calibration-outcomes",
                    "score-features", "native-estimates")),
        ("score", ("manifest", "decision-packet", "score-outcomes")),
    ):
        sub = phases.add_parser(phase)
        for key in keys:
            sub.add_argument("--" + key, required=True, type=Path)
            sub.add_argument("--" + key + "-sha256", required=True,
                             help="expected SHA256 of raw input file bytes, not canonical JSON")
        sub.add_argument("--output-dir", required=True, type=Path)
        if phase == "decide":
            sub.add_argument("--expected-methods", required=True,
                             help="AETTA, Baek_ALine, AETTA,Baek_ALine, or explicit none")
    return parser


def main(argv=None):
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _preflight_output(args.output_dir)
        keys = ("manifest", "development-features", "fit-outcomes", "calibration-outcomes",
                "score-features", "native-estimates") if args.phase == "decide" else (
                "manifest", "decision-packet", "score-outcomes")
        values, bindings = {}, {}
        for key in keys:
            attribute = key.replace("-", "_")
            values[key], bindings[key] = _read_bound(getattr(args, attribute), getattr(args, attribute + "_sha256"))
        # Import only the saved-data scoring component. No native inference module.
        import task3_common_scoring as scoring
        import sklearn
        source_files = [Path(__file__).resolve(), Path(scoring.__file__).resolve(),
                        Path(scoring.__file__).with_name("run_decision_baselines.py"),
                        Path(scoring.__file__).with_name("kbound_decide.py")]
        source_bindings = {str(path): _sha(path.read_bytes()) for path in source_files}
        expected = None
        if args.phase == "decide":
            expected = _expected_methods(args.expected_methods)
            _check_methods(scoring, values["manifest"], values["native-estimates"], expected)
            gate, timing = _fit_with_timing(scoring, values)
            artifact = scoring.decide_gate(gate, values["score-features"], values["native-estimates"])
            packet_digest = artifact["packet_sha256"]
            artifact_name = "decision_packet.json"
            status = "DECISIONS_WRITTEN_NOT_BENCHMARK_EVIDENCE"
        else:
            artifact = scoring.score_decisions(values["manifest"], values["decision-packet"], values["score-outcomes"])
            packet_digest = values["decision-packet"]["packet_sha256"]
            artifact_name = "metrics.json"
            status = "SAVED_OUTCOMES_SCORED_NOT_NATIVE_BENCHMARK_AUTHENTICATION"
            timing = {"predictor_fit_seconds": 0.0, "predictor_fit_call_count": 0,
                      "fit_and_calibration_seconds": 0.0,
                      "predictor_fit_scope": "no predictor fitting or calibration in score command"}
        if source_bindings != {str(path): _sha(path.read_bytes()) for path in source_files}:
            raise ValueError("implementation source changed during computation")
        receipt = {"schema": "task3-common-analysis-receipt-v1", "phase": args.phase,
            "status": status, "started_at_utc": started_at, "inputs": bindings,
            "expected_native_methods": expected, "timing": timing,
            "decision_packet_sha256_canonical_body": packet_digest,
            "implementation_source_sha256_bytes": source_bindings,
            "runtime": {"python_executable": sys.executable, "python_version": sys.version,
                        "sklearn_version": sklearn.__version__, "numpy_version": scoring.np.__version__},
            "independently_authenticated_native_execution": False,
            "binding_scope": "raw-byte input SHA256 verified against caller-supplied expected values; packet internal SHA256 is canonical JSON body without packet_sha256",
            "sealing_scope": "fresh exclusive output files chmod0444 and directory0555; caller must retain expected packet byte digest before opening scored outcomes; chronology and privileged tampering are not authenticated",
            "scope": "saved-input analysis only; no native invocation, image/backbone inference or benchmark-completion claim"}
        _write_output(args.output_dir, artifact_name, artifact, receipt, started)
        print(json.dumps({"status": status, "output_dir": str(args.output_dir), "artifact": artifact_name}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
