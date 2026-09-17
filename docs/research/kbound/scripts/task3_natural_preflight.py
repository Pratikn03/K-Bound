#!/usr/bin/env python3
"""Fail-closed input checks for the official-baseline and natural tracks.

This module intentionally does not launch training or evaluation.  It only
answers whether the inputs are complete enough for a later launch, keeping
official Task 3 evidence separate from the natural-shift study.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

OFFICIAL_CORRUPTIONS = ("gaussian_noise", "shot_noise", "impulse_noise")


def _path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value).expanduser().resolve()


def _class_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(item for item in path.iterdir() if item.is_dir())


def _source_status(name: str, source: str | Path | None, *, require_clean: bool) -> list[str]:
    path = _path(source)
    if path is None or not path.is_dir():
        return [f"{name}_source_missing"]
    git_entry = path / ".git"
    if not git_entry.exists():
        return [f"{name}_source_not_git_checkout"]
    if not require_clean:
        return []
    probe = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return [f"{name}_source_git_unreadable"]
    meaningful: list[str] = []
    for line in probe.stdout.splitlines():
        changed_path = line[3:].strip() if len(line) >= 3 else line.strip()
        parts = Path(changed_path).parts
        if (
            "__pycache__" in parts
            or changed_path.endswith(".pyc")
            or Path(changed_path).name == ".DS_Store"
            or Path(changed_path).name.startswith("._")
        ):
            continue
        if parts and parts[0] in {"results", ".pytest_cache"}:
            continue
        meaningful.append(line)
    return [f"{name}_source_dirty"] if meaningful else []


def _python_command(executable: str | Path | None) -> tuple[str | None, str | None]:
    """Resolve command names through PATH and explicit paths without reinterpretation."""
    if executable is None:
        return None, "python_missing"
    value = str(executable)
    if "/" in value or value.startswith("~"):
        # Resolving a venv's python symlink changes sys.prefix and can silently
        # probe the base environment instead of the selected dependency set.
        path = Path(value).expanduser().absolute()
        if not path.is_file() or not path.stat().st_mode & 0o111:
            return None, "python_not_executable"
        return str(path), None
    command = shutil.which(value)
    return (str(Path(command).absolute()), None) if command else (None, "python_not_found")


def _runtime_status(name: str, executable: str | Path | None, imports: tuple[str, ...]) -> list[str]:
    """Check an environment without importing its packages into this process."""
    command, error = _python_command(executable)
    if error:
        return [f"{name}_{error}"]
    code = "import " + ", ".join(imports)
    probe = subprocess.run([command, "-c", code], text=True, capture_output=True, check=False)
    return [] if probe.returncode == 0 else [f"{name}_dependency_missing"]


def _runtime_capabilities(executable: str | Path | None) -> dict[str, bool] | None:
    """Read accelerator availability from the pinned environment.

    The upstream POEM and AETTA trees contain unconditional CUDA calls.  A
    ``--help`` probe cannot see those calls because argument parsing exits
    before model construction, so the launch gate records the actual backend
    capabilities separately.
    """

    command, error = _python_command(executable)
    if error:
        return None
    code = (
        "import torch; "
        "mps = getattr(torch.backends, 'mps', None); "
        "print(int(torch.cuda.is_available()), int(bool(mps and mps.is_available())))"
    )
    probe = subprocess.run([command, "-c", code], text=True, capture_output=True, check=False)
    if probe.returncode != 0:
        return None
    fields = probe.stdout.strip().split()
    if len(fields) != 2 or any(value not in {"0", "1"} for value in fields):
        return None
    return {"cuda": fields[0] == "1", "mps": fields[1] == "1"}


def _unconditional_cuda_files(source: str | Path | None) -> list[str]:
    """Return upstream Python files that require CUDA before a real run.

    This is intentionally conservative: a source tree with any unconditional
    ``.cuda()`` or ``torch.cuda.set_device`` call cannot be called native on a
    machine whose pinned environment has no CUDA backend.  We do not rewrite
    the upstream tree to make this check pass, because such a rewrite would be
    a platform port rather than native official evidence.
    """

    path = _path(source)
    if path is None or not path.is_dir():
        return []
    markers = (
        re.compile(r"\.cuda\s*\("),
        re.compile(r"torch\.cuda\.set_device\s*\("),
    )
    matches: list[str] = []
    for file_path in sorted(path.rglob("*.py")):
        if any(part in {".git", "__pycache__", "dataset", "cached_data", "results"} for part in file_path.parts):
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(marker.search(text) for marker in markers):
            matches.append(file_path.relative_to(path).as_posix())
    return matches


def _tracked_source_file(source: str | Path | None, relative: str) -> bool:
    """Return whether a required file belongs to the pinned upstream tree."""

    path = _path(source)
    if path is None or not path.is_dir() or not (path / relative).is_file():
        return False
    probe = subprocess.run(
        ["git", "-C", str(path), "ls-files", "--error-unmatch", relative],
        text=True,
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == relative


def _entrypoint_status(
    name: str,
    source: str | Path | None,
    executable: str | Path | None,
    *,
    details: dict[str, Any],
    sar_source: str | Path | None = None,
) -> list[str]:
    """Require a known native entrypoint to start before a real launch.

    Import checks alone are insufficient: an upstream checkout can have a
    healthy environment while its runner imports a missing package (as the
    POEM checkout does when ``models/`` is absent).  This is deliberately a
    ``--help`` smoke check, never a data or model run.
    """

    path = _path(source)
    if path is None or not path.is_dir():
        return []
    entrypoint = path / "main.py"
    if not entrypoint.is_file():
        return [f"{name}_entrypoint_missing"]
    command, error = _python_command(executable)
    if error:
        return [f"{name}_{error}"]
    probe_command = [command, str(entrypoint), "--help"]
    if name == "poem" and sar_source is not None:
        probe_command = [command, str(Path(__file__).with_name("poem_dependency_bootstrap.py")),
                         "--poem-source", str(path), "--sar-source", str(_path(sar_source)),
                         "--", "--help"]
    try:
        probe = subprocess.run(
            probe_command,
            cwd=str(path),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        details[f"{name}_entrypoint"] = {"path": str(entrypoint), "status": "TIMEOUT"}
        return [f"{name}_entrypoint_not_runnable"]
    details[f"{name}_entrypoint"] = {
        "path": str(entrypoint),
        "returncode": probe.returncode,
        "stderr_tail": probe.stderr[-500:],
        "dependency_bootstrap": name == "poem" and sar_source is not None,
    }
    return [] if probe.returncode == 0 else [f"{name}_entrypoint_not_runnable"]


def _layout_check(
    *,
    imagenet_root: str | Path | None,
    imagenetc_root: str | Path | None,
    expected_classes: int,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    details: dict[str, Any] = {}
    clean = _path(imagenet_root)
    corrupt = _path(imagenetc_root)

    val = None if clean is None else clean / "val"
    val_classes = _class_dirs(val) if val else []
    details["clean_imagenet"] = {"root": str(clean) if clean else None, "val": str(val) if val else None, "class_count": len(val_classes)}
    if val is None or not val.is_dir():
        blockers.append("clean_imagenet_val_missing")
    elif len(val_classes) != expected_classes:
        blockers.append("clean_imagenet_class_count")

    corruption_details: dict[str, Any] = {}
    for corruption_name in OFFICIAL_CORRUPTIONS:
        level = None if corrupt is None else corrupt / corruption_name / "5"
        classes = _class_dirs(level) if level else []
        corruption_details[corruption_name] = {"path": str(level) if level else None, "class_count": len(classes)}
        if level is None or not level.is_dir():
            blockers.append(f"imagenetc_{corruption_name}_missing")
        elif len(classes) != expected_classes:
            blockers.append(f"imagenetc_{corruption_name}_class_count")
    details["imagenet_c"] = corruption_details
    return blockers, details


def check_task3_inputs(
    *,
    repo: str | Path,
    imagenet_root: str | Path | None,
    imagenetc_root: str | Path | None,
    poem_source: str | Path | None,
    aetta_source: str | Path | None,
    ttaline_source: str | Path | None,
    python_executable: str | Path | None,
    poem_python: str | Path | None = None,
    aetta_python: str | Path | None = None,
    expected_classes: int = 1000,
    require_clean_sources: bool = True,
    poem_sar_source: str | Path | None = None,
) -> dict[str, Any]:
    """Return ``READY`` only when all official-panel inputs are present."""

    blockers, details = _layout_check(
        imagenet_root=imagenet_root,
        imagenetc_root=imagenetc_root,
        expected_classes=expected_classes,
    )
    _, runtime_error = _python_command(python_executable)
    if runtime_error:
        blockers.append(runtime_error.replace("python_", "python_executable_", 1))

    for name, source in (("poem", poem_source), ("aetta", aetta_source), ("ttaline", ttaline_source)):
        blockers.extend(_source_status(name, source, require_clean=require_clean_sources))
    if poem_sar_source is not None:
        blockers.extend(_source_status("poem_sar_dependency", poem_sar_source, require_clean=True))
    if poem_python is not None:
        blockers.extend(_runtime_status("poem", poem_python, ("torch", "torchvision", "timm", "PIL")))
    if aetta_python is not None:
        blockers.extend(_runtime_status("aetta", aetta_python, ("torch", "torchvision", "numpy", "PIL")))
    for name, source, executable in (
        ("poem", poem_source, poem_python or python_executable),
        ("aetta", aetta_source, aetta_python or python_executable),
    ):
        if name == "poem" and require_clean_sources:
            # The pinned POEM commit imports ``models.Res`` but does not track
            # that package.  A local ignored shim can make ``--help`` pass;
            # it cannot be treated as native source provenance.
            dependency = _path(source) / "models" / "Res.py" if _path(source) else None
            dependency_relative = "models/Res.py"
            dependency_state = "tracked" if _tracked_source_file(source, dependency_relative) else (
                "untracked_or_omitted" if dependency and dependency.is_file() else "missing"
            )
            details["poem_native_dependency"] = {
                "path": dependency_relative,
                "state": dependency_state,
                "native_required": True,
            }
            if poem_sar_source is not None:
                details["poem_native_dependency"]["state"] = "explicit_pinned_bootstrap_pending_probe"
                details["poem_native_dependency"]["sar_source"] = str(_path(poem_sar_source))
            elif dependency_state != "tracked":
                blockers.append("poem_native_dependency_untracked")
        cuda_files = _unconditional_cuda_files(source)
        capabilities = _runtime_capabilities(executable)
        details[f"{name}_native_runtime"] = {
            "capabilities": capabilities,
            "unconditional_cuda_files": cuda_files,
            "native_cuda_required": bool(cuda_files),
        }
        if cuda_files:
            if capabilities is None:
                blockers.append(f"{name}_native_runtime_unverified")
            elif not capabilities["cuda"]:
                blockers.append(f"{name}_native_cuda_unavailable")
    # POEM and AETTA both expose main.py.  The pinned Agreement-on-the-Line
    # checkout is notebook/source material and intentionally has no assumed
    # native entrypoint; a Task-3 run must still supply a reviewed common
    # panel runner before any external-method result is promotable.
    poem_probe = _entrypoint_status("poem", poem_source, poem_python or python_executable,
                                    details=details, sar_source=poem_sar_source)
    blockers.extend(poem_probe)
    if poem_sar_source is not None and "poem_native_dependency" in details:
        details["poem_native_dependency"]["state"] = (
            "pinned_bootstrap_probe_failed" if poem_probe else "pinned_bootstrap_help_verified_not_benchmark"
        )
    blockers.extend(_entrypoint_status("aetta", aetta_source, aetta_python or python_executable, details=details))

    return {
        "schema": "kbound_task3_preflight_v1",
        "repo": str(_path(repo)),
        "status": "READY" if not blockers else "OPEN",
        "blockers": sorted(set(blockers)),
        "details": details,
    }


def check_natural_inputs(
    *,
    repo: str | Path,
    domainnet_root: str | Path | None,
    audit_path: str | Path | None,
    split_manifest: str | Path | None,
    checkpoint: str | Path | None,
) -> dict[str, Any]:
    """Return ``READY`` only for a complete, locked natural-study input set."""

    blockers: list[str] = []
    details: dict[str, Any] = {}
    root = _path(domainnet_root)
    if root is None or not root.is_dir():
        blockers.append("domainnet_root_missing")
    else:
        details["domainnet_root"] = str(root)

    audit = _path(audit_path)
    audit_doc: dict[str, Any] = {}
    if audit is None or not audit.is_file():
        blockers.append("domainnet_audit_missing")
    else:
        try:
            audit_doc = json.loads(audit.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("domainnet_audit_invalid")
        else:
            details["audit"] = {
                "path": str(audit),
                "integrity_status": audit_doc.get("integrity_status"),
                "total_images": audit_doc.get("total_images"),
            }
            if audit_doc.get("integrity_status") != "PASS" or not audit_doc.get("total_images"):
                blockers.append("domainnet_audit_not_pass")

    manifest = _path(split_manifest)
    if manifest is None or not manifest.is_file():
        blockers.append("natural_split_manifest_missing")
    else:
        try:
            manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("natural_split_manifest_invalid")
        else:
            details["split_manifest"] = {
                "path": str(manifest),
                "locked": bool(manifest_doc.get("lock", {}).get("locked")),
                "sha256": manifest_doc.get("lock", {}).get("sha256"),
            }
            if not manifest_doc.get("lock", {}).get("locked"):
                blockers.append("natural_split_manifest_unlocked")

    ckpt = _path(checkpoint)
    if ckpt is None or not ckpt.is_file():
        blockers.append("natural_checkpoint_missing")
    else:
        details["checkpoint"] = {"path": str(ckpt), "bytes": ckpt.stat().st_size}

    return {
        "schema": "kbound_natural_preflight_v1",
        "repo": str(_path(repo)),
        "status": "READY" if not blockers else "OPEN",
        "blockers": sorted(set(blockers)),
        "details": details,
    }


def _write_or_print(payload: dict[str, Any], output: str | Path | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        path = Path(output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task3", action="store_true")
    mode.add_argument("--natural", action="store_true")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--imagenet-root")
    parser.add_argument("--imagenetc-root")
    parser.add_argument("--poem-source")
    parser.add_argument("--poem-sar-source", help="Explicit pinned SAR source for authenticated POEM model dependency")
    parser.add_argument("--aetta-source")
    parser.add_argument("--ttaline-source")
    parser.add_argument("--python-executable")
    parser.add_argument("--poem-python")
    parser.add_argument("--aetta-python")
    parser.add_argument("--domainnet-root")
    parser.add_argument("--audit-path")
    parser.add_argument("--split-manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--expected-classes", type=int, default=1000)
    parser.add_argument("--allow-dirty-sources", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    if args.task3:
        payload = check_task3_inputs(
            repo=args.repo,
            imagenet_root=args.imagenet_root,
            imagenetc_root=args.imagenetc_root,
            poem_source=args.poem_source,
            poem_sar_source=args.poem_sar_source,
            aetta_source=args.aetta_source,
            ttaline_source=args.ttaline_source,
            python_executable=args.python_executable,
            poem_python=args.poem_python,
            aetta_python=args.aetta_python,
            expected_classes=args.expected_classes,
            require_clean_sources=not args.allow_dirty_sources,
        )
    else:
        payload = check_natural_inputs(
            repo=args.repo,
            domainnet_root=args.domainnet_root,
            audit_path=args.audit_path,
            split_manifest=args.split_manifest,
            checkpoint=args.checkpoint,
        )
    _write_or_print(payload, args.json_out)
    return 0 if payload["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
