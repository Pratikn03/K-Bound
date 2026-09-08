#!/usr/bin/env python3
"""Discover and run the preauthorized tracked repository verification surface.

The inventory is derived from fixed source-only pathspecs in the pinned Git tree,
never from an unrestricted whole-tree listing.  Immutable archives and exact
historical/timing writers are classified with machine-readable rationales.
Tests and validators that require protected So2Sat development authority are
recorded but run only after the caller supplies the explicit authorization flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import site
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from docs.research.kbound.scripts import release_privacy

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INVENTORY = (
    REPOSITORY_ROOT / "docs/research/kbound/audits/repository_test_inventory.json"
)
SCHEMA = "kbound-repository-test-inventory-v2"
PYTEST_PROCESS_MODEL = "one_module_per_fresh_interpreter"

# These files look like pytest modules but are retained historical writers or a
# benchmark utility.  Running either 3D-ADAM writer consumes already-opened
# target labels and mutates historical result files, so a release verification
# must classify rather than execute them.  New exceptions require source review.
EXPLICIT_TEST_EXCLUSIONS: dict[str, str] = {
    "docs/research/kbound/theory_v2/timing_test.py": (
        "timing_utility_without_assertions"
    ),
    "experiments/kbound/test_3dadam_bootstrap.py": (
        "historical_target_analysis_writer"
    ),
    "experiments/kbound/test_3dadam_namedcond.py": (
        "historical_target_analysis_writer"
    ),
}

# These tracked modules contain at least one test that reads the sealed So2Sat
# development authority from its real repository location.  They remain part
# of the complete inventory, but execution requires the explicit CLI opt-in.
PROTECTED_SO2SAT_TEST_MODULES = frozenset(
    {
        "tests/test_canonical_release_data.py",
        "tests/test_empirical_data_quality_audit_remediation.py",
        "tests/test_reconciled_panels.py",
        "tests/test_so2sat_numbers_builder.py",
        "tests/test_so2sat_prospective_v2.py",
    }
)
PROTECTED_SO2SAT_VALIDATORS = frozenset(
    {"docs/research/kbound/scripts/validate_canonical_release_data.py"}
)
PROTECTED_SO2SAT_EXCLUSION_REASON = "requires_explicit_protected_so2sat_authority"

EXPLICIT_VALIDATOR_EXCLUSIONS: dict[str, str] = {
    "docs/research/kbound/scripts/validate_closure_seed.py": "requires_run_specific_input",
    "docs/research/kbound/scripts/validate_pacs_replay.py": "requires_run_specific_input",
}

SCRIPT_VALIDATORS = frozenset(
    {
        "docs/research/kbound/scripts/validate_canonical_release_data.py",
        "docs/research/kbound/scripts/validate_cifar10c_sar_rebuild.py",
        "docs/research/kbound/scripts/validate_closure_protocol.py",
        "src/scripts/validate_manuscript_claims.py",
    }
)

VALIDATOR_PREFIXES = (
    "experiments/kbound/theory_validation/",
    "docs/research/kbound/theory_v2/",
    "docs/research/kbound/audit_validation/",
    "docs/research/kbound/ttc_extension/",
    "docs/research/kbound/gapclose_wave5/",
)

IMMUTABLE_ARCHIVE_PREFIXES = (
    "archive/",
    "docs/research/kbound/archive/",
)

FIRST_PARTY_PATHS = (
    "kga",
    "deploy/api",
    "scripts",
    "src",
    "tests",
    "experiments/kbound",
    "docs/research/kbound",
)

BYTE_PRESERVING_ARCHIVE_ROOTS = (
    "docs/research/kbound/archive/superseded_empirical_authorities_2026-09-02",
    "docs/research/kbound/archive/stale_publication_builds_2026-09-02",
    "docs/research/kbound/archive/legacy_publication_surfaces_2026-09-02",
)

RUFF_ARCHIVE_EXCLUDE_ARGS = (
    "--force-exclude",
    *(
        argument
        for root in BYTE_PRESERVING_ARCHIVE_ROOTS
        for argument in ("--extend-exclude", root)
    ),
)
RUFF_FORMAT_ARCHIVE_EXCLUDE_ARGS = (
    "--force-exclude",
    *(
        argument
        for root in BYTE_PRESERVING_ARCHIVE_ROOTS
        for argument in ("--exclude", root)
    ),
)

APLUS_PATHS = (
    "kga",
    "src/scripts/kbound/make_synth_archive.py",
    "src/scripts/kbound/smoke_trichotomy.py",
    "deploy/api/kga_routes.py",
    "kga/experiment_contract.py",
    "docs/research/kbound/scripts/validate_closure_protocol.py",
    "docs/research/kbound/scripts/run_closure_stage.py",
    "tests/test_kga_package.py",
    "tests/test_kga_experiment_contract.py",
    "tests/test_smoke_trichotomy.py",
)

PYTEST_BACKED_RELEASE_GATE_MODULES = frozenset(
    {
        "tests/test_production_release_hygiene.py",
        "tests/test_reproducibility_hygiene.py",
        "tests/test_release_privacy.py",
    }
)

# Scan only distributable runtime/operator sources. This keeps the release
# privacy gate real while deliberately avoiding experiment/result trees and
# adversarial test fixtures, both of which are outside the public source scope.
PRIVATE_PATH_SCAN_ROOTS = (
    "deploy/api",
    "docs/research/kbound/runbooks",
    "kga",
    "scripts",
    "src/scripts",
)
PRIVATE_PATH_SCAN_SUFFIXES = frozenset(
    {".py", ".sh", ".toml", ".yaml", ".yml", ".md", ".txt"}
)

# These positive Git roots contain maintained implementation, test, or validator
# source and no experiment data/result subtree.  The default inventory must not
# first enumerate the entire tree and then filter protected names in Python.
TRACKED_VERIFICATION_PATHSPECS = (
    "deploy/api",
    "docs/research/kbound/audit_validation",
    "docs/research/kbound/edge",
    "docs/research/kbound/gapclose_wave5",
    "docs/research/kbound/kbound_pkg",
    "docs/research/kbound/kbound_repro",
    "docs/research/kbound/scripts",
    "docs/research/kbound/tests",
    "docs/research/kbound/theory_v2",
    "docs/research/kbound/ttc_extension",
    "experiments/kbound/conj1_validator.py",
    "experiments/kbound/domainnet/__init__.py",
    "experiments/kbound/domainnet/source_data.py",
    "experiments/kbound/domainnet/train_source.py",
    "experiments/kbound/domainnet/pilot_data.py",
    "experiments/kbound/domainnet/pilot_data_v2.py",
    "experiments/kbound/domainnet/pilot_candidate.py",
    "experiments/kbound/domainnet/pilot_analysis.py",
    "experiments/kbound/domainnet/pilot_runner.py",
    "experiments/kbound/test_3dadam_bootstrap.py",
    "experiments/kbound/test_3dadam_namedcond.py",
    "experiments/kbound/theory_validation",
    "kga",
    "scripts",
    "src",
    "tests",
)


@dataclass(frozen=True)
class GateSpec:
    """A required release gate shown in the machine-readable inventory."""

    name: str
    required: bool = True


@dataclass(frozen=True)
class PytestProcessGroup:
    """One tracked test module executed in a fresh Python interpreter."""

    group_id: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class PytestRunReceipt:
    """Auditable result of executing the complete pytest process plan."""

    accepted_skips: tuple[dict[str, str], ...]
    executed_group_ids: tuple[str, ...]


# Each entry covers only an inverse optional-runtime branch or a later mandatory
# release phase. Missing scientific evidence and missing required runtimes are
# intentionally absent and therefore fail the release.
ALLOWED_PYTEST_SKIPS: dict[tuple[str, str], str] = {
    (
        "docs.research.kbound.kbound_repro.tests.test_runtime_paths_deps.test_resolve_device_defers_missing_torch",
        "torch is installed in this environment",
    ): "inverse optional-dependency branch is exercised only when torch is absent",
    (
        "docs.research.kbound.kbound_pkg.tests.test_kga.TestKGA.test_raises_importerror_without_torch",
        "torch is installed; cannot test torch-absent path here",
    ): "inverse optional-dependency branch is exercised only when torch is absent",
    (
        "docs.research.kbound.tests.test_cifar_tent_collection.test_dataset_base_is_object_without_torch",
        "torch present: base is torch.utils.data.Dataset",
    ): "inverse optional-dependency branch is exercised only when torch is absent",
    (
        "docs.research.kbound.tests.test_cifar_tent_collection.test_torch_only_path_raises_named_importerror",
        "torch present: the torch-only path runs normally",
    ): "inverse optional-dependency branch is exercised only when torch is absent",
    (
        "docs.research.kbound.tests.test_cifar_tent_collection.test_require_torch_is_not_sys_exit",
        "torch present",
    ): "inverse optional-dependency branch is exercised only when torch is absent",
    (
        "tests.test_kbound_pdf_build_isolation.test_resident_latex_toolchain_does_not_reuse_old_auxiliary_files",
        "A local TeX/Poppler toolchain is not installed",
    ): "the later mandatory PDF phase executes and verifies the resident TeX/Poppler toolchain",
    (
        "docs.research.kbound.edge.tests.test_dashboard_render.test_video_writer",
        "no mp4 codec available in this environment",
    ): "optional MP4 recording is not a publication artifact and still-image rendering is tested",
    (
        "tests.test_model_governance.test_model_versions_endpoint",
        "models/MANIFEST.json not present locally",
    ): "the release image is intentionally model-agnostic and the empty-manifest endpoint is tested",
    (
        "tests.test_model_governance.test_model_rollback_rejects_unknown_version",
        "models/MANIFEST.json not present locally",
    ): "the release image is intentionally model-agnostic and the empty-manifest endpoint is tested",
    (
        "tests.test_model_governance.test_model_rollback_applies_default",
        "models/MANIFEST.json not present locally",
    ): "the release image is intentionally model-agnostic and the empty-manifest endpoint is tested",
    (
        "tests.test_model_governance.test_model_versions_endpoint_empty_without_manifest",
        "local manifest present",
    ): "the mutually exclusive manifest-present endpoint and rollback tests execute instead",
}

ALLOWED_PYTEST_WARNINGS: tuple[dict[str, str], ...] = (
    {
        "filter": (
            "ignore:Using `httpx` with `starlette.testclient` is deprecated:UserWarning"
        ),
        "scope": "third-party FastAPI/Starlette TestClient compatibility import",
        "rationale": "the production API and offline container gates do not use TestClient",
    },
    {
        "filter": ("ignore:Warning:UserWarning:numpy.ma.core"),
        "scope": "adversarial masked-array rejection tests",
        "rationale": "the warning is the intentionally malformed value being rejected by the tests",
    },
    {
        "filter": (
            "ignore:'crypt' is deprecated and slated for removal in Python 3.13:DeprecationWarning"
        ),
        "scope": "third-party passlib import used only by API authentication tests",
        "rationale": (
            "the production API is Python 3.11-pinned and this third-party warning is not emitted by KGA code"
        ),
    },
)


class InventoryError(RuntimeError):
    """The tracked verification surface cannot be classified safely."""


_UF_DATALESS = 0x40000000


def _is_test_shaped(relative: str) -> bool:
    path = PurePosixPath(relative)
    return path.suffix == ".py" and (
        path.name.startswith("test_") or path.name.endswith("_test.py")
    )


def _is_archive(relative: str) -> bool:
    canonical = PurePosixPath(relative).as_posix()
    return any(canonical.startswith(prefix) for prefix in IMMUTABLE_ARCHIVE_PREFIXES)


def classify_test_paths(
    tracked_paths: Iterable[str],
    *,
    require_exception_classification: bool = True,
    authorize_protected_so2sat: bool = False,
) -> dict[str, Any]:
    """Classify every test-shaped tracked path exactly once."""

    pytest_paths: list[str] = []
    excluded: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in sorted(tracked_paths):
        relative = PurePosixPath(raw).as_posix()
        if relative in seen or not _is_test_shaped(relative):
            continue
        seen.add(relative)
        if _is_archive(relative):
            excluded.append({"path": relative, "reason": "immutable_archive"})
            continue
        if relative in PROTECTED_SO2SAT_TEST_MODULES and not authorize_protected_so2sat:
            excluded.append(
                {"path": relative, "reason": PROTECTED_SO2SAT_EXCLUSION_REASON}
            )
            continue
        reason = EXPLICIT_TEST_EXCLUSIONS.get(relative)
        if reason is not None:
            excluded.append({"path": relative, "reason": reason})
            continue
        if require_exception_classification and relative.endswith("/timing_test.py"):
            raise InventoryError(
                f"unclassified test-shaped path requires an explicit rationale: {relative}"
            )
        pytest_paths.append(relative)
    classified = len(pytest_paths) + len(excluded)
    if classified != len(seen):  # defensive invariant
        raise InventoryError("test-shaped inventory is not a complete partition")
    return {
        "test_shaped_count": len(seen),
        "classified_count": classified,
        "pytest_count": len(pytest_paths),
        "excluded_count": len(excluded),
        "pytest_paths": pytest_paths,
        "excluded_paths": excluded,
    }


def _is_validator_candidate(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        path.suffix == ".py"
        and (
            path.name.startswith(("val_", "validate_"))
            or path.name.endswith("_validator.py")
        )
        and (
            relative.startswith("docs/research/kbound/")
            or relative.startswith("experiments/kbound/")
            or relative.startswith("src/scripts/")
        )
    )


def _is_release_validator(relative: str) -> bool:
    return (
        relative in SCRIPT_VALIDATORS
        or relative == "experiments/kbound/conj1_validator.py"
        or any(relative.startswith(prefix) for prefix in VALIDATOR_PREFIXES)
    )


def classify_validator_paths(
    tracked_paths: Iterable[str],
    *,
    authorize_protected_so2sat: bool = False,
) -> dict[str, Any]:
    """Partition every maintained validator into executable or explained."""

    validators: list[str] = []
    excluded: list[dict[str, str]] = []
    candidates: set[str] = set()
    for raw in sorted(tracked_paths):
        relative = PurePosixPath(raw).as_posix()
        if relative in candidates or not _is_validator_candidate(relative):
            continue
        candidates.add(relative)
        if _is_archive(relative):
            excluded.append({"path": relative, "reason": "immutable_archive"})
        elif relative in PROTECTED_SO2SAT_VALIDATORS and not authorize_protected_so2sat:
            excluded.append(
                {"path": relative, "reason": PROTECTED_SO2SAT_EXCLUSION_REASON}
            )
        elif relative in EXPLICIT_VALIDATOR_EXCLUSIONS:
            excluded.append(
                {"path": relative, "reason": EXPLICIT_VALIDATOR_EXCLUSIONS[relative]}
            )
        elif _is_release_validator(relative):
            validators.append(relative)
        else:
            raise InventoryError(f"unclassified validator requires review: {relative}")
    return {
        "validator_candidate_count": len(candidates),
        "validator_count": len(validators),
        "excluded_validator_count": len(excluded),
        "validator_paths": validators,
        "excluded_validators": excluded,
    }


def release_gate_plan(*, python: str) -> tuple[GateSpec, ...]:
    """Return the required Task 7 surface in deterministic execution order."""

    del python  # The interpreter is an execution parameter, not gate identity.
    return tuple(
        GateSpec(name)
        for name in (
            "pytest",
            "standalone-validators",
            "immutable-hook-configuration",
            "ruff-critical",
            "ruff-lint",
            "ruff-format",
            "mypy",
            "bandit",
            "private-path-scan",
            "package-build-install-cli",
            "dashboard-lockfile-build",
            "lean-kbound",
            "lean-multiclass-vector-capacity",
            "docker-build",
            "docker-health-network-none",
            "docker-content-network-none",
        )
    )


def canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(document),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def inventory_digest(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("inventory_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def pytest_process_groups(paths: Sequence[str]) -> tuple[PytestProcessGroup, ...]:
    """Plan one deterministic fresh interpreter for every tracked module.

    Native runtimes such as OpenMP can leave process-global state behind after
    Torch or SciPy imports.  A later subprocess-heavy module must therefore not
    inherit the pytest interpreter that initialized that state.  A module is
    the smallest auditable isolation unit: it prevents cross-module leakage
    while preserving each module's fixtures and collection semantics.
    """

    ordered = sorted(paths)
    if len(ordered) != len(set(ordered)):
        duplicates = sorted(path for path in set(ordered) if ordered.count(path) > 1)
        raise InventoryError(f"duplicate pytest module in process plan: {duplicates}")
    groups = tuple(
        PytestProcessGroup(
            group_id=(
                f"pytest-module-{index:04d}-{hashlib.sha256(path.encode('utf-8')).hexdigest()[:16]}"
            ),
            paths=(path,),
        )
        for index, path in enumerate(ordered, start=1)
    )
    identifiers = [group.group_id for group in groups]
    if len(identifiers) != len(set(identifiers)):
        raise InventoryError("pytest process-group identifiers are not unique")
    return groups


def pytest_process_group_records(
    groups: Sequence[PytestProcessGroup],
) -> list[dict[str, Any]]:
    """Return the canonical JSON representation of a pytest process plan."""

    return [
        {
            "group_id": group.group_id,
            "module_count": len(group.paths),
            "pytest_paths": list(group.paths),
        }
        for group in groups
    ]


def verify_pytest_process_group_records(
    paths: Sequence[str], records: Sequence[Mapping[str, Any]]
) -> None:
    """Require the recorded groups to cover every module exactly once."""

    expected = pytest_process_group_records(pytest_process_groups(paths))
    if list(records) != expected:
        raise InventoryError(
            "pytest process-group coverage is not the canonical exact-once partition"
        )


def verify_pytest_backed_release_gate_coverage(paths: Sequence[str]) -> None:
    """Ensure pytest-backed release checks are in the exact-once module plan."""

    missing = sorted(PYTEST_BACKED_RELEASE_GATE_MODULES - set(paths))
    if missing:
        raise InventoryError(
            f"pytest-backed release gate modules are missing: {missing}"
        )


def _verify_executed_group_ids(
    groups: Sequence[PytestProcessGroup], executed_group_ids: Sequence[str]
) -> None:
    expected = [group.group_id for group in groups]
    if list(executed_group_ids) != expected:
        raise InventoryError(
            "executed pytest process-group coverage is not complete and exact-once"
        )


def build_inventory_payload(
    *,
    tracked_paths: Iterable[str],
    source_commit: str,
    source_tree: str,
    authorize_protected_so2sat: bool = False,
) -> dict[str, Any]:
    tracked = list(tracked_paths)
    inventory = classify_test_paths(
        tracked,
        authorize_protected_so2sat=authorize_protected_so2sat,
    )
    validators = classify_validator_paths(
        tracked,
        authorize_protected_so2sat=authorize_protected_so2sat,
    )
    process_groups = pytest_process_groups(inventory["pytest_paths"])
    process_group_records = pytest_process_group_records(process_groups)
    verify_pytest_process_group_records(
        inventory["pytest_paths"], process_group_records
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE_CLASSIFICATION",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "protected_so2sat_authorized": authorize_protected_so2sat,
        **inventory,
        **validators,
        "pytest_process_model": PYTEST_PROCESS_MODEL,
        "pytest_process_group_count": len(process_groups),
        "pytest_process_groups": process_group_records,
        "pytest_process_coverage_status": "PLANNED_COMPLETE_EXACTLY_ONCE",
        "release_gates": [
            {"name": gate.name, "required": gate.required}
            for gate in release_gate_plan(python=sys.executable)
        ],
        "execution_status": "NOT_RUN",
        "private_path_scan": {
            "status": "NOT_RUN",
            "scope_roots": list(PRIVATE_PATH_SCAN_ROOTS),
        },
        "unexpected_skips": [],
        "unexpected_warnings": [],
        "allowed_pytest_warnings": [dict(row) for row in ALLOWED_PYTEST_WARNINGS],
    }
    payload["inventory_sha256"] = inventory_digest(payload)
    return payload


def write_canonical_json(path: Path, document: Mapping[str, Any]) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    if any(Path(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1)):
        raise InventoryError(f"refusing to write inventory through a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(document)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except FileExistsError as exc:
        raise InventoryError(
            f"refusing to reuse inventory staging path: {temporary}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _is_dataless(path: Path) -> bool:
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and (
        (info.st_size > 0 and getattr(info, "st_blocks", 1) == 0)
        or bool(getattr(info, "st_flags", 0) & _UF_DATALESS)
    )


def _resident_head_oid(git_dir: Path) -> str:
    """Resolve HEAD without asking Git to hydrate cloud-only metadata."""

    try:
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if head.startswith("ref: "):
            ref_path = git_dir / head.removeprefix("ref: ")
            if _is_dataless(ref_path):
                raise InventoryError(f"Git HEAD ref is not resident: {ref_path}")
            head = ref_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise InventoryError("could not resolve resident Git HEAD") from exc
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise InventoryError("resident Git HEAD is not a full object ID")
    return head


def _has_dataless_loose_objects(objects_dir: Path) -> bool:
    """Detect cloud placeholders that can shadow a resident packed object."""

    try:
        for source_dir in objects_dir.iterdir():
            if (
                re.fullmatch(r"[0-9a-f]{2}", source_dir.name) is None
                or not source_dir.is_dir()
            ):
                continue
            for source in source_dir.iterdir():
                if source.is_file() and _is_dataless(source):
                    return True
    except OSError:
        return False
    return False


def _has_dataless_git_metadata(git_dir: Path) -> bool:
    """Detect Git metadata whose hydration can block a read-only command."""

    objects_dir = git_dir / "objects"
    candidates = [
        git_dir / "config",
        git_dir / "packed-refs",
        objects_dir / "info" / "commit-graph",
        objects_dir / "pack" / "multi-pack-index",
    ]
    commit_graphs = objects_dir / "info" / "commit-graphs"
    try:
        if commit_graphs.is_dir():
            candidates.extend(
                path for path in commit_graphs.iterdir() if path.is_file()
            )
    except OSError:
        pass
    return any(_is_dataless(path) for path in candidates)


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    git_dir = repo / ".git"
    command = ["git", *args]
    cwd = repo
    temporary_view: tempfile.TemporaryDirectory[str] | None = None
    objects_dir = git_dir / "objects"
    if git_dir.is_dir() and (
        _has_dataless_git_metadata(git_dir) or _has_dataless_loose_objects(objects_dir)
    ):
        # The worktree can be fully resident while iCloud has evicted Git
        # metadata or loose objects. A cloud-only loose object shadows the
        # same resident object in a pack and can block even read-only Git
        # commands. Build a temporary view containing resident objects only.
        head_oid = _resident_head_oid(git_dir)
        temporary_view = tempfile.TemporaryDirectory(prefix="kbound-git-view.")
        view = Path(temporary_view.name)
        (view / "objects" / "pack").mkdir(parents=True)
        (view / "refs").mkdir()
        (view / "HEAD").write_text(head_oid + "\n", encoding="ascii")
        (view / "config").write_text(
            "[core]\n\trepositoryformatversion = 0\n\tbare = true\n",
            encoding="ascii",
        )
        source_objects = objects_dir
        for source in (source_objects / "pack").iterdir():
            if source.is_file() and not _is_dataless(source):
                os.link(source, view / "objects" / "pack" / source.name)
        for source_dir in source_objects.iterdir():
            if (
                re.fullmatch(r"[0-9a-f]{2}", source_dir.name) is None
                or not source_dir.is_dir()
                or _is_dataless(source_dir)
            ):
                continue
            target_dir = view / "objects" / source_dir.name
            target_dir.mkdir()
            for source in source_dir.iterdir():
                if source.is_file() and not _is_dataless(source):
                    os.link(source, target_dir / source.name)
        command = ["git", f"--git-dir={view}", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=not binary,
            timeout=30,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise InventoryError(f"Git inventory command failed: {' '.join(args)}") from exc
    finally:
        if temporary_view is not None:
            temporary_view.cleanup()
    stdout = completed.stdout
    if not isinstance(stdout, (str, bytes)):
        raise InventoryError("git did not return a text or binary payload")
    return stdout


def tracked_paths(repo: Path, revision: str = "HEAD") -> list[str]:
    """Return source-only verification paths from an immutable Git revision."""

    raw = _git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        revision,
        "--",
        *TRACKED_VERIFICATION_PATHSPECS,
        binary=True,
    )
    assert isinstance(raw, bytes)
    return sorted(item.decode("utf-8") for item in raw.split(b"\0") if item)


def source_identity(repo: Path) -> tuple[str, str]:
    commit = str(_git(repo, "rev-parse", "--verify", "HEAD^{commit}")).strip()
    tree = str(_git(repo, "rev-parse", "--verify", "HEAD^{tree}")).strip()
    return commit, tree


def verify_expected_source_commit(*, actual: str, expected: str) -> str:
    """Bind an inventory to an exact workflow checkout, never an abbreviated ref."""

    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise InventoryError(
            "expected source commit must be a full lowercase 40-character Git object ID"
        )
    if actual != expected:
        raise InventoryError(
            f"expected source commit {expected} does not match HEAD {actual}"
        )
    return actual


def run_pytest_modules(paths: Sequence[str], *, python: str) -> int:
    if not paths:
        raise InventoryError("tracked pytest inventory is empty")
    first_failure = 0
    for group in pytest_process_groups(paths):
        command = [python, "-m", "pytest", "-q", *group.paths]
        returncode = subprocess.run(command, check=False).returncode
        if returncode != 0 and first_failure == 0:
            first_failure = returncode
    return first_failure


def validate_pytest_report(path: Path) -> list[dict[str, str]]:
    """Reject every runtime skip without an exact, non-load-bearing policy."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise InventoryError(
            f"pytest did not produce a valid JUnit receipt: {exc}"
        ) from exc
    accepted: list[dict[str, str]] = []
    for testcase in root.iter("testcase"):
        skipped = testcase.find("skipped")
        if skipped is None:
            continue
        scope = f"{testcase.get('classname', '')}.{testcase.get('name', '')}".strip(".")
        reason = skipped.get("message", "")
        rationale = ALLOWED_PYTEST_SKIPS.get((scope, reason))
        if rationale is None:
            raise InventoryError(f"unexpected pytest skip: {scope}: {reason}")
        accepted.append({"scope": scope, "reason": reason, "rationale": rationale})
    return accepted


def _run_pytest_release(
    paths: Sequence[str], *, repo: Path, python: str
) -> PytestRunReceipt:
    if not paths:
        raise InventoryError("tracked pytest inventory is empty")
    groups = pytest_process_groups(paths)
    with tempfile.TemporaryDirectory(prefix="kbound-pytest-") as temporary:
        accepted_skips: list[dict[str, str]] = []
        executed_group_ids: list[str] = []
        failures: list[str] = []
        report_failures: list[str] = []
        for group in groups:
            report = Path(temporary) / f"{group.group_id}.xml"
            command = [
                python,
                "-m",
                "pytest",
                "-q",
                "-W",
                "error",
            ]
            for row in ALLOWED_PYTEST_WARNINGS:
                command.extend(["-W", row["filter"]])
            command.extend(["--junitxml", str(report), *group.paths])
            result = subprocess.run(command, cwd=repo, check=False)
            executed_group_ids.append(group.group_id)
            if result.returncode != 0:
                failures.append(
                    f"{group.group_id} ({group.paths[0]}) failed with exit {result.returncode}"
                )
            try:
                accepted_skips.extend(validate_pytest_report(report))
            except InventoryError as exc:
                report_failures.append(f"{group.group_id} ({group.paths[0]}): {exc}")

        _verify_executed_group_ids(groups, executed_group_ids)
        if failures or report_failures:
            details = "; ".join([*failures, *report_failures])
            raise InventoryError(f"pytest process groups failed: {details}")
        return PytestRunReceipt(
            accepted_skips=tuple(accepted_skips),
            executed_group_ids=tuple(executed_group_ids),
        )


def _run(
    command: Sequence[str],
    *,
    repo: Path,
    cwd: Path | None = None,
    unset_env: Sequence[str] = (),
) -> None:
    environment = os.environ.copy()
    environment["PYTHONWARNINGS"] = "error"
    for name in unset_env:
        environment.pop(name, None)
    try:
        subprocess.run(
            list(command),
            cwd=cwd or repo,
            env=environment,
            check=True,
        )
    except FileNotFoundError as exc:
        raise InventoryError(
            f"required release tool is unavailable: {command[0]}"
        ) from exc


def _run_validators(paths: Sequence[str], *, repo: Path, python: str) -> None:
    if not paths:
        raise InventoryError("tracked standalone validator inventory is empty")
    for relative in paths:
        path = repo / relative
        if not path.is_file():
            raise InventoryError(
                f"tracked validator is absent from the checkout: {relative}"
            )
        _run([python, relative], repo=repo)


def run_private_path_scan(*, repo: Path) -> dict[str, Any]:
    """Scan the public runtime/operator source surface and return a receipt.

    The release plan used to advertise this gate without executing it.  The
    bounded source roots are intentional: scanning result/dataset trees would
    both exceed the public source scope and could dereference protected
    scientific artifacts.  Symlinks fail closed instead of extending the scan.
    """

    scanned: list[str] = []
    for root_relative in PRIVATE_PATH_SCAN_ROOTS:
        root = repo / root_relative
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise InventoryError(
                f"private-path scan root is missing or a symlink: {root_relative}"
            )
        for candidate in sorted(root.rglob("*")):
            if candidate.is_symlink():
                raise InventoryError(
                    f"private-path scan refuses symlink: {candidate.relative_to(repo)}"
                )
            if (
                not candidate.is_file()
                or candidate.suffix.lower() not in PRIVATE_PATH_SCAN_SUFFIXES
            ):
                continue
            relative = candidate.relative_to(repo).as_posix()
            try:
                release_privacy.scan_member(relative, candidate.read_bytes())
            except (OSError, release_privacy.PrivacyError) as exc:
                raise InventoryError(
                    f"private-path scan failed for {relative}: {exc}"
                ) from exc
            scanned.append(relative)
    if not scanned:
        raise InventoryError(
            "private-path scan found no public runtime/operator source files"
        )
    return {
        "status": "PASS",
        "scope_roots": list(PRIVATE_PATH_SCAN_ROOTS),
        "scanned_file_count": len(scanned),
        "scanned_paths_sha256": hashlib.sha256(
            "\n".join(scanned).encode("utf-8")
        ).hexdigest(),
    }


# Bandit 1.9.4 passes a deprecated no-op argument to pinned stevedore 5.9.1.
# Keep _run's warnings-as-errors policy; allow only this warning in the Bandit
# child. Python -W uses prefix matching, so use anchored filterwarnings regexes.
BANDIT_COMPATIBILITY_BOOTSTRAP = (
    "import runpy, warnings; "
    "warnings.filterwarnings('ignore', "
    "message=r'(?-i:\\AThe verify_requirements argument is now a no-op and is "
    "deprecated for removal\\. Remove the argument from calls\\.\\Z)', "
    "category=DeprecationWarning, module=r'\\Astevedore\\.extension\\Z'); "
    "runpy.run_module('bandit', run_name='__main__', alter_sys=True)"
)


def _run_quality_gates(*, repo: Path, python: str) -> None:
    commands = (
        [
            python,
            "-m",
            "ruff",
            "check",
            *RUFF_ARCHIVE_EXCLUDE_ARGS,
            "--select",
            "E9,F63,F7,F82",
            *FIRST_PARTY_PATHS,
        ],
        [python, "-m", "ruff", "check", *RUFF_ARCHIVE_EXCLUDE_ARGS, *APLUS_PATHS],
        [
            python,
            "-m",
            "ruff",
            "format",
            "--check",
            *RUFF_FORMAT_ARCHIVE_EXCLUDE_ARGS,
            *APLUS_PATHS,
        ],
        [python, "-m", "mypy", "kga", "--ignore-missing-imports"],
        [
            python,
            "-c",
            BANDIT_COMPATIBILITY_BOOTSTRAP,
            "-q",
            "-r",
            "kga",
            "deploy/api",
            "-c",
            "pyproject.toml",
        ],
    )
    for command in commands:
        _run(command, repo=repo)


def _copy_package_source(repo: Path, destination: Path) -> None:
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "CITATION.cff",
    ):
        shutil.copy2(repo / name, destination / name)
    shutil.copytree(repo / "kga", destination / "kga")


def validate_distribution_artifacts(paths: Sequence[Path]) -> tuple[Path, Path]:
    """Require the exact wheel/sdist pair expected from ``python -m build``."""

    wheels = [path for path in paths if path.suffix == ".whl"]
    sources = [path for path in paths if path.name.endswith(".tar.gz")]
    if len(paths) != 2 or len(wheels) != 1 or len(sources) != 1:
        raise InventoryError(
            "package build must produce exactly one wheel and one source archive"
        )
    return wheels[0], sources[0]


def _run_package_gate(*, repo: Path, python: str) -> None:
    with tempfile.TemporaryDirectory(prefix="kbound-package-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        _copy_package_source(repo, source)
        dist = root / "dist"
        # Exact locked setuptools supplies bdist_wheel. Avoid build isolation
        # (and therefore network resolution); artifacts are checked below.
        _run(
            [
                python,
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                str(dist),
                str(source),
            ],
            repo=repo,
        )
        artifacts = sorted(dist.iterdir())
        wheel, _source_archive = validate_distribution_artifacts(artifacts)
        _run([python, "-m", "twine", "check", *map(str, artifacts)], repo=repo)
        environment = root / "venv"
        _run(
            [python, "-m", "venv", "--system-site-packages", str(environment)],
            repo=repo,
        )
        installed_python = environment / "bin" / "python"
        child_site_packages = (
            environment
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        child_site_packages.mkdir(parents=True, exist_ok=True)
        locked_parent_sites = sorted(
            str(Path(candidate).resolve())
            for candidate in site.getsitepackages()
            if Path(candidate).is_dir()
        )
        if not locked_parent_sites:
            raise InventoryError(
                "locked parent environment exposes no site-packages directory"
            )
        (child_site_packages / "kbound-locked-parent-environment.pth").write_text(
            "".join(f"{candidate}\n" for candidate in locked_parent_sites),
            encoding="utf-8",
        )
        _run(
            [
                str(installed_python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                str(wheel),
            ],
            repo=repo,
            cwd=root,
            unset_env=("PYTHONPATH", "PYTHONHOME"),
        )
        _run(
            [
                str(installed_python),
                "-c",
                (
                    "from pathlib import Path; import kga; "
                    f"expected=Path({str(child_site_packages)!r}).resolve(); "
                    "observed=Path(kga.__file__).resolve(); "
                    "assert observed.is_relative_to(expected), (observed, expected)"
                ),
            ],
            repo=repo,
            cwd=root,
            unset_env=("PYTHONPATH", "PYTHONHOME"),
        )
        _run(
            [str(installed_python), "-m", "kga", "--help"],
            repo=repo,
            cwd=root,
            unset_env=("PYTHONPATH", "PYTHONHOME"),
        )
        _run(
            [str(environment / "bin" / "kga"), "--help"],
            repo=repo,
            cwd=root,
            unset_env=("PYTHONPATH", "PYTHONHOME"),
        )


def _run_dashboard_gate(*, repo: Path) -> None:
    dashboard = repo / "docs/research/kbound/dashboard"
    with tempfile.TemporaryDirectory(prefix="kbound-dashboard-") as temporary:
        temporary_root = Path(temporary)
        staged = temporary_root / "dashboard"
        npm_cache = temporary_root / "npm-cache"
        shutil.copytree(
            dashboard, staged, ignore=shutil.ignore_patterns("node_modules")
        )
        _run(
            [
                "npm",
                "ci",
                "--cache",
                str(npm_cache),
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
            ],
            repo=repo,
            cwd=staged,
        )
        _run(["npm", "run", "build"], repo=repo, cwd=staged)
        expected: set[Path] = set()
        for source in (staged / "src").rglob("*.ts"):
            relative = source.relative_to(staged / "src").with_suffix(".js")
            expected.update({relative, Path(f"{relative}.map")})
        actual = {
            path.relative_to(staged / "js")
            for path in (staged / "js").rglob("*")
            if path.is_file()
        }
        if not expected:
            raise InventoryError("dashboard build produced no JavaScript")
        if actual != expected:
            unexpected = sorted(map(str, actual - expected))
            missing = sorted(map(str, expected - actual))
            raise InventoryError(
                f"unexpected or missing dashboard outputs: unexpected={unexpected}, missing={missing}"
            )
        for relative in sorted(expected):
            rebuilt = staged / "js" / relative
            committed = dashboard / "js" / relative
            if (
                not committed.is_file()
                or rebuilt.read_bytes() != committed.read_bytes()
            ):
                raise InventoryError(
                    f"dashboard build is stale or nondeterministic: js/{relative}"
                )


def _run_lean_gates(*, repo: Path, python: str) -> None:
    receipt = repo / "docs/research/kbound/audits/formal_foundations_2026_08_31.json"
    _run(
        [
            "bash",
            "docs/research/kbound/formal/build.sh",
            "--json-out",
            str(receipt),
        ],
        repo=repo,
    )
    multiclass = repo / "docs/research/multiclass_vector_capacity/formal"
    _run(["lake", "--no-cache", "--wfail", "build"], repo=repo, cwd=multiclass)
    _run([python, "-B", "export_inventory.py"], repo=repo, cwd=multiclass)


def _run_docker_gates(*, repo: Path, source_commit: str) -> None:
    tag = f"kbound-api:release-{source_commit[:12]}"
    _run(["docker", "build", "--platform", "linux/amd64", "--tag", tag, "."], repo=repo)
    _run(
        [
            "docker",
            "run",
            "--network",
            "none",
            "--rm",
            "--entrypoint",
            "python",
            tag,
            "-c",
            "import asyncio; from deploy.api.main import health; result=asyncio.run(health()); assert result['status'] == 'ok', result",
        ],
        repo=repo,
    )
    _run(
        [
            "docker",
            "run",
            "--network",
            "none",
            "--rm",
            "--entrypoint",
            "python",
            tag,
            "-c",
            "from pathlib import Path; assert not Path('/app/src').exists(); assert not Path('/app/experiments').exists(); assert not Path('/app/docs').exists()",
        ],
        repo=repo,
    )


def run_all_gates(
    payload: Mapping[str, Any], *, repo: Path, python: str
) -> PytestRunReceipt:
    """Execute every required Task 7 gate, failing at the first bad boundary."""

    paths = payload["pytest_paths"]
    records = payload.get("pytest_process_groups")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise InventoryError("pytest process-group coverage record is missing")
    verify_pytest_process_group_records(paths, records)
    verify_pytest_backed_release_gate_coverage(paths)
    pytest_receipt = _run_pytest_release(paths, repo=repo, python=python)
    _run_validators(payload["validator_paths"], repo=repo, python=python)
    _run_quality_gates(repo=repo, python=python)
    private_path_scan = run_private_path_scan(repo=repo)
    if isinstance(payload, dict):
        # A PASS is written only after the scanner actually completes; failed
        # executions leave the inventory's NOT_RUN record intact.
        payload["private_path_scan"] = private_path_scan
    _run_package_gate(repo=repo, python=python)
    _run_dashboard_gate(repo=repo)
    _run_lean_gates(repo=repo, python=python)
    _run_docker_gates(repo=repo, source_commit=str(payload["source_commit"]))
    return pytest_receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--expected-source-commit",
        help="require HEAD to equal this full lowercase 40-character commit ID",
    )
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--all-gates",
        action="store_true",
        help="run every Task 7 quality, validator, package, dashboard, Lean, and Docker gate",
    )
    parser.add_argument(
        "--authorize-protected-so2sat",
        action="store_true",
        help=(
            "explicitly authorize tests and validators that read the real sealed "
            "So2Sat development authority"
        ),
    )
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    commit, tree = source_identity(repo)
    if args.expected_source_commit is not None:
        verify_expected_source_commit(
            actual=commit, expected=args.expected_source_commit
        )
    payload = build_inventory_payload(
        tracked_paths=tracked_paths(repo, commit),
        source_commit=commit,
        source_tree=tree,
        authorize_protected_so2sat=args.authorize_protected_so2sat,
    )
    output = args.output if args.output.is_absolute() else repo / args.output
    write_canonical_json(output, payload)
    print(
        f"repository test inventory: {payload['pytest_count']} executable, "
        f"{payload['excluded_count']} documented exclusions"
    )
    if args.inventory_only:
        return 0
    if not args.all_gates:
        return run_pytest_modules(payload["pytest_paths"], python=args.python)
    try:
        pytest_receipt = run_all_gates(payload, repo=repo, python=args.python)
    except (InventoryError, OSError, subprocess.SubprocessError) as exc:
        payload["execution_status"] = "FAIL"
        payload["failure"] = str(exc)
        payload["inventory_sha256"] = inventory_digest(payload)
        write_canonical_json(output, payload)
        print(f"repository verification: FAIL: {exc}", file=sys.stderr)
        return 1
    payload["execution_status"] = "PASS"
    payload["documented_pytest_skips"] = list(pytest_receipt.accepted_skips)
    payload["pytest_execution"] = {
        "status": "COMPLETE_EXACTLY_ONCE",
        "process_model": PYTEST_PROCESS_MODEL,
        "executed_group_count": len(pytest_receipt.executed_group_ids),
        "executed_module_count": len(payload["pytest_paths"]),
        "executed_group_ids": list(pytest_receipt.executed_group_ids),
    }
    payload["executed_gates"] = [
        gate.name for gate in release_gate_plan(python=args.python)
    ]
    payload["inventory_sha256"] = inventory_digest(payload)
    write_canonical_json(output, payload)
    print(
        "repository verification: PASS (all required gates, no unexpected skips/warnings)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
