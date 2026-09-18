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

import yaml

from docs.research.kbound.scripts import release_privacy

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INVENTORY = REPOSITORY_ROOT / "docs/research/kbound/audits/repository_test_inventory.json"
SCHEMA = "kbound-repository-test-inventory-v2"
PYTEST_PROCESS_MODEL = "one_module_per_fresh_interpreter"

# These files look like pytest modules but are retained historical writers, a
# benchmark utility, or vendored upstream-baseline checks.  Running either
# 3D-ADAM writer consumes already-opened target labels and mutates historical
# result files.  The AETTA module belongs to the upstream baseline's separately
# declared Conda/dataset/model/GPU environment; K-Bound release verification
# covers its integration/provenance contract, not upstream baseline internals.
# New exceptions require source review.
EXPLICIT_TEST_EXCLUSIONS: dict[str, str] = {
    "AETTA/tests/test_dnn_state_initialization.py": ("third_party_baseline_outside_kbound_release_contract"),
    "docs/research/kbound/theory_v2/timing_test.py": ("timing_utility_without_assertions"),
    "experiments/kbound/test_3dadam_bootstrap.py": ("historical_target_analysis_writer"),
    "experiments/kbound/test_3dadam_namedcond.py": ("historical_target_analysis_writer"),
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
PROTECTED_SO2SAT_VALIDATORS = frozenset({"docs/research/kbound/scripts/validate_canonical_release_data.py"})
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

BYTE_PRESERVING_ARCHIVE_ROOTS = (
    "docs/research/kbound/archive/superseded_empirical_authorities_2026-09-02",
    "docs/research/kbound/archive/stale_publication_builds_2026-09-02",
    "docs/research/kbound/archive/legacy_publication_surfaces_2026-09-02",
)

RUFF_ARCHIVE_EXCLUDE_ARGS = (
    "--force-exclude",
    *(argument for root in BYTE_PRESERVING_ARCHIVE_ROOTS for argument in ("--extend-exclude", root)),
)
RUFF_FORMAT_ARCHIVE_EXCLUDE_ARGS = (
    "--force-exclude",
    *(argument for root in BYTE_PRESERVING_ARCHIVE_ROOTS for argument in ("--exclude", root)),
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

PRE_COMMIT_CONFIG_PATH = ".pre-commit-config.yaml"
EXPECTED_PRE_COMMIT_HOOKS: dict[str, tuple[str, ...]] = {
    "https://github.com/astral-sh/ruff-pre-commit": ("ruff", "ruff", "ruff-format"),
    "https://github.com/pre-commit/pre-commit-hooks": (
        "trailing-whitespace",
        "end-of-file-fixer",
        "check-yaml",
        "check-json",
        "check-added-large-files",
        "check-merge-conflict",
        "debug-statements",
    ),
    "https://github.com/gitleaks/gitleaks": ("gitleaks",),
    "https://github.com/pre-commit/mirrors-mypy": ("mypy",),
}
APLUS_HOOK_INCLUDED_CANARIES = (
    "kga/policy.py",
    "kga/nested/policy.py",
    "src/scripts/kbound/make_synth_archive.py",
    "src/scripts/kbound/smoke_trichotomy.py",
    "deploy/api/kga_routes.py",
    "docs/research/kbound/scripts/validate_closure_protocol.py",
    "docs/research/kbound/scripts/run_closure_stage.py",
    "tests/test_kga_package.py",
    "tests/test_kga_experiment_contract.py",
    "tests/test_smoke_trichotomy.py",
)
APLUS_HOOK_EXCLUDED_CANARIES = (
    "kga/py.typed",
    "deploy/api/main.py",
    "docs/research/kbound/scripts/run_repository_verification.py",
    "tests/test_repository_verification_runner.py",
    "experiments/kbound/so2sat/gate.py",
    "experiments/kbound/results/run.py",
    "docs/research/kbound/archive/retired.py",
    "AETTA/upstream.py",
    "external/vendor.py",
)
IMMUTABLE_HOOK_EXCLUDED_CANARIES = (
    "AETTA/upstream.py",
    "external/vendor.py",
    "archive/retired.py",
    "research_lock/source.yaml",
    "audits/receipt.json",
    "experiments/kbound/results/run.json",
    "docs/research/kbound/kbound_pkg/module.py",
    "docs/research/kbound/paper/generated/table.tex",
    "docs/research/kbound/results/receipt.json",
    "docs/research/kbound/audits/receipt.json",
)
MUTABLE_HOOK_INCLUDED_CANARIES = (
    "kga/policy.py",
    "tests/test_repository_verification_runner.py",
    "experiments/kbound/so2sat/gate.py",
    "docs/research/kbound/scripts/run_repository_verification.py",
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
PRIVATE_PATH_SCAN_SUFFIXES = frozenset({".py", ".sh", ".toml", ".yaml", ".yml", ".md", ".txt"})

# These positive Git roots contain maintained implementation, test, or validator
# source and no experiment data/result subtree.  The default inventory must not
# first enumerate the entire tree and then filter protected names in Python.
SO2SAT_SOURCE_PATHS = (
    "experiments/kbound/so2sat/__init__.py",
    "experiments/kbound/so2sat/adapters.py",
    "experiments/kbound/so2sat/development.py",
    "experiments/kbound/so2sat/features.py",
    "experiments/kbound/so2sat/gate.py",
    "experiments/kbound/so2sat/integrity.py",
    "experiments/kbound/so2sat/label_firewall.py",
    "experiments/kbound/so2sat/metadata_manifest.py",
    "experiments/kbound/so2sat/model.py",
    "experiments/kbound/so2sat/precalibration_seal.py",
    "experiments/kbound/so2sat/prospective_runner_v2.py",
    "experiments/kbound/so2sat/prospective_v2.py",
    "experiments/kbound/so2sat/protocol.py",
    "experiments/kbound/so2sat/source_acceptance.py",
    "experiments/kbound/so2sat/source_data.py",
    "experiments/kbound/so2sat/source_preflight.py",
    "experiments/kbound/so2sat/target_amendment.py",
    "experiments/kbound/so2sat/target_contract.py",
    "experiments/kbound/so2sat/target_inference.py",
    "experiments/kbound/so2sat/target_runner.py",
    "experiments/kbound/so2sat/target_scorer.py",
    "experiments/kbound/so2sat/target_seal.py",
    "experiments/kbound/so2sat/train_source.py",
    "experiments/kbound/so2sat/v2_target.py",
)

TRACKED_VERIFICATION_PATHSPECS = (
    "AETTA/tests/test_dnn_state_initialization.py",
    "deploy/api",
    "docs/research/kbound/audit_validation",
    "docs/research/kbound/dashboard/tests/test_build_dashboard_snapshot.py",
    "docs/research/kbound/edge",
    "docs/research/kbound/gapclose_wave5",
    "docs/research/kbound/kbound_pkg",
    "docs/research/kbound/kbound_repro",
    "docs/research/kbound/scripts",
    "docs/research/kbound/tests",
    "docs/research/kbound/theory_v2",
    "docs/research/kbound/ttc_extension",
    "docs/research/multiclass_vector_capacity/formal/tests/test_export_inventory.py",
    "docs/research/multiclass_vector_capacity/tests/test_exact_oracle.py",
    "docs/research/multiclass_vector_capacity/tests/test_statement_bindings.py",
    "experiments/kbound/conj1_validator.py",
    "experiments/kbound/test_3dadam_bootstrap.py",
    "experiments/kbound/test_3dadam_namedcond.py",
    "experiments/kbound/theory_validation",
    *SO2SAT_SOURCE_PATHS,
    "kga",
    "scripts",
    "src",
    "tests",
)

ADDITIONAL_AUTHORED_SOURCE_PATHS = (
    "docs/research/kbound/panel_review_2026-07-25/recompute/kb_common.py",
    "docs/research/kbound/realshift_win/verify_realshift_win.py",
    "experiments/kbound/results/camelyon17_fullscale_B_v1/estimator_dryrun/dryrun.py",
    "experiments/kbound/wilds/analyze_camelyon_kbound.py",
    "experiments/kbound/wilds/analyze_iwildcam_kbound.py",
)

AUTHORED_SOURCE_PATHSPECS = (*TRACKED_VERIFICATION_PATHSPECS, *ADDITIONAL_AUTHORED_SOURCE_PATHS)

DOCKER_CONTEXT_PATHSPECS = (
    ".dockerignore",
    "Dockerfile",
    "requirements-api-py311-linux.lock.txt",
    "deploy/api",
    "kga",
)

QUALITY_EXCLUDED_PREFIXES = (
    *IMMUTABLE_ARCHIVE_PREFIXES,
    "experiments/kbound/data/",
    "experiments/kbound/results/",
    "experiments/kbound/so2sat/gate_calibration/",
    "experiments/kbound/so2sat/gate-calibration/",
    "experiments/kbound/so2sat/target_data/",
    "experiments/kbound/so2sat/target_results/",
)

AUTHORED_SOURCE_EXCLUDED_PREFIXES = (
    *IMMUTABLE_ARCHIVE_PREFIXES,
    "experiments/kbound/so2sat/gate_calibration/",
    "experiments/kbound/so2sat/gate-calibration/",
    "experiments/kbound/so2sat/target_data/",
    "experiments/kbound/so2sat/target_results/",
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
        "filter": ("ignore:Using `httpx` with `starlette.testclient` is deprecated:UserWarning"),
        "scope": "third-party FastAPI/Starlette TestClient compatibility import",
        "rationale": "the production API and offline container gates do not use TestClient",
    },
    {
        "filter": ("ignore:Warning:UserWarning:numpy.ma.core"),
        "scope": "adversarial masked-array rejection tests",
        "rationale": "the warning is the intentionally malformed value being rejected by the tests",
    },
    {
        "filter": ("ignore:'crypt' is deprecated and slated for removal in Python 3.13:DeprecationWarning"),
        "scope": "third-party passlib import used only by API authentication tests",
        "rationale": (
            "the production API is Python 3.11-pinned and this third-party warning is not emitted by KGA code"
        ),
    },
)


class InventoryError(RuntimeError):
    """The tracked verification surface cannot be classified safely."""


def _is_test_shaped(relative: str) -> bool:
    path = PurePosixPath(relative)
    return path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py"))


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
            excluded.append({"path": relative, "reason": PROTECTED_SO2SAT_EXCLUSION_REASON})
            continue
        reason = EXPLICIT_TEST_EXCLUSIONS.get(relative)
        if reason is not None:
            excluded.append({"path": relative, "reason": reason})
            continue
        if require_exception_classification and relative.endswith("/timing_test.py"):
            raise InventoryError(f"unclassified test-shaped path requires an explicit rationale: {relative}")
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
        and (path.name.startswith(("val_", "validate_")) or path.name.endswith("_validator.py"))
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
            excluded.append({"path": relative, "reason": PROTECTED_SO2SAT_EXCLUSION_REASON})
        elif relative in EXPLICIT_VALIDATOR_EXCLUSIONS:
            excluded.append({"path": relative, "reason": EXPLICIT_VALIDATOR_EXCLUSIONS[relative]})
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
            group_id=(f"pytest-module-{index:04d}-{hashlib.sha256(path.encode('utf-8')).hexdigest()[:16]}"),
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


def verify_pytest_process_group_records(paths: Sequence[str], records: Sequence[Mapping[str, Any]]) -> None:
    """Require the recorded groups to cover every module exactly once."""

    expected = pytest_process_group_records(pytest_process_groups(paths))
    if list(records) != expected:
        raise InventoryError("pytest process-group coverage is not the canonical exact-once partition")


def verify_pytest_backed_release_gate_coverage(paths: Sequence[str]) -> None:
    """Ensure pytest-backed release checks are in the exact-once module plan."""

    missing = sorted(PYTEST_BACKED_RELEASE_GATE_MODULES - set(paths))
    if missing:
        raise InventoryError(f"pytest-backed release gate modules are missing: {missing}")


def _verify_executed_group_ids(groups: Sequence[PytestProcessGroup], executed_group_ids: Sequence[str]) -> None:
    expected = [group.group_id for group in groups]
    if list(executed_group_ids) != expected:
        raise InventoryError("executed pytest process-group coverage is not complete and exact-once")


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
    verify_pytest_process_group_records(inventory["pytest_paths"], process_group_records)
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
            {"name": gate.name, "required": gate.required} for gate in release_gate_plan(python=sys.executable)
        ],
        "execution_status": "NOT_RUN",
        "immutable_hook_configuration": {
            "status": "NOT_RUN",
            "config_path": PRE_COMMIT_CONFIG_PATH,
            "execution": "NOT_RUN",
        },
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
        raise InventoryError(f"refusing to reuse inventory staging path: {temporary}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    """Run Git only against a fully resident repository, failing closed on error."""

    command = ["git", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            check=True,
            capture_output=True,
            text=not binary,
            timeout=int(os.environ.get("KBOUND_GIT_TIMEOUT", "180")),
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise InventoryError(f"Git inventory command failed: {' '.join(args)}") from exc
    stdout = completed.stdout
    if not isinstance(stdout, (str, bytes)):
        raise InventoryError("git did not return a text or binary payload")
    return stdout


def _parse_ls_tree_records(raw: bytes) -> dict[str, tuple[str, str, str]]:
    """Parse NUL-delimited ``git ls-tree`` records, including blob identities."""

    entries: dict[str, tuple[str, str, str]] = {}
    for encoded_record in raw.split(b"\0"):
        if not encoded_record:
            continue
        try:
            metadata, encoded_path = encoded_record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ", 2)
            relative = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise InventoryError("Git returned a malformed source-inventory record") from exc
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id) is None:
            raise InventoryError("Git returned a malformed source-inventory object ID")
        if relative in entries:
            raise InventoryError(f"Git returned a duplicate source-inventory path: {relative}")
        entries[relative] = (mode, object_type, object_id)
    return entries


def _parse_ls_tree_entries(raw: bytes) -> dict[str, tuple[str, str]]:
    """Parse NUL-delimited ``git ls-tree`` records without losing file modes."""

    return {
        relative: (mode, object_type)
        for relative, (mode, object_type, _object_id) in _parse_ls_tree_records(raw).items()
    }


def tracked_path_entries(repo: Path, revision: str = "HEAD") -> dict[str, tuple[str, str]]:
    """Return exact source-only Git entries, including their modes and types."""

    raw = _git(
        repo,
        "ls-tree",
        "-r",
        "-z",
        revision,
        "--",
        *TRACKED_VERIFICATION_PATHSPECS,
        binary=True,
    )
    assert isinstance(raw, bytes)
    return _parse_ls_tree_entries(raw)


def authored_source_entries(repo: Path, revision: str = "HEAD") -> dict[str, tuple[str, str]]:
    """Return exact Git entries from the bounded authored-source pathspecs."""

    raw = _git(
        repo,
        "ls-tree",
        "-r",
        "-z",
        revision,
        "--",
        *AUTHORED_SOURCE_PATHSPECS,
        binary=True,
    )
    assert isinstance(raw, bytes)
    return _parse_ls_tree_entries(raw)


def tracked_paths(repo: Path, revision: str = "HEAD") -> list[str]:
    """Return source-only verification paths from an immutable Git revision."""

    return sorted(tracked_path_entries(repo, revision))


def quality_python_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Return exact tracked Python sources without entering data/result trees."""

    selected: set[str] = set()
    for raw in paths:
        relative = PurePosixPath(raw).as_posix()
        if PurePosixPath(relative).suffix != ".py":
            continue
        if any(relative.startswith(prefix) for prefix in QUALITY_EXCLUDED_PREFIXES):
            continue
        selected.add(relative)
    return tuple(sorted(selected))


def authored_source_paths(
    paths: Iterable[str],
    *,
    suffixes: Iterable[str],
) -> tuple[str, ...]:
    """Select exact maintained source files before touching the worktree.

    ``paths`` comes from the positive Git inventory above.  The explicit
    protected/evidence prefixes are filtered here, before filesystem metadata
    or file contents are inspected, so a hostile entry cannot redirect a scan
    into target evidence.
    """

    allowed_suffixes = frozenset(suffixes)
    if not allowed_suffixes or any(not suffix.startswith(".") for suffix in allowed_suffixes):
        raise InventoryError("authored-source suffixes must be nonempty dotted extensions")
    selected: set[str] = set()
    for raw in paths:
        relative = PurePosixPath(raw).as_posix()
        if PurePosixPath(relative).suffix not in allowed_suffixes:
            continue
        if any(relative.startswith(prefix) for prefix in AUTHORED_SOURCE_EXCLUDED_PREFIXES):
            continue
        selected.add(relative)
    return tuple(sorted(selected))


def _regular_git_blob(relative: str, entry: tuple[str, str]) -> None:
    mode, object_type = entry
    if object_type != "blob" or mode not in {"100644", "100755"}:
        raise InventoryError(f"quality source is not a regular Git blob: {relative}")


def _safe_relative_source_path(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise InventoryError(f"quality source is not a safe relative path: {relative}")
    if path.as_posix() != relative:
        raise InventoryError(f"quality source is not a canonical Git path: {relative}")
    return path


def worktree_deleted_source_paths(
    repo: Path,
    revision: str,
    *,
    pathspecs: Iterable[str] = TRACKED_VERIFICATION_PATHSPECS,
) -> frozenset[str]:
    """Return exact tracked source paths deleted from the current worktree.

    This is a positive Git query over the same bounded source pathspecs as the
    immutable inventory.  It lets static-analysis tests honor intentional
    source retirements before those deletions are committed, without walking
    the repository or weakening validation for files that still exist.
    """

    raw = _git(
        repo,
        "diff",
        "--name-only",
        "--diff-filter=D",
        "-z",
        revision,
        "--",
        *pathspecs,
        binary=True,
    )
    assert isinstance(raw, bytes)
    deleted: set[str] = set()
    for encoded_path in raw.split(b"\0"):
        if not encoded_path:
            continue
        try:
            relative = encoded_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InventoryError("Git returned a malformed deleted-source path") from exc
        _safe_relative_source_path(relative)
        deleted.add(relative)
    return frozenset(deleted)


def _require_unredirected_regular_file(repo: Path, relative: str) -> None:
    """Reject symlinked parents and nonregular working-tree source entries."""

    root = repo.absolute()
    try:
        root_metadata = os.lstat(root)
    except OSError as exc:
        raise InventoryError(f"quality source root is unavailable: {root}") from exc
    if stat.S_ISLNK(root_metadata.st_mode):
        raise InventoryError(f"quality source root is a symlink redirection: {root}")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise InventoryError(f"quality source root is not a directory: {root}")

    path = _safe_relative_source_path(relative)
    current = root
    for part in path.parts[:-1]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise InventoryError(f"quality source parent is unavailable: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise InventoryError(f"quality source parent is a symlink redirection: {relative}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise InventoryError(f"quality source parent is not a directory: {relative}")

    candidate = root.joinpath(*path.parts)
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise InventoryError(f"quality source is unavailable: {relative}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise InventoryError(f"quality source is a symlink redirection: {relative}")
    if not stat.S_ISREG(metadata.st_mode):
        raise InventoryError(f"quality source is not a regular working-tree file: {relative}")


def validate_quality_paths(
    repo: Path,
    entries: Mapping[str, tuple[str, str]],
) -> tuple[str, ...]:
    """Validate the exact quality inventory before any tool receives its paths."""

    selected = quality_python_paths(entries)
    if not selected:
        raise InventoryError("source-only Python quality inventory is empty")
    for relative in selected:
        _regular_git_blob(relative, entries[relative])
        _require_unredirected_regular_file(repo, relative)
    return selected


def validate_authored_source_paths(
    repo: Path,
    entries: Mapping[str, tuple[str, str]],
    *,
    suffixes: Iterable[str],
) -> tuple[str, ...]:
    """Validate a source scan's exact Git and working-tree file boundary."""

    selected = authored_source_paths(entries, suffixes=suffixes)
    if not selected:
        raise InventoryError("authored-source inventory is empty")
    for relative in selected:
        _regular_git_blob(relative, entries[relative])
        _require_unredirected_regular_file(repo, relative)
    return selected


def verify_so2sat_source_allowlist(repo: Path, revision: str) -> tuple[str, ...]:
    """Validate only the exact declared So2Sat source files.

    Completeness review is a separately authorized maintenance operation. The
    default release path never queries the So2Sat parent or enumerates sibling
    names because that directory may also contain protected authorities.
    """

    raw = _git(
        repo,
        "ls-tree",
        "-z",
        revision,
        "--",
        *SO2SAT_SOURCE_PATHS,
        binary=True,
    )
    assert isinstance(raw, bytes)
    entries = _parse_ls_tree_entries(raw)
    observed: set[str] = set()
    for relative, entry in entries.items():
        _regular_git_blob(relative, entry)
        observed.add(relative)

    expected = set(SO2SAT_SOURCE_PATHS)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise InventoryError(f"So2Sat top-level source allowlist is stale: missing={missing}, unexpected={unexpected}")
    return tuple(sorted(observed))


def validated_quality_python_paths(*, repo: Path, revision: str) -> tuple[str, ...]:
    """Build and validate the exact source surface used by quality gates."""

    verify_so2sat_source_allowlist(repo, revision)
    return validate_quality_paths(repo, tracked_path_entries(repo, revision))


def validated_authored_source_paths(
    *,
    repo: Path,
    revision: str,
    suffixes: Iterable[str],
) -> tuple[str, ...]:
    """Return the tracked, source-only, worktree-safe static-scan surface."""

    entries = authored_source_entries(repo, revision)
    deleted = worktree_deleted_source_paths(repo, revision, pathspecs=AUTHORED_SOURCE_PATHSPECS)
    return validate_authored_source_paths(
        repo,
        {relative: entry for relative, entry in entries.items() if relative not in deleted},
        suffixes=suffixes,
    )


def source_identity(repo: Path) -> tuple[str, str]:
    commit = str(_git(repo, "rev-parse", "--verify", "HEAD^{commit}")).strip()
    tree = str(_git(repo, "rev-parse", "--verify", "HEAD^{tree}")).strip()
    return commit, tree


def verify_expected_source_commit(*, actual: str, expected: str) -> str:
    """Bind an inventory to an exact workflow checkout, never an abbreviated ref."""

    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise InventoryError("expected source commit must be a full lowercase 40-character Git object ID")
    if actual != expected:
        raise InventoryError(f"expected source commit {expected} does not match HEAD {actual}")
    return actual


def _local_snapshot_scopes() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reuse the reviewed source scopes, never a repository-wide data walk."""

    from docs.research.kbound.scripts import build_release_source_seal as seal

    exact = {path for paths in seal.EXPLICIT_FILES.values() for path in paths}
    exact.update(ADDITIONAL_AUTHORED_SOURCE_PATHS)
    exact.update(path for path in TRACKED_VERIFICATION_PATHSPECS if path.endswith(".py"))
    exact.add("kga/py.typed")
    roots = {path.rstrip("/") for path in TRACKED_VERIFICATION_PATHSPECS if not path.endswith(".py")}
    roots.update(prefix.rstrip("/") for _category, prefix, _suffixes in seal.SOURCE_PREFIX_RULES)
    return tuple(sorted(exact)), tuple(sorted(roots))


_SNAPSHOT_SOURCE_SUFFIXES = frozenset({".py", ".pyi", ".sh", ".lean", ".ts", ".js", ".map", ".css"})
_SNAPSHOT_RUNTIME_PARTS = frozenset(
    {".git", ".venv", "venv", "node_modules", ".lake", "__pycache__", "build", "dist", ".pytest_cache"}
)


def _local_snapshot_paths(repo: Path, exact: Sequence[str], roots: Sequence[str]) -> tuple[str, ...]:
    def reject_redirected_ancestors(relative: str) -> None:
        parts = PurePosixPath(relative).parts
        for index in range(1, len(parts) + 1):
            if repo.joinpath(*parts[:index]).is_symlink():
                raise InventoryError(f"snapshot source ancestor is a symlink: {relative}")

    def reject_unreadable_directory(error: OSError) -> None:
        raise InventoryError(f"snapshot source directory cannot be inventoried: {error}") from error

    selected: set[str] = set()
    for relative in exact:
        _safe_relative_source_path(relative)
        reject_redirected_ancestors(relative)
        candidate = repo / relative
        if candidate.exists() or candidate.is_symlink():
            _require_unredirected_regular_file(repo, relative)
            selected.add(relative)
    for relative_root in roots:
        _safe_relative_source_path(relative_root)
        reject_redirected_ancestors(relative_root)
        root = repo / relative_root
        if root.is_symlink():
            raise InventoryError(f"snapshot source root is a symlink: {relative_root}")
        if not root.exists():
            continue
        for current, directories, filenames in os.walk(root, followlinks=False, onerror=reject_unreadable_directory):
            current_path = Path(current)
            directories[:] = sorted(name for name in directories if name not in _SNAPSHOT_RUNTIME_PARTS)
            for name in directories:
                if (current_path / name).is_symlink():
                    raise InventoryError(f"snapshot source directory is a symlink: {current_path / name}")
            for name in sorted(filenames):
                candidate = current_path / name
                relative = candidate.relative_to(repo).as_posix()
                if candidate.suffix not in _SNAPSHOT_SOURCE_SUFFIXES or _is_archive(relative):
                    continue
                _require_unredirected_regular_file(repo, relative)
                selected.add(relative)
    return tuple(sorted(selected))


def _local_snapshot_bytes(repo: Path, relative: str) -> tuple[bytes, str]:
    _require_unredirected_regular_file(repo, relative)
    blob = _git_blob_digest(repo / relative, object_id_length=40)
    content = _read_bound_source_bytes(repo, relative, blob)
    metadata = (repo / relative).stat(follow_symlinks=False)
    # Some macOS Python builds omit SF_DATALESS even though stat flags expose it.
    if getattr(metadata, "st_flags", 0) & getattr(stat, "SF_DATALESS", 0x40000000):
        raise InventoryError(f"snapshot source is dataless; explicit recovery is required: {relative}")
    return content, "100755" if metadata.st_mode & 0o111 else "100644"


def create_local_source_snapshot(repo: Path, destination: Path, *, expected_source_commit: str) -> dict[str, Any]:
    """Commit an explicit local-source derivative in a new isolated repository.

    Its lineage names the original HEAD and every copied local byte sequence.
    The scratch commit is deliberately distinct from that HEAD. No release
    evidence, dataset, source-seal PASS, or original Git mutation is implied.
    """

    repo = repo.resolve()
    destination = destination.absolute()
    if destination != destination.resolve() or destination == repo or repo in destination.parents:
        raise InventoryError("snapshot destination must be outside the source checkout and unredirected")
    if destination.exists():
        raise FileExistsError(f"snapshot destination already exists: {destination}")
    original_commit, original_tree = source_identity(repo)
    verify_expected_source_commit(actual=original_commit, expected=expected_source_commit)
    exact, roots = _local_snapshot_scopes()
    paths = _local_snapshot_paths(repo, exact, roots)
    if not paths:
        raise InventoryError("local source snapshot inventory is empty")
    raw = _git(repo, "ls-tree", "-r", "-z", original_commit, "--", *exact, *roots, binary=True)
    assert isinstance(raw, bytes)
    original_entries = _parse_ls_tree_records(raw)
    original_source_paths = {
        relative
        for relative in original_entries
        if relative in exact
        or (
            PurePosixPath(relative).suffix in _SNAPSHOT_SOURCE_SUFFIXES
            and not _SNAPSHOT_RUNTIME_PARTS.intersection(PurePosixPath(relative).parts)
            and not _is_archive(relative)
        )
    }
    destination.mkdir(parents=False)
    rows: list[dict[str, Any]] = []
    for relative in paths:
        content, mode = _local_snapshot_bytes(repo, relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(content)
        target.chmod(0o755 if mode == "100755" else 0o644)
        blob = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
        original = original_entries.get(relative)
        relation = (
            "LOCAL_ONLY" if original is None else ("UNCHANGED" if original == (mode, "blob", blob) else "MODIFIED")
        )
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "git_mode": mode,
                "original_git_blob": original[2] if original else None,
                "relation_to_original_head": relation,
            }
        )
    if _local_snapshot_paths(repo, exact, roots) != paths:
        raise InventoryError("local source inventory changed during snapshot creation")
    for row in rows:
        current, mode = _local_snapshot_bytes(repo, row["path"])
        if hashlib.sha256(current).hexdigest() != row["sha256"] or mode != row["git_mode"]:
            raise InventoryError(f"local source changed during snapshot creation: {row['path']}")
    if source_identity(repo) != (original_commit, original_tree):
        raise InventoryError("original HEAD changed during snapshot creation")
    lineage = {
        "schema": "kbound-local-source-snapshot-v1",
        "original_source_commit": original_commit,
        "original_source_tree": original_tree,
        "scope": "current local source under reviewed exact files and source roots; no dataset or result trees",
        "exact_file_scopes": list(exact),
        "source_root_scopes": list(roots),
        "runtime_directory_exclusions": sorted(_SNAPSHOT_RUNTIME_PARTS),
        "source_suffixes": sorted(_SNAPSHOT_SOURCE_SUFFIXES),
        "files": rows,
        "deleted_from_original_head": sorted(original_source_paths - set(paths)),
        "missing_declared_files": sorted(set(exact) - set(paths)),
        "execution_status": "NOT_RUN",
        "limits": "A distinct local derivative, not the original HEAD or a complete release/evidence bundle.",
    }
    write_canonical_json(destination / "KBOUND_LOCAL_SOURCE_SNAPSHOT.json", lineage)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    for arguments in (
        ("init", "-q", "--template=", "--object-format=sha1"),
        ("add", "--all"),
        (
            "-c",
            "user.name=K-Bound Local Snapshot",
            "-c",
            "user.email=snapshot@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            f"Isolated local source snapshot from {original_commit}",
        ),
    ):
        subprocess.run(
            ["git", *arguments], cwd=destination, env=environment, check=True, capture_output=True, timeout=30
        )
    snapshot_commit, snapshot_tree = source_identity(destination)
    bindings = revision_blob_bindings(destination, snapshot_commit, paths)
    verify_worktree_blob_bindings(destination, bindings)
    return {
        "schema": "kbound-local-source-snapshot-receipt-v1",
        "status": "SOURCE_SNAPSHOT_CREATED",
        "original_source_commit": original_commit,
        "original_source_tree": original_tree,
        "source_commit": snapshot_commit,
        "source_tree": snapshot_tree,
        "snapshot_path": str(destination),
        "source_file_count": len(rows),
        "missing_declared_files": lineage["missing_declared_files"],
        "lineage_sha256": hashlib.sha256(canonical_json_bytes(lineage)).hexdigest(),
        "execution_status": "NOT_RUN",
    }


def revision_blob_bindings(repo: Path, revision: str, paths: Sequence[str]) -> dict[str, str]:
    """Bind exact release inputs to the regular blobs in ``revision``.

    Every path is validated before it is interpolated into a literal Git
    pathspec.  A caller therefore cannot broaden this query with pathspec
    magic, and a directory cannot silently expand into extra inputs.
    """

    ordered = tuple(dict.fromkeys(paths))
    if not ordered or len(ordered) != len(paths):
        raise InventoryError("revision blob binding paths must be nonempty and unique")
    for relative in ordered:
        _safe_relative_source_path(relative)
    raw = _git(
        repo,
        "ls-tree",
        "-r",
        "-z",
        revision,
        "--",
        *(f":(literal){relative}" for relative in ordered),
        binary=True,
    )
    assert isinstance(raw, bytes)
    records = _parse_ls_tree_records(raw)
    expected = set(ordered)
    observed = set(records)
    if observed != expected:
        raise InventoryError(
            "revision blob binding is incomplete: "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )
    bindings: dict[str, str] = {}
    for relative in ordered:
        mode, object_type, object_id = records[relative]
        _regular_git_blob(relative, (mode, object_type))
        bindings[relative] = object_id
    return bindings


def _git_blob_digest(path: Path, *, object_id_length: int) -> str:
    """Hash one no-follow regular file using Git's blob-object framing."""

    algorithm = {40: "sha1", 64: "sha256"}.get(object_id_length)
    if algorithm is None:
        raise InventoryError("unsupported Git object ID length in source binding")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InventoryError(f"bound source cannot be opened safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise InventoryError(f"bound source is not a regular file: {path}")
        digest = hashlib.new(algorithm)
        digest.update(f"blob {before.st_size}\0".encode("ascii"))
        consumed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            consumed += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if not stable or consumed != before.st_size:
            raise InventoryError(f"bound source changed while it was being verified: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def verify_worktree_blob_bindings(repo: Path, bindings: Mapping[str, str]) -> None:
    """Fail unless current no-follow worktree bytes equal their pinned blobs."""

    if not bindings:
        raise InventoryError("worktree blob binding set is empty")
    for relative, expected in bindings.items():
        _safe_relative_source_path(relative)
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", expected) is None:
            raise InventoryError(f"invalid pinned Git blob identity: {relative}")
        _require_unredirected_regular_file(repo, relative)
        observed = _git_blob_digest(repo / relative, object_id_length=len(expected))
        if observed != expected:
            raise InventoryError(f"source differs from the pinned Git blob: {relative}")


def _read_bound_source_bytes(repo: Path, relative: str, expected: str) -> bytes:
    """Read one no-follow source file and verify the bytes against its Git blob."""

    _safe_relative_source_path(relative)
    _require_unredirected_regular_file(repo, relative)
    algorithm = {40: "sha1", 64: "sha256"}.get(len(expected))
    if algorithm is None or re.fullmatch(r"[0-9a-f]+", expected) is None:
        raise InventoryError(f"invalid pinned Git blob identity: {relative}")
    path = repo / relative
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InventoryError(f"bound source cannot be opened safely: {relative}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise InventoryError(f"bound source is not a regular file: {relative}")
        chunks: list[bytes] = []
        consumed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            consumed += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stable or consumed != before.st_size:
        raise InventoryError(f"bound source changed while it was being read: {relative}")
    payload = b"".join(chunks)
    digest = hashlib.new(algorithm)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    if digest.hexdigest() != expected:
        raise InventoryError(f"source differs from the pinned Git blob: {relative}")
    return payload


def validate_immutable_hook_configuration(*, repo: Path, source_commit: str) -> dict[str, Any]:
    """Validate the pinned hook policy without installing or executing hooks."""

    bindings = revision_blob_bindings(repo, source_commit, (PRE_COMMIT_CONFIG_PATH,))
    raw = _read_bound_source_bytes(repo, PRE_COMMIT_CONFIG_PATH, bindings[PRE_COMMIT_CONFIG_PATH])
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise InventoryError("pre-commit configuration is not valid UTF-8 YAML") from exc
    if not isinstance(document, Mapping):
        raise InventoryError("pre-commit configuration root must be a mapping")
    repositories = document.get("repos")
    if not isinstance(repositories, list) or not repositories:
        raise InventoryError("pre-commit configuration must contain a nonempty repository list")

    observed: dict[str, list[dict[str, Any]]] = {}
    hook_count = 0
    for repository_index, repository in enumerate(repositories):
        context = f"pre-commit repository {repository_index}"
        if not isinstance(repository, Mapping):
            raise InventoryError(f"{context} must be a mapping")
        repository_url = repository.get("repo")
        revision = repository.get("rev")
        hooks = repository.get("hooks")
        if not isinstance(repository_url, str) or not repository_url:
            raise InventoryError(f"{context} has no repository URL")
        if repository_url in observed:
            raise InventoryError(f"duplicate pre-commit repository: {repository_url}")
        if (
            not isinstance(revision, str)
            or re.fullmatch(
                r"(?:v)?\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.-]+)?|[0-9a-f]{40}|[0-9a-f]{64}",
                revision,
            )
            is None
        ):
            raise InventoryError(f"pre-commit repository revision is not an immutable version: {repository_url}")
        if not isinstance(hooks, list) or not hooks:
            raise InventoryError(f"{context} has no hooks")
        normalized_hooks: list[dict[str, Any]] = []
        for hook_index, hook in enumerate(hooks):
            hook_context = f"{repository_url} hook {hook_index}"
            if not isinstance(hook, Mapping):
                raise InventoryError(f"{hook_context} must be a mapping")
            allowed_hook_keys = {"id", "name", "args", "files", "exclude", "additional_dependencies"}
            unreviewed_keys = [key for key in hook if key not in allowed_hook_keys]
            if unreviewed_keys:
                raise InventoryError(f"{hook_context} has unreviewed control keys: {unreviewed_keys}")
            hook_id = hook.get("id")
            if not isinstance(hook_id, str) or not hook_id:
                raise InventoryError(f"{hook_context} has no hook id")
            args = hook.get("args", [])
            if not isinstance(args, list) or not all(isinstance(argument, str) for argument in args):
                raise InventoryError(f"{hook_context} args must be a list of strings")
            for field in ("files", "exclude"):
                value = hook.get(field)
                if value is not None and (not isinstance(value, str) or not value):
                    raise InventoryError(f"{hook_context} {field} must be a nonempty regex string")
            normalized_hooks.append(
                {
                    "id": hook_id,
                    "args": tuple(args),
                    "files": hook.get("files"),
                    "exclude": hook.get("exclude"),
                }
            )
            hook_count += 1
        observed[repository_url] = normalized_hooks

    if set(observed) != set(EXPECTED_PRE_COMMIT_HOOKS):
        raise InventoryError(
            "pre-commit repository set differs from the reviewed hook policy: "
            f"expected={sorted(EXPECTED_PRE_COMMIT_HOOKS)}, observed={sorted(observed)}"
        )
    for repository_url, expected_ids in EXPECTED_PRE_COMMIT_HOOKS.items():
        observed_ids = tuple(hook["id"] for hook in observed[repository_url])
        if observed_ids != expected_ids:
            raise InventoryError(
                f"pre-commit hook selection differs for {repository_url}: "
                f"expected={expected_ids}, observed={observed_ids}"
            )

    def compile_hook_regex(hook: Mapping[str, Any], field: str, description: str) -> re.Pattern[str]:
        value = hook.get(field)
        if not isinstance(value, str) or not value:
            raise InventoryError(f"{description} has no {field} boundary")
        try:
            return re.compile(value)
        except re.error as exc:
            raise InventoryError(f"{description} has an invalid {field} regex") from exc

    ruff_repository = observed["https://github.com/astral-sh/ruff-pre-commit"]
    ruff_hooks = [hook for hook in ruff_repository if hook["id"] == "ruff"]
    autofix_hooks = [hook for hook in ruff_hooks if "--fix" in hook["args"]]
    critical_hooks = [hook for hook in ruff_hooks if "--fix" not in hook["args"]]
    if len(autofix_hooks) != 1 or autofix_hooks[0]["args"] != ("--fix", "--exit-non-zero-on-fix"):
        raise InventoryError("ruff autofix hook does not have the reviewed fail-on-fix policy")
    if len(critical_hooks) != 1 or critical_hooks[0]["args"] != ("--select", "E9,F63,F7,F82"):
        raise InventoryError("ruff critical hook does not have the reviewed non-mutating rule selection")
    formatter = next(hook for hook in ruff_repository if hook["id"] == "ruff-format")
    if formatter["args"]:
        raise InventoryError("ruff formatter has unreviewed command arguments")
    for description, hook in (("ruff autofix hook", autofix_hooks[0]), ("ruff formatter", formatter)):
        selector = compile_hook_regex(hook, "files", description)
        exclusion = compile_hook_regex(hook, "exclude", description) if hook["exclude"] is not None else None
        for relative in APLUS_HOOK_INCLUDED_CANARIES:
            if selector.search(relative) is None:
                raise InventoryError(f"{description} does not select required A+ path: {relative}")
            if exclusion is not None and exclusion.search(relative) is not None:
                raise InventoryError(f"{description} excludes required A+ path: {relative}")
        for relative in APLUS_HOOK_EXCLUDED_CANARIES:
            if selector.search(relative) is not None:
                raise InventoryError(f"mutating hook selector escapes the A+ boundary: {description}: {relative}")

    hygiene_repository = observed["https://github.com/pre-commit/pre-commit-hooks"]
    hygiene_mutators = [
        hook for hook in hygiene_repository if hook["id"] in {"trailing-whitespace", "end-of-file-fixer"}
    ]
    for hook in hygiene_mutators:
        description = f"{hook['id']} hook"
        exclusion = compile_hook_regex(hook, "exclude", description)
        for relative in IMMUTABLE_HOOK_EXCLUDED_CANARIES:
            if exclusion.search(relative) is None:
                raise InventoryError(f"mutable hygiene hook does not exclude immutable path: {description}: {relative}")
        for relative in MUTABLE_HOOK_INCLUDED_CANARIES:
            if exclusion.search(relative) is not None:
                raise InventoryError(f"mutable hygiene hook excludes maintained path: {description}: {relative}")

    mutating_hooks = (
        "ruff:--fix",
        "ruff-format",
        "trailing-whitespace",
        "end-of-file-fixer",
    )
    return {
        "status": "PASS",
        "config_path": PRE_COMMIT_CONFIG_PATH,
        "config_git_blob": bindings[PRE_COMMIT_CONFIG_PATH],
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "repository_count": len(repositories),
        "hook_count": hook_count,
        "mutating_hook_count": len(mutating_hooks),
        "mutating_hooks": list(mutating_hooks),
        "selection_canary_count": (
            len(APLUS_HOOK_INCLUDED_CANARIES)
            + len(APLUS_HOOK_EXCLUDED_CANARIES)
            + len(IMMUTABLE_HOOK_EXCLUDED_CANARIES)
            + len(MUTABLE_HOOK_INCLUDED_CANARIES)
        ),
        "execution": "CONFIGURATION_PARSED_NO_HOOKS_EXECUTED",
    }


def run_pytest_modules(
    paths: Sequence[str],
    *,
    repo: Path,
    python: str,
    source_commit: str,
) -> int:
    if not paths:
        raise InventoryError("tracked pytest inventory is empty")
    bindings = revision_blob_bindings(repo, source_commit, paths)
    first_failure = 0
    for group in pytest_process_groups(paths):
        verify_worktree_blob_bindings(repo, {relative: bindings[relative] for relative in group.paths})
        command = [python, "-m", "pytest", "-q", *group.paths]
        returncode = subprocess.run(command, cwd=repo, check=False).returncode
        if returncode != 0 and first_failure == 0:
            first_failure = returncode
    return first_failure


def validate_pytest_report(path: Path) -> list[dict[str, str]]:
    """Reject every runtime skip without an exact, non-load-bearing policy."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise InventoryError(f"pytest did not produce a valid JUnit receipt: {exc}") from exc
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
    paths: Sequence[str],
    *,
    repo: Path,
    python: str,
    source_commit: str,
) -> PytestRunReceipt:
    if not paths:
        raise InventoryError("tracked pytest inventory is empty")
    groups = pytest_process_groups(paths)
    bindings = revision_blob_bindings(repo, source_commit, paths)
    with tempfile.TemporaryDirectory(prefix="kbound-pytest-") as temporary:
        accepted_skips: list[dict[str, str]] = []
        executed_group_ids: list[str] = []
        failures: list[str] = []
        report_failures: list[str] = []
        for group in groups:
            verify_worktree_blob_bindings(repo, {relative: bindings[relative] for relative in group.paths})
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
                failures.append(f"{group.group_id} ({group.paths[0]}) failed with exit {result.returncode}")
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
        raise InventoryError(f"required release tool is unavailable: {command[0]}") from exc


def _run_validators(
    paths: Sequence[str],
    *,
    repo: Path,
    python: str,
    source_commit: str,
) -> None:
    if not paths:
        raise InventoryError("tracked standalone validator inventory is empty")
    bindings = revision_blob_bindings(repo, source_commit, paths)
    for relative in paths:
        verify_worktree_blob_bindings(repo, {relative: bindings[relative]})
        _run([python, relative], repo=repo)


def _private_path_scan_bindings(repo: Path, source_commit: str) -> dict[str, str]:
    """Inventory only fixed public roots from the pinned Git tree."""

    raw = _git(
        repo,
        "ls-tree",
        "-r",
        "-z",
        source_commit,
        "--",
        *PRIVATE_PATH_SCAN_ROOTS,
        binary=True,
    )
    assert isinstance(raw, bytes)
    records = _parse_ls_tree_records(raw)
    bindings: dict[str, str] = {}
    for relative, (mode, object_type, object_id) in records.items():
        path = _safe_relative_source_path(relative)
        if not any(relative == root or relative.startswith(f"{root}/") for root in PRIVATE_PATH_SCAN_ROOTS):
            raise InventoryError(f"private-path inventory escaped its public roots: {relative}")
        if path.suffix.lower() not in PRIVATE_PATH_SCAN_SUFFIXES:
            continue
        _regular_git_blob(relative, (mode, object_type))
        bindings[relative] = object_id
    if not bindings:
        raise InventoryError("private-path scan found no public runtime/operator source files")
    return bindings


def run_private_path_scan(*, repo: Path, source_commit: str) -> dict[str, Any]:
    """Scan the public runtime/operator source surface and return a receipt.

    The release plan used to advertise this gate without executing it.  The
    bounded source roots are intentional: scanning result/dataset trees would
    both exceed the public source scope and could dereference protected
    scientific artifacts.  Symlinks fail closed instead of extending the scan.
    """

    bindings = _private_path_scan_bindings(repo, source_commit)
    scanned: list[str] = []
    for relative in sorted(bindings):
        try:
            payload = _read_bound_source_bytes(repo, relative, bindings[relative])
            release_privacy.scan_member(relative, payload)
        except (OSError, release_privacy.PrivacyError) as exc:
            raise InventoryError(f"private-path scan failed for {relative}: {exc}") from exc
        scanned.append(relative)
    return {
        "status": "PASS",
        "scope_roots": list(PRIVATE_PATH_SCAN_ROOTS),
        "scanned_file_count": len(scanned),
        "scanned_paths_sha256": hashlib.sha256("\n".join(scanned).encode("utf-8")).hexdigest(),
    }


def _run_quality_gates(
    *,
    repo: Path,
    python: str,
    quality_paths: Sequence[str],
    source_commit: str,
) -> None:
    if not quality_paths:
        raise InventoryError("source-only Python quality inventory is empty")
    binding_paths = tuple(dict.fromkeys((*quality_paths, "pyproject.toml")))
    bindings = revision_blob_bindings(repo, source_commit, binding_paths)
    commands = (
        [
            python,
            "-m",
            "ruff",
            "check",
            *RUFF_ARCHIVE_EXCLUDE_ARGS,
            "--select",
            "E9,F63,F7,F82",
            *quality_paths,
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
            "-m",
            "bandit",
            "-q",
            "-r",
            "kga",
            "deploy/api",
            "-c",
            "pyproject.toml",
        ],
    )
    for command in commands:
        verify_worktree_blob_bindings(repo, bindings)
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
        raise InventoryError("package build must produce exactly one wheel and one source archive")
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
            environment / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        )
        child_site_packages.mkdir(parents=True, exist_ok=True)
        locked_parent_sites = sorted(
            str(Path(candidate).resolve()) for candidate in site.getsitepackages() if Path(candidate).is_dir()
        )
        if not locked_parent_sites:
            raise InventoryError("locked parent environment exposes no site-packages directory")
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
        shutil.copytree(dashboard, staged, ignore=shutil.ignore_patterns("node_modules"))
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
        actual = {path.relative_to(staged / "js") for path in (staged / "js").rglob("*") if path.is_file()}
        if not expected:
            raise InventoryError("dashboard build produced no JavaScript")
        if actual != expected:
            unexpected = sorted(map(str, actual - expected))
            missing = sorted(map(str, expected - actual))
            raise InventoryError(f"unexpected or missing dashboard outputs: unexpected={unexpected}, missing={missing}")
        for relative in sorted(expected):
            rebuilt = staged / "js" / relative
            committed = dashboard / "js" / relative
            if not committed.is_file() or rebuilt.read_bytes() != committed.read_bytes():
                raise InventoryError(f"dashboard build is stale or nondeterministic: js/{relative}")


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


def _is_declared_docker_context_path(relative: str) -> bool:
    return (
        relative in {".dockerignore", "Dockerfile", "requirements-api-py311-linux.lock.txt"}
        or relative.startswith("deploy/api/")
        or relative.startswith("kga/")
    )


def _stage_docker_context(*, repo: Path, source_commit: str, destination: Path) -> tuple[str, ...]:
    """Materialize only immutable, declared runtime blobs into a fresh context."""

    raw = _git(
        repo,
        "ls-tree",
        "-r",
        "-z",
        source_commit,
        "--",
        *DOCKER_CONTEXT_PATHSPECS,
        binary=True,
    )
    assert isinstance(raw, bytes)
    entries = _parse_ls_tree_entries(raw)
    for relative, entry in entries.items():
        if not _is_declared_docker_context_path(relative):
            raise InventoryError(f"Docker context contains an undeclared path: {relative}")
        _safe_relative_source_path(relative)
        _regular_git_blob(relative, entry)

    required = {".dockerignore", "Dockerfile", "requirements-api-py311-linux.lock.txt"}
    missing = sorted(required - entries.keys())
    if missing:
        raise InventoryError(f"Docker context is missing required release files: {missing}")
    if not any(relative.startswith("deploy/api/") for relative in entries):
        raise InventoryError("Docker context contains no deploy/api runtime source")
    if not any(relative.startswith("kga/") for relative in entries):
        raise InventoryError("Docker context contains no kga runtime source")

    for relative in sorted(entries):
        payload = _git(repo, "show", f"{source_commit}:{relative}", binary=True)
        if not isinstance(payload, bytes):
            raise InventoryError(f"Git did not return Docker-context bytes: {relative}")
        target = destination.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise InventoryError(f"could not stage Docker-context file: {relative}") from exc
        os.chmod(target, 0o755 if entries[relative][0] == "100755" else 0o644)
    return tuple(sorted(entries))


def _run_docker_gates(*, repo: Path, source_commit: str) -> None:
    tag = f"kbound-api:release-{source_commit[:12]}"
    with tempfile.TemporaryDirectory(prefix="kbound-docker-context-") as temporary:
        context = Path(temporary)
        _stage_docker_context(repo=repo, source_commit=source_commit, destination=context)
        _run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--tag",
                tag,
                str(context),
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


def run_all_gates(payload: Mapping[str, Any], *, repo: Path, python: str) -> PytestRunReceipt:
    """Execute every required Task 7 gate, failing at the first bad boundary."""

    paths = payload["pytest_paths"]
    records = payload.get("pytest_process_groups")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise InventoryError("pytest process-group coverage record is missing")
    verify_pytest_process_group_records(paths, records)
    verify_pytest_backed_release_gate_coverage(paths)
    source_commit = str(payload["source_commit"])
    pytest_receipt = _run_pytest_release(
        paths,
        repo=repo,
        python=python,
        source_commit=source_commit,
    )
    _run_validators(
        payload["validator_paths"],
        repo=repo,
        python=python,
        source_commit=source_commit,
    )
    hook_configuration = validate_immutable_hook_configuration(
        repo=repo,
        source_commit=source_commit,
    )
    if isinstance(payload, dict):
        # The advertised gate is promoted only after the exact source-bound
        # YAML and every reviewed mutating-hook selection boundary pass.
        payload["immutable_hook_configuration"] = hook_configuration
    quality_paths = validated_quality_python_paths(
        repo=repo,
        revision=source_commit,
    )
    _run_quality_gates(
        repo=repo,
        python=python,
        quality_paths=quality_paths,
        source_commit=source_commit,
    )
    private_path_scan = run_private_path_scan(repo=repo, source_commit=source_commit)
    if isinstance(payload, dict):
        # A PASS is written only after the scanner actually completes; failed
        # executions leave the inventory's NOT_RUN record intact.
        payload["private_path_scan"] = private_path_scan
    _run_package_gate(repo=repo, python=python)
    _run_dashboard_gate(repo=repo)
    _run_lean_gates(repo=repo, python=python)
    _run_docker_gates(repo=repo, source_commit=source_commit)
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
        help=("explicitly authorize tests and validators that read the real sealed So2Sat development authority"),
    )
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    commit, tree = source_identity(repo)
    if args.expected_source_commit is not None:
        verify_expected_source_commit(actual=commit, expected=args.expected_source_commit)
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
        payload["execution_scope"] = "pytest_only"
        try:
            returncode = run_pytest_modules(
                payload["pytest_paths"],
                repo=repo,
                python=args.python,
                source_commit=commit,
            )
        except (InventoryError, OSError, subprocess.SubprocessError) as exc:
            returncode = 1
            payload["failure"] = str(exc)
            print(f"repository pytest execution: FAIL: {exc}", file=sys.stderr)
        # This mode does not reconcile release skips/warnings or run the other
        # required gates; its receipt must never resemble an all-gates PASS.
        payload["execution_status"] = "PYTEST_ONLY_PASS" if returncode == 0 else "PYTEST_ONLY_FAIL"
        payload["pytest_exit_code"] = returncode
        payload["inventory_sha256"] = inventory_digest(payload)
        write_canonical_json(output, payload)
        return returncode
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
    payload["executed_gates"] = [gate.name for gate in release_gate_plan(python=args.python)]
    payload["inventory_sha256"] = inventory_digest(payload)
    write_canonical_json(output, payload)
    print("repository verification: PASS (all required gates, no unexpected skips/warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
