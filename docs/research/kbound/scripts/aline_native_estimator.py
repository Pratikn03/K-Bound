"""Authenticated Baek ALine array callable; no model loading or action controller.

Git or explicit fixed-commit source-receipt authentication is mandatory/offline.
No target labels/outcomes accepted. No automatic network or source fallback.
The exact three native notebook definitions execute; no clipping or regularizing.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

COMMIT = "99cb46ee1b50960c04b11d8e2f27943e68cbfbec"
NOTEBOOK_SHA256 = "cef78f7d196c2f86a96d466a30a6967c31eaa84cb176edd072d5356fa89344fc"
ORIGIN = "https://github.com/kebaek/Agreement-on-the-line.git"
URL = f"https://raw.githubusercontent.com/kebaek/Agreement-on-the-line/{COMMIT}/agreement_trajectory.ipynb"
FIELDS = {"model_ids", "checkpoint_sha256", "clean_sample_ids", "target_sample_ids",
          "clean_predictions", "target_predictions", "clean_accuracies", "class_count"}
SUPPORTED_RUNTIME = {"python": "3.12.12", "numpy": "2.4.4", "scipy": "1.13.1", "statsmodels": "0.14.6"}


def _runtime():
    declared = os.environ.get("ALINE_PYTHON")
    if not declared:
        raise ValueError("ALINE_PYTHON must explicitly name the prepared method interpreter")
    if Path(declared).resolve() != Path(sys.executable).resolve():
        raise ValueError("ALINE_PYTHON does not match the executing interpreter")
    versions = {"python": ".".join(map(str, sys.version_info[:3]))}
    try:
        versions.update({name: importlib.metadata.version(name) for name in ("numpy", "scipy", "statsmodels")})
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("declared ALine runtime is missing a required package") from exc
    if versions != SUPPORTED_RUNTIME:
        raise ValueError(f"unsupported ALine method runtime: {versions}")
    return {**versions, "executable": str(Path(sys.executable).resolve()), "declared_executable": declared}


def bank_sha256(bank):
    """Bind ordered identities, source accuracies and complete prediction arrays."""
    return hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _git(source, *arguments):
    try:
        return subprocess.check_output(
            ["git", "-c", "protocol.allow=never", "-C", str(source), *arguments],
            stderr=subprocess.PIPE, timeout=15,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0"))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError("offline pinned Git authentication unavailable; no working-copy fallback") from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate source receipt key")
        result[key] = value
    return result


def _native(source, source_receipt=None):
    import numpy as np
    import statsmodels.api as sm
    from scipy.stats import norm

    payload = (Path(source) / "agreement_trajectory.ipynb").read_bytes()
    if hashlib.sha256(payload).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("ALine notebook SHA-256 mismatch")
    if source_receipt is None:
        if _git(source, "rev-parse", "HEAD").decode().strip() != COMMIT:
            raise ValueError("ALine pinned commit mismatch")
        if _git(source, "show", f"{COMMIT}:agreement_trajectory.ipynb") != payload:
            raise ValueError("ALine working bytes differ from pinned Git blob")
        authority = {"method": "pinned_local_git_blob"}
    else:
        raw = Path(source_receipt).read_bytes()
        receipt = json.loads(raw, object_pairs_hook=_unique_object)
        expected = {"schema": "aline-fixed-commit-source-v1", "status": "EXACT_SOURCE_HASH_VERIFIED",
                    "origin": ORIGIN, "url": URL, "commit": COMMIT, "sha256": NOTEBOOK_SHA256,
                    "bytes": len(payload), "path": str((Path(source)/"agreement_trajectory.ipynb").resolve())}
        if not isinstance(receipt, dict) or any(receipt.get(k) != v or type(receipt.get(k)) is not type(v)
                                                for k, v in expected.items()):
            raise ValueError("source receipt does not bind exact fixed origin/commit/path/bytes")
        authority = {"method": "fixed_commit_public_source_receipt", "receipt_path": str(source_receipt),
                     "receipt_sha256": hashlib.sha256(raw).hexdigest(), "git_objects": "NOT_USED_EXPLICIT_RECEIPT"}
    notebook = json.loads(payload)
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(text)
    names = {"rescale", "compute_linear_fit", "aline"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(definitions) != 3 or {node.name for node in definitions} != names:
        raise ValueError("expected exactly three native function definitions")
    source_text = "\n".join(ast.get_source_segment(text, node) for node in definitions)
    namespace = dict(np=np, sm=sm, norm=norm)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "authenticated-aline-notebook", "exec"), namespace)
    return namespace["aline"], hashlib.sha256(source_text.encode()).hexdigest(), authority


def _validate(bank):
    import numpy as np
    from scipy.stats import norm

    if not isinstance(bank, dict) or set(bank) != FIELDS:
        raise ValueError("exact bank fields required; target outcomes are forbidden")
    def identities(key):
        values = bank[key]
        if (not isinstance(values, list) or not values or
                any(not isinstance(x, str) or not x.strip() for x in values) or len(set(values)) != len(values)):
            raise ValueError(f"invalid or duplicate {key}")
        return values
    ids = identities("model_ids")
    checkpoints = identities("checkpoint_sha256")
    if len(ids) < 3 or len(checkpoints) != len(ids) or any(not re.fullmatch("[0-9a-f]{64}", x) for x in checkpoints):
        raise ValueError("at least three distinct identified checkpoint members required")
    clean_ids, target_ids = identities("clean_sample_ids"), identities("target_sample_ids")
    if set(clean_ids) & set(target_ids):
        raise ValueError("clean and target identity namespaces must be disjoint")
    classes = bank["class_count"]
    if type(classes) is not int or classes < 2:
        raise ValueError("invalid class count")
    arrays = []
    for key, samples in (("clean_predictions", clean_ids), ("target_predictions", target_ids)):
        rows = bank[key]
        if (not isinstance(rows, list) or len(rows) != len(ids) or
                any(not isinstance(row, list) or len(row) != len(samples) for row in rows) or
                any(type(x) is not int or not 0 <= x < classes for row in rows for x in row)):
            raise ValueError("prediction arrays must align with ordered models/samples and integer class IDs")
        arrays.append(np.asarray(rows, dtype=np.int64))
    accuracy = bank["clean_accuracies"]
    if (not isinstance(accuracy, list) or len(accuracy) != len(ids) or
            any(type(x) not in (int, float) or not 0 < x < 1 for x in accuracy)):
        raise ValueError("finite clean accuracies strictly inside (0,1) required")
    if not np.isfinite(norm.ppf(accuracy)).all():
        raise ValueError("nonfinite clean probit")
    # Validate the exact native retained-pair system, without changing it.
    A, source_agreement = [], []
    clean, target = arrays
    for i in range(len(ids)):
        for j in range(i, len(ids)):
            ca, ta = np.mean(clean[i] == clean[j]), np.mean(target[i] == target[j])
            if .05 <= ca <= .98 and .05 <= ta <= .98:
                row = np.zeros(len(ids)); row[i] = .5; row[j] = .5
                A.append(row); source_agreement.append(ca)
    if not A or np.linalg.matrix_rank(np.asarray(A)) != len(ids):
        raise ValueError("native retained-pair model design is empty or rank deficient")
    fit = np.column_stack((np.ones(len(A)), norm.ppf(source_agreement)))
    if np.linalg.matrix_rank(fit) != 2:
        raise ValueError("native agreement regression lacks intercept/slope identification")
    return clean, np.asarray(accuracy), target, len(A)


def estimate(bank, *, expected_bank_sha256, source, source_receipt=None):
    """Return ALine-D accuracies and ALine-S probits, never action decisions.

    ``bank`` is a JSON-compatible mapping of ordered class-ID matrices and
    explicit identities. The expected digest must be locked externally first.
    """
    runtime = _runtime()
    import numpy as np
    clean, accuracy, target, pairs = _validate(bank)
    binding = bank_sha256(bank)
    if not isinstance(expected_bank_sha256, str) or binding != expected_bank_sha256:
        raise ValueError("bank identity/data binding mismatch")
    native, definitions_hash, authority = _native(Path(source), source_receipt)
    (s, d), bias, slope = native(clean, accuracy, target)
    if not all(np.isfinite(x).all() for x in (s, d, bias, slope)) or np.any(d < 0) or np.any(d > 1):
        raise ValueError("nonfinite or invalid native estimate; no repair applied")
    return {"method": "Baek_Agreement_on_the_Line", "aline_s_probit": s.tolist(),
            "aline_d_accuracy": d.tolist(), "bias": float(bias), "slope": float(slope),
            "model_ids": list(bank["model_ids"]), "bank_sha256": binding,
            "upstream_commit": COMMIT, "notebook_sha256": NOTEBOOK_SHA256,
            "native_definitions_sha256": definitions_hash, "retained_pairs": pairs,
            "source_authority": authority,
            "runtime": runtime, "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "benchmark_completed": False, "target_labels_consumed": False,
            "scope": "authenticated_native_estimator_under_array_protocol_adapter"}
