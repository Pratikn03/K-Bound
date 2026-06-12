"""Model registry, model cards, and artifact integrity verification.

This module provides a small, dependency-light registry for the supervised and
unsupervised scikit-learn artifacts that ship under ``models/`` in this repo.
It mirrors the integrity pattern already used by the deployment API
(``deploy/api/main.py``): artifacts are addressed by a SHA-256 hex digest, and
the digest can be overridden per-model via the environment variable
``UAIS_MODEL_SHA256_<NAME>`` (uppercased).

Design goals
------------
* **Standard library only** (``hashlib``, ``json``, ``pathlib``, ``dataclasses``).
  ``pyyaml`` is used *optionally* and only when reading ``.yaml`` manifests; the
  JSON path (the default) has no third-party dependency.
* **Fail closed on integrity.** :meth:`ModelRegistry.load` recomputes the
  artifact digest and raises :class:`IntegrityError` on any mismatch rather than
  silently returning a possibly-tampered artifact.
* **Honest metadata.** :class:`ModelCard` carries the human-authored model-card
  fields (task, framework, training data, intended use, metrics, limitations)
  so that the registry and the on-disk ``MODEL_CARD.md`` files stay aligned.

The registry deliberately does **not** import scikit-learn, joblib, or torch.
Loading the actual estimator object is left to the caller; :meth:`ModelRegistry.load`
verifies integrity and hands back the verified absolute :class:`pathlib.Path`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "IntegrityError",
    "ManifestError",
    "ModelCard",
    "ModelRegistry",
    "sha256_file",
]

# Default location of the artifact manifest, relative to the repository root.
DEFAULT_MANIFEST_PATH = "models/MANIFEST.json"

# Sentinel digest written by ``build_manifest`` before real hashes are filled in.
PENDING_SHA256 = "PENDING"

# Read size for streaming hash computation (1 MiB), matching deploy/api/main.py.
_HASH_CHUNK_BYTES = 1024 * 1024


class ManifestError(RuntimeError):
    """Raised when a manifest is missing, malformed, or internally inconsistent."""


class IntegrityError(RuntimeError):
    """Raised when an artifact's on-disk SHA-256 does not match the manifest.

    Carrying the offending fields on the exception makes failures actionable in
    logs and tests without re-deriving them at the call site.
    """

    def __init__(self, name: str, path: Path, expected: str, actual: str) -> None:
        self.name = name
        self.path = Path(path)
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Integrity check failed for model {name!r} at {self.path}: "
            f"expected sha256={expected}, computed sha256={actual}"
        )


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file.

    The file is read in fixed-size chunks so that arbitrarily large artifacts do
    not have to be held in memory. This matches the ``_sha256`` helper in
    ``deploy/api/main.py`` so digests are interchangeable across the codebase.

    Args:
        path: Filesystem path to the artifact to hash.

    Returns:
        The hex-encoded SHA-256 digest in lowercase.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ModelCard:
    """Structured, honest metadata for a single model artifact.

    The fields intentionally mirror the headings used in the human-readable
    ``MODEL_CARD.md`` files so the two never drift apart. ``metrics`` is a free
    -form mapping (e.g. ``{"test_roc_auc": 0.892}``); when a metric was not
    recorded in the repository the convention is to store the string
    ``"not recorded in repo"`` rather than to invent a number.

    Attributes:
        name: Unique registry key for the model (e.g. ``"fraud"``).
        task: Short description of the prediction task.
        framework: Modeling framework / estimator family (e.g.
            ``"scikit-learn (HistGradientBoostingClassifier)"``).
        train_data: Description of the training dataset.
        intended_use: Intended/appropriate use of the model.
        metrics: Mapping of metric name to value (numeric) or to a string such
            as ``"not recorded in repo"``.
        limitations: Free-text known limitations and caveats.
        version: Card/model version string.
        created: ISO-8601 creation timestamp (string; may be empty if unknown).
        sha256: Expected SHA-256 hex digest of the artifact, or
            ``"see MANIFEST.json"`` when the digest is tracked centrally.
        path: Repository-root-relative path to the artifact.
    """

    name: str
    task: str
    framework: str
    train_data: str
    intended_use: str
    metrics: dict[str, Any] = field(default_factory=dict)
    limitations: str = ""
    version: str = "1.0"
    created: str = ""
    sha256: str = "see MANIFEST.json"
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation of the card."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        """Construct a :class:`ModelCard` from a mapping.

        Unknown keys are ignored so that forward-compatible manifests (carrying
        extra fields) still load cleanly.

        Args:
            data: Mapping with at least the required card fields.

        Returns:
            A populated :class:`ModelCard`.

        Raises:
            KeyError: If a required field is missing.
        """
        required = ("name", "task", "framework", "train_data", "intended_use")
        for key in required:
            if key not in data:
                raise KeyError(f"ModelCard.from_dict missing required field: {key!r}")
        return cls(
            name=data["name"],
            task=data["task"],
            framework=data["framework"],
            train_data=data["train_data"],
            intended_use=data["intended_use"],
            metrics=dict(data.get("metrics", {})),
            limitations=data.get("limitations", ""),
            version=data.get("version", "1.0"),
            created=data.get("created", ""),
            sha256=data.get("sha256", "see MANIFEST.json"),
            path=data.get("path", ""),
        )


class ModelRegistry:
    """In-memory registry of model cards backed by an on-disk manifest.

    A manifest (see :mod:`uais.registry.build_manifest`) records, per artifact,
    its repository-relative ``path`` and expected ``sha256``. The registry holds
    :class:`ModelCard` objects keyed by name and uses the manifest to resolve and
    integrity-check artifacts at load time.

    Typical usage::

        reg = ModelRegistry(repo_root="/path/to/repo")
        reg.load_manifest()                 # reads models/MANIFEST.json
        reg.register(ModelCard(...))        # optional: attach rich metadata
        path = reg.load("fraud", verify=True)  # raises IntegrityError on mismatch

    The repository root defaults to four parents above this file
    (``src/uais/registry/model_registry.py`` -> repo root), which is correct for
    the in-repo layout but can be overridden for tests.
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[3]
        self.repo_root: Path = Path(repo_root)
        self._cards: dict[str, ModelCard] = {}
        # name -> {"path": str, "sha256": str, "size_bytes": int, "mtime": str}
        self._manifest: dict[str, dict[str, Any]] = {}
        self.manifest_path: Path | None = None

    # ------------------------------------------------------------------
    # Manifest handling
    # ------------------------------------------------------------------
    def load_manifest(self, path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, dict[str, Any]]:
        """Load a manifest file and index its artifacts by name.

        Supports JSON manifests (the default) and, if ``pyyaml`` is installed,
        ``.yaml``/``.yml`` manifests with the same schema.

        Args:
            path: Manifest path. Relative paths are resolved against the
                repository root.

        Returns:
            The internal name -> entry mapping that was loaded.

        Raises:
            ManifestError: If the file is missing, unparesable, lacks an
                ``artifacts`` list, or contains duplicate artifact names.
        """
        manifest_path = self._resolve(path)
        if not manifest_path.exists():
            raise ManifestError(f"Manifest not found: {manifest_path}")

        text = manifest_path.read_text(encoding="utf-8")
        suffix = manifest_path.suffix.lower()
        try:
            if suffix in {".yaml", ".yml"}:
                data = self._parse_yaml(text)
            else:
                data = json.loads(text)
        except ValueError as exc:  # json.JSONDecodeError subclasses ValueError
            raise ManifestError(f"Could not parse manifest {manifest_path}: {exc}") from exc

        if not isinstance(data, dict) or "artifacts" not in data:
            raise ManifestError(f"Manifest {manifest_path} must be a mapping with an 'artifacts' list")

        artifacts = data.get("artifacts")
        if not isinstance(artifacts, list):
            raise ManifestError(f"Manifest {manifest_path} 'artifacts' must be a list")

        indexed: dict[str, dict[str, Any]] = {}
        for entry in artifacts:
            if not isinstance(entry, dict) or "name" not in entry or "path" not in entry:
                raise ManifestError(f"Manifest {manifest_path} artifact entries require 'name' and 'path'")
            name = str(entry["name"])
            if name in indexed:
                raise ManifestError(f"Duplicate artifact name in manifest: {name!r}")
            indexed[name] = dict(entry)

        self._manifest = indexed
        self.manifest_path = manifest_path
        return self._manifest

    @staticmethod
    def _parse_yaml(text: str) -> Any:
        """Parse YAML text, raising :class:`ManifestError` if pyyaml is absent."""
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without pyyaml
            raise ManifestError("Reading a YAML manifest requires the optional 'pyyaml' dependency") from exc
        return yaml.safe_load(text)

    def manifest_entries(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of the loaded manifest entries by name."""
        return dict(self._manifest)

    # ------------------------------------------------------------------
    # Card handling
    # ------------------------------------------------------------------
    def register(self, card: ModelCard) -> None:
        """Register (or replace) a model card by its ``name``.

        Args:
            card: The :class:`ModelCard` to store.
        """
        self._cards[card.name] = card

    def get(self, name: str) -> ModelCard:
        """Return the registered :class:`ModelCard` for ``name``.

        Args:
            name: The model name.

        Returns:
            The stored model card.

        Raises:
            KeyError: If no card is registered under ``name``.
        """
        if name not in self._cards:
            raise KeyError(f"No model card registered for {name!r}")
        return self._cards[name]

    def names(self) -> Iterable[str]:
        """Return the registered model names."""
        return tuple(self._cards.keys())

    # ------------------------------------------------------------------
    # Integrity-checked loading
    # ------------------------------------------------------------------
    def _expected_sha256(self, name: str, entry: dict[str, Any]) -> str:
        """Resolve the expected digest for ``name``, honouring env overrides.

        ``UAIS_MODEL_SHA256_<NAME>`` (uppercased) takes precedence over the
        manifest value, mirroring the deployment API's override convention.
        """
        override = os.getenv(f"UAIS_MODEL_SHA256_{name.upper()}")
        if override:
            return override.strip().lower()
        return str(entry.get("sha256", "")).strip().lower()

    def resolve_path(self, name: str) -> Path:
        """Return the absolute artifact path for ``name`` from the manifest.

        Args:
            name: The model name.

        Returns:
            Absolute path to the artifact (not checked for existence here).

        Raises:
            ManifestError: If no manifest is loaded or ``name`` is unknown.
        """
        entry = self._manifest_entry(name)
        return self._resolve(entry["path"])

    def load(self, name: str, verify: bool = True) -> Path:
        """Resolve and integrity-check an artifact, returning its path.

        This method does **not** unpickle or otherwise deserialise the artifact;
        it returns the verified absolute path so the caller can load it with the
        appropriate framework (joblib/torch). Verification recomputes the
        SHA-256 of the file and compares it against the expected digest.

        Args:
            name: The model name as recorded in the manifest.
            verify: When ``True`` (default), recompute and compare the digest.

        Returns:
            Absolute :class:`pathlib.Path` to the verified artifact.

        Raises:
            ManifestError: If no manifest is loaded or ``name`` is unknown.
            FileNotFoundError: If the artifact file does not exist.
            IntegrityError: If verification is enabled and the digest mismatches,
                or if the manifest digest is still the ``"PENDING"`` sentinel.
        """
        entry = self._manifest_entry(name)
        artifact_path = self._resolve(entry["path"])
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact for {name!r} not found: {artifact_path}")

        if not verify:
            return artifact_path

        expected = self._expected_sha256(name, entry)
        if not expected or expected == PENDING_SHA256.lower():
            raise IntegrityError(
                name=name,
                path=artifact_path,
                expected=expected or "(empty)",
                actual=sha256_file(artifact_path),
            )

        actual = sha256_file(artifact_path)
        if actual.lower() != expected:
            raise IntegrityError(name=name, path=artifact_path, expected=expected, actual=actual)
        return artifact_path

    def verify_all(self) -> dict[str, Any]:
        """Verify every artifact in the manifest and return a structured report.

        The report does not raise on mismatch; instead each artifact is recorded
        with a status so callers (CLIs, CI gates) can decide how to react.

        Returns:
            A dict of the form::

                {
                    "ok": bool,                # True iff every artifact verified
                    "checked": int,
                    "artifacts": {
                        "<name>": {
                            "status": "ok" | "mismatch" | "missing" | "pending",
                            "path": "<relative path>",
                            "expected": "<hex or sentinel>",
                            "actual": "<hex or null>",
                        },
                        ...
                    },
                }

        Raises:
            ManifestError: If no manifest has been loaded.
        """
        if not self._manifest:
            raise ManifestError("No manifest loaded; call load_manifest() first")

        report: dict[str, Any] = {"ok": True, "checked": 0, "artifacts": {}}
        for name, entry in self._manifest.items():
            rel_path = str(entry.get("path", ""))
            artifact_path = self._resolve(rel_path)
            expected = self._expected_sha256(name, entry)
            record: dict[str, Any] = {
                "status": "ok",
                "path": rel_path,
                "expected": expected or "(empty)",
                "actual": None,
            }
            if not artifact_path.exists():
                record["status"] = "missing"
                report["ok"] = False
            elif not expected or expected == PENDING_SHA256.lower():
                record["status"] = "pending"
                record["actual"] = sha256_file(artifact_path)
                report["ok"] = False
            else:
                actual = sha256_file(artifact_path)
                record["actual"] = actual
                if actual.lower() != expected:
                    record["status"] = "mismatch"
                    report["ok"] = False
            report["artifacts"][name] = record
            report["checked"] += 1
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _manifest_entry(self, name: str) -> dict[str, Any]:
        if not self._manifest:
            raise ManifestError("No manifest loaded; call load_manifest() first")
        if name not in self._manifest:
            raise ManifestError(f"No artifact named {name!r} in manifest")
        return self._manifest[name]

    def _resolve(self, path: str | Path) -> Path:
        """Resolve ``path`` against the repository root if it is relative."""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.repo_root / candidate).resolve()
