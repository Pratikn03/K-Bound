"""Phase 2.B — raw per-seed test prediction archive contract.

Every Phase 2 run produces, per (cell, method, seed, split), an
immutable archive of the per-sample prediction vector + the metadata
that makes the prediction inferentially usable. This module defines
the schema, the writer, and the index file.

Layout:

  experiments/phase2/predictions/<cell>/<method>/<split>/seed_<NN>.parquet

Each parquet file contains one row per sample with the columns in
`PREDICTION_ARCHIVE_SCHEMA`. The matching index row is appended to
`experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv`.

Immutability rule: an archive file may not be overwritten once its
hash is recorded in a locked analysis. Re-runs of the same (cell,
method, seed, split) tuple under a different code/config hash are
written as `seed_<NN>__rerun_<idx>.parquet` and indexed separately.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PREDICTION_ARCHIVE_SCHEMA: tuple[str, ...] = (
    # identity
    "sample_id",
    "benchmark",
    "protocol",
    "analysis_family",
    "pairing_strength",
    "split",
    "seed",
    "method",
    "method_variant",
    # selection provenance
    "selected_head_or_comparator_status",
    "selection_rule",
    "selection_used_test_metrics",
    # labels + predictions
    "label",
    "raw_score",
    "calibrated_score_if_used",
    "prediction_threshold_if_used",
    # gate behaviour
    "gate_mode",
    "gate_fired_if_applicable",
    "mean_reliability_if_applicable",
    "min_reliability_if_applicable",
    # corruption / failure context
    "failure_type_if_applicable",
    "failed_domain_count_if_applicable",
    "fault_severity_if_applicable",
    # provenance hashes
    "source_artifact_version",
    "config_hash",
    "code_commit_hash",
)


INDEX_COLUMNS: tuple[str, ...] = (
    "run_id",
    "experiment_id",
    "benchmark",
    "protocol",
    "seed",
    "method",
    "split",
    "artifact_path",
    "rows",
    "sha256",
    "config_hash",
    "commit_hash",
    "validation_only_selection_verified",
    "usable_for_inference",
    "created_utc",
)


@dataclass(frozen=True)
class ArchiveEntry:
    """In-memory descriptor for one archive file's index row."""

    run_id: str
    experiment_id: str
    benchmark: str
    protocol: str
    seed: int
    method: str
    split: str
    artifact_path: str
    rows: int
    sha256: str
    config_hash: str
    commit_hash: str
    validation_only_selection_verified: bool
    usable_for_inference: bool
    created_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_dict(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class PredictionArchive:
    """Writer + index for Phase 2 prediction artifacts.

    Usage:

        archive = PredictionArchive(root=Path("experiments/phase2/predictions"))
        for seed in range(30):
            for split, preds in [...]:
                df = archive.build_frame(...)
                entry = archive.write(
                    experiment_id="A-POWERED-1",
                    benchmark="MVTec 3D-AD",
                    protocol="PatchCore supervised-paired",
                    seed=seed,
                    method="rga_meta_router",
                    split=split,
                    frame=df,
                    config=cfg_dict,
                    selection_used_test_metrics=False,
                    validation_only_selection_verified=True,
                )
                archive.append_index(entry)
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "PREDICTION_ARCHIVE_INDEX.csv"
        if not self.index_path.exists():
            with self.index_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
                writer.writeheader()

    @staticmethod
    def build_frame(
        *,
        sample_ids: Iterable[Any],
        labels: np.ndarray,
        raw_scores: np.ndarray,
        method: str,
        method_variant: str | None,
        benchmark: str,
        protocol: str,
        analysis_family: str,
        pairing_strength: str,
        split: str,
        seed: int,
        selection_rule: str,
        selection_used_test_metrics: bool,
        selected_head_or_comparator_status: str,
        calibrated_scores: np.ndarray | None = None,
        prediction_threshold: float | None = None,
        gate_mode: str | None = None,
        gate_fired: np.ndarray | None = None,
        mean_reliability: np.ndarray | None = None,
        min_reliability: np.ndarray | None = None,
        failure_type: str | None = None,
        failed_domain_count: int | None = None,
        fault_severity: float | None = None,
        source_artifact_version: str = "phase2_v1",
        config_hash: str = "",
        code_commit_hash: str | None = None,
    ) -> pd.DataFrame:
        sample_ids = list(sample_ids)
        n = len(sample_ids)
        if labels.shape[0] != n or raw_scores.shape[0] != n:
            raise ValueError(
                f"sample_ids ({n}), labels ({labels.shape[0]}), raw_scores ({raw_scores.shape[0]}) must agree"
            )
        if code_commit_hash is None:
            code_commit_hash = _git_commit_hash()

        def _col(arr, default=None):
            if arr is None:
                return [default] * n
            return list(arr)

        data = {
            "sample_id": sample_ids,
            "benchmark": [benchmark] * n,
            "protocol": [protocol] * n,
            "analysis_family": [analysis_family] * n,
            "pairing_strength": [pairing_strength] * n,
            "split": [split] * n,
            "seed": [int(seed)] * n,
            "method": [method] * n,
            "method_variant": [method_variant or ""] * n,
            "selected_head_or_comparator_status": [selected_head_or_comparator_status] * n,
            "selection_rule": [selection_rule] * n,
            "selection_used_test_metrics": [bool(selection_used_test_metrics)] * n,
            "label": labels.astype(int).tolist(),
            "raw_score": raw_scores.astype(float).tolist(),
            "calibrated_score_if_used": _col(calibrated_scores, default=None),
            "prediction_threshold_if_used": [prediction_threshold if prediction_threshold is not None else None] * n,
            "gate_mode": [gate_mode or ""] * n,
            "gate_fired_if_applicable": _col(gate_fired, default=None),
            "mean_reliability_if_applicable": _col(mean_reliability, default=None),
            "min_reliability_if_applicable": _col(min_reliability, default=None),
            "failure_type_if_applicable": [failure_type or ""] * n,
            "failed_domain_count_if_applicable": [
                failed_domain_count if failed_domain_count is not None else None
            ] * n,
            "fault_severity_if_applicable": [
                fault_severity if fault_severity is not None else None
            ] * n,
            "source_artifact_version": [source_artifact_version] * n,
            "config_hash": [config_hash] * n,
            "code_commit_hash": [code_commit_hash] * n,
        }
        return pd.DataFrame(data, columns=list(PREDICTION_ARCHIVE_SCHEMA))

    def _path_for(
        self, experiment_id: str, benchmark: str, protocol: str, method: str, split: str, seed: int
    ) -> Path:
        cell_slug = (
            f"{experiment_id}__"
            + benchmark.replace(" ", "_").replace("/", "_")
            + "__"
            + protocol.replace(" ", "_").replace("/", "_")
        )
        out_dir = self.root / cell_slug / method / split
        out_dir.mkdir(parents=True, exist_ok=True)
        # immutability: don't overwrite existing files; append rerun suffix.
        base = out_dir / f"seed_{int(seed):02d}.parquet"
        if not base.exists():
            return base
        idx = 1
        while True:
            cand = out_dir / f"seed_{int(seed):02d}__rerun_{idx}.parquet"
            if not cand.exists():
                return cand
            idx += 1

    def write(
        self,
        *,
        experiment_id: str,
        benchmark: str,
        protocol: str,
        seed: int,
        method: str,
        split: str,
        frame: pd.DataFrame,
        config: dict | None = None,
        selection_used_test_metrics: bool = False,
        validation_only_selection_verified: bool = True,
    ) -> ArchiveEntry:
        if list(frame.columns) != list(PREDICTION_ARCHIVE_SCHEMA):
            missing = set(PREDICTION_ARCHIVE_SCHEMA) - set(frame.columns)
            extra = set(frame.columns) - set(PREDICTION_ARCHIVE_SCHEMA)
            raise ValueError(
                f"frame columns do not match the schema. missing={missing} extra={extra}"
            )
        if selection_used_test_metrics:
            raise ValueError(
                "selection_used_test_metrics=True is FORBIDDEN under Phase 2 contract"
            )
        path = self._path_for(experiment_id, benchmark, protocol, method, split, seed)
        try:
            frame.to_parquet(path, index=False)
        except Exception as e:  # parquet engine missing → fallback CSV
            path = path.with_suffix(".csv")
            frame.to_csv(path, index=False)
        sha = _hash_file(path)
        cfg_hash = _hash_dict(config or {})
        commit = _git_commit_hash()
        run_id = f"{experiment_id}_{method}_{split}_seed{int(seed):02d}_{sha[:8]}"
        # Try relative-to-cwd; fall back to absolute path if outside cwd.
        try:
            artifact_path = str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            artifact_path = str(path.resolve())
        return ArchiveEntry(
            run_id=run_id,
            experiment_id=experiment_id,
            benchmark=benchmark,
            protocol=protocol,
            seed=int(seed),
            method=method,
            split=split,
            artifact_path=artifact_path,
            rows=int(len(frame)),
            sha256=sha,
            config_hash=cfg_hash,
            commit_hash=commit,
            validation_only_selection_verified=bool(validation_only_selection_verified),
            usable_for_inference=(not selection_used_test_metrics),
            created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def append_index(self, entry: ArchiveEntry) -> None:
        with self.index_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
            writer.writerow(entry.to_dict())

    def load_index(self) -> pd.DataFrame:
        if not self.index_path.exists():
            return pd.DataFrame(columns=list(INDEX_COLUMNS))
        return pd.read_csv(self.index_path)


__all__ = [
    "PREDICTION_ARCHIVE_SCHEMA",
    "INDEX_COLUMNS",
    "ArchiveEntry",
    "PredictionArchive",
]
