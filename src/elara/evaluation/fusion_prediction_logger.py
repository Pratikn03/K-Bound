"""Write Master Scenario C prediction archives from fusion experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from elara.evaluation.prediction_archive import PredictionArchive, _hash_dict


def write_seed_archives(
    archive: PredictionArchive,
    *,
    experiment_id: str,
    benchmark: str,
    protocol: str,
    analysis_family: str,
    pairing_strength: str,
    seed: int,
    sample_ids: list[Any],
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    split: str = "test",
    selection_rule: str = "validation_frozen",
    gate_fired: np.ndarray | None = None,
    mean_reliability: np.ndarray | None = None,
    config: dict | None = None,
) -> list[str]:
    """Write one parquet per method; return artifact paths."""
    cfg_hash = _hash_dict(config or {})
    paths: list[str] = []
    for method, probs in predictions.items():
        frame = PredictionArchive.build_frame(
            sample_ids=sample_ids,
            labels=labels,
            raw_scores=np.asarray(probs, dtype=float),
            method=method,
            method_variant="master_c",
            benchmark=benchmark,
            protocol=protocol,
            analysis_family=analysis_family,
            pairing_strength=pairing_strength,
            split=split,
            seed=int(seed),
            selection_rule=selection_rule,
            selection_used_test_metrics=False,
            selected_head_or_comparator_status="frozen_development",
            gate_mode="mean" if gate_fired is not None else None,
            gate_fired=gate_fired if method == "craf_attention" else None,
            mean_reliability=mean_reliability if method == "craf_attention" else None,
            config_hash=cfg_hash,
            source_artifact_version="master_c_v1",
        )
        entry = archive.write(
            experiment_id=experiment_id,
            benchmark=benchmark,
            protocol=protocol,
            seed=int(seed),
            method=method,
            split=split,
            frame=frame,
            config=config,
            validation_only_selection_verified=True,
        )
        archive.append_index(entry)
        paths.append(entry.artifact_path)
    return paths
