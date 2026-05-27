"""Validation helper for attention fusion inputs."""

from __future__ import annotations

import json
from pathlib import Path

from uais.fusion.attention.attention_utils import (
    hash_file,
    infer_feature_columns,
    load_fusion_dataframe,
    prepare_fusion_dataframe,
    validate_fusion_schema,
)
from uais.utils.config_loader import load_yaml
from uais.utils.paths import PROJECT_ROOT

DEFAULT_CONFIG = Path("src/uais/fusion/attention/attention_config.yaml")


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def validate_attention_inputs(cfg_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    cfg = load_yaml(cfg_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    validation_cfg = cfg.get("validation", {})

    data_path = _resolve_path(data_cfg.get("path", ""))
    df = load_fusion_dataframe(data_path)

    feature_columns = infer_feature_columns(
        df,
        score_column=data_cfg.get("score_column", "score"),
        confidence_column=data_cfg.get("confidence_column", "confidence"),
        embedding_prefix=data_cfg.get("embedding_prefix", "embedding_"),
        feature_columns=data_cfg.get("feature_columns") or None,
        include_confidence=bool(model_cfg.get("use_input_confidence", True)),
    )

    schema_stats = validate_fusion_schema(
        df,
        id_column=data_cfg.get("id_column", "sample_id"),
        domain_column=data_cfg.get("domain_column", "domain"),
        score_column=data_cfg.get("score_column", "score"),
        label_column=data_cfg.get("label_column", "label"),
        confidence_column=data_cfg.get("confidence_column", "confidence"),
        embedding_prefix=data_cfg.get("embedding_prefix", "embedding_"),
        timestamp_column=data_cfg.get("timestamp_column"),
    )

    _, prep_stats = prepare_fusion_dataframe(
        df,
        id_column=data_cfg.get("id_column", "sample_id"),
        domain_column=data_cfg.get("domain_column", "domain"),
        feature_columns=feature_columns,
        label_column=data_cfg.get("label_column", "label"),
        timestamp_column=data_cfg.get("timestamp_column"),
    )

    report = {
        "data_path": str(data_path),
        "data_hash": hash_file(data_path),
        "feature_columns": feature_columns,
        "schema_stats": schema_stats,
        "prep_stats": prep_stats,
    }

    report_path = validation_cfg.get("report_path")
    if report_path:
        out_path = _resolve_path(report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":  # pragma: no cover
    validate_attention_inputs()
