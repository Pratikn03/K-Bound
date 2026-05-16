import pandas as pd
import pytest

from uais.fusion.attention.attention_utils import (
    build_fusion_tensors,
    validate_fusion_schema,
    validate_incident_protocol,
)


def test_validate_fusion_schema_basic():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b"],
            "domain": ["fraud", "cyber", "fraud"],
            "score": [0.9, 0.2, 0.1],
            "confidence": [0.8, 0.7, 0.6],
            "embedding_0": [0.1, 0.2, 0.3],
            "label": [1, 1, 0],
        }
    )
    stats = validate_fusion_schema(df)
    assert stats["rows"] == 3
    assert stats["embedding_dim"] == 1


def test_build_fusion_tensors_with_timestamp():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "a", "a", "a"],
            "timestamp": ["t1", "t1", "t2", "t2"],
            "domain": ["fraud", "cyber", "fraud", "cyber"],
            "score": [0.9, 0.2, 0.4, 0.6],
            "label": [1, 1, 0, 0],
        }
    )
    features, masks, labels, sample_ids, domain_order = build_fusion_tensors(
        df,
        id_column="sample_id",
        domain_column="domain",
        label_column="label",
        timestamp_column="timestamp",
        feature_columns=["score"],
    )
    assert features.shape == (2, len(domain_order), 1)
    assert labels.shape[0] == 2
    assert len(sample_ids) == 2


def test_validate_incident_protocol_accepts_naturally_coobserved_temporal_splits():
    df = pd.DataFrame(
        {
            "incident_id": ["i1", "i1", "i2", "i2", "i3", "i3"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:03",
                    "2026-02-01T00:00:00",
                    "2026-02-01T00:00:02",
                    "2026-03-01T00:00:00",
                    "2026-03-01T00:00:04",
                ]
            ),
            "split": ["train", "train", "validation", "validation", "test", "test"],
            "domain": ["network", "auth", "network", "auth", "network", "auth"],
            "label": [0, 0, 1, 1, 1, 1],
            "score": [0.1, 0.2, 0.8, 0.7, 0.9, 0.6],
        }
    )

    stats = validate_incident_protocol(df, min_domains_per_incident=2)

    assert stats["natural_pairing"] is True
    assert stats["incident_count"] == 3
    assert stats["min_domains_per_incident"] == 2
    assert stats["temporal_order_valid"] is True
    assert stats["split_leakage_count"] == 0


def test_validate_incident_protocol_rejects_split_leakage():
    df = pd.DataFrame(
        {
            "incident_id": ["i1", "i1", "i1", "i2"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:02",
                    "2026-01-02T00:00:00",
                    "2026-02-01T00:00:00",
                ]
            ),
            "split": ["train", "train", "test", "test"],
            "domain": ["network", "auth", "network", "network"],
            "label": [0, 0, 0, 1],
            "score": [0.1, 0.2, 0.3, 0.8],
        }
    )

    with pytest.raises(ValueError, match="split leakage"):
        validate_incident_protocol(df, min_domains_per_incident=2)


def test_validate_incident_protocol_can_report_non_temporal_replay_without_accepting_it_as_temporal():
    df = pd.DataFrame(
        {
            "incident_id": ["i1", "i1", "i2", "i2", "i3", "i3"],
            "timestamp": pd.to_datetime(
                [
                    "2026-03-01T00:00:00",
                    "2026-03-01T00:00:03",
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:02",
                    "2026-02-01T00:00:00",
                    "2026-02-01T00:00:04",
                ]
            ),
            "split": ["train", "train", "validation", "validation", "test", "test"],
            "domain": ["network", "auth", "network", "auth", "network", "auth"],
            "label": [0, 0, 1, 1, 1, 1],
            "score": [0.1, 0.2, 0.8, 0.7, 0.9, 0.6],
        }
    )

    with pytest.raises(ValueError, match="Temporal split order"):
        validate_incident_protocol(df, min_domains_per_incident=2)

    stats = validate_incident_protocol(
        df,
        min_domains_per_incident=2,
        require_temporal_order=False,
    )

    assert stats["natural_pairing"] is True
    assert stats["temporal_order_valid"] is False
