import pandas as pd

from uais.fusion.attention.attention_utils import build_fusion_tensors, validate_fusion_schema


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
