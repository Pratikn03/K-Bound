import pandas as pd
import torch
from torch.utils.data import DataLoader

from uais.fusion.attention.attention_utils import FusionDataset, build_fusion_tensors
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel


def test_attention_fusion_end_to_end():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b", "c", "c"],
            "domain": ["fraud", "cyber", "fraud", "cyber", "fraud", "cyber"],
            "score": [0.8, 0.2, 0.1, 0.4, 0.7, 0.6],
            "label": [1, 1, 0, 0, 1, 1],
        }
    )
    features, masks, labels, _, domain_order = build_fusion_tensors(
        df,
        id_column="sample_id",
        domain_column="domain",
        label_column="label",
        domain_order=None,
        feature_columns=["score"],
    )

    dataset = FusionDataset(features, masks, labels)
    loader = DataLoader(dataset, batch_size=2)

    model = AttentionFusionModel(
        num_domains=len(domain_order),
        input_dim=1,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        use_confidence=True,
    )

    batch = next(iter(loader))
    feats, masks_batch, labels_batch = batch
    logits, attn, conf = model(feats, key_padding_mask=masks_batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_batch.unsqueeze(-1))
    loss.backward()

    assert logits.shape[0] == labels_batch.shape[0]
    assert attn is not None
    assert conf is not None
