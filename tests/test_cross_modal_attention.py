import torch

from uais.fusion.attention.cross_modal_attention import CrossModalAttentionFusion


def test_cross_modal_attention_shapes():
    model = CrossModalAttentionFusion(num_domains=3, domain_embed_dim=16, num_heads=4)
    x = torch.randn(5, 3, 16)
    logits, attn = model(x)
    assert logits.shape == (5, 1)
    assert attn.shape[0] == 5


def test_cross_modal_attention_mask():
    model = CrossModalAttentionFusion(num_domains=2, domain_embed_dim=8, num_heads=2)
    x = torch.randn(2, 2, 8)
    mask = torch.tensor([[False, True], [False, False]])
    logits, attn = model(x, key_padding_mask=mask)
    assert logits.shape == (2, 1)
    assert attn.shape[-1] == 2


def test_cross_modal_attention_all_missing_row_is_finite():
    model = CrossModalAttentionFusion(num_domains=3, domain_embed_dim=12, num_heads=3)
    x = torch.randn(2, 3, 12)
    mask = torch.tensor([[True, True, True], [False, True, False]])
    logits, attn = model(x, key_padding_mask=mask)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attn).all()
