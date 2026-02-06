import torch

from uais.fusion.attention.confidence_estimator import DomainConfidenceEstimator
from uais.fusion.attention.domain_encoders import DomainEncoder


def test_domain_encoder_output_shape():
    encoder = DomainEncoder(input_dim=3, embed_dim=8)
    x = torch.randn(4, 3)
    y = encoder(x)
    assert y.shape == (4, 8)


def test_confidence_estimator_output_shape():
    estimator = DomainConfidenceEstimator(embed_dim=8)
    x = torch.randn(2, 5, 8)
    conf = estimator(x)
    assert conf.shape == (2, 5)
    assert torch.all(conf >= 0)
    assert torch.all(conf <= 1)
