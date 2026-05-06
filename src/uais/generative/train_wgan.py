"""Wasserstein GAN with Gradient Penalty (WGAN-GP) for tabular synthetic data.

Architecture follows Gulrajani et al. (2017): a critic trained without
BatchNorm using a gradient-penalty regulariser instead of weight clipping,
and a generator with LayerNorm for training stability on tabular inputs.

Reference: https://arxiv.org/abs/1704.00028
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class WGANConfig:
    dataset_path: Path
    latent_dim: int = 128
    generator_hidden: Tuple[int, ...] = field(default_factory=lambda: (256, 256))
    critic_hidden: Tuple[int, ...] = field(default_factory=lambda: (256, 256))
    epochs: int = 100
    batch_size: int = 128
    # Critic steps per generator step — recommended >=5 for WGAN-GP
    n_critic: int = 5
    # Gradient penalty coefficient (lambda in the paper)
    lambda_gp: float = 10.0
    lr: float = 1e-4
    # Adam beta1=0 recommended for WGAN-GP (Gulrajani et al.)
    beta1: float = 0.0
    beta2: float = 0.9
    random_state: int = 42
    test_size: float = 0.2

    def resolve_path(self) -> Path:
        path = Path(self.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path


class _Generator(nn.Module):
    """Maps latent noise z → synthetic tabular sample."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: Tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _Critic(nn.Module):
    """Scores how 'real' a tabular sample is — no sigmoid, no BatchNorm."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            # BatchNorm is intentionally omitted: it destabilises WGAN-GP training
            layers += [nn.Linear(in_dim, h), nn.LeakyReLU(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _gradient_penalty(
    critic: _Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Two-sided gradient penalty: enforces ||∇D(x̂)||₂ ≈ 1 at interpolations."""
    batch = real.size(0)
    alpha = torch.rand(batch, 1, device=device)
    interpolated = (alpha * real + (1.0 - alpha) * fake.detach()).requires_grad_(True)
    d_interp = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = gradients.view(batch, -1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def run_wgan_pipeline(cfg: WGANConfig) -> Dict[str, object]:
    """Train a WGAN-GP on numeric tabular data.

    Returns a dict containing training history, final losses, estimated
    Wasserstein distance on held-out data, and the trained model objects
    needed for synthetic data generation.
    """
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    path = cfg.resolve_path()
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df = df.select_dtypes(include=[np.number]).dropna()
    if df.empty:
        raise ValueError("No numeric columns found for WGAN training.")

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values).astype("float32")
    X_train, X_test = train_test_split(X, test_size=cfg.test_size, random_state=cfg.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = X.shape[1]

    G = _Generator(cfg.latent_dim, output_dim, cfg.generator_hidden).to(device)
    C = _Critic(output_dim, cfg.critic_hidden).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_C = torch.optim.Adam(C.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)

    g_losses: list[float] = []
    c_losses: list[float] = []
    g_loss = torch.tensor(0.0)

    for epoch in range(cfg.epochs):
        perm = torch.randperm(len(X_tensor), device=device)
        X_tensor = X_tensor[perm]
        c_epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_tensor) - cfg.batch_size, cfg.batch_size):
            real = X_tensor[start : start + cfg.batch_size]
            n_batches += 1

            # Critic: maximise E[C(real)] - E[C(fake)] subject to GP constraint
            for _ in range(cfg.n_critic):
                z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device)
                fake = G(z).detach()
                gp = _gradient_penalty(C, real, fake, device)
                c_loss = C(fake).mean() - C(real).mean() + cfg.lambda_gp * gp
                opt_C.zero_grad()
                c_loss.backward()
                opt_C.step()
                c_epoch_loss += c_loss.item()

            # Generator: minimise -E[C(fake)]
            z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device)
            fake = G(z)
            g_loss = -C(fake).mean()
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        g_losses.append(float(g_loss.item()))
        c_losses.append(c_epoch_loss / max(n_batches * cfg.n_critic, 1))

    # Approximate Wasserstein distance on held-out test set
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    G.eval()
    C.eval()
    with torch.no_grad():
        z = torch.randn(len(X_test), cfg.latent_dim, device=device)
        synth = G(z)
        w_dist = float((C(X_test_tensor).mean() - C(synth).mean()).abs().item())

    return {
        "wasserstein_distance": w_dist,
        "final_g_loss": g_losses[-1],
        "final_c_loss": c_losses[-1],
        "generator": G,
        "critic": C,
        "scaler": scaler,
        "history": {"g_losses": g_losses, "c_losses": c_losses},
    }


def generate_synthetic_samples(
    generator: _Generator,
    n_samples: int,
    latent_dim: int,
    scaler: StandardScaler,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Draw n_samples from a trained generator, inverse-scaled to original units."""
    if device is None:
        device = next(generator.parameters()).device
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        synth = generator(z).cpu().numpy()
    return scaler.inverse_transform(synth)


__all__ = ["WGANConfig", "run_wgan_pipeline", "generate_synthetic_samples"]
