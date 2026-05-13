"""Dimensionality reduction applied CONSISTENTLY to all baselines.

Addresses reviewer comment (Part 1 + Part 2):
  "The Autoencoder is applied only to the proposed BGMM, while baseline models
   are evaluated without dimensionality reduction. This gives BGMM an unfair
   advantage due to noise reduction.
   → Apply Autoencoder or PCA consistently to all models, or clearly justify."

This module provides a unified DimReducer interface so the experiment script
can apply the EXACT same dimensionality reduction transform to every baseline
in the comparison.  Reducers are fit on the training fold only and applied
unchanged to validation and test folds — no leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DimReducer:
    """Abstract DR transform.  Subclasses implement fit() and transform()."""

    def fit(self, X: np.ndarray) -> "DimReducer":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class NoOpReducer(DimReducer):
    """Identity reducer — used as the baseline ('no DR applied')."""

    def fit(self, X: np.ndarray) -> "NoOpReducer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


@dataclass
class PCAReducerConfig:
    n_components: int = 16
    whiten: bool = False
    svd_solver: str = "auto"
    random_state: int = 42


class PCAReducer(DimReducer):
    """PCA wrapper with explicit train-fit / test-transform discipline."""

    def __init__(self, config: Optional[PCAReducerConfig] = None) -> None:
        self.config = config or PCAReducerConfig()
        self._scaler = StandardScaler()
        self._pca: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> "PCAReducer":
        X_s = self._scaler.fit_transform(X)
        n = min(self.config.n_components, X_s.shape[1], max(1, X_s.shape[0] - 1))
        self._pca = PCA(
            n_components=n,
            whiten=self.config.whiten,
            svd_solver=self.config.svd_solver,
            random_state=self.config.random_state,
        )
        self._pca.fit(X_s)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Call fit() first.")
        return self._pca.transform(self._scaler.transform(X)).astype(np.float32)

    def get_explained_variance_ratio(self) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Call fit() first.")
        return self._pca.explained_variance_ratio_.copy()


@dataclass
class AEReducerConfig:
    """Autoencoder feature-extractor architecture (encoder output used as features).

    Identical to AEConfig but with the encoder output exposed as the features
    rather than reconstruction error.
    """
    encoder_dims: List[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 16
    activation: str = "ReLU"
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    random_state: int = 42


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, cfg: AEReducerConfig) -> None:
        super().__init__()
        act = getattr(nn, cfg.activation)
        layers: List[nn.Module] = []
        prev = input_dim
        for h in cfg.encoder_dims:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, cfg.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, output_dim: int, cfg: AEReducerConfig) -> None:
        super().__init__()
        act = getattr(nn, cfg.activation)
        layers: List[nn.Module] = []
        prev = cfg.latent_dim
        for h in reversed(cfg.encoder_dims):
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AutoencoderReducer(DimReducer):
    """Trained AE — encoder output is used as the reduced representation.

    Reconstruction-loss-trained on training data only.  Transform applies just
    the encoder so it can be reused on val/test without leakage.
    """

    def __init__(self, config: Optional[AEReducerConfig] = None,
                 device: Optional[torch.device] = None) -> None:
        self.config = config or AEReducerConfig()
        self.device = device or torch.device("cpu")
        self._scaler = StandardScaler()
        self._encoder: Optional[_Encoder] = None
        self._decoder: Optional[_Decoder] = None

    def fit(self, X: np.ndarray) -> "AutoencoderReducer":
        c = self.config
        torch.manual_seed(c.random_state)
        np.random.seed(c.random_state)
        X_s = self._scaler.fit_transform(X).astype(np.float32)

        self._encoder = _Encoder(X_s.shape[1], c).to(self.device)
        self._decoder = _Decoder(X_s.shape[1], c).to(self.device)
        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        opt = torch.optim.Adam(params, lr=c.lr, weight_decay=c.weight_decay)

        # 10% validation split for early stopping (training data only)
        n_val = max(int(0.1 * len(X_s)), 1)
        perm = np.random.permutation(len(X_s))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_tr, X_val = X_s[tr_idx], X_s[val_idx]

        best_val, no_imp, best_state = float("inf"), 0, None
        for _ in range(c.epochs):
            self._encoder.train(); self._decoder.train()
            perm = np.random.permutation(len(X_tr))
            for start in range(0, len(X_tr), c.batch_size):
                idx = perm[start:start + c.batch_size]
                xb = torch.tensor(X_tr[idx], dtype=torch.float32, device=self.device)
                opt.zero_grad()
                recon = self._decoder(self._encoder(xb))
                ((recon - xb) ** 2).mean().backward()
                opt.step()
            self._encoder.eval(); self._decoder.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                val_loss = float(((self._decoder(self._encoder(xv)) - xv) ** 2).mean().item())
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = (
                    {k: v.clone() for k, v in self._encoder.state_dict().items()},
                    {k: v.clone() for k, v in self._decoder.state_dict().items()},
                )
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= c.patience:
                    break
        if best_state is not None:
            self._encoder.load_state_dict(best_state[0])
            self._decoder.load_state_dict(best_state[1])
        self._encoder.eval(); self._decoder.eval()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("Call fit() first.")
        with torch.no_grad():
            xt = torch.tensor(self._scaler.transform(X).astype(np.float32),
                              device=self.device)
            z = self._encoder(xt).cpu().numpy()
        return z.astype(np.float32)


# ---------------------------------------------------------------------------
# Factory: select DR by name (used by config-driven scripts)
# ---------------------------------------------------------------------------

def make_reducer(name: str, **kwargs) -> DimReducer:
    """Build a DimReducer by name.  Supported: 'none', 'pca', 'autoencoder'."""
    name = (name or "none").lower()
    if name in ("none", "noop", "identity"):
        return NoOpReducer()
    if name == "pca":
        return PCAReducer(PCAReducerConfig(**kwargs))
    if name in ("autoencoder", "ae"):
        return AutoencoderReducer(AEReducerConfig(**kwargs))
    raise ValueError(f"Unknown reducer: {name}")


__all__ = [
    "DimReducer", "NoOpReducer",
    "PCAReducerConfig", "PCAReducer",
    "AEReducerConfig", "AutoencoderReducer",
    "make_reducer",
]
