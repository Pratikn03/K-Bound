"""Unsupervised anomaly-detection baselines (normal-only training protocol).

Addresses reviewer comments (Part 2):
  - "Anomaly detection models must be trained only on non-fraudulent transactions,
     with fraud samples used only during testing."
  - "Inconsistency between oversampling and unsupervised learning"
  - "Missing baseline configuration details: number of clusters for K-means
     and GMM, initialization methods, number of trees and subsampling size for
     Isolation Forest, random seeds"

Every detector here:
  1. Receives (features, masks, labels) at fit() time but discards labels except
     to FILTER to the normal-only subset (labels == 0).  Anomaly labels are
     never seen during training.
  2. Exposes a fully-specified hyperparameter registry on the class.
  3. Sets every random seed it controls.
  4. Returns anomaly scores in [0, 1] from predict_proba().

Detectors provided:
  BGMMAnomalyDetector      — Bayesian Gaussian Mixture (Dirichlet-process prior,
                             unbounded effective component count)
  GMMAnomalyDetector       — Standard EM-fit Gaussian Mixture
  KMeansAnomalyDetector    — distance-to-nearest-centroid as anomaly score
  IsolationForestDetector  — sklearn IsolationForest (fully-configurable)
  OneClassSVMDetector      — sklearn OneClassSVM (RBF kernel by default)
  LOFAnomalyDetector       — Local Outlier Factor in novelty mode
  AutoencoderAnomalyDetector — reconstruction-error scoring with a fully-specified
                               symmetric MLP autoencoder

All scores are normalised to [0, 1] using train-set percentile calibration so
they can be directly compared and used as input to the fusion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _flatten_with_mask(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """[N, D, F] + [N, D] bool → [N, D*F + D] float (masked → 0.5 + indicator)."""
    n, d, f = features.shape
    flat = features.copy().astype(np.float32)
    for di in range(d):
        flat[:, di, :][masks[:, di]] = 0.5
    flat = flat.reshape(n, d * f)
    indicators = masks.astype(np.float32)
    return np.concatenate([flat, indicators], axis=1)


def _normal_only(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter to label==0 rows.  Anomaly labels are never seen during fit().

    This enforces the reviewer's required protocol for unsupervised models.
    """
    if labels is None:
        return features, masks
    keep = np.asarray(labels) == 0
    return features[keep], masks[keep]


def _percentile_calibrate(
    raw_scores: np.ndarray,
    train_raw_scores: np.ndarray,
) -> np.ndarray:
    """Map raw scores to [0, 1] using the training-set rank distribution.

    score = P(train_raw <= raw); larger raw → higher anomaly probability.
    """
    sorted_train = np.sort(train_raw_scores)
    ranks = np.searchsorted(sorted_train, raw_scores, side="right")
    return (ranks / max(len(sorted_train), 1)).astype(np.float32).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Hyperparameter registries — every value documented for reproducibility
# ---------------------------------------------------------------------------


@dataclass
class BGMMConfig:
    n_components: int = 10
    weight_concentration_prior_type: str = "dirichlet_process"
    weight_concentration_prior: float = 1.0
    covariance_type: str = "full"
    max_iter: int = 200
    tol: float = 1e-3
    init_params: str = "kmeans"
    random_state: int = 42


@dataclass
class GMMConfig:
    n_components: int = 8
    covariance_type: str = "full"
    max_iter: int = 200
    tol: float = 1e-3
    init_params: str = "kmeans"
    n_init: int = 1
    random_state: int = 42


@dataclass
class KMeansConfig:
    n_clusters: int = 8
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42


@dataclass
class IForestConfig:
    n_estimators: int = 200
    max_samples: str = "auto"  # 256 or len(X)
    contamination: str = "auto"  # let sklearn pick
    max_features: float = 1.0
    bootstrap: bool = False
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class OCSVMConfig:
    kernel: str = "rbf"
    nu: float = 0.05
    gamma: str = "scale"
    tol: float = 1e-3
    cache_size: int = 200


@dataclass
class LOFConfig:
    n_neighbors: int = 20
    algorithm: str = "auto"
    metric: str = "minkowski"
    contamination: str = "auto"
    novelty: bool = True  # required for predict() on new data
    n_jobs: int = -1


@dataclass
class AEConfig:
    """Fully-specified Autoencoder architecture.

    Addresses reviewer comment: 'Full Autoencoder architecture (layers,
    activations, latent size)'.
    """

    encoder_dims: list[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 8
    decoder_dims: list[int] = field(default_factory=lambda: [32, 64])
    activation: str = "ReLU"
    dropout: float = 0.0
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    random_state: int = 42


# ---------------------------------------------------------------------------
# Mixin: shared interface
# ---------------------------------------------------------------------------


class _BaseUnsupervised:
    """Shared init / scaler / state for all unsupervised baselines."""

    _config_cls = None

    def __init__(self, config=None) -> None:
        self.config = config or self._config_cls()
        self._scaler = StandardScaler()
        self._train_raw_scores: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API — every detector implements _fit_normal and _raw_scores
    # ------------------------------------------------------------------
    def fit(self, features: np.ndarray, masks: np.ndarray, labels: np.ndarray | None = None) -> _BaseUnsupervised:
        f_norm, m_norm = _normal_only(features, masks, labels)
        if len(f_norm) == 0:
            raise ValueError(
                f"{type(self).__name__}.fit: no normal samples after filtering. "
                "Did you pass labels with at least one 0?"
            )
        X_norm = self._scaler.fit_transform(_flatten_with_mask(f_norm, m_norm))
        self._fit_normal(X_norm)
        self._train_raw_scores = self._raw_scores(X_norm)
        self._fitted = True
        return self

    def predict_proba(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}.fit must be called first.")
        X = self._scaler.transform(_flatten_with_mask(features, masks))
        raw = self._raw_scores(X)
        return _percentile_calibrate(raw, self._train_raw_scores)

    def get_hyperparameters(self) -> dict:
        """Return all hyperparameters used by this baseline (for reproducibility)."""
        return {k: getattr(self.config, k) for k in self.config.__dataclass_fields__}

    # ------------------------------------------------------------------
    # Subclasses implement these two
    # ------------------------------------------------------------------
    def _fit_normal(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. BGMM — Bayesian Gaussian Mixture (paper's headline proposal)
# ---------------------------------------------------------------------------


class BGMMAnomalyDetector(_BaseUnsupervised):
    """Bayesian Gaussian Mixture with Dirichlet-process prior.

    Anomaly score: negative log-likelihood under the fitted mixture.
    """

    _config_cls = BGMMConfig

    def _fit_normal(self, X: np.ndarray) -> None:
        c = self.config
        self._model = BayesianGaussianMixture(
            n_components=c.n_components,
            weight_concentration_prior_type=c.weight_concentration_prior_type,
            weight_concentration_prior=c.weight_concentration_prior,
            covariance_type=c.covariance_type,
            max_iter=c.max_iter,
            tol=c.tol,
            init_params=c.init_params,
            random_state=c.random_state,
        )
        self._model.fit(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X).astype(np.float32)

    def get_effective_n_components(self) -> int:
        """Count mixture components with non-negligible weight (>1%).

        Addresses reviewer comment about reporting BGMM component count.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return int((self._model.weights_ > 0.01).sum())


# ---------------------------------------------------------------------------
# 2. GMM — Standard Gaussian Mixture
# ---------------------------------------------------------------------------


class GMMAnomalyDetector(_BaseUnsupervised):
    _config_cls = GMMConfig

    def _fit_normal(self, X: np.ndarray) -> None:
        c = self.config
        self._model = GaussianMixture(
            n_components=c.n_components,
            covariance_type=c.covariance_type,
            max_iter=c.max_iter,
            tol=c.tol,
            init_params=c.init_params,
            n_init=c.n_init,
            random_state=c.random_state,
        )
        self._model.fit(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. KMeans distance-from-centroid
# ---------------------------------------------------------------------------


class KMeansAnomalyDetector(_BaseUnsupervised):
    _config_cls = KMeansConfig

    def _fit_normal(self, X: np.ndarray) -> None:
        c = self.config
        self._model = KMeans(
            n_clusters=c.n_clusters,
            init=c.init,
            n_init=c.n_init,
            max_iter=c.max_iter,
            tol=c.tol,
            random_state=c.random_state,
        )
        self._model.fit(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        # Distance to nearest centroid
        dists = self._model.transform(X)
        return dists.min(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Isolation Forest
# ---------------------------------------------------------------------------


class IsolationForestDetector(_BaseUnsupervised):
    _config_cls = IForestConfig

    def _fit_normal(self, X: np.ndarray) -> None:
        c = self.config
        self._model = IsolationForest(
            n_estimators=c.n_estimators,
            max_samples=c.max_samples,
            contamination=c.contamination,
            max_features=c.max_features,
            bootstrap=c.bootstrap,
            random_state=c.random_state,
            n_jobs=c.n_jobs,
        )
        self._model.fit(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        # Higher score_samples = more normal; negate for "anomaly" score
        return -self._model.score_samples(X).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. One-Class SVM
# ---------------------------------------------------------------------------


class OneClassSVMDetector(_BaseUnsupervised):
    _config_cls = OCSVMConfig

    def _fit_normal(self, X: np.ndarray) -> None:
        c = self.config
        self._model = OneClassSVM(
            kernel=c.kernel,
            nu=c.nu,
            gamma=c.gamma,
            tol=c.tol,
            cache_size=c.cache_size,
        )
        self._model.fit(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X).astype(np.float32)


# ---------------------------------------------------------------------------
# 6. LOF novelty mode
# ---------------------------------------------------------------------------


class LOFAnomalyDetector(_BaseUnsupervised):
    _config_cls = LOFConfig

    def _fit_normal(self, X: np.ndarray) -> None:
        c = self.config
        self._model = LocalOutlierFactor(
            n_neighbors=min(c.n_neighbors, max(1, len(X) - 1)),
            algorithm=c.algorithm,
            metric=c.metric,
            contamination=c.contamination,
            novelty=c.novelty,
            n_jobs=c.n_jobs,
        )
        self._model.fit(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X).astype(np.float32)


# ---------------------------------------------------------------------------
# 7. Autoencoder anomaly detector (fully-specified architecture)
# ---------------------------------------------------------------------------


class _SymmetricAE(nn.Module):
    """Symmetric MLP autoencoder.  Architecture printed by repr() for paper tables."""

    def __init__(self, input_dim: int, cfg: AEConfig) -> None:
        super().__init__()
        act = getattr(nn, cfg.activation)
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in cfg.encoder_dims:
            enc_layers += [nn.Linear(prev, h), act()]
            if cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            prev = h
        enc_layers.append(nn.Linear(prev, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        prev = cfg.latent_dim
        for h in cfg.decoder_dims:
            dec_layers += [nn.Linear(prev, h), act()]
            if cfg.dropout > 0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderAnomalyDetector(_BaseUnsupervised):
    """Train on normal-only, score by per-sample reconstruction MSE."""

    _config_cls = AEConfig

    def __init__(self, config: AEConfig | None = None, device: torch.device | None = None) -> None:
        super().__init__(config)
        self.device = device or torch.device("cpu")
        self._model: _SymmetricAE | None = None

    def _fit_normal(self, X: np.ndarray) -> None:
        c = self.config
        torch.manual_seed(c.random_state)
        np.random.seed(c.random_state)

        self._model = _SymmetricAE(X.shape[1], c).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=c.lr, weight_decay=c.weight_decay)

        # Internal validation split (10%) for early stopping — uses only
        # normal training data, no anomalies leaked.
        n_val = max(int(0.1 * len(X)), 1)
        perm = np.random.permutation(len(X))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        X_tr, X_val = X[tr_idx], X[val_idx]

        best_val = float("inf")
        no_imp = 0
        best_state = None

        for _ in range(c.epochs):
            self._model.train()
            perm = np.random.permutation(len(X_tr))
            for start in range(0, len(X_tr), c.batch_size):
                idx = perm[start : start + c.batch_size]
                xb = torch.tensor(X_tr[idx], dtype=torch.float32, device=self.device)
                opt.zero_grad()
                ((self._model(xb) - xb) ** 2).mean().backward()
                opt.step()

            self._model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                val_loss = float(((self._model(xv) - xv) ** 2).mean().item())

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= c.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            recon = self._model(xt)
            err = ((recon - xt) ** 2).mean(dim=1).cpu().numpy()
        return err.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience runner — all unsupervised baselines, identical protocol
# ---------------------------------------------------------------------------


def run_unsupervised_suite(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    bgmm_cfg: BGMMConfig | None = None,
    gmm_cfg: GMMConfig | None = None,
    kmeans_cfg: KMeansConfig | None = None,
    iforest_cfg: IForestConfig | None = None,
    ocsvm_cfg: OCSVMConfig | None = None,
    lof_cfg: LOFConfig | None = None,
    ae_cfg: AEConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, dict]:
    """Fit every unsupervised baseline on training-set normals only,
    score the test set, and return {name: classification_metrics}."""
    from uais.utils.metrics import classification_metrics

    train_feat, train_mask, train_labels = features[train_idx], masks[train_idx], labels[train_idx]
    test_feat, test_mask, test_labels = features[test_idx], masks[test_idx], labels[test_idx]

    detectors = {
        "bgmm": BGMMAnomalyDetector(bgmm_cfg),
        "gmm": GMMAnomalyDetector(gmm_cfg),
        "kmeans": KMeansAnomalyDetector(kmeans_cfg),
        "isolation_forest": IsolationForestDetector(iforest_cfg),
        "one_class_svm": OneClassSVMDetector(ocsvm_cfg),
        "lof": LOFAnomalyDetector(lof_cfg),
        "autoencoder": AutoencoderAnomalyDetector(ae_cfg, device=device),
    }

    results: dict[str, dict] = {}
    for name, det in detectors.items():
        try:
            det.fit(train_feat, train_mask, train_labels)
            probs = det.predict_proba(test_feat, test_mask)
            metrics_d = classification_metrics(test_labels, probs)
            metrics_d["hyperparameters"] = det.get_hyperparameters()
            if hasattr(det, "get_effective_n_components"):
                metrics_d["effective_n_components"] = det.get_effective_n_components()
            results[name] = metrics_d
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}", "hyperparameters": det.get_hyperparameters()}

    return results


__all__ = [
    "BGMMConfig",
    "GMMConfig",
    "KMeansConfig",
    "IForestConfig",
    "OCSVMConfig",
    "LOFConfig",
    "AEConfig",
    "BGMMAnomalyDetector",
    "GMMAnomalyDetector",
    "KMeansAnomalyDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "LOFAnomalyDetector",
    "AutoencoderAnomalyDetector",
    "run_unsupervised_suite",
]
