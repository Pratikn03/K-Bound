"""Strong multimodal fusion baselines for comparison against CRAF/RGA.

Six baselines covering static fusion and test-time adaptive fusion:

  EarlyFusionMLP          — concat all domain features + missing indicators → MLP
  LateFusionEnsemble      — per-domain logistic regressor → stacked meta-learner
  RandomForestFusion      — sklearn RF on flattened features + missing indicators
  ConfidenceWeightedMean  — sharpness-weighted score mean (CRAF's direct predecessor)
  TentAdapter             — test-time entropy minimization on BN params (Wang et al. 2021)
  PseudoLabelTTTAdapter   — pseudo-label test-time training (Sun et al. 2020)

All models expose a consistent `fit(features, masks, labels)` / `predict_proba(features, masks)`
interface so the experiment script can treat them uniformly.

Missing-domain handling strategy:
  - EarlyFusionMLP, RandomForest, TTT variants: fill masked features with 0.5 and append
    a [D]-dim binary missingness indicator vector.
  - LateFusionEnsemble: per-domain model outputs 0.5 when the domain is absent.
  - ConfidenceWeightedMean: excluded from weighted sum; result falls back to 0.5 if all absent.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from uais.utils.metrics import classification_metrics


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _flatten_with_mask(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """[N, D, F] + [N, D] bool → [N, D*F + D] float.

    Masked entries are set to 0.5 (uninformative midpoint for score-like features).
    The missingness indicator appended at the end lets models learn a bias for absence.
    """
    n, d, f = features.shape
    flat = features.copy().astype(np.float32)
    for di in range(d):
        flat[:, di, :][masks[:, di]] = 0.5
    flat = flat.reshape(n, d * f)
    indicators = masks.astype(np.float32)  # 1.0 = domain was missing
    return np.concatenate([flat, indicators], axis=1)


# ---------------------------------------------------------------------------
# 1. EarlyFusionMLP
# ---------------------------------------------------------------------------

class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EarlyFusionMLP:
    """Concatenate all domain features + missingness indicators, then train a 2-hidden-layer MLP.

    This is the standard early-fusion baseline in multimodal learning papers
    (Baltrusaitis et al., 2018; Ramachandram & Taylor, 2017).
    """

    def __init__(
        self,
        hidden_dims: List[int] = (128, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        patience: int = 8,
        batch_size: int = 128,
        device: Optional[torch.device] = None,
        random_seed: int = 42,
    ) -> None:
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.random_seed = random_seed
        self._model: Optional[_MLPNet] = None
        self._scaler = StandardScaler()

    def fit(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_masks: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> "EarlyFusionMLP":
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        X = self._scaler.fit_transform(_flatten_with_mask(features, masks))
        y = labels.astype(np.float32)

        pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

        if val_features is not None:
            X_val = self._scaler.transform(_flatten_with_mask(val_features, val_masks))
            y_val = val_labels.astype(np.float32)
        else:
            # Simple 10% internal validation split
            n_val = max(int(0.1 * len(X)), 1)
            perm = np.random.permutation(len(X))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            X_val, y_val = X[val_idx], y[val_idx]
            X, y = X[train_idx], y[train_idx]

        self._model = _MLPNet(X.shape[1], self.hidden_dims, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))

        best_val_loss = float("inf")
        no_improve = 0
        best_state = None

        for _ in range(self.epochs):
            self._model.train()
            perm = np.random.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = torch.tensor(X[idx], dtype=torch.float32, device=self.device)
                yb = torch.tensor(y[idx], dtype=torch.float32, device=self.device)
                optimizer.zero_grad()
                criterion(self._model(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            self._model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                yv = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                val_loss = criterion(self._model(xv), yv).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    def predict_proba(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X = self._scaler.transform(_flatten_with_mask(features, masks))
        probs = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                xb = torch.tensor(X[start:start + 256], dtype=torch.float32, device=self.device)
                probs.append(torch.sigmoid(self._model(xb)).cpu().numpy())
        return np.concatenate(probs)


# ---------------------------------------------------------------------------
# 2. LateFusionEnsemble (stacked)
# ---------------------------------------------------------------------------

class LateFusionEnsemble:
    """Stacked late fusion: per-domain logistic regressors + learned meta-combiner.

    This mirrors the standard late-fusion approach in multimodal anomaly detection
    (e.g., Zhao et al., ICDM 2020; Wang et al., 2021).  Missing domains contribute
    0.5 (uninformative) to the meta-learner input.
    """

    def __init__(
        self,
        score_index: int = 0,
        min_samples_per_domain: int = 10,
        meta_C: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        self.score_index = score_index
        self.min_samples_per_domain = min_samples_per_domain
        self.meta_C = meta_C
        self.random_seed = random_seed
        self._domain_models: Dict[int, LogisticRegression] = {}
        self._domain_scalers: Dict[int, StandardScaler] = {}
        self._meta: Optional[LogisticRegression] = None
        self._n_domains: int = 0

    def fit(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
    ) -> "LateFusionEnsemble":
        n, d, f = features.shape
        self._n_domains = d

        # Train per-domain first-level models
        domain_preds = np.full((n, d), 0.5, dtype=np.float32)
        for di in range(d):
            available = ~masks[:, di]
            if available.sum() < self.min_samples_per_domain:
                continue
            X_d = features[available, di, :]
            y_d = labels[available]
            if len(np.unique(y_d)) < 2:
                continue
            scaler = StandardScaler()
            X_d_s = scaler.fit_transform(X_d)
            lr = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=500,
                random_state=self.random_seed,
            )
            lr.fit(X_d_s, y_d)
            preds = lr.predict_proba(X_d_s)[:, 1].astype(np.float32)
            domain_preds[available, di] = preds
            self._domain_models[di] = lr
            self._domain_scalers[di] = scaler

        # Train meta-learner on domain predictions
        self._meta = LogisticRegression(
            C=self.meta_C, class_weight="balanced", max_iter=500,
            random_state=self.random_seed,
        )
        self._meta.fit(domain_preds, labels)
        return self

    def predict_proba(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if self._meta is None:
            raise RuntimeError("Call fit() first.")
        n, d, _ = features.shape
        domain_preds = np.full((n, d), 0.5, dtype=np.float32)
        for di, lr in self._domain_models.items():
            available = ~masks[:, di]
            if not available.any():
                continue
            X_d = self._domain_scalers[di].transform(features[available, di, :])
            domain_preds[available, di] = lr.predict_proba(X_d)[:, 1].astype(np.float32)
        return self._meta.predict_proba(domain_preds)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# 3. RandomForestFusion
# ---------------------------------------------------------------------------

class RandomForestFusion:
    """Random forest on flattened features + missingness indicators.

    Ensembling over many decision trees tends to be the toughest tabular
    baseline to beat, making it an essential comparison point.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_features: str = "sqrt",
        min_samples_leaf: int = 5,
        random_seed: int = 42,
    ) -> None:
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed,
        )
        self._scaler = StandardScaler()

    def fit(self, features: np.ndarray, masks: np.ndarray, labels: np.ndarray) -> "RandomForestFusion":
        X = self._scaler.fit_transform(_flatten_with_mask(features, masks))
        self.rf.fit(X, labels)
        return self

    def predict_proba(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        X = self._scaler.transform(_flatten_with_mask(features, masks))
        return self.rf.predict_proba(X)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# 4. ConfidenceWeightedMean (CRAF predecessor / ablation)
# ---------------------------------------------------------------------------

class ConfidenceWeightedMean:
    """Sharpness-weighted score mean: w_d = 2 * |score_d - 0.5|.

    This is the direct predecessor to CRAF's TTRA — it applies the same
    distance-from-0.5 heuristic that the original AttentionFusionModel uses
    internally, but without any calibration correction or drift detection.
    Including it as a baseline isolates the contribution of CRAF's CRS over
    this simpler static heuristic.

    Parameter-free — no fitting required (fit() is a no-op).
    """

    def __init__(self, score_index: int = 0) -> None:
        self.score_index = score_index

    def fit(self, features: np.ndarray, masks: np.ndarray, labels: np.ndarray) -> "ConfidenceWeightedMean":
        return self  # no-op

    def predict_proba(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        n, d, _ = features.shape
        scores = features[:, :, self.score_index].copy()  # [N, D]
        weights = 2.0 * np.abs(scores - 0.5)             # sharpness weight
        weights[masks] = 0.0                              # zero out missing
        scores[masks] = 0.0

        total_weight = weights.sum(axis=1, keepdims=True)
        safe_weight = np.where(total_weight > 1e-9, total_weight, 1.0)
        weighted_scores = (scores * weights).sum(axis=1) / safe_weight.squeeze(1)

        # Fall back to 0.5 where all domains are absent
        all_missing = masks.all(axis=1)
        weighted_scores[all_missing] = 0.5
        return weighted_scores.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. TentAdapter  (test-time entropy minimisation, Wang et al. NeurIPS 2020)
# ---------------------------------------------------------------------------

class TentAdapter:
    """Adapt a fitted EarlyFusionMLP at test time by minimising prediction entropy.

    Only the BatchNorm affine parameters (weight, bias) are updated — all other
    weights stay frozen.  This is the "Tent" approach: the model is moved back
    to its fitted state before every predict_proba() call so adaptation is
    stateless across batches.

    Reference: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
    Minimization", ICLR 2021.
    """

    def __init__(
        self,
        base: Optional[EarlyFusionMLP] = None,
        n_steps: int = 1,
        lr: float = 1e-3,
    ) -> None:
        self.base = base if base is not None else EarlyFusionMLP()
        self.n_steps = n_steps
        self.lr = lr
        self._base_state: Optional[Dict] = None

    def fit(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
    ) -> "TentAdapter":
        self.base.fit(features, masks, labels)
        self._base_state = {k: v.clone() for k, v in self.base._model.state_dict().items()}
        return self

    def predict_proba(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if self._base_state is None:
            raise RuntimeError("Call fit() first.")
        # Reset to fitted state — adaptation is per-call, not accumulated
        self.base._model.load_state_dict(self._base_state)

        # Collect BN affine parameters; set BN layers to train mode
        bn_params: List[torch.nn.Parameter] = []
        for module in self.base._model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train()
                bn_params.extend(p for p in module.parameters() if p.requires_grad)
        # All other layers: eval + no grad
        for name, module in self.base._model.named_modules():
            if not isinstance(module, nn.BatchNorm1d):
                module.eval()
        for p in self.base._model.parameters():
            p.requires_grad_(False)
        for p in bn_params:
            p.requires_grad_(True)

        if bn_params:
            opt = torch.optim.Adam(bn_params, lr=self.lr)
            X = self.base._scaler.transform(_flatten_with_mask(features, masks))
            xt = torch.tensor(X, dtype=torch.float32, device=self.base.device)
            for _ in range(self.n_steps):
                opt.zero_grad()
                logits = self.base._model(xt)
                p_hat = torch.sigmoid(logits)
                eps = 1e-7
                entropy = -(
                    p_hat * (p_hat + eps).log()
                    + (1.0 - p_hat) * (1.0 - p_hat + eps).log()
                )
                entropy.mean().backward()
                opt.step()

        self.base._model.eval()
        for p in self.base._model.parameters():
            p.requires_grad_(True)

        X = self.base._scaler.transform(_flatten_with_mask(features, masks))
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                xb = torch.tensor(X[start:start + 256], dtype=torch.float32, device=self.base.device)
                probs.append(torch.sigmoid(self.base._model(xb)).cpu().numpy())
        return np.concatenate(probs)


# ---------------------------------------------------------------------------
# 6. PseudoLabelTTTAdapter  (test-time training with pseudo-labels)
# ---------------------------------------------------------------------------

class PseudoLabelTTTAdapter:
    """Test-Time Training via self-generated pseudo-labels.

    After fit(), adapts the MLP on each test batch by:
    1. Computing initial predictions with the fitted model.
    2. Selecting high-confidence predictions (p ≥ threshold or p ≤ 1-threshold)
       as pseudo-labelled samples.
    3. Running n_steps gradient updates on the whole model using pseudo-label BCE.
    4. Returning predictions from the adapted model.

    The model is reset to its fitted state before every predict_proba() call
    so adaptation is stateless across batches.

    Reference: Sun et al., "Test-Time Training with Self-Supervision for
    Generalization under Distribution Shifts", ICML 2020.
    """

    def __init__(
        self,
        base: Optional[EarlyFusionMLP] = None,
        confidence_threshold: float = 0.85,
        n_steps: int = 3,
        lr: float = 5e-4,
    ) -> None:
        self.base = base if base is not None else EarlyFusionMLP()
        self.confidence_threshold = confidence_threshold
        self.n_steps = n_steps
        self.lr = lr
        self._base_state: Optional[Dict] = None

    def fit(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
    ) -> "PseudoLabelTTTAdapter":
        self.base.fit(features, masks, labels)
        self._base_state = {k: v.clone() for k, v in self.base._model.state_dict().items()}
        return self

    def predict_proba(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if self._base_state is None:
            raise RuntimeError("Call fit() first.")
        self.base._model.load_state_dict(self._base_state)

        X = self.base._scaler.transform(_flatten_with_mask(features, masks))
        xt = torch.tensor(X, dtype=torch.float32, device=self.base.device)

        # Initial predictions → pseudo-labels
        self.base._model.eval()
        with torch.no_grad():
            init_probs = torch.sigmoid(self.base._model(xt)).cpu().numpy()

        pos_mask = init_probs >= self.confidence_threshold
        neg_mask = init_probs <= (1.0 - self.confidence_threshold)
        pseudo_mask = pos_mask | neg_mask

        if pseudo_mask.sum() >= 2:
            pseudo_X = xt[pseudo_mask]
            pseudo_y = torch.tensor(
                (init_probs[pseudo_mask] >= self.confidence_threshold).astype(np.float32),
                device=self.base.device,
            )
            optimizer = torch.optim.Adam(self.base._model.parameters(), lr=self.lr)
            criterion = nn.BCEWithLogitsLoss()
            torch.manual_seed(0)   # deterministic dropout across calls
            self.base._model.train()
            for _ in range(self.n_steps):
                optimizer.zero_grad()
                criterion(self.base._model(pseudo_X), pseudo_y).backward()
                nn.utils.clip_grad_norm_(self.base._model.parameters(), 1.0)
                optimizer.step()

        self.base._model.eval()
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                xb = xt[start:start + 256]
                probs.append(torch.sigmoid(self.base._model(xb)).cpu().numpy())
        return np.concatenate(probs)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_baseline_suite(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    score_index: int = 0,
    device: Optional[torch.device] = None,
    random_seed: int = 42,
) -> Dict[str, Dict]:
    """Fit all four baselines and return classification metrics for each.

    Returns a dict keyed by baseline name, each value being a classification_metrics dict.
    All baselines receive the same train/val/test split for a fair comparison.
    """
    device = device or torch.device("cpu")

    train_feat, train_mask, train_labels = features[train_idx], masks[train_idx], labels[train_idx]
    val_feat, val_mask, val_labels = features[val_idx], masks[val_idx], labels[val_idx]
    test_feat, test_mask, test_labels = features[test_idx], masks[test_idx], labels[test_idx]

    results: Dict[str, Dict] = {}

    # 1. Early Fusion MLP
    mlp = EarlyFusionMLP(device=device, random_seed=random_seed)
    mlp.fit(train_feat, train_mask, train_labels, val_feat, val_mask, val_labels)
    results["early_fusion_mlp"] = classification_metrics(test_labels, mlp.predict_proba(test_feat, test_mask))

    # 2. Late Fusion Ensemble
    lfe = LateFusionEnsemble(score_index=score_index, random_seed=random_seed)
    # Combine train+val for domain models (standard practice for stacked ensembles)
    all_idx_for_lfe = np.concatenate([train_idx, val_idx])
    lfe.fit(features[all_idx_for_lfe], masks[all_idx_for_lfe], labels[all_idx_for_lfe])
    results["late_fusion_ensemble"] = classification_metrics(test_labels, lfe.predict_proba(test_feat, test_mask))

    # 3. Random Forest
    rf = RandomForestFusion(random_seed=random_seed)
    rf.fit(train_feat, train_mask, train_labels)
    results["random_forest"] = classification_metrics(test_labels, rf.predict_proba(test_feat, test_mask))

    # 4. Confidence-Weighted Mean (parameter-free)
    cwm = ConfidenceWeightedMean(score_index=score_index)
    cwm.fit(train_feat, train_mask, train_labels)
    results["confidence_weighted_mean"] = classification_metrics(test_labels, cwm.predict_proba(test_feat, test_mask))

    # 5. Tent (test-time entropy minimisation) — wraps a fresh EarlyFusionMLP
    tent = TentAdapter(base=EarlyFusionMLP(device=device, random_seed=random_seed))
    tent.fit(train_feat, train_mask, train_labels)
    results["tent_ttt"] = classification_metrics(test_labels, tent.predict_proba(test_feat, test_mask))

    # 6. Pseudo-Label TTT — wraps a fresh EarlyFusionMLP
    plt_adapter = PseudoLabelTTTAdapter(
        base=EarlyFusionMLP(device=device, random_seed=random_seed)
    )
    plt_adapter.fit(train_feat, train_mask, train_labels)
    results["pseudo_label_ttt"] = classification_metrics(test_labels, plt_adapter.predict_proba(test_feat, test_mask))

    return results


__all__ = [
    "EarlyFusionMLP",
    "LateFusionEnsemble",
    "RandomForestFusion",
    "ConfidenceWeightedMean",
    "TentAdapter",
    "PseudoLabelTTTAdapter",
    "run_baseline_suite",
]
