"""Trainer registry and thin adapters over existing ``train_*.py`` scripts.

This module wires the unified :class:`~uais.training.trainer.Trainer` lifecycle
to the project's pre-existing training entrypoints **without duplicating their
logic**. Each adapter is a small subclass that, inside its lifecycle methods,
lazily imports and calls the original function (e.g.
``uais.anomaly.train_isolation_forest.train_isolation_forest``).

Lazy, method-local imports are deliberate: heavyweight backends (TensorFlow,
PyTorch) are only imported when the corresponding trainer is actually *run*,
so importing :mod:`uais.training` -- and registering every adapter -- never
requires those optional dependencies.

Use the :func:`register` decorator to add a trainer and :func:`get_trainer`
to resolve one by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from uais.training.trainer import TrainConfig, Trainer

# Central registry mapping a string key to a Trainer subclass.
TRAINERS: dict[str, type[Trainer]] = {}


def register(*names: str) -> Callable[[type[Trainer]], type[Trainer]]:
    """Class decorator registering a :class:`Trainer` subclass under ``names``.

    Parameters
    ----------
    *names:
        One or more registry keys the decorated class should be reachable by.

    Returns
    -------
    Callable[[Type[Trainer]], Type[Trainer]]
        The decorator, which registers and then returns the class unchanged.

    Raises
    ------
    ValueError
        If no names are supplied or a name is already registered to a
        different class.
    """
    if not names:
        raise ValueError("register() requires at least one trainer name.")

    def _decorator(cls: type[Trainer]) -> type[Trainer]:
        for name in names:
            key = name.strip().lower()
            existing = TRAINERS.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(f"Trainer name {key!r} already registered to {existing.__name__}.")
            TRAINERS[key] = cls
        return cls

    return _decorator


def get_trainer(name: str, config: TrainConfig) -> Trainer:
    """Instantiate the registered trainer for ``name`` with ``config``.

    Parameters
    ----------
    name:
        Registry key (case-insensitive). Typically ``config.trainer_key``.
    config:
        The :class:`TrainConfig` passed to the trainer constructor.

    Returns
    -------
    Trainer
        A ready-to-run trainer instance.

    Raises
    ------
    KeyError
        If ``name`` is not present in :data:`TRAINERS`.
    """
    key = name.strip().lower()
    if key not in TRAINERS:
        available = ", ".join(sorted(TRAINERS)) or "<none>"
        raise KeyError(f"No trainer registered for {key!r}. Available: {available}.")
    return TRAINERS[key](config)


def available_trainers() -> list[str]:
    """Return the sorted list of registered trainer keys.

    Returns
    -------
    List[str]
        All keys currently present in :data:`TRAINERS`.
    """
    return sorted(TRAINERS)


# ---------------------------------------------------------------------------
# Helpers shared by adapters.
# ---------------------------------------------------------------------------


def _load_tabular(path: Path) -> pd.DataFrame:
    """Load a tabular dataset from a Parquet or CSV file.

    Parameters
    ----------
    path:
        Path to a ``.parquet``/``.pq`` or delimited text file.

    Returns
    -------
    pandas.DataFrame
        The loaded table.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found for training: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Anomaly-detection adapters (scikit-learn based, no torch/tf required).
# ---------------------------------------------------------------------------


@register("isolation_forest", "iforest")
class IsolationForestTrainer(Trainer):
    """Adapter for :func:`uais.anomaly.train_isolation_forest.train_isolation_forest`."""

    def build(self) -> None:
        """No-op build; the model is constructed inside :meth:`fit`."""
        return None

    def fit(self) -> Any:
        """Load data and fit an Isolation Forest, returning ``(model, scaler)``."""
        from uais.anomaly.train_isolation_forest import train_isolation_forest

        data_path = self.config.resolved_data_path()
        if data_path is None:
            raise ValueError("isolation_forest trainer requires 'data_path' in the config.")
        frame = _load_tabular(data_path)

        target = self.config.hyperparams.get("target")
        features = frame.drop(columns=[target]) if target and target in frame.columns else frame

        model, scaler = train_isolation_forest(
            features,
            random_state=self.config.seed,
            contamination=float(self.config.hyperparams.get("contamination", 0.01)),
        )
        self._scaler = scaler
        self._features = features
        return {"model": model, "scaler": scaler}

    def evaluate(self) -> dict[str, Any]:
        """Summarise anomaly scores over the training features."""
        from uais.anomaly.train_isolation_forest import compute_anomaly_score

        bundle = self._model or {}
        model = bundle.get("model")
        scaler = bundle.get("scaler")
        if model is None or scaler is None:
            return {"status": "not_fitted"}
        scores = compute_anomaly_score(model, scaler, self._features)
        return {
            "n_samples": int(len(scores)),
            "anomaly_score_mean": float(scores.mean()),
            "anomaly_score_std": float(scores.std()),
            "anomaly_rate_at_0_5": float((scores >= 0.5).mean()),
        }


class _PreprocessorAnomalyTrainer(Trainer):
    """Base adapter for LOF / OCSVM / autoencoder anomaly trainers.

    These existing entrypoints share the same signature
    ``(df, target, preprocessor, config, domain)`` and return
    ``(model, scores, y)``. Subclasses only declare which underlying function
    to call.
    """

    #: Subclasses set this to the underlying ``train_*`` callable name resolver.
    def _train_fn(self) -> Callable[..., Any]:  # pragma: no cover - overridden
        """Return the underlying training callable. Overridden by subclasses."""
        raise NotImplementedError

    def _build_preprocessor(self) -> Any:
        """Construct a numeric/categorical ColumnTransformer preprocessor.

        Returns
        -------
        sklearn.compose.ColumnTransformer
            A preprocessor imputing+scaling numerics and one-hot encoding
            low-cardinality categoricals.
        """
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric = self._features.select_dtypes(include="number").columns.tolist()
        categorical = [c for c in self._features.columns if c not in numeric]

        numeric_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # pragma: no cover - older sklearn
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric),
                ("cat", categorical_pipe, categorical),
            ],
            remainder="drop",
        )

    def build(self) -> None:
        """Load data and prepare features; defer model creation to :meth:`fit`."""
        data_path = self.config.resolved_data_path()
        if data_path is None:
            raise ValueError(f"{self.config.trainer_key} trainer requires 'data_path' in the config.")
        frame = _load_tabular(data_path)
        target = self.config.hyperparams.get("target", "label")
        if target not in frame.columns:
            # Synthesise an all-zero target so the unsupervised fit can proceed.
            frame[target] = 0
        self._frame = frame
        self._target = target
        self._features = frame.drop(columns=[target])
        return None

    def fit(self) -> Any:
        """Fit the underlying anomaly model and capture ``(scores, y)``."""
        train_fn = self._train_fn()
        preprocessor = self._build_preprocessor()
        domain = self.config.hyperparams.get("domain", self.config.name)
        config = {
            "seed": self.config.seed,
            "training": {"anomaly_contamination": float(self.config.hyperparams.get("contamination", 0.05))},
        }
        model, scores, y = train_fn(self._frame, self._target, preprocessor, config, domain)
        self._scores = scores
        self._y = y
        return model

    def evaluate(self) -> dict[str, Any]:
        """Summarise anomaly scores; add ROC-AUC when labels are informative."""
        scores = getattr(self, "_scores", None)
        if scores is None:
            return {"status": "not_fitted"}
        import numpy as np

        scores = np.asarray(scores, dtype=float)
        metrics: dict[str, Any] = {
            "n_samples": int(len(scores)),
            "anomaly_score_mean": float(np.mean(scores)),
            "anomaly_score_std": float(np.std(scores)),
        }
        y = getattr(self, "_y", None)
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim == 1 and len(np.unique(y_arr)) > 1:
                try:
                    from sklearn.metrics import roc_auc_score

                    metrics["roc_auc"] = float(roc_auc_score(y_arr, scores))
                except (ValueError, ImportError):  # pragma: no cover - degenerate labels
                    pass
        return metrics


@register("lof", "local_outlier_factor")
class LOFTrainer(_PreprocessorAnomalyTrainer):
    """Adapter for :func:`uais.anomaly.train_lof.train_lof`."""

    def _train_fn(self) -> Callable[..., Any]:
        """Resolve the LOF training callable lazily."""
        from uais.anomaly.train_lof import train_lof

        return train_lof


@register("ocsvm", "one_class_svm")
class OCSVMTrainer(_PreprocessorAnomalyTrainer):
    """Adapter for :func:`uais.anomaly.train_ocsvm.train_ocsvm`."""

    def _train_fn(self) -> Callable[..., Any]:
        """Resolve the One-Class SVM training callable lazily."""
        from uais.anomaly.train_ocsvm import train_ocsvm

        return train_ocsvm


@register("autoencoder", "sk_autoencoder")
class AutoencoderTrainer(_PreprocessorAnomalyTrainer):
    """Adapter for :func:`uais.anomaly.train_autoencoder.train_autoencoder`.

    Note: this targets the scikit-learn MLP autoencoder in
    ``uais.anomaly.train_autoencoder`` (no deep-learning backend required), not
    the Keras vision autoencoder.
    """

    def _train_fn(self) -> Callable[..., Any]:
        """Resolve the MLP autoencoder training callable lazily."""
        from uais.anomaly.train_autoencoder import train_autoencoder

        return train_autoencoder


# ---------------------------------------------------------------------------
# Fusion adapter (scikit-learn meta-model, no torch/tf required).
# ---------------------------------------------------------------------------


@register("fusion_meta", "fusion")
class FusionMetaTrainer(Trainer):
    """Adapter for :func:`uais.fusion.train_fusion_meta.train_fusion_model`.

    Delegates dataset assembly and meta-model fitting to the existing fusion
    pipeline, forwarding per-domain score paths from the config when provided.
    """

    def build(self) -> None:
        """No-op build; the fusion pipeline assembles its own dataset."""
        return None

    def fit(self) -> Any:
        """Run the fusion pipeline, returning the fitted meta-model."""
        from uais.fusion.train_fusion_meta import train_fusion_model

        score_paths = None
        raw_scores = self.config.hyperparams.get("scores")
        if isinstance(raw_scores, dict):
            score_paths = {domain: Path(p) for domain, p in raw_scores.items()}

        config = {
            "seed": self.config.seed,
            "data": {"test_size": float(self.config.hyperparams.get("test_size", 0.2))},
        }
        model, metrics = train_fusion_model(score_paths=score_paths, config=config)
        self._fusion_metrics = metrics
        return model

    def evaluate(self) -> dict[str, Any]:
        """Return the classification metrics produced by the fusion pipeline."""
        metrics = getattr(self, "_fusion_metrics", None)
        if metrics is None:
            return {"status": "not_fitted"}
        return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------------------
# Generative VAE adapter (TensorFlow/Keras -- imported only when run).
# ---------------------------------------------------------------------------


@register("vae", "tabular_vae")
class VAETrainer(Trainer):
    """Adapter for :func:`uais.generative.train_vae.run_vae_pipeline`.

    TensorFlow is imported lazily *inside* :meth:`fit`, so registering this
    trainer never requires TensorFlow to be installed.
    """

    def build(self) -> None:
        """Validate that a dataset path is configured; defer model build."""
        if self.config.resolved_data_path() is None:
            raise ValueError("vae trainer requires 'data_path' in the config.")
        return None

    def fit(self) -> Any:
        """Run the VAE training pipeline and capture its metrics."""
        from uais.generative.train_vae import VAEConfig, run_vae_pipeline

        data_path = self.config.resolved_data_path()
        hp = self.config.hyperparams
        cfg = VAEConfig(
            dataset_path=data_path,
            latent_dim=int(hp.get("latent_dim", 16)),
            epochs=int(hp.get("epochs", 20)),
            batch_size=int(hp.get("batch_size", 128)),
            test_size=float(hp.get("test_size", 0.2)),
            random_state=self.config.seed,
        )
        self._vae_metrics = run_vae_pipeline(cfg)
        # The pipeline returns metrics, not a serialisable model; nothing to
        # persist as an artifact here.
        return None

    def evaluate(self) -> dict[str, Any]:
        """Return scalar reconstruction-error metrics from the VAE run."""
        metrics = getattr(self, "_vae_metrics", None)
        if metrics is None:
            return {"status": "not_fitted"}
        return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


__all__ = [
    "TRAINERS",
    "register",
    "get_trainer",
    "available_trainers",
    "IsolationForestTrainer",
    "LOFTrainer",
    "OCSVMTrainer",
    "AutoencoderTrainer",
    "FusionMetaTrainer",
    "VAETrainer",
]
