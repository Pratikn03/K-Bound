"""Convolutional autoencoder for unsupervised visual anomaly detection.

Anomaly score = per-image MSE reconstruction error.  A percentile-based
threshold computed on the training set flags unseen images as anomalous when
their reconstruction error exceeds that threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class VisionAEConfig:
    data_dir: str
    image_size: int = 128
    latent_dim: int = 64
    batch_size: int = 32
    epochs: int = 20
    validation_split: float = 0.15
    # Percentile of training reconstruction errors used as anomaly threshold
    anomaly_percentile: float = 95.0
    learning_rate: float = 1e-3
    seed: int = 42


def _build_autoencoder(image_size: int, latent_dim: int, learning_rate: float) -> tf.keras.Model:
    """Symmetric conv-encoder → dense bottleneck → conv-decoder."""
    n_strides = 3
    bottleneck_spatial = image_size // (2**n_strides)  # spatial dim after encoding
    bottleneck_channels = 128

    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    # Encoder: downsample 3x with strided convolutions
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(bottleneck_channels, 3, strides=2, padding="same", activation="relu")(x)

    # Dense bottleneck
    flat_dim = bottleneck_spatial * bottleneck_spatial * bottleneck_channels
    x = tf.keras.layers.Flatten()(x)
    encoded = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder: project back then upsample symmetrically
    x = tf.keras.layers.Dense(flat_dim, activation="relu")(encoded)
    x = tf.keras.layers.Reshape((bottleneck_spatial, bottleneck_spatial, bottleneck_channels))(x)
    x = tf.keras.layers.Conv2DTranspose(bottleneck_channels, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    # Output in [0, 1] to match Rescaling normalisation
    decoded = tf.keras.layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid", name="reconstruction")(x)

    model = tf.keras.Model(inputs, decoded, name="conv_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def _iter_images(directory: Path) -> bool:
    """Return True if the directory has direct image children."""
    for ext in IMAGE_EXTENSIONS:
        if next(directory.glob(f"*{ext}"), None) is not None:
            return True
        if next(directory.glob(f"*{ext.upper()}"), None) is not None:
            return True
    return False


def _find_dataset_root(directory: Path) -> Path:
    """Handle Kaggle-style double-nested directories gracefully."""
    if _iter_images(directory) or not directory.exists():
        return directory
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        candidate = subdirs[0]
        grandkids = [d for d in candidate.iterdir() if d.is_dir()]
        if grandkids:
            return candidate
    return directory


def _reconstruction_errors(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """Compute per-image MSE between input and reconstruction."""
    errors = []
    for batch_x, _ in dataset:
        recon = model(batch_x, training=False)
        # Normalise inputs the same way as the Rescaling layer
        x_norm = tf.cast(batch_x, tf.float32) / 255.0
        mse = tf.reduce_mean(tf.square(x_norm - recon), axis=[1, 2, 3])
        errors.append(mse.numpy())
    return np.concatenate(errors)


def train_autoencoder_vision(
    data_dir: str,
    image_size: int = 128,
    latent_dim: int = 64,
    batch_size: int = 32,
    epochs: int = 20,
    validation_split: float = 0.15,
    anomaly_percentile: float = 95.0,
    learning_rate: float = 1e-3,
    seed: int = 42,
    **_: object,
) -> dict[str, float]:
    """Train a convolutional autoencoder and return reconstruction-based anomaly metrics.

    All images in data_dir are used for unsupervised training. Anomaly threshold
    is set at ``anomaly_percentile`` of training-set reconstruction errors, and
    the anomaly rate on the validation split is reported.
    """
    cfg = VisionAEConfig(
        data_dir=data_dir,
        image_size=image_size,
        latent_dim=latent_dim,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        anomaly_percentile=anomaly_percentile,
        learning_rate=learning_rate,
        seed=seed,
    )

    root = _find_dataset_root(Path(cfg.data_dir))
    if not root.exists():
        raise FileNotFoundError(f"Vision data directory not found: {root}")

    load_kwargs = {
        "directory": root,
        "image_size": (cfg.image_size, cfg.image_size),
        "batch_size": cfg.batch_size,
        "validation_split": cfg.validation_split,
        "seed": cfg.seed,
    }
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(subset="training", **load_kwargs)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(subset="validation", **load_kwargs)

    # Autoencoder targets are the inputs themselves
    ae_train = train_ds.map(lambda x, _: (x, x), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    ae_val = val_ds.map(lambda x, _: (x, x), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    model = _build_autoencoder(cfg.image_size, cfg.latent_dim, cfg.learning_rate)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    history = model.fit(ae_train, validation_data=ae_val, epochs=cfg.epochs, callbacks=callbacks, verbose=1)

    # Compute reconstruction errors on training set to derive anomaly threshold
    train_errors = _reconstruction_errors(model, train_ds)
    threshold = float(np.percentile(train_errors, cfg.anomaly_percentile))

    # Evaluate anomaly rate on validation set
    val_errors = _reconstruction_errors(model, val_ds)
    anomaly_rate = float(np.mean(val_errors > threshold))

    best_val_loss = float(min(history.history.get("val_loss", [0.0])))

    return {
        "reconstruction_loss": best_val_loss,
        "anomaly_threshold": threshold,
        "train_recon_error_mean": float(np.mean(train_errors)),
        "train_recon_error_std": float(np.std(train_errors)),
        "val_anomaly_rate": anomaly_rate,
        "epochs_trained": len(history.history.get("loss", [])),
    }


__all__ = ["VisionAEConfig", "train_autoencoder_vision"]
