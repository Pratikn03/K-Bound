"""TensorFlow multi-sequence model for the 30-sequence behavior task.

The model accepts a dict input with keys ``seq_1`` ... ``seq_30``. Each entry is
shaped ``(batch, seq_len, n_features)``. A shared 1-D causal-conv stack encodes
every sub-sequence, the 30 latent vectors are concatenated, and a small MLP
emits the final logits.

Kept compact so the TensorFlow path stays import-checkable on CPU.
"""

from __future__ import annotations


def build_30_sequence_model(
    seq_len: int,
    n_features: int,
    latent_dim: int,
    num_outputs: int,
    num_sequences: int = 30,
    dropout: float = 0.1,
):
    """Construct a Keras model expecting ``{seq_1...seq_30: (batch, seq_len, n_features)}``.

    Parameters mirror ``MultiSequenceTCNClassifier``. The returned model can be
    compiled and called like any Keras model. The activation is left linear so
    callers can use ``from_logits=True`` cross-entropy.
    """
    import tensorflow as tf  # local import keeps the module CPU-importable

    keys = [f"seq_{i}" for i in range(1, num_sequences + 1)]
    inputs = {k: tf.keras.Input(shape=(seq_len, n_features), name=k) for k in keys}

    # Shared encoder applied to every sub-sequence.
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(
                filters=max(latent_dim, 2 * n_features),
                kernel_size=3,
                padding="causal",
                activation="relu",
            ),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv1D(
                filters=latent_dim,
                kernel_size=3,
                padding="causal",
                activation="relu",
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
        ],
        name="per_sequence_tcn",
    )

    latents = [encoder(inputs[k]) for k in keys]
    merged = tf.keras.layers.Concatenate(axis=-1)(latents)
    hidden = max(latent_dim, 2 * num_outputs)
    x = tf.keras.layers.Dense(hidden, activation="relu")(merged)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_outputs)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="multi_sequence_30_tcn_tf")


__all__ = ["build_30_sequence_model"]
