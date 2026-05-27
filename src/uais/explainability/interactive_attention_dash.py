"""Interactive Streamlit dashboard for attention analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from uais.explainability.attention_visualizer import plot_attention_heatmap


def run_attention_dashboard(attention_weights, domain_labels: Sequence[str] | None = None) -> None:
    try:
        import streamlit as st  # type: ignore
    except Exception as exc:
        raise RuntimeError("Streamlit is required for the attention dashboard.") from exc

    weights = attention_weights
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().numpy()
    weights = np.asarray(weights)

    st.title("Attention Fusion Explorer")
    sample_idx = st.number_input("Sample index", min_value=0, max_value=max(0, weights.shape[0] - 1), value=0)
    head_idx = None
    if weights.ndim == 4:
        head_idx = st.number_input("Head index", min_value=0, max_value=max(0, weights.shape[1] - 1), value=0)
    fig = plot_attention_heatmap(weights, domain_labels=domain_labels, sample_idx=int(sample_idx), head_idx=head_idx)
    st.pyplot(fig)


__all__ = ["run_attention_dashboard"]
