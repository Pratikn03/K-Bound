"""Explainability helpers (SHAP, LIME, sequence)."""

from uais.explainability.attention_visualizer import plot_attention_heatmap
from uais.explainability.domain_contribution import summarize_domain_contributions
from uais.explainability.interactive_attention_dash import run_attention_dashboard

__all__ = [
    "plot_attention_heatmap",
    "summarize_domain_contributions",
    "run_attention_dashboard",
]
