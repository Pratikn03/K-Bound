"""Generate attention fusion report plots from harness outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uais.explainability.attention_visualizer import (
    plot_attention_heatmap,
    plot_domain_contributions,
    summarize_attention_weights,
)
from uais.utils.config_loader import load_yaml
from uais.utils.paths import PROJECT_ROOT

DEFAULT_CONFIG = Path("src/uais/fusion/attention/attention_config.yaml")


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_harness_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _plot_from_attention_weights(weights_path: Path, out_dir: Path) -> None:
    data = np.load(weights_path, allow_pickle=True)
    weights = data["attention_weights"]
    domain_order = data.get("domain_order")
    domain_labels = domain_order.tolist() if domain_order is not None else None

    if weights.ndim == 4:
        avg_weights = weights.mean(axis=0).mean(axis=0)
    elif weights.ndim == 3:
        avg_weights = weights.mean(axis=0)
    else:
        avg_weights = weights

    plot_attention_heatmap(
        avg_weights,
        domain_labels=domain_labels,
        out_path=out_dir / "attention_heatmap.png",
    )
    contributions = summarize_attention_weights(avg_weights, domain_labels=domain_labels)
    plot_domain_contributions(
        contributions,
        out_path=out_dir / "domain_contributions.png",
    )


def _plot_from_harness_summary(harness: dict, out_dir: Path) -> None:
    if not harness:
        return
    summary = harness.get("baseline", {}).get("summary", {})
    interpretability = summary.get("interpretability", {})
    attention_mean = interpretability.get("attention_mean")
    if not attention_mean:
        return
    contributions = {k: v.get("mean") for k, v in attention_mean.items() if isinstance(v, dict)}
    plot_domain_contributions(
        contributions,
        out_path=out_dir / "domain_contributions.png",
    )


def generate_attention_reports(cfg_path: Path = DEFAULT_CONFIG) -> None:
    cfg = load_yaml(cfg_path)
    eval_cfg = cfg.get("evaluation", {})
    output_dir = _resolve_path(eval_cfg.get("plots_dir", "experiments/fusion/attention_fusion/plots"))
    output_dir.mkdir(parents=True, exist_ok=True)

    attention_dir = _resolve_path(
        eval_cfg.get("attention_weights_dir", "experiments/fusion/attention_fusion/attention_weights")
    )
    weight_files = sorted(attention_dir.glob("attention_weights_seed_*.npz"))
    if weight_files:
        _plot_from_attention_weights(weight_files[0], output_dir)

    harness_path = _resolve_path(
        eval_cfg.get("metrics_path", "experiments/fusion/attention_fusion/harness_metrics.json")
    )
    harness = _load_harness_summary(harness_path)
    _plot_from_harness_summary(harness, output_dir)

    print(f"[ok] Attention fusion report plots saved to {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    generate_attention_reports()
