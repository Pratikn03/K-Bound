# Reproducibility Guide

This project supports deterministic training and evaluation for Phase 2 attention fusion.

## Seeds

- The default seed is set in `src/uais/fusion/attention/attention_config.yaml`.
- Evaluation harness uses the list in `evaluation.seeds`.

## Training

```bash
python3 -m uais.fusion.attention.train_attention_fusion
```

Artifacts:
- `models/fusion/attention/attention_fusion.pt`
- `experiments/fusion/attention_fusion/metrics.json`

## Evaluation

```bash
python3 -m uais.fusion.attention.evaluate_attention_harness
python3 -m scripts.generate_attention_reports
```

Artifacts:
- `experiments/fusion/attention_fusion/harness_metrics.json`
- `experiments/fusion/attention_fusion/attention_weights/`
- `experiments/fusion/attention_fusion/plots/`

## Notes

- Performance results can vary with different hardware and PyTorch versions.
- Keep dataset splits fixed when comparing ablations.
