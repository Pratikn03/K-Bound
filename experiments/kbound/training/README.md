# Raw-Data Training and Multi-Seed Evaluation

This directory is the only maintained raw-data experiment entry point in the
clean K-Bound repository. It has no runtime dependency on the historical
AutoML repository.

The deep-TTA, PACS, Camelyon helper, and ImageNet-R runners were migrated from
the last audited internal implementation on 2026-07-14. During migration:

- empirical residual radii were changed from interpolated NumPy quantiles to
  exact observed order statistics;
- stress-grid calibration was labeled cross-fitted empirical rather than exact
  split conformal;
- PACS calibration and serialization were separated by adapter;
- the outer scorer was replaced by one fit/calibration/test seed protocol used
  across every supported dataset;
- the superseded pooled-condition multiseed aggregator was removed so only one
  cross-dataset scoring path remains;
- outputs were moved to ignored immutable run directories, while only compact
  reviewed artifacts may be promoted into results.

The protocol lock is
research_lock/MULTISEED_COMPLETION_PROTOCOL_v1.json.

## Commands

From the repository root:

    bash docs/research/kbound/scripts/kbtrain.sh preflight
    bash docs/research/kbound/scripts/kbtrain.sh plan
    bash docs/research/kbound/scripts/kbtrain.sh status
    bash docs/research/kbound/scripts/kbtrain.sh run --yes
    bash docs/research/kbound/scripts/kbtrain.sh analyze

The default queue runs:

1. a clean five-seed CIFAR-10-C SAR tree;
2. ImageNet-C SAR seeds 1-4, joined with the imported immutable seed 0;
3. PACS seeds 0-2 with adapter-separated outputs;
4. ImageNet-R seed 3, joined with the committed seeds 0-2.

Optional queues:

    bash docs/research/kbound/scripts/kbtrain.sh plan --jobs imagenetc_vit_architecture
    bash docs/research/kbound/scripts/kbtrain.sh plan --jobs imagenetc_batch_sensitivity

Set dataset overrides only when T9 uses a different layout:

    export KBOUND_CIFAR_ROOT=/Volumes/T9/uav/data/cifar
    export KBOUND_IMAGENETC_ROOT=/Volumes/T9/uav/data/imagenet-c
    export KBOUND_PACS_ROOT=/Volumes/T9/uav/data/domainbed
    export KBOUND_IMAGENETR_ROOT=/Volumes/T9/uav/data/imagenet-r
    export KBOUND_IMAGENET_CLASS_INDEX=/Volumes/T9/uav/data/imagenet_class_index.json

The launcher fails before model loading when T9, a dataset marker, the research
Python stack, the class-index file, the requested accelerator, or local output
capacity is missing. Long runs also require the explicit --yes flag and an
exclusive accelerator lock. CPU use must be requested explicitly.

Run only selected tracks by passing space-separated or comma-separated names:

    bash docs/research/kbound/scripts/kbtrain.sh run \
      --jobs cifar10c_sar,imagenetc_sar --device mps --yes

The common analysis distinguishes three results: `beats_both_point`,
`beats_both_gain_ci`, and `beats_both_ci_robust`. The last requires both gain
intervals to exclude zero in KGA's favor and the hierarchical FA_u upper bound
to remain at or below alpha.

## Post-Run Work

Uniform analysis:

    bash docs/research/kbound/scripts/kbtrain.sh analyze

Calibration-size, batch, and architecture sensitivity:

    python experiments/kbound/training/ablate_multiseed.py --help

Held-out label-free iWildCam stream routing:

    python experiments/kbound/training/iwildcam_label_free_stream.py --help

Full candidate/controller timing:

    python experiments/kbound/training/end_to_end_runtime.py --help

No new run changes the canonical paper manifest automatically. Promotion
requires a separate artifact audit, protocol-hash check, and manuscript review.
