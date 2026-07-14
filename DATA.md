# K-Bound Data and Artifact Contract

Raw benchmark datasets, phone video, and model checkpoints are not stored in
Git. The repository contains compact result records, protocol locks, hashes,
and code needed to audit or regenerate paper-facing artifacts.

## Canonical paper data

`docs/research/kbound/paper/generated/kbound_result_manifest.json` is the only
promoted-number index. It records metric definitions, policy order, seed counts,
source paths, quantile provenance, verdicts, and caveats.

The compact release includes the exact artifacts referenced by that manifest:

- CIFAR-10-C Tent/EATA locked five-seed aggregate;
- ImageNet-C SAR 27-cell paired bootstrap;
- Office-Home/iWildCam OOF lock records;
- Camelyon17 reconciliation records;
- RxRx1 locked summary in the manifest;
- constructed three-source OOF result;
- protocol-matched POEM/AETTA head-to-head record.

Missing raw tensors or images do not weaken artifact auditing, but they do mean
full model replay requires separately obtained benchmark data and checkpoints.

## Evaluation units

Controlled stress tracks score cross-fitted evaluation cells. Semantic natural
tracks fit on development data, calibrate on a separate split where available,
freeze all choices, and score held-out target conditions once. Target labels are
joined only after KGA decisions for offline regret and false-adapt evaluation.

The benefit convention is `Delta = R_T(f_0) - R_T(f_a)`. The canonical policy
order in saved regret arrays is KGA, always-adapt, always-freeze.

## Dataset status

| Track | Release status | Claim scope |
|---|---|---|
| CIFAR-10-C | compact five-seed aggregate | controlled CI beats-both for Tent/EATA |
| ImageNet-C | compact 27-cell seed-0 record | paired-bootstrap beats-both; single-seed caveat |
| Office-Home | OOF summary | natural no-harm |
| iWildCam | OOF summary | natural no-harm |
| Camelyon17 | reconciliation records | natural no-harm |
| RxRx1 | locked summary | natural no-harm |
| CIFAR-10.1 | summary only | diagnostic failure |
| ImageNet-R | three of four planned seeds | diagnostic/incomplete |
| PACS | one of three planned seeds | diagnostic/incomplete |

## Physical study

The locked `edge_real_phone_v1` plan requires 608 fresh physical clips across
S01-S10 and produces 736 evaluation windows after deterministic
batch-composition windows are derived. Raw clips remain local. Public release
artifacts may include privacy-reviewed manifests, SHA-256 hashes, split seals,
policy logs, metrics, and generated LaTeX tables.

Mock, browser-preview, pilot, reconstructed, or synthetic clips cannot satisfy
the publication gate. See
`docs/research/kbound/edge/PHYSICAL_STUDY_RUNBOOK.md`.

## External data placement

Dataset locations are configured by experiment scripts or environment-specific
paths and are intentionally excluded from Git. Never commit benchmark archives,
WILDS images, CIFAR-10-C arrays, ImageNet data, raw phone video, checkpoints, or
personal capture metadata.

The locked multiseed launcher discovers the following default T9 layouts. An
environment override may point to a different read-only location.

| Dataset | Default root | Required marker | Override |
|---|---|---|---|
| CIFAR-10/CIFAR-10-C | `/Volumes/T9/uav/data/cifar` | `CIFAR-10-C/labels.npy`, `resnet18_cifar.pt` | `KBOUND_CIFAR_ROOT` |
| ImageNet-C | `/Volumes/T9/uav/data/imagenet-c` | the three locked noise corruption directories | `KBOUND_IMAGENETC_ROOT` |
| PACS | `/Volumes/T9/uav/data/domainbed` | all four `PACS/` domains | `KBOUND_PACS_ROOT` |
| ImageNet-R | `/Volumes/T9/uav/data/imagenet-r` | ImageNet synset directories | `KBOUND_IMAGENETR_ROOT` |
| ImageNet class map | `/Volumes/T9/uav/data/imagenet_class_index.json` | JSON class-index file | `KBOUND_IMAGENET_CLASS_INDEX` |

The code repository itself need not live on T9. Run outputs are written under
`experiments/kbound/runs/multiseed_completion_v1/`, which is intentionally
ignored by Git until compact artifacts are audited and promoted. No launcher
command writes into a raw dataset directory.
