# Running cifar_tent_mps_v2.py on ImageNet-C and with ResNet-50 / ViT-B

This note describes exactly how to extend the existing `cifar_tent_mps_v2.py`
DECISIVE benchmark to ImageNet-C and to larger architectures.

---

## 1. Running on ImageNet-C

ImageNet-C (`Hendrycks & Dietterich, 2019`) follows the same structure as
CIFAR-10-C: per-corruption `.npy` files at 5 severities, total 50,000 images
each.  The script already has `--benchmarks imagenetc` and `IMAGENET_C_QUICK`
defined; you only need to supply the paths.

### One-line command

```bash
source .venv/bin/activate   # or .venv_wilds for the wilds env

python docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks imagenetc \
    --imagenetc-root /path/to/ImageNet-C \
    --imagenet-val   /path/to/ILSVRC2012/val \
    --quick
```

`--imagenetc-root` must contain the standard ImageNet-C directory layout:
```
/path/to/ImageNet-C/
    gaussian_noise/
        1/  2/  3/  4/  5/    ← severity subdirs, each with 1000 class dirs
    defocus_blur/...
    snow/...
```

`--imagenet-val` is the plain ImageNet validation set (needed to build the
frozen f0 baseline and the balanced eval pool).  Download from
`https://www.image-net.org/` (academic licence).

`--quick` runs only `IMAGENET_C_QUICK = ["gaussian_noise","defocus_blur","snow",
"contrast","elastic_transform","jpeg_compression"]` to keep wall-time under 4h
on an M-series Mac.  Remove for the full 19-corruption suite.

### Downloading ImageNet-C

ImageNet-C (Zenodo record `2235448`) is distributed as **5 grouped tars**, not
per-corruption files:

| tar | size | corruptions |
|-----|------|-------------|
| `noise.tar`   | 21 GB | gaussian_noise, shot_noise, impulse_noise |
| `blur.tar`    | 7 GB  | defocus_blur, glass_blur, motion_blur, zoom_blur |
| `weather.tar` | 12 GB | snow, frost, fog, brightness |
| `digital.tar` | 7 GB  | contrast, elastic_transform, pixelate, jpeg_compression |
| `extra.tar`   | 15 GB | speckle_noise, spatter, gaussian_blur, saturate (optional) |

The 15-corruption standard set = noise+blur+weather+digital ≈ **47 GB**. Just use the
consolidated downloader (handles Camelyon17 + ImageNet-C + backbones, resumable):

```bash
bash docs/research/kbound/scripts/download_all_datasets.sh           # 15 corruptions
WITH_EXTRA=1 bash docs/research/kbound/scripts/download_all_datasets.sh   # +extra.tar
```

Or fetch ImageNet-C manually:

```bash
IC=experiments/kbound/data/imagenet-c; mkdir -p "$IC"; cd "$IC"
for t in noise blur weather digital; do
    wget -c "https://zenodo.org/records/2235448/files/${t}.tar?download=1" -O ${t}.tar
    tar xf ${t}.tar && rm ${t}.tar
done
# -> $IC/<corruption>/<severity 1-5>/<class>/*.JPEG
```

---

## 2. Using ResNet-50 as Backbone

ResNet-50 is drop-in compatible with the existing script.  The script currently
uses `WideResNet40` (loaded from `resnet18_cifar.pt`).

To add ResNet-50 support, find the `build_model` / `load_model` section in
`cifar_tent_mps_v2.py` and add:

```python
# In the model-loading block (search "resnet18_cifar" in the script):
import torchvision.models as tvm

if args.arch == "resnet50":
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
```

The Tent adaptation targets **BatchNorm affine** parameters (`weight` and `bias`
of every `nn.BatchNorm2d` layer) — ResNet-50 has BatchNorm after every conv
block, so `configure_tent()` works identically.

Suggested command:

```bash
python docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks imagenetc \
    --imagenetc-root /path/to/ImageNet-C \
    --imagenet-val   /path/to/val \
    --arch resnet50 --quick
```

---

## 3. Using ViT-B/16 as Backbone — IMPORTANT DIFFERENCE

ViT-B/16 (`torchvision.models.vit_b_16`) uses **LayerNorm**, not BatchNorm.
This is a **real, substantive difference** from ResNet: LayerNorm normalises
over the feature dimension per token, not over the batch, so the standard
BN-affine Tent target does not exist.

### What changes for ViT

Instead of targeting `nn.BatchNorm2d`, Tent on ViT targets the **LayerNorm
affine** parameters (`weight` and `bias` of every `nn.LayerNorm` module):

```python
def configure_tent_vit(model, lr=1e-3):
    """Tent for ViT: adapt LayerNorm affine params (NOT BatchNorm)."""
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            params += [m.weight, m.bias]
    assert len(params) > 0, "No LayerNorm found — wrong model?"
    return torch.optim.Adam(params, lr=lr)
```

Key implications for the paper:
- Tent on ViT can still minimise entropy, but the adaptation magnitude is
  different (LayerNorm has no running-mean to update; every batch it normalises
  online).  EATA and SAR are similarly applicable.
- The harmful/helpful regime balance may differ: ViT's internal representations
  are more robust to common corruptions, so the base rate of harmful adaptation
  is likely lower than for ResNet.  **Report separately; do not merge results.**
- The conformal radius ε may be smaller for ViT (lower variance in B across
  conditions) — test empirically.

### Suggested ViT command (after adding the `--arch vit_b16` flag stub):

```bash
python docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks imagenetc \
    --imagenetc-root /path/to/ImageNet-C \
    --imagenet-val   /path/to/val \
    --arch vit_b16 --quick
```

---

## 4. Adding `--arch` Flag to cifar_tent_mps_v2.py

The flag stub below can be added to the `argparse` block in `cifar_tent_mps_v2.py`:

```python
parser.add_argument("--arch", default="wideresnet40",
                    choices=["wideresnet40", "resnet50", "vit_b16"],
                    help="Backbone architecture. "
                         "wideresnet40: original CIFAR paper model. "
                         "resnet50: torchvision ImageNet-1K pretrain. "
                         "vit_b16: torchvision ViT-B/16 — adapts LayerNorm "
                         "instead of BatchNorm (see IMAGENETC_AND_ARCH.md).")
```

And in the model-loading block:

```python
def load_model(args, device):
    import torchvision.models as tvm
    if args.arch == "wideresnet40":
        # existing code — load resnet18_cifar.pt
        ...
    elif args.arch == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        configure_fn = configure_tent        # BN-affine
    elif args.arch == "vit_b16":
        model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        configure_fn = configure_tent_vit   # LayerNorm-affine ← REAL DIFFERENCE
    return model.to(device), configure_fn
```

---

## 5. Quick Sanity Check (no data download)

Syntax-validate all scripts before a full run:

```bash
python -c "import ast; ast.parse(open('docs/research/kbound/scripts/cifar_tent_mps_v2.py').read()); print('OK')"
python -c "import ast; ast.parse(open('docs/research/kbound/scripts/run_wilds_camelyon17.py').read()); print('OK')"
bash -n docs/research/kbound/scripts/run_wilds.sh && echo "shell OK"

# Smoke-test analysis core (no GPU):
python docs/research/kbound/scripts/run_wilds_camelyon17.py --smoke-test
python docs/research/kbound/scripts/cifar_tent_mps_v2.py --dry-run
```
