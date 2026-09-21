"""Strict offline AdaContrast Clipart-2020 source checkpoint preparation.

No download, image read, optimizer restoration, adaptation or scoring. The
generic byte/state helpers do not authenticate an origin; only load_clipart2020
binds the reviewed author release. CPU-only single-worker preparation: callers
must not concurrently mutate Torch global dtype/device/RNG configuration.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import re
import stat

from task3_image_panel import _open_directory_chain

RELEASE_SHA256 = '0b8cb099b93890ff9b3fdd993368846ea506c89f81fc33b670959e04ab16f8de'
RELEASE_BYTES = 96586871
CLASS_MAP_SHA256 = '5a1a4ba6b7f738a69c2b83b405f54f65cee5404920fa382d40d89cd08afd57fd'
REFERENCE_COMMIT = 'c3c8b880131f2658d6fd0d5ed14d71f326174d57'


def read_verified_checkpoint(path, expected_sha256, expected_bytes):
    """Authenticate one descriptor-bound buffer; never reopen it to deserialize."""
    if (not isinstance(expected_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None
            or type(expected_bytes) is not int or not 0 < expected_bytes <= 128 * 1024**2):
        raise ValueError('invalid bounded checkpoint identity')
    path = Path(os.path.abspath(path))
    parent = _open_directory_chain(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, 'rb') as handle:
            before = os.fstat(handle.fileno())
            if (not stat.S_ISREG(before.st_mode)
                    or getattr(before, 'st_flags', 0) & 0x40000000):
                raise ValueError('checkpoint must be a materialized regular file')
            if before.st_size != expected_bytes:
                raise ValueError('checkpoint size mismatch')
            data = handle.read(expected_bytes + 1)
            after = os.fstat(handle.fileno())
            fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
            if len(data) != expected_bytes or any(getattr(before, k) != getattr(after, k) for k in fields):
                raise ValueError('checkpoint changed during read')
    finally:
        os.close(parent)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('checkpoint digest mismatch')
    return data


@contextmanager
def _cpu_preparation():
    """Include first-time imports in the CPU/FP32 and RNG isolation boundary."""
    import torch
    old_dtype = torch.get_default_dtype()
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        try:
            torch.set_default_dtype(torch.float32)
            yield
        finally:
            torch.set_default_dtype(old_dtype)


def build_reference_classifier():
    """Native architecture without an incidental ImageNet weight download."""
    import torch

    class ReferenceClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            backbone = torchvision.models.resnet50(weights=None)
            backbone.fc = torch.nn.Linear(2048, 256)
            self.encoder = torch.nn.Sequential(backbone, torch.nn.BatchNorm1d(256))
            self.fc = torch.nn.utils.weight_norm(torch.nn.Linear(256, 126), dim=0)

        def forward(self, x, return_feats=False):
            features = torch.flatten(self.encoder(x), 1)
            logits = self.fc(features)
            return (features, logits) if return_feats else logits

        @property
        def output_dim(self):
            return self.encoder[0].fc.out_features

        @property
        def num_classes(self):
            return self.fc.out_features

        def get_params(self):
            """Native bottleneck recipe: backbone 1x, bottleneck/head 10x LR."""
            backbone = [p for child in list(self.encoder[0].children())[:-1]
                        for p in child.parameters() if p.requires_grad]
            extra = [p for module in (self.encoder[0].fc, self.encoder[1], self.fc)
                     for p in module.parameters() if p.requires_grad]
            return backbone, extra

    with _cpu_preparation():
        import torchvision.models
        model = ReferenceClassifier().eval()
    return model


def apply_strict_state(model, authenticated_bytes):
    """Load a previously authenticated buffer; return no origin/official claim.

    weights_only constrains deserialization but may deserialize the native
    checkpoint's optimizer data. No optimizer is constructed/restored/stepped.
    All declared state validation completes before copying any model parameter.
    """
    import torch
    from torch.nn.utils.weight_norm import WeightNorm

    checkpoint = torch.load(io.BytesIO(authenticated_bytes), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state_dict'), dict):
        raise ValueError('checkpoint must contain explicit state_dict')
    original = checkpoint['state_dict']
    if not original or any(type(k) is not str for k in original):
        raise ValueError('state keys must be nonempty strings')
    prefixed = [k.startswith('module.') for k in original]
    if any(prefixed) and not all(prefixed):
        raise ValueError('mixed checkpoint prefixes')
    prefix = 'module.' if all(prefixed) else ''
    state = {k[len(prefix):]: v for k, v in original.items()}
    if any(k.startswith('module.') for k in state) or len(state) != len(original):
        raise ValueError('double or colliding checkpoint prefixes')
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError('exact state keys required; missing or unexpected tensors')
    for name, tensor in state.items():
        want = expected[name]
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                or tensor.layout != torch.strided or tensor.is_quantized
                or tensor.shape != want.shape or tensor.dtype != want.dtype):
            raise ValueError('unsupported tensor, shape or dtype: ' + name)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('nonfinite state: ' + name)
        if name.endswith(('running_var', 'num_batches_tracked')) and bool((tensor < 0).any()):
            raise ValueError('negative BatchNorm state: ' + name)

    weight_norm_hooks = []
    # Use the installed native legacy calculation, not an approximate formula.
    for module_name, module in model.named_modules():
        base = module_name + '.' if module_name else ''
        for hook in module._forward_pre_hooks.values():
            if not isinstance(hook, WeightNorm):
                continue
            if hook.dim != 0:
                raise ValueError('only native dimension-zero weight normalization supported')
            v = state[base + hook.name + '_v']
            g = state[base + hook.name + '_g']
            norms = torch.norm_except_dim(v, 2, 0)
            effective = torch._weight_norm(v, g, 0)
            if not bool((norms > 0).all()) or not bool(torch.isfinite(effective).all()):
                raise ValueError('unusable weight-normalized head')
            weight_norm_hooks.append((module, hook))

    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError('strict checkpoint load mismatch')
    for module, hook in weight_norm_hooks:
        # Refresh only the known WeightNorm hook; this is not a forward pass.
        hook(module, None)
    return {'state_entries': len(state), 'prefix_removed': prefix,
            'strict_keys_shapes_dtypes_finite': True, 'derived_weights_refreshed': True}


def load_clipart2020(path):
    """Only the reviewed release can reach the full offline model factory."""
    data = read_verified_checkpoint(path, RELEASE_SHA256, RELEASE_BYTES)
    import torch
    with _cpu_preparation():
        import torchvision
        if (torch.__version__, torchvision.__version__) != ('2.8.0', '0.23.0'):
            raise ValueError('reviewed source loader requires torch2.8.0/torchvision0.23.0')
        model = build_reference_classifier()
        checks = apply_strict_state(model, data)
    for tensor in model.state_dict().values():
        if tensor.device.type != 'cpu':
            raise ValueError('CPU source state required')
    provenance = {
        'schema': 'kbound-domainnet-source-load-v1',
        'status': 'STRICT_OFFLINE_SOURCE_LOAD_ONLY',
        'release_sha256': RELEASE_SHA256, 'release_bytes': len(data),
        'class_map_sha256': CLASS_MAP_SHA256,
        'reference_commit': REFERENCE_COMMIT,
        'architecture': 'ResNet50_256BN_weightnorm126',
        'documented_release_seed': 2020,
        'seed_provenance': 'author_recipe_and_filename_not_independent_training_history',
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'loader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'io_helper_sha256': hashlib.sha256(Path(__import__('task3_image_panel').__file__).read_bytes()).hexdigest(),
        'compatibility_change': 'weights=None construction then exact release load; CPU; strict state checks',
        'study_locked': False, 'native_adaptation_executed': False,
        'images_read': 0, 'forward_passes': 0, 'optimizer_updates': 0,
        'training_history_reproduced': False,
        **checks,
    }
    return model, provenance
