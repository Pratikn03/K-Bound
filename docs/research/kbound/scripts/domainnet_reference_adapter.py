"""Pinned official AdaContrast, single-process CPU/MPS bridge.

No dataset IO, target labels, scoring, telemetry, or training-on-import. This
executes selected hash-verified official definitions, changing only device
allocations. Unmodified losses, optimizer, LR schedule, augmentation and epoch
training order are retained. The MoCo DDP shuffle is replaced by its one-process
equivalent. This is infrastructure for a transferred development variant;
it is not a reproduction of the published seven-domain-shift benchmark.
"""
import ast
import hashlib
import logging
import math
from pathlib import Path
import copy
import numpy as np
import random
import time
from types import SimpleNamespace

from PIL import Image, ImageFilter
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import yaml

from domainnet_reference_source import load_clipart2020, read_verified_checkpoint

# Source roots are supplied explicitly; no scratch path or automatic download.
REFERENCE_COMMIT = "c3c8b880131f2658d6fd0d5ed14d71f326174d57"
CHECKPOINT_SHA256 = "0b8cb099b93890ff9b3fdd993368846ea506c89f81fc33b670959e04ab16f8de"
SOURCE_HASHES = {
    "target.py": "bb10bc1d2fc26169e4f1926cfb1cea85d7797404e7b628bcefdd09a4c96f8595",
    "utils.py": "be3cdd3ad41b24bee81b9ddf061cc0e6c99fbdb83a82ab10dc8878bed8eeacd3",
    "classifier.py": "08a70ece6985ca65805a816500ec3de7da5de5bbb48ae45b155af869b2c34b1f",
    "moco/builder.py": "921973c692883b7b71251f8e32917a9f5cf85ba10a43b92c410ca4096ea6e5f1",
    "moco/loader.py": "accde6613dfe9ae11e55adba53dfa13a4ffa0818f0931e35fea39bf7b7d6d6d4",
}


class _DeviceAllocations(ast.NodeTransformer):
    """No arithmetic/algorithm edits: CUDA-only allocations become DEVICE."""
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            if node.args or node.keywords:
                raise ValueError("unexpected argument in pinned CUDA allocation")
            node.func.attr = "to"
            node.args = [ast.Name(id="DEVICE", ctx=ast.Load())]
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == "to"
              and node.args and isinstance(node.args[0], ast.Constant)
              and node.args[0].value == "cuda"):
            node.args[0] = ast.Name(id="DEVICE", ctx=ast.Load())
        return node


def _extract(root, filename, names, namespace):
    source = _verified_source(root, filename, SOURCE_HASHES[filename])
    tree = ast.parse(source, filename=str(root / filename))
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in tree.body} != set(names):
        raise ValueError(f"missing pinned definitions: {filename}")
    tree = ast.fix_missing_locations(_DeviceAllocations().visit(tree))
    exec(compile(tree, str(root / filename), "exec"), namespace)


def load_reference(root, device="cpu"):
    """Extract only local algorithm definitions; never import wandb or launchers."""
    device = _device(device)
    namespace = dict(torch=torch, nn=nn, F=F, math=math, time=time,
                     logging=logging, random=random, Image=Image,
                     ImageFilter=ImageFilter, transforms=transforms, DEVICE=device,
                     concat_all_gather=lambda tensor: tensor, use_wandb=lambda args: False)
    root = Path(root)
    # Authenticate also declared files not AST-executed by this bridge (the
    # full classifier is constructed by the separate strict offline loader).
    for filename, digest in SOURCE_HASHES.items():
        _verified_source(root, filename, digest)
    _extract(root, "moco/loader.py", {"GaussianBlur", "NCropsTransform"}, namespace)
    _extract(root, "utils.py", {"get_augmentation", "get_distances", "adjust_learning_rate", "AverageMeter", "ProgressMeter"}, namespace)
    _extract(root, "target.py", {
        "soft_k_nearest_neighbors", "update_labels", "refine_predictions",
        "get_augmentation_versions", "get_target_optimizer", "train_epoch",
        "calculate_acc", "instance_loss", "classification_loss", "div",
        "diversification_loss", "smoothed_cross_entropy", "cross_entropy_loss",
        "entropy_minimization",
    }, namespace)
    _extract(root, "moco/builder.py", {"AdaMoCo"}, namespace)
    original = namespace["AdaMoCo"]

    class SingleProcessAdaMoCo(original):
        @torch.no_grad()
        def _batch_shuffle_ddp(self, x):
            # Exactly the world_size=1 permutation; no rank communication.
            shuffle = torch.randperm(len(x)).to(x.device)
            return x[shuffle], torch.argsort(shuffle)

        @torch.no_grad()
        def _batch_unshuffle_ddp(self, x, idx_unshuffle):
            return x[idx_unshuffle]

    exported = dict(namespace, AdaMoCo=SingleProcessAdaMoCo)
    return SimpleNamespace(**exported)




def validate_loss_summary(summary, expected_count):
    """Native per-update mean (not an image-weighted or recomputed loss)."""
    if (not isinstance(summary, dict) or set(summary) != {'count', 'sum', 'mean', 'last'}
            or type(summary['count']) is not int or summary['count'] != expected_count
            or expected_count <= 0):
        raise ValueError('incomplete native loss summary')
    if any(type(summary[k]) not in (int, float) or not math.isfinite(summary[k])
           for k in ('sum', 'mean', 'last')):
        raise ValueError('nonfinite native loss summary')
    if not math.isclose(summary['mean'], summary['sum'] / expected_count, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('inconsistent native loss mean')
    return summary


class ReferenceSession:
    """Resettable per-cell U-only adapter. Inputs contain no ground-truth labels.

    source_factory must construct a fresh source model with identical weights.
    reset reconstructs both encoders (avoiding weight-norm deepcopy problems),
    memories and optimizer, and resets stochastic seeds. Full target evaluation
    from the upstream launcher is deliberately excluded; bank initialization
    uses only caller-provided unlabeled adaptation inputs.
    """
    def __init__(self, source_factory, args, device="cpu", seed=2020,
                 reference_root=None):
        if args.distributed:
            raise ValueError("single-process bridge requires distributed=False")
        self.source_factory = source_factory
        self.args = copy.deepcopy(args)
        self._configured_args = _plain_args(self.args)
        self.device = _device(device)
        self.seed = seed
        self._configured_seed = seed
        self._configured_device = str(self.device)
        self.reference_root = Path(reference_root)
        self.ref = load_reference(reference_root, device)
        self.reset()

    def reset(self):
        self._ensure_configuration()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        source = self.source_factory()
        momentum = self.source_factory()
        # A source constructor may consume RNG; both encoders must start equal.
        momentum.load_state_dict(source.state_dict(), strict=True)
        cfg = self.args.model_tta
        self.model = self.ref.AdaMoCo(source, momentum, K=cfg.queue_size,
                                     m=cfg.m, T_moco=cfg.T_moco).to(self.device)
        self.device = next(self.model.parameters()).device
        self.optimizer = self.ref.get_target_optimizer(self.model, self.args)
        self.banks = None
        self.completed_steps = 0
        self.failed = False
        self.identity = {
            'device': str(self.device), 'seed': self.seed,
            'args': _plain_args(self.args), 'source_hashes': dict(SOURCE_HASHES),
            'initial_model': _state_digest(self.model.state_dict()),
            'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def _ensure_configuration(self):
        if (_plain_args(self.args) != self._configured_args
                or self.seed != self._configured_seed
                or str(self.device) != self._configured_device):
            raise ValueError('session configuration changed')
        epochs, total = self.args.learn.epochs, self.args.learn.full_progress
        if (type(epochs) is not int or type(total) is not int
                or epochs <= 0 or total <= 0 or total % epochs):
            raise ValueError('invalid declared epoch schedule')
        return total // epochs

    @torch.no_grad()
    def initialize_bank(self, batches):
        self._ensure_configuration()
        if self.failed or self.banks is not None or self.completed_steps:
            raise ValueError('bank initialization requires reset')
        self.model.eval()
        features, probabilities = [], []
        try:
            for images in batches:
                _validate_images(images)
                feats, logits = self.model(images.to(self.device), cls_only=True)
                probs = F.softmax(logits, dim=1)
                if not bool(torch.isfinite(feats).all()) or not bool(torch.isfinite(probs).all()):
                    raise ValueError('nonfinite bank model outputs')
                features.append(feats)
                probabilities.append(probs)
            if not features:
                raise ValueError("unlabeled adaptation bank cannot be empty")
            features = torch.cat(features)
            probabilities = torch.cat(probabilities)
            size = len(features) if self.args.learn.queue_size == -1 else self.args.learn.queue_size
            if size < self.args.learn.num_neighbors or len(features) < self.args.learn.num_neighbors:
                raise ValueError("insufficient adaptation examples for configured neighbors")
            permutation = torch.randperm(len(features)).to(self.device)
            self._ensure_configuration()
            self.banks = {"features": features[permutation][:size],
                          "probs": probabilities[permutation][:size], "ptr": 0}
        except Exception:
            self.failed = True
            raise

    def train_epoch(self, batches, epoch):
        epoch_length = self._ensure_configuration()
        if self.banks is None:
            raise ValueError("initialize unlabeled adaptation bank first")
        if self.failed:
            raise ValueError("failed partial epoch requires reset; no silent continuation")
        if type(epoch) is not int or not 0 <= epoch < self.args.learn.epochs:
            raise ValueError("epoch outside declared schedule")
        stream = StreamingEpoch(batches, len(self.banks["features"]), self.args.model_tta.queue_size)
        if len(stream) != epoch_length or self.completed_steps != epoch * epoch_length:
            raise ValueError("epoch/schedule cursor mismatch")
        # Each extracted reference has its own globals. Observe the *existing*
        # native meter, with no new tensor operations or random draws.
        namespace = self.ref.get_target_optimizer.__globals__
        native_meter = namespace['AverageMeter']
        loss_meters = []

        def observed_meter(name, fmt=':f'):
            meter = native_meter(name, fmt)
            if name == 'Loss':
                loss_meters.append(meter)
            return meter

        namespace['AverageMeter'] = observed_meter
        try:
            self.ref.train_epoch(stream, self.model, self.banks, self.optimizer, epoch, copy.deepcopy(self.args))
            self._ensure_configuration()
            _require_finite_state(self.model.state_dict())
            _require_finite_state(self.optimizer.state_dict())
            _require_finite_state(self.banks)
            if len(loss_meters) != 1:
                raise ValueError('missing or duplicate native loss meter')
            meter = loss_meters[0]
            summary = validate_loss_summary(
                {'count': meter.count, 'sum': meter.sum, 'mean': meter.avg, 'last': meter.val}, epoch_length
            )
        except Exception:
            self.failed = True
            raise
        finally:
            namespace['AverageMeter'] = native_meter
        self.completed_steps += len(stream)
        return summary

    @torch.no_grad()
    def predict(self, batches, refine=False):
        self._ensure_configuration()
        if self.failed:
            raise ValueError('failed partial epoch requires reset')
        self.model.eval()
        predicted = []
        for images in batches:
            _validate_images(images)
            features, logits = self.model(images.to(self.device), cls_only=True)
            _require_finite_state((features, logits))
            if refine:
                if self.banks is None:
                    raise ValueError("refined inference requires initialized bank")
                labels, _, _ = self.ref.refine_predictions(features, F.softmax(logits, 1), self.banks, self.args)
            else:
                labels = logits.argmax(1)
            predicted.append(labels.cpu())
        return torch.cat(predicted) if predicted else torch.empty(0, dtype=torch.long)

    def snapshot_state(self):
        """In-memory epoch-boundary state; the runner must bind its sampler/cursor.

        Digest detects corruption, not external authenticity. The outer runner
        must bind this packet to an immutable manifest before persistent resume.
        A partial/failed epoch cannot yield a resumable or scored candidate.
        """
        self._ensure_configuration()
        if self.failed:
            raise ValueError('failed partial epoch requires reset')
        if _plain_args(self.args) != self.identity['args']:
            raise ValueError('session configuration changed')
        packet = {
            'schema': 'kbound-adacontrast-epoch-state-v1',
            'identity': copy.deepcopy(self.identity),
            'model': _cpu_copy(self.model.state_dict()),
            'optimizer': _cpu_copy(self.optimizer.state_dict()),
            'banks': _cpu_copy(self.banks), 'queue_ptr': self.model.queue_ptr,
            'completed_steps': self.completed_steps,
            'training': {n: m.training for n, m in self.model.named_modules()},
            'gradients': {n: _cpu_copy(p.grad) for n, p in self.model.named_parameters()},
            'rng': _rng_state(self.device),
        }
        packet['digest'] = _state_digest(packet)
        return packet

    def restore_state(self, state):
        """Validate in a fresh candidate, then swap; rejected packets change nothing."""
        fields = {'schema', 'identity', 'model', 'optimizer', 'banks', 'queue_ptr',
                  'completed_steps', 'training', 'gradients', 'rng', 'digest'}
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError('invalid state fields')
        body = {k: v for k, v in state.items() if k != 'digest'}
        if state['digest'] != _state_digest(body):
            raise ValueError('state digest mismatch')
        if (state['schema'] != 'kbound-adacontrast-epoch-state-v1'
                or state['identity'] != self.identity
                or _plain_args(self.args) != self.identity['args']):
            raise ValueError('state source/config/runtime/device identity mismatch')
        prior_rng = _rng_state(self.device)
        try:
            candidate = ReferenceSession(self.source_factory, self.args, device=self.device,
                                         seed=self.seed, reference_root=self.reference_root)
            if candidate.identity != self.identity:
                raise ValueError('fresh source identity changed')
            expected = candidate.model.state_dict()
            if set(state['model']) != set(expected):
                raise ValueError('state tensor keys mismatch')
            for name, want in expected.items():
                tensor = state['model'][name]
                _validate_tensor(tensor, want, name)
                if name.endswith(('running_var', 'num_batches_tracked')) and bool((tensor < 0).any()):
                    raise ValueError('invalid BatchNorm state')
            labels = state['model']['mem_labels']
            if bool(((labels < 0) | (labels >= candidate.model.src_model.num_classes)).any()):
                raise ValueError('invalid queue labels')
            if type(state['queue_ptr']) is not int or not 0 <= state['queue_ptr'] < self.args.model_tta.queue_size:
                raise ValueError('invalid queue pointer')
            steps = state['completed_steps']
            if (type(steps) is not int or not 0 <= steps <= self.args.learn.full_progress
                    or steps % self._ensure_configuration()):
                raise ValueError('invalid progress')
            if steps:
                candidate.ref.adjust_learning_rate(candidate.optimizer, steps - 1, candidate.args)
            _validate_optimizer(state['optimizer'], candidate.optimizer, trained=bool(steps))
            candidate.model.load_state_dict(state['model'], strict=True)
            _refresh_weightnorm(candidate.model)
            candidate.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            candidate.model.queue_ptr = state['queue_ptr']
            candidate.completed_steps = steps
            bank = state['banks']
            if bank is not None:
                if not isinstance(bank, dict) or set(bank) != {'features', 'probs', 'ptr'}:
                    raise ValueError('invalid neighbor bank')
                feature, probability = bank['features'], bank['probs']
                if (type(feature) is not torch.Tensor or feature.ndim != 2
                        or feature.shape[1] != candidate.model.src_model.output_dim
                        or len(feature) < self.args.learn.num_neighbors):
                    raise ValueError('invalid bank feature shape')
                if (type(probability) is not torch.Tensor
                        or probability.shape != (len(feature), candidate.model.src_model.num_classes)):
                    raise ValueError('invalid bank probability shape')
                for tensor in (feature, probability):
                    if tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                        raise ValueError('invalid bank tensor')
                if (bool((probability < 0).any()) or bool((probability > 1).any())
                        or not torch.allclose(probability.sum(1), torch.ones(len(feature)), atol=1e-5, rtol=0)):
                    raise ValueError('invalid bank probabilities')
                if type(bank['ptr']) is not int or not 0 <= bank['ptr'] < len(feature):
                    raise ValueError('invalid bank pointer')
                candidate.banks = {k: v.clone().to(self.device) if torch.is_tensor(v) else v for k, v in bank.items()}
            elif steps:
                raise ValueError('trained state missing bank')
            modules = dict(candidate.model.named_modules())
            if set(state['training']) != set(modules) or any(type(v) is not bool for v in state['training'].values()):
                raise ValueError('invalid normalization/training mode state')
            for name, value in state['training'].items():
                modules[name].training = value
            params = dict(candidate.model.named_parameters())
            if set(state['gradients']) != set(params):
                raise ValueError('invalid gradient keys')
            for name, value in state['gradients'].items():
                if value is not None:
                    _validate_tensor(value, params[name], name)
                    if not params[name].requires_grad:
                        raise ValueError('gradient on frozen teacher')
                    params[name].grad = value.clone().to(self.device)
            # Validate CPU/Python/NumPy/device RNG on isolated globals, restoring
            # the caller's current state even if any backend rejects the packet.
            _set_rng(state['rng'], self.device)
        except (KeyError, TypeError, RuntimeError, OverflowError) as exc:
            raise ValueError('malformed resume state') from exc
        finally:
            _set_rng(prior_rng, self.device)
        self.model, self.optimizer, self.banks = candidate.model, candidate.optimizer, candidate.banks
        self.completed_steps, self.failed = candidate.completed_steps, False
        _set_rng(state['rng'], self.device)


def _plain_args(value):
    if isinstance(value, SimpleNamespace):
        return {k: _plain_args(v) for k, v in vars(value).items()}
    if isinstance(value, (tuple, list)):
        return [_plain_args(v) for v in value]
    return value


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def _state_digest(value):
    """Deterministic typed hash without pickle or executable deserialization."""
    digest = hashlib.sha256()
    def walk(item):
        def add(data):
            digest.update(str(len(data)).encode() + b':' + data)
        if type(item) is torch.Tensor:
            add(b'tensor'); add(str(item.dtype).encode()); add(str(tuple(item.shape)).encode())
            add(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            add(b'array'); add(str(item.dtype).encode()); add(str(item.shape).encode()); add(item.tobytes())
        elif isinstance(item, dict):
            add(b'dict'); add(str(len(item)).encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                walk(key); walk(item[key])
        elif isinstance(item, (list, tuple)):
            add(type(item).__name__.encode()); add(str(len(item)).encode())
            for child in item: walk(child)
        elif item is None or type(item) in (str, int, float, bool):
            add(type(item).__name__.encode()); add(repr(item).encode())
        else:
            raise ValueError('unsupported state value')
    walk(value)
    return digest.hexdigest()


def _rng_state(device):
    return {'torch': torch.get_rng_state().clone(), 'python': random.getstate(),
            'numpy': copy.deepcopy(np.random.get_state()),
            'mps': torch.mps.get_rng_state().clone() if device.type == 'mps' else None}


def _set_rng(state, device):
    if not isinstance(state, dict) or set(state) != {'torch', 'python', 'numpy', 'mps'}:
        raise ValueError('invalid RNG state')
    if (device.type == 'cpu') != (state['mps'] is None):
        raise ValueError('RNG device mismatch')
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if device.type == 'mps': torch.mps.set_rng_state(state['mps'])


def _validate_tensor(value, expected, name):
    if (type(value) is not torch.Tensor or value.device.type != 'cpu'
            or value.layout != torch.strided or value.shape != expected.shape
            or value.dtype != expected.dtype or not bool(torch.isfinite(value).all())):
        raise ValueError('invalid state tensor: ' + name)


def _validate_optimizer(state, optimizer, *, trained):
    if not isinstance(state, dict) or set(state) != {'state', 'param_groups'}:
        raise ValueError('invalid optimizer state')
    expected = optimizer.state_dict()
    if len(state['param_groups']) != len(expected['param_groups']):
        raise ValueError('optimizer groups mismatch')
    params = {}
    for actual, wanted, live in zip(state['param_groups'], expected['param_groups'], optimizer.param_groups):
        if set(actual) != set(wanted) or any(actual[k] != wanted[k] for k in wanted):
            raise ValueError('optimizer recipe mismatch')
        if type(actual['lr']) not in (int, float) or not math.isfinite(actual['lr']) or not 0 <= actual['lr'] <= wanted['lr']:
            raise ValueError('invalid optimizer learning rate')
        params.update(zip(wanted['params'], live['params']))
    if not isinstance(state['state'], dict) or not set(state['state']) <= set(params):
        raise ValueError('unknown optimizer parameter')
    if set(state['state']) != (set(params) if trained else set()):
        raise ValueError('SGD momentum slots do not match trained progress')
    for key, slot in state['state'].items():
        if not isinstance(slot, dict) or set(slot) != {'momentum_buffer'}:
            raise ValueError('invalid SGD momentum slot')
        _validate_tensor(slot['momentum_buffer'], params[key], 'momentum')


def _refresh_weightnorm(model):
    from torch.nn.utils.weight_norm import WeightNorm
    for module in model.modules():
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm):
                weight = hook.compute_weight(module)
                if not bool(torch.isfinite(weight).all()):
                    raise ValueError('invalid derived WeightNorm tensor')
                setattr(module, hook.name, weight)


def _validate_images(images):
    if (type(images) is not torch.Tensor or images.ndim < 2 or len(images) == 0
            or images.layout != torch.strided or images.dtype != torch.float32
            or not bool(torch.isfinite(images).all())):
        raise ValueError('nonempty finite FP32 image tensors required')


def _require_finite_state(value):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite native execution state')
    elif isinstance(value, dict):
        for child in value.values(): _require_finite_state(child)
    elif isinstance(value, (tuple, list)):
        for child in value: _require_finite_state(child)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('nonfinite native execution state')


RECIPE_HASHES = {
    "configs/optim/sgd.yaml": "942d71b10b16990ee80a97ecad5ee3c6cdf5d3f8f0fc4d23cbc2ae2d00c29ae6",
    "configs/learn/target.yaml": "2ef18a628f05286fab53a4a47c13251aa608b7e27e4ab253963310e6e13e5da3",
    "configs/data/basic.yaml": "10f60aff7c0018717786b3a8ad02881eab732260e82b052e7c979e39f48ca562",
    "configs/model_tta/moco.yaml": "112826845588e4403323f01bd3992bfaea48fd006c13a9c679099b3c8237f060",
    "train_domainnet-126_target.sh": "66e443c2223de50871c00abf74f5d5de2d2aef3e71b07c1842d37642e4fb388b",
}


def _device(value):
    if str(value) not in {"cpu", "mps"}:
        raise ValueError("only explicit cpu or mps is supported; no device fallback")
    if str(value) == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS device is unavailable")
    return torch.device(value)


def _verified_source(root, name, digest):
    path = Path(root) / name
    size = path.lstat().st_size
    if not 0 < size <= 1024 * 1024:
        raise ValueError("reference source exceeds bounded size")
    try:
        return read_verified_checkpoint(path, digest, size)
    except ValueError as exc:
        raise ValueError("reference integrity mismatch: " + name) from exc


def reference_args(reference_root, *, epochs, steps_per_epoch):
    """Authenticate every YAML byte plus the shell's LR override before parsing."""
    if any(type(n) is not int or n <= 0 for n in (epochs, steps_per_epoch)):
        raise ValueError("positive explicit schedule lengths required")
    raw = {name: _verified_source(reference_root, name, digest)
           for name, digest in RECIPE_HASHES.items()}
    paths = {"optim": "optim/sgd.yaml", "learn": "learn/target.yaml",
             "data": "data/basic.yaml", "model_tta": "model_tta/moco.yaml"}
    result = {key: SimpleNamespace(**yaml.safe_load(raw["configs/" + name]))
              for key, name in paths.items()}
    # Exact authenticated target shell override; the shell is never executed.
    result["optim"].lr = 2e-4
    for key in ("lr", "momentum", "weight_decay", "gamma"):
        setattr(result["optim"], key, float(getattr(result["optim"], key)))
    result["learn"].epochs = epochs
    result["learn"].full_progress = epochs * steps_per_epoch
    return SimpleNamespace(**result, distributed=False, use_wandb=False), {
        "reference_commit": REFERENCE_COMMIT,
        "source_hashes": dict(RECIPE_HASHES),
        "epochs": epochs, "steps_per_epoch": steps_per_epoch,
        "compatibility": "single-process allocation/shuffle bridge; adaptation-only bank",
        "full_published_protocol_reproduction": False,
    }


class StreamingEpoch:
    """Sized, lazy input wrapper retaining augmentation/update RNG interleaving."""
    def __init__(self, batches, bank_size, queue_size):
        self.batches = batches
        try:
            self.count = len(batches)
        except TypeError as exc:
            raise ValueError("explicit sized epoch required by native scheduler") from exc
        if self.count <= 0:
            raise ValueError("empty training epoch")
        self.capacity = min(bank_size, queue_size)

    def __len__(self):
        return self.count

    def __iter__(self):
        seen = 0
        for item in self.batches:
            if seen >= self.count:
                raise ValueError("epoch has more batches than declared")
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("expected outcome-free views and indices")
            views, indices = item
            if not isinstance(views, (tuple, list)) or len(views) != 3:
                raise ValueError("reference training requires three wss views")
            if (type(indices) is not torch.Tensor or indices.dtype != torch.int64
                    or indices.ndim != 1 or not 2 <= len(indices) <= self.capacity
                    or bool((indices < 0).any()) or len(indices.unique()) != len(indices)):
                raise ValueError("invalid indices, singleton or queue/bank capacity")
            shape = views[0].shape if type(views[0]) is torch.Tensor else None
            for view in views:
                if (type(view) is not torch.Tensor or view.ndim < 2
                        or view.shape != shape or len(view) != len(indices)
                        or view.dtype != torch.float32 or not bool(torch.isfinite(view).all())):
                    raise ValueError("invalid aligned finite FP32 views")
            seen += 1
            yield views, None, indices
        if seen != self.count:
            raise ValueError("epoch has fewer batches than declared")


def build_session(checkpoint_path, config, device):
    """Full authenticated classifier only. No public test-network injection."""
    _device(device)  # reject unavailable devices before any file access
    fields = {"reference_root", "epochs", "steps_per_epoch", "seed"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("exact explicit reference root, epochs, steps_per_epoch and seed required")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("invalid stream seed")
    args, identity = reference_args(config["reference_root"], epochs=config["epochs"],
                                    steps_per_epoch=config["steps_per_epoch"])
    def factory():
        model, receipt = load_clipart2020(checkpoint_path)
        source_receipts.append(receipt)
        return model
    source_receipts = []
    session = ReferenceSession(factory, args, device=device, seed=config["seed"],
                               reference_root=config["reference_root"])
    session.provenance = {**identity, "native_source_hashes": dict(SOURCE_HASHES),
                          "checkpoint": source_receipts[0],
                          "stream_seed": config["seed"], "device": str(session.device),
                          "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "study_locked": False}
    return session
