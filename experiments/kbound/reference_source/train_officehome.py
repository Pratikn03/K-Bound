"""Pinned SHOT source-only training; new explicit 90/10 membership, then separate KGA.

Run as python -m experiments.kbound.reference_source.train_officehome.
No target list, target selection, sample cap, or reduced recipe is accepted.
"""
from __future__ import annotations

import argparse
import fcntl
import io
import math
import os
import random
import struct
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Reuse only dataset-independent safe I/O, provenance and RNG helpers.
from .train_rxrx1 import (SourceError, safe_path, read_bytes, file_sha, sha,
                         json_bytes, read_json, write_json, prepare_output,
                         rng_state, set_rng, runtime_identity, synchronize,
                         authenticated_torch_load)

REVISION = '9a8110cd16d6a52758459a2ea2c35f4d667666b8'
INITIALIZATION_SHA256 = '19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097'
COUNTS = {'Art': 2427, 'Clipart': 4365, 'Product': 4439, 'Real_World': 4357}
EPOCHS, BATCH_SIZE = 50, 64


def appledouble_entries(data):
    """Validate the RFC1740 outer container, including Apple's macOS filler.

    Payloads remain opaque metadata, never training examples. This validates
    format structure, not the historical origin of user-supplied metadata.
    """
    if len(data) < 26:
        raise SourceError('truncated AppleDouble header')
    magic, version, filler, count = struct.unpack_from('>II16sH', data)
    end = 26 + 12 * count
    if (magic != 0x00051607 or version != 0x00020000
            or filler not in {bytes(16), b'Mac OS X        '}
            or count == 0 or end > len(data)):
        raise SourceError('invalid AppleDouble header/version/entry table')
    entries, spans = [], []
    for i in range(count):
        entry_id, offset, length = struct.unpack_from('>III', data, 26 + 12 * i)
        # Only defined metadata IDs are admitted; data-fork ID1 is forbidden.
        if (entry_id not in {2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}
                or entry_id in entries or offset < end or offset + length > len(data)):
            raise SourceError('invalid AppleDouble metadata entry')
        entries.append(entry_id)
        if length:
            spans.append((offset, offset + length))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise SourceError('overlapping AppleDouble metadata entries')
    return entries


def inventory(root, domain, source_list, digest, *, expected_count=None, classes=65):
    """Accept explicit source-only mapping/order; never inspect other domains."""
    root = safe_path(root)
    raw = read_bytes(source_list)
    if sha(raw) != digest:
        raise SourceError('source list SHA256 mismatch')
    if domain not in COUNTS:
        raise SourceError('unknown source domain')
    expected_count = COUNTS[domain] if expected_count is None else expected_count
    rows, seen, mapping = [], set(), {}
    try:
        lines = raw.decode('utf-8').splitlines()
        for line in lines:
            path_text, label_text = line.rsplit(' ', 1)
            path = Path(path_text)
            if (len(path.parts) != 3 or path.parts[0] != domain or path.is_absolute()
                    or '..' in path.parts or str(path) != path_text
                    or not label_text.isdecimal()):
                raise SourceError('source list must contain canonical domain/class/file label rows')
            label, category = int(label_text), path.parts[1]
            if not 0 <= label < classes or mapping.get(category, label) != label or path_text in seen:
                raise SourceError('source list duplicate or inconsistent class mapping')
            mapping[category] = label; seen.add(path_text)
            data = read_bytes(root / path)
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                raise SourceError('truncated image decoding is forbidden')
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.width < 1 or image.height < 1:
                    raise SourceError('invalid image dimensions')
                image.convert('RGB').load()
                dims = [image.width, image.height]
            rows.append({'row_id': len(rows), 'path': path_text, 'label': label,
                         'sha256': sha(data), 'bytes': len(data), 'dimensions': dims})
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        raise SourceError(f'invalid source list/image: {exc}') from exc
    if len(rows) != expected_count or len(mapping) != classes or set(mapping.values()) != set(range(classes)):
        raise SourceError('full source population and bijective class mapping required')
    actual, sidecars = set(), []
    # Walk only the selected source domain, rejecting links and nonregular files.
    def walk_error(exc):
        raise SourceError(f'source tree cannot be enumerated: {exc}') from exc
    for folder, directories, files in os.walk(safe_path(root / domain), followlinks=False, onerror=walk_error):
        for name in directories:
            directory = safe_path(Path(folder) / name)
            if directory.parent != root / domain or name not in mapping:
                raise SourceError('unexpected source directory outside exact class mapping')
        for name in files:
            p = safe_path(Path(folder) / name)
            if not p.is_file():
                raise SourceError('source tree contains nonregular file')
            relative = str(p.relative_to(root))
            if name.startswith('._'):
                companion = safe_path(p.with_name(name[2:]))
                companion_relative = str(companion.relative_to(root))
                if companion_relative in seen and companion.is_file():
                    companion_kind = 'image'
                elif (companion.parent == root / domain and companion.name in mapping
                      and companion.is_dir()):
                    companion_kind = 'class_directory'
                else:
                    raise SourceError('AppleDouble companion must be a listed regular image or exact mapped source class directory')
                data = read_bytes(p)
                entries = appledouble_entries(data)
                sidecars.append({'path': relative, 'companion': companion_relative,
                                 'companion_kind': companion_kind,
                                 'sha256': sha(data), 'bytes': len(data),
                                 'version': 0x00020000, 'entry_ids': entries})
            else:
                actual.add(relative)
    if seen != actual:
        raise SourceError('source list does not cover the exact full source-domain tree')
    return rows, {'root': str(root), 'domain': domain, 'count': len(rows),
                  'source_list_sha256': sha(raw), 'class_mapping': mapping,
                  'ordered_content_sha256': sha(json_bytes(rows)),
                  'metadata_sidecars': sorted(sidecars, key=lambda r: r['path']),
                  'metadata_sidecar_policy': 'RFC1740 AppleDouble v2 or macOS filler; defined metadata IDs; listed regular image or exact mapped source class-directory companion; hashes separate from image population',
                  'historical_list_or_membership_equivalence_claimed': False}


def partition(rows):
    # Same torch random_split/global RNG semantics; list order is explicitly new.
    n = int(.9 * len(rows))
    train, val = torch.utils.data.random_split(range(len(rows)), [n, len(rows) - n])
    return {'train': list(train.indices), 'val': list(val.indices)}


def image_transform(training):
    return T.Compose([T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
                      *([T.RandomCrop(224), T.RandomHorizontalFlip()] if training else [T.CenterCrop(224)]),
                      T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class SourceDataset(Dataset):
    def __init__(self, root, rows, indices, *, training):
        self.root, self.rows, self.indices = Path(root), rows, indices
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.rows[self.indices[i]]
        data = read_bytes(self.root / row['path'])
        if len(data) != row['bytes'] or sha(data) != row['sha256']:
            raise SourceError('source image content changed')
        try:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                raise SourceError('truncated image decoding is forbidden')
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                x = self.transform(image.convert('RGB'))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise SourceError(f'source image decode failed: {exc}') from exc
        return x, row['label']


class ResBase(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        for name in ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'):
            setattr(self, name, getattr(backbone, name))

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x.view(x.size(0), -1)


class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(256, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(.5)
        self.bottleneck = nn.Linear(2048, 256)
        nn.init.xavier_normal_(self.bottleneck.weight)
        nn.init.zeros_(self.bottleneck.bias)

    def forward(self, x):
        # Upstream declares but does not use ReLU or dropout.
        return self.bn(self.bottleneck(x))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.utils.weight_norm(nn.Linear(256, 65), name='weight')
        # Preserve upstream initialization ORDER, including legacy transient weight.
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


def load_model(initialization):
    state = authenticated_torch_load(initialization, INITIALIZATION_SHA256)
    backbone = torchvision.models.resnet50(weights=None)
    backbone.load_state_dict(state, strict=True)
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in backbone.state_dict().values()):
        raise SourceError('nonfinite initialization')
    return nn.ModuleDict({'F': ResBase(backbone), 'B': Bottleneck(), 'C': Classifier()})


def forward(model, x):
    return model['C'](model['B'](model['F'](x)))


def optimizer_for(model):
    groups = []
    for name in ('F', 'B', 'C'):
        for parameter in model[name].parameters():
            lr = .001 if name == 'F' else .01
            groups.append({'params': [parameter], 'lr': lr, 'lr0': lr})
    return torch.optim.SGD(groups)


def schedule(optimizer, step, total_steps):
    if not 1 <= step <= total_steps:
        raise SourceError('scheduler step outside full source recipe')
    for group in optimizer.param_groups:
        group.update(lr=group['lr0'] * (1 + 10 * step / total_steps) ** -.75,
                     weight_decay=.001, momentum=.9, nesterov=True)


def smooth_loss(logits, labels):
    targets = torch.zeros_like(logits).scatter_(1, labels[:, None], 1)
    targets = .9 * targets + .1 / logits.shape[1]
    return -(targets * logits.log_softmax(1)).sum(1).mean()


def run_epoch(model, loader, device, *, optimizer=None, step=0, total_steps=1):
    training = optimizer is not None
    model.train(training)
    synchronize(device); start = time.monotonic()
    count = correct = batches = 0
    loss_sum = 0.
    with torch.set_grad_enabled(training):
        for x, y in loader:
            if training and len(y) == 1:
                raise SourceError('singleton train batch would be skipped upstream; prohibited')
            if training:
                step += 1
                schedule(optimizer, step, total_steps)
            x, y = x.to(device), y.to(device)
            logits = forward(model, x)
            loss = smooth_loss(logits, y)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise SourceError('nonfinite source loss/logits')
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            count += len(y); batches += 1
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.detach().argmax(1) == y).sum())
    synchronize(device)
    if count == 0:
        raise SourceError('empty source epoch')
    return {'smoothed_ce': loss_sum / count, 'accuracy': correct / count,
            'examples': count, 'batches': batches, 'wall_seconds': time.monotonic() - start}, step


def save_state(path, model, optimizer, identity, history, step, device):
    synchronize(device)
    state = {'schema': 'officehome_shot_epoch_v1', 'identity': identity,
             'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
             'optimizer': optimizer.state_dict(), 'step': step, 'history': history,
             'rng': rng_state(device)}
    with safe_path(path).open('xb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())


def restore_state(path, digest, model, optimizer, identity, device):
    data = read_bytes(path)
    if sha(data) != digest:
        raise SourceError('resume checkpoint hash identity mismatch')
    state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if state.get('schema') != 'officehome_shot_epoch_v1' or state.get('identity') != identity:
        raise SourceError('resume configuration identity mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    set_rng(state['rng'], device)
    return state['history'], state['step']


def selected_epoch(history):
    eligible = [i for i in range(len(history)) if (i + 1) % 5 == 0]
    return max(eligible, key=lambda i: (history[i]['val']['accuracy'], i)) if eligible else None


def geometry(domain):
    n = int(.9 * COUNTS[domain])
    return {'train': {'examples': n, 'batches': math.ceil(n / BATCH_SIZE)},
            'val': {'examples': COUNTS[domain] - n, 'batches': math.ceil((COUNTS[domain] - n) / BATCH_SIZE)}}


def valid_history(history, domain):
    expected = geometry(domain)
    return isinstance(history, list) and len(history) <= EPOCHS and all(
        type(r.get('epoch')) is int and r['epoch'] == i and all(
            type(r.get(s, {}).get(k)) is int and r[s][k] == v
            for s in expected for k, v in expected[s].items()) for i, r in enumerate(history))


def reference_complete(history, domain):
    return len(history) == EPOCHS and valid_history(history, domain)


def verify_best(out, pointer, history, identity):
    epoch = selected_epoch(history)
    if epoch is None:
        if pointer is not None:
            raise SourceError('selection before reference validation cadence')
        return
    if (not isinstance(pointer, dict) or pointer.get('epoch') != epoch
            or pointer.get('val_accuracy') != history[epoch]['val']['accuracy']
            or Path(pointer['checkpoint']).name != pointer['checkpoint']):
        raise SourceError('best pointer disagrees with reference selection history')
    data = read_bytes(out / pointer['checkpoint'])
    if sha(data) != pointer['sha256']:
        raise SourceError('best checkpoint hash mismatch')
    state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if (state.get('schema') != 'officehome_shot_epoch_v1' or state.get('identity') != identity
            or state.get('history') != history[:epoch + 1]):
        raise SourceError('best checkpoint provenance mismatch')


def execute(args):
    if args.workers < 0 or args.seed < 0 or args.seed >= 2**32:
        raise SourceError('invalid worker count or seed')
    if args.preflight and args.resume:
        raise SourceError('preflight cannot resume source training')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise SourceError('MPS unavailable')
    if not args.resume and safe_path(args.output_dir).exists():
        raise SourceError('fresh output directory required')
    rows, source = inventory(args.data_root, args.domain, args.source_list, args.source_list_sha256)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    split = partition(rows)  # Must precede model construction, as upstream.
    identity = {'schema': 'officehome_public_shot_source_v1', 'reference_revision': REVISION,
                'source': source, 'seed': args.seed, 'membership': split,
                'initialization_sha256': INITIALIZATION_SHA256,
                'training_code_sha256': file_sha(__file__),
                'helpers_code_sha256': file_sha(Path(__file__).with_name('train_rxrx1.py')),
                'runtime': runtime_identity(args.device), 'workers': args.workers,
                'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'drop_last': False,
                'mode': 'preflight' if args.preflight else 'train',
                'recipe': {'optimizer': 'SGD', 'feature_lr': .001, 'head_lr': .01,
                           'weight_decay': .001, 'momentum': .9, 'nesterov': True,
                           'scheduler': '(1+10*step/total_steps)^(-0.75),before optimizer,step1 first',
                           'label_smoothing': .1, 'bn': 'train normally',
                           'preprocessing': 'RGB;square resize256 bilinear;random crop224+flip train;center crop224 val;ImageNet normalization',
                           'selection': 'full source val accuracy every5epochs;latest tie wins',
                           'snapshot_correction': 'immutable real selected tensors, not shallow live state_dict',
                           'diagnostic_validation': 'intervening epochs preserve all RNG states'},
                'source_validation_ineligible_for_independent_kga_check': True,
                'historical_released_checkpoint_equivalence_claimed': False}
    out = prepare_output(args.output_dir, identity, resume=args.resume)
    with safe_path(out / 'run.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SourceError('source output already owned by running process') from exc
        if (out / 'completion.json').exists():
            raise SourceError('source output already completed')
        try:
            if not args.resume:
                write_json(out / 'source-population.json', source)
                write_json(out / 'source-membership.json', {'rows': rows, 'membership': split,
                           'source_validation_ineligible_for_independent_kga_check': True})
            elif (read_json(out / 'source-population.json') != source
                  or read_json(out / 'source-membership.json') != {'rows': rows, 'membership': split,
                    'source_validation_ineligible_for_independent_kga_check': True}):
                raise SourceError('resume source membership mismatch')
            model = load_model(args.initialization).to(args.device)
            opt = optimizer_for(model)
            loaders = {s: DataLoader(SourceDataset(args.data_root, rows, split[s], training=s == 'train'),
                                    batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                                    num_workers=args.workers, persistent_workers=False) for s in split}
            total_steps = EPOCHS * len(loaders['train'])
            if args.preflight:
                batch = next(iter(loaders['train']))
                if len(batch[1]) != BATCH_SIZE:
                    raise SourceError('full batch64 preflight required')
                metrics, _ = run_epoch(model, [batch], args.device, optimizer=opt, total_steps=total_steps)
                write_json(out / 'preflight.json', {'complete': False, 'source_training_complete': False,
                           'full_batch64_exercised': True, 'metrics': metrics})
                return
            history, best, step = [], None, 0
            if args.resume:
                journal = read_json(out / 'last-epoch.json')
                if Path(journal['checkpoint']).name != journal['checkpoint']:
                    raise SourceError('unsafe resume checkpoint path')
                history, step = restore_state(out / journal['checkpoint'], journal['sha256'], model, opt, identity, args.device)
                if not history or not valid_history(history, args.domain) or step != len(history) * len(loaders['train']):
                    raise SourceError('resume epoch geometry/scheduler mismatch')
                best = journal['best']
                verify_best(out, best, history, identity)
            for epoch in range(len(history), EPOCHS):
                train, step = run_epoch(model, loaders['train'], args.device, optimizer=opt, step=step, total_steps=total_steps)
                # Additional monitoring does not perturb upstream sampler/augmentation RNG.
                state = rng_state(args.device) if (epoch + 1) % 5 else None
                val, _ = run_epoch(model, loaders['val'], args.device)
                if state is not None:
                    set_rng(state, args.device)
                record = {'epoch': epoch, 'train': train, 'val': val, 'scheduler_step': step,
                          'selection_eligible': (epoch + 1) % 5 == 0,
                          'accuracy_gap_train_minus_val': train['accuracy'] - val['accuracy'],
                          'smoothed_ce_gap_val_minus_train': val['smoothed_ce'] - train['smoothed_ce'],
                          'learning_rates': [g['lr'] for g in opt.param_groups]}
                history.append(record)
                if not valid_history(history, args.domain):
                    raise SourceError('incomplete source epoch')
                name = f'epoch-{epoch:03d}-{uuid.uuid4().hex}.pt'
                save_state(out / name, model, opt, identity, history, step, args.device)
                digest = file_sha(out / name)
                if selected_epoch(history) == epoch:
                    best = {'checkpoint': name, 'sha256': digest, 'epoch': epoch, 'val_accuracy': val['accuracy']}
                journal = {'checkpoint': name, 'sha256': digest, 'epoch': epoch, 'best': best}
                write_json(out / 'last-epoch.json', journal, replace=True)
                write_json(out / 'learning-curves.json', history, replace=True)
                print(json_bytes(record).decode(), flush=True)
            if not reference_complete(history, args.domain):
                raise SourceError('full50 source epochs required')
            verify_best(out, best, history, identity)
            write_json(out / 'completion.json', {'complete': True, 'stage': 'source_training_only',
                       'all_nine_complete': False, 'epochs': EPOCHS, 'best': best, 'last': journal,
                       'configuration_sha256': file_sha(out / 'configuration.json'),
                       'population_sha256': file_sha(out / 'source-population.json'),
                       'membership_sha256': file_sha(out / 'source-membership.json'),
                       'source_validation_ineligible_for_independent_kga_check': True})
        except (Exception, KeyboardInterrupt) as exc:
            write_json(out / 'failure.json', {'complete': False, 'error_type': type(exc).__name__,
                       'error': str(exc), 'resume_policy': 'explicit verified completed epoch only'}, replace=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--domain', choices=list(COUNTS), required=True)
    parser.add_argument('--source-list', type=Path, required=True)
    parser.add_argument('--source-list-sha256', required=True)
    parser.add_argument('--initialization', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    try:
        execute(parser.parse_args(argv))
    except (Exception, KeyboardInterrupt) as exc:
        print(json_bytes({'complete': False, 'error_type': type(exc).__name__, 'error': str(exc)}).decode(), flush=True)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
