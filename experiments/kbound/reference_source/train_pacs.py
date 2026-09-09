"""Pinned DomainBed ERM PACS source sweep, followed separately by KGA.

Workers=0, authenticated modern torchvision V1 initialization, strict decoding,
source-only evaluation and replayable samplers are explicit reproduction boundaries.
No target image or target label is loaded. No reduced training recipe is exposed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import math
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import Dataset, RandomSampler
from torchvision import transforms as T

from .train_rxrx1 import (SourceError, safe_path, read_bytes, file_sha, sha,
                         json_bytes, read_json, write_json, prepare_output,
                         rng_state, set_rng, runtime_identity, synchronize,
                         authenticated_torch_load)
from .train_officehome import appledouble_entries

REVISION = '7df6f06a6f9062284812a3f174c306218932c5e4'
INITIALIZATION_SHA256 = '19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097'
COUNTS = {'art_painting': 2048, 'cartoon': 2344, 'photo': 1670, 'sketch': 3929}
DOMAINS = tuple(COUNTS)
STEPS, CHECKPOINT_FREQUENCY = 5001, 300


def seed_hash(*args):
    return int(hashlib.md5(str(args).encode()).hexdigest(), 16) % (2 ** 31)


def hparams(hparams_seed, trial_seed):
    if (type(hparams_seed) is not int or type(trial_seed) is not int
            or not 0 <= hparams_seed < 20 or not 0 <= trial_seed < 3):
        raise SourceError('the full reference grid requires hparams_seed0..19 and trial_seed0..2')
    result = dict(data_augmentation=True, resnet18=False, class_balanced=False,
                  resnet_dropout=0., lr=5e-5, batch_size=32, weight_decay=0.)
    if hparams_seed:
        rng = np.random.RandomState(seed_hash(hparams_seed, trial_seed))
        result.update(resnet_dropout=float(rng.choice([0., .1, .5])),
                      lr=float(10 ** rng.uniform(-5, -3.5)),
                      batch_size=int(2 ** rng.uniform(3, 5.5)),
                      weight_decay=float(10 ** rng.uniform(-6, -2)))
    return result


def job(target, hparams_seed, trial_seed):
    if target not in DOMAINS:
        raise SourceError('unknown PACS target domain')
    params = hparams(hparams_seed, trial_seed)
    return dict(target=target, target_index=DOMAINS.index(target),
                hparams_seed=hparams_seed, trial_seed=trial_seed, hparams=params,
                seed=seed_hash('PACS', 'ERM', [DOMAINS.index(target)], hparams_seed, trial_seed))


def jobs():
    return [job(target, hp, trial) for target in DOMAINS for trial in range(3) for hp in range(20)]


def partition(n, trial_seed, env_index):
    indices = list(range(n))
    np.random.RandomState(seed_hash(trial_seed, env_index)).shuffle(indices)
    return {'out': indices[:int(.2 * n)], 'in': indices[int(.2 * n):]}


def decode(data):
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise SourceError('truncated image decoding is forbidden')
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.convert('RGB')
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise SourceError(f'source image decode failed: {exc}') from exc


def accept_population(root, manifest, digest, target, *, expected_counts=None, classes=7):
    """Manifest binds three sorted ImageFolder-style lists; target is never walked.

    expected_counts/classes are test-fixture seams, never exposed by the CLI.
    """
    root = safe_path(root)
    raw = read_bytes(manifest)
    if sha(raw) != digest:
        raise SourceError('source manifest SHA256 mismatch')
    import json
    try:
        spec = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise SourceError('invalid source manifest JSON') from exc
    sources = [d for d in DOMAINS if d != target]
    if (target not in DOMAINS or not isinstance(spec, dict)
            or set(spec) != {'schema', 'target', 'domains'}
            or spec['schema'] != 'pacs_source_manifest_v1' or spec['target'] != target
            or not isinstance(spec['domains'], dict) or set(spec['domains']) != set(sources)):
        raise SourceError('manifest must declare exactly the three source domains, never target')
    expected_counts = COUNTS if expected_counts is None else expected_counts
    all_rows, identities, common_mapping = {}, {}, None
    extensions = torchvision.datasets.folder.IMG_EXTENSIONS
    for domain in sources:
        entry = spec['domains'][domain]
        if not isinstance(entry, dict) or set(entry) != {'list', 'sha256'}:
            raise SourceError('source manifest domain entry malformed')
        listing = read_bytes(entry['list'])
        if sha(listing) != entry['sha256']:
            raise SourceError('source list SHA256 mismatch')
        rows, mapping, seen = [], {}, set()
        try:
            for line in listing.decode('utf-8').splitlines():
                path_text, label_text = line.rsplit(' ', 1)
                path = Path(path_text)
                if (len(path.parts) != 3 or path.parts[0] != domain or path.is_absolute()
                        or '..' in path.parts or str(path) != path_text or not label_text.isdecimal()
                        or path.name.startswith('._') or path.suffix.lower() not in extensions):
                    raise SourceError('canonical source domain/class/image label required')
                label, category = int(label_text), path.parts[1]
                if (not 0 <= label < classes or mapping.get(category, label) != label
                        or path_text in seen):
                    raise SourceError('duplicate or inconsistent source label/path')
                mapping[category] = label; seen.add(path_text)
                data = read_bytes(root / path); image = decode(data)
                rows.append(dict(row_id=len(rows), path=path_text, label=label, sha256=sha(data),
                                 bytes=len(data), dimensions=list(image.size)))
        except (ValueError, OSError) as exc:
            raise SourceError(f'invalid source list/image: {exc}') from exc
        if (len(rows) != expected_counts[domain] or len(mapping) != classes
                or mapping != {name: i for i, name in enumerate(sorted(mapping))}
                or [r['path'] for r in rows] != sorted(seen)):
            raise SourceError('full sorted ImageFolder source population and seven-class mapping required')
        if common_mapping is not None and mapping != common_mapping:
            raise SourceError('source domains disagree on class mapping')
        common_mapping = mapping
        actual, sidecars = set(), []
        def walk_error(exc):
            raise SourceError(f'source tree enumeration failed: {exc}') from exc
        for folder, directories, files in os.walk(safe_path(root / domain), followlinks=False, onerror=walk_error):
            for name in directories:
                directory = safe_path(Path(folder) / name)
                if directory.parent != root / domain or name not in mapping:
                    raise SourceError('unexpected source directory outside exact class mapping')
            for name in files:
                path = safe_path(Path(folder) / name)
                if not path.is_file():
                    raise SourceError('nonregular source tree entry')
                relative = str(path.relative_to(root))
                if name.startswith('._'):
                    companion = safe_path(path.with_name(name[2:]))
                    rel = str(companion.relative_to(root))
                    if rel in seen and companion.is_file():
                        kind = 'image'
                    elif companion.parent == root / domain and companion.name in mapping and companion.is_dir():
                        kind = 'class_directory'
                    else:
                        raise SourceError('AppleDouble companion is not an exact source image/class')
                    data = read_bytes(path)
                    sidecars.append(dict(path=relative, companion=rel, companion_kind=kind,
                                         sha256=sha(data), bytes=len(data), entry_ids=appledouble_entries(data)))
                else:
                    actual.add(relative)
        if actual != seen:
            raise SourceError('source list does not cover exact source domain tree')
        all_rows[domain] = rows
        identities[domain] = dict(source_list=str(safe_path(entry['list'])), source_list_sha256=sha(listing),
            count=len(rows), class_mapping=mapping, ordered_content_sha256=sha(json_bytes(rows)),
            metadata_sidecars=sorted(sidecars, key=lambda r: r['path']))
    return all_rows, dict(root=str(root), source_manifest_sha256=sha(raw), domains=identities,
        source_validation_ineligible_for_independent_kga_check=True,
        target_images_or_labels_accessed=False)


def image_transform():
    # Source out-splits inherit augmentation from their ImageFolder environment.
    return T.Compose([T.RandomResizedCrop(224, scale=(.7, 1.)), T.RandomHorizontalFlip(),
        T.ColorJitter(.3, .3, .3, .3), T.RandomGrayscale(), T.ToTensor(),
        T.Normalize([.485, .456, .406], [.229, .224, .225])])


class SourceDataset(Dataset):
    def __init__(self, root, rows, indices):
        self.root, self.rows, self.indices = Path(root), rows, indices
        self.transform = image_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.rows[self.indices[index]]
        data = read_bytes(self.root / row['path'])
        if len(data) != row['bytes'] or sha(data) != row['sha256']:
            raise SourceError('source image identity changed')
        return self.transform(decode(data)), row['label']


class ReplaySampler:
    """Pinned cycling RandomSampler/BatchSampler distribution with workers=0.

    Materializing a local-generator cycle does not advance global augmentation RNG.
    The discarded tail is retained in state as evidence, never counted as visited.
    """
    def __init__(self, n, batch_size, *, replacement, drop_last):
        if n < 1 or batch_size < 1 or (drop_last and n < batch_size):
            raise SourceError('empty sampler/full reference batch unavailable')
        self.n, self.batch_size = n, batch_size
        self.replacement, self.drop_last = replacement, drop_last
        self.base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.cycle, self.cursor, self.order, self.seed, self.generator_state = 0, 0, [], None, None

    def _order(self, seed):
        generator = torch.Generator().manual_seed(seed)
        order = list(RandomSampler(range(self.n), replacement=self.replacement, generator=generator))
        return order, generator.get_state()

    def next(self):
        usable = self.n - self.n % self.batch_size if self.drop_last else self.n
        if not self.order or self.cursor >= usable:
            self.seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.order, self.generator_state = self._order(self.seed)
            self.cursor = 0; self.cycle += 1
        end = min(self.cursor + self.batch_size, usable)
        result = self.order[self.cursor:end]; self.cursor = end
        return result

    def state_dict(self):
        return dict(n=self.n, batch_size=self.batch_size, replacement=self.replacement,
                    drop_last=self.drop_last, base_seed=self.base_seed, cycle=self.cycle,
                    cursor=self.cursor, order=self.order[:], seed=self.seed,
                    generator_state=self.generator_state)

    def load_state_dict(self, state):
        try:
            if set(state) != set(self.state_dict()) or any(state[k] != getattr(self, k)
                    for k in ('n', 'batch_size', 'replacement', 'drop_last')):
                raise SourceError('sampler configuration mismatch')
            if any(type(state[k]) is not int or state[k] < 0 for k in ('base_seed', 'cycle', 'cursor')):
                raise SourceError('sampler integer identity invalid')
            usable = self.n - self.n % self.batch_size if self.drop_last else self.n
            if state['cycle'] == 0:
                if state['order'] or state['cursor'] or state['seed'] is not None or state['generator_state'] is not None:
                    raise SourceError('invalid initial sampler state')
            else:
                if type(state['seed']) is not int or not 0 <= state['seed'] < 2 ** 63:
                    raise SourceError('sampler cycle seed invalid')
                order, generator = self._order(state['seed'])
                if (state['order'] != order or not torch.equal(state['generator_state'], generator)
                        or not 0 < state['cursor'] <= usable
                        or (state['cursor'] != usable and state['cursor'] % self.batch_size)):
                    raise SourceError('sampler order/cursor/generator mismatch')
            for key, value in state.items(): setattr(self, key, value)
        except (KeyError, TypeError, RuntimeError) as exc:
            raise SourceError('invalid sampler state') from exc


class Featurizer(nn.Module):
    def __init__(self, backbone, dropout):
        super().__init__(); self.network = backbone; self.network.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.freeze_bn()

    def freeze_bn(self):
        for module in self.network.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    def train(self, mode=True):
        super().train(mode); self.freeze_bn(); return self

    def forward(self, x):
        return self.dropout(self.network(x))


class ERM(nn.Module):
    def __init__(self, backbone, dropout, *, features=2048, classes=7):
        super().__init__(); self.featurizer = Featurizer(backbone, dropout)
        self.classifier = nn.Linear(features, classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)

    def forward(self, x):
        return self.network(x)


def load_model(initialization, dropout):
    # Authenticating/loading tensors consumes no model RNG; construct full1000fc first.
    state = authenticated_torch_load(initialization, INITIALIZATION_SHA256)
    backbone = torchvision.models.resnet50(weights=None)
    backbone.load_state_dict(state, strict=True)
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise SourceError('nonfinite V1 initialization')
    return ERM(backbone, dropout)


def optimizer_for(model, params):
    return torch.optim.Adam(model.network.parameters(), lr=params['lr'],
                            weight_decay=params['weight_decay'], betas=(.9, .999), eps=1e-8)


def update(model, batches, optimizer, device):
    if len(batches) != 3 or len({len(y) for _, y in batches}) != 1:
        raise SourceError('ERM requires three equal-sized source batches')
    model.train(); synchronize(device); start = time.monotonic()
    x = torch.cat([x for x, _ in batches]).to(device)
    y = torch.cat([y for _, y in batches]).to(device)
    logits = model(x); loss = nn.functional.cross_entropy(logits, y)
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        raise SourceError('nonfinite source loss/logits')
    optimizer.zero_grad(); loss.backward(); optimizer.step(); synchronize(device)
    return dict(loss=float(loss.detach()), examples=len(y), wall_seconds=time.monotonic() - start)


def batch(dataset, indices):
    pairs = [dataset[i] for i in indices]
    return torch.stack([x for x, _ in pairs]), torch.tensor([y for _, y in pairs], dtype=torch.long)


def evaluate(model, dataset, sampler, device):
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for _ in range(math.ceil(len(dataset) / 64)):
            indices = sampler.next(); x, y = batch(dataset, indices)
            logits = model(x.to(device))
            if not torch.isfinite(logits).all(): raise SourceError('nonfinite source validation logits')
            correct += int((logits.argmax(1).cpu() == y).sum()); total += len(y)
    if total != len(dataset): raise SourceError('source validation coverage incomplete')
    return correct / total


def selection_steps():
    return list(range(0, STEPS, CHECKPOINT_FREQUENCY))


def score(record):
    values = list(record['out'].values())
    if len(values) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise SourceError('three finite source-out accuracies required')
    return sum(values) / 3


def best_checkpoint(records):
    return max(records, key=score)  # Earliest occurrence wins equal checkpoint scores.


def best_cell(cells):
    return max(cells, key=lambda c: (c['score'], c['hparams_seed']))


def save_state(path, model, optimizer, samplers, identity, history, evaluations, visits, device):
    synchronize(device)
    state = dict(schema='pacs_source_state_v1', identity=identity,
        algorithm={k: v.detach().cpu() for k, v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), samplers={k: v.state_dict() for k, v in samplers.items()},
        rng=rng_state(device), history=history, evaluations=evaluations, visits=visits)
    with safe_path(path).open('xb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())


def validate_adam_state(payload, model, optimizer, completed_updates):
    """Require complete bound Adam history, not merely a loadable state_dict.

    Validate serialized values before load_state_dict can cast malformed moments
    or synthesize a scalar step, and validate the restored state again afterward.
    Parameter IDs/order come from the unchanged model/optimizer construction.
    """
    if (type(optimizer) is not torch.optim.Adam or type(completed_updates) is not int
            or not 1 <= completed_updates <= STEPS):
        raise SourceError('Adam resume requires a positive exact completed-update count')
    parameters = [parameter for parameter in model.network.parameters() if parameter.requires_grad]
    live_parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if (not parameters or len(optimizer.param_groups) != 1 or len(live_parameters) != len(parameters)
            or any(actual is not expected for actual, expected in zip(live_parameters, parameters))):
        raise SourceError('Adam optimizer does not cover every expected trainable parameter in order')
    expected_groups = optimizer.state_dict()['param_groups']
    if (not isinstance(payload, dict) or set(payload) != {'state', 'param_groups'}
            or not isinstance(payload['state'], dict) or not isinstance(payload['param_groups'], list)
            or len(payload['param_groups']) != 1):
        raise SourceError('Adam optimizer state/group structure invalid')
    group, expected = payload['param_groups'][0], expected_groups[0]
    if not isinstance(group, dict) or set(group) != set(expected):
        raise SourceError('Adam optimizer parameter-group recipe keys mismatch')
    for key, value in expected.items():
        actual = group[key]
        if type(actual) is not type(value):
            raise SourceError(f'Adam optimizer parameter-group recipe type mismatch: {key}')
        if isinstance(value, (tuple, list)) and (len(actual) != len(value)
                or any(type(a) is not type(e) for a, e in zip(actual, value))):
            raise SourceError(f'Adam optimizer parameter-group element identity mismatch: {key}')
        if actual != value:
            raise SourceError(f'Adam optimizer parameter-group recipe/order mismatch: {key}')
    ids = group['params']
    if (any(type(index) is not int for index in ids)
            or len(set(ids)) != len(parameters)
            or any(type(index) is not int for index in payload['state'])
            or set(payload['state']) != set(ids)):
        raise SourceError('Adam optimizer state missing, partial, extra or misidentified parameters')
    for index, parameter in zip(ids, parameters):
        state = payload['state'][index]
        if not isinstance(state, dict) or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise SourceError('Adam optimizer per-parameter state keys mismatch')
        step = state['step']
        if (not isinstance(step, torch.Tensor) or step.layout != torch.strided
                or step.shape != torch.Size([]) or step.dtype != torch.float32
                or step.requires_grad or not bool(torch.isfinite(step))
                or step.item() != completed_updates):
            raise SourceError('Adam optimizer step must be an exact finite integral completed-update scalar')
        for key in ('exp_avg', 'exp_avg_sq'):
            moment = state[key]
            if (not isinstance(moment, torch.Tensor) or moment.layout != torch.strided
                    or moment.shape != parameter.shape or moment.dtype != parameter.dtype
                    or moment.requires_grad or not bool(torch.isfinite(moment).all())
                    or (key == 'exp_avg_sq' and not bool((moment >= 0).all()))):
                raise SourceError(f'Adam optimizer {key} shape/dtype/finiteness invalid')


def load_state(path, digest, model, optimizer, samplers, identity, device):
    state = authenticated_torch_load(path, digest)
    if state.get('schema') != 'pacs_source_state_v1' or state.get('identity') != identity:
        raise SourceError('resume configuration/input/code/runtime mismatch')
    if set(state.get('samplers', {})) != set(samplers):
        raise SourceError('resume sampler identities mismatch')
    history = state.get('history')
    if not isinstance(history, list):
        raise SourceError('Adam resume completed-update history is malformed')
    validate_adam_state(state.get('optimizer'), model, optimizer, len(history))
    for key, sampler in samplers.items(): sampler.load_state_dict(state['samplers'][key])
    model.load_state_dict(state['algorithm'], strict=True); optimizer.load_state_dict(state['optimizer'])
    validate_adam_state(optimizer.state_dict(), model, optimizer, len(history))
    set_rng(state['rng'], device)
    return state


def validate_progress(history, evaluations, visits, batch_size, memberships):
    sources = list(memberships)
    if not 1 <= len(history) <= STEPS: raise SourceError('invalid number of completed source updates')
    realized = {d: [0] * len(memberships[d]['in']) for d in sources}
    for step, record in enumerate(history):
        if (type(record.get('step')) is not int or record['step'] != step
                or type(record.get('examples')) is not int or record['examples'] != 3 * batch_size
                or type(record.get('loss')) not in (int, float) or not math.isfinite(record['loss'])
                or record['loss'] < 0):
            raise SourceError('source training history geometry invalid')
        sampled = record.get('sampled_source_in_indices', {})
        if set(sampled) != set(sources): raise SourceError('source sample trace domains missing')
        for domain in sources:
            if (len(sampled[domain]) != batch_size or any(type(i) is not int or
                    not 0 <= i < len(realized[domain]) for i in sampled[domain])):
                raise SourceError('source sample trace batch geometry invalid')
            for index in sampled[domain]: realized[domain][index] += 1
    expected_steps = [s for s in selection_steps() if s < len(history)]
    if [r.get('step') for r in evaluations] != expected_steps:
        raise SourceError('source validation cadence mismatch')
    for record in evaluations:
        if set(record.get('in', {})) != set(sources) or set(record.get('out', {})) != set(sources):
            raise SourceError('source validation domain identities mismatch')
        score(record)
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in record['in'].values()):
            raise SourceError('invalid source-in evaluation')
    if set(visits) != set(sources): raise SourceError('source visitation domain mismatch')
    for domain in sources:
        counts = visits[domain]
        if (len(counts) != len(memberships[domain]['in'])
                or any(type(v) is not int or v < 0 for v in counts)
                or sum(counts) != len(history) * batch_size or counts != realized[domain]):
            raise SourceError('source visitation count geometry mismatch')


def reference_complete(history, evaluations, visits, batch_size, memberships=None):
    if len(history) != STEPS or memberships is None: return False
    try:
        if len(memberships) != 3 or not set(memberships) < set(DOMAINS): return False
        for domain, splits in memberships.items():
            n = COUNTS[domain]
            ids = splits['in'] + splits['out']
            if (set(splits) != {'in', 'out'} or len(splits['out']) != int(.2 * n)
                    or len(ids) != n or any(type(i) is not int for i in ids)
                    or set(ids) != set(range(n))):
                return False
        validate_progress(history, evaluations, visits, batch_size, memberships)
        return True
    except (SourceError, KeyError, TypeError):
        return False


def validate_sampler_progress(samplers, memberships, updates, evaluations, batch_size):
    for domain, splits in memberships.items():
        sampler = samplers[f'train/{domain}']
        batches_per_cycle = len(splits['in']) // batch_size
        cycle = (updates - 1) // batches_per_cycle + 1
        cursor = ((updates - 1) % batches_per_cycle + 1) * batch_size
        if (sampler.cycle, sampler.cursor) != (cycle, cursor):
            raise SourceError('training sampler cycle/cursor disagrees with completed updates')
        for split in ('in', 'out'):
            sampler = samplers[f'{domain}/{split}']
            if (sampler.cycle, sampler.cursor) != (evaluations, len(splits[split])):
                raise SourceError('validation sampler cycle/cursor disagrees with checkpoint count')


def configuration(args, population, memberships):
    return dict(schema='pacs_public_reference_source_v1', reference_revision=REVISION,
        job=job(args.target, args.hparams_seed, args.trial_seed), steps=STEPS, checkpoint_frequency=300,
        initialization=str(safe_path(args.initialization)), initialization_sha256=INITIALIZATION_SHA256,
        population=population, memberships=memberships, runtime=runtime_identity(args.device), workers=0,
        code_sha256={name: file_sha(Path(__file__).with_name(name)) for name in
                     ('train_pacs.py', 'train_rxrx1.py', 'train_officehome.py')},
        historical_bitwise_equivalence_claimed=False,
        sampler_policy='RandomSampler cycles; replacement train/drop_last; random nonreplacement eval/batch64',
        source_transforms='RGB;RandomResizedCrop224 scale0.7..1;HFlip;ColorJitter0.3x4;RandomGrayscale;ImageNetNormalize;including source out',
        batchnorm_policy='eval running statistics; affine gradients enabled',
        optimizer_policy='Adam defaults; constant lr; equal three-source concatenated CE',
        selection_policy='unweighted source-out mean; earliest checkpoint; highest hparams_seed on cell ties',
        source_validation_ineligible_for_independent_kga_check=True)


def local_checkpoint(out, pointer):
    if (not isinstance(pointer, dict) or not isinstance(pointer.get('checkpoint'), str)
            or Path(pointer['checkpoint']).name != pointer['checkpoint']
            or not re.fullmatch(r'[0-9a-f]{64}', pointer.get('sha256', ''))):
        raise SourceError('invalid checkpoint pointer')
    return out / pointer['checkpoint']


def validate_reference_identity(identity):
    """Do not accept mutually consistent receipts for a different source recipe."""
    try:
        cell = identity['job']
        args = argparse.Namespace(target=cell['target'], hparams_seed=cell['hparams_seed'],
            trial_seed=cell['trial_seed'], initialization=identity['initialization'],
            device=identity['runtime']['device'])
        if identity != configuration(args, identity['population'], identity['memberships']):
            raise SourceError('source reference policy/code/runtime identity mismatch')
        sources = set(DOMAINS) - {cell['target']}
        if set(identity['population']['domains']) != sources or set(identity['memberships']) != sources:
            raise SourceError('source population domain identities invalid')
        mapping = None
        for domain in sources:
            population = identity['population']['domains'][domain]
            current = population['class_mapping']
            if (population['count'] != COUNTS[domain] or len(current) != 7
                    or current != {name: i for i, name in enumerate(sorted(current))}
                    or (mapping is not None and current != mapping)
                    or identity['memberships'][domain] != partition(COUNTS[domain], cell['trial_seed'], DOMAINS.index(domain))):
                raise SourceError('source population class/count/seeded membership mismatch')
            mapping = current
    except (KeyError, TypeError, AttributeError) as exc:
        raise SourceError('malformed source reference identity') from exc


def execute(args):
    if not args.resume and safe_path(args.output_dir).exists():
        raise SourceError('fresh output directory required')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise SourceError('requested MPS hardware unavailable')
    cell = job(args.target, args.hparams_seed, args.trial_seed)
    rows, population = accept_population(args.data_root, args.source_manifest, args.source_sha256, args.target)
    memberships = {d: partition(len(rows[d]), args.trial_seed, DOMAINS.index(d)) for d in rows}
    identity = configuration(args, population, memberships)
    out = prepare_output(args.output_dir, identity, resume=args.resume)
    with safe_path(out / 'run.lock').open('a+b') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc: raise SourceError('source output already owned by a running process') from exc
        if (out / 'completion.json').exists(): raise SourceError('source run already complete')
        try:
            random.seed(cell['seed']); np.random.seed(cell['seed']); torch.manual_seed(cell['seed'])
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
            b = cell['hparams']['batch_size']
            datasets = {f'{d}/{split}': SourceDataset(args.data_root, rows[d], memberships[d][split])
                        for split in ('in', 'out') for d in rows}
            # Preserve source-only workers0 iterator construction order before model RNG.
            samplers = {f'train/{d}': ReplaySampler(len(datasets[f'{d}/in']), b, replacement=True, drop_last=True) for d in rows}
            samplers.update({f'{d}/{split}': ReplaySampler(len(datasets[f'{d}/{split}']), 64,
                replacement=False, drop_last=False) for split in ('in', 'out') for d in rows})
            model = load_model(args.initialization, cell['hparams']['resnet_dropout']).to(args.device)
            optimizer = optimizer_for(model, cell['hparams'])
            history, evaluations = [], []
            visits = {d: [0] * len(memberships[d]['in']) for d in rows}
            checkpoints = []
            eligible_source = dict(rows=rows, memberships=memberships,
                note='All eligible IDs; replacement sampling does not guarantee visitation. Source out excluded from independent KGA.')
            if args.resume:
                if read_json(out / 'eligible-source.json') != eligible_source:
                    raise SourceError('resume eligible source IDs/content/membership mismatch')
                journal = read_json(out / 'resume.json')
                state = load_state(local_checkpoint(out, journal), journal['sha256'], model, optimizer,
                                   samplers, identity, args.device)
                history, evaluations, visits = state['history'], state['evaluations'], state['visits']
                validate_progress(history, evaluations, visits, b, memberships)
                validate_sampler_progress(samplers, memberships, len(history), len(evaluations), b)
                checkpoints = journal['checkpoints']
                verify_checkpoints(out, checkpoints, evaluations, identity, history)
                if journal.get('step') != len(history) - 1:
                    raise SourceError('resume pointer update count mismatch')
            else:
                write_json(out / 'eligible-source.json', eligible_source)
            for step in range(len(history), STEPS):
                batches, sampled = [], {}
                for domain in rows:
                    indices = samplers[f'train/{domain}'].next(); sampled[domain] = indices
                    batches.append(batch(datasets[f'{domain}/in'], indices))
                record = update(model, batches, optimizer, args.device); record['step'] = step
                record['sampled_source_in_indices'] = sampled
                history.append(record)
                for domain, indices in sampled.items():
                    for index in indices: visits[domain][index] += 1
                eligible = step % CHECKPOINT_FREQUENCY == 0
                if eligible:
                    result = {'step': step, 'in': {}, 'out': {}}
                    for split in ('in', 'out'):
                        for domain in rows:
                            key = f'{domain}/{split}'
                            result[split][domain] = evaluate(model, datasets[key], samplers[key], args.device)
                    evaluations.append(result)
                if eligible or step == STEPS - 1:
                    validate_progress(history, evaluations, visits, b, memberships)
                    validate_sampler_progress(samplers, memberships, len(history), len(evaluations), b)
                    name = f'step-{step:04d}-{uuid.uuid4().hex}.pt'
                    save_state(out / name, model, optimizer, samplers, identity, history, evaluations, visits, args.device)
                    pointer = dict(checkpoint=name, sha256=file_sha(out / name), step=step)
                    if eligible: checkpoints.append(pointer)
                    write_json(out / 'resume.json', dict(**pointer, checkpoints=checkpoints), replace=True)
                    write_json(out / 'progress.json', dict(updates=len(history), last_train=record,
                        best_step=best_checkpoint(evaluations)['step'], source_complete=False), replace=True)
            if not reference_complete(history, evaluations, visits, b, memberships):
                raise SourceError('full reference source completion gate failed')
            winner = best_checkpoint(evaluations)
            selected = next(c for c in checkpoints if c['step'] == winner['step'])
            write_json(out / 'completion.json', dict(schema='pacs_source_completion_v1', complete=True,
                source_training_complete=True, identity=identity, configuration_sha256=file_sha(out / 'configuration.json'),
                eligible_source_sha256=file_sha(out / 'eligible-source.json'), updates=STEPS,
                last=pointer, checkpoints=checkpoints, best=selected, source_selection_score=score(winner),
                evaluations=evaluations, history=history, visits=visits, target_evaluation_performed=False))
        except BaseException as exc:
            write_json(out / ('failure-' + uuid.uuid4().hex + '.json'), dict(complete=False,
                source_training_complete=False, error=type(exc).__name__, message=str(exc),
                resume_only_from_last_authenticated_checkpoint=True))
            raise


def verify_checkpoints(out, checkpoints, evaluations, identity, history):
    if [p.get('step') for p in checkpoints] != [e['step'] for e in evaluations]:
        raise SourceError('checkpoint/selection cadence mismatch')
    for i, pointer in enumerate(checkpoints):
        state = authenticated_torch_load(local_checkpoint(out, pointer), pointer['sha256'])
        if (state.get('schema') != 'pacs_source_state_v1' or state.get('identity') != identity
                or state.get('evaluations') != evaluations[:i + 1]
                or state.get('history') != history[:pointer['step'] + 1]):
            raise SourceError('selection checkpoint provenance mismatch')


def select(args):
    if safe_path(args.output_dir).exists(): raise SourceError('fresh output directory required')
    raw = read_bytes(args.runs_manifest)
    if sha(raw) != args.runs_sha256: raise SourceError('runs manifest SHA256 mismatch')
    import json
    receipts = json.loads(raw)
    if not isinstance(receipts, list) or len(receipts) != 20:
        raise SourceError('selection requires exactly twenty completed source cells')
    cells, seen, common = [], set(), None
    for entry in receipts:
        if not isinstance(entry, dict) or set(entry) != {'completion', 'sha256'}:
            raise SourceError('completion manifest entry invalid')
        receipt_path = safe_path(entry['completion']); data = read_bytes(receipt_path)
        if sha(data) != entry['sha256']: raise SourceError('source completion SHA256 mismatch')
        receipt = json.loads(data); identity = receipt['identity']; cell = identity['job']
        validate_reference_identity(identity)
        if (receipt.get('schema') != 'pacs_source_completion_v1' or receipt.get('complete') is not True
                or receipt.get('source_training_complete') is not True
                or receipt.get('target_evaluation_performed') is not False
                or cell != job(args.target, cell['hparams_seed'], args.trial_seed)
                or cell['hparams_seed'] in seen):
            raise SourceError('completed source grid identity invalid')
        seen.add(cell['hparams_seed'])
        shared = {k: v for k, v in identity.items() if k != 'job'}
        if common is not None and shared != common: raise SourceError('source cells disagree on population/configuration/runtime')
        common = shared
        directory = receipt_path.parent
        if (read_json(directory / 'configuration.json') != identity
                or file_sha(directory / 'configuration.json') != receipt['configuration_sha256']
                or file_sha(directory / 'eligible-source.json') != receipt['eligible_source_sha256']
                or not reference_complete(receipt['history'], receipt['evaluations'], receipt['visits'],
                                          cell['hparams']['batch_size'], identity['memberships'])):
            raise SourceError('source completion evidence invalid')
        verify_checkpoints(directory, receipt['checkpoints'], receipt['evaluations'], identity, receipt['history'])
        last = authenticated_torch_load(local_checkpoint(directory, receipt['last']), receipt['last']['sha256'])
        if (last.get('identity') != identity or last.get('history') != receipt['history']
                or last.get('evaluations') != receipt['evaluations'] or last.get('visits') != receipt['visits']):
            raise SourceError('final source checkpoint disagrees with completion')
        winner = best_checkpoint(receipt['evaluations'])
        best = next(c for c in receipt['checkpoints'] if c['step'] == winner['step'])
        if receipt['best'] != best or receipt['source_selection_score'] != score(winner):
            raise SourceError('source selection pointer disagrees with source metrics')
        cells.append(dict(hparams_seed=cell['hparams_seed'], score=score(winner), best=best,
                          directory=str(directory), completion_sha256=sha(data), identity=identity))
    if seen != set(range(20)): raise SourceError('full twenty-cell source grid missing')
    winner = best_cell(cells)
    state = authenticated_torch_load(local_checkpoint(Path(winner['directory']), winner['best']), winner['best']['sha256'])
    out = prepare_output(args.output_dir, dict(schema='pacs_source_selection_v1', target=args.target,
        trial_seed=args.trial_seed, runs_manifest_sha256=sha(raw)), resume=False)
    with (out / 'selected-model.pt').open('xb') as f:
        torch.save(dict(schema='pacs_source_selected_model_v1', algorithm=state['algorithm'],
                        identity=winner['identity'], selected_step=winner['best']['step']), f)
        f.flush(); os.fsync(f.fileno())
    write_json(out / 'selection.json', dict(complete=True, source_grid_complete=True, target=args.target,
        trial_seed=args.trial_seed, candidates=cells, winner=winner, model='selected-model.pt',
        model_sha256=file_sha(out / 'selected-model.pt'), target_evaluation_performed=False))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    subs = result.add_subparsers(dest='command', required=True)
    plan = subs.add_parser('plan'); plan.add_argument('--output-dir', required=True)
    train = subs.add_parser('train')
    for name in ('data-root', 'source-manifest', 'source-sha256', 'initialization', 'output-dir'):
        train.add_argument('--' + name, required=True)
    train.add_argument('--target', choices=DOMAINS, required=True)
    train.add_argument('--hparams-seed', type=int, choices=range(20), required=True)
    train.add_argument('--trial-seed', type=int, choices=range(3), required=True)
    train.add_argument('--device', choices=('cpu', 'mps'), required=True)
    train.add_argument('--resume', action='store_true')
    selection = subs.add_parser('select')
    for name in ('runs-manifest', 'runs-sha256', 'output-dir'): selection.add_argument('--' + name, required=True)
    selection.add_argument('--target', choices=DOMAINS, required=True)
    selection.add_argument('--trial-seed', type=int, choices=range(3), required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'plan':
            out = prepare_output(args.output_dir, dict(schema='pacs_source_plan_v1', revision=REVISION), resume=False)
            write_json(out / 'plan.json', dict(jobs=jobs(), full_source_updates=STEPS,
                source_runs=240, selected_models=12, source_training_complete=False))
        elif args.command == 'train': execute(args)
        else: select(args)
        return 0
    except (SourceError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f'PACS source stage failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
