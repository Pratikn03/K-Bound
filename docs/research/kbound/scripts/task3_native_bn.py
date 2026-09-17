#!/usr/bin/env python3
"""Shared full-BN POEM trajectory and native AETTA measurements.

Executes authenticated computational definitions through a disclosed CPU driver,
not unchanged upstream main programs. No target outcomes or gate fitting occur
here. A common-panel controller must bind these records before scoring them.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import types

import task3_image_panel as images
import run_poem_real_image_smoke as poem_io
import run_aetta_real_image_smoke as aetta_io
import aetta_native_estimator as aetta

FIELDS = {'schema', 'scope', 'clean_root', 'corruption_root', 'condition', 'seed',
          'batch_size', 'source', 'target'}


def validate_panel(panel):
    if not isinstance(panel, dict) or set(panel) != FIELDS:
        raise ValueError('exact cell fields required; outcomes forbidden')
    if (panel['schema'] != 'kbound-native-bn-cell-v1'
            or panel['scope'] not in ('ENGINEERING_SMOKE_NOT_BENCHMARK', 'LOCKED_DEVELOPMENT_CELL')
            or type(panel['seed']) is not int or not 0 <= panel['seed'] < 2**31
            or type(panel['batch_size']) is not int or panel['batch_size'] != 4
            or panel['condition'] not in {f'{c}/{s}' for c in ('gaussian_noise', 'shot_noise', 'impulse_noise')
                                         for s in (1, 3, 5)}):
        raise ValueError('unsupported locked cell recipe')
    for root in ('clean_root', 'corruption_root'):
        if not isinstance(panel[root], str) or not Path(panel[root]).is_absolute():
            raise ValueError('absolute dataset root required')
    identities = []
    for role in ('source', 'target'):
        rows = panel[role]
        minimum = 2 if role == 'source' else 101
        maximum = 12500 if role == 'source' else 4000
        if not isinstance(rows, list) or not minimum <= len(rows) <= maximum:
            raise ValueError('invalid role count or unpassed native warmup')
        if len(rows) % 4:
            raise ValueError('complete batch-of-four cells required')
        for row in rows:
            if not isinstance(row, dict) or set(row) != poem_io.ROW_FIELDS:
                raise ValueError('exact image fields required; outcomes forbidden')
            prefix = '' if role == 'source' else panel['condition']+'/'
            relative = row['relative_path']
            if (not isinstance(relative, str) or not relative.startswith(prefix)
                    or not images.NAME.fullmatch(relative[len(prefix):])
                    or Path(relative).stem != row['sample_id']):
                raise ValueError('image condition or sample identity mismatch')
            if (not isinstance(row['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['sha256'])
                    or type(row['size_bytes']) is not int or not 0 < row['size_bytes'] <= 32*1024**2
                    or any(type(row[k]) is not int or row[k] < 1 for k in ('width', 'height'))
                    or row['decoded_rgb'] is not True):
                raise ValueError('invalid bound image metadata')
            identities.append(row['sample_id'])
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate or overlapping source/target identities')


def read_panel(path, expected):
    data = Path(path).read_bytes()
    if images.digest(data) != expected:
        raise ValueError('cell manifest digest mismatch')
    value = json.loads(data, object_pairs_hook=poem_io.unique_object)
    validate_panel(value)
    return value


def bn_recipe(batch_size):
    if type(batch_size) is not int or batch_size != 4:
        raise ValueError('reviewed BN driver requires batch4')
    return dict(temperature=1.0, lr=.00025/64*batch_size*2, momentum=.9,
                native_warmup_samples=100, dropout_draws=10, dropout_probability=.5,
                e0=.4*math.log(1000), gamma=1/(8*math.sqrt(3)), eps_clip=1.8,
                vanilla_loss=True, steps=1,
                prediction_timing='prequential native POEM return before current optimizer step',
                reset='fresh model, optimizer, protector, AETTA EMA and RNG states per cell',
                checkpoint_policy='authenticated torchvision BN V2, NOT upstream original BN default',
                controller='none: produces paired predictions and native estimates, not native gate actions')


def isolated_estimate(function, *, seed):
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return function()


class EstimatorStream:
    """Private continuous CPU RNG per estimator; adapter RNG is restored."""
    def __init__(self, episode, seed):
        import torch
        self.episode = episode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.state = torch.get_rng_state().clone()

    def estimate(self, batch):
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.state)
            result = self.episode.estimate(batch)
            self.state = torch.get_rng_state().clone()
        return result


def checked_estimate(stream, model, batch, expected_logits):
    import torch
    calls, deterministic = [], []
    def observe(module, inputs, kwargs, output):
        probability = kwargs.get('dropout', 0.)
        calls.append(probability)
        if probability == 0.: deterministic.append(output.detach())
    handle = model.register_forward_hook(observe, with_kwargs=True)
    try:
        result = stream.estimate(batch)
    finally:
        handle.remove()
    if calls != [0.] + [.5]*10 or len(deterministic) != 1:
        raise ValueError('native AETTA requires one deterministic and ten dropout forwards')
    if not torch.equal(deterministic[0], expected_logits):
        raise ValueError('AETTA deterministic prediction differs from shared candidate prediction')
    return result


def finite_tree(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray): return bool(np.isfinite(value).all())
    if isinstance(value, dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list,tuple)): return all(finite_tree(v) for v in value)
    if value is None: return True
    return bool(np.isfinite(value))


def observe_updates(native_module, traces, *, allow_empty_selection=False):
    """Observe unchanged native computation; never repair or skip its loss."""
    import torch
    original = native_module.forward_and_adapt
    def wrapped(*args, **kwargs):
        optimizer = args[2] if len(args)>2 else None
        parameters = [] if optimizer is None else [p for group in optimizer.param_groups for p in group['params']]
        before = [p.detach().clone() for p in parameters]
        output, loss = original(*args, **kwargs)
        finite = bool(torch.isfinite(loss).all())
        selected = int((-(output.softmax(1)*output.log_softmax(1)).sum(1)
                        < kwargs.get('e_margin',.4*math.log(1000))).sum())
        classified = (not finite and bool(torch.isnan(loss)) and kwargs.get('vanilla_loss') is True
                      and selected==0 and bool(torch.isfinite(output).all()))
        traces.append({'finite': finite, 'loss': float(loss.detach()) if finite else None,
                       'selected_count': selected,
                       'classification': 'empty_reliable_selection_native_nan' if classified else 'finite' if finite else 'unexplained_nonfinite',
                       'changed_parameter_tensors':sum(not torch.equal(p,old) for p,old in zip(parameters,before))})
        if optimizer is not None and not finite_tree(optimizer.state):
            raise ValueError('nonfinite native optimizer state')
        if not finite_tree(parameters) or not finite_tree([p.grad for p in parameters]):
            raise ValueError('nonfinite native model or gradient state')
        model = args[1] if len(args)>1 else None
        if model is not None and not finite_tree([p.grad for p in model.parameters()]):
            raise ValueError('nonfinite native gradient outside optimizer parameter set')
        if not finite and not (allow_empty_selection and classified):
            raise ValueError('nonfinite native update loss; upstream numerical outcome retained as failure')
        return output, loss
    native_module.forward_and_adapt = wrapped


def tensor_record(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': images.digest(value.detach().cpu().contiguous().numpy().tobytes())}


def mutation_signature(model):
    """Include parameters, buffers, gradients, BN flags and modes."""
    entries = {'state': aetta_io.state_digest(model),
               'gradients': {name: None if p.grad is None else tensor_record(p.grad)
                             for name, p in model.named_parameters()},
               'modules': {name: [m.training, getattr(m, 'track_running_stats', None)]
                           for name, m in model.named_modules()},
               'requires_grad': {name: p.requires_grad for name, p in model.named_parameters()}}
    return images.digest(json.dumps(entries, sort_keys=True).encode())


def execute(args, panel, record):
    import numpy as np
    import torch
    import torchvision
    from torchvision import transforms
    if (torch.__version__, torchvision.__version__) != ('2.1.1', '0.16.1'):
        raise ValueError('shared BN native execution requires prepared POEM torch2.1.1/torchvision0.16.1')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(panel['seed'])
    np.random.seed(panel['seed'])
    source = aetta.NativeAETTASource(args.aetta_source, auth_receipt=args.auth_receipt)
    native = poem_io.load_native(args.poem_source, record)
    record['native_update_losses'] = []
    observe_updates(native['poem'], record['native_update_losses'],allow_empty_selection=args.record_native_empty_selection)
    record['instrumentation'] = 'observe unchanged forward_and_adapt return and selected count; preserve backward/SGD/zero_grad exactly'
    record['numerical_policy'] = ('explicit empty reliable-set NaN warning; all predictions and continuing states must remain finite'
                                  if args.record_native_empty_selection else 'abort on any nonfinite native loss')
    weights = aetta_io.read_checkpoint(args.checkpoint)
    state = torch.load(io.BytesIO(weights), map_location='cpu', weights_only=True)
    frozen = source.create_model()
    frozen.load_state_dict(state, strict=True)
    del state, weights
    candidate = deepcopy(frozen)
    normalize = source.new_episode(frozen)._native.net[0]
    recipe = bn_recipe(panel['batch_size'])
    record.update(recipe=recipe, aetta_source=source.provenance,
                  checkpoint_sha256=aetta_io.CHECKPOINT_SHA256,
                  checkpoint_path=str(args.checkpoint), model='full_ResNet50_BN_V2',
                  model_parameters=sum(p.numel() for p in frozen.parameters()),
                  runtime={'python': sys.version, 'torch': torch.__version__, 'torchvision': torchvision.__version__},
                  original_main_executed=False, cuda_parity_verified=False,
                  checkpoint_default_reproduction=False,
                  sample_schedule='ordered locked IDs; no data shuffle; private continuous AETTA RNG',
                  preprocessing={'source':'PILRGB Resize256 bilinear CenterCrop224 ToTensor nativeImageNetNorm',
                                 'target':'PILRGB CenterCrop224 ToTensor nativeImageNetNorm'})
    raw_transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
    clean_transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                                         transforms.CenterCrop(224), transforms.ToTensor()])
    def batches(role):
        root = panel['clean_root' if role == 'source' else 'corruption_root']
        transform = clean_transform if role == 'source' else raw_transform
        for offset in range(0, len(panel[role]), 4):
            rows = panel[role][offset:offset+4]
            yield rows, torch.stack([transform(images.read_bound_image(root, row)) for row in rows])
            poem_io.check_resources()
    entropy = native['poem'].softmax_entropy
    source_ents = []
    source_classes = []
    started = time.monotonic()
    with torch.no_grad():
        for rows, raw in batches('source'):
            out = frozen(normalize(raw))
            source_ents.extend(entropy(out).tolist())
            source_classes.extend(out.argmax(1).tolist())
    record['source_seconds'] = time.monotonic()-started
    if len(set(source_ents)) < 2 or not np.isfinite(source_ents).all():
        raise ValueError('nonfinite/constant native entropy reference')
    before_frozen = mutation_signature(frozen)
    candidate = native['sar'].configure_model(candidate).eval()
    params, names = native['sar'].collect_params(candidate)
    optimizer = torch.optim.SGD(params, lr=recipe['lr'], momentum=.9)
    protector = native['protector'].get_protector_from_ents(source_ents,
                 types.SimpleNamespace(gamma=recipe['gamma'], eps_clip=recipe['eps_clip'], device='cpu'))
    adapter = native['poem'].POEM(candidate, optimizer, protector, e0=recipe['e0'], vanilla_loss=True).eval()
    frozen_est = EstimatorStream(source.new_episode(frozen), panel['seed']+100000)
    candidate_est = EstimatorStream(source.new_episode(candidate), panel['seed']+200000)
    record['parameter_names'] = names
    record['candidate_initial_state_sha256'] = aetta_io.state_digest(candidate)
    outputs0, outputs1, estimates, seen = [], [], [], []
    started = time.monotonic()
    for batch_index, (rows, raw) in enumerate(batches('target')):
        normalized = normalize(raw)
        with torch.no_grad():
            before = candidate(normalized).detach()
            frozen_logits = frozen(normalized).detach()
        if not args.without_estimators:
            signature = mutation_signature(candidate)
            estimate0 = checked_estimate(frozen_est, frozen, raw, frozen_logits)
            estimate1 = checked_estimate(candidate_est, candidate, raw, before)
            if mutation_signature(candidate) != signature:
                raise ValueError('AETTA changed candidate parameters, buffers, gradients or modes')
            estimates.append({'batch_index': batch_index, 'sample_ids':[r['sample_id'] for r in rows],
                              'frozen':estimate0, 'candidate':estimate1})
        with torch.no_grad():
            out = adapter(normalized).detach()
        if not torch.equal(out, before):
            raise ValueError('native POEM return differs from same-state pre-update prediction')
        if not torch.isfinite(out).all() or not torch.isfinite(frozen_logits).all():
            raise ValueError('nonfinite logits')
        if not all(torch.isfinite(p).all() for p in candidate.parameters()):
            raise ValueError('nonfinite candidate parameters')
        if not finite_tree([list(candidate.buffers()),optimizer.state,protector.export_info(),protector.gradients]):
            raise ValueError('nonfinite native continuing buffer, optimizer or protector state')
        outputs0.append(frozen_logits.numpy()); outputs1.append(out.numpy())
        seen.extend(r['sample_id'] for r in rows)
        print(json.dumps({'stage':'shared_native_bn', 'complete':len(seen),'total':len(panel['target'])}),flush=True)
    record.update(target_seconds=time.monotonic()-started,
                  candidate_final_state_sha256=aetta_io.state_digest(candidate),
                  changed_parameter_tensors=sum(not torch.equal(p, adapter.model_state[n]) for n,p in candidate.named_parameters()),
                  frozen_state_unchanged=mutation_signature(frozen)==before_frozen,
                  processed_ids=seen, native_optimizer_populated=bool(optimizer.state),
                  protector_count=len(protector.info['z_before']),
                  native_counter=adapter.curr_n_samples,
                  estimator_execution=not args.without_estimators)
    if (record['frozen_state_unchanged'] is not True
            or seen != [r['sample_id'] for r in panel['target']]
            or adapter.curr_n_samples != len(seen) or record['protector_count'] != len(seen)
            or not record['native_optimizer_populated']):
        raise ValueError('shared execution lifecycle check failed')
    buffer=io.BytesIO()
    np.savez_compressed(buffer, frozen=np.concatenate(outputs0), candidate=np.concatenate(outputs1),
                        sample_ids=np.array(seen), source_classes=np.array(source_classes,dtype=np.int64))
    record['outputs'] = {
        'predictions.npz': poem_io.write_artifact(args.output_fd,'predictions.npz',buffer.getvalue()),
        'aetta_estimates.json': poem_io.write_artifact(args.output_fd,'aetta_estimates.json',
                             (json.dumps(estimates,indent=2,allow_nan=False)+'\n').encode())}
    warnings=sum(not entry['finite'] for entry in record['native_update_losses'])
    record['native_numerical_warning_batches']=warnings
    record['status']=('COMPLETED_WITH_NATIVE_NUMERICAL_WARNING' if warnings else 'PASS_SHARED_NATIVE_BN_EXECUTION')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','checkpoint','poem-source','aetta-source','auth-receipt','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--without-estimators',action='store_true',help='explicit POEM-only conformance arm, not full comparison')
    parser.add_argument('--record-native-empty-selection',action='store_true',
                        help='explicit protocol amendment: retain native empty-reliable-set NaN loss as warning; never skip or modify update')
    parser.add_argument('--deadline-seconds',type=int,default=600)
    args=parser.parse_args()
    if not 1 <= args.deadline_seconds <= 86400:
        parser.error('deadline must be between1 and86400seconds')
    runtime=os.environ.get('POEM_PYTHON')
    if not runtime or Path(runtime).resolve()!=Path(sys.executable).resolve():
        parser.error('POEM_PYTHON must explicitly match the executing interpreter')
    panel=read_panel(args.manifest,args.manifest_sha256)
    for root in (panel['clean_root'],panel['corruption_root'],args.poem_source,args.aetta_source):
        if Path(root).resolve() in (args.output.resolve(),*args.output.resolve().parents):
            parser.error('output must be outside data and native sources')
    fd=images._open_directory_chain(args.output.parent)
    try:
        os.mkdir(args.output.name,mode=0o700,dir_fd=fd)
        args.output_fd=os.open(args.output.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
    finally: os.close(fd)
    record={'schema':'kbound-shared-native-bn-execution-v1','status':'INCOMPLETE',
            'scope':panel['scope'],'manifest_sha256':args.manifest_sha256,
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'harness_sha256':images.digest(Path(__file__).read_bytes()),
            'helpers':{Path(m.__file__).name:images.digest(Path(m.__file__).read_bytes())
                       for m in (images,poem_io,aetta_io,aetta)},
            'target_outcomes_read':False,'benchmark_completed':False,'no_automatic_downloads':True}
    start=time.monotonic()
    def timeout(signum,frame): raise TimeoutError('declared cell deadline exceeded')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(args.deadline_seconds)
    try:
        execute(args,panel,record)
        actual, expected = args.output.lstat(), os.fstat(args.output_fd)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('output pathname replaced; retained descriptor preserves original artifacts')
    except Exception as exc:
        record.update(status='FAILED_SHARED_NATIVE_BN_EXECUTION',error=f'{type(exc).__name__}: {exc}')
    finally:
        signal.alarm(0)
        record['elapsed_seconds']=time.monotonic()-start
        poem_io.write_artifact(args.output_fd,'receipt.json',(json.dumps(record,indent=2,allow_nan=False)+'\n').encode())
        os.close(args.output_fd)
    print(json.dumps({'status':record['status'],'output':str(args.output)}),flush=True)
    return 0 if record['status'] in ('PASS_SHARED_NATIVE_BN_EXECUTION','COMPLETED_WITH_NATIVE_NUMERICAL_WARNING') else 2


if __name__=='__main__': raise SystemExit(main())
