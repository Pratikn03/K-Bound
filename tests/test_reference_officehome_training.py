import copy
import io
import os
import errno
import struct
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from experiments.kbound.reference_source import train_officehome as m


def fixture_source(tmp_path, per_class=5):
    root = tmp_path / 'data'
    lines = []
    for c in range(2):
        folder = root / 'Art' / f'class{c}'
        folder.mkdir(parents=True)
        for i in range(per_class):
            path = folder / f'{i}.png'
            Image.new('L', (32, 40), 30 + i).save(path)
            lines.append(f'Art/class{c}/{i}.png {c}\n')
    listing = tmp_path / 'source.txt'
    listing.write_text(''.join(lines))
    return root, listing


def toy():
    return torch.nn.ModuleDict({k: torch.nn.Linear(3, 3) for k in ('F', 'B', 'C')})


def test_inventory_split_identity_and_rgb(tmp_path):
    root, listing = fixture_source(tmp_path)
    rows, identity = m.inventory(root, 'Art', listing, m.file_sha(listing), expected_count=10, classes=2)
    torch.manual_seed(9)
    split = m.partition(rows)
    assert len(split['train']) == 9 and len(split['val']) == 1
    assert set(split['train']).isdisjoint(split['val'])
    assert identity['class_mapping'] == {'class0': 0, 'class1': 1}
    x, y = m.SourceDataset(root, rows, split['val'], training=False)[0]
    assert x.shape == (3, 224, 224) and y in (0, 1)


@pytest.mark.parametrize('kind', ['wrong_hash', 'duplicate', 'wrong_domain', 'bad_mapping', 'extra', 'symlink', 'corrupt'])
def test_inventory_fail_closed(tmp_path, kind):
    root, listing = fixture_source(tmp_path)
    digest = m.file_sha(listing)
    if kind == 'duplicate':
        listing.write_text(listing.read_text() + listing.read_text().splitlines()[0] + '\n')
    elif kind == 'wrong_domain':
        listing.write_text(listing.read_text().replace('Art/', 'Product/'))
    elif kind == 'bad_mapping':
        listing.write_text(listing.read_text().replace('class1/0.png 1', 'class1/0.png 0'))
    elif kind == 'extra':
        Image.new('RGB', (20, 20)).save(root / 'Art/class0/extra.png')
    elif kind == 'symlink':
        p = root / 'Art/class0/0.png'
        p.unlink(); p.symlink_to(root / 'Art/class0/1.png')
    elif kind == 'corrupt':
        (root / 'Art/class0/0.png').write_bytes(b'broken')
    if kind != 'wrong_hash':
        digest = m.file_sha(listing)
    else:
        digest = '0' * 64
    with pytest.raises(m.SourceError):
        m.inventory(root, 'Art', listing, digest, expected_count=10, classes=2)


def test_optimizer_schedule_and_smoothing():
    model = toy()
    opt = m.optimizer_for(model)
    assert len(opt.param_groups) == 6
    m.schedule(opt, 1, 50)
    factor = (1 + 10 / 50) ** -.75
    assert opt.param_groups[0]['lr'] == pytest.approx(.001 * factor)
    assert opt.param_groups[-1]['lr'] == pytest.approx(.01 * factor)
    assert all(g['momentum'] == .9 and g['nesterov'] and g['weight_decay'] == .001 for g in opt.param_groups)
    logits = torch.tensor([[2., 0., -1.]])
    expected = -(torch.tensor([[.9 + .1/3, .1/3, .1/3]]) * logits.log_softmax(1)).sum()
    assert m.smooth_loss(logits, torch.tensor([0])) == expected


def test_actual_update_resume_and_rng(tmp_path):
    torch.manual_seed(23)
    model = toy(); opt = m.optimizer_for(model)
    batch = [(torch.randn(4, 3), torch.tensor([0, 1, 2, 1]))]
    initial = copy.deepcopy(model.state_dict())
    metrics, step = m.run_epoch(model, batch, 'cpu', optimizer=opt, step=0, total_steps=50)
    assert step == 1 and metrics['examples'] == 4
    assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
    path = tmp_path / 'epoch.pt'
    m.save_state(path, model, opt, {'fixture': True}, [{'epoch': 0}], 1, 'cpu')
    expected_random = torch.rand(5)
    m.run_epoch(model, batch, 'cpu', optimizer=opt, step=1, total_steps=50)
    expected = copy.deepcopy(model.state_dict())
    restored = toy(); restored_opt = m.optimizer_for(restored)
    history, step = m.restore_state(path, m.file_sha(path), restored, restored_opt, {'fixture': True}, 'cpu')
    assert torch.equal(torch.rand(5), expected_random)
    m.run_epoch(restored, batch, 'cpu', optimizer=restored_opt, step=step, total_steps=50)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in expected.items())
    assert history == [{'epoch': 0}]
    with pytest.raises(m.SourceError, match='identity'):
        m.restore_state(path, m.file_sha(path), restored, restored_opt, {}, 'cpu')


def test_selection_cadence_latest_tie_and_no_false_completion():
    history = [{'epoch': i, 'val': {'accuracy': .5}} for i in range(10)]
    assert m.selected_epoch(history) == 9
    assert m.selected_epoch(history[:4]) is None
    assert not m.reference_complete(history, 'Art')
    good = [{'epoch': i, 'train': {'examples': 2184, 'batches': 35}, 'val': {'examples': 243, 'batches': 4}} for i in range(50)]
    assert m.reference_complete(good, 'Art')
    good[20]['train']['examples'] -= 1
    assert not m.reference_complete(good, 'Art')


def test_fresh_output_and_fifo(tmp_path):
    out = tmp_path / 'existing'; out.mkdir()
    marker = out / 'keep'; marker.write_text('preserved')
    with pytest.raises(m.SourceError):
        m.prepare_output(out, {}, resume=False)
    assert marker.read_text() == 'preserved'
    fifo = tmp_path / 'fifo'
    try:
        os.mkfifo(fifo)
    except OSError as exc:
        if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
            pytest.skip('fixture filesystem does not support FIFO creation')
        raise
    with pytest.raises(m.SourceError):
        m.read_bytes(fifo)


def test_singleton_and_nonfinite_fail():
    model = toy(); opt = m.optimizer_for(model)
    with pytest.raises(m.SourceError, match='singleton'):
        m.run_epoch(model, [(torch.zeros(1, 3), torch.tensor([0]))], 'cpu', optimizer=opt, total_steps=50)
    with pytest.raises(m.SourceError, match='nonfinite'):
        m.run_epoch(model, [(torch.full((2, 3), float('nan')), torch.tensor([0, 1]))], 'cpu')


def test_actual_v1_composite_construction_and_bottleneck_semantics():
    path = os.getenv('KBOUND_TEST_V1_INITIALIZATION')
    if not path:
        pytest.skip('optional real initializer not supplied')
    model = m.load_model(path)
    assert model['B'].bottleneck.in_features == 2048
    assert model['B'].bottleneck.out_features == 256
    assert set(model['C'].state_dict()) == {'fc.bias', 'fc.weight_g', 'fc.weight_v'}
    assert model['C'].fc.out_features == 65
    x = torch.randn(2, 2048)
    model.eval()
    expected = model['B'].bn(model['B'].bottleneck(x))
    assert torch.equal(model['B'](x), expected)
    assert torch.equal(model['B'](x), model['B'](x))
    assert model['C'](expected).shape == (2, 65)


def test_snapshot_immutable_and_selection_provenance(tmp_path):
    model = toy(); opt = m.optimizer_for(model)
    history = [{'epoch': i, 'val': {'accuracy': .5}} for i in range(5)]
    path = tmp_path / 'epoch.pt'
    m.save_state(path, model, opt, {'fixture': True}, history, 5, 'cpu')
    digest = m.file_sha(path)
    pointer = {'epoch': 4, 'checkpoint': path.name, 'sha256': digest, 'val_accuracy': .5}
    with torch.no_grad():
        next(model.parameters()).add_(100)
    assert m.file_sha(path) == digest
    m.verify_best(tmp_path, pointer, history, {'fixture': True})
    pointer['epoch'] = 0
    with pytest.raises(m.SourceError, match='selection'):
        m.verify_best(tmp_path, pointer, history, {'fixture': True})


def test_resume_rejects_changed_checkpoint(tmp_path):
    model = toy(); opt = m.optimizer_for(model)
    path = tmp_path / 'epoch.pt'
    m.save_state(path, model, opt, {}, [], 0, 'cpu')
    digest = m.file_sha(path)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(m.SourceError, match='hash'):
        m.restore_state(path, digest, model, opt, {}, 'cpu')


def test_cli_failure_receipt_and_explicit_epoch_resume(tmp_path, monkeypatch):
    root, listing = fixture_source(tmp_path, per_class=6)
    # An unrelated target directory must not be opened or interpreted.
    target = root / 'Product'; target.mkdir()
    (target / 'forbidden-labels').write_bytes(b'not an image')
    original_inventory, original_epoch = m.inventory, m.run_epoch
    monkeypatch.setitem(m.COUNTS, 'Art', 12)
    monkeypatch.setattr(m, 'BATCH_SIZE', 4)
    monkeypatch.setattr(m, 'inventory', lambda *a: original_inventory(*a, expected_count=12, classes=2))
    def model_fixture(_):
        return torch.nn.ModuleDict({'F': torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()),
                                    'B': torch.nn.Linear(3, 3), 'C': torch.nn.Linear(3, 2)})
    monkeypatch.setattr(m, 'load_model', model_fixture)
    args = SimpleNamespace(data_root=root, domain='Art', source_list=listing, source_list_sha256=m.file_sha(listing),
                           initialization=tmp_path/'unused', seed=2020, output_dir=tmp_path/'run',
                           device='cpu', workers=0, resume=False, preflight=False)
    stop_step = [3]
    def interrupt(model, loader, device, **kw):
        if kw.get('optimizer') is not None and kw.get('step', 0) >= stop_step[0]:
            raise RuntimeError('injected source interruption')
        return original_epoch(model, loader, device, **kw)
    monkeypatch.setattr(m, 'run_epoch', interrupt)
    with pytest.raises(RuntimeError, match='injected'):
        m.execute(args)
    assert not (args.output_dir / 'completion.json').exists()
    assert m.read_json(args.output_dir / 'failure.json')['complete'] is False
    assert m.read_json(args.output_dir / 'source-population.json')['metadata_sidecars'] == []
    assert m.read_json(args.output_dir / 'last-epoch.json')['epoch'] == 0
    args.resume = True; stop_step[0] = 6
    with pytest.raises(RuntimeError, match='injected'):
        m.execute(args)
    assert m.read_json(args.output_dir / 'last-epoch.json')['epoch'] == 1
    assert len(m.read_json(args.output_dir / 'learning-curves.json')) == 2
    assert not (args.output_dir / 'completion.json').exists()


def appledouble_fixture(entries=((9, 50, 32), (2, 82, 0)), *, magic=0x00051607, version=0x00020000):
    return (struct.pack('>II16sH', magic, version, b'Mac OS X        ', len(entries))
            + b''.join(struct.pack('>III', *e) for e in entries) + bytes(32))


def test_valid_appledouble_is_recorded_not_counted_as_image(tmp_path):
    root, listing = fixture_source(tmp_path)
    sidecar = root / 'Art/class0/._0.png'
    sidecar.write_bytes(appledouble_fixture())
    rows, identity = m.inventory(root, 'Art', listing, m.file_sha(listing), expected_count=10, classes=2)
    assert len(rows) == 10
    assert identity['metadata_sidecars'] == [{
        'path': 'Art/class0/._0.png', 'companion': 'Art/class0/0.png',
        'companion_kind': 'image',
        'sha256': m.file_sha(sidecar), 'bytes': 82, 'version': 0x00020000,
        'entry_ids': [9, 2]}]


def test_class_directory_sidecar_requires_exact_source_class_mapping(tmp_path):
    root, listing = fixture_source(tmp_path)
    sidecar = root / 'Art/._class0'
    sidecar.write_bytes(appledouble_fixture())
    rows, identity = m.inventory(root, 'Art', listing, m.file_sha(listing), expected_count=10, classes=2)
    assert len(rows) == 10
    assert identity['metadata_sidecars'][0]['companion_kind'] == 'class_directory'
    assert identity['metadata_sidecars'][0]['companion'] == 'Art/class0'
    (root / 'Art/unlisted').mkdir()
    (root / 'Art/._unlisted').write_bytes(appledouble_fixture())
    with pytest.raises(m.SourceError, match='companion|directory'):
        m.inventory(root, 'Art', listing, m.file_sha(listing), expected_count=10, classes=2)


@pytest.mark.parametrize('relative', ['Art/unlisted_empty', 'Art/class0/nested_empty'])
def test_inventory_rejects_arbitrary_empty_directories(tmp_path, relative):
    root, listing = fixture_source(tmp_path)
    (root / relative).mkdir()
    with pytest.raises(m.SourceError, match='directory'):
        m.inventory(root, 'Art', listing, m.file_sha(listing), expected_count=10, classes=2)


@pytest.mark.parametrize('kind', ['magic', 'version', 'truncated', 'descriptor_overflow', 'data_fork',
                                 'duplicate_id', 'overlap', 'offset_inside_header', 'outside_file',
                                 'orphan', 'symlink', 'unknown_file', 'unknown_entry'])
def test_appledouble_fail_closed(tmp_path, kind):
    root, listing = fixture_source(tmp_path)
    sidecar = root / 'Art/class0/._0.png'
    data = appledouble_fixture()
    if kind == 'magic':
        data = appledouble_fixture(magic=0x00051600)
    elif kind == 'version':
        data = appledouble_fixture(version=1)
    elif kind == 'truncated':
        data = data[:12]
    elif kind == 'descriptor_overflow':
        data = data[:24] + b'\xff\xff' + data[26:]
    elif kind == 'data_fork':
        data = appledouble_fixture(((1, 50, 32), (2, 82, 0)))
    elif kind == 'duplicate_id':
        data = appledouble_fixture(((9, 50, 32), (9, 82, 0)))
    elif kind == 'overlap':
        data = appledouble_fixture(((9, 50, 32), (2, 60, 10)))
    elif kind == 'offset_inside_header':
        data = appledouble_fixture(((9, 40, 32), (2, 82, 0)))
    elif kind == 'outside_file':
        data = appledouble_fixture(((9, 50, 100), (2, 82, 0)))
    elif kind == 'orphan':
        sidecar = sidecar.with_name('._absent.png')
    elif kind == 'unknown_file':
        sidecar = sidecar.with_name('unrecognized.bin')
    elif kind == 'unknown_entry':
        data = appledouble_fixture(((99, 50, 32), (2, 82, 0)))
    if kind == 'symlink':
        target = tmp_path / 'metadata'; target.write_bytes(data)
        sidecar.symlink_to(target)
    else:
        sidecar.write_bytes(data)
    with pytest.raises(m.SourceError):
        m.inventory(root, 'Art', listing, m.file_sha(listing), expected_count=10, classes=2)


def test_actual_exfat_appledouble_inventory():
    destination = os.getenv('KBOUND_TEST_EXFAT_ROOT')
    if not destination:
        pytest.skip('optional exFAT fixture directory not supplied')
    # Deliberately retain this isolated fixture: exFAT cleanup semantics differ.
    fixture = Path(tempfile.mkdtemp(prefix='officehome-exfat-fixture-', dir=destination))
    root, listing = fixture_source(fixture)
    for path in (root / 'Art').glob('*/*.png'):
        if not path.name.startswith('._'):
            subprocess.run(['/usr/bin/xattr', '-w', 'com.kbound.regression', 'fixture-only', str(path)], check=True)
    rows, identity = m.inventory(root, 'Art', listing, m.file_sha(listing), expected_count=10, classes=2)
    assert len(rows) == 10 and len(identity['metadata_sidecars']) == 12
    assert sum(r['companion_kind'] == 'image' for r in identity['metadata_sidecars']) == 10
    assert sum(r['companion_kind'] == 'class_directory' for r in identity['metadata_sidecars']) == 2
    assert all(r['version'] == 0x00020000 and r['bytes'] >= 50 for r in identity['metadata_sidecars'])
    assert all(m.file_sha(root / r['path']) == r['sha256'] for r in identity['metadata_sidecars'])
