"""Synthetic archives/processes only; never open the proposed real pilot inputs."""
from copy import deepcopy
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from PIL import Image, PngImagePlugin
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts'
sys.path.insert(0, str(SCRIPTS))


def broker():
    assert (SCRIPTS / 'domainnet_pilot_images.py').is_file(), 'selected-member broker missing'
    return importlib.import_module('domainnet_pilot_images')


def runner():
    assert (SCRIPTS / 'domainnet_capacity_pilot.py').is_file(), 'bounded pilot runner missing'
    return importlib.import_module('domainnet_capacity_pilot')


def picture(color, text=''):
    buf = io.BytesIO(); meta = PngImagePlugin.PngInfo(); meta.add_text('note', text)
    Image.new('RGB', (8, 9), color).save(buf, format='PNG', pnginfo=meta)
    return buf.getvalue()


def archive(tmp, payloads=None, names=None):
    values = payloads or [picture((12, 21, 30)), picture((50, 60, 70)), b'never decode this']
    path = tmp / 'painting.zip'
    names = names or ['painting/a/one.png', 'painting/b/two.png', 'real/c/forbidden.png']
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, value in zip(names, values): z.writestr(name, value)
    members = []
    with zipfile.ZipFile(path) as z:
        for name, role in zip(names[:2], ['adaptation', 'diagnostic_evaluation']):
            info = z.getinfo(name)
            members.append({'image_id': name, 'role': role, 'zip_crc32_metadata_not_authentication': info.CRC,
                            'compressed_bytes': info.compress_size, 'uncompressed_bytes': info.file_size})
    identity = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return identity, members


def test_only_selected_members_are_decompressed_once(tmp_path, monkeypatch):
    m = broker(); identity, members = archive(tmp_path); opened = []
    original = zipfile.ZipFile.open
    def observed(self, name, *a, **kw):
        key = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert key != 'real/c/forbidden.png', 'unauthorized domain payload opened'
        opened.append(key); return original(self, name, *a, **kw)
    monkeypatch.setattr(zipfile.ZipFile, 'open', observed)
    cache, inventory = m.load_selected(identity, members, cache_limit=4096)
    assert opened == [r['image_id'] for r in members]
    assert list(cache) == opened and all(im.mode == 'RGB' for im in cache.values())
    assert cache[opened[0]].getpixel((0, 0)) == (12, 21, 30)
    assert inventory['decoded_rgb_bytes'] == 432 and inventory['images_decoded'] == 2
    assert inventory['entries'][0]['decoded_rgb_sha256'] != inventory['entries'][1]['decoded_rgb_sha256']


@pytest.mark.parametrize('fault', ['digest', 'size', 'symlink', 'parent_symlink', 'role', 'traversal', 'repeat', 'metadata'])
def test_bad_archive_or_selection_fails_before_payload(tmp_path, monkeypatch, fault):
    m = broker(); identity, members = archive(tmp_path)
    if fault == 'digest': identity['sha256'] = '0' * 64
    if fault == 'size': identity['bytes'] += 1
    if fault == 'symlink':
        link = tmp_path / 'redirect.zip'; link.symlink_to(identity['path']); identity['path'] = str(link)
    if fault == 'parent_symlink':
        link = tmp_path / 'parent'; link.symlink_to(tmp_path, target_is_directory=True); identity['path'] = str(link / 'painting.zip')
    if fault == 'role': members[0]['role'] = 'score'
    if fault == 'traversal': members[0]['image_id'] = 'painting/../outside.png'
    if fault == 'repeat': members[1]['image_id'] = members[0]['image_id']
    if fault == 'metadata': members[0]['uncompressed_bytes'] += 1
    def trap(*a, **kw): raise AssertionError('member payload accessed')
    monkeypatch.setattr(zipfile.ZipFile, 'open', trap)
    with pytest.raises((ValueError, OSError)): m.load_selected(identity, members, cache_limit=4096)


@pytest.mark.parametrize('same_encoded', [True, False])
def test_cross_role_encoded_or_decoded_duplicates_stop_without_replacement(tmp_path, same_encoded):
    m = broker()
    a, b = picture((1, 2, 3)), picture((1, 2, 3), 'different encoding')
    identity, members = archive(tmp_path, [a, a if same_encoded else b, b'not selected'])
    with pytest.raises(ValueError, match='cross-role'): m.load_selected(identity, members, cache_limit=4096)


def test_cache_ceiling_and_corrupt_image_stop(tmp_path):
    m = broker(); identity, members = archive(tmp_path)
    with pytest.raises(ValueError, match='cache'): m.load_selected(identity, members, cache_limit=215)
    damaged = tmp_path / 'damaged'; damaged.mkdir()
    identity, members = archive(damaged, [b'not an image', picture((1, 2, 3)), b'unused'])
    with pytest.raises((ValueError, OSError)): m.load_selected(identity, members, cache_limit=4096)


def test_approval_identity_is_not_a_caller_chosen_config(tmp_path):
    m = runner(); path = tmp_path / 'proposal.json'; path.write_text('{}')
    with pytest.raises(ValueError): m.read_approved(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize('mode', ['wall', 'rss', 'disk'])
def test_resource_monitor_stops_only_its_child(tmp_path, mode):
    m = runner()
    code = {'wall': 'import time; time.sleep(10)',
            'rss': 'import time; x=bytearray(50*1024**2); time.sleep(10)',
            'disk': 'from pathlib import Path; import time; Path("big").write_bytes(b"x"*10000); time.sleep(10)'}[mode]
    budget = {'hard_wall_ceiling_seconds': .3 if mode == 'wall' else 5,
              'peak_rss_stop_bytes': 10*1024**2 if mode == 'rss' else 1024**3,
              'max_new_disk_bytes': 100 if mode == 'disk' else 1024**3}
    result = m.run_bounded([sys.executable, '-c', code], tmp_path, budget)
    assert result['status'] == 'STOP_' + mode.upper()
    assert result['child_reaped'] is True


def test_resource_monitor_retains_normal_and_failed_exit(tmp_path):
    m = runner()
    budget = {'hard_wall_ceiling_seconds': 5, 'peak_rss_stop_bytes': 1024**3, 'max_new_disk_bytes': 1024**3}
    for code, expected in [('print("ok")', 'PROCESS_EXIT_0'), ('raise ValueError("retained")', 'PROCESS_FAILED')]:
        dest = tmp_path / expected; dest.mkdir()
        result = m.run_bounded([sys.executable, '-c', code], dest, budget)
        assert result['status'] == expected
        assert (dest / 'worker.log').is_file()


def test_fixed_prediction_scoring_changes_only_metrics():
    m = runner()
    packet = {'sample_ids': ['painting/a/one.png', 'painting/b/two.png'],
              'frozen': [0, 1], 'candidate_direct': [1, 1], 'candidate_refined': [1, 0]}
    before = deepcopy(packet)
    result = m.score_predictions(packet, {'painting/a/one.png': 0, 'painting/b/two.png': 1}, classes=2)
    assert result['frozen_accuracy'] == 1.
    assert result['candidate_direct_accuracy'] == .5
    assert result['candidate_refined_accuracy'] == 0.
    assert result['direct_benefit'] == -.5 and result['refined_benefit'] == -1.
    other = m.score_predictions(packet, {'painting/a/one.png': 1, 'painting/b/two.png': 0}, classes=2)
    assert other['refined_benefit'] == 1. and packet == before
    with pytest.raises(ValueError): m.score_predictions({**packet, 'frozen': [0]}, {}, classes=2)


def test_no_unapproved_cli_options_can_open_inputs(tmp_path):
    m = runner()
    result = subprocess.run([sys.executable, str(SCRIPTS/'domainnet_capacity_pilot.py'), 'run',
                             '--proposal', str(tmp_path/'missing'), '--proposal-sha256', 'a'*64,
                             '--device', 'mps'], capture_output=True, timeout=15)
    assert result.returncode == 2 and b'unrecognized arguments' in result.stderr


def fake_collection(monkeypatch, tmp_path, fail=False):
    """Real broker/transforms/runner, synthetic image bytes and instrumented model."""
    import warnings
    from types import SimpleNamespace
    import torch
    import domainnet_reference_adapter as adapter
    import domainnet_reference_source as source
    m = runner()
    path = tmp_path/'painting48.zip'
    names = [f'painting/c/im{i}.png' for i in range(48)]
    with zipfile.ZipFile(path, 'w') as z:
        for i, name in enumerate(names): z.writestr(name, picture((i, 20, 30)))
    with zipfile.ZipFile(path) as z:
        entries = [{'image_id': n, 'role': 'adaptation' if i < 16 else 'diagnostic_evaluation',
                    'zip_crc32_metadata_not_authentication': z.getinfo(n).CRC,
                    'compressed_bytes': z.getinfo(n).compress_size,
                    'uncompressed_bytes': z.getinfo(n).file_size} for i, n in enumerate(names)]
    seen = []
    def transform(im): return torch.tensor([[im.getpixel((0, 0))[0]]], dtype=torch.float32)
    class Model:
        queue_ptr = 0
        def state_dict(self): return {'x': torch.tensor([0.])}
    class Session:
        args = {}; identity = {}; provenance = {}; completed_steps = 0; banks = {}
        model = Model()
        ref = SimpleNamespace(get_augmentation=lambda _: transform,
                              get_augmentation_versions=lambda _: lambda im: [transform(im)]*3)
        def predict(self, batches, refine=False):
            data = torch.cat(list(batches)).flatten().tolist()
            assert data == list(range(16, 48)), 'evaluation selection changed'
            seen.append('refined' if refine else 'direct')
            return torch.zeros(32, dtype=torch.long)
        def initialize_bank(self, batches):
            assert torch.cat(list(batches)).flatten().tolist() == list(range(16))
            seen.append('bank')
        def train_epoch(self, batches, epoch):
            for i, (views, indices) in enumerate(batches):
                assert indices.tolist() == list(range(i*4, i*4+4))
                assert views[0].flatten().tolist() == list(range(i*4, i*4+4))
            seen.append('train')
            warnings.warn('retained native warning', UserWarning)
            if fail: raise ValueError('native failure')
            self.completed_steps = 4; self.model.queue_ptr = 16
    monkeypatch.setattr(adapter, 'build_session', lambda *a, **k: Session())
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda _: None)
    monkeypatch.setattr(torch, 'set_num_threads', lambda _: None)
    def trap(*a, **k): raise AssertionError('label/source file was read after fake source construction')
    monkeypatch.setattr(source, 'read_verified_checkpoint', trap)
    proposal = {'reference': {'root': 'fake'}, 'checkpoint_path': 'fake',
        'execution': {'resolved_args': {}, 'queue_caveat': 'capacity only'},
        'selection': {'members': entries}, 'scope': 'capacity only', 'prior_access': {'real': 'UNKNOWN'},
        'archive': {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
        'budget': {'peak_rss_stop_bytes': 10*1024**3, 'decoded_rgb_cache_stop_bytes': 4096*48}}
    return m, proposal, seen


def test_collection_is_label_free_and_never_adapts_evaluation_images(tmp_path, monkeypatch):
    m, proposal, seen = fake_collection(monkeypatch, tmp_path)
    result = m.collect_payload(proposal)
    assert seen == ['direct', 'bank', 'train', 'direct', 'refined']
    assert result['completed_updates'] == 4 and result['inventory']['images_decoded'] == 48
    assert result['warnings'][0]['message'] == 'retained native warning'
    assert 'correctness' not in result and 'accuracy' not in result
    assert result['benchmark_complete'] is False and result['study_locked'] is False


def test_native_warning_survives_failed_collection(tmp_path, monkeypatch, capsys):
    m, proposal, _ = fake_collection(monkeypatch, tmp_path, fail=True)
    with pytest.raises(ValueError, match='native failure'): m.collect_payload(proposal)
    captured = capsys.readouterr()
    assert 'retained native warning' in captured.out + captured.err


def test_bad_prediction_identity_is_rejected_before_label_authority(tmp_path, monkeypatch):
    m = runner()
    import domainnet_reference_source as source
    def trap(*a, **k): raise AssertionError('labels opened before prediction validation')
    monkeypatch.setattr(source, 'read_verified_checkpoint', trap)
    proposal = {'selection': {'members': [{'image_id': 'painting/a/a.png', 'role': 'diagnostic_evaluation'}]}}
    path = tmp_path/'predictions.json'; path.write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        m.score_payload(proposal, path, hashlib.sha256(path.read_bytes()).hexdigest(), {})


def test_score_rejects_wrong_adaptation_ids_before_outcome_read():
    m = runner()
    proposal = {'selection': {'members': [{'image_id': 'painting/a/a.png', 'role': 'adaptation'},
                                          {'image_id': 'painting/b/b.png', 'role': 'diagnostic_evaluation'}]}}
    packet = {'schema': 'domainnet-capacity-predictions-v1', 'status': 'COLLECTED_NOT_SCORED',
        'proposal_sha256': m.APPROVED_SHA, 'implementation': {}, 'sample_ids': ['painting/b/b.png'],
        'adaptation_ids': ['real/forbidden.jpg'], 'completed_updates': 4,
        'study_locked': False, 'benchmark_complete': False,
        'frozen': [0], 'candidate_direct': [0], 'candidate_refined': [0]}
    with pytest.raises(ValueError): m.validate_collection(packet, proposal, {})


def test_exact_approved_member_vocabulary_passes_metadata_validation(monkeypatch):
    """Reads proposal metadata only, never the referenced real archive or images."""
    m = broker()
    path = SCRIPTS.parents[3] / 'experiments/kbound/results/natural_calibration_value_v1/DOMAINNET_PILOT_PROPOSAL_V1.json'
    proposal = runner().read_approved(path, runner().APPROVED_SHA)
    class Validated(Exception): pass
    def sentinel(identity):
        assert identity == proposal['archive']
        raise Validated('selection validated; real archive never opened')
    monkeypatch.setattr(m, 'authenticated_archive', sentinel)
    with pytest.raises(Validated):
        m.load_selected(proposal['archive'], proposal['selection']['members'],
                        cache_limit=proposal['budget']['decoded_rgb_cache_stop_bytes'])


def test_synthetic_hyphenated_category_decodes_without_relaxing_domain(tmp_path):
    identity, members = archive(tmp_path, names=['painting/teddy-bear/one.png',
                               'painting/axe/two.png', 'real/c/forbidden.png'])
    cache, inventory = broker().load_selected(identity, members, cache_limit=4096)
    assert list(cache)[0] == 'painting/teddy-bear/one.png' and inventory['images_decoded'] == 2
