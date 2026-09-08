from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import sqlite3
import tarfile
from functools import lru_cache
from pathlib import Path

import pytest
from PIL import Image


@lru_cache(None)
def module():
    path = Path(__file__).parents[1] / 'experiments/kbound/reference_source/imagenet_archives.py'
    assert path.is_file(), 'strict full-population ImageNet archive reader missing'
    spec = importlib.util.spec_from_file_location('image_archives', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_tar(tmp_path, *, omit=None, extra=None):
    path = tmp_path / 'weather.tar'
    raw = io.BytesIO()
    Image.new('RGB', (4, 4), (30, 60, 90)).save(raw, format='JPEG')
    payload = raw.getvalue()
    with tarfile.open(path, 'w') as tf:
        for severity in (1, 2):
            for image_id, synset in [(1, 'n00000001'), (2, 'n00000002')]:
                name = f'frost/{severity}/{synset}/ILSVRC2012_val_{image_id:08d}.JPEG'
                if name == omit:
                    continue
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                tf.addfile(member, io.BytesIO(payload))
        if extra:
            member = tarfile.TarInfo(extra)
            member.size = len(payload)
            tf.addfile(member, io.BytesIO(payload))
    return path


PROFILE = {'corruptions': ['frost'], 'severities': [1, 2], 'images_per_cell': 2, 'classes': 2}


def build(path, out):
    return module().index_archive(path, 'weather', out,
                                  expected_md5=hashlib.md5(path.read_bytes()).hexdigest(), profile=PROFILE)


def receipt_sha(out):
    return hashlib.sha256((out / 'completion.json').read_bytes()).hexdigest()


def test_archive_index_has_all_cells_and_roundtrips_original_bytes_without_extraction(tmp_path):
    path = make_tar(tmp_path)
    receipt = build(path, tmp_path / 'index')
    assert receipt['complete'] is True
    assert receipt['full_reference_complete'] is False
    assert receipt['images'] == 4
    assert receipt['cell_counts'] == {'frost/1': 2, 'frost/2': 2}
    assert not (tmp_path / 'frost').exists()
    connection = sqlite3.connect(tmp_path / 'index/members.sqlite3')
    connection.row_factory = sqlite3.Row
    row = dict(connection.execute('SELECT * FROM images WHERE severity=2 AND image_id=1').fetchone())
    connection.close()
    image = module().read_indexed_image(path, row)
    assert image.mode == 'RGB' and image.size == (4, 4)
    assert row['synset'] == 'n00000001'
    with path.open('r+b') as handle:
        handle.seek(row['offset'])
        handle.write(b'broken')
    with pytest.raises(module().ArchiveError, match='hash'):
        module().read_indexed_image(path, row)


def test_missing_member_cannot_complete_full_declared_cell(tmp_path):
    path = make_tar(tmp_path, omit='frost/2/n00000002/ILSVRC2012_val_00000002.JPEG')
    with pytest.raises(module().ArchiveError, match='count'):
        build(path, tmp_path / 'index')
    assert json.loads((tmp_path / 'index/completion.json').read_text())['complete'] is False


@pytest.mark.parametrize('extra', [
    'frost/1/n00000001/ILSVRC2012_val_00000001.JPEG',
    'frost/1/../../escape.JPEG',
    'frost/2/n00000002/ILSVRC2012_val_00000001.JPEG',
])
def test_duplicate_escape_or_changed_class_mapping_is_rejected(tmp_path, extra):
    path = make_tar(tmp_path, extra=extra)
    with pytest.raises(module().ArchiveError):
        build(path, tmp_path / 'index')
    assert not (tmp_path / 'escape.JPEG').exists()


def test_bad_archive_hash_stops_before_outputs_and_existing_output_is_preserved(tmp_path):
    path = make_tar(tmp_path)
    with pytest.raises(module().ArchiveError, match='MD5'):
        module().index_archive(path, 'weather', tmp_path / 'new', expected_md5='0' * 32, profile=PROFILE)
    assert not (tmp_path / 'new').exists()
    out = tmp_path / 'old'
    out.mkdir()
    (out / 'keep').write_text('original')
    with pytest.raises(module().ArchiveError, match='fresh'):
        build(path, out)
    assert (out / 'keep').read_text() == 'original'


def test_default_full_profile_rejects_synthetic_archive_identity(tmp_path):
    path = make_tar(tmp_path)
    with pytest.raises(module().ArchiveError, match='MD5|official'):
        module().index_archive(path, 'weather', tmp_path / 'full')
    assert not (tmp_path / 'full').exists()


def test_gzip_disguised_as_tar_has_logical_offsets_and_reusable_stream_reader(tmp_path):
    path = make_tar(tmp_path)
    path.write_bytes(gzip.compress(path.read_bytes()))
    receipt = build(path, tmp_path / 'index')
    assert receipt['archive_compression'] == 'gzip'
    assert receipt['index_offset_space'] == 'uncompressed_tar_stream'
    db = sqlite3.connect(tmp_path / 'index/members.sqlite3')
    db.row_factory = sqlite3.Row
    rows = [dict(row) for row in db.execute('SELECT * FROM images ORDER BY offset')]
    db.close()
    with module().IndexedImageReader(path) as reader:
        for row in rows:
            assert reader.read(row).size == (4, 4)
        # A backward read must be correct, though it requires replaying gzip.
        assert reader.read(rows[0]).mode == 'RGB'
    assert module().read_indexed_image(path, rows[-1]).size == (4, 4)
    assert not (tmp_path / 'frost').exists()


def test_accept_index_binds_receipt_archive_and_read_only_database_for_tiny_fixture(tmp_path):
    path = make_tar(tmp_path)
    out = tmp_path / 'index'
    build(path, out)

    connection, receipt = module().accept_index(out, receipt_sha(out), require_full=False)
    try:
        assert receipt['complete'] is True
        assert receipt['full_reference_complete'] is False
        assert connection.execute('PRAGMA query_only').fetchone()[0] == 1
        connection.row_factory = sqlite3.Row
        record = dict(connection.execute('SELECT * FROM images ORDER BY offset LIMIT 1').fetchone())
        assert module().read_indexed_image(path, record).size == (4, 4)
        with pytest.raises(sqlite3.OperationalError):
            connection.execute('DELETE FROM images')
    finally:
        connection.close()


def test_accept_index_never_promotes_incomplete_fixture_to_full(tmp_path):
    path = make_tar(tmp_path)
    out = tmp_path / 'index'
    build(path, out)

    with pytest.raises(module().ArchiveError, match='full|official'):
        module().accept_index(out, receipt_sha(out))


def test_accept_index_rejects_false_full_claim_even_when_fixture_mode_is_allowed(tmp_path):
    path = make_tar(tmp_path)
    out = tmp_path / 'index'
    build(path, out)
    receipt_path = out / 'completion.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['full_reference_complete'] = True
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(module().ArchiveError, match='full|official'):
        module().accept_index(out, receipt_sha(out), require_full=False)


@pytest.mark.parametrize('kind', ['receipt', 'database', 'schema', 'archive'])
def test_accept_index_rejects_tampered_evidence(tmp_path, kind):
    path = make_tar(tmp_path)
    out = tmp_path / 'index'
    build(path, out)
    expected_receipt_sha = receipt_sha(out)
    receipt_path = out / 'completion.json'
    database = out / 'members.sqlite3'

    if kind == 'receipt':
        receipt_path.write_bytes(receipt_path.read_bytes() + b' ')
    elif kind == 'database':
        database.write_bytes(database.read_bytes() + b'tampered')
    elif kind == 'schema':
        connection = sqlite3.connect(database)
        connection.execute('ALTER TABLE images ADD COLUMN unbound TEXT')
        connection.commit()
        connection.close()
        receipt = json.loads(receipt_path.read_text())
        receipt['index_sha256'] = hashlib.sha256(database.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt))
        expected_receipt_sha = receipt_sha(out)
    else:
        with path.open('ab') as stream:
            stream.write(b'tampered')

    with pytest.raises(module().ArchiveError, match='receipt|index|schema|archive|MD5'):
        module().accept_index(out, expected_receipt_sha, require_full=False)
