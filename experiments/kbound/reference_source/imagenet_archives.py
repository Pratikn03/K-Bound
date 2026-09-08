"""Full ImageNet-C tar indexes and strict image reads, without extraction.

The published MD5 authenticates an original archive; per-image SHA256 and
offsets bind subsequent reads. An index does not establish KGA partitions,
model performance, or successful decoding of images not yet evaluated.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import stat
import tarfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

GROUPS = {
    'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
    'extra': ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate'],
}
MD5 = {
    'noise': 'e80562d7f6c3f8834afb1ecf27252745',
    'blur': '2d8e81fdd8e07fef67b9334fa635e45c',
    'weather': '33ffea4db4d93fe4a428c40a6ce0c25d',
    'digital': '89157860d7b10d5797849337ca2e5c03',
    'extra': 'd492dfba5fc162d8ec2c3cd8ee672984',
}


class ArchiveError(ValueError):
    """A declared archive or image could not be verified completely."""


def _path(value):
    value = Path(value).absolute()
    if '..' in value.parts or any(p.is_symlink() for p in (value, *value.parents)):
        raise ArchiveError('archive/output path must not use symlinks or parent traversal')
    return value


def _open_regular(path):
    try:
        descriptor = os.open(_path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        handle = os.fdopen(descriptor, 'rb')
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            handle.close()
            raise ArchiveError('archive must be a regular file')
        return handle
    except OSError as exc:
        raise ArchiveError(f'archive unavailable: {exc}') from exc


def _identity(handle):
    s = os.fstat(handle.fileno())
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _digest(handle, kind):
    value = hashlib.new(kind)
    handle.seek(0)
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def _write_receipt(path, receipt):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def index_archive(path, group, output_dir, *, expected_md5=None, profile=None):
    """Index plain/gzip tar streams; only the official profile certifies full input coverage."""
    if group not in GROUPS:
        raise ArchiveError('unknown official archive group')
    official = {'corruptions': GROUPS[group], 'severities': [1, 2, 3, 4, 5],
                'images_per_cell': 50000, 'classes': 1000}
    profile = official if profile is None else profile
    if (set(profile) != set(official) or not profile['corruptions'] or not profile['severities']
            or not set(profile['corruptions']).issubset(GROUPS[group])
            or len(set(profile['corruptions'])) != len(profile['corruptions'])
            or len(set(profile['severities'])) != len(profile['severities'])
            or any(type(s) is not int or s not in range(1, 6) for s in profile['severities'])
            or any(type(profile[k]) is not int or profile[k] < 1 for k in ('images_per_cell', 'classes'))
            or profile['images_per_cell'] % profile['classes']):
        raise ArchiveError('invalid exact population profile')
    expected_md5 = MD5[group] if expected_md5 is None else expected_md5
    if not isinstance(expected_md5, str) or not re.fullmatch('[0-9a-f]{32}', expected_md5):
        raise ArchiveError('invalid expected MD5')
    path, out = _path(path), _path(output_dir)
    if out.exists():
        raise ArchiveError('fresh index output directory required')
    started = time.monotonic()
    with _open_regular(path) as handle:
        identity = _identity(handle)
        observed = _digest(handle, 'md5')
        if observed != expected_md5 or identity != _identity(handle):
            raise ArchiveError('archive MD5 mismatch or changed during authentication')
        handle.seek(0)
        compression = 'gzip' if handle.read(2) == b'\x1f\x8b' else 'none'
        try:
            out.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            raise ArchiveError('fresh index output directory required') from exc
        receipt = {'schema': 'imagenetc_archive_index_v1', 'archive': str(path), 'group': group,
                   'md5': observed, 'archive_bytes': identity[2], 'profile': profile,
                   'archive_compression': compression, 'index_offset_space': 'uncompressed_tar_stream',
                   'complete': False, 'full_reference_complete': False,
                   'all_images_decoded': False, 'kga_partitions_verified': False}
        database = out / 'members.sqlite3'
        db = sqlite3.connect(database)
        db.execute('CREATE TABLE images(corruption TEXT, severity INTEGER, image_id INTEGER, '
                   'synset TEXT, offset INTEGER, bytes INTEGER, sha256 TEXT, '
                   'PRIMARY KEY(corruption,severity,image_id))')
        counts, class_counts, canonical = Counter(), Counter(), {}
        try:
            handle.seek(0)
            with tarfile.open(fileobj=handle, mode='r:gz' if compression == 'gzip' else 'r:') as archive:
                for member in archive:
                    name = member.name.removeprefix('./').rstrip('/')
                    parts = name.split('/')
                    if (PurePosixPath(name).is_absolute() or any(p in ('', '.', '..') for p in parts)
                            or '\\' in name or any(ord(c) < 32 for c in name)):
                        raise ArchiveError('unsafe archive member path')
                    if member.isdir():
                        continue
                    if not member.isfile() or len(parts) != 4:
                        raise ArchiveError('archive member is not an expected regular image')
                    corruption, severity_text, synset, basename = parts
                    match = re.fullmatch(r'ILSVRC2012_val_([0-9]{8})\.(?:JPEG|jpeg|jpg)', basename)
                    if (corruption not in profile['corruptions'] or not severity_text.isdecimal()
                            or int(severity_text) not in profile['severities']
                            or not re.fullmatch(r'n[0-9]{8}', synset) or not match or member.size < 1):
                        raise ArchiveError('archive member does not match declared population')
                    severity, image_id = int(severity_text), int(match[1])
                    if not 1 <= image_id <= profile['images_per_cell']:
                        raise ArchiveError('original image ID outside full population')
                    if image_id in canonical and canonical[image_id] != synset:
                        raise ArchiveError('original image class mapping changed across cells')
                    canonical[image_id] = synset
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ArchiveError('missing archive image bytes')
                    digest, size = hashlib.sha256(), 0
                    with stream:
                        for block in iter(lambda stream=stream: stream.read(1024 * 1024), b''):
                            digest.update(block)
                            size += len(block)
                    if size != member.size:
                        raise ArchiveError('truncated archive image')
                    db.execute('INSERT INTO images VALUES(?,?,?,?,?,?,?)',
                               (corruption, severity, image_id, synset, member.offset_data, size, digest.hexdigest()))
                    counts[f'{corruption}/{severity}'] += 1
                    class_counts[(corruption, severity, synset)] += 1
                    if sum(counts.values()) % 50000 == 0:
                        db.commit()
                        archive.members.clear()
                        print(json.dumps({'indexed': sum(counts.values()), 'group': group,
                                          'wall_seconds': time.monotonic() - started}), flush=True)
            want = {f'{c}/{s}': profile['images_per_cell'] for c in profile['corruptions'] for s in profile['severities']}
            if (dict(counts) != want or len(canonical) != profile['images_per_cell']
                    or len(set(canonical.values())) != profile['classes']
                    or any(v != profile['images_per_cell'] // profile['classes'] for v in class_counts.values())):
                raise ArchiveError('archive population/class count mismatch')
            if identity != _identity(handle):
                raise ArchiveError('archive changed while indexing')
            db.commit()
            db.close()
            with database.open('rb') as indexed:
                database_sha = _digest(indexed, 'sha256')
            receipt.update(complete=True, full_reference_complete=profile == official and observed == MD5[group],
                           images=sum(counts.values()), cell_counts=dict(counts), index_sha256=database_sha,
                           index_file='members.sqlite3', wall_seconds=time.monotonic() - started)
        except (ArchiveError, OSError, EOFError, tarfile.TarError, sqlite3.Error) as exc:
            db.close()
            receipt.update(error=str(exc), images=sum(counts.values()), wall_seconds=time.monotonic() - started)
            _write_receipt(out / 'completion.json', receipt)
            raise ArchiveError(str(exc)) from exc
        _write_receipt(out / 'completion.json', receipt)
        return receipt


def _strict_json(data):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ArchiveError(f'duplicate receipt key: {key}')
            value[key] = item
        return value

    try:
        return json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ArchiveError(f'nonfinite receipt value: {value}')),
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ArchiveError(f'invalid receipt JSON: {exc}') from exc


def _validated_profile(group, profile):
    official = {'corruptions': GROUPS[group], 'severities': [1, 2, 3, 4, 5],
                'images_per_cell': 50000, 'classes': 1000}
    if (not isinstance(profile, dict) or set(profile) != set(official)
            or not isinstance(profile['corruptions'], list) or not profile['corruptions']
            or not isinstance(profile['severities'], list) or not profile['severities']
            or not set(profile['corruptions']).issubset(GROUPS[group])
            or any(not isinstance(c, str) for c in profile['corruptions'])
            or len(set(profile['corruptions'])) != len(profile['corruptions'])
            or len(set(profile['severities'])) != len(profile['severities'])
            or any(type(s) is not int or s not in range(1, 6) for s in profile['severities'])
            or any(type(profile[k]) is not int or profile[k] < 1 for k in ('images_per_cell', 'classes'))
            or profile['images_per_cell'] % profile['classes']):
        raise ArchiveError('invalid exact population profile in receipt')
    return official


def _validate_index_rows(connection, profile):
    expected_cells = {
        (corruption, severity): profile['images_per_cell']
        for corruption in profile['corruptions']
        for severity in profile['severities']
    }
    observed_cells = {
        (row[0], row[1]): row[2]
        for row in connection.execute(
            'SELECT corruption,severity,COUNT(*) FROM images GROUP BY corruption,severity'
        )
    }
    if observed_cells != expected_cells:
        raise ArchiveError('index population cell counts mismatch')
    invalid = connection.execute(
        "SELECT 1 FROM images WHERE typeof(corruption)!='text' OR typeof(severity)!='integer' "
        "OR typeof(image_id)!='integer' OR typeof(synset)!='text' OR typeof(offset)!='integer' "
        "OR typeof(bytes)!='integer' OR typeof(sha256)!='text' OR image_id<1 OR image_id>? "
        "OR offset<0 OR bytes<1 OR length(synset)!=9 OR substr(synset,1,1)!='n' "
        "OR substr(synset,2) GLOB '*[^0-9]*' OR length(sha256)!=64 "
        "OR sha256 GLOB '*[^0-9a-f]*' LIMIT 1",
        (profile['images_per_cell'],),
    ).fetchone()
    if invalid is not None:
        raise ArchiveError('invalid index image record')
    for corruption, severity, count, distinct_ids, low, high in connection.execute(
        'SELECT corruption,severity,COUNT(*),COUNT(DISTINCT image_id),MIN(image_id),MAX(image_id) '
        'FROM images GROUP BY corruption,severity'
    ):
        expected = expected_cells[(corruption, severity)]
        if (count, distinct_ids, low, high) != (expected, expected, 1, expected):
            raise ArchiveError('index image IDs do not cover the declared population')
    if connection.execute(
        'SELECT 1 FROM images GROUP BY image_id HAVING COUNT(DISTINCT synset)!=1 LIMIT 1'
    ).fetchone() is not None:
        raise ArchiveError('index image class mapping changes across cells')
    per_class = profile['images_per_cell'] // profile['classes']
    class_rows = list(connection.execute(
        'SELECT corruption,severity,synset,COUNT(*) FROM images '
        'GROUP BY corruption,severity,synset'
    ))
    if (len(class_rows) != len(expected_cells) * profile['classes']
            or any(row[3] != per_class for row in class_rows)):
        raise ArchiveError('index class counts mismatch')


def accept_index(output_dir, expected_receipt_sha256, require_full=True):
    """Return ``(read_only_connection, receipt)``; the caller must close the connection."""
    if (not isinstance(expected_receipt_sha256, str)
            or not re.fullmatch(r'[0-9a-f]{64}', expected_receipt_sha256)):
        raise ArchiveError('invalid expected receipt SHA256')
    out = _path(output_dir)
    if not out.is_dir():
        raise ArchiveError('index output directory is missing')
    with _open_regular(out / 'completion.json') as handle:
        receipt_bytes = handle.read()
    if hashlib.sha256(receipt_bytes).hexdigest() != expected_receipt_sha256:
        raise ArchiveError('receipt SHA256 mismatch')
    receipt = _strict_json(receipt_bytes)
    if not isinstance(receipt, dict) or receipt.get('schema') != 'imagenetc_archive_index_v1':
        raise ArchiveError('invalid archive index receipt schema')
    for field in ('complete', 'full_reference_complete', 'all_images_decoded', 'kga_partitions_verified'):
        if type(receipt.get(field)) is not bool:
            raise ArchiveError(f'receipt {field} must be boolean')
    if (receipt['complete'] is not True or receipt['all_images_decoded'] is not False
            or receipt['kga_partitions_verified'] is not False):
        raise ArchiveError('completed source-only archive index receipt required')
    group = receipt.get('group')
    if not isinstance(group, str) or group not in GROUPS:
        raise ArchiveError('invalid official archive group in receipt')
    profile = receipt.get('profile')
    official = _validated_profile(group, profile)
    if receipt.get('index_file') != 'members.sqlite3':
        raise ArchiveError('invalid archive index filename')
    if receipt.get('index_offset_space') != 'uncompressed_tar_stream':
        raise ArchiveError('invalid archive index offset semantics')
    compression = receipt.get('archive_compression')
    if compression not in ('none', 'gzip'):
        raise ArchiveError('invalid archive compression in receipt')
    observed_md5 = receipt.get('md5')
    if not isinstance(observed_md5, str) or not re.fullmatch(r'[0-9a-f]{32}', observed_md5):
        raise ArchiveError('invalid archive MD5 in receipt')
    is_official = profile == official and observed_md5 == MD5[group]
    if receipt['full_reference_complete'] is not is_official:
        raise ArchiveError('receipt full-reference claim disagrees with official profile/MD5')
    if require_full and not is_official:
        raise ArchiveError('full official archive index receipt required')
    archive_path = receipt.get('archive')
    if not isinstance(archive_path, str) or not archive_path:
        raise ArchiveError('archive path missing from receipt')
    with _open_regular(archive_path) as archive:
        archive_identity = _identity(archive)
        actual_md5 = _digest(archive, 'md5')
        archive.seek(0)
        actual_compression = 'gzip' if archive.read(2) == b'\x1f\x8b' else 'none'
        if archive_identity != _identity(archive):
            raise ArchiveError('archive changed during index acceptance')
    if (actual_md5 != observed_md5 or actual_compression != compression
            or type(receipt.get('archive_bytes')) is not int
            or receipt['archive_bytes'] != archive_identity[2]):
        raise ArchiveError('archive identity does not match receipt MD5/bytes/compression')

    expected_cells = {
        f'{corruption}/{severity}': profile['images_per_cell']
        for corruption in profile['corruptions']
        for severity in profile['severities']
    }
    expected_images = len(expected_cells) * profile['images_per_cell']
    if (type(receipt.get('images')) is not int or receipt['images'] != expected_images
            or receipt.get('cell_counts') != expected_cells
            or any(type(value) is not int for value in receipt['cell_counts'].values())):
        raise ArchiveError('receipt population counts mismatch')

    database = _path(out / 'members.sqlite3')
    with _open_regular(database) as indexed:
        database_identity = _identity(indexed)
        actual_index_sha = _digest(indexed, 'sha256')
        if database_identity != _identity(indexed):
            raise ArchiveError('index changed during authentication')
    if (not isinstance(receipt.get('index_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', receipt['index_sha256'])
            or actual_index_sha != receipt['index_sha256']):
        raise ArchiveError('index SHA256 mismatch')

    connection = None
    try:
        connection = sqlite3.connect(f'{database.as_uri()}?mode=ro', uri=True)
        connection.execute('PRAGMA query_only=ON')
        objects = connection.execute(
            "SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        ).fetchall()
        columns = [
            (row[1], row[2].upper(), row[3], row[4], row[5])
            for row in connection.execute('PRAGMA table_info(images)')
        ]
        expected_columns = [
            ('corruption', 'TEXT', 0, None, 1), ('severity', 'INTEGER', 0, None, 2),
            ('image_id', 'INTEGER', 0, None, 3), ('synset', 'TEXT', 0, None, 0),
            ('offset', 'INTEGER', 0, None, 0), ('bytes', 'INTEGER', 0, None, 0),
            ('sha256', 'TEXT', 0, None, 0),
        ]
        if objects != [('table', 'images')] or columns != expected_columns:
            raise ArchiveError('index SQLite schema mismatch')
        if connection.execute('PRAGMA integrity_check').fetchone() != ('ok',):
            raise ArchiveError('index SQLite integrity check failed')
        _validate_index_rows(connection, profile)
        current = os.stat(database)
        current_identity = (current.st_dev, current.st_ino, current.st_size,
                            current.st_mtime_ns, current.st_ctime_ns)
        if current_identity != database_identity:
            raise ArchiveError('index changed during acceptance')
        return connection, receipt
    except (ArchiveError, OSError, sqlite3.Error, TypeError, KeyError) as exc:
        if connection is not None:
            connection.close()
        if isinstance(exc, ArchiveError):
            raise
        raise ArchiveError(f'index acceptance failed: {exc}') from exc


class IndexedImageReader:
    """Reuse a plain/gzip stream for rows from a caller-verified index.

    Iterate rows in increasing offset order for gzip. Backward seeks are correct
    but replay the compressed stream; reopening for every image is unsuitable
    for full evaluation. Offsets always address the UNCOMPRESSED tar stream.
    """

    def __init__(self, archive_path):
        self.raw = _open_regular(archive_path)
        self.identity = _identity(self.raw)
        compressed = self.raw.read(2) == b'\x1f\x8b'
        self.raw.seek(0)
        self.stream = gzip.GzipFile(fileobj=self.raw) if compressed else self.raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()
        if self.stream is not self.raw:
            self.raw.close()

    def read(self, record):
        if (any(type(record.get(k)) is not int or record[k] < 0 for k in ('offset', 'bytes'))
                or record['bytes'] < 1 or not isinstance(record.get('sha256'), str)):
            raise ArchiveError('invalid indexed image record')
        if _identity(self.raw) != self.identity:
            raise ArchiveError('archive changed during indexed reading')
        try:
            self.stream.seek(record['offset'])
            payload = self.stream.read(record['bytes'])
        except (OSError, EOFError, ValueError) as exc:
            raise ArchiveError(f'indexed archive read failed: {exc}') from exc
        if _identity(self.raw) != self.identity:
            raise ArchiveError('archive changed during indexed reading')
        if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ArchiveError('indexed image hash mismatch')
        return _decode(payload)


def _decode(payload):
    from PIL import Image, ImageFile
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            return image.convert('RGB')
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ArchiveError(f'indexed image cannot be decoded: {exc}') from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def read_indexed_image(archive_path, record):
    """Convenience single read; use IndexedImageReader for a whole gzip population."""
    with IndexedImageReader(archive_path) as reader:
        return reader.read(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--group', choices=sorted(GROUPS), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(index_archive(args.archive, args.group, args.output_dir), sort_keys=True))
    except (ArchiveError, OSError) as exc:
        print(json.dumps({'complete': False, 'error': str(exc)}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
