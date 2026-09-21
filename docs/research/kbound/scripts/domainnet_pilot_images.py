"""Outcome-free, selected-member painting broker. Never extracts an archive.

The caller supplies an authenticated archive identity and preselected roles.
Only those members are decoded, once, into a bounded RGB cache. This verifies
bytes and role disjointness, not historical nonaccess or population sampling.
"""
from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import re
import stat
import zipfile

from PIL import Image
from task3_image_panel import _open_directory_chain


def signature(s):
    return tuple(getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


@contextmanager
def authenticated_archive(identity):
    path = Path(os.path.abspath(identity['path']))
    if (type(identity['bytes']) is not int or not 0 < identity['bytes'] < 8 * 1024**3
            or not re.fullmatch('[0-9a-f]{64}', identity['sha256'])):
        raise ValueError('invalid archive identity')
    parent = _open_directory_chain(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, 'rb') as handle:
            before = os.fstat(handle.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_size != identity['bytes']
                    or getattr(before, 'st_flags', 0) & 0x40000000):
                raise ValueError('archive must be resident regular file with matching size')
            digest = hashlib.sha256()
            remaining = identity['bytes']
            while remaining:
                block = handle.read(min(4 * 1024**2, remaining))
                if not block:
                    raise ValueError('short archive read')
                digest.update(block); remaining -= len(block)
            if (digest.hexdigest() != identity['sha256']
                    or signature(before) != signature(os.fstat(handle.fileno()))):
                raise ValueError('archive digest or identity mismatch')
            handle.seek(0)
            with zipfile.ZipFile(handle) as archive:
                yield archive
            if signature(before) != signature(os.fstat(handle.fileno())):
                raise ValueError('archive changed during decoding')
    finally:
        os.close(parent)


def load_selected(identity, members, *, cache_limit):
    if type(cache_limit) is not int or not 0 < cache_limit <= 256 * 1024**2:
        raise ValueError('invalid cache ceiling')
    if not isinstance(members, list) or not 1 <= len(members) <= 48:
        raise ValueError('invalid selection size')
    fields = {'image_id', 'role', 'zip_crc32_metadata_not_authentication', 'compressed_bytes', 'uncompressed_bytes'}
    seen = set()
    for entry in members:
        if (not isinstance(entry, dict) or set(entry) != fields
                or not isinstance(entry['image_id'], str)
                or not re.fullmatch(r'painting/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\.(jpg|jpeg|png)', entry['image_id'])
                or '..' in entry['image_id'] or entry['image_id'] in seen
                or entry['role'] not in ('adaptation', 'diagnostic_evaluation')):
            raise ValueError('invalid selected member or role')
        for key in fields - {'image_id', 'role'}:
            if type(entry[key]) is not int or entry[key] < 0:
                raise ValueError('invalid selected metadata')
        seen.add(entry['image_id'])
    cache, inventory, encoded_roles, pixel_roles = {}, [], {}, {}
    used = 0
    with authenticated_archive(identity) as archive:
        infos = archive.infolist()
        if len(infos) > 100000 or len({i.filename for i in infos}) != len(infos):
            raise ValueError('excessive or duplicate ZIP directory')
        # Validate the complete selection before touching any member payload.
        for entry in members:
            try:
                info = archive.getinfo(entry['image_id'])
            except KeyError as exc:
                raise ValueError('selected member missing') from exc
            mode = info.external_attr >> 16
            if (info.is_dir() or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)) or info.flag_bits & 1
                    or not 0 < info.file_size <= 50 * 1024**2
                    or (info.CRC, info.compress_size, info.file_size) !=
                    (entry['zip_crc32_metadata_not_authentication'], entry['compressed_bytes'], entry['uncompressed_bytes'])):
                raise ValueError('selected member metadata mismatch or unsupported kind')
        for entry in members:
            name, role = entry['image_id'], entry['role']
            # ZipFile.read verifies CRC; authentication comes from the archive SHA.
            encoded = archive.read(name)
            encoded_sha = hashlib.sha256(encoded).hexdigest()
            with Image.open(io.BytesIO(encoded)) as image:
                width, height = image.size
                size = width * height * 3
                if width <= 0 or height <= 0 or used + size > cache_limit:
                    raise ValueError('decoded RGB cache ceiling')
                image.load()
                rgb = image.convert('RGB')
            pixel_sha = hashlib.sha256(f'{width},{height}:'.encode() + rgb.tobytes()).hexdigest()
            for digest, roles in ((encoded_sha, encoded_roles), (pixel_sha, pixel_roles)):
                if digest in roles and roles[digest] != role:
                    raise ValueError('cross-role encoded/decoded duplicate; no replacement permitted')
                roles[digest] = role
            cache[name] = rgb; used += size
            inventory.append({**entry, 'encoded_sha256': encoded_sha, 'decoded_rgb_sha256': pixel_sha,
                              'width': width, 'height': height})
    return cache, {'entries': inventory, 'decoded_rgb_bytes': used, 'images_decoded': len(cache),
                   'unique_encoded_images': len(encoded_roles), 'unique_decoded_images': len(pixel_roles)}
