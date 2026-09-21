"""Role-restricted painting staging and bounded lazy RGB decoding.

No truth, model, sampler selection, automatic replacement or archive extraction.
The production wrapper requires the complete approved manifest. Tests exercise
the same lower-level broker on synthetic archives; they do not authorize IO.
"""

import hashlib
import io
import os
import stat
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from domainnet_feasibility_contract import (
    BUDGET,
    read_bytes,
    read_json,
    validate_manifest,
    validate_member,
    write_bytes,
    write_json,
)
from domainnet_pilot_images import authenticated_archive
from PIL import Image
from task3_image_panel import _open_directory_chain


def fresh_directory(path):
    path = Path(os.path.abspath(path))
    parent = _open_directory_chain(path.parent)
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
        child = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        os.close(child)
        os.fsync(parent)
    finally:
        os.close(parent)
    return path


def rgb_from_bytes(data, ceiling):
    with Image.open(io.BytesIO(data)) as source:
        width, height = source.size
        if width <= 0 or height <= 0 or width * height * 3 > ceiling:
            raise ValueError("decoded RGB cache ceiling")
        source.load()
        image = source.convert("RGB")
    return image


def pixel_digest(image):
    return hashlib.sha256(f"{image.width},{image.height}:".encode() + image.tobytes()).hexdigest()


def directory_metadata(archive):
    infos = archive.infolist()
    if len(infos) > 100000 or len({i.filename for i in infos}) != len(infos):
        raise ValueError("excessive or duplicate ZIP directory")
    return {i.filename: i for i in infos}


def validate_zip_member(info, member):
    mode = info.external_attr >> 16
    if (
        info.is_dir()
        or (stat.S_IFMT(mode) and not stat.S_ISREG(mode))
        or info.flag_bits & 1
        or (info.CRC, info.file_size, info.compress_size)
        != (member["crc32"], member["encoded_bytes"], member["compressed_bytes"])
    ):
        raise ValueError("selected ZIP member metadata/kind mismatch")


def _stage_members(archive_identity, members, prior, fresh_root):
    """Private broker core; production callers must pass stage_selected gate."""
    if not members or len(members) > 18432:
        raise ValueError("invalid selection count")
    for member in members:
        validate_member(member)
    if len({x["image_id"] for x in members}) != len(members) or len({x["stage_name"] for x in members}) != len(members):
        raise ValueError("duplicate selected/staged name")
    root = fresh_directory(fresh_root)
    encoded_seen = {x["encoded_sha256"] for x in prior}
    pixels_seen = {x["decoded_rgb_sha256"] for x in prior}
    entries = []
    with authenticated_archive(archive_identity) as archive:
        directory = directory_metadata(archive)
        for member in members:
            if member["image_id"] not in directory:
                raise ValueError("selected member missing")
            validate_zip_member(directory[member["image_id"]], member)
        for member in members:
            data = archive.read(member["image_id"])
            sha = hashlib.sha256(data).hexdigest()
            with rgb_from_bytes(data, BUDGET["cache_bytes"]) as image:
                rgb_sha = pixel_digest(image)
                width, height = image.size
            # Strictly reject duplicates even within a role. Never redraw.
            if sha in encoded_seen or rgb_sha in pixels_seen:
                raise ValueError("encoded/pixel duplicate of prior or selected image; INPUT_STOP no replacement")
            encoded_seen.add(sha)
            pixels_seen.add(rgb_sha)
            write_bytes(root / member["stage_name"], data)
            entries.append(
                {**member, "encoded_sha256": sha, "decoded_rgb_sha256": rgb_sha, "width": width, "height": height}
            )
    inventory = {
        "schema": "painting-staged-inventory-v1",
        "root": str(root),
        "entries": entries,
        "encoded_bytes": sum(x["encoded_bytes"] for x in entries),
        "complete": True,
    }
    write_json(root / "inventory.json", inventory)
    return inventory


def stage_selected(manifest, fresh_root):
    validate_manifest(manifest)
    prior_bound = manifest["prior_inventory"]
    prior_packet = read_json(prior_bound["path"], prior_bound["sha256"])
    prior = prior_packet["inventory"]["entries"]
    if len(prior) != 48 or {x["image_id"] for x in prior} != set(manifest["excluded_ids"]):
        raise ValueError("prior pilot inventory identity mismatch")
    return _stage_members(manifest["archive"], manifest["members"], prior, fresh_root)


def validate_inventory(manifest, inventory):
    validate_manifest(manifest)
    if (
        inventory.get("schema") != "painting-staged-inventory-v1"
        or inventory.get("complete") is not True
        or len(inventory["entries"]) != 18432
        or inventory["encoded_bytes"] != 914238704
    ):
        raise ValueError("incomplete inventory")
    for entry, member in zip(inventory["entries"], manifest["members"]):
        if {k: entry[k] for k in member} != member:
            raise ValueError("inventory role/identity mismatch")
        if (
            len(entry["encoded_sha256"]) != 64
            or len(entry["decoded_rgb_sha256"]) != 64
            or type(entry["width"]) is not int
            or type(entry["height"]) is not int
            or not 0 < entry["width"] * entry["height"] * 3 <= BUDGET["cache_bytes"]
        ):
            raise ValueError("invalid image inventory")
    return inventory


class ImageView:
    """An episode/role capability; no getter for another role's images.

    Caller gets a temporary copy, never the cached object. Every access verifies
    staged encoded bytes, including cache hits. Cache ceiling excludes that one
    temporary copy and transform tensors; total RSS is independently monitored.
    """

    def __init__(self, inventory, episode, role, *, cache_bytes=BUDGET["cache_bytes"]):
        if (
            type(episode) is not int
            or episode not in range(3)
            or role not in ("adaptation", "evaluation")
            or type(cache_bytes) is not int
            or not 0 < cache_bytes <= BUDGET["cache_bytes"]
        ):
            raise ValueError("invalid view capability/cache")
        self.root = Path(inventory["root"])
        self.entries = {x["image_id"]: x for x in inventory["entries"] if x["episode"] == episode and x["role"] == role}
        self.ids = list(self.entries)
        self.cache = OrderedDict()
        self.cache_bytes = cache_bytes
        self.used = 0
        self.cache_peak_bytes = 0

    @contextmanager
    def image(self, name):
        if name not in self.entries:
            raise ValueError("image outside role/episode capability")
        entry = self.entries[name]
        data = read_bytes(self.root / entry["stage_name"], 50 * 1024**2)
        if len(data) != entry["encoded_bytes"] or hashlib.sha256(data).hexdigest() != entry["encoded_sha256"]:
            raise ValueError("staged image bytes changed")
        size = entry["width"] * entry["height"] * 3
        if size > self.cache_bytes:
            raise ValueError("decoded RGB cache ceiling")
        if name in self.cache:
            image = self.cache.pop(name)
            self.cache[name] = image
        else:
            while self.used + size > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.used -= old.width * old.height * 3
                old.close()
            image = rgb_from_bytes(data, self.cache_bytes)
            if image.size != (entry["width"], entry["height"]) or pixel_digest(image) != entry["decoded_rgb_sha256"]:
                image.close()
                raise ValueError("decoded image identity mismatch")
            self.cache[name] = image
            self.used += size
            self.cache_peak_bytes = max(self.cache_peak_bytes, self.used)
        temporary = image.copy()
        try:
            yield temporary
        finally:
            temporary.close()

    def close(self):
        for image in self.cache.values():
            image.close()
        self.cache.clear()
        self.used = 0
