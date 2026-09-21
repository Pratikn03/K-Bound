"""Synthetic ZIPs only. Production manifest gate is exercised separately."""

import hashlib
import importlib
import io
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts"))


def module():
    return importlib.import_module("domainnet_feasibility_images")


def picture(i, note=""):
    out = io.BytesIO()
    meta = PngImagePlugin.PngInfo()
    meta.add_text("note", note)
    Image.new("RGB", (8, 8), (i, 2, 3)).save(out, "PNG", pnginfo=meta)
    return out.getvalue()


def inputs(tmp, values=None):
    values = values or [picture(1), picture(2), picture(3)]
    path = tmp / "painting.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, value in enumerate(values):
            z.writestr(f"painting/a/{i}.png", value)
        z.writestr("real/forbidden/image.png", b"forbidden")
    members = []
    with zipfile.ZipFile(path) as z:
        for i in range(3):
            name = f"painting/a/{i}.png"
            info = z.getinfo(name)
            members.append(
                {
                    "image_id": name,
                    "episode": 0 if i < 2 else 1,
                    "role": "adaptation" if i == 0 else "evaluation",
                    "stage_name": f"{i:05d}.bin",
                    "crc32": info.CRC,
                    "encoded_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                }
            )
    identity = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return identity, members


def test_stage_only_selected_once_and_role_view(tmp_path, monkeypatch):
    m = module()
    archive, members = inputs(tmp_path)
    opened = []
    native = zipfile.ZipFile.open

    def watched(self, name, *a, **kw):
        name = name.filename if hasattr(name, "filename") else name
        assert name.startswith("painting/")
        opened.append(name)
        return native(self, name, *a, **kw)

    monkeypatch.setattr(zipfile.ZipFile, "open", watched)
    inventory = m._stage_members(archive, members, [], tmp_path / "stage")
    assert opened == [x["image_id"] for x in members]
    view = m.ImageView(inventory, 0, "adaptation", cache_bytes=192)
    assert view.ids == ["painting/a/0.png"]
    with view.image(view.ids[0]) as image:
        assert image.getpixel((0, 0)) == (1, 2, 3)
    with pytest.raises(ValueError):
        view.image("painting/a/1.png").__enter__()
    with pytest.raises(ValueError):
        view.image("painting/a/2.png").__enter__()


@pytest.mark.parametrize("kind", ["encoded", "pixel", "old", "corrupt"])
def test_duplicates_or_corrupt_stop_without_replacement(tmp_path, kind):
    m = module()
    values = [picture(1), picture(2), picture(3)]
    prior = []
    if kind == "encoded":
        values[1] = values[0]
    if kind == "pixel":
        values[2] = picture(1, "different bytes")
    if kind == "corrupt":
        values[0] = b"not a picture"
    if kind == "old":
        prior = [{"encoded_sha256": hashlib.sha256(values[0]).hexdigest(), "decoded_rgb_sha256": "0" * 64}]
    archive, members = inputs(tmp_path, values)
    with pytest.raises((ValueError, OSError)):
        m._stage_members(archive, members, prior, tmp_path / "stage")


@pytest.mark.parametrize("kind", ["parent", "leaf", "metadata", "duplicate_stage", "outside_role"])
def test_staging_checks_before_payload(tmp_path, monkeypatch, kind):
    m = module()
    archive, members = inputs(tmp_path)
    output = tmp_path / "stage"
    if kind == "parent":
        link = tmp_path / "link"
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / "stage"
    if kind == "leaf":
        output.symlink_to(tmp_path, target_is_directory=True)
    if kind == "metadata":
        members[0]["encoded_bytes"] += 1
    if kind == "duplicate_stage":
        members[1]["stage_name"] = members[0]["stage_name"]
    if kind == "outside_role":
        members[0]["role"] = "calibration"

    def trap(*a, **kw):
        raise AssertionError("payload opened before metadata validation")

    monkeypatch.setattr(zipfile.ZipFile, "open", trap)
    with pytest.raises((ValueError, OSError)):
        m._stage_members(archive, members, [], output)


def test_cache_authenticates_changed_bytes_even_on_hit(tmp_path):
    m = module()
    archive, members = inputs(tmp_path)
    inventory = m._stage_members(archive, members, [], tmp_path / "stage")
    view = m.ImageView(inventory, 0, "adaptation", cache_bytes=192)
    with view.image(view.ids[0]):
        pass
    path = tmp_path / "stage/00000.bin"
    path.write_bytes(picture(42))
    with pytest.raises(ValueError):
        view.image(view.ids[0]).__enter__()


def test_lru_bound_and_no_cache_object_escape(tmp_path):
    m = module()
    archive, members = inputs(tmp_path)
    for member in members:
        member["episode"] = 0
        member["role"] = "adaptation"
    inventory = m._stage_members(archive, members, [], tmp_path / "stage")
    view = m.ImageView(inventory, 0, "adaptation", cache_bytes=192)
    for name in view.ids:
        with view.image(name) as image:
            image.putpixel((0, 0), (99, 99, 99))
    assert view.cache_peak_bytes <= 192 and len(view.cache) == 1
    with view.image(view.ids[0]) as image:
        assert image.getpixel((0, 0)) == (1, 2, 3)
    tiny = m.ImageView(inventory, 0, "adaptation", cache_bytes=191)
    with pytest.raises(ValueError):
        tiny.image(tiny.ids[0]).__enter__()


def test_prior_inventory_uses_actual_capacity_packet_schema(monkeypatch, tmp_path):
    m = module()
    prior = [{"image_id": f"painting/a/{i}.jpg"} for i in range(48)]
    manifest = {
        "prior_inventory": {"path": "bound", "sha256": "bound"},
        "excluded_ids": [x["image_id"] for x in prior],
        "archive": {},
        "members": [],
    }
    monkeypatch.setattr(m, "validate_manifest", lambda value: value)
    monkeypatch.setattr(m, "read_json", lambda *a: {"inventory": {"entries": prior}})
    monkeypatch.setattr(m, "_stage_members", lambda a, b, old, root: old)
    assert m.stage_selected(manifest, tmp_path) == prior
