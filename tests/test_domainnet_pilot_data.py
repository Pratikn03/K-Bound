"""Painting ingestion contracts; all archives and annotations here are synthetic."""

import hashlib
import importlib
import importlib.util
import io
import json
import stat
import struct
import subprocess
import sys
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

DESIGN = Path(__file__).resolve().parents[1] / "protocols/confirmatory_v2/DOMAINNET_DEV_PILOT_DESIGN_v1.json"


def module():
    assert importlib.util.find_spec("experiments.kbound.domainnet.pilot_data") is not None, (
        "painting preparation missing"
    )
    return importlib.import_module("experiments.kbound.domainnet.pilot_data")


def png(color=(1, 2, 3)):
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color).save(buffer, format="PNG")
    return buffer.getvalue()


def fixture(tmp_path, entries=None, raw_list=None, extras=(), source_group="0" * 64, source_map=None):
    entries = (
        entries
        if entries is not None
        else [("painting/ant/z.png", 0, png()), ("painting/bear/a.png", 1, png((4, 5, 6)))]
    )
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as stream:
        for path, _, raw in entries:
            stream.writestr(path, raw)
        for info, raw in extras:
            stream.writestr(info, raw)
    listing = tmp_path / "synthetic.txt"
    raw = raw_list if raw_list is not None else "".join(f"{path} {label}\n" for path, label, _ in entries).encode()
    listing.write_bytes(raw)
    source = tmp_path / "synthetic-source.json"
    source.write_text(
        json.dumps(
            {
                "schema": "kbound-domainnet-source-inventory/2",
                "domain": "clipart",
                "role": "source",
                "status": "SOURCE_ONLY_PREPARED_NOT_SCIENTIFIC_LOCK",
                "identity": {"class_count": 2},
                "rows": [
                    {"path": f"clipart/{name}/x.png", "label": label, "group_id": source_group}
                    for name, label in (source_map or [("ant", 0), ("bear", 1)])
                ],
            }
        )
    )
    identity = module().SyntheticIdentity(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        list_sha256=hashlib.sha256(raw).hexdigest(),
        list_git_blob=hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(),
        source_inventory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        class_count=2,
    )
    return archive, listing, source, identity


def prepare(tmp_path, **kwargs):
    archive, listing, source, identity = fixture(tmp_path, **kwargs)
    return module().prepare_painting(archive, listing, source, DESIGN, tmp_path / "out", synthetic_identity=identity)


@pytest.mark.parametrize(
    "rank,want",
    [
        (0, "DEV_fit"),
        ((3 * 2**256) // 5 - 1, "DEV_fit"),
        ((3 * 2**256) // 5, "DEV_radius"),
        ((4 * 2**256) // 5 - 1, "DEV_radius"),
        ((4 * 2**256) // 5, "DEV_check"),
        (2**256 - 1, "DEV_check"),
    ],
)
def test_literal_partition_boundaries(rank, want):
    assert module().partition_for_rank(rank) == want


def test_literal_packing_groups_overshoot_and_incomplete_cell_tail():
    # A changed threshold, split group or rescored partial cell breaks these literals.
    groups = [["a0", "a1", "a2"], ["b0"], ["c0", "c1"], ["d0", "d1", "d2"], ["e0"], ["f0"]]
    packed = module().pack_ordered_groups(groups, {"U": 2, "V": 2, "E": 3})
    assert packed == {
        "cells": [{"U": ["a0", "a1", "a2"], "V": ["b0", "c0", "c1"], "E": ["d0", "d1", "d2"]}],
        "tail": {"U": ["e0", "f0"], "V": [], "E": []},
    }
    flat = [s for cell in packed["cells"] for window in cell.values() for s in window]
    flat += [s for window in packed["tail"].values() for s in window]
    assert flat == ["a0", "a1", "a2", "b0", "c0", "c1", "d0", "d1", "d2", "e0", "f0"]


def test_literal_salted_group_order_is_independent_of_labels():
    rows = [
        {"sample_id": f"s{i}", "group_id": str(i) * 64, "label": i % 2, "image_sha256": "a" * 64}
        for i in [5, 4, 3, 2, 1, 0]
    ]
    packing, _, _ = module()._packing(rows)
    assert packing == {
        "DEV_fit": {"cells": [], "tail": {"U": ["s0", "s4", "s3", "s2", "s5"], "V": [], "E": []}},
        "DEV_radius": {"cells": [], "tail": {"U": [], "V": [], "E": []}},
        "DEV_check": {"cells": [], "tail": {"U": ["s1"], "V": [], "E": []}},
    }
    for row in rows:
        row["label"] = 1 - row["label"]
    assert module()._packing(rows)[0] == packing


def test_default_packing_uses_exact_64_64_128_boundaries():
    groups = [
        [f"u{i}" for i in range(64)],
        [f"v{i}" for i in range(64)],
        [f"e{i}" for i in range(127)],
        ["last"],
        ["tail"],
    ]
    packed = module().pack_ordered_groups(groups)
    assert len(packed["cells"]) == 1
    cell = packed["cells"][0]
    assert {w: len(ids) for w, ids in cell.items()} == {"U": 64, "V": 64, "E": 128}
    assert cell["U"][0] == "u0" and cell["U"][-1] == "u63"
    assert cell["V"][0] == "v0" and cell["V"][-1] == "v63"
    assert cell["E"][-1] == "last"
    assert packed["tail"] == {"U": ["tail"], "V": [], "E": []}


def test_small_geometry_preserves_order_labels_hashes_and_immutable_private_public_outputs(tmp_path):
    result = prepare(tmp_path)
    assert result["status"] == "INCONCLUSIVE_DATA_GEOMETRY" and result["eligible_for_pilot"] is False
    out = tmp_path / "out"
    inventory = json.loads((out / "inventory.json").read_text())
    assert [r["path"] for r in inventory["rows"]] == ["painting/ant/z.png", "painting/bear/a.png"]
    assert [r["label"] for r in inventory["rows"]] == [0, 1]
    assert inventory["rows"][0]["group_id"] == hashlib.sha256(struct.pack(">QQ", 1, 1) + b"\x01\x02\x03").hexdigest()
    assert inventory["rows"][0]["image_sha256"] == hashlib.sha256(png()).hexdigest()
    opaque = hashlib.sha256(b"kbound-painting-pilot-v1:sample:painting/ant/z.png").hexdigest()
    assert inventory["rows"][0]["sample_id"] == opaque
    assert json.loads((out / "labels.json").read_text())[opaque] == 0
    public = json.loads((out / "public_manifest.json").read_text())
    assert len(public["rows"]) == 2
    for row in public["rows"]:
        assert set(row) == {
            "sample_id",
            "official_index",
            "image_sha256",
            "group_id",
            "width",
            "height",
            "partition",
            "cell",
            "window",
        }
    for forbidden in ("label", "painting/", "ant", "bear"):
        assert forbidden not in (out / "public_manifest.json").read_text()
    assert sum(p["tail_rows"] for p in result["counts"].values()) == 2
    for name in ("inventory.json", "public_manifest.json", "labels.json"):
        assert result["output_sha256"][name] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    for path in out.iterdir():
        assert not path.stat().st_mode & stat.S_IWUSR
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        module().prepare_painting(tmp_path / "absent", tmp_path / "absent", tmp_path / "absent", DESIGN, out)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_conflicting_labels_remain_one_group_and_source_overlap_stops(tmp_path):
    group = hashlib.sha256(struct.pack(">QQ", 1, 1) + b"\x01\x02\x03").hexdigest()
    result = prepare(
        tmp_path, entries=[("painting/ant/z.png", 0, png()), ("painting/bear/a.png", 1, png())], source_group=group
    )
    assert result["status"] == "STOP_SOURCE_OVERLAP" and result["eligible_for_pilot"] is False
    inventory = json.loads((tmp_path / "out/inventory.json").read_text())
    assert inventory["conflicts"]["group_count"] == 1 and inventory["conflicts"]["row_count"] == 2
    assert inventory["conflicts"]["groups"][0]["label_multiplicity"] == {"0": 1, "1": 1}
    assert inventory["source_overlap"]["group_count"] == 1 and inventory["source_overlap"]["painting_row_count"] == 2
    assert inventory["source_overlap"]["source_row_count"] == 2
    assert len({(r["partition"], r["cell"], r["window"]) for r in inventory["rows"]}) == 1


@pytest.mark.parametrize(
    "bad",
    [
        b"painting/ant/../x.png 0\npainting/bear/a.png 1\n",
        b"painting/ant/z.png 00\npainting/bear/a.png 1\n",
        b"painting/ant/z.png -1\npainting/bear/a.png 1\n",
        b"painting/ant/z.png 2\npainting/bear/a.png 1\n",
        b"painting/ant/z.png 0\npainting/ant/z.png 0\npainting/bear/a.png 1\n",
        b"painting/ant/z.png 0\npainting/Ant/Z.png 0\npainting/bear/a.png 1\n",
        b"painting/ant/z.png 0\npainting/bear/a.png 0\n",
        b"painting/ant/z.png 0\n",
        b"painting/ant/z.png 0\n\npainting/bear/a.png 1\n",
        b"painting/ant/z.png 0\rpainting/bear/a.png 1\n",
    ],
)
def test_strict_list_rejects_paths_labels_aliases_and_noncanonical_lines(tmp_path, bad):
    with pytest.raises(ValueError):
        prepare(tmp_path, raw_list=bad)
    assert not (tmp_path / "out/summary.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "sketch/a/x.png",
        "painting/../x.png",
        "painting/ant/Z.png",
        "painting/a/e\u0301.png",
        "painting/a/x\x7f.png",
        "painting/a/b/c.png",
        "painting\\ant\\x.png",
        "painting/a./x.png",
    ],
)
def test_all_zip_headers_rejected_even_when_unlisted(tmp_path, name):
    with pytest.raises(ValueError):
        prepare(tmp_path, extras=[(name, png())])
    assert not (tmp_path / "out/summary.json").exists()


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK])
def test_zip_special_members_rejected(tmp_path, kind):
    info = zipfile.ZipInfo("painting/ant/extra.png")
    info.create_system = 3
    info.external_attr = (kind | 0o644) << 16
    with pytest.raises(ValueError):
        prepare(tmp_path, extras=[(info, b"payload")])


@pytest.mark.parametrize(
    "fault", ["crc", "encryption", "strong_encryption", "nul_name", "oversize", "members", "total"]
)
def test_zip_crc_encryption_alias_and_resource_bounds(tmp_path, monkeypatch, fault):
    archive, listing, source, identity = fixture(tmp_path, extras=[("painting/ant/extra.png", png())])
    payload = bytearray(archive.read_bytes())
    central = payload.index(b"PK\x01\x02")
    if fault == "crc":
        # Stored first image bytes: mutation retains valid ZIP structure but invalid CRC.
        local = payload.index(b"PK\x03\x04")
        name_size, extra_size = struct.unpack_from("<HH", payload, local + 26)
        payload[local + 30 + name_size + extra_size] ^= 1
    elif fault in {"encryption", "strong_encryption"}:
        # Unlisted final member: header screening must catch this before image iteration.
        central = payload.rindex(b"PK\x01\x02")
        struct.pack_into("<H", payload, central + 8, 1 if fault == "encryption" else 64)
    elif fault == "nul_name":
        payload[central + 46 + len("painting/ant/z")] = 0
    elif fault == "oversize":
        struct.pack_into("<I", payload, central + 24, 50 * 1024**2 + 1)
    elif fault == "members":
        monkeypatch.setattr(module().source_data, "MAX_MEMBERS", 2)
    elif fault == "total":
        monkeypatch.setattr(module().source_data, "MAX_TOTAL_BYTES", 100)
    archive.write_bytes(payload)
    identity = replace(identity, archive_sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        module().prepare_painting(archive, listing, source, DESIGN, tmp_path / "out", synthetic_identity=identity)
    assert not (tmp_path / "out/summary.json").exists()


@pytest.mark.parametrize("fault", ["archive", "list", "blob", "source", "design", "class_map", "missing", "corrupt"])
def test_identity_mapping_missing_and_corrupt_image_fail_without_success(tmp_path, fault):
    kwargs = {"source_map": [("ant", 1), ("bear", 0)]} if fault == "class_map" else {}
    if fault == "missing":
        kwargs["raw_list"] = b"painting/ant/missing.png 0\npainting/bear/a.png 1\n"
    if fault == "corrupt":
        kwargs["entries"] = [("painting/ant/z.png", 0, b"broken"), ("painting/bear/a.png", 1, png())]
    archive, listing, source, identity = fixture(tmp_path, **kwargs)
    design = DESIGN
    if fault in {"archive", "list", "blob", "source"}:
        field = {
            "archive": "archive_sha256",
            "list": "list_sha256",
            "blob": "list_git_blob",
            "source": "source_inventory_sha256",
        }[fault]
        identity = replace(identity, **{field: "f" * (40 if fault == "blob" else 64)})
    if fault == "design":
        design = tmp_path / "changed.json"
        design.write_bytes(DESIGN.read_bytes() + b" ")
    with pytest.raises((ValueError, OSError)):
        module().prepare_painting(archive, listing, source, design, tmp_path / "out", synthetic_identity=identity)
    assert not (tmp_path / "out/summary.json").exists()


def test_decode_warnings_are_errors_without_caller_warning_flags(tmp_path, monkeypatch):
    archive, listing, source, identity = fixture(tmp_path)
    original = module().source_data.decode_rgb

    def warned(raw):
        warnings.warn("synthetic decoder warning", UserWarning, stacklevel=2)
        return original(raw)

    monkeypatch.setattr(module().source_data, "decode_rgb", warned)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(UserWarning):
            module().prepare_painting(archive, listing, source, DESIGN, tmp_path / "out", synthetic_identity=identity)
    assert not (tmp_path / "out/summary.json").exists()


def test_partial_publication_retains_files_and_has_no_success_receipt(tmp_path, monkeypatch):
    archive, listing, source, identity = fixture(tmp_path)
    original = module().source_data.write_new_json

    def failing(path, value):
        if path.name == "labels.json":
            raise OSError("synthetic storage failure")
        original(path, value)

    monkeypatch.setattr(module().source_data, "write_new_json", failing)
    with pytest.raises(OSError, match="synthetic storage"):
        module().prepare_painting(archive, listing, source, DESIGN, tmp_path / "out", synthetic_identity=identity)
    assert (tmp_path / "out/inventory.json").exists()
    assert (tmp_path / "out/public_manifest.json").exists()
    assert not (tmp_path / "out/summary.json").exists()


def test_synthetic_injection_rejects_real_identities_and_full_class_count(tmp_path):
    *_, identity = fixture(tmp_path)
    for changes in (
        {"class_count": 126},
        {"class_count": True},
        {"archive_sha256": module().ARCHIVE_SHA256},
        {"list_sha256": module().LIST_SHA256},
        {"list_git_blob": module().LIST_GIT_BLOB},
        {"source_inventory_sha256": module().SOURCE_INVENTORY_SHA256},
    ):
        with pytest.raises(ValueError):
            replace(identity, **changes)


def test_cli_rejects_synthetic_identity_overrides_and_unpinned_inputs(tmp_path):
    archive, listing, source, _ = fixture(tmp_path)
    args = [
        sys.executable,
        "-Werror",
        "-m",
        "experiments.kbound.domainnet.pilot_data",
        "--archive",
        str(archive),
        "--painting-list",
        str(listing),
        "--source-inventory",
        str(source),
        "--design",
        str(DESIGN),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    for extra in ([], ["--class-count", "2"], ["--archive-sha256", "f" * 64], ["--partition-salt", "changed"]):
        run = subprocess.run(args + extra, capture_output=True, text=True, check=False)
        assert run.returncode != 0
        assert not (tmp_path / "out/summary.json").exists()


@pytest.mark.parametrize("overlap", [False, True])
def test_cli_entrypoint_returns_nonzero_for_complete_diagnostic_inventory(tmp_path, monkeypatch, overlap):
    source_group = hashlib.sha256(struct.pack(">QQ", 1, 1) + b"\x01\x02\x03").hexdigest() if overlap else "0" * 64
    archive, listing, source, identity = fixture(tmp_path, source_group=source_group)
    original = module().prepare_painting

    def synthetic(*args):
        return original(*args, synthetic_identity=identity)

    monkeypatch.setattr(module(), "prepare_painting", synthetic)
    result = module().main(
        [
            "--archive",
            str(archive),
            "--painting-list",
            str(listing),
            "--source-inventory",
            str(source),
            "--design",
            str(DESIGN),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert result == 2
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["status"] == ("STOP_SOURCE_OVERLAP" if overlap else "INCONCLUSIVE_DATA_GEOMETRY")
    assert summary["eligible_for_pilot"] is False


def test_full_synthetic_geometry_prepares_all_rows_once_deterministically(tmp_path):
    # 360 small decoded groups, each 128 rows: group-preserving overshoot with real production geometry.
    entries = [
        (f"painting/{'ant' if g % 2 == 0 else 'bear'}/{g:03d}-{m:03d}.png", g % 2, png((g // 256, g % 256, 7)))
        for g in range(360)
        for m in range(128)
    ]
    archive, listing, source, identity = fixture(tmp_path, entries=entries)
    out = tmp_path / "out"
    result = module().prepare_painting(archive, listing, source, DESIGN, out, synthetic_identity=identity)
    assert result["status"] == "DEV_ONLY_PREPARED_NOT_SCIENTIFIC_LOCK" and result["eligible_for_pilot"] is True
    inventory = json.loads((out / "inventory.json").read_text())
    assert len(inventory["rows"]) == 46080
    assert {p: (c["decoded_groups"], c["complete_cells"], c["tail_rows"]) for p, c in result["counts"].items()} == {
        "DEV_fit": (221, 73, 256),
        "DEV_radius": (72, 24, 0),
        "DEV_check": (67, 22, 128),
    }
    assert len({r["sample_id"] for r in inventory["rows"]}) == 46080
    seen = set()
    for partition, packing in inventory["packing"].items():
        for cell in packing["cells"]:
            windows = [set(cell[w]) for w in ("U", "V", "E")]
            assert [len(w) for w in windows] == [128, 128, 128]
            assert not (windows[0] & windows[1] or windows[1] & windows[2] or windows[0] & windows[2])
            for window in windows:
                assert not seen & window
                seen.update(window)
        for window in packing["tail"].values():
            assert not seen & set(window)
            seen.update(window)
        assert (
            result["counts"][partition]["complete_cells"]
            >= {"DEV_fit": 40, "DEV_radius": 10, "DEV_check": 20}[partition]
        )
    assert seen == {r["sample_id"] for r in inventory["rows"]}
    assert all(
        len({(r["partition"], r["cell"], r["window"]) for r in group["members"]}) == 1 for group in inventory["groups"]
    )
    again = module().prepare_painting(archive, listing, source, DESIGN, tmp_path / "again", synthetic_identity=identity)
    assert again == result
    assert all(
        (out / name).read_bytes() == (tmp_path / "again" / name).read_bytes()
        for name in ["inventory.json", "public_manifest.json", "labels.json", "summary.json"]
    )
