from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
from PIL import Image


@lru_cache(maxsize=1)
def module():
    path = Path(__file__).parents[1] / "experiments/kbound/reference_source/population.py"
    assert path.is_file(), "full reference population indexer is not implemented"
    spec = importlib.util.spec_from_file_location("reference_population", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def rx(tmp_path):
    root = tmp_path / "rx"
    root.mkdir()
    metadata = root / "metadata.csv"
    metadata.write_text(
        "dataset,experiment,plate,well,site,sirna_id\n"
        "train,HEPG2-01,1,A01,1,DO_NOT_READ\n"
        "train,HEPG2-01,1,A01,2,DO_NOT_READ\n"
        "val,HEPG2-02,2,B01,1,DO_NOT_READ\n"
        "test,HEPG2-03,3,C01,2,DO_NOT_READ\n"
    )
    paths = [
        "images/HEPG2-01/Plate1/A01_s1.png", "images/HEPG2-01/Plate1/A01_s2.png",
        "images/HEPG2-02/Plate2/B01_s1.png", "images/HEPG2-03/Plate3/C01_s2.png",
    ]
    for i, path in enumerate(paths):
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(20 * i, 10, 40)).save(dest)
    return root, metadata, paths


def build(rx, tmp_path):
    root, metadata, _ = rx
    return module().build_population(
        "rxrx1", root, sha(metadata), tmp_path / "inventory",
        expected_counts={"train": 1, "id_test": 1, "val": 1, "test": 1},
    )


def test_rxrx_official_site_two_is_never_source_training(rx):
    rows = list(module().metadata_entries("rxrx1", rx[0]))
    assert [row["split"] for row in rows] == ["train", "id_test", "val", "test"]
    assert [row["row_id"] for row in rows] == [0, 1, 2, 3]
    assert [row["path"] for row in rows] == rx[2]
    assert all("sirna_id" not in row and "label" not in row for row in rows)


def test_camelyon_official_centers_override_csv_and_preserve_patient_padding(tmp_path):
    (tmp_path / "metadata.csv").write_text(
        ",patient,node,x_coord,y_coord,tumor,slide,center,split\n"
        "0,004,1,10,20,DO_NOT_READ,3,1,0\n"
        "1,005,2,30,40,DO_NOT_READ,4,2,0\n"
        "2,006,1,0,0,DO_NOT_READ,5,0,1\n"
    )
    rows = list(module().metadata_entries("camelyon17", tmp_path))
    assert [r["split"] for r in rows] == ["val", "test", "id_val"]
    assert rows[0]["path"] == "patches/patient_004_node_1/patch_patient_004_node_1_x_10_y_20.png"
    assert rows[0]["group"] == {"center": "1", "slide": "3"}


def test_iwildcam_paths_do_not_filter_a_missing_image_or_access_label(tmp_path):
    (tmp_path / "metadata.csv").write_text(
        "split,location_remapped,sequence_remapped,filename,y\n"
        "id_val,8,15,not-downloaded.jpg,DO_NOT_READ\n"
    )
    rows = list(module().metadata_entries("iwildcam", tmp_path))
    assert rows == [{"row_id": 0, "path": "train/not-downloaded.jpg", "split": "id_val",
                     "group": {"location": "8", "sequence": "15"}}]


def test_index_binds_actual_images_and_is_not_claimed_as_full_reference(rx, tmp_path):
    result = build(rx, tmp_path)
    assert result["complete"] is True
    assert result["reference_population_complete"] is False
    assert result["counts"] == {"train": 1, "id_test": 1, "val": 1, "test": 1}
    index = tmp_path / "inventory/image-index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    assert result["index_sha256"] == sha(index)
    assert rows[0]["sha256"] == sha(rx[0] / rx[2][0])
    assert rows[0]["bytes"] == (rx[0] / rx[2][0]).stat().st_size
    assert "DO_NOT_READ" not in index.read_text()
    assert "sirna_id" not in index.read_text()


@pytest.mark.parametrize("kind", ["missing", "corrupt", "symlink"])
def test_unusable_image_fails_instead_of_substituting(rx, tmp_path, kind):
    image = rx[0] / rx[2][0]
    image.unlink()
    if kind == "corrupt":
        image.write_bytes(b"not an image")
    elif kind == "symlink":
        image.symlink_to(rx[0] / rx[2][1])
    mod = module()
    with pytest.raises(mod.PopulationError, match="image"):
        build(rx, tmp_path)
    assert json.loads((tmp_path / "inventory/completion.json").read_text())["complete"] is False


def test_fifo_expected_path_fails_promptly_and_writes_incomplete_receipt(rx, tmp_path):
    image = rx[0] / rx[2][0]
    image.unlink()
    os.mkfifo(image)
    output = tmp_path / "fifo-inventory"
    source = Path(module().__file__)
    script = f"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("fifo_population", {str(source)!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    mod.build_population(
        "rxrx1",
        Path({str(rx[0])!r}),
        {sha(rx[1])!r},
        Path({str(output)!r}),
        expected_counts={{"train": 1, "id_test": 1, "val": 1, "test": 1}},
    )
except mod.PopulationError:
    raise SystemExit(0)
raise SystemExit(3)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((output / "completion.json").read_text())
    assert receipt["complete"] is False
    assert "regular file" in receipt["error"]


def test_decompression_bomb_writes_incomplete_receipt(rx, tmp_path, monkeypatch):
    def raise_decompression_bomb(*_args, **_kwargs):
        raise Image.DecompressionBombError("synthetic oversized image")

    monkeypatch.setattr(Image, "open", raise_decompression_bomb)
    mod = module()
    with pytest.raises(mod.PopulationError, match="synthetic oversized image"):
        build(rx, tmp_path)
    receipt = json.loads((tmp_path / "inventory/completion.json").read_text())
    assert receipt["complete"] is False
    assert "synthetic oversized image" in receipt["error"]


def test_reference_convertible_grayscale_is_indexed_with_original_mode(rx, tmp_path):
    grayscale = rx[0] / rx[2][0]
    Image.new("L", (8, 8), color=37).save(grayscale)
    result = build(rx, tmp_path)
    assert result["complete"] is True
    first = json.loads((tmp_path / "inventory/image-index.jsonl").read_text().splitlines()[0])
    assert first["original_mode"] == "L"
    assert first["width"] == 8
    assert first["height"] == 8


def test_existing_output_is_preserved(rx, tmp_path):
    out = tmp_path / "inventory"
    out.mkdir()
    sentinel = out / "do-not-overwrite"
    sentinel.write_text("keep")
    with pytest.raises(module().PopulationError, match="output"):
        build(rx, tmp_path)
    assert sentinel.read_text() == "keep"


def test_wrong_metadata_identity_stops_before_creating_output(rx, tmp_path):
    mod = module()
    with pytest.raises(mod.PopulationError, match="metadata.*SHA"):
        mod.build_population("rxrx1", rx[0], "0" * 64, tmp_path / "inventory")
    assert not (tmp_path / "inventory").exists()


def test_incomplete_official_population_never_gets_complete_receipt(rx, tmp_path):
    mod = module()
    with pytest.raises(mod.PopulationError, match="counts"):
        mod.build_population("rxrx1", rx[0], sha(rx[1]), tmp_path / "inventory")
    assert json.loads((tmp_path / "inventory/completion.json").read_text())["complete"] is False


def test_repeated_metadata_path_is_not_counted_twice(rx, tmp_path):
    lines = rx[1].read_text().splitlines()
    rx[1].write_text("\n".join(lines + [lines[1]]) + "\n")
    with pytest.raises(module().PopulationError, match="duplicate"):
        build(rx, tmp_path)


def test_metadata_path_escape_is_rejected(tmp_path):
    (tmp_path / "metadata.csv").write_text(
        "split,location_remapped,sequence_remapped,filename\ntrain,1,1,../../outside.jpg\n"
    )
    with pytest.raises(module().PopulationError, match="path"):
        list(module().metadata_entries("iwildcam", tmp_path))


def test_symlink_ancestor_in_input_root_is_rejected(rx, tmp_path):
    link = tmp_path / "linked"
    link.symlink_to(rx[0], target_is_directory=True)
    with pytest.raises(module().PopulationError, match="symlink"):
        list(module().metadata_entries("rxrx1", link))
