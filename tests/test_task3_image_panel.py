"""The real-image smoke panel is explicit, immutable, and outcome-independent."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/task3_image_panel.py"


@pytest.fixture
def panel():
    assert SCRIPT.exists(), "real-image panel builder is missing"
    spec = importlib.util.spec_from_file_location("task3_image_panel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (256, 256), (40, 80, 120)).save(stream, format="JPEG")
    return stream.getvalue()


@pytest.fixture
def fixture(tmp_path):
    clean, corrupt = tmp_path / "clean", tmp_path / "corrupt"
    names = [f"n00000001/ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, 161)]
    for name in names:
        for path in (clean / name, corrupt / "gaussian_noise/5" / name):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(jpeg())
    return clean, corrupt, names


def test_plan_fixed_before_outcomes_and_disjoint_by_underlying_image(panel, fixture):
    clean, corrupt, names = fixture
    plan = panel.plan_smoke(names, seed=0, source_count=32, target_count=104)
    assert plan == panel.plan_smoke(names, seed=0, source_count=32, target_count=104)
    assert len(plan["source"]) == 32 and len(plan["target"]) == 104
    assert not {Path(p).name for p in plan["source"]} & {Path(p).name for p in plan["target"]}
    assert all("label" not in key and "accuracy" not in key for key in plan)
    assert panel.plan_smoke(names, seed=1, source_count=32, target_count=104) != plan


@pytest.mark.parametrize("names", [["../secret.JPEG"], ["/secret.JPEG"],
                                    ["n00000001/ILSVRC2012_val_00000001.JPEG"] * 160])
def test_reject_unsafe_or_duplicate_inventory_before_reads(panel, names):
    with pytest.raises(ValueError):
        panel.plan_smoke(names, seed=0, source_count=32, target_count=104)


def test_source_fraction_is_fixed_and_too_small_pool_fails(panel, fixture):
    _, _, names = fixture
    with pytest.raises(ValueError, match="source"):
        panel.plan_smoke(names[:120], seed=0, source_count=32, target_count=104)
    with pytest.raises(ValueError, match="warmup"):
        panel.plan_smoke(names, seed=0, source_count=32, target_count=96)


def test_decode_and_hash_bind_identical_image_bytes(panel, fixture):
    clean, _, names = fixture
    row = panel.inspect_image(clean, names[0])
    assert row["sha256"] == hashlib.sha256((clean / names[0]).read_bytes()).hexdigest()
    assert row["width"] == 256 and row["height"] == 256
    assert row["relative_path"] == names[0]
    assert row["decoded_rgb"] is True


def test_symlink_and_nonimage_fail_closed(panel, fixture, tmp_path):
    clean, _, names = fixture
    path = clean / names[0]
    path.unlink()
    outside = tmp_path / "outside.JPEG"
    outside.write_bytes(jpeg())
    path.symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        panel.inspect_image(clean, names[0])
    path.unlink()
    path.write_bytes(b"not a jpeg")
    with pytest.raises((OSError, ValueError)):
        panel.inspect_image(clean, names[0])


def test_symlink_ancestor_rejected(panel, fixture, tmp_path):
    clean, _, names = fixture
    alias = tmp_path / "alias"
    alias.symlink_to(clean, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        panel.inspect_image(alias, names[0])


def test_inventory_receipt_requires_full_metadata_and_exact_path_hash(panel, fixture, tmp_path):
    clean, _, names = fixture
    inventory = tmp_path / "relative_paths.txt"
    inventory.write_text("\n".join(names) + "\n")
    receipt = {"status": "LABEL_PATHS_VERIFIED_NOT_IMAGE_CONTENT", "root": str(clean),
               "image_count": 50000, "class_count": 1000,
               "class_mapping_authenticated": True, "directory_inventory_complete": True,
               "devkit_sha256": panel.DEVKIT_SHA256,
               "relative_paths_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest()}
    audit = tmp_path / "report.json"
    audit.write_text(json.dumps(receipt))
    # Counts in a report do not replace validation of the actual inventory.
    with pytest.raises(ValueError, match="50000"):
        panel.load_inventory(clean, audit, inventory)
    receipt["class_mapping_authenticated"] = False
    audit.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="metadata"):
        panel.load_inventory(clean, audit, inventory)


def test_mutated_bytes_rejected_before_model_read(panel, fixture):
    clean, _, names = fixture
    row = panel.inspect_image(clean, names[0])
    (clean / names[0]).write_bytes(jpeg() + b"changed")
    with pytest.raises(ValueError, match="digest"):
        panel.read_bound_image(clean, row)


def test_root_ancestor_swap_cannot_redirect_reads(panel, tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    root = parent / "clean"
    foreign = tmp_path / "foreign"
    name = "n00000001/ILSVRC2012_val_00000001.JPEG"
    for base in (root, foreign / "clean"):
        (base / name).parent.mkdir(parents=True)
        (base / name).write_bytes(jpeg())
    original_open = panel.os.open
    swapped = False
    def race(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and (Path(path) == root or str(path) == "parent"):
            parent.rename(tmp_path / "original")
            parent.symlink_to(foreign, target_is_directory=True)
            swapped = True
        return original_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(panel.os, "open", race)
    with pytest.raises((OSError, ValueError)):
        panel.inspect_image(root, name)


def test_output_symlink_parent_is_rejected_before_write(panel, fixture, tmp_path):
    clean, corrupt, _ = fixture
    outside = tmp_path / "alias"
    outside.symlink_to(clean, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        panel.write_fresh_output(outside / "result.json", b"{}", [clean, corrupt])
    assert not (clean / "result.json").exists()
