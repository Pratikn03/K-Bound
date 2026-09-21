"""A historical location is not permission to hide changed or active code."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

MODULE = "docs.research.kbound.kbound_repro.source_hygiene"
FRAGMENT = "/" + "Users/" + "pratik" + "/private"


def checker():
    assert importlib.util.find_spec(MODULE) is not None, "preservation-aware guard is missing"
    return __import__(MODULE, fromlist=["check_source_hygiene"]).check_source_hygiene


def fixture(tmp_path):
    relative = "archive/history/old.py"
    data = ("ROOT = '" + FRAGMENT + "'\n").encode()
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    manifest = {
        "schema": "kbound-preserved-executables-v1",
        "roots": ["archive/history"],
        "records": [{"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}],
    }
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(manifest))
    return relative, manifest, inventory


def test_preserved_exact_bytes_pass_but_active_machine_path_fails(tmp_path):
    check = checker()
    relative, _, inventory = fixture(tmp_path)
    active = tmp_path / "active.py"
    active.write_text("ROOT = '" + FRAGMENT + "'\n")
    assert check(tmp_path, [relative, "active.py"], inventory, ["active.py"], [FRAGMENT]) == {"active.py"}
    active.write_text("ROOT = '.'\n")
    assert check(tmp_path, [relative, "active.py"], inventory, ["active.py"], [FRAGMENT]) == set()


@pytest.mark.parametrize("mutation", ["tamper", "missing", "duplicate", "new_clean_file", "new_bad_file", "symlink", "nonfinite_size", "outside_root"])
def test_invalid_preservation_fails_closed(tmp_path, mutation):
    check = checker()
    relative, manifest, inventory = fixture(tmp_path)
    path = tmp_path / relative
    if mutation == "tamper":
        path.write_text("changed but portable\n")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "duplicate":
        manifest["records"] *= 2
    elif mutation in {"new_clean_file", "new_bad_file"}:
        (path.parent / "new.sh").write_text("echo ok" if mutation == "new_clean_file" else FRAGMENT)
    elif mutation == "symlink":
        data = path.read_bytes()
        path.unlink()
        (tmp_path / "target.py").write_bytes(data)
        path.symlink_to(tmp_path / "target.py")
    elif mutation == "nonfinite_size":
        manifest["records"][0]["bytes"] = float("nan")
    elif mutation == "outside_root":
        manifest["records"][0]["path"] = "../escape.py"
    inventory.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        check(tmp_path, [relative], inventory, [], [FRAGMENT])


def test_preserved_code_cannot_enter_release_source_even_without_banned_path(tmp_path):
    check = checker()
    relative, _, inventory = fixture(tmp_path)
    with pytest.raises(ValueError, match="release"):
        check(tmp_path, [relative], inventory, [relative], [FRAGMENT])


def test_new_active_source_is_checked_even_when_not_in_release_inventory(tmp_path):
    check = checker()
    relative, _, inventory = fixture(tmp_path)
    (tmp_path / "new.py").write_text(FRAGMENT)
    assert check(tmp_path, [relative, "new.py"], inventory, [], [FRAGMENT]) == {"new.py"}


def test_unreadable_active_source_is_not_silently_clean(tmp_path):
    check = checker()
    relative, _, inventory = fixture(tmp_path)
    (tmp_path / "stub.py").write_bytes(b"\x00\x00")
    with pytest.raises(ValueError, match="text"):
        check(tmp_path, [relative, "stub.py"], inventory, [], [FRAGMENT])


def test_source_in_scratch_is_checked_when_release_explicitly_requires_it(tmp_path):
    check = checker()
    _, _, inventory = fixture(tmp_path)
    (tmp_path / "output").mkdir()
    (tmp_path / "output/required.py").write_text(FRAGMENT)
    assert check(tmp_path, [], inventory, ["output/required.py"], [FRAGMENT]) == {"output/required.py"}


def test_empty_package_markers_are_valid_but_still_hash_bound(tmp_path):
    check = checker()
    relative, manifest, inventory = fixture(tmp_path)
    (tmp_path / relative).write_bytes(b"")
    manifest["records"][0].update(bytes=0, sha256=hashlib.sha256(b"").hexdigest())
    inventory.write_text(json.dumps(manifest))
    (tmp_path / "__init__.py").write_bytes(b"")
    assert check(tmp_path, [relative, "__init__.py"], inventory, [], [FRAGMENT]) == set()


def test_unknown_archive_script_in_cache_directory_is_rejected(tmp_path):
    check = checker()
    relative, _, inventory = fixture(tmp_path)
    hidden = tmp_path / "archive/history/__pycache__"
    hidden.mkdir()
    (hidden / "unknown.py").write_text("print('portable but unregistered')")
    with pytest.raises(ValueError, match="unregistered"):
        check(tmp_path, [relative], inventory, [], [FRAGMENT])


def test_unreadable_archive_directory_is_not_a_clean_inventory(tmp_path, monkeypatch):
    check = checker()
    relative, _, inventory = fixture(tmp_path)
    hidden = tmp_path / "archive/history/unreadable"
    hidden.mkdir()
    (hidden / "unknown.py").write_text("print('portable but unregistered')")
    original = os.scandir

    def inaccessible(path):
        if Path(path) == hidden:
            raise PermissionError("synthetic inaccessible archive")
        return original(path)

    monkeypatch.setattr(os, "scandir", inaccessible)
    with pytest.raises(PermissionError, match="inaccessible"):
        check(tmp_path, [relative], inventory, [], [FRAGMENT])
