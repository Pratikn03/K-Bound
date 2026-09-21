"""No real dataset payloads: allocation and immutable contract regressions."""

import copy
import hashlib
import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts"
sys.path.insert(0, str(SCRIPTS))


def module():
    return importlib.import_module("domainnet_feasibility_contract")


def rows():
    return "\n".join(f"painting/c{i % 126}/image{i:05}.jpg {i % 126}" for i in range(30042)).encode()


def test_selection_is_label_blind_disjoint_and_repeatable():
    m = module()
    source = rows()
    changed = "\n".join(line.split()[0] + " 0" for line in source.decode().splitlines()).encode()
    excluded = [line.split()[0] for line in source.decode().splitlines()[:48]]
    selected = m.select_ids(source, 2026091901, excluded)
    assert len(selected) == len(set(selected)) == 18432
    assert not set(selected) & set(excluded)
    assert selected == m.select_ids(changed, 2026091901, excluded)
    assert selected == m.select_ids(source, 2026091901, excluded)


@pytest.mark.parametrize("fault", ["seed", "count", "duplicate", "domain", "excluded"])
def test_bad_allocation_rejected(fault):
    m = module()
    source = rows()
    seed = 2026091901
    excluded = [line.split()[0] for line in source.decode().splitlines()[:48]]
    if fault == "seed":
        seed += 1
    if fault == "count":
        source = source.split(b"\n", 1)[1]
    if fault == "duplicate":
        source = source.replace(b"image00001.jpg", b"image00000.jpg").replace(
            b"painting/c1/image00000", b"painting/c0/image00000"
        )
    if fault == "domain":
        source = source.replace(b"painting/", b"real/", 1)
    if fault == "excluded":
        excluded[0] = "painting/missing/no.jpg"
    with pytest.raises(ValueError):
        m.select_ids(source, seed, excluded)


def test_approval_cannot_be_caller_supplied_or_inferred(tmp_path):
    m = module()
    fake = tmp_path / "approval.json"
    fake.write_text("{}")
    with pytest.raises(ValueError):
        m.read_approval(fake)


def test_epoch_orders_are_bound_without_global_rng_changes():
    import random

    m = module()
    before = random.getstate()
    orders = m.epoch_orders(2020)
    assert len(orders) == 15 and len({m.digest(x) for x in orders}) == 15
    assert all(sorted(x) == list(range(2048)) for x in orders)
    assert random.getstate() == before
    assert 3 * len(orders) * len(orders[0]) // 4 == 23040
    assert 15 * 2048 % 16384 == 14336


def test_create_only_bound_json_and_symlink_parents(tmp_path):
    m = module()
    dest = tmp_path / "record.json"
    value = {"a": [1, 2]}
    m.write_json(dest, value)
    assert m.read_json(dest, m.file_identity(dest)["sha256"]) == value
    with pytest.raises(FileExistsError):
        m.write_json(dest, value)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        m.write_json(link / "other.json", value)
    with pytest.raises(ValueError):
        m.read_json(dest, "0" * 64)


def test_contract_rejects_unbound_scope_changes():
    m = module()
    # A caller-created digest is integrity, not launch authorization.
    fake = {"schema": "painting-feasibility-manifest-v1", "episodes": [], "prospective": True}
    with pytest.raises(ValueError):
        m.validate_manifest(fake)


def test_member_metadata_is_validated_before_payload():
    m = module()
    entry = {
        "image_id": "painting/a/one.jpg",
        "episode": 0,
        "role": "adaptation",
        "stage_name": "00000.bin",
        "crc32": 1,
        "encoded_bytes": 12,
        "compressed_bytes": 10,
    }
    m.validate_member(entry)
    for key, value in [
        ("image_id", "real/a/one.jpg"),
        ("role", "score"),
        ("stage_name", "../x"),
        ("episode", True),
        ("encoded_bytes", 0),
        ("crc32", -1),
    ]:
        bad = copy.deepcopy(entry)
        bad[key] = value
        with pytest.raises(ValueError):
            m.validate_member(bad)


def test_pilot_authority_matches_preserved_repository_bytes():
    path = (
        SCRIPTS.parents[3] / "experiments/kbound/results/natural_calibration_value_v1/DOMAINNET_PILOT_PROPOSAL_V1.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == module().PILOT_SHA
