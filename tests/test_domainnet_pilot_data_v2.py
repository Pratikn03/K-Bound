"""Metadata-only amendment contracts; fixtures never inspect real model outcomes."""

import hashlib
import importlib
import importlib.util
import json
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.kbound.domainnet import pilot_data, source_data

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "protocols/confirmatory_v2/DOMAINNET_DEV_PILOT_DESIGN_v1.json"
AMENDMENT = ROOT / "protocols/confirmatory_v2/DOMAINNET_DEV_PILOT_AMENDMENT_v2.json"
STOP = ROOT / "protocols/confirmatory_v2/DOMAINNET_DEV_PILOT_v1_STOP_PORTABLE.json"
ARTIFACTS = ("inventory.json", "public_manifest.json", "labels.json", "quarantine.json")


def module():
    assert importlib.util.find_spec("experiments.kbound.domainnet.pilot_data_v2") is not None, (
        "approved exact-intersection quarantine builder is missing"
    )
    return importlib.import_module("experiments.kbound.domainnet.pilot_data_v2")


def write(path, value):
    path.write_bytes(source_data.json_bytes(value))
    return source_data.file_hash(path)


def fixture(tmp_path, *, geometry=False):
    """Real parent-shaped metadata; expectations below are independent literals."""
    parent = tmp_path / "v1"
    parent.mkdir()
    source = tmp_path / "source.json"
    source_hash = write(
        source,
        {
            "schema": "kbound-domainnet-source-inventory/2",
            "domain": "clipart",
            "role": "source",
            "status": "SOURCE_ONLY_PREPARED_NOT_SCIENTIFIC_LOCK",
            "identity": {"class_count": 2},
            "rows": [
                {"path": "clipart/ant/shared.png", "label": 0, "group_id": "9" * 64},
                {"path": "clipart/bear/unshared.png", "label": 1, "group_id": "8" * 64},
            ],
        },
    )
    identity = {
        "archive_sha256": "a" * 64,
        "list_sha256": "b" * 64,
        "list_git_blob": "c" * 40,
        "source_inventory_sha256": source_hash,
        "class_count": 2,
    }
    selected = [str(i) * 64 for i in [5, 4, 3, 2, 1, 0, 3, 9, 9]]
    if geometry:
        # Three groups per cell, each with 128 members: no group is ever split.
        need = {"DEV_fit": 120, "DEV_radius": 30, "DEV_check": 60}
        selected = ["9" * 64, "9" * 64]
        for i in range(2000):
            group = hashlib.sha256(f"large-fixture-{i}".encode()).hexdigest()
            partition = pilot_data.partition_for_group(group)
            if need[partition]:
                selected.extend([group] * 128)
                need[partition] -= 1
            if not any(need.values()):
                break
        assert not any(need.values())
    rows = []
    for i, group in enumerate(selected):
        label = 1 if i == 6 else i % 2
        path = f"painting/{'ant' if label == 0 else 'bear'}/{i}.png"
        rows.append(
            {
                "sample_id": hashlib.sha256(("kbound-painting-pilot-v1:sample:" + path).encode()).hexdigest(),
                "official_index": i,
                "path": path,
                "label": label,
                "image_sha256": hashlib.sha256(f"encoded-{i}".encode()).hexdigest(),
                "group_id": group,
                "width": 1,
                "height": 2,
            }
        )
    packing, counts, groups = pilot_data._packing(rows)
    conflicts = [group for group in groups if len(group["labels"]) > 1]
    overlap = [
        {
            "group_id": "9" * 64,
            "painting_members": [r for r in rows if r["group_id"] == "9" * 64],
            "source_members": json.loads(source.read_text())["rows"][:1],
        }
    ]
    inventory = {
        "schema": "kbound-painting-pilot-inventory/1",
        "status": "STOP_SOURCE_OVERLAP",
        "eligible_for_pilot": False,
        "eligible_for_confirmatory": False,
        "execution_scope": "SYNTHETIC_TEST",
        "domain": "painting",
        "role": "development",
        "identity": identity,
        "design": {"sha256": pilot_data.DESIGN_SHA256, "document": json.loads(DESIGN.read_text())},
        "class_map": {"ant": 0, "bear": 1},
        "rows": rows,
        "groups": groups,
        "packing": packing,
        "counts": counts,
        "partition_salt": pilot_data.PARTITION_SALT,
        "group_order_salt": pilot_data.ORDER_SALT,
        "conflicts": {
            "groups": conflicts,
            "group_count": len(conflicts),
            "row_count": sum(g["multiplicity"] for g in conflicts),
        },
        "source_overlap": {"groups": overlap, "group_count": 1, "painting_row_count": 2, "source_row_count": 1},
        "custody_scope": json.loads(DESIGN.read_text())["custody_scope"],
    }
    public = {
        "schema": "kbound-painting-pilot-public/1",
        "status": "STOP_SOURCE_OVERLAP",
        "eligible_for_pilot": False,
        "execution_scope": "SYNTHETIC_TEST",
        "design_sha256": pilot_data.DESIGN_SHA256,
        "packing": packing,
        "rows": [{k: v for k, v in r.items() if k not in {"path", "label"}} for r in rows],
    }
    write(parent / "inventory.json", inventory)
    write(parent / "public_manifest.json", public)
    write(parent / "labels.json", {r["sample_id"]: r["label"] for r in rows})
    summary = {
        "schema": "kbound-painting-pilot-summary/1",
        "status": "STOP_SOURCE_OVERLAP",
        "eligible_for_pilot": False,
        "eligible_for_confirmatory": False,
        "execution_scope": "SYNTHETIC_TEST",
        "identity": identity,
        "design_sha256": pilot_data.DESIGN_SHA256,
        "counts": counts,
        "row_count": len(rows),
        "group_count": len(groups),
        "conflict_group_count": len(conflicts),
        "source_overlap_group_count": 1,
    }
    stop = json.loads(STOP.read_text())["historical_stop"]
    stop.update(
        rows=len(rows),
        decoded_content_groups=len(groups),
        within_painting_cross_label_groups=len(conflicts),
        within_painting_cross_label_rows=sum(g["multiplicity"] for g in conflicts),
        cross_source_duplicate_groups=1,
        overlapping_painting_rows=2,
        overlapping_clipart_rows=1,
        affected_painting_rows_by_partition={
            p: sum(r["partition"] == p for r in overlap[0]["painting_members"]) for p in pilot_data.PARTITIONS
        },
        complete_cells_before_stop={p: counts[p]["complete_cells"] for p in pilot_data.PARTITIONS},
        unscored_tail_rows=sum(c["tail_rows"] for c in counts.values()),
        minimum_geometry_passes=geometry,
    )
    amendment = json.loads(AMENDMENT.read_text())
    amendment.update(
        expected_quarantined_groups=1,
        expected_quarantined_painting_rows=2,
        expected_original_painting_rows=len(rows),
        expected_remaining_painting_rows=len(rows) - 2,
    )
    data = {
        "parent": parent,
        "source": source,
        "identity": identity,
        "summary": summary,
        "stop": stop,
        "amendment": amendment,
        "tmp": tmp_path,
    }
    seal(data)
    return data


def seal(data):
    parent = data["parent"]
    output = {
        name: source_data.file_hash(parent / name) for name in ("inventory.json", "public_manifest.json", "labels.json")
    }
    data["summary"]["output_sha256"] = output
    summary_hash = write(parent / "summary.json", data["summary"])
    data["stop"]["preparation"].update(
        directory=str(parent),
        summary_sha256=summary_hash,
        inventory_sha256=output["inventory.json"],
        public_manifest_sha256=output["public_manifest.json"],
        labels_sha256=output["labels.json"],
    )
    stop_hash = write(data["tmp"] / "stop.json", data["stop"])
    data["amendment"].update(
        v1_stop_sha256=stop_hash,
        v1_summary_sha256=summary_hash,
        v1_inventory_sha256=output["inventory.json"],
        source_inventory_sha256=source_data.file_hash(data["source"]),
    )
    amendment_hash = write(data["tmp"] / "amendment.json", data["amendment"])
    data["pins"] = dict(
        data["identity"],
        amendment_sha256=amendment_hash,
        v1_stop_sha256=stop_hash,
        v1_summary_sha256=summary_hash,
        v1_inventory_sha256=output["inventory.json"],
        v1_public_manifest_sha256=output["public_manifest.json"],
        v1_labels_sha256=output["labels.json"],
    )


def prepare(data, output=None, **overrides):
    args = {
        "v1_dir": data["parent"],
        "source_inventory": data["source"],
        "design": DESIGN,
        "amendment": data["tmp"] / "amendment.json",
        "v1_stop": data["tmp"] / "stop.json",
        "output_dir": output or data["tmp"] / "out",
        "synthetic_identity": module().SyntheticIdentity(**data["pins"]),
    }
    args.update(overrides)
    return module().prepare_painting_v2(**args)


def test_exact_intersection_preserves_all_other_rows_conflicts_and_original_records(tmp_path):
    # Omitting one source-shared member, extra exclusions or relabeling fails this reconciliation.
    data = fixture(tmp_path)
    original = json.loads((data["parent"] / "inventory.json").read_text())
    before = {p: p.read_bytes() for p in [*data["parent"].iterdir(), data["tmp"] / "stop.json", data["source"]]}
    summary = prepare(data)
    out = tmp_path / "out"
    active = json.loads((out / "inventory.json").read_text())
    quarantine = json.loads((out / "quarantine.json").read_text())
    assert active["rows"] == original["rows"][:7]
    assert quarantine["rows"] == original["rows"][7:]
    assert quarantine["groups"] == original["source_overlap"]["groups"]
    assert active["source_overlap"] == {"groups": [], "group_count": 0, "painting_row_count": 0, "source_row_count": 0}
    assert active["conflicts"]["group_count"] == 1
    assert active["conflicts"]["groups"][0]["label_multiplicity"] == {"0": 1, "1": 1}
    assert summary["original_row_count"] == 9 and summary["row_count"] == 7
    assert summary["quarantined_row_count"] == 2 and summary["quarantined_group_count"] == 1
    assert summary["eligible_for_confirmatory"] is False
    assert summary["source_overlap_group_count"] == 0
    assert before == {p: p.read_bytes() for p in before}
    assert sum(c["total_rows"] for c in summary["counts"].values()) == 7
    assert active["amendment"]["sha256"] == data["pins"]["amendment_sha256"]
    assert active["parents"]["v1_inventory_sha256"] == data["pins"]["v1_inventory_sha256"]


def test_literal_partition_order_and_packing_tails_exclude_quarantine(tmp_path):
    # Changed salts, reordered group members or accidentally reused V1 windows fails literal indices.
    data = fixture(tmp_path)
    summary = prepare(data)
    out = tmp_path / "out"
    public = json.loads((out / "public_manifest.json").read_text())
    lookup = {r["sample_id"]: r["official_index"] for r in public["rows"]}
    assert {p: {w: [lookup[s] for s in ids] for w, ids in v["tail"].items()} for p, v in public["packing"].items()} == {
        "DEV_fit": {"U": [5, 1, 2, 6, 3, 0], "V": [], "E": []},
        "DEV_radius": {"U": [], "V": [], "E": []},
        "DEV_check": {"U": [4], "V": [], "E": []},
    }
    assert all(not v["cells"] for v in public["packing"].values())
    assert summary["status"] == "INCONCLUSIVE_DATA_GEOMETRY" and summary["eligible_for_pilot"] is False
    labels = json.loads((out / "labels.json").read_text())
    assert labels["schema"] == "kbound-painting-pilot-labels/2"
    original = json.loads((data["parent"] / "inventory.json").read_text())["rows"]
    assert labels["labels"] == {r["sample_id"]: r["label"] for r in original[:7]}
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
    assert "painting/" not in (out / "public_manifest.json").read_text()
    assert '"label"' not in (out / "public_manifest.json").read_text()


def test_complete_geometry_uses_original_minima_without_rescoring_tail(tmp_path):
    # Incorrect cell minima or geometry decisions turn this exact 40/10/20 panel into a different result.
    data = fixture(tmp_path, geometry=True)
    result = prepare(data)
    assert result["eligible_for_pilot"] is True
    assert result["status"] == "DEV_ONLY_PREPARED_NOT_SCIENTIFIC_LOCK"
    assert result["eligible_for_confirmatory"] is False
    assert {p: v["complete_cells"] for p, v in result["counts"].items()} == {
        "DEV_fit": 40,
        "DEV_radius": 10,
        "DEV_check": 20,
    }
    assert all(c == {"U": 128, "V": 128, "E": 128} for v in result["counts"].values() for c in v["cell_rows"])
    assert sum(v["tail_rows"] for v in result["counts"].values()) == 0


def test_deterministic_immutable_publication_binds_every_artifact_and_parent(tmp_path):
    # Nondeterministic identifiers or an unbound quarantine allow unnoticed substitutions.
    data = fixture(tmp_path)
    first = prepare(data, tmp_path / "one")
    assert prepare(data, tmp_path / "two") == first
    for name in (*ARTIFACTS, "summary.json"):
        a, b = tmp_path / "one" / name, tmp_path / "two" / name
        assert a.read_bytes() == b.read_bytes()
        assert not a.stat().st_mode & stat.S_IWUSR
        assert json.loads(a.read_text())["schema"].endswith("/2")
    assert first["output_sha256"] == {name: source_data.file_hash(tmp_path / "one" / name) for name in ARTIFACTS}
    assert first["parents"] == {
        k: v for k, v in data["pins"].items() if k.startswith("v1_") or k == "source_inventory_sha256"
    }


@pytest.mark.parametrize(
    "target",
    [
        "inventory.json",
        "summary.json",
        "public_manifest.json",
        "labels.json",
        "source.json",
        "amendment.json",
        "stop.json",
        "design",
    ],
)
def test_altered_parent_or_amendment_bytes_fail_before_output(tmp_path, target):
    data = fixture(tmp_path)
    if target == "design":
        changed = tmp_path / "design.json"
        changed.write_bytes(DESIGN.read_bytes() + b" ")
        args = {"design": changed}
    else:
        path = (
            data["parent"]
            if target in {"inventory.json", "summary.json", "public_manifest.json", "labels.json"}
            else tmp_path
        ) / target
        path.write_bytes(path.read_bytes() + b" ")
        args = {}
    with pytest.raises(ValueError, match="SHA256"):
        prepare(data, **args)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "summary_hash",
        "stop_count",
        "overlap_members",
        "packing",
        "public_label",
        "labels",
        "row_path",
        "amendment_rule",
        "amendment_count",
    ],
)
def test_resealed_synthetic_parents_still_require_consistent_metadata(tmp_path, fault):
    data = fixture(tmp_path)
    if fault == "stop_count":
        data["stop"]["overlapping_painting_rows"] = 1
    elif fault == "amendment_count":
        data["amendment"]["expected_quarantined_painting_rows"] = 1
    elif fault == "amendment_rule":
        data["amendment"]["quarantine_rule"] = "Drop arbitrary labels"
    elif fault == "summary_hash":
        data["summary"]["row_count"] = 8
    else:
        name = (
            "public_manifest.json"
            if fault == "public_label"
            else "labels.json"
            if fault == "labels"
            else "inventory.json"
        )
        value = json.loads((data["parent"] / name).read_text())
        if fault == "overlap_members":
            value["source_overlap"]["groups"][0]["painting_members"].pop()
        elif fault == "packing":
            value["packing"]["DEV_fit"]["tail"]["U"].reverse()
        elif fault == "public_label":
            value["rows"][0]["label"] = 0
        elif fault == "labels":
            value[next(iter(value))] = 1
        else:
            value["rows"][0]["path"] = "painting/../bad.png"
        write(data["parent"] / name, value)
    seal(data)
    with pytest.raises(ValueError):
        prepare(data)
    assert not (tmp_path / "out/summary.json").exists()


@pytest.mark.parametrize(
    "target", ["v1_dir", "source_inventory", "design", "amendment", "v1_stop", "output_parent", "parent_file", "dotdot"]
)
def test_symlinked_or_unsafe_paths_are_rejected(tmp_path, target):
    data = fixture(tmp_path)
    paths = {
        "v1_dir": data["parent"],
        "source_inventory": data["source"],
        "design": DESIGN,
        "amendment": tmp_path / "amendment.json",
        "v1_stop": tmp_path / "stop.json",
    }
    if target == "parent_file":
        path = data["parent"] / "inventory.json"
        moved = tmp_path / "original.json"
        path.rename(moved)
        path.symlink_to(moved)
        args = {}
    elif target == "dotdot":
        args = {"output_dir": tmp_path / "v1" / ".." / "out"}
    else:
        link = tmp_path / "link"
        link.symlink_to(tmp_path if target == "output_parent" else paths[target])
        args = {"output_dir": link / "out"} if target == "output_parent" else {target: link}
    with pytest.raises((ValueError, OSError)):
        prepare(data, **args)
    assert not (tmp_path / "out/summary.json").exists()


def test_output_collision_and_partial_publication_never_overwrite_or_resume(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    out = tmp_path / "out"
    real_write = source_data.write_new_json

    def interrupted(path, value):
        if path.name == "public_manifest.json":
            (path.parent / ".public_manifest.json.staged").write_bytes(b"partial")
            raise OSError("simulated partial publication")
        real_write(path, value)

    monkeypatch.setattr(module().source_data, "write_new_json", interrupted)
    with pytest.raises(OSError, match="partial publication"):
        prepare(data)
    assert (out / "inventory.json").exists()
    assert (out / ".public_manifest.json.staged").read_bytes() == b"partial"
    assert not (out / "summary.json").exists()
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(data)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_synthetic_hook_rejects_production_class_and_every_real_identity(tmp_path):
    data = fixture(tmp_path)
    identity = module().SyntheticIdentity(**data["pins"])
    with pytest.raises(ValueError):
        replace(identity, class_count=126)
    for name, value in module().PRODUCTION_IDENTITIES.items():
        with pytest.raises(ValueError):
            replace(identity, **{name: value})
    with pytest.raises(ValueError):
        prepare(data, synthetic_identity=data["pins"])
    with pytest.raises(ValueError):
        prepare(data, synthetic_identity=None)


def test_cli_exposes_no_identity_or_exclusion_override_and_geometry_exit_is_two(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    # CLI delegation uses the real builder fixture; only inaccessible production identity binding is supplied here.
    actual = module().prepare_painting_v2

    def fixture_builder(*args, **kwargs):
        return actual(*args, **kwargs, synthetic_identity=module().SyntheticIdentity(**data["pins"]))

    monkeypatch.setattr(module(), "prepare_painting_v2", fixture_builder)
    args = [
        "--v1-dir",
        str(data["parent"]),
        "--source-inventory",
        str(data["source"]),
        "--design",
        str(DESIGN),
        "--amendment",
        str(tmp_path / "amendment.json"),
        "--v1-stop",
        str(tmp_path / "stop.json"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert module().main(args) == 2
    for option in ("--synthetic-identity", "--exclude", "--partition-salt", "--class-count"):
        with pytest.raises(SystemExit) as raised:
            module().main([*args, option, "arbitrary"])
        assert raised.value.code == 2


@pytest.mark.parametrize("fault", [None, "summary", "inventory", "quarantine", "resealed_public", "summary_symlink"])
def test_read_only_verification_replays_parents_and_rejects_tampered_v2(tmp_path, fault):
    # Even a newly hashed public file may not select different windows or rows.
    data = fixture(tmp_path)
    prepare(data)
    out = tmp_path / "out"
    summary_path = out / "summary.json"
    expected = source_data.file_hash(summary_path)
    if fault == "summary_symlink":
        moved = tmp_path / "copied-summary.json"
        summary_path.rename(moved)
        summary_path.symlink_to(moved)
    elif fault:
        name = "public_manifest.json" if fault == "resealed_public" else fault + ".json"
        path = out / name
        path.chmod(0o644)
        if fault == "resealed_public":
            public = json.loads(path.read_text())
            public["rows"].pop()
            write(path, public)
            summary = json.loads(summary_path.read_text())
            summary["output_sha256"][name] = source_data.file_hash(path)
            summary_path.chmod(0o644)
            expected = write(summary_path, summary)
        else:
            path.write_bytes(path.read_bytes() + b" ")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    args = (out, data["parent"], data["source"], DESIGN, tmp_path / "amendment.json", tmp_path / "stop.json")
    kwargs = {"expected_summary_sha256": expected, "synthetic_identity": module().SyntheticIdentity(**data["pins"])}
    if fault:
        with pytest.raises((ValueError, OSError)):
            module().verify_prepared_painting_v2(*args, **kwargs)
    else:
        inventory, public = module().verify_prepared_painting_v2(*args, **kwargs)
        assert inventory == json.loads((out / "inventory.json").read_text())
        assert public == json.loads((out / "public_manifest.json").read_text())
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
