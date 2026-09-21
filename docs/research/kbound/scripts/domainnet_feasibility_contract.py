"""Fixed painting feasibility authorization and metadata-only allocation.

No image payload reads, adaptation, labels-as-outcomes, or scoring here.
Self-digests are integrity checks, not permission; production uses the exact
user-approved proposal and approval bytes. Historical pilot limits are intact.
"""

import hashlib
import json
import os
import random
import re
import stat
from pathlib import Path

from domainnet_reference_source import CLASS_MAP_SHA256, RELEASE_SHA256
from task3_image_panel import _open_directory_chain

APPROVAL_SHA = "df208647765c39428b55103609cb7ff787e877234256135d9202b06caaec19f7"
PROPOSAL_SHA = "c6e2f3c159289f4a2d99f9366dd2d2907e71eeffa83cd9e5a23ec51d9988667a"
PILOT_SHA = "8580ce1a2ac494ee12f25716edf5d8bdc4deb5eac974d36e6466463a6e41a072"
LIST_SHA = "0d91298e5777ff28b8d1dabf673af566d1002e2331bea1972ac9dea27978f0db"
ARCHIVE_SHA = "fa47e6d405503ea0286cabd767f176bd30e988b63ddd7b9db16cf030c9770015"
IDS_SHA = "a109435085bc78736a9c29f985e257caee25389a9e5f541708958451b82e67a0"
EPISODE_SHA = [
    "d2f5726f47976dd2eda929099c4de73d7664e2648ea4d771c6995439adf6c341",
    "f500596f2a16963a855027d74f8693b681f0e306d16cb6d4945111efd97639c2",
    "e6cd96c2a5f44f61a334301a73dd2b4a6f2afd32c25688b42db9db660e4ce702",
]
SEED = 2026091901
SCOPE = "opened painting development feasibility; not KGA routing or prospective evidence"
BUDGET = {"wall_seconds": 43200, "rss_bytes": 4 * 1024**3, "disk_bytes": 8 * 1024**3, "cache_bytes": 256 * 1024**2}
CODE_FILES = [
    "domainnet_feasibility_contract.py",
    "domainnet_feasibility_images.py",
    "domainnet_feasibility_runner.py",
    "domainnet_feasibility_score.py",
    "domainnet_reference_source.py",
    "domainnet_reference_adapter.py",
    "domainnet_pilot_images.py",
    "task3_image_panel.py",
]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_bytes(path, limit=32 * 1024**2):
    """One descriptor-bound, no-symlink read with stable identity."""
    path = Path(os.path.abspath(path))
    parent = _open_directory_chain(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or not 0 < before.st_size <= limit
                or getattr(before, "st_flags", 0) & 0x40000000
            ):
                raise ValueError("resident bounded regular file required")
            data = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
            keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if len(data) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in keys):
                raise ValueError("file changed during read")
            return data
    finally:
        os.close(parent)


def file_identity(path):
    data = read_bytes(path)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def read_json(path, sha):
    data = read_bytes(path)
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError("bound JSON digest mismatch")
    return json.loads(data)


def write_bytes(path, data):
    """Create-only, fsync publication; never follows parent/leaf symlinks."""
    path = Path(os.path.abspath(path))
    parent = _open_directory_chain(path.parent)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent)
    finally:
        os.close(parent)


def write_json(path, value):
    write_bytes(path, encoded(value) + b"\n")


def read_approval(path):
    return read_json(path, APPROVAL_SHA)


def valid_id(value):
    return (
        isinstance(value, str)
        and ".." not in value
        and re.fullmatch(r"painting/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\.(jpg|jpeg|png)", value)
    )


def select_ids(list_bytes, seed, excluded):
    if type(seed) is not int or seed != SEED:
        raise ValueError("fixed approved sampling seed required")
    rows = [line.split() for line in list_bytes.decode().splitlines()]
    if len(rows) != 30042 or any(len(row) != 2 or not valid_id(row[0]) for row in rows):
        raise ValueError("wrong population count or painting paths")
    ids = [row[0] for row in rows]
    if len(set(ids)) != 30042 or len(excluded) != 48 or len(set(excluded)) != 48 or not set(excluded) <= set(ids):
        raise ValueError("duplicate paths or invalid previous exclusion")
    return random.Random(seed).sample(sorted(set(ids) - set(excluded)), 18432)


def epoch_orders(seed):
    if type(seed) is not int or seed not in (2020, 2021, 2022):
        raise ValueError("invalid stream seed")
    orders = []
    for epoch in range(15):
        order = list(range(2048))
        random.Random(seed + epoch).shuffle(order)
        orders.append(order)
    return orders


def validate_member(entry):
    fields = {"image_id", "episode", "role", "stage_name", "crc32", "encoded_bytes", "compressed_bytes"}
    if (
        not isinstance(entry, dict)
        or set(entry) != fields
        or not valid_id(entry["image_id"])
        or type(entry["episode"]) is not int
        or entry["episode"] not in range(3)
        or entry["role"] not in ("adaptation", "evaluation")
        or not re.fullmatch(r"\d{5}\.bin", entry["stage_name"])
    ):
        raise ValueError("invalid member role/path/episode")
    for key, upper in [("crc32", 2**32 - 1), ("encoded_bytes", 50 * 1024**2), ("compressed_bytes", 50 * 1024**2)]:
        if type(entry[key]) is not int or not (0 if key == "crc32" else 1) <= entry[key] <= upper:
            raise ValueError("invalid ZIP member metadata")


def implementation():
    root = Path(__file__).parent
    return {name: file_identity(root / name) for name in CODE_FILES}


def build_manifest(list_bytes, metadata, seed=SEED):
    """Metadata contains authenticated proposal/pilot, ZIP directory and bindings."""
    if hashlib.sha256(list_bytes).hexdigest() != LIST_SHA:
        raise ValueError("source list identity mismatch")
    pilot, proposal = metadata["pilot"], metadata["proposal"]
    excluded = [x["image_id"] for x in pilot["selection"]["members"]]
    selected = select_ids(list_bytes, seed, excluded)
    if digest(selected) != IDS_SHA or proposal["selected_ids_ordered_sha256"] != IDS_SHA:
        raise ValueError("approved selected identities mismatch")
    members, episodes = [], []
    for e in range(3):
        ids = selected[e * 6144 : (e + 1) * 6144]
        episodes.append(
            {
                "episode": e,
                "stream_seed": 2020 + e,
                "adaptation_ids": ids[:2048],
                "evaluation_ids": ids[2048:],
                "epoch_order_sha256": [digest(x) for x in epoch_orders(2020 + e)],
            }
        )
        for j, name in enumerate(ids):
            info = metadata["directory"][name]
            members.append(
                {
                    "image_id": name,
                    "episode": e,
                    "role": "adaptation" if j < 2048 else "evaluation",
                    "stage_name": f"{e * 6144 + j:05d}.bin",
                    "crc32": info["crc32"],
                    "encoded_bytes": info["encoded_bytes"],
                    "compressed_bytes": info["compressed_bytes"],
                }
            )
    result = {
        "schema": "painting-feasibility-manifest-v1",
        "scope": SCOPE,
        "prospective": False,
        "approval_sha256": APPROVAL_SHA,
        "proposal_sha256": PROPOSAL_SHA,
        "pilot_sha256": PILOT_SHA,
        "sampling_seed": seed,
        "source_list": metadata["source_list"],
        "archive": {k: pilot["archive"][k] for k in ("path", "sha256", "bytes")},
        "checkpoint_path": pilot["checkpoint_path"],
        "checkpoint_sha256": RELEASE_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "reference_root": pilot["reference"]["root"],
        "excluded_ids": excluded,
        "episodes": episodes,
        "members": members,
        "budget": BUDGET,
        "recipe": {
            "epochs": 15,
            "steps_per_epoch": 512,
            "batch_size": 4,
            "device": "cpu",
            "torch_threads": 1,
            "loader_workers": 0,
        },
        "code": metadata["code"],
        "resolved_args": metadata["resolved_args"],
        "runtime": metadata["runtime"],
        "prior_inventory": metadata["prior_inventory"],
        "output_root": proposal["fresh_output_proposed"],
    }
    validate_manifest(result)
    return result


def validate_manifest(m):
    """Validate fixed production scope, NOT independent-review acceptance."""
    try:
        if (
            m["schema"] != "painting-feasibility-manifest-v1"
            or m["scope"] != SCOPE
            or m["prospective"] is not False
            or m["approval_sha256"] != APPROVAL_SHA
            or m["proposal_sha256"] != PROPOSAL_SHA
            or m["pilot_sha256"] != PILOT_SHA
            or m["sampling_seed"] != SEED
            or m["source_list"]["sha256"] != LIST_SHA
            or m["archive"]["sha256"] != ARCHIVE_SHA
            or m["archive"]["bytes"] != 3679366174
            or m["checkpoint_sha256"] != RELEASE_SHA256
            or m["class_map_sha256"] != CLASS_MAP_SHA256
            or m["budget"] != BUDGET
            or m["recipe"]
            != {
                "epochs": 15,
                "steps_per_epoch": 512,
                "batch_size": 4,
                "device": "cpu",
                "torch_threads": 1,
                "loader_workers": 0,
            }
        ):
            raise ValueError("manifest authorization/source/recipe mismatch")
        if len(m["episodes"]) != 3 or len(m["members"]) != 18432 or len(set(m["excluded_ids"])) != 48:
            raise ValueError("wrong complete panel count")
        ids = []
        for e, episode in enumerate(m["episodes"]):
            a, v = episode["adaptation_ids"], episode["evaluation_ids"]
            if (
                episode["episode"] != e
                or episode["stream_seed"] != 2020 + e
                or len(a) != 2048
                or len(v) != 4096
                or digest(a + v) != EPISODE_SHA[e]
                or episode["epoch_order_sha256"] != [digest(x) for x in epoch_orders(2020 + e)]
            ):
                raise ValueError("episode/order identity mismatch")
            ids.extend(a + v)
        if digest(ids) != IDS_SHA or len(set(ids)) != 18432 or set(ids) & set(m["excluded_ids"]):
            raise ValueError("selection/role overlap")
        for j, member in enumerate(m["members"]):
            validate_member(member)
            if (
                member["image_id"] != ids[j]
                or member["episode"] != j // 6144
                or member["role"] != ("adaptation" if j % 6144 < 2048 else "evaluation")
                or member["stage_name"] != f"{j:05d}.bin"
            ):
                raise ValueError("member role binding mismatch")
        if sum(x["encoded_bytes"] for x in m["members"]) != 914238704:
            raise ValueError("approved staged byte count mismatch")
        if set(m["code"]) != set(CODE_FILES):
            raise ValueError("implementation closure missing")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed manifest") from exc
    return m
