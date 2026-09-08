#!/usr/bin/env python3
"""Generate six publication macros from hash-bound sufficient-statistic records.

The supplied manifest hash is a trust anchor, not a signature or evidence of
original count correctness. No source outcomes, labels, images, or models load.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

COUNTS = {"entropy": 24, "bridge": 24, "officehome": 54, "smoke": 1}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("Nonfinite JSON constant")


def parse(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA256")


def _number(value, low=0, high=None, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected finite real number, not bool/string")
    if value < low or (high is not None and value > high) or (positive and value <= 0):
        raise ValueError("Number out of bounds")


def _rows(role, data, binding):
    if not isinstance(data, dict) or set(data) != {"schema", "role", "rows"}:
        raise ValueError("Invalid extract structure")
    if data["schema"] != "kbound-empirical-macro-extract-v1" or data["role"] != role:
        raise ValueError("Invalid extract identity")
    rows = data["rows"]
    if (
        not isinstance(rows, list)
        or len(rows) != COUNTS[role]
        or type(binding["row_count"]) is not int
        or binding["row_count"] != COUNTS[role]
    ):
        raise ValueError("Missing or extra cells")
    expected_fields = (
        {"cell_id", "n_images", "frozen_correct", "candidate_correct", "action"}
        if role in ("entropy", "bridge")
        else {"cell_id", "a0", "a_adapted", "update_norm"}
        if role == "officehome"
        else {"cell_id", "wall_seconds"}
    ) | {"content_id"}
    ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("Invalid selected fields")
        cell_id = row["cell_id"]
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or len(cell_id) > 200
            or not re.fullmatch(r"[A-Za-z0-9_:|.\-]+", cell_id)
        ):
            raise ValueError("Invalid portable cell identity")
        ids.append(cell_id)
        if role in ("entropy", "bridge"):
            n = row["n_images"]
            if type(n) is not int or n <= 0:
                raise ValueError("Image count must be positive integer")
            for field in ("frozen_correct", "candidate_correct"):
                if type(row[field]) is not int or not 0 <= row[field] <= n:
                    raise ValueError("Correct count outside integer bounds")
            if row["action"] not in ("ADAPT", "FREEZE", "ABSTAIN"):
                raise ValueError("Unknown action")
        elif role == "officehome":
            _number(row["a0"], high=1)
            _number(row["a_adapted"], high=1)
            _number(row["update_norm"])
        else:
            _number(row["wall_seconds"], positive=True)
        _hash(row["content_id"])
        if digest(canonical({k: v for k, v in row.items() if k != "content_id"})) != row["content_id"]:
            raise ValueError("Cell content ID mismatch")
    if len(set(ids)) != len(ids) or ids != binding["ordered_cell_ids"]:
        raise ValueError("Duplicate, missing, unexpected or reordered cells")
    return rows


def compute(bundle, manifest_sha256):
    bundle = Path(bundle)
    _hash(manifest_sha256)
    manifest_path = bundle / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Manifest symlink is not allowed")
    raw = manifest_path.read_bytes()
    if digest(raw) != manifest_sha256:
        raise ValueError("Manifest hash mismatch")
    manifest = parse(raw)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "kbound-empirical-macro-bundle-v1"
        or type(manifest.get("extraction_version")) is not int
        or manifest.get("extraction_version") != 1
    ):
        raise ValueError("Unknown bundle schema/version")
    records = manifest.get("records")
    if not isinstance(records, dict) or set(records) != set(COUNTS):
        raise ValueError("Exactly four input roles required")
    inputs = {}
    for role, binding in records.items():
        if not isinstance(binding, dict):
            raise ValueError("Invalid source binding")
        for key in ("path", "sha256", "source_id", "original_sha256", "row_count", "ordered_cell_ids"):
            if key not in binding:
                raise ValueError("Incomplete source binding")
        if binding["path"] != role + ".json":
            raise ValueError("Only fixed portable child filenames allowed")
        _hash(binding["sha256"])
        _hash(binding["original_sha256"])
        child = bundle / binding["path"]
        if child.is_symlink():
            raise ValueError("Child symlink is not allowed")
        raw = child.read_bytes()
        if digest(raw) != binding["sha256"]:
            raise ValueError("Extract hash mismatch")
        inputs[role] = _rows(role, parse(raw), binding)
    values = {}
    for role, prefix in (("entropy", "KBDNEntropy"), ("bridge", "KBDNBridge")):
        rows = inputs[role]
        deltas = [(r["candidate_correct"] - r["frozen_correct"]) / r["n_images"] for r in rows]
        gate = 100 * (math.fsum(max(d, 0) - d * (r["action"] == "ADAPT") for d, r in zip(deltas, rows)) / len(rows))
        adapt = 100 * (math.fsum(max(-d, 0) for d in deltas) / len(rows))
        values[prefix + "GatePP"] = f"{gate:.6f}"
        values[prefix + "AdaptPP"] = f"{adapt:.6f}"
    rows = inputs["officehome"]
    bs = [r["a_adapted"] - r["a0"] for r in rows]
    regret = math.fsum(max(b, 0) - b * (r["update_norm"] == 0.0) for b, r in zip(bs, rows)) / len(rows)
    values["KBOHZeroRegret"] = f"{regret:.8f}"
    values["KBOHSmokeWallSeconds"] = f"{inputs['smoke'][0]['wall_seconds']:.2f}"
    return values


def render(values):
    return "% Generated from verified saved rows; see empirical-evidence-values.json.\n" + "".join(
        "\\providecommand{\\" + key + "}{" + value + "}\n" for key, value in values.items()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="Create a new macro file; never overwrite.")
    mode.add_argument("--check", type=Path, help="Verify an existing macro file without writing.")
    args = parser.parse_args()
    result = render(compute(args.bundle, args.manifest_sha256))
    if args.check is not None:
        if args.check.is_symlink() or args.check.read_bytes() != result.encode("ascii"):
            raise ValueError("Existing macro file does not match the pinned inputs")
        print("Six saved-evidence macros match the pinned inputs; no files changed.")
        return
    # Exclusive create prevents replacing scientific inputs or existing outputs.
    with args.output.open("x", encoding="ascii") as handle:
        handle.write(result)


if __name__ == "__main__":
    main()
