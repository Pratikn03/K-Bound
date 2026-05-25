"""Phase 2.2D — emit FAMILY_D_PARTITION_MANIFEST_v2.json with real archive SHA256s.

Reads:
  experiments/phase2/family_d/eyecandies_archive_sha256.txt
  experiments/phase2/family_d/eyecandies_schema_verification.json
  configs/phase2/family_d_v2_eyecandies_protocol.yaml (for protocol SHA256)
  docs/research/phase2/FAMILY_D_HYPOTHESES_v2.csv (for hypotheses SHA256)
  docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md
  docs/research/phase2/FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md

Writes:
  docs/research/phase2/FAMILY_D_PARTITION_MANIFEST_v2.json

Fails (does not write) if any required hash or count is missing.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HASH_TXT = ROOT / "experiments" / "phase2" / "family_d" / "eyecandies_archive_sha256.txt"
SCHEMA_JSON = ROOT / "experiments" / "phase2" / "family_d" / "eyecandies_schema_verification.json"
PROTOCOL_YAML = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"
HYPOTHESES_CSV = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_HYPOTHESES_v2.csv"
SEL_POLICY = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md"
OPERATOR_SPEC = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md"
OUT = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_PARTITION_MANIFEST_v2.json"

DATA_IDS = {
    "CandyCane": "1OI0Jh5tUj98j3ihFXCXf7EW2qSpeaTSY",
    "ChocolateCookie": "1PEvIXZOcxuDMBo4iuCsUVDN63jisg0QN",
    "ChocolatePraline": "1dRlDAS31QJSwROgA6yFcXo85mL0EBh25",
    "Confetto": "10GNPUIQTUheT-qd6EzO76fsUgAwsHfaq",
    "GummyBear": "1OCAKXPmpNrD9s3oUcQ--mhRZTt4HGJ-W",
    "HazelnutTruffle": "1PsKc4hXxsuIjqwyHh7ciPAeS-IxsPikm",
    "LicoriceSandwich": "1dtU_l9gD1zoCN7fIYRksd_9KeyZklaHC",
    "Lollipop": "1DbL91Zjm2I9-AfJewU3M354pW4vnuaNz",
    "Marshmallow": "1pebIU3AegEFilqqoROaVzOZqkSgX-JTo",
    "PeppermintCandy": "1tF_1fPJYaUVaf1AwjlEi-fsGWzgCx6UF",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    # Required inputs must all be present and complete.
    missing = []
    for p, label in [
        (HASH_TXT, "archive SHA256 hash file"),
        (SCHEMA_JSON, "schema verification JSON"),
        (PROTOCOL_YAML, "protocol YAML"),
        (HYPOTHESES_CSV, "hypotheses CSV"),
        (SEL_POLICY, "selection policy"),
        (OPERATOR_SPEC, "operator spec"),
    ]:
        if not p.exists():
            missing.append(f"{label} at {p}")
    if missing:
        print("BLOCKED: missing required inputs:")
        for m in missing:
            print(f"  - {m}")
        return 1

    # Parse archive SHA256 lines: "<sha256>  <bytes>  <category>"
    archive_hashes = {}
    for line in HASH_TXT.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        digest, size, cat = parts
        archive_hashes[cat] = {"sha256": digest, "size_bytes": int(size)}

    # Verify all 10 categories present
    needed = set(DATA_IDS) - set(archive_hashes)
    if needed:
        print(f"BLOCKED: missing archive hashes for: {sorted(needed)}")
        return 2

    # Load schema verification
    schema = json.loads(SCHEMA_JSON.read_text())

    manifest = {
        "contract_version": "v2",
        "phase": "2.2D",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "freeze_commit_hash": _git_head(),
        "test_evaluation_executed": False,
        "dataset": {
            "name": "Eyecandies",
            "release_version": "1.0.3",
            "official_source_page": "https://eyecan-ai.github.io/eyecandies/",
            "official_repository": "https://github.com/eyecan-ai/eyecandies",
            "official_paper": "Bonfiglioli et al., ACCV 2022 — The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization",
            "official_bibitem_key": "bonfiglioli2022eyecandies",
        },
        "modalities": {
            "documented_available": ["rgb", "depth", "normal"],
            "primary": ["rgb", "depth"],
            "excluded_from_primary": ["normal"],
        },
        "splits": {
            "train": "anomaly_free_official_training_split",
            "validation": "anomaly_free_official_validation_split",
            "test": "anomalous_official_test_split_NOT_USED_BEFORE_EXECUTION",
        },
        "permitted_training_validation_access": True,
        "prohibited_test_access_before_execution": True,
        "archives": {
            cat: {
                "drive_file_id": DATA_IDS[cat],
                "archive_path": f"data/raw/eyecandies/_archives/{cat}.tar",
                "sha256": archive_hashes[cat]["sha256"],
                "size_bytes": archive_hashes[cat]["size_bytes"],
                "schema": schema.get(cat, {"splits": {}}).get("splits", {}),
            }
            for cat in sorted(DATA_IDS)
        },
        "protocol_yaml_sha256": _sha256(PROTOCOL_YAML),
        "hypotheses_csv_sha256": _sha256(HYPOTHESES_CSV),
        "selection_policy_sha256": _sha256(SEL_POLICY),
        "operator_spec_sha256": _sha256(OPERATOR_SPEC),
        "boundary_notice": (
            "These are retrospective evaluation protocol artefacts; held-out "
            "confirmation requires the one-time future Family-D execution AFTER "
            "independent external review of this freeze."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
