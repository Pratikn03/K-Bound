"""Phase 2.2D — hash-only Eyecandies download via gdown.

The bundled `eyec ec-get` CLI (eyecandies==1.0.3) uses an outdated Google
Drive download flow that returns HTTP 400 for the large category archives.
`gdown` keeps up with Google Drive's confirm-token UI. We use the same
official per-category file IDs (from the eyecandies==1.0.3 source) so
the downloaded content is bit-identical to what `eyec ec-get` would have
produced.

This script:
- downloads each category archive (.tar) to data/raw/eyecandies/_archives/<Category>.tar
- computes SHA256 of each archive
- writes experiments/phase2/family_d/eyecandies_archive_sha256.txt
- DOES NOT extract or inspect anomaly labels
- DOES NOT train any model

Per the Phase 2.2D specification it is a hash-only pass.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "data" / "raw" / "eyecandies" / "_archives"
HASH_FILE = ROOT / "experiments" / "phase2" / "family_d" / "eyecandies_archive_sha256.txt"

# Official per-category file IDs from eyecandies==1.0.3 source
# (see .venv/lib/python3.14/site-packages/eyecandies/commands/download.py)
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default=None,
                   help="comma-separated subset of category names (else all 10)")
    args = p.parse_args()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)

    cats = list(DATA_IDS) if args.only is None else [c.strip() for c in args.only.split(",")]

    hashes = {}
    for cat in cats:
        if cat not in DATA_IDS:
            raise SystemExit(f"unknown category {cat!r}")
        # The official archives are .tar (per download.py source: tarfile.open)
        archive_path = ARCHIVE_DIR / f"{cat}.tar"
        if archive_path.exists():
            print(f"[{cat}] already downloaded; computing SHA256...")
        else:
            print(f"[{cat}] downloading via gdown...", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", "gdown",
                 f"https://drive.google.com/uc?id={DATA_IDS[cat]}",
                 "-O", str(archive_path)],
                check=False,
            )
            if r.returncode != 0 or not archive_path.exists():
                print(f"[{cat}] DOWNLOAD FAILED (return code {r.returncode})", flush=True)
                continue
        digest = _sha256(archive_path)
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        hashes[cat] = (digest, archive_path.stat().st_size)
        print(f"[{cat}] sha256={digest}  size={size_mb:.1f} MB", flush=True)

    with HASH_FILE.open("w") as f:
        f.write("# Eyecandies 1.0.3 per-category archive SHA256\n")
        f.write("# Phase 2.2D hash-only download pass\n")
        f.write("# format: <sha256>  <bytes>  <category>\n")
        for cat, (digest, size) in sorted(hashes.items()):
            f.write(f"{digest}  {size}  {cat}\n")
    print(f"\nwrote {HASH_FILE}")
    print(f"successfully hashed {len(hashes)}/{len(cats)} categories")
    return 0 if len(hashes) == len(cats) else 1


if __name__ == "__main__":
    raise SystemExit(main())
