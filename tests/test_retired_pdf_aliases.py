"""Retiring PDF aliases must retain exact historical bytes, not rename a build."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/research/kbound/archive/retired_pdf_aliases_2026-09-20"
EXPECTED = {
    "kbound_short_main.pdf": (844455, "49a7e5f3a1aac162a96f7fef69e0684e20f1135596fa436cedd03e77bd1698fc"),
    "kbound_submission.pdf": (1300682, "80018377ca04807bdd375addd8851977400ddc1105e7c4ae5f0747b64b8ea697"),
}


def test_retired_aliases_are_preserved_outside_active_publication_set():
    manifest = ARCHIVE / "MANIFEST.json"
    assert manifest.is_file(), "publication aliases must be preserved before retirement"
    saved = json.loads(manifest.read_text())
    assert saved["publication_authority"] is False
    assert {row["name"] for row in saved["records"]} == set(EXPECTED)
    for row in saved["records"]:
        path = ARCHIVE / row["name"]
        assert not any(p.is_symlink() for p in (path, *path.parents))
        data = path.read_bytes()
        actual = len(data), hashlib.sha256(data).hexdigest()
        assert actual == EXPECTED[row["name"]] == (row["bytes"], row["sha256"])
        assert not (ROOT / "docs/research/kbound" / row["name"]).exists()
