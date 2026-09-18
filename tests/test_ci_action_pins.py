from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (ROOT / ".github" / "workflows").glob("*.yml")
PINNED_ACTIONS = {
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/setup-python": "5fda3b95a4ea91299a34e894583c3862153e4b97",
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
}


def test_every_third_party_workflow_action_is_immutable_and_reviewed() -> None:
    observed: set[tuple[str, str]] = set()
    for path in sorted(WORKFLOWS):
        text = path.read_text(encoding="utf-8")
        for action, revision in re.findall(r"\buses:\s*([^\s@]+)@([^\s#]+)", text):
            observed.add((action, revision))
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                f"{path.relative_to(ROOT)} uses mutable action revision {action}@{revision}"
            )
            assert PINNED_ACTIONS.get(action) == revision, (
                f"{path.relative_to(ROOT)} uses an unreviewed action pin {action}@{revision}"
            )
    assert {action for action, _ in observed} == set(PINNED_ACTIONS)
