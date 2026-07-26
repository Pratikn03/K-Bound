"""Research naming invariants: an acronym is expanded where it is introduced.

History (defect D7).  This file used to assert that
``Evidence-Layered Anomaly Reliability Architecture`` (ELARA) and
``Reliability-Gated Attention`` (RGA) appear in ``README.md`` **and** in
``src/uais/fusion/attention/__init__.py``.  Both assertions now describe a
subject this repository no longer has:

* ``src/uais/`` does not exist -- the multimodal-anomaly code was removed when
  the tree became K-Bound / KGA only (``MONOREPO.md``), so the second path in
  each loop could not be read at all;
* ``README.md`` was rewritten to describe K-Bound, and correctly no longer
  brands itself with an architecture it does not ship.

So the README is right and the test was stale.  The invariant it was protecting
-- *an acronym must be expanded in the document that introduces it and in the
module that implements it* -- is real and worth keeping, so it is re-pointed at
the naming this repository actually carries, and a tripwire is added so that a
half-restored ELARA/RGA surface fails loudly instead of passing vacuously.

The paths are resolved from this file's location rather than from the process's
current working directory, which is a second reason the old assertions could
fail: they only worked when pytest happened to be invoked from the repo root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

#: (acronym, expansion, module that implements it).  The expansion must appear
#: in README.md *and* in the module, so the two can never drift apart.
LIVE_NAMING: list[tuple[str, str, Path]] = [
    ("KGA", "Knowability-Guided Adaptation", REPO / "kga" / "__init__.py"),
]

#: (acronym, expansion, module that used to implement it).  These left the
#: repository.  If the module comes back, the expansion must come back with it
#: -- and the entry must move up into LIVE_NAMING.
RETIRED_NAMING: list[tuple[str, str, Path]] = [
    (
        "ELARA",
        "Evidence-Layered Anomaly Reliability Architecture",
        REPO / "src" / "uais" / "fusion" / "attention" / "__init__.py",
    ),
    (
        "RGA",
        "Reliability-Gated Attention",
        REPO / "src" / "uais" / "fusion" / "attention" / "__init__.py",
    ),
]


@pytest.mark.parametrize("acronym,expansion,module", LIVE_NAMING, ids=[n[0] for n in LIVE_NAMING])
def test_live_acronym_is_expanded_in_readme_and_in_its_module(acronym, expansion, module):
    readme = README.read_text(encoding="utf-8")
    assert acronym in readme, f"README.md never mentions {acronym}"
    assert expansion in readme, (
        f"README.md uses the acronym {acronym} but never expands it to {expansion!r}. "
        "Expand it on first use, or change both here and there together."
    )
    assert module.exists(), f"{module} is missing but {acronym} is listed as live naming"
    assert expansion in module.read_text(encoding="utf-8"), (
        f"{module.relative_to(REPO)} implements {acronym} but does not spell out {expansion!r}, "
        "so the docs and the code disagree about what the acronym means."
    )


@pytest.mark.parametrize(
    "acronym,expansion,module", RETIRED_NAMING, ids=[n[0] for n in RETIRED_NAMING]
)
def test_retired_acronym_is_fully_gone_or_fully_documented(acronym, expansion, module):
    """A retired subsystem must be absent, or present *with* its expansion.

    The failure this guards against is the middle state: the module returns to
    the tree while README.md still says nothing about it, which is how the
    original assertion came to be checking a file that did not exist.
    """
    if not module.exists():
        # Retirement is complete: assert the whole package is gone, so this
        # cannot pass on a half-removed tree with the module merely renamed.
        assert not (REPO / "src" / "uais").exists(), (
            f"{module.relative_to(REPO)} is gone but src/uais still exists; "
            f"{acronym} is in an undefined half-retired state."
        )
        return
    text = module.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    assert expansion in text, (
        f"{module.relative_to(REPO)} is back in the tree, so {acronym} must be expanded "
        f"to {expansion!r} in it."
    )
    assert expansion in readme, (
        f"{module.relative_to(REPO)} is back in the tree, so README.md must expand "
        f"{acronym} to {expansion!r} again, and this entry belongs in LIVE_NAMING."
    )
