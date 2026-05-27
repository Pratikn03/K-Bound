"""Phase 2.1 — the v2 Family-A report must reference the registry cells, not the v1 drift cells."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_V2 = ROOT / "docs" / "research" / "phase2" / "FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT_v2.md"


def _text() -> str:
    return REPORT_V2.read_text()


def test_v2_report_lists_locked_family_a_cells():
    t = _text()
    # The v2 report must reference every locked benchmark/protocol pair.
    for s in (
        "MVTec 3D-AD",
        "PatchCore supervised-paired",
        "PatchCore held-out category",
        "MVTec LOCO-AD",
        "VisA",
        "RGB+edge supervised-paired",
        "UNSW-NB15",
        "flow/conn/context",
    ):
        assert s in t, f"v2 report missing locked cell substring: {s!r}"


def test_v2_report_does_not_present_efficientad_or_real3d_as_family_a():
    """The v1 drift was to list EfficientAD / Real3D as A-POWERED-2..5.
    The v2 report must not present them under Family-A cell IDs in
    *assertive cell-identity rows* — markdown bullet/table entries
    that start with the cell ID. Prose acknowledging the drift is fine."""
    import re

    lines = _text().splitlines()
    pat = re.compile(r"^[-*\s]*\*\*A-POWERED-[2-5]\*\*\s*[—\-:]")  # bullet-form claim line
    table_pat = re.compile(r"^\|\s*A-POWERED-[2-5]\s*\|")  # table-row claim line
    offenders = []
    for ln in lines:
        if pat.match(ln) or table_pat.match(ln):
            if "Real3D" in ln or "EfficientAD" in ln:
                offenders.append(ln.strip())
    assert not offenders, f"v2 report still claims EfficientAD/Real3D as Family-A cells: {offenders}"


def test_v2_report_labels_existing_output_as_secondary_audit():
    t = _text()
    assert (
        "SECONDARY_ALL_COMPARATOR_PILOT_AUDIT" in t
    ), "v2 report must label the K=10 output as SECONDARY_ALL_COMPARATOR_PILOT_AUDIT"
    assert "PRIMARY_FAMILY_A_CELL_LEVEL" in t, "v2 report must define the PRIMARY_FAMILY_A_CELL_LEVEL surface"


def test_v2_report_records_k5_not_final_until_all_cells_complete():
    t = _text().lower()
    # one of these phrasings must be present
    assert any(
        s in t
        for s in (
            "k = 5 holm",
            "k=5 holm",
            "pending_full_family",
            "cannot be reported until all five",
            "until all 5",
            "all five primary cells",
        )
    ), "v2 report must state that K=5 Holm is not final until all 5 cells exist"
