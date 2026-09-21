"""The builder's exact derived PDF belongs to the outer artifact checksum."""

from pathlib import Path

from docs.research.kbound.scripts import build_release_source_seal as seal
from docs.research.kbound.scripts import verify_release_checksums as checksums

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = "docs/research/kbound/figures/fig_frontier_schematic.pdf"


def test_rebuilt_frontier_is_a_bound_output_not_immutable_source():
    assert FRONTIER in seal.GENERATED_OUTPUT_ALLOWLIST
    assert FRONTIER not in {path for _, path in seal._inventory(ROOT, "HEAD")}


def test_rebuilt_frontier_is_required_by_the_outer_checksum_contract():
    assert FRONTIER in checksums.REQUIRED_RELEASE_PATHS


def test_frontier_output_exception_preserves_its_producer_source_bindings():
    paths = {path for _, path in seal._inventory(ROOT, "HEAD")}
    assert "docs/research/kbound/scripts/build_pdfs.sh" in paths
    assert "docs/research/kbound/scripts/make_submission_figures.py" in paths
    assert "docs/research/kbound/figures/unreviewed.pdf" not in seal.GENERATED_OUTPUT_ALLOWLIST
