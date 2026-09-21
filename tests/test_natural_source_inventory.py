"""Sealed tests must not depend on omitted development implementation files."""

from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_release_source_seal as seal

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    "docs/research/kbound/scripts/domainnet_batch_probe.py",
    "docs/research/kbound/scripts/domainnet_bn_diagnostic.py",
    "docs/research/kbound/scripts/domainnet_capacity_pilot.py",
    "docs/research/kbound/scripts/domainnet_feasibility_contract.py",
    "docs/research/kbound/scripts/domainnet_feasibility_images.py",
    "docs/research/kbound/scripts/domainnet_feasibility_runner.py",
    "docs/research/kbound/scripts/domainnet_feasibility_score.py",
    "docs/research/kbound/scripts/domainnet_pilot_images.py",
    "docs/research/kbound/scripts/domainnet_reference_adapter.py",
    "docs/research/kbound/scripts/domainnet_reference_source.py",
    "docs/research/kbound/scripts/natural_study_contract.py",
    "docs/research/kbound/scripts/natural_study_runner.py",
    "experiments/kbound/results/natural_calibration_value_v1/DOMAINNET_PILOT_PROPOSAL_V1.json",
)


@pytest.mark.parametrize("relative", DEPENDENCIES)
def test_release_test_dependencies_have_exact_source_bindings(relative):
    # Inventory construction is source/metadata enumeration, not an experiment
    # gateway. It does not read referenced images, tensors or scored outcomes.
    paths = {path for _, path in seal._inventory(ROOT, "HEAD")}
    assert relative in paths, "sealed test depends on unsealed implementation/metadata"


def test_natural_metadata_binding_does_not_expand_to_experiment_outputs():
    selected = {
        path
        for _, path in seal._inventory(ROOT, "HEAD")
        if path.startswith("experiments/kbound/results/natural_calibration_value_v1/")
    }
    assert selected == {DEPENDENCIES[-1]}
