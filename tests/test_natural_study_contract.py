"""Data-free development-margin selection; no natural evaluation authorization."""
import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts'))


def select(*args):
    return importlib.import_module('natural_study_contract').select_margin(*args)


def test_development_cost_selects_margin_without_score_inputs():
    assert select([.2, .3], [-.1, .1], [0., .25, .5], 1.) == .25


def test_tied_objective_selects_larger_margin():
    assert select([.2], [0.], [0., .5], 1.) == .5
    assert select([.2], [0.], [.5, 0., .5], 1.) == .5


def test_boundary_is_strict_and_cost_has_effect():
    assert select([.2], [-.1], [0., .2], 1.) == .2
    assert select([.2, .3], [-.1, .2], [0., .5], 1.) == 0.
    assert select([.2, .3], [-.1, .2], [0., .5], 3.) == .5


@pytest.mark.parametrize('values', [
    ([], [], [0.], 1.), ([.1], [], [0.], 1.), ([.1], [.2], [], 1.),
    ([float('nan')], [.2], [0.], 1.), ([float('inf')], [.2], [0.], 1.),
    ([.1], [float('nan')], [0.], 1.), ([.1], [1.1], [0.], 1.),
    ([.1], [-1.1], [0.], 1.), ([.1], [.2], [-.1], 1.),
    ([.1], [.2], [float('inf')], 1.), ([.1], [.2], [0.], 0.),
    ([.1], [.2], [0.], -1.), ([.1], [.2], [0.], float('nan')),
    ([True], [.2], [0.], 1.), ([.1], [False], [0.], 1.),
    ([.1], [.2], [True], 1.), ([.1], [.2], [0.], True),
    ('bad', [.2], [0.], 1.), ([.1], [.2], [0.], '1'),
])
def test_invalid_selection_inputs_fail_closed(values):
    with pytest.raises(ValueError): select(*values)
