"""test_features -- the edge evidence schema is stable AND complete (14 features)."""

import numpy as np

from kbound_edge.evidence import (
    edge_evidence_vector, evidence_dict,
    EDGE_EVIDENCE_NAMES, EDGE_EXTRA_NAMES, N_EDGE_FEATURES,
)
from kbound_edge._bridge import PAPER_EVIDENCE_NAMES


def _probs(n=16, c=4, seed=0):
    return np.random.default_rng(seed).dirichlet(np.ones(c), size=n)


class TestSchema:
    def test_feature_count_is_14(self):
        assert N_EDGE_FEATURES == 14
        assert len(EDGE_EVIDENCE_NAMES) == 14

    def test_first_11_are_the_paper_schema(self):
        # the certificate-facing features must be IDENTICAL to the paper's
        assert tuple(EDGE_EVIDENCE_NAMES[:11]) == tuple(PAPER_EVIDENCE_NAMES)

    def test_three_edge_features_appended(self):
        assert EDGE_EXTRA_NAMES == ("mean_js_div", "pred_flip_rate", "post_top2_margin")
        assert tuple(EDGE_EVIDENCE_NAMES[11:]) == EDGE_EXTRA_NAMES

    def test_names_unique(self):
        assert len(set(EDGE_EVIDENCE_NAMES)) == len(EDGE_EVIDENCE_NAMES)


class TestVector:
    def test_shape_and_finite(self):
        z = edge_evidence_vector(_probs(seed=1), _probs(seed=2), upd_norm=0.1)
        assert z.shape == (14,)
        assert np.all(np.isfinite(z))

    def test_deterministic(self):
        p0, pa = _probs(seed=3), _probs(seed=4)
        assert np.array_equal(edge_evidence_vector(p0, pa, 0.2), edge_evidence_vector(p0, pa, 0.2))

    def test_dict_keys_match_schema(self):
        d = evidence_dict(_probs(seed=5), _probs(seed=6))
        assert list(d.keys()) == list(EDGE_EVIDENCE_NAMES)
        assert len(d) == 14

    def test_label_free_signature(self):
        # The function takes only (p0, pa, upd_norm) -- there is NO label input.
        # Same probability inputs -> identical features, guaranteeing label-freeness.
        p0, pa = _probs(seed=7), _probs(seed=8)
        assert np.array_equal(edge_evidence_vector(p0, pa), edge_evidence_vector(p0, pa))

    def test_update_norm_passthrough(self):
        z = edge_evidence_vector(_probs(seed=9), _probs(seed=10), upd_norm=0.42)
        assert z[EDGE_EVIDENCE_NAMES.index("update_norm")] == 0.42

    def test_collapse_raises_high_flip_and_conf(self):
        # sanity: a collapsed pa (all class 0) vs diffuse p0 -> high flip rate
        n, c = 16, 4
        p0 = _probs(n=n, c=c, seed=11)
        pa = np.full((n, c), 1e-4)
        pa[:, 0] = 1.0 - (c - 1) * 1e-4
        d = evidence_dict(p0, pa)
        assert 0.0 <= d["pred_flip_rate"] <= 1.0
        assert d["post_top2_margin"] > 0.5  # collapsed -> large top1-top2 gap
