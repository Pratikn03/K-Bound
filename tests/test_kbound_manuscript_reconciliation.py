"""Bounded source guards for reviewed claim corrections; no empirical data reads.

These guards prevent the specific overclaims identified by the semantic review.
They do not establish theorem fidelity, numerical rank, empirical premises, or
coverage. Independent semantic review remains required for changed prose.
"""
from pathlib import Path

from tests.kbound_tex_helpers import live_tex

PAPER = Path(__file__).resolve().parents[1] / "docs/research/kbound"


def words(name):
    return " ".join(live_tex((PAPER / name).read_text(encoding="utf-8")).split())


def test_matched_world_evidence_has_a_declared_common_experiment():
    body = words("kbound_submission_body.tex")
    setup = body.split(r"\paragraph{Scope of the matched-world observation experiment.}", 1)[-1]
    assert "same fixed observation experiment" in setup
    assert "finite iid input sampling" in setup
    assert "common joint law of observation and random seed" in setup
    assert "equal seed marginals alone" in setup
    lemma = body.split(r"\label{lem:fibre}", 1)[1].split(r"\end{proof}", 1)[0]
    assert "under the declared common observation experiment" in lemma
    assert "preserves every label-free evidence law" not in lemma


def test_fixed_pair_scope_does_not_assert_transductive_sampling_validity():
    body = words("kbound_submission_body.tex")
    assert "does not justify conditioning on a pair adapted on that same batch" in body
    assert "We do not establish that the transductive benchmark satisfies this observation model" in body
    assert "does not establish exchangeability" in body


def test_individual_subclass_direction_needs_a_realized_member():
    core = words("paper/sections/theory_core_main.tex")
    assert "For a nonempty compatible subclass" in core
    assert "both universal directional statements are vacuous" in core


def test_finite_adverse_outcomes_are_not_deductive_refutations():
    supp = words("kbound_submission_supplement.tex")
    passage = supp.split(r"\paragraph{What would falsify the practical claim.}", 1)[1].split(r"\FloatBarrier", 1)[0]
    assert "do not by themselves logically refute a marginal guarantee" in passage
    assert "Poor utility can also coexist with valid coverage" in passage
    assert "they refute the operational assumptions" not in passage


def test_no_bound_claim_is_limited_by_the_compatible_fibre_radius():
    abstract = words("kbound_abstract_core.tex")
    supp = words("kbound_submission_supplement.tex")
    assert "the smallest residual bound that any uniformly valid audit of the same information must cover" in abstract
    assert "cannot uniformly validate a residual bound below the unresolved radius" in supp
    assert r"An unlabeled deployment batch cannot verify $|\gamma|\le\beta$" not in supp


def test_feature_rank_separates_proved_bound_and_unverified_empirical_rank():
    supp = words("kbound_submission_supplement.tex")
    passage = supp.split(r"\subsection{Protocol-Specific Feature and Model Schemas}", 1)[1].split("The exact common coordinate order", 1)[0]
    assert "row rank at most nine" in passage
    assert "effective rank nine remains an unverified empirical assertion here" in passage
    assert "matrix-specific receipt and numerical tolerance" in passage
    assert "with effective rank 9" not in passage


def test_integrity_failure_is_not_silently_mapped_to_a_certified_action():
    body = words("kbound_submission_body.tex")
    supp = words("kbound_submission_supplement.tex")
    assert "integrity failure" in body
    assert "integrity exception" in supp
    assert "must not relabel it as certified FREEZE" in supp
    assert "CCT-20 Mahalanobis distance is diagnostic" in supp
    assert "withhold claim; recalibrate or justify transfer" in supp
    assert "Evidence outside support & distance or support audit & abstain" not in supp


def test_domainnet_records_are_separate_and_keep_zero_exposure_undefined():
    supp = words("kbound_submission_supplement.tex")
    rows = (
        ("DomainNet five-checkpoint", "120", "120"),
        ("DomainNet R50 entropy", "24", "24"),
        ("DomainNet AdaContrast-derived", "24", "24"),
    )
    for label, count, abstentions in rows:
        assert f"{label} & {count} & 0 & 0 & {abstentions} & -- & -- &" in supp
    assert "not 120 independent environments" in supp
    assert "conditional false-ADAPT and false-FREEZE rates are undefined" in supp
    assert "not a matched head-to-head" in supp
    assert "not an exact reproduction of the published pipeline" in supp


def test_domainnet_summary_values_and_diagnostic_limits_remain_visible():
    supp = words("kbound_submission_supplement.tex")
    for expected in (
        "12 helpful, 15 harmful, 93 tied", "7 helpful, 14 harmful, 3 tied",
        "1 helpful, 23 harmful, 0 tied", "1387/1366", "1384/1131",
        "2.6099386731", "10.8115435382", "21/24", "24/24",
        "0.1106770833", "0.1235465116", "0.390625", "1.07421875",
        "0.0651041667", "8.2995195413", "187", "181", "243", "107", "133",
        "does not isolate random initialization", "does not rescue the adapted model",
        "not validation of nominal coverage on unseen groups",
    ):
        assert expected in supp


def test_current_formal_count_is_bounded_and_not_paper_wide_certification():
    supp = words("kbound_submission_supplement.tex")
    assert r"\input{paper/sections/proof_traceability}" in supp
    proof = words("paper/sections/proof_traceability.tex")
    assert "315 registered declarations: 65 retained core declarations and 250 additional results" in proof
    assert "326 distinct" in proof
    assert "Thirteen public-root example files contain 80 examples" in proof
    assert "not a percentage of the paper proved" in proof
    assert "does not supply the reported experiments with a population sampling-error radius" in proof
    assert "incremental with cached dependencies, not a cold dependency rebuild" in proof
    assert r"\path{formal/full_report_bridges_verification_20260908.portable.json}" in proof
    assert "common observation law" in proof
    assert "common joint law of observation and random seed" in words("kbound_submission_body.tex")
