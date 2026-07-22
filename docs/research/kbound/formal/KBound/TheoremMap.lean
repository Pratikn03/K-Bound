import KBound.Basics
import KBound.Certificate
import KBound.Conformal
import KBound.Corollaries
import KBound.Disagreement
import KBound.Dichotomy
import KBound.FiniteTesting
import KBound.Frontier
import KBound.Gate
import KBound.Impossibility
import KBound.LeCam
import KBound.ThreeWorld
import KBound.Multicandidate
import KBound.Probability.ConformalExchangeability
import KBound.Probability.EProcess
import KBound.Probability.Exchangeable
import KBound.Probability.LeCam
import KBound.Probability.LeCamMeasure
import KBound.Probability.Rates
import KBound.Probability.Ville
import KBound.Probability.MeasureCertificate
import KBound.Probability.RankCounting
import KBound.Probability.UniformConformal

/-!
# K-Bound paper theorem index (Wave 4 spine + Wave 5 measure + Wave 6 foundations)

## Clause-level coverage of the short-paper spine

`lem:reduction`
* FORMALIZED: binary benefit/sign algebra (`binary_sign_reduction`, `binary_margin_split`).
* PEN-AND-PAPER: construction of the conditional correctness function from a target law.

`lem:nonid` and `cor:matched-abstain`
* FORMALIZED: opposite fixed benefits force abstention; action-probability arithmetic
  (`matched_opposite_worlds_force_abstain`, `forced_abstention_probability`).
* PEN-AND-PAPER: measurable target-label kernels, equality of induced evidence laws, and
  membership of the constructed laws in the declared drift class.

`prop:closed-band` and `thm:frontier`
* FORMALIZED: frontier sufficiency, the three deterministic decision branches, the
  closed-band zero witness, open-band opposite-sign witnesses, and both boundary
  zero-versus-strict witnesses
  (`frontier_identifiable_positive`, `frontier_identifiable_negative`,
  `frontier_decision_adapt`, `frontier_decision_freeze`, `frontier_decision_abstain`,
  `frontier_band_zero_witness`, `frontier_open_band_opposite_witnesses`).
* PEN-AND-PAPER: realization of these algebraic drifts by measurable target laws,
  target-class richness, and the lift from the witnesses to distributional
  necessity/pointwise maximality.

`thm:certificate`
* FORMALIZED: pointwise containment of strict directional errors in coverage failure,
  measure-level error bounds conditional on coverage, and finite uniform-rank/exchangeable-score
  reductions (certificate, measure, and probability declarations checked below).
* NOT CLAIMED FORMALIZED: calibration transfer for the paper's heterogeneous deployment tracks or
  a general theorem that leave-one-condition-out empirical calibration is exact conformal.

Successful compilation checks only the declarations listed here; it is not an assertion that the
pen-and-paper clauses above have been kernel-checked.
-/

namespace KBoundTheoremMap

-- `thm:certificate`: interval-decision algebra and measure containment.
#check KBound.cert_false_adapt_sound
#check KBound.cert_false_freeze_sound
-- Wave 5: measure-theoretic certificate + uniform-index conformal coverage
#check KBound.measure_false_adapt_le_alpha
#check KBound.measure_false_freeze_le_alpha
#check KBound.measure_false_adapt_le_alpha_of_measurable
#check KBound.card_high_strictRank_le
#check KBound.card_low_strictRank_ge
#check KBound.uniformIndex_miss_eq
#check KBound.uniformIndex_miss_le
#check KBound.uniformIndex_coverage_ge
#check KBound.uniformIndex_false_adapt_le
#check KBound.uniformIndex_false_freeze_le
-- `lem:nonid` corollary only: fixed opposite worlds and probability arithmetic.
#check KBound.gate_regret_identity
#check KBound.forced_abstention_probability
#check KBound.matched_opposite_worlds_force_abstain
#check KBound.lecam_regret_floor_two_point
#check KBound.lecam_testing_two_point
-- `thm:frontier`: sufficiency, rule branches, and algebraic necessity witnesses.
#check KBound.frontier_identifiable_positive
#check KBound.frontier_identifiable_negative
#check KBound.frontier_decision_adapt
#check KBound.frontier_decision_freeze
#check KBound.frontier_decision_abstain
#check KBound.frontier_band_zero_witness
#check KBound.frontier_open_band_opposite_witnesses
#check KBound.frontier_positive_boundary_zero_strict
#check KBound.frontier_negative_boundary_zero_strict
-- `lem:reduction`: algebraic sign reductions.
#check KBound.binary_sign_reduction
#check KBound.binary_margin_split
#check KBound.multiclass_sign_reduction
#check KBound.multiclass_harm_iff_nonpos
#check KBound.multiclass_benefit_pos_of_pa_gt
#check KBound.multiclass_routing_harm_equiv
#check KBound.single_candidate_false_adapt_sound
#check KBound.one_sided_commit_when_radius_small
#check KBound.two_sided_sign_certified
#check KBound.finite_uniform_rank_coverage_add_miss
#check KBound.finite_uniform_rank_miss_le_alpha
#check KBound.exchangeable_conformal_miss_le_alpha
#check KBound.exchangeable_cert_false_adapt_sound
#check KBound.bettingFactor_le_one
#check KBound.betting_wealth_step_le
#check KBound.binary_benefit_neg_accuracy
#check KBound.binary_sign_flip_on_accuracy_complement
#check KBound.multiclass_benefit_swap_pa_p0
#check KBound.lecam_tv_identity
#check KBound.lecam_single_error_ge_one_sub_tv
#check KBound.rate_implies_commit
#check KBound.rate_conformal_miss
-- Wave 6: paper-faithful foundation closures
#check KBound.exchangeable_scores_miss_le_alpha
#check KBound.exchangeable_scores_false_adapt_le
#check KBound.betting_wealth_supermartingale_step
#check KBound.ville_bound_false_adapt
#check KBound.lecam_tv_two_point_measure
#check KBound.lecam_testing_error_ge_one_sub_tv_measure
#check KBound.hoeffding_radius_le
#check KBound.rate_commit_from_concentration
#check KBound.evidence_swap_involution
#check KBound.swap_flips_benefit_preserves_evidence

end KBoundTheoremMap
