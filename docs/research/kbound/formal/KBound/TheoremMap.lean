import KBound.Basics
import KBound.UnitMismatch
import KBound.Stability
import KBound.JackknifePlus
import KBound.NonFinite
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
import KBound.TargetLaw
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
import KBound.Probability.MeasureConformal
import KBound.Probability.FilteredVille
import KBound.Probability.InformationBound
import KBound.Probability.GeneralLeCam
import KBound.Probability.Concentration
import KBound.Probability.MeasureSwap
import KBound.Probability.MeasureTarget
import KBound.Probability.MeasureFrontier
import KBound.Probability.ChannelCounterexample

/-!
# K-Bound paper theorem index: finite spine and measurable foundations

## Clause-level coverage of the short-paper spine

`lem:reduction`
* FORMALIZED: binary benefit/sign algebra (`binary_sign_reduction`, `binary_margin_split`).
* FORMALIZED: measurable label kernels, actual joint target laws and the population
  zero-one loss integral (`MeasureTarget`, `MeasureFrontier`). On disagreement the
  constructed class is supported on the two predicted labels.

`lem:nonid` and `cor:matched-abstain`
* FORMALIZED: opposite fixed benefits force abstention; action-probability arithmetic
  (`matched_opposite_worlds_force_abstain`, `abstention_mass_ge_one_sub_two_alpha_arith`).
* FORMALIZED: measurable target-label kernels and equality of all measurable
  input-evidence laws; opposite risks within the declared full correctness-field
  class subject to a calibration-residual budget (`MeasureTarget`, `MeasureFrontier`).
* NOT INFERRED: membership in an arbitrary restricted deployment class.

`prop:closed-band` and `thm:frontier`
* FORMALIZED: frontier sufficiency, the three deterministic decision branches, the
  closed-band zero witness, open-band opposite-sign witnesses, and both boundary
  zero-versus-strict witnesses
  (`frontier_identifiable_positive`, `frontier_identifiable_negative`,
  `frontier_decision_adapt`, `frontier_decision_freeze`, `frontier_decision_abstain`,
  `frontier_band_zero_witness`, `frontier_open_band_opposite_witnesses`).
* FORMALIZED: a canonical finite discrete measurable target-law construction,
  matched induced evidence laws, and the distributional necessity/pointwise-maximality
  lift under the explicit `RichAt` target-class premise (`KBound/TargetLaw.lean`).
* FORMALIZED: arbitrary measurable input spaces, clipped feasible identified
  interval and exact strict ADAPT/FREEZE equivalences over the full measurable
  correctness-field class, with unchanged off-disagreement kernel and input law.
  This construction does not assume `RichAt` (`MeasureFrontier`). It does not
  establish richness of an arbitrary restricted target subclass.

`thm:certificate`
* FORMALIZED: pointwise containment of strict directional errors in coverage failure,
  measure-level error bounds, and one-shot residual coverage derived from
  exchangeable measurable scores (including ties) and a calibration threshold
  (`MeasureConformal`), not an assumed uniform-rank conclusion.
* NOT CLAIMED FORMALIZED: calibration transfer for the paper's heterogeneous deployment tracks or
  a general theorem that leave-one-condition-out empirical calibration is exact conformal.

`FilteredVille`, `Concentration`, and `GeneralLeCam` add filtered maximal and
bounded optional-stopping results, genuine concentration inequalities, and
general randomized-testing/KL bounds for finite product experiments.

`MeasureSwap` lifts the label-swap obstruction to arbitrary measurable evidence
channels. `ChannelCounterexample` disproves the historical orbit-selection
sufficiency claim and proves a set-theoretic fibre-consistency criterion. This
does not close the historical full one-bit/H/ratio-rate extension.

Successful compilation proves encoded propositions under their explicit
assumptions; it does not certify empirical preprocessing or calibration transfer.
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
#check KBound.abstention_mass_ge_one_sub_two_alpha_arith
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
#check KBound.finiteEvidence_measurable
#check KBound.finite_target_laws_matched_evidence
#check KBound.positiveTargetLaw_benefit
#check KBound.negativeTargetLaw_benefit
#check KBound.finite_target_world_pair
#check KBound.rich_closed_band_forces_abstain
#check KBound.frontierDecision_uniformly_sound
#check KBound.distributional_frontier_maximal
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
-- Wave 8: (A5) unit mismatch -- deterministic core of the LOO undercoverage witness
#check KBound.unit_mismatch_forces_miss
#check KBound.covering_requires_across_unit_radius
#check KBound.miss_mono
-- Wave 8: (A7) estimator stability transfers leave-one-out coverage
#check KBound.stability_transfers_loo_coverage
#check KBound.unstable_fit_voids_transfer
-- Wave 9: jackknife+ counting core (the factor two) and finiteness-free impossibility
#check KBoundJK.two_mul_pairs_le
#check KBoundJK.card_le_two_mul
#check KBoundNF.matched_opposite_forces_abstain
#check KBoundNF.continuum_matched_witness
#check KBoundNF.continuum_impossibility
-- Historical finite probability reductions (retained for compatibility).
#check KBound.uniformIndexLaw_miss_le_alpha
#check KBound.uniformIndexLaw_false_adapt_le
#check KBound.betting_wealth_supermartingale_step
#check KBound.ville_bound_false_adapt
#check KBound.lecam_tv_two_point_measure
#check KBound.lecam_testing_error_ge_one_sub_tv_measure
#check KBound.hoeffding_radius_le
#check KBound.rate_commit_from_concentration
#check KBound.evidence_swap_involution
#check KBound.swap_flips_benefit_preserves_evidence

-- Measurable foundation scope: MeasureConformal.
#check KBound.exchangeable_scoreLaw_miss_le
#check KBound.exchangeable_scores_rank_miss_le
#check KBound.exchangeable_scores_rank_coverage_ge
#check KBound.calibrationThreshold_rank_le
#check KBound.exchangeable_calibration_threshold_miss_le
#check KBound.exchangeable_calibration_threshold_coverage_ge
#check KBound.exchangeable_residual_coverage_ge
#check KBound.exchangeable_residual_false_adapt_le
#check KBound.exchangeable_residual_false_freeze_le
#check KBound.exchangeable_residual_either_error_le

-- Measurable foundation scope: FilteredVille.
#check KBound.filtered_optional_stopping_le
#check KBound.filtered_ville_finite
#check KBound.filtered_ville
#check KBound.filtered_ville_alpha
#check KBound.dominated_eprocess_ville
#check KBound.eprocess_finite_time_crossing
#check KBound.filtered_betting_supermartingale
#check KBound.filtered_betting_anytime
#check KBound.predictable_betting_wealth_bounds
#check KBound.predictable_betting_wealth_adapted
#check KBound.bounded_predictable_betting_anytime

-- Measurable foundation scope: InformationBound.
#check KBound.binary_bretagnolle_huber

-- Measurable foundation scope: GeneralLeCam.
#check KBound.measurableTotalVariation_eq_abs_sup
#check KBound.measurableTotalVariation_symm
#check KBound.measurableTotalVariation_map_le
#check KBound.general_lecam_testing_error_ge
#check KBound.exists_lecam_optimal_test
#check KBound.general_lecam_inf_testing_error
#check KBound.general_lecam_worst_case_error_ge
#check KBound.general_lecam_regret_floor
#check KBound.general_lecam_iid_testing_identity
#check KBound.binary_partition_kl_le
#check KBound.binary_partition_support
#check KBound.klDiv_map_measurableEquiv
#check KBound.klDiv_prod_add
#check KBound.klDiv_iidObservationLaw
#check KBound.general_bretagnolle_huber_finite
#check KBound.general_bretagnolle_huber
#check KBound.general_lecam_exponential_regret_floor
#check KBound.general_lecam_iid_exponential_regret_floor

-- Measurable foundation scope: Concentration.
#check KBound.subgaussian_abs_tail
#check KBound.bounded_independent_sum_tail
#check KBound.unit_interval_mean_tail
#check KBound.unit_interval_hoeffding_coverage
#check KBound.common_mean_hoeffding_coverage
#check KBound.paired_benefit_hoeffding_coverage
#check KBound.adapted_subgaussian_sum_tail
#check KBound.conditional_hoeffding_of_bounded_zero_mean
#check KBound.bounded_martingale_difference_tail

-- Measurable foundation scope: MeasureSwap.
#check KBound.measurable_predictionSwap
#check KBound.predictionSwap_law_involutive
#check KBound.predictionSwap_preserves_evidence
#check KBound.predictionSwap_preserves_channel
#check KBound.predictionSwap_negates_populationBenefit
#check KBound.evidence_definable_opposite_target

-- Measurable foundation scope: MeasureTarget.
#check KBound.targetLabelKernel_isMarkov
#check KBound.joint_target_probability
#check KBound.target_label_free_law
#check KBound.measurable_label_kernel_freedom
#check KBound.measurable_label_kernel_freedom_subtype
#check KBound.constructed_target_population_benefit
#check KBound.constant_target_population_benefit
#check KBound.disagreementMean_bounds
#check KBound.measurable_target_benefit_reduction
#check KBound.measurable_correctness_identified_interval
#check KBound.measurable_target_frontier_attainment

-- Measurable foundation scope: MeasureFrontier.
#check KBound.correctnessFieldTarget_properties
#check KBound.correctnessFieldTarget_benefit
#check KBound.measurable_frontier_class_nonempty
#check KBound.measurable_frontier_adapt_iff
#check KBound.measurable_frontier_freeze_iff
#check KBound.measurable_closed_band_zero_target
#check KBound.measurable_open_band_opposite_targets

-- Measurable foundation scope: ChannelCounterexample.
#check KBound.OrbitFibreCounterexample.selected_exactly_one
#check KBound.OrbitFibreCounterexample.orbit_selection_not_fibre_orientation
#check KBound.OrbitFibreCounterexample.no_evidence_decoder
#check KBound.bool_decoder_iff_constant_on_fibres

end KBoundTheoremMap
