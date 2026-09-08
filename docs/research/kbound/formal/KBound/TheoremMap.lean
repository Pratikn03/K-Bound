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
import KBound.Probability.RandomizedActionLaw
import KBound.Probability.ExtendedRadiusCertificate
import KBound.Probability.EvidenceTransport
import KBound.Probability.JointKernelScore
import KBound.Probability.ActualWorldFrontier
import KBound.Probability.ActualWorldEvidence
import KBound.Probability.ActualWorldSubclass
import KBound.WeightedHelpful
import KBound.FeatureRank
import KBound.Probability.RiskAlignment
import KBound.Probability.ActualFibreRadius
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
import KBound.Probability.AuditFloor
import KBound.Probability.AuditFloorCorrectness
import KBound.Probability.JointTargetReduction
import KBound.Probability.PaperCounterexamples
import KBound.Probability.DependentSignFlip
import KBound.Probability.IndependentSignFlip
import KBound.PaperDecisionAlgebra
import KBound.Probability.ExactConformal

/-!
# K-Bound paper theorem index: finite spine and measurable foundations

## Clause-level coverage of the short-paper spine

`lem:reduction`
* FORMALIZED: binary benefit/sign algebra (`binary_sign_reduction`, `binary_margin_split`).
* FORMALIZED: measurable label kernels, actual joint target laws and the population
  zero-one loss integral (`MeasureTarget`, `MeasureFrontier`). On disagreement the
  constructed class is supported on the two predicted labels.
* FORMALIZED: arbitrary finite joint laws, binary complementarity or multiclass
  correct-on-disagreement event differences, and actual conditional set-integral
  score/residual linearity (`JointTargetReduction`). The subsequent kernel and
  full-class score-frontier bridges are listed below.
* FORMALIZED: `JointKernelScore` uses the actual disintegration of an arbitrary
  joint binary target law, with no Standard Borel assumption on the input space.
  It identifies kernel correctness with joint event probability, supplies bounded
  score integrability, and proves the exact score/residual/benefit/sign identities.
  Positive disagreement mass is explicit for conditional interpretation. The
  full actual-class frontier is supplied by the separate module below.
* FORMALIZED: `ActualWorldFrontier` now quantifies the full class of actual binary
  joint probability laws with fixed input marginal and actual scoreResidual budget.
  Constructed fields supply witnesses, not a restriction on the universal class.
  It proves strict-sign iff frontiers, the clipped benefit interval, nonemptiness,
  closed-band zero attainment, the zero-margin/budget case and rule maximality.
  The actual augmented-evidence/interior/randomized-probability assembly is
  supplied by `ActualWorldEvidence` under a fixed observation experiment.
  `ActualWorldSubclass` supplies the restricted-class interface from legal
  interior opposite-pair membership and boundary-only zero membership. These
  minimal conditions follow from the paper construction closure; they are not
  asserted equivalent to closure under every legal construction. Individual
  directional iff and uniqueness require a realized/nonempty subclass explicitly.

`lem:nonid` and `cor:matched-abstain`
* FORMALIZED: opposite fixed benefits force abstention; action-probability arithmetic
  (`matched_opposite_worlds_force_abstain`, `abstention_mass_ge_one_sub_two_alpha_arith`).
* FORMALIZED: measurable randomized rules on arbitrary world sample spaces, with
  equal induced action laws and world-specific directional-error probability
  bounds, force abstention probability at least `1 - 2 * alpha` in both worlds
  (`RandomizedActionLaw.randomized_rules_force_abstention`). Equality of action
  laws is explicit; equal seed marginals alone do not establish it.
* FORMALIZED: `EvidenceTransport` derives input-batch evidence laws under finite
  iid sampling, and derives common action laws from a fixed Markov policy or an
  independent common seed. Its two abstention capstones use actual probabilities.
  Neither a sampling design nor joint observation/seed equality is inferred from
  separate marginals. `ActualWorldEvidence` composes the actual score-class
  witnesses with a fixed marginal-based observation experiment, derives its
  concrete finite iid realization, and proves the fixed augmented-law fibre iff
  and closed-band randomized abstention bound. It does not infer arbitrary
  world-dependent batch or seed couplings from marginal equalities.
* FORMALIZED: measurable target-label kernels and equality of all measurable
  input-evidence laws; opposite risks within the declared full correctness-field
  class subject to a calibration-residual budget (`MeasureTarget`, `MeasureFrontier`).
* FORMALIZED: `ActualWorldSubclass` transports restricted directional-event bounds
  through the same actual augmented law, using separate opposite interior worlds
  and the boundary zero world. No interior zero membership is required.
* NOT INFERRED: construction closure or nonemptiness of an arbitrary deployment class.

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
* FORMALIZED: measurable radii taking finite and infinite values on a common
  probability space. Infinity abstains; every finite nonnegative real radius
  agrees with the original rule. Marginal coverage bounds the measurable union
  of both strict directional errors by alpha (`ExtendedRadiusCertificate`).
  Coverage for the declared target remains an explicit premise.
* FORMALIZED: literal ceiling index and measurable k-th absolute-residual order
  statistic, with infinity at insufficient calibration size (`ExactConformal`).
  Finite min/max characterization retains ties and excludes the held-out score;
  exchangeability yields marginal coverage and the union directional-error bound.
  Full-protocol score exchangeability and operational label custody are not
  inferred from cross-fitting or from these formal function-dependence proofs.
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

The maintained appendix's actual calibration and cell/population probability
witnesses are encoded in `PaperCounterexamples`. `DependentSignFlip` proves the
nine-coordinate common-sign witness and its exact 1/512 reference calculation;
`IndependentSignFlip` proves coordinate-flip invariance under mutual independence
and symmetric marginal laws. `PaperDecisionAlgebra` covers pointwise regret,
radius monotonicity and the printed numeric branches. Its helpful-only result is
composed by `WeightedHelpful` over finite common nonnegative weights, with positive
total weight for means. Per-cell helpfulness stays a premise. `RiskAlignment`
proves strict implies aligned on a fixed fibre, and supplies genuine zero/positive
target laws showing the converse fails at M=beta>0 under the fixed observation model.
`FeatureRank` encodes the literal eleven-coordinate order and both difference
relations, proves dimension9 of that real-linear schema and rank at most9 for
any satisfying row family. Observed exact rank9, probabilistic independence and
physical feature realizability do not follow.
`ActualFibreRadius` assembles the actual scoreResidual-class supremum and fixed
augmented-law fibre radius equality, using actual target attainment, positive
disagreement, the full fixed-marginal class and0<=beta<=1/2. The actual-class
supremum needs no sampling premise; the augmented-law fibre uses the explicit
fixed observation model. Literal paper sampling language remains qualified.
These additions do not establish exhaustive paper-wide semantic closure.
-/

namespace KBoundTheoremMap

-- Literal measurable conformal ceiling/order-statistic/infinity construction.
#check KBound.ExactConformal.orderRadius_le_iff
#check KBound.ExactConformal.orderRadius_eq_top
#check KBound.ExactConformal.orderRadius_real_characterization
#check KBound.ExactConformal.orderRadius_ignores_heldout
#check KBound.ExactConformal.orderRadius_at_full_rank
#check KBound.ExactConformal.orderRadius_lt_top
#check KBound.ExactConformal.finite_orderRadius_threshold
#check KBound.ExactConformal.measurable_orderRadius
#check KBound.ExactConformal.rank_positive
#check KBound.ExactConformal.rank_le_total
#check KBound.ExactConformal.ceiling_rank_budget
#check KBound.ExactConformal.no_calibration_radius
#check KBound.ExactConformal.orderRadius_attained
#check KBound.ExactConformal.real_order_statistic_attained
#check KBound.ExactConformal.coverage
#check KBound.ExactConformal.residual_directional_error
#check KBound.ExactConformal.literal_residual_certificate

-- Reviewed current-paper reductions, probability counterexamples, and algebra.
#check KBound.JointTargetReduction.measurableSet_disagreementEvent
#check KBound.JointTargetReduction.measurableSet_correctOnDisagreement
#check KBound.JointTargetReduction.zeroOneBenefit_eq_indicator_sub
#check KBound.JointTargetReduction.population_benefit_event_difference
#check KBound.JointTargetReduction.multiclass_population_reduction
#check KBound.JointTargetReduction.binary_benefit_eq_indicators
#check KBound.JointTargetReduction.binary_population_event_reduction
#check KBound.JointTargetReduction.binary_population_reduction
#check KBound.JointTargetReduction.conditional_score_residual_identity
#check KBound.PaperCounterexamples.calibration_correct_probability
#check KBound.PaperCounterexamples.calibration_score_calibrated
#check KBound.PaperCounterexamples.calibration_top_label_calibrated
#check KBound.PaperCounterexamples.calibration_disagreement_accuracy
#check KBound.PaperCounterexamples.calibration_actual_score_margin
#check KBound.PaperCounterexamples.calibration_margin_residual
#check KBound.PaperCounterexamples.calibration_population_benefit
#check KBound.PaperCounterexamples.calibration_witness_exists
#check KBound.PaperCounterexamples.cellEstimate_label_free
#check KBound.PaperCounterexamples.cell_population_benefit
#check KBound.PaperCounterexamples.cell_coverage_probability
#check KBound.PaperCounterexamples.cell_adapt_probability
#check KBound.PaperCounterexamples.cell_false_adapt_probability
#check KBound.PaperCounterexamples.cell_conditional_false_adapt_one
#check KBound.PaperCounterexamples.no_adaptation_no_false_adaptation
#check KBound.PaperCounterexamples.cell_population_witness_exists
#check KBound.PaperCounterexamples.multiclass_neither_correct
#check KBound.DependentSignFlip.coordinate_exchangeable
#check KBound.DependentSignFlip.positive_coordinate_probability
#check KBound.DependentSignFlip.exchangeable_fair_witness
#check KBound.DependentSignFlip.fair_map
#check KBound.DependentSignFlip.measurable_globalFlip
#check KBound.DependentSignFlip.global_sign_symmetric
#check KBound.DependentSignFlip.marginal_sign_symmetric
#check KBound.DependentSignFlip.measurable_flipFirst
#check KBound.DependentSignFlip.equal_coordinates_probability
#check KBound.DependentSignFlip.flipped_equal_coordinates_probability
#check KBound.DependentSignFlip.not_coordinate_sign_symmetric
#check KBound.DependentSignFlip.sign_le_one
#check KBound.DependentSignFlip.all_positive_iff
#check KBound.DependentSignFlip.mean_comparison_iff
#check KBound.DependentSignFlip.sign_vector_count
#check KBound.DependentSignFlip.positive_upper_tail_count
#check KBound.DependentSignFlip.negative_upper_tail_count
#check KBound.DependentSignFlip.positive_reference_value
#check KBound.DependentSignFlip.negative_reference_value
#check KBound.DependentSignFlip.small_reference_probability
#check KBound.IndependentSignFlip.independent_symmetric_coordinates
#check KBound.PaperDecisionAlgebra.frozen_oracle_regret
#check KBound.PaperDecisionAlgebra.adapted_oracle_regret
#check KBound.PaperDecisionAlgebra.abstain_regret_eq_frozen
#check KBound.PaperDecisionAlgebra.equal_score_zero_regret
#check KBound.PaperDecisionAlgebra.adapt_iff
#check KBound.PaperDecisionAlgebra.freeze_iff
#check KBound.PaperDecisionAlgebra.larger_radius_adapt_subset
#check KBound.PaperDecisionAlgebra.larger_radius_freeze_subset
#check KBound.PaperDecisionAlgebra.larger_radius_commit_subset
#check KBound.PaperDecisionAlgebra.helpful_only_no_score_improvement
#check KBound.PaperDecisionAlgebra.radius_expansion_can_increase_regret
#check KBound.PaperDecisionAlgebra.printed_numeric_branches
#check KBound.PaperDecisionAlgebra.universal_unit_residual_bound

-- `thm:certificate`: interval-decision algebra and measure containment.
#check KBound.cert_false_adapt_sound
#check KBound.cert_false_freeze_sound
-- Wave 5: measure-theoretic certificate + uniform-index conformal coverage
#check KBound.measure_false_adapt_le_alpha
#check KBound.measure_false_freeze_le_alpha
#check KBound.measure_false_adapt_le_alpha_of_measurable
-- `thm:certificate`: measurable random finite-or-infinite radius.
#check KBound.ExtendedRadiusCertificate.action_top
#check KBound.ExtendedRadiusCertificate.action_finite
#check KBound.ExtendedRadiusCertificate.action_ofReal
#check KBound.ExtendedRadiusCertificate.measurableSet_coverage
#check KBound.ExtendedRadiusCertificate.false_direction_subset_failure
#check KBound.ExtendedRadiusCertificate.directional_error_probability
#check KBound.ExtendedRadiusCertificate.measurableSet_falseDirection
#check KBound.card_high_strictRank_le
#check KBound.card_low_strictRank_ge
#check KBound.uniformIndex_miss_eq
#check KBound.uniformIndex_miss_le
#check KBound.uniformIndex_coverage_ge
#check KBound.uniformIndex_false_adapt_le
#check KBound.uniformIndex_false_freeze_le
-- `cor:matched-abstain`: retained arithmetic and full common action-law interface.
#check KBound.gate_regret_identity
#check KBound.abstention_mass_ge_one_sub_two_alpha_arith
#check KBound.matched_opposite_worlds_force_abstain
#check KBound.RandomizedActionLaw.abstention_lower_bound
#check KBound.RandomizedActionLaw.common_law_forces_abstention
#check KBound.RandomizedActionLaw.randomized_rules_force_abstention
#check KBound.EvidenceTransport.common_observable_pushforward
#check KBound.EvidenceTransport.iid_input_batch_law
#check KBound.EvidenceTransport.iid_batch_observable_law
#check KBound.EvidenceTransport.independent_seed_joint_law
#check KBound.EvidenceTransport.independent_seed_action_law
#check KBound.EvidenceTransport.kernel_action_law
#check KBound.EvidenceTransport.kernel_randomized_abstention
#check KBound.EvidenceTransport.independent_seed_randomized_abstention
#check KBound.JointKernelScore.joint_disagreement_mass
#check KBound.JointKernelScore.kernel_correctness_event
#check KBound.JointKernelScore.joint_correctness_event
#check KBound.JointKernelScore.joint_conditional_accuracy
#check KBound.JointKernelScore.joint_score_identity
#check KBound.JointKernelScore.joint_score_benefit
#check KBound.JointKernelScore.scoreMargin_bounds
#check KBound.JointKernelScore.joint_score_sign
#check KBound.ActualWorldFrontier.fieldWorld_fst
#check KBound.ActualWorldFrontier.fieldWorld_residual
#check KBound.ActualWorldFrontier.actual_frontier_adapt_iff
#check KBound.ActualWorldFrontier.actual_frontier_freeze_iff
#check KBound.ActualWorldFrontier.actual_identified_interval
#check KBound.ActualWorldFrontier.actual_class_nonempty
#check KBound.ActualWorldFrontier.actual_closed_band_zero_target
#check KBound.ActualWorldFrontier.actual_zero_margin_budget
#check KBound.ActualWorldFrontier.actual_strict_direction_iff
#check KBound.ActualWorldFrontier.actual_pointwise_maximal_rule
#check KBound.RandomizedActionLaw.zero_errors_full_abstention
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

-- General randomized audit floor and separate full correctness-field radius.
#check KBound.measurable_audit_floor
#check KBound.fibrewise_randomized_audit_floor
#check KBound.constant_fibreRadius_valid
#check KBound.constant_fibreRadius_audit_valid
#check KBound.audit_floor_frontier_inert
#check KBound.fibreRadius_eq_of_bound_and_witness
#check KBound.correctness_fibreRadius_eq_beta
#check KBound.correctness_target_fibreRadius_eq_beta

-- Actual score-class witnesses and explicit augmented observation experiments.
#check KBound.ActualWorldEvidence.augmentedLaw_eq
#check KBound.ActualWorldEvidence.iidObservation_realization
#check KBound.ActualWorldEvidence.augmentedLaw_iid
#check KBound.ActualWorldEvidence.actual_open_band_field_witnesses
#check KBound.ActualWorldEvidence.actual_open_band_matched_targets
#check KBound.ActualWorldEvidence.actual_closed_band_matched_zero
#check KBound.ActualWorldEvidence.actual_augmented_fibre_strict_direction_iff
#check KBound.ActualWorldEvidence.actual_closed_band_abstention

#check KBound.ActualWorldSubclass.zeroWorld_residual
#check KBound.ActualWorldSubclass.zeroWorld_benefit
#check KBound.ActualWorldSubclass.zeroWorld_admissible_iff
#check KBound.ActualWorldSubclass.tiltWorld_residual
#check KBound.ActualWorldSubclass.tiltWorld_benefit
#check KBound.ActualWorldSubclass.tiltWorld_admissible
#check KBound.ActualWorldSubclass.full_class_tilt_witnesses
#check KBound.ActualWorldSubclass.subclass_closed_band_obstructions
#check KBound.ActualWorldSubclass.subclass_strict_direction_iff
#check KBound.ActualWorldSubclass.subclass_adapt_iff_of_nonempty
#check KBound.ActualWorldSubclass.subclass_freeze_iff_of_nonempty
#check KBound.ActualWorldSubclass.subclass_pointwise_maximal_rule
#check KBound.ActualWorldSubclass.subclass_augmented_fibre_strict_direction_iff
#check KBound.ActualWorldSubclass.subclass_closed_band_abstention

end KBoundTheoremMap

#check KBound.WeightedHelpful.helpful_weighted_sum
#check KBound.WeightedHelpful.helpful_weighted_mean
#check KBound.WeightedHelpful.helpful_weighted_regret_mean

#check KBound.RiskAlignment.strict_implies_aligned
#check KBound.RiskAlignment.aligned_of_nonnegative
#check KBound.RiskAlignment.not_strict_of_zero
#check KBound.RiskAlignment.actual_positive_boundary_nonnegative
#check KBound.RiskAlignment.actual_boundary_risk_aligned_not_strict

#check KBound.FeatureRank.encode_relations
#check KBound.FeatureRank.encode_injective
#check KBound.FeatureRank.mem_range_iff
#check KBound.FeatureRank.schema_finrank
#check KBound.FeatureRank.rowspan_finrank_le

#check KBound.ActualFibreRadius.actual_abs_residual_attained
#check KBound.ActualFibreRadius.actual_class_radius
#check KBound.ActualFibreRadius.actual_augmented_fibre_eq_class
#check KBound.ActualFibreRadius.actual_augmented_fibre_radius
