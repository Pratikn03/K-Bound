"""Central registry mapping thesis theorems T1--T7 to code and artifacts.

Every theorem entry lists:
  - the validating script(s)
  - the core library module(s)
  - the expected artifact path(s) relative to the repository root
  - the manuscript table (if any)

Use ``validate_theorem_stack.py`` to check that artifacts exist after a rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TheoremSpec:
    theorem_id: str
    title: str
    core_modules: tuple[str, ...]
    validation_scripts: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    table_path: str | None
    status_note: str


THEOREM_REGISTRY: dict[str, TheoremSpec] = {
    "T1": TheoremSpec(
        theorem_id="T1",
        title="Quality-blind fusion impossibility (T1a sparse-linear + T1b coherent-all-rules)",
        core_modules=(
            "src/uais/fusion/attention/reliability_estimator.py",
            "src/elara/theory/t1_impossibility.py",
        ),
        validation_scripts=("src/scripts/validate_t1_impossibility.py",),
        artifact_paths=(
            "experiments/fusion/t1_impossibility_validation.json",
            "docs/research/tables/t1_impossibility.tex",
        ),
        table_path="docs/research/tables/t1_impossibility.tex",
        status_note="Formal lemma + constructive adversary. T1a: linear rules dominated under sparse corruption; T1b: every rule (incl. median) dominated under coherent corruption.",
    ),
    "T2": TheoremSpec(
        theorem_id="T2",
        title="Global-KS mixture confounding (mixture-entropy bound + real-cohort validation)",
        core_modules=(
            "src/uais/fusion/attention/reliability_estimator.py",
            "src/elara/family_b/mixture_shift.py",
            "src/elara/theory/t2_mixture_entropy.py",
        ),
        validation_scripts=(
            "src/scripts/validate_category_mixture_t2.py",
            "src/scripts/emit_category_mixture_t2_table.py",
            "src/scripts/run_phase2_mixture_shift.py",
            "src/scripts/validate_t2_eyecandies_categories.py",
        ),
        artifact_paths=(
            "experiments/fusion/category_mixture_t2_validation.json",
            "docs/research/tables/category_mixture_t2.tex",
            "experiments/phase2/mechanism/domain_composition_shift_metrics.csv",
            "experiments/fusion/t2_category_ks_validation.json",
            "docs/research/tables/t2_category_ks.tex",
        ),
        table_path="docs/research/tables/t2_category_ks.tex",
        status_note="Real-cohort confirmed: on MVTec 3D-AD (separated categories) global KS false-fires under pure mixture re-weighting (p=4e-5) while category-aware KS stays null (0/8). Eyecandies negative control shows the effect requires inter-category separation (degenerate near-chance detector -> no confounding).",
    ),
    "T3": TheoremSpec(
        theorem_id="T3",
        title="Mean-gate dilution failure (closed-form miss probability)",
        core_modules=(
            "src/uais/fusion/attention/reliability_estimator.py",
            "src/elara/family_b/corruption.py",
            "src/elara/theory/t3_mean_gate_miss.py",
        ),
        validation_scripts=(
            "src/scripts/run_breakthrough_experiment.py",
            "src/scripts/emit_k_of_d_corruption_table.py",
            "src/scripts/run_phase2_mechanism_replication.py",
            "src/scripts/validate_t3_mean_gate_miss.py",
        ),
        artifact_paths=(
            "experiments/fusion/craf_real_k_domain_results.json",
            "docs/research/tables/elara_k_domain_corruption_results.tex",
            "docs/research/figures/elara_k_domain_corruption.png",
            "experiments/fusion/t3_mean_gate_miss_validation.json",
            "docs/research/tables/t3_mean_gate_miss.tex",
        ),
        table_path="docs/research/tables/elara_k_domain_corruption_results.tex",
        status_note="Closed-form P(miss) = Phi((mu_bar - tau)/sigma_bar); det. boundary k* = D(mu_h - tau)/(mu_h - mu_c); calibrated form matches empirical fire-rate within 0.027-0.10 abs error on k=1..3.",
    ),
    "T4": TheoremSpec(
        theorem_id="T4",
        title="Reliability-switch risk dominance (finite-sample sample complexity)",
        core_modules=("src/elara/certification/risk_dominance.py",),
        validation_scripts=(
            "src/scripts/run_phase2_b_cert_1_v2.py",
            "src/scripts/emit_risk_dominance_t4_table.py",
            "src/scripts/validate_t4_risk_dominance_sample_complexity.py",
        ),
        artifact_paths=(
            "experiments/phase2/certification/risk_dominance_terms_v2.csv",
            "docs/research/tables/risk_dominance_t4_prevalence.tex",
            "experiments/fusion/t4_risk_dominance_sample_complexity.json",
            "docs/research/tables/t4_risk_dominance_sample_complexity.tex",
        ),
        table_path="docs/research/tables/risk_dominance_t4_prevalence.tex",
        status_note="Finite-sample LCB on the dominance margin + closed-form min-n. On max_attack the point margin is positive but certifying deployment-prevalence dominance needs ~82k fired samples vs 1.6k available -- quantifies why the retrospective certificate is not a deployment guarantee.",
    ),
    "T5": TheoremSpec(
        theorem_id="T5",
        title="Finite-sample switching certificate (empirical-Bernstein closed form)",
        core_modules=(
            "src/uais/utils/metrics.py",
            "src/elara/certification/switching_certificate.py",
        ),
        validation_scripts=(
            "src/scripts/audit_switching_certificate_t5.py",
            "src/scripts/emit_switching_certificate_t5_table.py",
            "src/scripts/audit_switching_certificate_t5_persample.py",
        ),
        artifact_paths=(
            "experiments/fusion/switching_certificate_t5_audit.json",
            "docs/research/tables/switching_certificate_t5.tex",
            "experiments/fusion/switching_certificate_t5_persample_audit.json",
            "docs/research/tables/switching_certificate_t5_persample.tex",
        ),
        table_path="docs/research/tables/switching_certificate_t5.tex",
        status_note="Added deterministic empirical-Bernstein LCB (Maurer-Pontil) alongside bootstrap. At the per-sample level (n=48k) the closed form is tight (EB -0.0015 vs bootstrap -0.0011); at per-seed n=5 it is vacuous, honestly revealing the n=5 bootstrap certificates are not finite-sample valid.",
    ),
    "T6": TheoremSpec(
        theorem_id="T6",
        title="KS drift gate as sequential detector (CUSUM-style ARL/AED trade-off)",
        core_modules=(
            "src/uais/fusion/attention/reliability_estimator.py",
            "src/elara/theory/t6_sequential_detection.py",
        ),
        validation_scripts=(
            "src/scripts/run_phase2_ks_power_sweep.py",
            "src/scripts/emit_ks_power_t6_table.py",
            "src/scripts/validate_t6_sequential_detection.py",
        ),
        artifact_paths=(
            "experiments/phase2/mechanism/ks_window_size_power.csv",
            "docs/research/tables/ks_power_t6.tex",
            "experiments/fusion/t6_sequential_detection_validation.json",
            "docs/research/tables/t6_sequential_detection.tex",
        ),
        table_path="docs/research/tables/t6_sequential_detection.tex",
        status_note="Reformulated from empirical sweep to a sequential-detection theorem: closed-form ARL_0(W)=1/(2 exp(-2 W h^2)) and detection power Phi((delta-h)sqrt(2W)). ARL_0 monotone-exponential growth in W strongly validated on the B-MECH-4 sweep; power directionally validated (MAE 0.11, residual from KS-vs-likelihood-ratio efficiency + 5-seed noise).",
    ),
    "T7": TheoremSpec(
        theorem_id="T7",
        title="PAC bound on RGA+ meta-router (tightened via empirical Rademacher + empirical-Bernstein)",
        core_modules=(
            "src/scripts/audit_meta_router_pac.py",
            "src/scripts/audit_meta_router_pac_tight.py",
        ),
        validation_scripts=(
            "src/scripts/audit_meta_router_pac.py",
            "src/scripts/audit_meta_router_pac_tight.py",
            "src/scripts/emit_meta_router_pac_t7_table.py",
            "src/scripts/emit_meta_router_pac_t7_tight_table.py",
        ),
        artifact_paths=(
            "experiments/fusion/meta_router_pac_audit.json",
            "experiments/fusion/meta_router_pac_audit_tight.json",
            "docs/research/tables/meta_router_pac_t7.tex",
            "docs/research/tables/meta_router_pac_t7_tight.tex",
        ),
        table_path="docs/research/tables/meta_router_pac_t7_tight.tex",
        status_note="Tightened: slack 1.5-2.8 (loose) -> 0.11-0.48 (tight), x2-x5 reduction across 5 of 6 cells; Eyecandies skipped (single-class validation under one-class protocol).",
    ),
    "GDR": TheoremSpec(
        theorem_id="GDR",
        title="Coherence-certified gate decision rule (minimax-optimal; real-validation partial)",
        core_modules=(
            "src/uais/fusion/attention/gate_decision_rule.py",
            "src/elara/theory/gdr_minimax.py",
        ),
        validation_scripts=(
            "src/scripts/audit_gate_decision_rule_e2e.py",
            "src/scripts/emit_gate_decision_rule_table.py",
            "src/scripts/validate_gdr_minimax.py",
            "src/scripts/audit_gdr_real_benchmark.py",
        ),
        artifact_paths=(
            "experiments/fusion/gate_decision_rule_e2e_audit.json",
            "docs/research/tables/gate_decision_rule_e2e.tex",
            "experiments/fusion/gdr_minimax_validation.json",
            "docs/research/tables/gdr_minimax.tex",
            "experiments/fusion/gdr_real_benchmark_validation.json",
            "docs/research/tables/gdr_real_benchmark.tex",
        ),
        table_path="docs/research/tables/gdr_minimax.tex",
        status_note="MINIMAX PROVEN in the two-regime model (worst-case regret 0.0006 vs 0.10 for always/never-switch). Real-benchmark separation PARTIAL (1/3 at theta=0.5): bottlenecked by near-chance base detectors that under-disperse the coherence signal. Honest scope: theory A-level, empirics base-detector-limited.",
    ),
    "T8": TheoremSpec(
        theorem_id="T8",
        title="Certified heterogeneous fusion (CHF) under batch/category shift",
        core_modules=(
            "src/elara/theory/t8_certified_heterogeneous_fusion.py",
            "src/uais/fusion/attention/certified_heterogeneous_fusion.py",
        ),
        validation_scripts=("src/scripts/validate_t8_chf.py",),
        artifact_paths=(
            "experiments/fusion/t8_chf_validation.json",
            "docs/research/tables/t8_chf.tex",
        ),
        table_path="docs/research/tables/t8_chf.tex",
        status_note="Validation-only route/stack selection among SAR, RGA+, and coherence-gated paths; extends switching certificate to heterogeneous batches.",
    ),
}


def list_theorems() -> list[TheoremSpec]:
    return [THEOREM_REGISTRY[k] for k in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "GDR")]


def artifact_status(repo_root: Path, spec: TheoremSpec) -> dict[str, bool]:
    return {rel: (repo_root / rel).exists() for rel in spec.artifact_paths}
