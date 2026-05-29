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
        title="Quality-blind fusion impossibility",
        core_modules=("src/uais/fusion/attention/reliability_estimator.py",),
        validation_scripts=(),
        artifact_paths=(),
        table_path=None,
        status_note="Theoretical motivation; illustrated by B1/B2 collapse construction.",
    ),
    "T2": TheoremSpec(
        theorem_id="T2",
        title="Global-KS mixture confounding",
        core_modules=(
            "src/uais/fusion/attention/reliability_estimator.py",
            "src/elara/family_b/mixture_shift.py",
        ),
        validation_scripts=(
            "src/scripts/validate_category_mixture_t2.py",
            "src/scripts/emit_category_mixture_t2_table.py",
            "src/scripts/run_phase2_mixture_shift.py",
        ),
        artifact_paths=(
            "experiments/fusion/category_mixture_t2_validation.json",
            "docs/research/tables/category_mixture_t2.tex",
            "experiments/phase2/mechanism/domain_composition_shift_metrics.csv",
        ),
        table_path="docs/research/tables/category_mixture_t2.tex",
        status_note="Synthetic mixture validated; real cohort shift remains deferred (B-MECH-3S).",
    ),
    "T3": TheoremSpec(
        theorem_id="T3",
        title="Mean-gate dilution failure",
        core_modules=(
            "src/uais/fusion/attention/reliability_estimator.py",
            "src/elara/family_b/corruption.py",
        ),
        validation_scripts=(
            "src/scripts/run_breakthrough_experiment.py",
            "src/scripts/emit_k_of_d_corruption_table.py",
            "src/scripts/run_phase2_mechanism_replication.py",
        ),
        artifact_paths=(
            "experiments/fusion/craf_real_k_domain_results.json",
            "docs/research/tables/elara_k_domain_corruption_results.tex",
            "docs/research/figures/elara_k_domain_corruption.png",
        ),
        table_path="docs/research/tables/elara_k_domain_corruption_results.tex",
        status_note="Best-validated theorem; k-of-D sweep + Family B replication.",
    ),
    "T4": TheoremSpec(
        theorem_id="T4",
        title="Reliability-switch risk dominance",
        core_modules=("src/elara/certification/risk_dominance.py",),
        validation_scripts=(
            "src/scripts/run_phase2_b_cert_1_v2.py",
            "src/scripts/emit_risk_dominance_t4_table.py",
        ),
        artifact_paths=(
            "experiments/phase2/certification/risk_dominance_terms_v2.csv",
            "docs/research/tables/risk_dominance_t4_prevalence.tex",
        ),
        table_path="docs/research/tables/risk_dominance_t4_prevalence.tex",
        status_note="Retrospective prevalence sensitivity from locked B-CERT-1 terms.",
    ),
    "T5": TheoremSpec(
        theorem_id="T5",
        title="Finite-sample switching certificate",
        core_modules=(
            "src/uais/utils/metrics.py",
            "src/elara/certification/switching_certificate.py",
        ),
        validation_scripts=(
            "src/scripts/audit_switching_certificate_t5.py",
            "src/scripts/emit_switching_certificate_t5_table.py",
        ),
        artifact_paths=(
            "experiments/fusion/switching_certificate_t5_audit.json",
            "docs/research/tables/switching_certificate_t5.tex",
        ),
        table_path="docs/research/tables/switching_certificate_t5.tex",
        status_note="Retrospective LCB audit; not a production safety certificate.",
    ),
    "T6": TheoremSpec(
        theorem_id="T6",
        title="KS false-fire and detection boundary",
        core_modules=("src/uais/fusion/attention/reliability_estimator.py",),
        validation_scripts=(
            "src/scripts/run_phase2_ks_power_sweep.py",
            "src/scripts/emit_ks_power_t6_table.py",
        ),
        artifact_paths=(
            "experiments/phase2/mechanism/ks_window_size_power.csv",
            "docs/research/tables/ks_power_t6.tex",
        ),
        table_path="docs/research/tables/ks_power_t6.tex",
        status_note="Locked window grid; bounded power tradeoff only.",
    ),
    "T7": TheoremSpec(
        theorem_id="T7",
        title="PAC bound on RGA+ meta-router",
        core_modules=("src/scripts/audit_meta_router_pac.py",),
        validation_scripts=("src/scripts/audit_meta_router_pac.py", "src/scripts/emit_meta_router_pac_t7_table.py"),
        artifact_paths=(
            "experiments/fusion/meta_router_pac_audit.json",
            "docs/research/tables/meta_router_pac_t7.tex",
        ),
        table_path="docs/research/tables/meta_router_pac_t7.tex",
        status_note="Capacity certificate on meta-router folds; thesis + paper appendix.",
    ),
    "GDR": TheoremSpec(
        theorem_id="GDR",
        title="Coherence-certified gate decision rule",
        core_modules=("src/uais/fusion/attention/gate_decision_rule.py",),
        validation_scripts=(
            "src/scripts/audit_gate_decision_rule_e2e.py",
            "src/scripts/emit_gate_decision_rule_table.py",
        ),
        artifact_paths=(
            "experiments/fusion/gate_decision_rule_e2e_audit.json",
            "docs/research/tables/gate_decision_rule_e2e.tex",
        ),
        table_path="docs/research/tables/gate_decision_rule_e2e.tex",
        status_note="Novel predictive rule combining T2 heterogeneity detection with T5 certificate.",
    ),
}


def list_theorems() -> list[TheoremSpec]:
    return [THEOREM_REGISTRY[k] for k in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "GDR")]


def artifact_status(repo_root: Path, spec: TheoremSpec) -> dict[str, bool]:
    return {rel: (repo_root / rel).exists() for rel in spec.artifact_paths}
