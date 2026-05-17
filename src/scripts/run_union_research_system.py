"""Run the unified ELARA research system.

This orchestrator intentionally joins two layers that used to live separately:

* legacy standalone domain experiments for fraud, cyber, and behavior;
* paper-grade fusion benchmarks, healthcare gap audits, tables, figures, PDFs,
  and verification checks.

Each step writes its own log and the runner emits a machine-readable summary so
the repository has one end-to-end entry point without hiding which parts are
legacy dashboard support versus current research evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = Path("experiments/union_research_system/run_summary.json")
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tif", "*.tiff")


@dataclass(frozen=True)
class Step:
    name: str
    group: str
    command: tuple[str, ...]
    description: str
    required_paths: tuple[str, ...] = ()
    optional: bool = False
    evidence_scope: str = "research"


@dataclass
class StepResult:
    name: str
    group: str
    status: str
    command: list[str]
    description: str
    evidence_scope: str
    duration_seconds: float = 0.0
    log_path: str | None = None
    reason: str | None = None
    returncode: int | None = None


def _python() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _path_has_data(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return (
        any(path.rglob("*.csv"))
        or any(path.rglob("*.parquet"))
        or any(next(path.rglob(pattern), None) is not None for pattern in IMAGE_EXTENSIONS)
    )


def _missing_required(step: Step, repo_root: Path = REPO_ROOT) -> list[str]:
    missing: list[str] = []
    for raw_path in step.required_paths:
        path = repo_root / raw_path
        if not _path_has_data(path):
            missing.append(raw_path)
    return missing


def build_plan(
    *,
    mode: str = "full",
    include_deprecated_dashboard: bool = False,
    include_optional_nlp_vision: bool = False,
    with_tests: bool = False,
) -> list[Step]:
    """Build the ordered union-research execution plan."""
    py = _python()
    steps: list[Step] = [
        Step(
            name="legacy_fraud",
            group="legacy",
            command=(py, "src/scripts/run_fraud_experiment.py"),
            required_paths=("data/raw/fraud",),
            description="Train the standalone fraud supervised/anomaly pipeline.",
            evidence_scope="legacy-domain",
        ),
        Step(
            name="legacy_cyber",
            group="legacy",
            command=(py, "src/scripts/run_cyber_experiment.py"),
            required_paths=("data/raw/cyber",),
            description="Train the standalone cyber intrusion supervised/anomaly pipeline.",
            evidence_scope="legacy-domain",
        ),
        Step(
            name="legacy_behavior",
            group="legacy",
            command=(py, "src/scripts/run_behavior_experiment.py"),
            required_paths=("data/raw/behavior",),
            description="Train the standalone behavior anomaly pipeline.",
            evidence_scope="legacy-domain",
        ),
        Step(
            name="legacy_dashboard_fusion",
            group="legacy",
            command=(py, "src/scripts/run_fusion_experiment.py"),
            required_paths=("data/raw/fraud", "data/raw/cyber", "data/raw/behavior"),
            optional=True,
            description="Run the deprecated dashboard fusion bridge; not paper evidence.",
            evidence_scope="legacy-dashboard",
        ),
        Step(
            name="standalone_nlp",
            group="legacy",
            command=("bash", "scripts/run_train_nlp.sh"),
            required_paths=("data/processed/nlp/enron_emails.csv",),
            optional=True,
            description="Train the standalone NLP model when processed Enron data exists.",
            evidence_scope="optional-domain",
        ),
        Step(
            name="standalone_vision",
            group="legacy",
            command=("bash", "scripts/run_train_vision.sh"),
            required_paths=("data/processed/vision",),
            optional=True,
            description="Train the standalone vision classifier when processed image folders exist.",
            evidence_scope="optional-domain",
        ),
        Step(
            name="prep_mvtec_standard",
            group="paper_prep",
            command=(
                py,
                "src/scripts/prepare_mvtec3d_fusion_benchmark.py",
                "--dataset-root",
                "data/raw/mvtec3d",
                "--output",
                "experiments/fusion/mvtec3d_fusion_inputs.csv",
                "--metadata",
                "experiments/fusion/mvtec3d_fusion_metadata.json",
            ),
            required_paths=("data/raw/mvtec3d",),
            description="Build the naturally paired MVTec 3D-AD score-fusion table.",
        ),
        Step(
            name="prep_mvtec_heldout",
            group="paper_prep",
            command=(
                py,
                "src/scripts/prepare_mvtec3d_fusion_benchmark.py",
                "--dataset-root",
                "data/raw/mvtec3d",
                "--output",
                "experiments/fusion/mvtec3d_heldout_inputs.csv",
                "--metadata",
                "experiments/fusion/mvtec3d_heldout_metadata.json",
                "--train-categories",
                "bagel",
                "cable_gland",
                "cookie",
                "dowel",
            ),
            required_paths=("data/raw/mvtec3d",),
            description="Build the MVTec held-out-category variant.",
        ),
        Step(
            name="prep_mvtec_m3dm",
            group="paper_prep",
            command=(
                py,
                "src/scripts/prepare_mvtec3d_fusion_benchmark.py",
                "--dataset-root",
                "data/raw/mvtec3d",
                "--feature-mode",
                "m3dm",
                "--embedding-dim",
                "16",
                "--output",
                "experiments/fusion/mvtec3d_m3dm_inputs.csv",
                "--metadata",
                "experiments/fusion/mvtec3d_m3dm_metadata.json",
            ),
            required_paths=("data/raw/mvtec3d",),
            description="Build the M3DM-style ResNet/PCA MVTec variant.",
        ),
        Step(
            name="prep_realfusion",
            group="paper_prep",
            command=(
                py,
                "src/scripts/prepare_real_fusion_benchmark.py",
                "--output",
                "experiments/fusion/real_domain_fusion_inputs.csv",
                "--metadata",
                "experiments/fusion/real_domain_fusion_metadata.json",
            ),
            required_paths=("data/raw/fraud", "data/raw/cyber", "data/raw/behavior", "data/raw/nlp"),
            description="Build the label-aligned RealFusion benchmark.",
        ),
        Step(
            name="prep_realfusion_hard",
            group="paper_prep",
            command=(
                py,
                "src/scripts/prepare_real_fusion_benchmark.py",
                "--scorer-train-fraction",
                "0.05",
                "--output",
                "experiments/fusion/real_domain_fusion_hard_inputs.csv",
                "--metadata",
                "experiments/fusion/real_domain_fusion_hard_metadata.json",
            ),
            required_paths=("data/raw/fraud", "data/raw/cyber", "data/raw/behavior", "data/raw/nlp"),
            description="Build the hard RealFusion benchmark with weaker domain scorers.",
        ),
        Step(
            name="train_mvtec_standard",
            group="paper_train",
            command=(
                py,
                "src/scripts/run_breakthrough_experiment.py",
                "--config",
                "configs/attention_mvtec3d_fusion.yaml",
                "--output",
                "experiments/fusion/mvtec3d_results.json",
            ),
            required_paths=("experiments/fusion/mvtec3d_fusion_inputs.csv",),
            description="Run the 5-seed MVTec standard fusion experiment.",
        ),
        Step(
            name="train_mvtec_heldout",
            group="paper_train",
            command=(
                py,
                "src/scripts/run_breakthrough_experiment.py",
                "--config",
                "configs/attention_mvtec3d_heldout.yaml",
                "--output",
                "experiments/fusion/mvtec3d_heldout_results.json",
            ),
            required_paths=("experiments/fusion/mvtec3d_heldout_inputs.csv",),
            description="Run the 5-seed MVTec held-out-category experiment.",
        ),
        Step(
            name="train_mvtec_m3dm",
            group="paper_train",
            command=(
                py,
                "src/scripts/run_breakthrough_experiment.py",
                "--config",
                "configs/attention_mvtec3d_m3dm.yaml",
                "--output",
                "experiments/fusion/mvtec3d_m3dm_results.json",
            ),
            required_paths=("experiments/fusion/mvtec3d_m3dm_inputs.csv",),
            description="Run the 5-seed MVTec M3DM-style experiment.",
        ),
        Step(
            name="train_realfusion",
            group="paper_train",
            command=(
                py,
                "src/scripts/run_breakthrough_experiment.py",
                "--config",
                "configs/attention_real_fusion.yaml",
                "--output",
                "experiments/fusion/craf_real_results.json",
            ),
            required_paths=("experiments/fusion/real_domain_fusion_inputs.csv",),
            description="Run the 5-seed RealFusion experiment.",
        ),
        Step(
            name="train_realfusion_hard",
            group="paper_train",
            command=(
                py,
                "src/scripts/run_breakthrough_experiment.py",
                "--config",
                "configs/attention_real_fusion_hard.yaml",
                "--output",
                "experiments/fusion/craf_real_results_hard.json",
            ),
            required_paths=("experiments/fusion/real_domain_fusion_hard_inputs.csv",),
            description="Run the 5-seed hard RealFusion experiment.",
        ),
    ]

    healthcare_commands = [
        (
            "healthcare_provided",
            (
                py,
                "src/scripts/validate_healthcare_gap_closure.py",
                "--data-root",
                "data/raw/healthcare/gridpulse",
                "--report",
                "experiments/fusion/healthcare_gap_validation.json",
                "--fusion-output",
                "experiments/fusion/healthcare_clinical_fusion_inputs.csv",
            ),
            "Run the provided time-forward healthcare reference audit.",
        ),
        (
            "healthcare_gap1",
            (
                py,
                "src/scripts/validate_healthcare_gap_closure.py",
                "--data-root",
                "data/raw/healthcare/gridpulse",
                "--split-strategy",
                "patient_stratified",
                "--report",
                "experiments/fusion/healthcare_gap1_patient_stratified_validation.json",
                "--fusion-output",
                "experiments/fusion/healthcare_gap1_patient_stratified_fusion_inputs.csv",
            ),
            "Run the patient-disjoint healthcare Gap 1 replay.",
        ),
        (
            "healthcare_gap2",
            (
                py,
                "src/scripts/validate_healthcare_gap_closure.py",
                "--data-root",
                "data/raw/healthcare/gridpulse",
                "--split-strategy",
                "patient_stratified",
                "--report",
                "experiments/fusion/healthcare_gap2_reliability_stress_validation.json",
                "--fusion-output",
                "experiments/fusion/healthcare_gap2_reliability_stress_fusion_inputs.csv",
            ),
            "Run the healthcare reliability stress audit.",
        ),
        (
            "healthcare_gap4",
            (
                py,
                "src/scripts/validate_healthcare_gap_closure.py",
                "--data-root",
                "data/raw/healthcare/gridpulse",
                "--split-strategy",
                "patient_stratified",
                "--report",
                "experiments/fusion/healthcare_gap4_deployment_audit_validation.json",
                "--no-fusion-output",
            ),
            "Run the healthcare deployment-audit replay.",
        ),
    ]
    steps.extend(
        Step(
            name=name,
            group="healthcare",
            command=command,
            required_paths=("data/raw/healthcare/gridpulse/processed/splits",),
            description=description,
        )
        for name, command, description in healthcare_commands
    )
    steps.append(
        Step(
            name="rebuild_paper",
            group="assets",
            command=("bash", "scripts/rebuild_paper.sh"),
            required_paths=("experiments/fusion/craf_real_results.json", "experiments/fusion/mvtec3d_results.json"),
            description="Regenerate manuscript tables, figures, and PDFs.",
        )
    )
    if with_tests:
        steps.extend(
            [
                Step(
                    name="verify_metadata_tests",
                    group="verify",
                    command=(py, "-m", "pytest", "tests/test_paper_asset_metadata.py", "tests/test_healthcare_gap_closure.py", "-q"),
                    description="Run focused manuscript/healthcare regression tests.",
                ),
                Step(
                    name="verify_ruff_correctness",
                    group="verify",
                    command=(str(REPO_ROOT / ".venv" / "bin" / "ruff"), "check", "--select=E9,F63,F7,F82", "."),
                    description="Run ruff correctness checks.",
                ),
            ]
        )

    enabled_groups = {
        "full": {"legacy", "paper_prep", "paper_train", "healthcare", "assets", "verify"},
        "legacy": {"legacy", "verify"},
        "paper": {"paper_prep", "paper_train", "assets", "verify"},
        "healthcare": {"healthcare", "assets", "verify"},
        "assets": {"assets", "verify"},
    }[mode]
    filtered = [step for step in steps if step.group in enabled_groups]
    if not include_deprecated_dashboard:
        filtered = [step for step in filtered if step.name != "legacy_dashboard_fusion"]
    if not include_optional_nlp_vision:
        filtered = [step for step in filtered if step.name not in {"standalone_nlp", "standalone_vision"}]
    return filtered


def _write_summary(results: list[StepResult], summary_path: Path, started_at: float) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runner": "run_union_research_system.py",
        "duration_seconds": round(time.time() - started_at, 3),
        "results": [asdict(result) for result in results],
        "status_counts": {
            status: sum(1 for result in results if result.status == status)
            for status in sorted({result.status for result in results})
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_step(
    step: Step,
    *,
    logs_dir: Path,
    dry_run: bool = False,
    repo_root: Path = REPO_ROOT,
) -> StepResult:
    missing = _missing_required(step, repo_root)
    if missing:
        return StepResult(
            name=step.name,
            group=step.group,
            status="skipped",
            command=list(step.command),
            description=step.description,
            evidence_scope=step.evidence_scope,
            reason=f"missing required path(s): {', '.join(missing)}",
        )
    if dry_run:
        return StepResult(
            name=step.name,
            group=step.group,
            status="planned",
            command=list(step.command),
            description=step.description,
            evidence_scope=step.evidence_scope,
        )

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{step.name}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" if not env.get("PYTHONPATH") else f"src{os.pathsep}{env['PYTHONPATH']}"
    venv_bin = REPO_ROOT / ".venv" / "bin"
    if venv_bin.exists():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(step.command) + "\n\n")
        completed = subprocess.run(
            list(step.command),
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration = time.time() - start
    return StepResult(
        name=step.name,
        group=step.group,
        status="passed" if completed.returncode == 0 else "failed",
        command=list(step.command),
        description=step.description,
        evidence_scope=step.evidence_scope,
        duration_seconds=round(duration, 3),
        log_path=str(log_path),
        returncode=completed.returncode,
    )


def run_plan(
    steps: Iterable[Step],
    *,
    summary_path: Path,
    logs_dir: Path,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> list[StepResult]:
    started_at = time.time()
    results: list[StepResult] = []
    for step in steps:
        result = run_step(step, logs_dir=logs_dir, dry_run=dry_run)
        results.append(result)
        print(f"[{result.status}] {step.name}: {result.description}")
        if result.reason:
            print(f"  reason: {result.reason}")
        if result.log_path:
            print(f"  log: {result.log_path}")
        _write_summary(results, summary_path, started_at)
        if result.status == "failed" and not continue_on_error:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "legacy", "paper", "healthcare", "assets"], default="full")
    parser.add_argument("--dry-run", action="store_true", help="Print and summarize the plan without executing steps.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after failed executable steps.")
    parser.add_argument("--include-deprecated-dashboard", action="store_true", help="Also run legacy dashboard fusion.")
    parser.add_argument("--include-optional-nlp-vision", action="store_true", help="Run optional NLP/vision standalone scripts when data exists.")
    parser.add_argument("--with-tests", action="store_true", help="Append focused tests and ruff correctness checks.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--logs-dir", type=Path, default=Path("experiments/union_research_system/logs"))
    args = parser.parse_args()

    steps = build_plan(
        mode=args.mode,
        include_deprecated_dashboard=args.include_deprecated_dashboard,
        include_optional_nlp_vision=args.include_optional_nlp_vision,
        with_tests=args.with_tests,
    )
    results = run_plan(
        steps,
        summary_path=args.summary,
        logs_dir=args.logs_dir,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )
    failed = [result for result in results if result.status == "failed"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
