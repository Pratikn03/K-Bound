"""Claim-language firewall for the prospective CCT-20 manuscript section."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "src/scripts/validate_manuscript_claims.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("cct20_manuscript_claim_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        (
            r"\section{CCT-20 result} The target labels remained unopened until scoring.",
            "literal label-unopened wording is forbidden",
        ),
        (
            r"\section{CCT-20 result} The crossed design supplies 45 independent environments.",
            "not 45 independent environments",
        ),
        (
            r"\section{CCT-20 result} Holm-adjusted confidence intervals exclude zero.",
            "Holm adjusts p-values, not confidence intervals",
        ),
        (
            r"\section{CCT-20 result} The analysis reproduces the official CCT-20 single-label leaderboard.",
            "not an official single-label leaderboard reproduction",
        ),
        (
            r"\section{CCT-20 protocol} The study was publicly preregistered.",
            "requires a hashed public-registry record",
        ),
        (
            r"\section{CCT-20 result} This establishes a universal natural-shift win.",
            "cannot establish a universal natural-shift win",
        ),
        (
            r"\section{CCT-20 result} Five checkpoints constitute five independent studies.",
            "not five independent studies",
        ),
        (
            r"\section{CCT-20 result} The cis-heavy calibration provides formal trans-location conformal coverage.",
            "does not establish formal trans-location conformal coverage",
        ),
        (
            "\\section{CCT-20 result} The target labels remained un% hidden split\nopened until scoring.",
            "literal label-unopened wording is forbidden",
        ),
        (
            r"\section{CCT-20 result} The 45 checkpoint-by-location cells are independent evaluation units.",
            "not 45 independent environments",
        ),
        (
            r"\section{CCT-20 result} We use Holm-corrected simultaneous intervals.",
            "Holm adjusts p-values, not confidence intervals",
        ),
        (
            r"\section{CCT-20 protocol} The study was registered publicly before target execution.",
            "requires a hashed public-registry record",
        ),
        (
            r"\section{CCT-20 result} This establishes a win on every natural shift.",
            "cannot establish a universal natural-shift win",
        ),
        (
            r"\section{CCT-20 result} Five checkpoints constitute five independent runs.",
            "not five independent studies",
        ),
        (
            r"\section{CCT-20 result} The calibration provides finite-sample guarantees across trans locations.",
            "does not establish formal trans-location conformal coverage",
        ),
    ],
)
def test_rejects_cct20_overclaims(snippet: str, expected: str, tmp_path: Path) -> None:
    problems = VALIDATOR.validate_cct20_claims(
        snippet,
        release_manifest_path=tmp_path / "missing_release_manifest.json",
        repository_root=tmp_path,
    )
    assert any(expected in problem for problem in problems), problems


def test_approved_pre_result_wording_is_accepted_without_release_manifest(tmp_path: Path) -> None:
    approved = r"""
    \section{Planned CCT-20 evaluation}
    The planned evaluation is outcome-unopened before model execution; aggregate target metadata had
    already been inspected, so it is not described as literally label-unopened. The design crosses
    five independently trained checkpoints with nine target camera locations, yielding 45
    checkpoint-by-location cells rather than 45 independent environments. Confidence intervals use
    a Bonferroni adjustment, whereas Holm adjusts the two p-values. The set-valued scorer is not an
    official CCT-20 single-label leaderboard reproduction. Because calibration is mostly from cis
    locations, formal trans-location conformal coverage is not claimed. Any completed finding will
    be specific to this locked experiment, not a universal natural-shift win. The checkpoints are
    repeated models, not five independent studies. No completed CCT-20 outcome is claimed here.
    """
    assert (
        VALIDATOR.validate_cct20_claims(
            approved,
            release_manifest_path=tmp_path / "not_created_yet.json",
            repository_root=tmp_path,
        )
        == []
    )


def test_negation_is_local_to_the_claim_and_approved_negation_is_accepted(
    tmp_path: Path,
) -> None:
    contradictory = (
        r"\section{CCT-20 result} We do not claim formal conformal coverage, "
        r"but we provide exact trans-location conformal coverage."
    )
    problems = VALIDATOR.validate_cct20_claims(
        contradictory,
        release_manifest_path=tmp_path / "missing.json",
        repository_root=tmp_path,
    )
    assert any("formal trans-location conformal coverage" in problem for problem in problems)

    approved = (
        r"\section{CCT-20 protocol} Holm-adjusted confidence intervals were not used. "
        r"Formal trans-location conformal coverage is not claimed."
    )
    assert (
        VALIDATOR.validate_cct20_claims(
            approved,
            release_manifest_path=tmp_path / "missing.json",
            repository_root=tmp_path,
        )
        == []
    )


def _stable_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _plain_json_identity(path: Path, document: dict) -> dict:
    _write_json(path, document)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_document_sha256": _stable_sha256(document),
    }


def _received_json_identity(path: Path, document: dict) -> dict:
    _write_json(path, document)
    receipt_path = path.with_name(path.name + ".receipt.json")
    receipt = {
        "schema": "kbound_cct20_artifact_receipt_v1",
        "artifact_path": str(path.resolve()),
        "artifact_bytes": path.stat().st_size,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_document_sha256": _stable_sha256(document),
    }
    _write_json(receipt_path, receipt)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": receipt["artifact_sha256"],
        "canonical_document_sha256": receipt["canonical_document_sha256"],
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }


def _plain_file_identity(path: Path, payload: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _seal_release_manifest(path: Path, manifest: dict) -> None:
    receipt_path = path.with_name(path.name + ".receipt.json")
    for output in (path, receipt_path):
        if output.exists():
            output.chmod(0o644)
    unsigned = dict(manifest)
    unsigned.pop("release_sha256", None)
    manifest["release_sha256"] = _stable_sha256(unsigned)
    _write_json(path, manifest)
    receipt = {
        "schema": "kbound_cct20_artifact_receipt_v1",
        "artifact_path": str(path.resolve()),
        "artifact_bytes": path.stat().st_size,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_document_sha256": _stable_sha256(manifest),
    }
    _write_json(receipt_path, receipt)
    path.chmod(0o444)
    receipt_path.chmod(0o444)


def _verdict_fixture(code: str) -> tuple[dict, bool, bool, bool]:
    protocol_strong = code in {
        "CONFIRMATORY_STRONG_SUCCESS",
        "CONFIRMATORY_PRIMARY_SUCCESS_MIXED_EFFECTS_MISSING",
    }
    expanded = code == "CONFIRMATORY_STRONG_SUCCESS"
    safe_pass = code != "NO_CONFIRMATORY_SUCCESS"
    expected_code = VALIDATOR._expected_cct20_verdict_code(
        protocol_strong_success=protocol_strong,
        expanded_mixed_effects_success=expanded,
        safe_utility_passes=safe_pass,
    )
    assert code == expected_code
    return (
        {
            "code": code,
            "confirmatory_strong_claim_supported": expanded,
            "primary_confirmatory_claim_supported": protocol_strong,
            "protocol_strong_success": protocol_strong,
            "expanded_mixed_effects_success": expanded,
            "safe_utility_passes": safe_pass,
            "manuscript_claim": VALIDATOR.CCT20_VERDICT_CLAIMS[code],
        },
        protocol_strong,
        expanded,
        safe_pass,
    )


def _numbers_payload(
    *,
    verdict: dict,
    safe_pass: bool,
    inference_sha256: str,
    secondary_disclosure: str,
) -> str:
    values = dict.fromkeys(VALIDATOR.REQUIRED_CCT20_NUMBER_MACROS, "0")
    values.update(
        {
            "CCTFalseAdaptCount": "0",
            "CCTSafeUtilityPass": rf"\textnormal{{{'yes' if safe_pass else 'no'}}}",
            "CCTVerdict": rf"\textnormal{{{verdict['code'].replace('_', ' ')}}}",
            "CCTManuscriptClaim": rf"\textnormal{{{verdict['manuscript_claim']}}}",
            "CCTSecondaryMetricDisclosure": rf"\textnormal{{{secondary_disclosure}}}",
            "CCTInferenceSHA": inference_sha256,
        }
    )
    return "\n".join(
        rf"\newcommand{{\{name}}}{{{values[name]}}}" for name in sorted(values)
    ) + "\n"


def _write_release_manifest(
    tmp_path: Path,
    *,
    verdict_code: str = "NO_CONFIRMATORY_SUCCESS",
) -> tuple[Path, Path]:
    artifacts = tmp_path / "artifacts"
    builder = tmp_path / "docs/research/kbound/scripts/build_cct20_release.py"
    release_generator = _plain_file_identity(builder, "# synthetic release builder fixture\n")

    dataset_items = [
        {
            "name": f"dataset_{index:03d}",
            **_plain_file_identity(
                artifacts / f"dataset_{index:03d}.bin",
                f"dataset {index}\n",
            ),
        }
        for index in range(4)
    ]
    code_items = [
        {
            "name": f"code_{index:03d}",
            **_plain_file_identity(
                artifacts / f"code_{index:03d}.py",
                f"# code {index}\n",
            ),
        }
        for index in range(138)
    ]
    execution_dependencies = {
        "dataset_count": 4,
        "code_count": 138,
        "total_count": 142,
        "aggregate_sha256": _stable_sha256(
            {
                "dataset_dependencies": dataset_items,
                "code_dependencies": code_items,
            }
        ),
        "dataset_items": dataset_items,
        "code_items": code_items,
    }

    development_items = []
    for index in range(55):
        identity = _received_json_identity(
            artifacts / f"development_trace_{index:02d}.json",
            {"trace_id": f"trace_{index:02d}"},
        )
        identity.update(
            {
                "trace_id": f"trace_{index:02d}",
                "trace_sha256": hashlib.sha256(f"trace {index}".encode()).hexdigest(),
            }
        )
        development_items.append(identity)
    development_traces = {
        "count": 55,
        "aggregate_sha256": _stable_sha256(development_items),
        "items": development_items,
    }

    prediction_items = []
    prediction_cell_hashes = []
    for index in range(45):
        cell_sha256 = hashlib.sha256(f"cell {index}".encode()).hexdigest()
        prediction_cell_hashes.append(cell_sha256)
        prediction_items.append(
            _received_json_identity(
                artifacts / f"prediction_cell_{index:02d}.json",
                {"cell": index, "cell_sha256": cell_sha256},
            )
        )
    prediction_items.sort(key=lambda row: row["path"])
    prediction_cells = {
        "count": 45,
        "aggregate_sha256": _stable_sha256(prediction_items),
        "items": prediction_items,
    }

    action_items = []
    for index in range(45):
        identity = _received_json_identity(
            artifacts / f"action_{index:02d}.json",
            {"action": index},
        )
        identity.update(
            {
                "checkpoint_seed": index // 9,
                "location_id": str(index % 9),
                "action_sha256": hashlib.sha256(f"action {index}".encode()).hexdigest(),
            }
        )
        action_items.append(identity)
    prediction_actions = {
        "count": 45,
        "aggregate_sha256": _stable_sha256(action_items),
        "items": action_items,
    }

    root_received = {
        name: _received_json_identity(artifacts / f"{name}.json", {"name": name})
        for name in (
            "development_gate",
            "development_trace_collection",
            "execution_seal",
            "two_way_inference",
        )
    }
    collection_sha256 = hashlib.sha256(b"prediction collection").hexdigest()
    root_received["prediction_collection"] = _received_json_identity(
        artifacts / "prediction_collection.json",
        {"name": "prediction_collection", "collection_sha256": collection_sha256},
    )
    score_path = artifacts / "one_shot_score.json"
    root_received["one_shot_score"] = _received_json_identity(
        score_path,
        {
            "name": "one_shot_score",
            "execution_seal_artifact_sha256": root_received["execution_seal"]["sha256"],
            "prediction_collection_sha256": collection_sha256,
            "target_image_count": 23_275,
            "checkpoint_count": 5,
            "location_count": 9,
            "cell_count": 45,
        },
    )
    checkpoint_audit = _plain_json_identity(
        artifacts / "checkpoint_audit.json",
        {"name": "checkpoint_audit"},
    )
    marker_request = {
        "execution_seal_artifact_sha256": root_received["execution_seal"]["sha256"],
        "prediction_collection_sha256": collection_sha256,
        "prediction_cell_sha256": sorted(prediction_cell_hashes),
        "output_path": str(score_path.resolve()),
        "expected_target_images": 23_275,
        "label_contract": "set_membership_top1_and_16_indicator_multilabel_macro_f1",
    }
    scoring_marker = _plain_json_identity(
        artifacts / "one_shot_scoring_marker.json",
        {
            "schema": "kbound_cct20_one_shot_score_marker_v1",
            "status": "SPENT_BEFORE_GROUND_TRUTH_LOAD",
            "request": marker_request,
            "request_sha256": _stable_sha256(marker_request),
        },
    )
    upstream = {
        "checkpoint_audit": checkpoint_audit,
        **root_received,
        "one_shot_scoring_marker": scoring_marker,
        "release_generator": release_generator,
        "execution_dependencies": execution_dependencies,
        "development_traces": development_traces,
        "prediction_cells": prediction_cells,
        "prediction_actions": prediction_actions,
    }

    verdict, protocol_strong, expanded, safe_pass = _verdict_fixture(verdict_code)
    safe_lower = 0.01 if safe_pass else -0.01
    primary_comparisons = {
        "versus_always_adapt": {
            "comparator": "always_adapt",
            "point_estimate": 0.02 if safe_pass else -0.001,
            "pointwise_95_ci": [safe_lower, 0.03],
            "simultaneous_bonferroni_confidence_level": 0.975,
            "simultaneous_bonferroni_97_5_ci": [0.005 if protocol_strong else -0.02, 0.04],
            "exact_location_sign_flip_p_one_sided": 0.01,
            "holm_adjusted_p": 0.02,
            "holm_reject_at_familywise_0_05": protocol_strong,
        },
        "versus_always_freeze": {
            "comparator": "always_freeze",
            "point_estimate": 0.01 if safe_pass else -0.002,
            "pointwise_95_ci": [-0.004 if safe_pass else -0.01, 0.02],
            "simultaneous_bonferroni_confidence_level": 0.975,
            "simultaneous_bonferroni_97_5_ci": [0.001 if protocol_strong else -0.02, 0.03],
            "exact_location_sign_flip_p_one_sided": 0.02,
            "holm_adjusted_p": 0.02,
            "holm_reject_at_familywise_0_05": protocol_strong,
        },
    }
    safe_utility = {
        "contrast_sign": "baseline_regret_minus_kga_regret; positive_favors_kga",
        "frozen_noninferiority_margin": -0.005,
        "versus_always_adapt": {
            "point_estimate": primary_comparisons["versus_always_adapt"]["point_estimate"],
            "pointwise_95_ci": primary_comparisons["versus_always_adapt"]["pointwise_95_ci"],
        },
        "versus_always_freeze": {
            "point_estimate": primary_comparisons["versus_always_freeze"]["point_estimate"],
            "pointwise_95_ci": primary_comparisons["versus_always_freeze"]["pointwise_95_ci"],
        },
        "passes": safe_pass,
    }
    secondary_disclosure = (
        "Cell-level 16-indicator macro-F1 is archived as descriptive secondary evidence; "
        "no post-hoc aggregate or inference claim is made."
    )
    secondary = {
        "metric": "16-indicator multilabel macro-F1 with top-1 as a one-hot prediction set",
        "scope": "complete cell-level score artifact",
        "aggregate_claim": None,
        "disclosure": secondary_disclosure,
    }
    generated = {}
    for name in (
        "cct20_numbers_tex",
        "cct20_primary_table_tex",
        "cct20_location_effects_tex",
    ):
        generated_path = tmp_path / "generated" / f"{name.removesuffix('_tex')}.tex"
        payload = (
            _numbers_payload(
                verdict=verdict,
                safe_pass=safe_pass,
                inference_sha256=root_received["two_way_inference"]["canonical_document_sha256"],
                secondary_disclosure=secondary_disclosure,
            )
            if name == "cct20_numbers_tex"
            else f"% {name}\n"
        )
        generated[name] = _plain_file_identity(generated_path, payload)
        generated_path.chmod(0o444)

    manifest = {
        "schema": "kbound_cct20_release_manifest_v1",
        "status": "RELEASE_COMPLETE",
        "artifacts_complete": True,
        "prospective_disclosure": (
            "outcome-unopened before model execution; aggregate target metadata had already been "
            "inspected during candidate ranking, so this is not described as literally label-unopened"
        ),
        "sign_conventions": {
            "adaptation_benefit": "adapted_accuracy_minus_frozen_accuracy",
            "primary_contrast": "baseline_regret_minus_kga_regret; positive_favors_kga",
            "helpful": "adaptation_benefit > 0",
            "zero": "adaptation_benefit = 0",
            "harmful": "adaptation_benefit < 0",
            "target_selection_lock_nonpositive_boundary": (
                "adaptation_benefit <= 0; the release separates exact zero from strictly harmful"
            ),
        },
        "verdict": verdict,
        "design": {
            "checkpoint_count": 5,
            "location_cluster_count": 9,
            "matrix_shape": [5, 9],
            "cell_count": 45,
            "cluster_unit_for_exact_test": "camera_location",
            "independent_checkpoint_tensor_identities_verified": True,
        },
        "action_exposure": {"counts": {"ADAPT": 15, "FREEZE": 15, "ABSTAIN": 15}},
        "false_adapt_accounting": {
            "event": "decision == ADAPT and adaptation_benefit <= 0",
            "unit": "checkpoint_x_location",
            "denominator_all_cells": 45,
            "adapt_count": 15,
            "false_adapt_count": 0,
            "false_adapt_rate_unconditional": 0.0,
            "false_adapt_rate_conditional": 0.0,
        },
        "adaptation_effect_mix": {},
        "primary_comparisons": primary_comparisons,
        "safe_utility": safe_utility,
        "secondary_outcome_reporting": secondary,
        "strong_success_checks": {
            "protocol_strong_success": protocol_strong,
            "expanded_empirical_bundle_including_mixed_effects": expanded,
        },
        "upstream_artifacts": upstream,
        "generated_artifacts": generated,
    }
    manifest_path = tmp_path / "cct20_release_manifest.json"
    _seal_release_manifest(manifest_path, manifest)
    return manifest_path, Path(root_received["two_way_inference"]["path"])


def _write_safe_utility_only_no_adapt_release(tmp_path: Path) -> Path:
    manifest_path, _ = _write_release_manifest(
        tmp_path,
        verdict_code="SAFE_UTILITY_ONLY",
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["action_exposure"]["counts"] = {"ADAPT": 0, "FREEZE": 44, "ABSTAIN": 1}
    manifest["false_adapt_accounting"].update(
        {
            "adapt_count": 0,
            "false_adapt_count": 0,
            "false_adapt_rate_unconditional": 0.0,
            "false_adapt_rate_conditional": None,
        }
    )
    freeze = manifest["primary_comparisons"]["versus_always_freeze"]
    freeze.update(
        {
            "point_estimate": 0.0,
            "pointwise_95_ci": [0.0, 0.0],
            "simultaneous_bonferroni_97_5_ci": [0.0, 0.0],
            "exact_location_sign_flip_p_one_sided": 1.0,
            "holm_adjusted_p": 1.0,
            "holm_reject_at_familywise_0_05": False,
        }
    )
    manifest["safe_utility"]["versus_always_freeze"] = {
        "point_estimate": 0.0,
        "pointwise_95_ci": [0.0, 0.0],
    }
    _seal_release_manifest(manifest_path, manifest)
    return manifest_path


def test_completed_cct20_claim_requires_release_manifest(tmp_path: Path) -> None:
    snippet = r"\section{CCT-20 evaluation} We report the completed CCT-20 evaluation."
    problems = VALIDATOR.validate_cct20_claims(
        snippet,
        release_manifest_path=tmp_path / "missing.json",
        repository_root=tmp_path,
    )
    assert any("lacks a readable release manifest" in problem for problem in problems)


def test_inputting_generated_cct20_results_activates_release_gate(tmp_path: Path) -> None:
    snippet = (
        r"\section{CCT-20 evaluation}"
        r"\input{paper/generated/cct20_primary_table}"
    )
    problems = VALIDATOR.validate_cct20_claims(
        snippet,
        release_manifest_path=tmp_path / "missing.json",
        repository_root=tmp_path,
    )
    assert any("lacks a readable release manifest" in problem for problem in problems)


def test_completed_cct20_claim_requires_generated_verdict_macros(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    snippet = r"\section{CCT-20 evaluation} We report the completed CCT-20 evaluation."
    problems = VALIDATOR.validate_cct20_claims(
        snippet,
        release_manifest_path=manifest_path,
        repository_root=tmp_path,
    )
    assert any("must consume generated \\CCTVerdict" in problem for problem in problems)
    assert any("must consume generated \\CCTManuscriptClaim" in problem for problem in problems)


@pytest.mark.parametrize("verdict_code", sorted(VALIDATOR.CCT20_VERDICT_CLAIMS))
def test_completed_cct20_claim_accepts_exact_generated_verdict(
    tmp_path: Path,
    verdict_code: str,
) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path, verdict_code=verdict_code)
    snippet = (
        r"\section{CCT-20 evaluation} We report the completed CCT-20 evaluation. "
        r"\CCTVerdict. \CCTManuscriptClaim"
    )
    assert (
        VALIDATOR.validate_cct20_claims(
            snippet,
            release_manifest_path=manifest_path,
            repository_root=tmp_path,
        )
        == []
    )


def test_safe_utility_verdict_allows_expanded_negative_claim_text(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path, verdict_code="SAFE_UTILITY_ONLY")
    snippet = (
        r"\section{CCT-20 evaluation} We report the completed CCT-20 evaluation. "
        r"The prospective CCT-20 result passes only the locked safe-utility check; "
        r"it does not establish the preregistered strong-success claim. "
        r"\CCTVerdict. \CCTManuscriptClaim"
    )
    assert (
        VALIDATOR.validate_cct20_claims(
            snippet,
            release_manifest_path=manifest_path,
            repository_root=tmp_path,
        )
        == []
    )


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        (
            "The CCT-20 study confirms a natural shift in which KGA uses both ADAPT and "
            "FREEZE.",
            "claims both ADAPT and FREEZE exposure",
        ),
        (
            "The CCT-20 study confirms that KGA improves on both fixed policies.",
            "claims improvement over both fixed policies",
        ),
    ],
)
def test_safe_utility_no_adapt_release_rejects_stale_success_claims(
    tmp_path: Path,
    claim: str,
    expected: str,
) -> None:
    manifest_path = _write_safe_utility_only_no_adapt_release(tmp_path)
    snippet = (
        rf"\section{{CCT-20 evaluation}} We report the completed CCT-20 evaluation. {claim} "
        r"\CCTVerdict. \CCTManuscriptClaim"
    )
    problems = VALIDATOR.validate_cct20_claims(
        snippet,
        release_manifest_path=manifest_path,
        repository_root=tmp_path,
    )
    assert any(expected in problem for problem in problems), problems


def test_safe_utility_no_adapt_release_accepts_matching_negative_claim(tmp_path: Path) -> None:
    manifest_path = _write_safe_utility_only_no_adapt_release(tmp_path)
    snippet = (
        r"\section{CCT-20 evaluation} We report the completed CCT-20 evaluation. "
        r"CCT-20 KGA made 44 FREEZE decisions, one ABSTAIN decision, and no ADAPT decisions. "
        r"It matched always freeze and improved on always adapt, not both fixed policies. "
        r"\CCTVerdict. \CCTManuscriptClaim"
    )
    assert (
        VALIDATOR.validate_cct20_claims(
            snippet,
            release_manifest_path=manifest_path,
            repository_root=tmp_path,
        )
        == []
    )


def test_completed_cct20_claim_rejects_verdict_specific_overstatement(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    snippet = (
        r"\section{CCT-20 evaluation} We report the completed CCT-20 evaluation. "
        r"The CCT-20 safe-utility check passes. \CCTVerdict. \CCTManuscriptClaim"
    )
    problems = VALIDATOR.validate_cct20_claims(
        snippet,
        release_manifest_path=manifest_path,
        repository_root=tmp_path,
    )
    assert any("claims safe utility" in problem for problem in problems), problems


def test_primary_only_verdict_rejects_mixed_effects_overstatement(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(
        tmp_path,
        verdict_code="CONFIRMATORY_PRIMARY_SUCCESS_MIXED_EFFECTS_MISSING",
    )
    snippet = (
        r"\section{CCT-20 evaluation} We report the completed CCT-20 evaluation. "
        r"For CCT-20, both helpful and harmful cases are present. "
        r"\CCTVerdict. \CCTManuscriptClaim"
    )
    problems = VALIDATOR.validate_cct20_claims(
        snippet,
        release_manifest_path=manifest_path,
        repository_root=tmp_path,
    )
    assert any("mixed helpful/harmful evidence" in problem for problem in problems), problems


def test_completed_cct20_claim_rejects_a_different_literal_verdict(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    snippet = (
        r"\section{CCT-20 evaluation} We report the completed CCT-20 evaluation as "
        r"SAFE UTILITY ONLY. \CCTVerdict. \CCTManuscriptClaim"
    )
    problems = VALIDATOR.validate_cct20_claims(
        snippet,
        release_manifest_path=manifest_path,
        repository_root=tmp_path,
    )
    assert any("names verdict SAFE_UTILITY_ONLY" in problem for problem in problems), problems


def test_completed_cct20_claim_rejects_stale_upstream_hash(tmp_path: Path) -> None:
    manifest_path, artifact = _write_release_manifest(tmp_path)
    artifact.write_text('{"complete": false}\n')
    snippet = r"\section{CCT-20 evaluation} CCT-20 results show a completed evaluation."
    problems = VALIDATOR.validate_cct20_claims(
        snippet,
        release_manifest_path=manifest_path,
        repository_root=tmp_path,
        deep_local_provenance=True,
    )
    assert any("SHA-256 mismatch" in problem for problem in problems), problems


def test_portable_release_validation_does_not_open_machine_local_upstreams(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    (tmp_path / "artifacts").rename(tmp_path / "disconnected-machine-local-artifacts")

    document, portable_problems = VALIDATOR.validate_cct20_release_manifest(
        manifest_path,
        repository_root=tmp_path,
    )
    assert document is not None
    assert portable_problems == []

    _, deep_problems = VALIDATOR.validate_cct20_release_manifest(
        manifest_path,
        repository_root=tmp_path,
        deep_local_provenance=True,
    )
    assert any("artifact is missing" in problem for problem in deep_problems)


def test_release_runbook_keeps_deep_local_cct_provenance_explicit() -> None:
    runbook = (
        ROOT / "docs/research/kbound/runbooks/release_candidate.sh"
    ).read_text(encoding="utf-8")
    assert "deep-local-provenance" in runbook
    assert "KBOUND_DEEP_LOCAL_CCT20_PROVENANCE=1" in runbook
    assert "--deep-local-cct20-provenance" in runbook


def test_release_manifest_requires_builder_ledgers_and_aggregate_hashes(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["upstream_artifacts"].pop("one_shot_scoring_marker")
    manifest["upstream_artifacts"]["execution_dependencies"]["aggregate_sha256"] = "0" * 64
    _seal_release_manifest(manifest_path, manifest)

    _, problems = VALIDATOR.validate_cct20_release_manifest(
        manifest_path,
        repository_root=tmp_path,
    )
    assert any("builder-required upstream ledgers" in problem for problem in problems)
    assert any("execution dependency aggregate" in problem for problem in problems)


def test_release_manifest_rejects_stale_received_receipt(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    binding = manifest["upstream_artifacts"]["development_gate"]
    receipt_path = Path(binding["receipt_path"])
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    receipt_path.chmod(0o444)

    _, problems = VALIDATOR.validate_cct20_release_manifest(
        manifest_path,
        repository_root=tmp_path,
        deep_local_provenance=True,
    )
    assert any("receipt" in problem and "SHA-256 mismatch" in problem for problem in problems)


def test_release_manifest_cross_binds_spent_scoring_marker(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    marker_binding = manifest["upstream_artifacts"]["one_shot_scoring_marker"]
    marker_path = Path(marker_binding["path"])
    marker = json.loads(marker_path.read_text())
    marker["request"]["expected_target_images"] = 1
    marker["request_sha256"] = _stable_sha256(marker["request"])
    manifest["upstream_artifacts"]["one_shot_scoring_marker"] = _plain_json_identity(
        marker_path,
        marker,
    )
    _seal_release_manifest(manifest_path, manifest)

    _, problems = VALIDATOR.validate_cct20_release_manifest(
        manifest_path,
        repository_root=tmp_path,
        deep_local_provenance=True,
    )
    assert any("scoring marker differs from the release chain" in problem for problem in problems)


def test_release_manifest_recomputes_safe_utility_pass_from_intervals(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["safe_utility"]["passes"] = True
    _seal_release_manifest(manifest_path, manifest)

    _, problems = VALIDATOR.validate_cct20_release_manifest(
        manifest_path,
        repository_root=tmp_path,
    )
    assert any("stale safe-utility pass flag" in problem for problem in problems), problems


def test_release_manifest_binds_generated_verdict_text_to_evidence(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    binding = manifest["generated_artifacts"]["cct20_numbers_tex"]
    numbers_path = Path(binding["path"])
    numbers_path.chmod(0o644)
    payload = numbers_path.read_text().replace(
        r"\newcommand{\CCTVerdict}{\textnormal{NO CONFIRMATORY SUCCESS}}",
        r"\newcommand{\CCTVerdict}{\textnormal{SAFE UTILITY ONLY}}",
    )
    manifest["generated_artifacts"]["cct20_numbers_tex"] = _plain_file_identity(
        numbers_path,
        payload,
    )
    numbers_path.chmod(0o444)
    _seal_release_manifest(manifest_path, manifest)

    _, problems = VALIDATOR.validate_cct20_release_manifest(
        manifest_path,
        repository_root=tmp_path,
    )
    assert any("number macros are inconsistent" in problem for problem in problems), problems


def test_public_preregistration_claim_accepts_hashed_registry_record(tmp_path: Path) -> None:
    manifest_path, _ = _write_release_manifest(tmp_path)
    registry_snapshot = tmp_path / "registry/cct20_registry_record.json"
    registry_snapshot.parent.mkdir()
    registry_snapshot.write_text('{"registered": "before-target-execution"}\n')
    manifest = json.loads(manifest_path.read_text())
    manifest["public_registry_evidence"] = {
        "registry_id": "CCT20-SYNTHETIC-TEST",
        "url": "https://registry.example/CCT20-SYNTHETIC-TEST",
        "registered_before_target_execution": True,
        "snapshot": {
            "path": str(registry_snapshot.resolve()),
            "bytes": registry_snapshot.stat().st_size,
            "sha256": hashlib.sha256(registry_snapshot.read_bytes()).hexdigest(),
        },
    }
    _seal_release_manifest(manifest_path, manifest)

    snippet = r"\section{CCT-20 protocol} The study was publicly preregistered."
    assert (
        VALIDATOR.validate_cct20_claims(
            snippet,
            release_manifest_path=manifest_path,
            repository_root=tmp_path,
        )
        == []
    )
