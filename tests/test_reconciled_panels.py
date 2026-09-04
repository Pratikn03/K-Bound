"""Regression checks for the source-backed natural/corruption panel replay."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "experiments/kbound/results/reconciled_panels_v1"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYNC = _load_script(ROOT / "scripts/sync_reconciled_panels.py", "three_source_sync")
COMPAT = _load_script(
    ROOT / "docs/research/kbound/scripts/build_results_source_compat.py",
    "three_source_compat",
)
RECON = _load_script(ROOT / "scripts/reconcile_result_panels.py", "controlled_grid_reconciliation")


def load(name: str) -> dict:
    return json.loads((PANEL_ROOT / name).read_text())


def test_sync_loader_rejects_non_object_json(tmp_path: Path) -> None:
    source = tmp_path / "not-an-object.json"
    source.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        SYNC._load(source)


def test_compact_sources_are_complete_and_hash_locked() -> None:
    manifest = load("source_manifest.json")
    generator_hash = hashlib.sha256((ROOT / "scripts/reconcile_result_panels.py").read_bytes()).hexdigest()
    assert manifest["file_count"] == 106
    assert len(manifest["files"]) == 106
    assert manifest["generator_sha256"] == generator_hash
    assert load("canonical_panel_results.json")["generator_sha256"] == generator_hash
    assert "/Users/" not in json.dumps(manifest)
    for row in manifest["files"]:
        path = ROOT / row["destination"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["compact_sha256"]
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/" + "Volumes/T9" not in text


def test_reconciled_conflicts_and_negative_panels_are_locked() -> None:
    canonical = load("canonical_panel_results.json")
    assert canonical["runtime"]["numpy"] == "2.4.4"
    assert canonical["runtime"]["scikit_learn"] == "1.8.0"
    panels = canonical["panels"]

    office = panels["officehome"]["primary"]["exact_rank_transfer_score"]
    assert np.isclose(office["regret"]["kga"], 0.01582417582417583)
    assert office["regret"]["kga"] == office["regret"]["always_freeze"]
    assert office["adapt_count"] == 0
    assert not office["point_beats_both"]
    assert not office["seed_inference"]["ci_robust_beats_both"]

    replication = panels["officehome"]["test_stream_seed_replication"]
    assert np.isclose(
        replication["exact_rank_transfer_score"]["regret"]["kga"],
        0.02150997150997151,
    )
    assert replication["calibration"]["a7_status"] == "not_established"

    iwildcam = panels["iwildcam"]["primary"]["exact_rank_transfer_score"]
    assert iwildcam["adapt_count"] == 0
    assert iwildcam["regret"]["kga"] == iwildcam["regret"]["always_freeze"]
    assert iwildcam["freeze_count"] == 21

    reconciliation = panels["iwildcam"]["historical_reconciliation"]
    assert reconciliation["historical_claim"]["beats_both"]
    assert not reconciliation["corrected_claim"]["point_beats_both"]
    assert reconciliation["historical_claim"]["epsilon"] < reconciliation["corrected_claim"]["epsilon"]
    assert reconciliation["status"] == "superseded_not_promotable"

    sar = panels["imagenetc"]["panel"]["candidates"]["sar"]
    assert sar["false_adapt_count"] == 0
    assert sar["point_beats_both"]
    assert not sar["seed_inference"]["ci_robust_beats_both"]
    assert all(row["crossfit_protocol"]["status"] == "ok" for row in sar["per_file"])
    assert all(row["historical_fields"]["b_hat"]["current_authority"] is False for row in sar["per_file"])

    pacs = panels["pacs"]
    assert pacs["aggregate_matches_seed_files"]
    assert not pacs["decision_replay_available"]

    imagenet_r = panels["imagenet_r"]["panel"]["architecture_panel_aggregate"]
    assert imagenet_r["n"] == 480
    assert imagenet_r["regret"]["kga"] > imagenet_r["regret"]["always_adapt"]
    assert not imagenet_r["point_beats_both"]
    kappa_one = next(row for row in imagenet_r["kappa_sweep"] if row["kappa"] == 1.0)
    assert np.isclose(kappa_one["regret"], imagenet_r["regret"]["kga"])
    assert np.isclose(kappa_one["yield"], imagenet_r["decision_coverage"])
    assert all(
        file_row["crossfit_protocol"]["status"] == "ok"
        for row in panels["imagenet_r"]["panel"]["candidates"].values()
        for file_row in row["per_file"]
    )

    imagenetc = panels["imagenetc"]["panel"]["architecture_panel_aggregate"]
    assert np.isclose(imagenetc["radius_diagnostics"]["yield"], imagenetc["decision_coverage"])
    assert imagenetc["radius_diagnostics"]["eps_mean"] > 0.0


def test_imagenetc_source_is_bound_to_canonical_crossfit_authority() -> None:
    source = load("source/imagenetc/per_condition_imagenetc_sar_seed0.json")
    records = source["records"]
    features = np.asarray([row["Z"] for row in records], dtype=float)
    benefit = np.asarray([row["B"] for row in records], dtype=float)
    generated = load("canonical_panel_results.json")["panels"]["imagenetc"]["panel"]
    seed0 = generated["candidates"]["sar"]["per_file"][0]
    protocol = seed0["crossfit_protocol"]
    compact_sha256 = lambda value: hashlib.sha256(  # noqa: E731 - compact independent hash oracle
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert protocol["input_sha256"] == {
        "Z": compact_sha256(features.tolist()),
        "B": compact_sha256(benefit.tolist()),
        "sample_ids": compact_sha256(RECON._grid_cell_ids(records)),
    }
    assert protocol["software_versions"] == {
        "python": "3.12.13",
        "numpy": "2.4.4",
        "scikit_learn": "1.8.0",
    }
    cells = seed0["current_cell_authority"]["cells"]
    assert hashlib.sha256(
        (json.dumps(cells, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest() == seed0["current_cell_authority"]["sha256"]
    assert seed0["historical_fields"]["b_hat"]["current_authority"] is False


def test_missing_locked_tracks_use_exact_rank_and_retain_negative_scope() -> None:
    panels = load("canonical_panel_results.json")["panels"]

    cifar10c = panels["cifar10c"]
    assert len(cifar10c["source_provenance"]) == 15
    tent = cifar10c["panel"]["candidates"]["tent"]
    eata = cifar10c["panel"]["candidates"]["eata"]
    sar = cifar10c["panel"]["candidates"]["sar"]
    assert tent["n"] == eata["n"] == sar["n"] == 2160
    assert tent["point_beats_both"]
    assert eata["point_beats_both"]
    assert not sar["point_beats_both"]
    assert sar["regret"]["kga"] > sar["regret"]["always_adapt"]
    assert cifar10c["headline_promotion"]["sar"].startswith("withheld")

    camelyon = panels["camelyon17"]
    ood = camelyon["ood"]["replay"]["exact_rank_transfer_score"]
    assert ood["n"] == 18
    assert ood["adapt_count"] == 18
    assert np.isclose(ood["regret"]["kga"], 0.0)
    assert np.isclose(ood["regret"]["always_adapt"], 0.0)
    assert np.isclose(ood["regret"]["always_freeze"], 0.1381293402777778)
    assert not camelyon["ood"]["headline_promotion"]["eligible"]

    b_v2 = camelyon["b_v2_diagnostic"]
    assert not b_v2["panel"]["candidates"]["sar"]["point_beats_both"]
    assert not b_v2["headline_promotion"]["eligible"]
    assert "diagnostic" in b_v2["claim_scope"]

    rxrx1 = panels["rxrx1"]
    rx_primary = rxrx1["primary_model_seed0"]["exact_rank_transfer_score"]
    assert rx_primary["n"] == 60
    assert rx_primary["adapt_count"] == 0
    assert np.isclose(rx_primary["regret"]["always_adapt"], 0.2530598958333333)
    robustness = rxrx1["model_seed_robustness"]["aggregate"]
    assert robustness["n_model_seeds"] == 3
    assert robustness["all_tie_always_freeze"]
    assert not robustness["any_point_beats_both"]
    assert not rxrx1["headline_promotion"]["eligible"]

    cifar101 = panels["cifar101"]
    k_score = cifar101["replay"]["exact_rank_transfer_score"]
    assert k_score["n"] == 48
    assert k_score["adapt_count"] == 0
    assert k_score["fa_u"] == 0.0
    assert np.isclose(k_score["regret"]["kga"], k_score["regret"]["always_freeze"])
    assert not k_score["point_beats_both"]
    assert not cifar101["headline_promotion"]["eligible"]


def test_new_panel_provenance_is_complete_and_portable() -> None:
    panels = load("canonical_panel_results.json")["panels"]
    provenance_groups = [
        panels["cifar10c"]["source_provenance"],
        panels["camelyon17"]["ood"]["source_provenance"],
        panels["camelyon17"]["b_v2_diagnostic"]["source_provenance"],
        panels["rxrx1"]["source_provenance"],
        panels["cifar101"]["source_provenance"],
    ]
    for rows in provenance_groups:
        assert rows
        for row in rows:
            assert row["compact_path"].startswith("experiments/kbound/results/reconciled_panels_v1/source/")
            assert len(row["compact_sha256"]) == 64
            assert row["archive_relative_path"].startswith("experiments/kbound/results/")
            assert len(row["original_sha256"]) == 64
            assert row["original_bytes"] > 0
            assert "/Users/" not in json.dumps(row)
            assert "/Volumes/" not in json.dumps(row)


def test_evidence_schemas_are_explicit_and_track_specific() -> None:
    panels = load("canonical_panel_results.json")["panels"]
    expected_dimensions = {
        "officehome": panels["officehome"]["primary"]["evidence_contract"],
        "iwildcam": panels["iwildcam"]["primary"]["evidence_contract"],
        "camelyon17_ood": panels["camelyon17"]["ood"]["replay"]["evidence_contract"],
        "cifar10c": panels["cifar10c"]["panel"]["evidence_contract"],
        "imagenetc": panels["imagenetc"]["panel"]["evidence_contract"],
        "imagenet_r": panels["imagenet_r"]["panel"]["evidence_contract"],
    }
    assert {name: row["dimension"] for name, row in expected_dimensions.items()} == {
        "officehome": 18,
        "iwildcam": 11,
        "camelyon17_ood": 17,
        "cifar10c": 11,
        "imagenetc": 11,
        "imagenet_r": 11,
    }
    for row in expected_dimensions.values():
        assert row["names_recovered"]
        assert len(row["feature_names"]) == row["dimension"]
        assert len(row["schema_sha256"]) == 64


def test_run_seed_bootstrap_cannot_promote_independent_checkpoint_claims() -> None:
    panels = load("canonical_panel_results.json")["panels"]
    seed_inference_rows = [
        panels["officehome"]["primary"]["exact_rank_transfer_score"]["seed_inference"],
        panels["cifar10c"]["panel"]["architecture_panel_aggregate"]["seed_inference"],
        panels["imagenetc"]["panel"]["architecture_panel_aggregate"]["seed_inference"],
        panels["imagenet_r"]["panel"]["architecture_panel_aggregate"]["seed_inference"],
    ]
    for row in seed_inference_rows:
        assert not row["ci_robust_beats_both"]
        assert "independent checkpoint identities are not recorded" in row["reason"]
        bootstrap = row["descriptive_seed_bootstrap"]
        assert "conditional on the archived checkpoint/protocol" in bootstrap["unit"]


def test_generated_paper_manifest_matches_every_cifar_candidate() -> None:
    panels = load("canonical_panel_results.json")["panels"]
    paper_manifest = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text()
    )["tracks"]
    for candidate in ("tent", "eata", "sar"):
        score = panels["cifar10c"]["panel"]["candidates"][candidate]
        paper = paper_manifest[f"cifar10c_{candidate}"]
        assert np.allclose(
            paper["regret"],
            [
                score["regret"]["kga"],
                score["regret"]["always_adapt"],
                score["regret"]["always_freeze"],
            ],
        )
        assert paper["decision_counts"] == {
            "ADAPT": score["adapt_count"],
            "FREEZE": score["freeze_count"],
            "ABSTAIN": score["abstain_count"],
        }
        assert paper["point_beats_both"] == score["point_beats_both"]
        assert not paper["ci_robust_beats_both"]
        assert "one archived checkpoint/protocol" in paper["inference_scope"]


def test_three_source_oof_uses_diagnostic_claim_authority() -> None:
    manifest = json.loads((ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text())
    ledger = json.loads((ROOT / "docs/research/kbound/claim_ledger.json").read_text())
    row = manifest["tracks"]["three_source_oof"]
    authority = next(claim for claim in ledger["claims"] if claim["claim_id"] == "KB-CLAIM-024")
    historical_wording = (
        "historical researcher-constructed routing aggregate; rerun required under reconciled per-track decisions"
    )

    assert row["claim_id"] == "KB-CLAIM-024"
    assert row["claim_status"] == "diagnostic"
    assert row["status"] == "historical_policy_only"
    assert row["current_policy_authority"] is False
    assert row["numeric_release_eligible"] is False
    assert row["headline_promotion_eligible"] is False
    assert row["policy_synchronized"] is False
    assert "seal" not in row
    assert row["historical_audit_seal"].startswith(
        "docs/research/kbound/archive/superseded_empirical_authorities_2026-09-02/retired_tree/"
    )
    assert row["historical_audit_seal"].endswith("#three_source_oof")
    assert authority["claim_text"] == historical_wording
    assert authority["allowed_wording"] == historical_wording
    assert row["verdict"] == authority["claim_text"]
    lowered = row["verdict"].lower()
    assert "beats-both" not in lowered
    assert "natural-shift" not in lowered
    assert "transfer result" not in lowered


def test_three_source_oof_helper_overwrites_stale_row_from_synthetic_authority() -> None:
    stale = {
        "regret": [0.0059117, 0.0632323, 0.0342043],
        "false_adapt": 0.0,
        "n_conditions": 143,
        "verdict": "stale CI win wording",
        "source": "historical/source.json",
        "caveat": "historical scorer caveat",
        "seal": "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json#three_source_oof",
    }
    original_numeric_and_evidence = {
        key: copy.deepcopy(stale[key]) for key in ("regret", "false_adapt", "n_conditions", "source", "caveat")
    }
    authority = {
        "claim_id": "KB-CLAIM-024",
        "status": "diagnostic",
        "claim_text": (
            "historical researcher-constructed routing aggregate; rerun required under reconciled per-track decisions"
        ),
        "allowed_wording": (
            "historical researcher-constructed routing aggregate; rerun required under reconciled per-track decisions"
        ),
        "forbidden_wording": ["natural-shift win", "transfer result"],
    }

    SYNC._sync_three_source_oof(stale, {"claims": [authority]})

    assert {key: stale[key] for key in original_numeric_and_evidence} == original_numeric_and_evidence
    assert stale["verdict"] == authority["claim_text"]
    assert stale["claim_id"] == authority["claim_id"]
    assert stale["claim_status"] == "diagnostic"
    assert stale["status"] == "historical_policy_only"
    assert stale["current_policy_authority"] is False
    assert stale["numeric_release_eligible"] is False
    assert stale["headline_promotion_eligible"] is False
    assert stale["release_eligible_win"] is False
    assert stale["policy_synchronized"] is False
    assert "seal" not in stale
    assert stale["historical_audit_seal"].startswith(
        "docs/research/kbound/archive/superseded_empirical_authorities_2026-09-02/retired_tree/"
    )

    first = copy.deepcopy(stale)
    SYNC._sync_three_source_oof(stale, {"claims": [authority]})
    assert stale == first


def test_three_source_claim_normalization_then_projection_is_idempotent() -> None:
    ledger = {
        "claims": [
            {
                "claim_id": "KB-CLAIM-024",
                "claim_text": "This aggregate beats both fixed policies.",
                "status": "supported",
                "allowed_wording": "promoted win",
                "forbidden_wording": [],
                "supporting_artifacts": ["historical/source.json"],
            }
        ]
    }
    row = {
        "regret": [0.0059117, 0.0632323, 0.0342043],
        "false_adapt": 0.0,
        "n_conditions": 143,
        "verdict": "stale CI win wording",
        "source": "historical/source.json",
    }

    SYNC._sync_three_source_claim_authority(ledger)
    SYNC._sync_three_source_oof(row, ledger)
    first = copy.deepcopy((ledger, row))
    SYNC._sync_three_source_claim_authority(ledger)
    SYNC._sync_three_source_oof(row, ledger)

    authority = ledger["claims"][0]
    assert authority["claim_text"] == SYNC.THREE_SOURCE_HISTORICAL_WORDING
    assert authority["allowed_wording"] == authority["claim_text"]
    assert authority["status"] == "diagnostic"
    assert row["verdict"] == authority["claim_text"]
    assert (ledger, row) == first


@pytest.mark.parametrize(
    "authority",
    [
        {"claim_id": "KB-CLAIM-024", "status": "supported", "allowed_wording": "win"},
        {"claim_id": "KB-CLAIM-024", "status": "diagnostic"},
        {
            "claim_id": "KB-CLAIM-024",
            "status": "diagnostic",
            "allowed_wording": "not a historical aggregate",
        },
        {
            "claim_id": "KB-CLAIM-024",
            "status": "diagnostic",
            "claim_text": "This constructed aggregate beats both fixed policies.",
            "allowed_wording": (
                "historical researcher-constructed routing aggregate; rerun required under "
                "reconciled per-track decisions"
            ),
            "forbidden_wording": ["beats both"],
        },
        {
            "claim_id": "KB-CLAIM-024",
            "status": "diagnostic",
            "claim_text": (
                "historical researcher-constructed routing aggregate; rerun required under "
                "reconciled per-track decisions; CI beats both"
            ),
            "allowed_wording": (
                "historical researcher-constructed routing aggregate; rerun required under "
                "reconciled per-track decisions; CI beats both"
            ),
            "forbidden_wording": ["promoted beats-both claim", "natural-shift win", "transfer result"],
        },
    ],
)
def test_three_source_oof_helper_rejects_malformed_or_non_diagnostic_authority(authority) -> None:
    with pytest.raises(ValueError, match="KB-CLAIM-024 authority"):
        SYNC._sync_three_source_oof({"verdict": "stale"}, {"claims": [authority]})


def test_results_source_compat_projects_normalized_track_without_source_tree_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    normalized = {
        "regret": [0.0059117, 0.0632323, 0.0342043],
        "false_adapt": 0.0,
        "n_conditions": 143,
        "verdict": "stale CI win wording",
        "source": "historical/source.json",
        "caveat": "historical scorer caveat",
        "seal": "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json#three_source_oof",
    }
    authority = {
        "claim_id": "KB-CLAIM-024",
        "status": "diagnostic",
        "claim_text": (
            "historical researcher-constructed routing aggregate; rerun required under reconciled per-track decisions"
        ),
        "allowed_wording": (
            "historical researcher-constructed routing aggregate; rerun required under reconciled per-track decisions"
        ),
        "forbidden_wording": [],
    }
    SYNC._sync_three_source_oof(normalized, {"claims": [authority]})

    table_path = tmp_path / "docs/research/kbound/paper/generated/kbound_result_manifest.json"
    claims_path = tmp_path / "docs/research/kbound/RESULT_MANIFEST.json"
    output_path = tmp_path / "docs/research/kbound/results_source.json"
    source_tree_output = ROOT / "docs/research/kbound/results_source.json"
    source_tree_bytes = source_tree_output.read_bytes()
    table_path.parent.mkdir(parents=True)
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        json.dumps({"regenerated_utc": "test", "alpha": 0.1, "tracks": {"three_source_oof": normalized}})
    )
    claims_path.write_text("{}")
    monkeypatch.setattr(COMPAT, "ROOT", tmp_path)
    monkeypatch.setattr(COMPAT, "TABLE", table_path)
    monkeypatch.setattr(COMPAT, "CLAIMS", claims_path)
    monkeypatch.setattr(COMPAT, "OUT", output_path)

    COMPAT.main()

    generated = json.loads(output_path.read_text())
    assert generated["tracks"]["three_source_oof"] == normalized
    assert source_tree_output.read_bytes() == source_tree_bytes


def test_secondary_release_surfaces_match_counts_and_withhold_iwildcam() -> None:
    panels = load("canonical_panel_results.json")["panels"]
    expected = {
        candidate: {
            "ADAPT": panels["cifar10c"]["panel"]["candidates"][candidate]["adapt_count"],
            "FREEZE": panels["cifar10c"]["panel"]["candidates"][candidate]["freeze_count"],
            "ABSTAIN": panels["cifar10c"]["panel"]["candidates"][candidate]["abstain_count"],
        }
        for candidate in ("tent", "eata")
    }

    generated = json.loads((ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text())
    uniform = json.loads((ROOT / "docs/research/kbound/paper/generated/uniform_verdicts.json").read_text())
    metrics = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json").read_text()
    )

    accounting = {row["track"]: row for row in generated["decision_accounting_summary"]["rows"]}
    uniform_rows = {row["track"]: row for row in uniform["wave"]}
    metric_rows = {row["track"]: row for row in metrics["tracks"]}
    for candidate, title in (("tent", "Tent"), ("eata", "EATA")):
        assert generated["tracks"][f"cifar10c_{candidate}"]["decision_counts"] == expected[candidate]
        assert {
            action: accounting[f"CIFAR-10-C {title}"][action] for action in ("ADAPT", "FREEZE", "ABSTAIN")
        } == expected[candidate]
        assert uniform_rows[f"CIFAR-10-C {title}"]["decision_counts"] == expected[candidate]
        assert {
            action.upper(): metric_rows[f"CIFAR-10-C {title.upper()}"]["actions"][action]["count"]
            for action in ("adapt", "freeze", "abstain")
        } == expected[candidate]

    iwild_manifest = generated["tracks"]["iwildcam_H_v2"]
    assert iwild_manifest["numeric_release_eligible"] is False
    assert iwild_manifest["regret"] is None
    assert all(value is None for value in iwild_manifest["decision_counts"].values())
    assert "ci_vs_adapt" not in iwild_manifest
    assert "ci_vs_freeze" not in iwild_manifest
    assert iwild_manifest["seal"] is None
    iwild_uniform = uniform_rows["iWildCam H v2"]
    assert iwild_uniform["numeric_release_eligible"] is False
    assert all(iwild_uniform[field] is None for field in ("regret_kga", "regret_adapt", "regret_freeze", "FA_u"))
    iwild_metrics = metric_rows["iWildCam"]
    assert iwild_metrics["numeric_release_eligible"] is False
    assert all(
        iwild_metrics["actions"][action]["count"] is None and iwild_metrics["actions"][action]["rate"] is None
        for action in ("adapt", "freeze", "abstain")
    )

    claim_ledger = json.loads((ROOT / "docs/research/kbound/claim_ledger.json").read_text())
    iwild_claim = next(row for row in claim_ledger["claims"] if row["claim_id"] == "KB-CLAIM-021")
    assert iwild_claim["status"] == "withheld"
    result_manifest = json.loads((ROOT / "docs/research/kbound/RESULT_MANIFEST.json").read_text())
    assert all(row["claim_id"] != "KB-CLAIM-021" for row in result_manifest["results"])
    results_source = json.loads((ROOT / "docs/research/kbound/results_source.json").read_text())
    source_iwild = results_source["tracks"]["iwildcam_H_v2"]
    assert source_iwild["regret"] is None
    assert "ci_vs_adapt" not in source_iwild
    assert "ci_vs_freeze" not in source_iwild
    assert source_iwild["seal"] is None


def test_generated_current_policy_surfaces_match_canonical_exactly() -> None:
    panels = load("canonical_panel_results.json")["panels"]
    generated = json.loads((ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text())
    uniform = json.loads((ROOT / "docs/research/kbound/paper/generated/uniform_verdicts.json").read_text())
    metrics = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json").read_text()
    )
    uniform_rows = {row["track"]: row for row in uniform["wave"]}
    metric_rows = {row["track"]: row for row in metrics["tracks"]}
    display_name = {"tent": "Tent", "eata": "EATA", "sar": "SAR"}

    cases = (
        *(
            (
                f"cifar10c_{candidate}",
                f"CIFAR-10-C {display_name[candidate]}",
                f"CIFAR-10-C {candidate.upper()}",
                panels["cifar10c"]["panel"]["candidates"][candidate],
            )
            for candidate in ("tent", "eata", "sar")
        ),
        *(
            (
                f"imagenetc_{candidate}",
                f"ImageNet-C {display_name[candidate]}",
                f"ImageNet-C {candidate.upper()}",
                panels["imagenetc"]["panel"]["candidates"][candidate],
            )
            for candidate in ("tent", "eata", "sar")
        ),
        (
            "officehome_M_v2",
            "Office-Home M v2",
            "OfficeHome",
            panels["officehome"]["primary"]["exact_rank_transfer_score"],
        ),
        (
            "camelyon17_ood",
            "Camelyon17 OOD",
            "Camelyon17",
            panels["camelyon17"]["ood"]["replay"]["exact_rank_transfer_score"],
        ),
        (
            "rxrx1_J",
            "RxRx1 J",
            "RxRx1 sar_online",
            panels["rxrx1"]["primary_model_seed0"]["exact_rank_transfer_score"],
        ),
        (
            "cifar10_1_K",
            "CIFAR-10.1 K",
            "CIFAR-10.1 TENT",
            panels["cifar101"]["replay"]["exact_rank_transfer_score"],
        ),
    )
    for manifest_key, uniform_key, metric_key, score in cases:
        expected_regret = [
            score["regret"]["kga"],
            score["regret"]["always_adapt"],
            score["regret"]["always_freeze"],
        ]
        expected_counts = {
            "ADAPT": score["adapt_count"],
            "FREEZE": score["freeze_count"],
            "ABSTAIN": score["abstain_count"],
        }
        assert np.allclose(generated["tracks"][manifest_key]["regret"], expected_regret)
        assert generated["tracks"][manifest_key]["decision_counts"] == expected_counts
        assert np.allclose(
            [
                uniform_rows[uniform_key]["regret_kga"],
                uniform_rows[uniform_key]["regret_adapt"],
                uniform_rows[uniform_key]["regret_freeze"],
            ],
            expected_regret,
        )
        assert uniform_rows[uniform_key]["decision_counts"] == expected_counts
        assert metric_rows[metric_key]["regret_kga_adapt_freeze"] == expected_regret

    sar = panels["cifar10c"]["panel"]["candidates"]["sar"]
    sar_metrics = metric_rows["CIFAR-10-C SAR"]
    assert sar_metrics["n_decisions"] == sar["n"]
    assert sar_metrics["actions"]["adapt"]["count"] == sar["adapt_count"]
    assert sar_metrics["regret_kga_adapt_freeze"] == [
        sar["regret"]["kga"],
        sar["regret"]["always_adapt"],
        sar["regret"]["always_freeze"],
    ]


def test_historical_policy_artifacts_cannot_imply_a_current_win() -> None:
    generated = json.loads((ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text())
    uniform = json.loads((ROOT / "docs/research/kbound/paper/generated/uniform_verdicts.json").read_text())
    result_manifest = json.loads((ROOT / "docs/research/kbound/RESULT_MANIFEST.json").read_text())
    current_tent_regret = load("canonical_panel_results.json")["panels"]["cifar10c"]["panel"]["candidates"]["tent"]["regret"]["kga"]

    head = generated["headtohead"]
    assert head["policy_synchronized"] is False
    assert head["current_policy_authority"] is False
    assert head["numeric_release_eligible"] is False
    assert head["release_eligible_win"] is False
    assert head["current_exact_rank_reference"]["kga_regret"] == pytest.approx(current_tent_regret)
    assert "0.0015851849" not in json.dumps(generated)

    cluster = generated["tracks"]["cifar10c_tent"]["historical_cluster_resampling"]
    assert cluster["policy_synchronized"] is False
    assert cluster["current_policy_authority"] is False
    assert cluster["release_eligible"] is False
    assert cluster["convention"] == generated["ci_convention"]
    assert cluster["as_shipped_cell_out"]["comparisons"]["always_adapt"]["point"] > 0

    current_cluster = generated["tracks"]["cifar10c_tent"]["current_policy_family_sensitivity"]
    assert current_cluster["current_policy_authority"] is True
    assert current_cluster["retrospective"] is True
    assert current_cluster["confirmatory"] is False
    assert current_cluster["pointwise_family_intervals_positive_vs_both"] is True
    assert current_cluster["within_candidate_posthoc_holm_rejects_both"] is True
    assert current_cluster["retrospective_six_contrast_holm_rejects_both"] is False
    expected_holm = {"always_adapt": 0.140625, "always_freeze": 0.09375}
    expected_posthoc = {"always_adapt": 0.046875, "always_freeze": 0.03125}
    for baseline in ("always_adapt", "always_freeze"):
        comparison = current_cluster["comparisons"][baseline]
        assert comparison["ci95_unadjusted_family_bootstrap"][0] > 0
        assert comparison["p_value_holm_within_candidate_posthoc"] == pytest.approx(expected_posthoc[baseline])
        assert comparison["p_value_retrospective_holm_six_prospectively_named_contrasts"] == pytest.approx(expected_holm[baseline])

    uniform_head = next(
        row for row in uniform["wave"] if row["track"] == "Mixed head-to-head (CIFAR-10-C Tent primary)"
    )
    assert uniform_head["policy_synchronized"] is False
    assert uniform_head["numeric_release_eligible"] is False
    assert uniform_head["release_eligible_win"] is False
    assert uniform_head["regret_kga"] is None
    assert uniform["wave_holm"] == []
    assert uniform["migration"]["historical_only"] == [
        "CIFAR-10-C Tent cluster resampling",
        "Mixed head-to-head (CIFAR-10-C Tent primary)",
    ]
    assert "no candidate passes the retrospective Holm gate over the six" in uniform["migration"]["note"]

    current_claim = next(row for row in result_manifest["results"] if row["claim_id"] == "KB-CLAIM-010")
    sensitivity = current_claim["metrics"]["current_policy_family_sensitivity"]
    assert sensitivity["confirmatory"] is False
    assert sensitivity["candidates"]["tent"]["retrospective_six_contrast_holm_rejects_both"] is False

    claim = next(row for row in result_manifest["results"] if row["claim_id"] == "KB-CLAIM-026")
    assert claim["status"] == "diagnostic"
    assert claim["metrics"]["policy_synchronized"] is False
    assert claim["metrics"]["release_eligible_win"] is False


def test_sync_claim_text_is_derived_from_current_crossfit_authorities() -> None:
    cluster = load("current_policy_cluster_inference.json")
    assert SYNC._tent_holm_values(cluster) == (0.140625, 0.09375)
    source = (ROOT / "scripts/sync_reconciled_panels.py").read_text()
    for stale in (
        "one false adaptation in 135 cells",
        "ImageNet-C SAR exact-LOO panel",
        "Camelyon17 B-v2 SAR has lower point regret than both fixed policies",
        "0.09375 for both Tent contrasts",
        "within-candidate two-contrast Holm p-values are 0.03125",
    ):
        assert stale not in source
    assert "three-way cell-outcome-disjoint cross-fit" in source


def test_current_crossfit_wording_reaches_release_and_dashboard_surfaces() -> None:
    kbound = ROOT / "docs/research/kbound"
    result_manifest = json.loads((kbound / "RESULT_MANIFEST.json").read_text())
    policy_binding = result_manifest["reconciliation_source"]["current_policy_family_sensitivity"]
    assert policy_binding["status"] == "retrospective_current_policy_family_sensitivity"

    interpretation = next(
        row for row in result_manifest["results"] if row["claim_id"] == "KB-CLAIM-010"
    )["metrics"]["current_policy_family_sensitivity"]["release_interpretation"]
    assert "0.140625 against always-adapt" in interpretation
    assert "0.09375 against always-freeze" in interpretation
    assert "against both baselines" not in interpretation

    dashboard = json.loads((kbound / "dashboard/data/snapshot.json").read_text())
    serialized_dashboard = json.dumps(dashboard)
    assert "0 false adaptations in 135 cells" in serialized_dashboard
    assert "both descriptive run-seed gap intervals have positive lower bounds" in serialized_dashboard
    for stale in (
        "both 0.09375",
        "one false adaptation in 135 cells",
        "freeze-side seed interval touches zero",
    ):
        assert stale not in serialized_dashboard

    claim_manifest = (kbound / "KBOUND_SHORT_CLAIM_MANIFEST.md").read_text()
    assert "cell-outcome-disjoint three-way cross-fit" in claim_manifest
    assert "0.140625" in claim_manifest and "0.09375" in claim_manifest
    assert "exact LOO replay" not in claim_manifest

    producer_text = "\n".join(
        (kbound / relative).read_text()
        for relative in (
            "scripts/build_result_manifest.py",
            "scripts/build_dashboard_snapshot.py",
        )
    )
    for stale in (
        "0.09375 against both baselines",
        "prospectively named contrasts are both 0.09375",
        "one false adaptation in 135 cells",
        "freeze-side seed interval touches zero",
    ):
        assert stale not in producer_text


def test_current_controlled_grid_methods_and_imagenetc_reason_are_synchronized() -> None:
    canonical = load("canonical_panel_results.json")["panels"]
    table = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text()
    )
    ledger = json.loads((ROOT / "docs/research/kbound/claim_ledger.json").read_text())
    decision_metrics = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json").read_text()
    )
    claims = {row["claim_id"]: row for row in ledger["claims"]}

    sar = canonical["imagenetc"]["panel"]["candidates"]["sar"]
    inference = sar["seed_inference"]
    assert inference["descriptive_seed_bootstrap"]["both_lower_bounds_positive"] is True
    assert all(
        inference["descriptive_seed_bootstrap"]["gaps"][baseline]["ci95"][0] > 0
        for baseline in ("always_adapt", "always_freeze")
    )
    assert inference["reason"] in table["tracks"]["imagenetc_sar"]["verdict"]
    assert "touches zero" not in table["tracks"]["imagenetc_sar"]["verdict"]
    assert inference["reason"] in claims["KB-CLAIM-011"]["claim_text"]

    for track in ("cifar10c_tent", "imagenetc_sar", "imagenet_r_D", "camelyon17_b_v2_sar"):
        method = table["tracks"][track]["quantile_rule"]
        assert "three-way cell-outcome-disjoint" in method
        assert "loo" not in method.lower() and "leave-one" not in method.lower()

    validator = (
        ROOT / "docs/research/kbound/scripts/validate_canonical_release_data.py"
    ).read_text()
    assert 'camelyon_b_v2.get("point_beats_both") is not False' in validator
    assert 'cam_metrics.get("point_beats_both") is not False' in validator
    for claim_id in ("KB-CLAIM-010", "KB-CLAIM-011", "KB-CLAIM-042", "KB-CLAIM-053"):
        method = claims[claim_id]["calibration_method"]
        assert "three-way cell-outcome-disjoint" in method
        assert "loo" not in method.lower() and "leave-one" not in method.lower()
    controlled_metric_rows = [
        row for row in decision_metrics["tracks"]
        if row.get("current_policy_authority") is True
        and row.get("track", "").startswith(("CIFAR-10-C", "ImageNet-C", "ImageNet-R", "Camelyon17 B"))
    ]
    assert controlled_metric_rows
    for row in controlled_metric_rows:
        method = row["radius_rule"]
        assert "three-way cell-outcome-disjoint" in method
        assert "leave-one-out-of-pool" not in method

def test_natural_diagnostic_inventory_preserves_evidence_boundaries() -> None:
    canonical = load("canonical_panel_results.json")
    panels = canonical["panels"]
    table = json.loads((ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text())
    ledger = json.loads((ROOT / "docs/research/kbound/claim_ledger.json").read_text())
    result_manifest = json.loads((ROOT / "docs/research/kbound/RESULT_MANIFEST.json").read_text())
    claims = {row["claim_id"]: row for row in ledger["claims"]}
    results = {row["claim_id"]: row for row in result_manifest["results"]}

    cam_score = panels["camelyon17"]["b_v2_diagnostic"]["panel"]["candidates"]["sar"]
    cam_track = table["tracks"]["camelyon17_b_v2_sar"]
    assert cam_track["n_test"] == cam_score["n"] == 108
    assert cam_track["regret"] == [
        cam_score["regret"]["kga"],
        cam_score["regret"]["always_adapt"],
        cam_score["regret"]["always_freeze"],
    ]
    assert cam_track["point_beats_both"] is False
    assert cam_track["ci_robust_beats_both"] is False
    assert cam_track["headline_promotion_eligible"] is False
    assert cam_track["untouched_target_domain_evaluation"] is False
    assert cam_track["independent_checkpoint_identities_recorded"] is False
    cam_metrics = results["KB-CLAIM-053"]["metrics"]
    assert cam_metrics["within_seed_diagnostic"] is True
    assert cam_metrics["point_beats_both"] is False
    assert cam_metrics["ci_robust_beats_both"] is False
    assert cam_metrics["headline_promotion_eligible"] is False

    office_replication_score = panels["officehome"]["test_stream_seed_replication"]["exact_rank_transfer_score"]
    office_replication = results["KB-CLAIM-020"]["metrics"]["test_stream_seed_replication"]
    assert office_replication["n_decisions"] == office_replication_score["n"] == 54
    assert office_replication["decision_counts"] == {
        "ADAPT": 1,
        "FREEZE": 14,
        "ABSTAIN": 39,
    }
    assert office_replication["point_beats_both"] is True
    assert office_replication["ci_robust_beats_both"] is False
    assert office_replication["a7_status"] == "not_established"
    assert office_replication["headline_promotion_eligible"] is False

    fmow_rel = "experiments/kbound/results/fmow_protocol_L_v1/VERIFIED_FINDINGS.json"
    poverty_rel = "experiments/kbound/results/poverty_protocol_L_dev/VERIFIED_FINDINGS.json"
    for claim_id, artifact in (
        (
            "KB-CLAIM-053",
            "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
        ),
        ("KB-CLAIM-054", fmow_rel),
        ("KB-CLAIM-055", poverty_rel),
    ):
        assert claims[claim_id]["status"] == "diagnostic"
        assert artifact in claims[claim_id]["supporting_artifacts"]

    historical = ledger["reconciliation_source"]["separate_historical_diagnostic_authorities"]
    for key, artifact in (
        ("fmow_protocol_l", fmow_rel),
        ("poverty_protocol_l_development", poverty_rel),
    ):
        path = ROOT / artifact
        assert historical[key]["artifact"] == artifact
        assert historical[key]["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert historical[key]["artifact_bytes"] == path.stat().st_size
        assert historical[key]["canonical_panel_member"] is False
        assert historical[key]["headline_promotion_eligible"] is False

    fmow_metrics = results["KB-CLAIM-054"]["metrics"]
    assert fmow_metrics["n_decisions"] == 180
    assert fmow_metrics["false_adapt_conditional_rate"] == pytest.approx(0.375)
    assert fmow_metrics["false_adapt_unconditional_rate"] is None
    assert fmow_metrics["point_beats_both"] is False
    assert fmow_metrics["canonical_panel_member"] is False

    poverty_metrics = results["KB-CLAIM-055"]["metrics"]
    assert poverty_metrics["development_screen"] == "STOP"
    assert poverty_metrics["held_out_evaluation_run"] is False
    assert poverty_metrics["target_score"] is None
    assert poverty_metrics["point_beats_both"] is None
    assert poverty_metrics["canonical_panel_member"] is False


def test_phase1_release_keeps_long_manuscript_synchronized() -> None:
    kbound = ROOT / "docs/research/kbound"
    active = "\n".join((kbound / name).read_text() for name in ("kbound_submission.tex", "kbound_submission_body.tex"))
    normalized_active = " ".join(active.split())
    assert "0.001585" not in active
    assert "Verdict: WIN" not in active
    assert "cluster-robust for Tent" not in active
    assert "current-policy cluster inference is pending" not in active
    assert "retrospective" in active
    assert "six prospectively named candidate-by-baseline contrasts" in normalized_active

    tmlr = (kbound / "kbound_tmlr.tex").read_text()
    assert r"\input{kbound_submission_body}" in tmlr
    assert "kbound_short_body" not in tmlr
    assert "kbound_short_appendix" not in tmlr
    assert "SUPERSEDED HISTORICAL" not in tmlr

    build = (kbound / "scripts/build_pdfs.sh").read_text()
    assert 'BUILD_LONG_TMLR="${BUILD_LONG_TMLR:-${BUILD_HISTORICAL_TMLR:-0}}"' in build

    result_audit = (kbound / "KBOUND_SHORT_RESULT_AUDIT.md").read_text()
    normalized_result_audit = " ".join(result_audit.split())
    assert (
        "adjustment over the six prospectively named contrasts gives 0.140625 against "
        "always-adapt and 0.09375 against always-freeze"
    ) in normalized_result_audit
    assert "earlier KGA policy" in result_audit
    assert "confidence intervals are unadjusted" in result_audit

    claim_manifest = (kbound / "KBOUND_SHORT_CLAIM_MANIFEST.md").read_text()
    assert "retrospective Holm over the six prospectively named contrasts" in claim_manifest
    assert "Holm applies only to archived p-values" in claim_manifest


def test_generated_ci_direction_is_unambiguous() -> None:
    convention = "baseline_regret_minus_kga_regret; positive values favor KGA"
    paths = (
        ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json",
        ROOT / "docs/research/kbound/paper/generated/uniform_verdicts.json",
        ROOT / "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json",
        ROOT / "docs/research/kbound/RESULT_MANIFEST.json",
    )
    stale_fragments = ("ci_vs_", "gap_kga_minus", "gap_vs_adapt_ci", "gap_vs_freeze_ci")
    for path in paths:
        document = json.loads(path.read_text())
        serialized = json.dumps(document)
        assert all(fragment not in serialized for fragment in stale_fragments)
        for node in _walk_json(document):
            if isinstance(node, dict) and "comparisons" in node and "convention" in node:
                assert node["convention"] == convention


def _walk_json(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)
