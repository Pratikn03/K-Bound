"""Regression checks for the source-backed natural/corruption panel replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from kga.policy import decide_kga

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "experiments/kbound/results/reconciled_panels_v1"


def load(name: str) -> dict:
    return json.loads((PANEL_ROOT / name).read_text())


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
    assert np.isclose(sar["regret"]["kga"], 0.028892592368302522)
    assert sar["false_adapt_count"] == 1
    assert sar["point_beats_both"]
    assert not sar["seed_inference"]["ci_robust_beats_both"]

    pacs = panels["pacs"]
    assert pacs["aggregate_matches_seed_files"]
    assert not pacs["decision_replay_available"]

    imagenet_r = panels["imagenet_r"]["panel"]["architecture_panel_aggregate"]
    assert imagenet_r["n"] == 480
    assert np.isclose(imagenet_r["regret"]["kga"], 0.014968749999999998)
    assert imagenet_r["regret"]["kga"] > imagenet_r["regret"]["always_adapt"]
    assert not imagenet_r["point_beats_both"]
    kappa_one = next(row for row in imagenet_r["kappa_sweep"] if row["kappa"] == 1.0)
    assert np.isclose(kappa_one["regret"], imagenet_r["regret"]["kga"])
    assert np.isclose(kappa_one["yield"], imagenet_r["decision_coverage"])
    worse = sum(
        row["regret"]["kga"] > row["regret"]["always_adapt"]
        for row in panels["imagenet_r"]["panel"]["candidates"].values()
    )
    assert worse == 8

    imagenetc = panels["imagenetc"]["panel"]["architecture_panel_aggregate"]
    assert np.isclose(imagenetc["radius_diagnostics"]["yield"], imagenetc["decision_coverage"])
    assert np.isclose(imagenetc["radius_diagnostics"]["eps_mean"], 0.05320361619896232)


def test_imagenetc_source_replays_through_canonical_rule() -> None:
    source = load("source/imagenetc/per_condition_imagenetc_sar_seed0.json")
    records = source["records"]
    prediction = np.asarray([row["b_hat"] for row in records], dtype=float)
    benefit = np.asarray([row["B"] for row in records], dtype=float)
    epsilon, decision = decide_kga(prediction, benefit, alpha=0.1, calibration="loo")

    generated = load("canonical_panel_results.json")["panels"]["imagenetc"]["panel"]
    seed0 = generated["candidates"]["sar"]["per_file"][0]
    assert np.isclose(seed0["epsilon_min"], epsilon.min())
    assert np.isclose(seed0["epsilon_mean"], epsilon.mean())
    assert np.isclose(seed0["epsilon_max"], epsilon.max())
    assert seed0["score"]["adapt_count"] == int(np.sum(decision == "ADAPT"))


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
    assert b_v2["panel"]["candidates"]["sar"]["point_beats_both"]
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

    generated = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text()
    )
    uniform = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/uniform_verdicts.json").read_text()
    )
    metrics = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json").read_text()
    )

    accounting = {row["track"]: row for row in generated["decision_accounting_summary"]["rows"]}
    uniform_rows = {row["track"]: row for row in uniform["wave"]}
    metric_rows = {row["track"]: row for row in metrics["tracks"]}
    for candidate, title in (("tent", "Tent"), ("eata", "EATA")):
        assert generated["tracks"][f"cifar10c_{candidate}"]["decision_counts"] == expected[candidate]
        assert {
            action: accounting[f"CIFAR-10-C {title}"][action]
            for action in ("ADAPT", "FREEZE", "ABSTAIN")
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
    assert all(
        iwild_uniform[field] is None
        for field in ("regret_kga", "regret_adapt", "regret_freeze", "FA_u")
    )
    iwild_metrics = metric_rows["iWildCam"]
    assert iwild_metrics["numeric_release_eligible"] is False
    assert all(
        iwild_metrics["actions"][action]["count"] is None
        and iwild_metrics["actions"][action]["rate"] is None
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
    generated = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text()
    )
    uniform = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/uniform_verdicts.json").read_text()
    )
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
    generated = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json").read_text()
    )
    uniform = json.loads(
        (ROOT / "docs/research/kbound/paper/generated/uniform_verdicts.json").read_text()
    )
    result_manifest = json.loads((ROOT / "docs/research/kbound/RESULT_MANIFEST.json").read_text())

    head = generated["headtohead"]
    assert head["policy_synchronized"] is False
    assert head["current_policy_authority"] is False
    assert head["numeric_release_eligible"] is False
    assert head["release_eligible_win"] is False
    assert head["current_exact_rank_reference"]["kga_regret"] == pytest.approx(
        0.0016453700209105456
    )
    assert "0.0015851849" not in json.dumps(generated)

    cluster = generated["tracks"]["cifar10c_tent"]["historical_cluster_resampling"]
    assert cluster["policy_synchronized"] is False
    assert cluster["current_policy_authority"] is False
    assert cluster["release_eligible"] is False
    assert cluster["convention"] == generated["ci_convention"]
    assert cluster["as_shipped_cell_out"]["comparisons"]["always_adapt"]["point"] > 0

    current_cluster = generated["tracks"]["cifar10c_tent"][
        "current_policy_family_sensitivity"
    ]
    assert current_cluster["current_policy_authority"] is True
    assert current_cluster["retrospective"] is True
    assert current_cluster["confirmatory"] is False
    assert current_cluster["pointwise_family_intervals_positive_vs_both"] is True
    assert current_cluster["within_candidate_posthoc_holm_rejects_both"] is True
    assert current_cluster["retrospective_six_contrast_holm_rejects_both"] is False
    for baseline in ("always_adapt", "always_freeze"):
        comparison = current_cluster["comparisons"][baseline]
        assert comparison["ci95_unadjusted_family_bootstrap"][0] > 0
        assert comparison["p_value_holm_within_candidate_posthoc"] == pytest.approx(0.03125)
        assert comparison[
            "p_value_retrospective_holm_six_prospectively_named_contrasts"
        ] == pytest.approx(
            0.09375
        )

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
    assert "no candidate passes the retrospective Holm gate over the six" in uniform[
        "migration"
    ]["note"]

    current_claim = next(
        row for row in result_manifest["results"] if row["claim_id"] == "KB-CLAIM-010"
    )
    sensitivity = current_claim["metrics"]["current_policy_family_sensitivity"]
    assert sensitivity["confirmatory"] is False
    assert sensitivity["candidates"]["tent"][
        "retrospective_six_contrast_holm_rejects_both"
    ] is False

    claim = next(row for row in result_manifest["results"] if row["claim_id"] == "KB-CLAIM-026")
    assert claim["status"] == "diagnostic"
    assert claim["metrics"]["policy_synchronized"] is False
    assert claim["metrics"]["release_eligible_win"] is False


def test_phase1_release_keeps_long_manuscript_synchronized() -> None:
    kbound = ROOT / "docs/research/kbound"
    active = "\n".join(
        (kbound / name).read_text()
        for name in ("kbound_submission.tex", "kbound_submission_body.tex")
    )
    assert "0.001585" not in active
    assert "Verdict: WIN" not in active
    assert "cluster-robust for Tent" not in active
    assert "current-policy cluster inference is pending" not in active
    assert "retrospective" in active
    assert "six prospectively named contrasts" in active

    tmlr = (kbound / "kbound_tmlr.tex").read_text()
    assert r"\input{kbound_submission_body}" in tmlr
    assert "kbound_short_body" not in tmlr
    assert "kbound_short_appendix" not in tmlr
    assert "SUPERSEDED HISTORICAL" not in tmlr

    build = (kbound / "scripts/build_pdfs.sh").read_text()
    assert 'BUILD_LONG_TMLR="${BUILD_LONG_TMLR:-${BUILD_HISTORICAL_TMLR:-0}}"' in build

    result_audit = (kbound / "KBOUND_SHORT_RESULT_AUDIT.md").read_text()
    assert "adjustment over the six prospectively named contrasts gives 0.09375" in result_audit
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
