"""Regression checks for the source-backed natural/corruption panel replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from kga.policy import decide_kga

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "experiments/kbound/results/reconciled_panels_v1"


def load(name: str) -> dict:
    return json.loads((PANEL_ROOT / name).read_text())


def test_compact_sources_are_complete_and_hash_locked() -> None:
    manifest = load("source_manifest.json")
    generator_hash = hashlib.sha256((ROOT / "scripts/reconcile_result_panels.py").read_bytes()).hexdigest()
    assert manifest["file_count"] == 105
    assert len(manifest["files"]) == 105
    assert manifest["generator_sha256"] == generator_hash
    assert load("canonical_panel_results.json")["generator_sha256"] == generator_hash
    assert "/Users/" not in json.dumps(manifest)
    for row in manifest["files"]:
        path = ROOT / row["destination"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["compact_sha256"]
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/Volumes/T9" not in text


def test_reconciled_conflicts_and_negative_panels_are_locked() -> None:
    panels = load("canonical_panel_results.json")["panels"]

    office = panels["officehome"]["primary"]["exact_rank_transfer_score"]
    assert np.isclose(office["regret"]["kga"], 0.01582417582417583)
    assert office["regret"]["kga"] == office["regret"]["always_freeze"]
    assert office["adapt_count"] == 0
    assert not office["point_beats_both"]
    assert not office["seed_inference"]["ci_robust_beats_both"]

    replication = panels["officehome"]["independent_seed_replication"]
    assert np.isclose(
        replication["exact_rank_transfer_score"]["regret"]["kga"],
        0.02150997150997151,
    )
    assert replication["calibration"]["a7_status"] == "not_established"

    iwildcam = panels["iwildcam"]["primary"]["exact_rank_transfer_score"]
    assert iwildcam["adapt_count"] == 0
    assert iwildcam["regret"]["kga"] == iwildcam["regret"]["always_freeze"]
    assert iwildcam["freeze_count"] == 21

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
            assert row["compact_path"].startswith(
                "experiments/kbound/results/reconciled_panels_v1/source/"
            )
            assert len(row["compact_sha256"]) == 64
            assert row["archive_relative_path"].startswith("experiments/kbound/results/")
            assert len(row["original_sha256"]) == 64
            assert row["original_bytes"] > 0
            assert "/Users/" not in json.dumps(row)
            assert "/Volumes/" not in json.dumps(row)
