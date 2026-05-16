from __future__ import annotations

import pandas as pd

from uais.validation.healthcare_gap_closure import (
    assign_patient_stratified_splits,
    build_clinical_fusion_frame,
    clinical_domain_attribution_report,
    clinical_incident_detection_report,
    diagnostic_switching_certificate_from_stress,
    fusion_schema_integration_report,
    gap2_reliability_stress_report,
    gap4_deployment_audit_report,
    gap_statuses,
    label_diversity_report,
    patient_overlap_report,
)


def _toy_split_frame() -> pd.DataFrame:
    rows = [
        {
            "patient_id": "p1",
            "timestamp": "2026-01-01T00:00:00Z",
            "source_split": "train",
            "source_dataset": "site_a",
            "hr_bpm": 70.0,
            "spo2_pct": 98.0,
            "respiratory_rate": 16.0,
            "shock_index": 0.6,
            "reliability": 0.9,
            "is_critical": False,
        },
        {
            "patient_id": "p2",
            "timestamp": "2026-01-02T00:00:00Z",
            "source_split": "val",
            "source_dataset": "site_b",
            "hr_bpm": 132.0,
            "spo2_pct": 88.0,
            "respiratory_rate": 31.0,
            "shock_index": 1.2,
            "reliability": 0.7,
            "is_critical": True,
        },
        {
            "patient_id": "p3",
            "timestamp": "2026-01-03T00:00:00Z",
            "source_split": "test",
            "source_dataset": "site_b",
            "hr_bpm": 136.0,
            "spo2_pct": 87.0,
            "respiratory_rate": 34.0,
            "shock_index": 1.3,
            "reliability": 0.8,
            "is_critical": True,
        },
    ]
    return pd.DataFrame(rows)


def _toy_stratified_frame() -> pd.DataFrame:
    rows = []
    patients = [
        ("n1", 0, 70.0, 98.0, 16.0, 0.6),
        ("n2", 0, 72.0, 97.0, 15.0, 0.6),
        ("n3", 0, 74.0, 98.0, 17.0, 0.6),
        ("p1", 1, 135.0, 88.0, 32.0, 1.3),
        ("p2", 1, 138.0, 87.0, 34.0, 1.4),
        ("p3", 1, 132.0, 89.0, 31.0, 1.2),
    ]
    for idx, (patient_id, label, hr, spo2, resp, shock) in enumerate(patients):
        for step in range(2):
            rows.append(
                {
                    "patient_id": patient_id,
                    "timestamp": pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=idx, hours=step),
                    "source_dataset": "toy",
                    "hr_bpm": hr,
                    "spo2_pct": spo2,
                    "respiratory_rate": resp,
                    "shock_index": shock,
                    "reliability": 0.9,
                    "is_critical": bool(label),
                }
            )
    return pd.DataFrame(rows)


def test_build_clinical_fusion_frame_hashes_incidents_and_emits_four_domains():
    fusion = build_clinical_fusion_frame(_toy_split_frame())

    assert sorted(fusion["domain"].unique()) == [
        "heart_rate",
        "oxygenation",
        "respiration",
        "shock_index",
    ]
    assert fusion.groupby("incident_id")["domain"].nunique().tolist() == [4, 4, 4]
    assert set(fusion["fusion_split"]) == {"train", "validation", "test"}
    assert "p1" not in set(fusion["incident_id"])
    assert "patient_id" not in fusion.columns
    assert fusion["score"].between(0.0, 1.0).all()
    assert fusion["confidence"].between(0.0, 1.0).all()


def test_label_diversity_report_blocks_single_class_empirical_claims():
    fusion = build_clinical_fusion_frame(_toy_split_frame())

    report = label_diversity_report(fusion)

    assert report["train"]["label_classes"] == 1
    assert report["validation"]["label_classes"] == 1
    assert report["test"]["label_classes"] == 1
    assert report["all_required_splits_have_two_classes"] is False


def test_gap_statuses_keep_structural_and_empirical_closure_separate():
    fusion = build_clinical_fusion_frame(_toy_split_frame())
    label_report = label_diversity_report(fusion)
    overlap_report = patient_overlap_report(fusion)

    statuses = gap_statuses(
        incident_protocol_passed=True,
        label_report=label_report,
        patient_overlap=overlap_report,
        schema_passed=True,
        category_aware_report={"category_aware_gate_rate": 0.1, "global_gate_rate": 0.4},
        switching_report={"certificate": {"certified": True}},
        calibration_report={"alert": False},
    )

    assert statuses["1_true_multimodal_incident_detection"]["engineering_closed"] is True
    assert statuses["1_true_multimodal_incident_detection"]["empirical_closed"] is False
    assert statuses["2_autonomous_zero_misfire_adaptation"]["engineering_closed"] is True
    assert statuses["3_universal_system_integration"]["engineering_closed"] is True
    assert statuses["4_deployable_auditable_ai"]["engineering_closed"] is True
    assert statuses["4_deployable_auditable_ai"]["empirical_closed"] is False


def test_patient_stratified_split_closes_gap1_local_replay_conditions():
    split_frame = assign_patient_stratified_splits(_toy_stratified_frame(), seed=7)
    fusion = build_clinical_fusion_frame(split_frame)

    label_report = label_diversity_report(fusion)
    overlap_report = patient_overlap_report(fusion)
    detection_report = clinical_incident_detection_report(fusion)
    statuses = gap_statuses(
        incident_protocol_passed=True,
        label_report=label_report,
        patient_overlap=overlap_report,
        schema_passed=True,
        category_aware_report={"category_aware_gate_rate": 0.1, "global_gate_rate": 0.4},
        switching_report={"certificate": {"certified": False}},
        calibration_report={"alert": False},
        incident_detection_report=detection_report,
    )

    assert label_report["all_required_splits_have_two_classes"] is True
    assert overlap_report["any_overlap"] is False
    assert detection_report["closed"] is True
    assert statuses["1_true_multimodal_incident_detection"]["empirical_closed"] is True


def test_gap2_stress_report_closes_when_natural_is_quiet_and_collapse_fires():
    split_frame = assign_patient_stratified_splits(_toy_stratified_frame(), seed=7)
    fusion = build_clinical_fusion_frame(split_frame)

    stress_report = gap2_reliability_stress_report(
        fusion,
        natural_safety_margin=0.01,
        max_natural_fire_rate=0.05,
        min_collapse_fire_rate=0.5,
    )
    statuses = gap_statuses(
        incident_protocol_passed=True,
        label_report=label_diversity_report(fusion),
        patient_overlap=patient_overlap_report(fusion),
        schema_passed=True,
        category_aware_report={"category_aware_gate_rate": 1.0, "global_gate_rate": 1.0},
        switching_report={"certificate": {"certified": False}},
        calibration_report={"alert": False},
        incident_detection_report=clinical_incident_detection_report(fusion),
        reliability_stress_report=stress_report,
    )

    assert stress_report["closed"] is True
    assert stress_report["test"]["natural_fire_rate"] <= 0.05
    assert stress_report["test"]["collapse_fire_rate_mean"] >= 0.5
    assert statuses["2_autonomous_zero_misfire_adaptation"]["empirical_closed"] is True


def test_gap3_schema_integration_report_requires_hashes_and_tensor_readiness():
    fusion = build_clinical_fusion_frame(_toy_split_frame())
    schema_stats = {
        "score_out_of_range": 0,
        "confidence_out_of_range": 0,
    }
    incident_protocol = {
        "split_leakage_count": 0,
        "min_domains_per_incident": 4,
    }

    report = fusion_schema_integration_report(fusion, schema_stats, incident_protocol)
    statuses = gap_statuses(
        incident_protocol_passed=True,
        label_report=label_diversity_report(fusion),
        patient_overlap=patient_overlap_report(fusion),
        schema_passed=True,
        category_aware_report={"category_aware_gate_rate": 0.1, "global_gate_rate": 0.4},
        switching_report={"certificate": {"certified": False}},
        calibration_report={"alert": False},
        schema_integration_report=report,
    )

    assert report["closed"] is True
    assert report["raw_patient_id_absent"] is True
    assert report["patient_keys_are_hashed"] is True
    assert report["tensor_ready"] is True
    assert statuses["3_universal_system_integration"]["empirical_closed"] is True


def test_gap4_deployment_audit_closes_on_local_stress_certificate(tmp_path):
    split_frame = assign_patient_stratified_splits(_toy_stratified_frame(), seed=7)
    fusion = build_clinical_fusion_frame(split_frame)
    stress_report = gap2_reliability_stress_report(
        fusion,
        natural_safety_margin=0.01,
        max_natural_fire_rate=0.05,
        min_collapse_fire_rate=0.5,
    )
    data_root = tmp_path / "gridpulse"
    manifest_dir = data_root / "mimic3" / "processed"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "mimic3_manifest.json").write_text(
        '{"license_notes":"Follow PhysioNet credential requirements.","source_urls":["https://physionet.org"]}',
        encoding="utf-8",
    )

    audit = gap4_deployment_audit_report(
        fusion,
        data_root=data_root,
        temporal_reference={"temporal_order_valid": True},
        reliability_stress_report=stress_report,
        switching_report={"evaluated": True, "calibration": {"alert": False, "reasons": []}},
        label_report=label_diversity_report(fusion),
    )
    statuses = gap_statuses(
        incident_protocol_passed=True,
        label_report=label_diversity_report(fusion),
        patient_overlap=patient_overlap_report(fusion),
        schema_passed=True,
        category_aware_report={"category_aware_gate_rate": 1.0, "global_gate_rate": 1.0},
        switching_report={"evaluated": True, "calibration": {"alert": False}},
        calibration_report={"alert": False},
        incident_detection_report=clinical_incident_detection_report(fusion),
        reliability_stress_report=stress_report,
        deployment_audit_report=audit,
    )

    assert clinical_domain_attribution_report(fusion)["evaluated"] is True
    assert diagnostic_switching_certificate_from_stress(stress_report)["certified"] is True
    assert audit["closed"] is True
    assert audit["privacy_license_review"]["passed"] is True
    assert statuses["4_deployable_auditable_ai"]["empirical_closed"] is True
