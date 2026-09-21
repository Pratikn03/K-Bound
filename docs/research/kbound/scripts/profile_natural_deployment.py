#!/usr/bin/env python3
"""Stage costs on one already-opened development city; no outcome evaluation.

This measurement does not authorize natural target access or a serving claim.
It runs all five checkpoints required by the city-mean v2 controller, resetting
each candidate from its verified source checkpoint on every repetition.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import resource
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_paths = [Path(__file__)] + [
        ROOT / "experiments/kbound/so2sat" / name
        for name in ("development.py", "adapters.py", "features.py", "prospective_v2.py", "source_data.py", "model.py")
    ]
    snapshots = args.output_dir / "execution_sources"
    snapshots.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshots / path.name)
    protocol = {
        "schema": "kbound-opened-development-cost-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city": "cairo",
        "role": "gate_fit",
        "repetitions": 3,
        "checkpoint_ids": list("01234"),
        "device": "mps",
        "threads": 4,
        "batch_size": 128,
        "warmup_policy": "no explicit warmup; cache state uncontrolled because setup already reads the source container and checkpoints",
        "statistical_scope": "timing only on previously opened development data; no target outcomes or new efficacy result",
        "labels_read": 0,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    import torch

    from experiments.kbound.so2sat import adapters, development, features, prospective_v2
    from experiments.kbound.so2sat.integrity import load_verified_json_mapping_with_receipt
    from experiments.kbound.so2sat.source_data import load_sealed_band_normalizer

    torch.set_num_threads(4)
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError("fixed profiling device MPS unavailable")
    manifest, _ = load_verified_json_mapping_with_receipt(args.manifest)
    binding, loaded_manifest, inventory = development._load_manifest_and_inventory(
        args.manifest, args.data_dir / "training_geo.h5"
    )
    if manifest != loaded_manifest:
        raise RuntimeError("manifest changed during inventory construction")
    _, checkpoints = development.load_verified_checkpoints(args.checkpoint_dir)
    normalizer = load_sealed_band_normalizer(args.checkpoint_dir / "so2sat_sen2_source_normalizer.json")
    data = development.DevelopmentData(args.data_dir / "training.h5", inventory, normalizer, authorized_role="gate_fit")
    if "cairo" not in binding["gate_fit_cities"]:
        raise RuntimeError("Cairo is not in the already-opened development partition")
    partition = inventory.partitions["gate_fit"]["cairo"]
    controller = prospective_v2.load_controller_v2()
    setup_seconds = time.perf_counter() - started
    rows, repetitions = [], []
    max_driver = 0

    def measured(stages, name, function):
        nonlocal max_driver
        torch.mps.synchronize()
        stamp = time.perf_counter()
        result = function()
        torch.mps.synchronize()
        stages[name] = time.perf_counter() - stamp
        max_driver = max(max_driver, torch.mps.driver_allocated_memory())
        return result

    for repetition in range(3):
        repetition_started = time.perf_counter()
        city_features = {}
        for checkpoint in checkpoints:
            stages = {}
            source = measured(
                stages, "checkpoint_reset", lambda checkpoint=checkpoint: checkpoint.fresh_model(device=device)
            )
            frozen_probe, divergence = measured(
                stages,
                "source_probe_inference_and_bn_statistics",
                lambda source=source: adapters.frozen_logits_and_bn_divergence(
                    source, data.pixel_batches(partition, half="probe"), device=device
                ),
            )
            candidate, diagnostic = measured(
                stages,
                "tent_probe_adaptation",
                lambda source=source, divergence=divergence: adapters.adapt_on_probe(
                    source,
                    data.pixel_batches(partition, half="probe"),
                    candidate_id=adapters.TENT_CANDIDATE_ID,
                    device=device,
                    batchnorm_source_statistic_divergence=divergence,
                ),
            )
            adapted_probe = measured(
                stages,
                "candidate_probe_inference",
                lambda candidate=candidate: adapters.fixed_model_logits(
                    candidate, data.pixel_batches(partition, half="probe"), device=device
                ),
            )
            feature = measured(
                stages,
                "label_free_feature_extraction",
                lambda frozen_probe=frozen_probe, adapted_probe=adapted_probe, diagnostic=diagnostic: (
                    features.extract_label_free_features(
                        frozen_probe.numpy(),
                        adapted_probe.numpy(),
                        normalized_adapter_update_norm=diagnostic.normalized_adapter_update_norm,
                        batchnorm_source_statistic_divergence=diagnostic.batchnorm_source_statistic_divergence,
                    )
                ),
            )
            city_features[checkpoint.checkpoint_id] = {
                name: feature["features"][name] for name in prospective_v2.FEATURE_NAMES
            }
            # Both inference alternatives are timed independently of gate decisions.
            # Only the selected alternative is normally needed after deployment.
            source_outputs = measured(
                stages,
                "source_evaluation_pixel_inference",
                lambda source=source: adapters.fixed_model_logits(
                    source, data.pixel_batches(partition, half="evaluation"), device=device
                ),
            )
            candidate_outputs = measured(
                stages,
                "candidate_evaluation_pixel_inference",
                lambda candidate=candidate: adapters.fixed_model_logits(
                    candidate, data.pixel_batches(partition, half="evaluation"), device=device
                ),
            )
            row = {
                "repetition": repetition,
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_sha256": checkpoint.checkpoint_file_sha256,
                "stage_seconds": stages,
                "feature_sha256": feature["feature_sha256"],
                "probe_n": len(partition.probe_rows),
                "evaluation_n": len(partition.evaluation_rows),
            }
            rows.append(row)
            with (args.output_dir / "measurements.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            del source, candidate, frozen_probe, adapted_probe, source_outputs, candidate_outputs
            gc.collect()
            torch.mps.empty_cache()
        gate_stages = {}
        predictions = measured(
            gate_stages,
            "five_checkpoint_city_mean_prediction",
            lambda city_features=city_features: [
                prospective_v2.raw_prediction_v2(controller, checkpoint_id=key, city_probe_features=city_features)
                for key in "01234"
            ],
        )
        repetitions.append(
            {
                "repetition": repetition,
                "seconds_including_cleanup_and_both_inference_alternatives": time.perf_counter() - repetition_started,
                "prediction_seconds": gate_stages["five_checkpoint_city_mean_prediction"],
                "predictions": predictions,
            }
        )

    names = rows[0]["stage_seconds"]
    stages_summary = {
        name: {
            "n": len(rows),
            "median_seconds": statistics.median(row["stage_seconds"][name] for row in rows),
            "max_seconds": max(row["stage_seconds"][name] for row in rows),
            "total_seconds": sum(row["stage_seconds"][name] for row in rows),
        }
        for name in names
    }
    report = {
        "status": "PASS",
        "protocol": protocol,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "setup_seconds_including_full_source_identity_checks": setup_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "stage_summary": stages_summary,
        "repetitions": repetitions,
        "cpu_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "maximum_sampled_post_stage_mps_driver_bytes": max_driver,
        "controller_sha256": controller["controller_sha256"],
        "partition_sha256": partition.partition_sha256,
        "labels_read": 0,
        "target_pixels_read": 0,
        "limitations": [
            "one previously opened city, five related checkpoints and three repeated measurements",
            "MPS sampled after stages is not a continuously measured peak",
            "HDF5 reads, normalization and GPU transfers are included in their respective stages",
            "gate profile is a point computation; calibration validity and target authorization are evaluated separately",
            "both serving alternatives are measured, not a deployed ensemble",
            "no sustained load, energy meter, external field deployment or production SLA",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output_dir),
                "setup_seconds": setup_seconds,
                "total_seconds": report["total_wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
