"""One retrospective fixed-frame audit; replay uses the original saved draws.

No training, new candidate generation, future-environment inference or F1
certificate. The public CCT records were already opened. Randomization is only
over a fresh label-budget simulation on fixed recorded predictions/outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

from kga.finite_population_audit import make_plan, draw_sample, assess_sample

SCORE = "e609483a59e626fdee1c1fa1816bef821fdba3f5d925d2ce145c1fb754b64779"
LABELS = "4676ae9a846cee177d0c47d4c0db9b1874043f96ba5c94e922f6ef32148b3c60"
CONTRACT = "8eb31fe60629e4114e97fa98addb8e701d530757ca0330885ec172e8c0468755"
ROOT = Path(__file__).resolve().parents[4]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, record):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")


def read_object(archive, authority):
    raw = archive.read("objects/" + authority)
    if digest(raw) != authority:
        raise ValueError("Public object hash mismatch")
    return json.loads(raw)


def run(bundle, output, replay=False):
    started = time.monotonic()
    if not replay:
        output.mkdir(parents=True, exist_ok=False)
        protocol = {
            "schema": "kbound-cct20-retrospective-finite-frame-audit-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "retrospective_preexisting_opened_outcomes",
            "cohort": "all 45 published checkpoint-by-location evaluation cells",
            "draws_per_cell": 512, "sampling": "uniform_with_replacement_on_prediction_disagreement",
            "family_alpha_exact": "1/20", "cell_alpha_exact": "1/900",
            "metric": "finite-frame mean adapted-minus-frozen set-membership top1 correctness",
            "success_criterion": "report all intervals, sign decisions and full-frame containment; no superiority endpoint",
            "prospective_deployment_validated": False, "macro_f1_inference": False,
            "reroll_rule": "never; replay saved draw inventories only",
            "source_commit_before_execution": "11a3c6aed1f44f8b8c8b558e151c1b0bbdb6c462",
            "executed_sources": {p: digest((ROOT/p).read_bytes()) for p in (
                "kga/finite_population_audit.py", "kga/transport_numerics.py",
                "docs/research/kbound/scripts/run_cct20_finite_frame_audit.py")},
            "bundle_sha256": digest(bundle.read_bytes()),
            "scoring_object": SCORE, "label_object": LABELS, "label_contract_object": CONTRACT,
        }
        write(output / "protocol.json", protocol)
        (output / "executed_sources").mkdir()
        for name in protocol["executed_sources"]:
            (output / "executed_sources" / Path(name).name).write_bytes((ROOT/name).read_bytes())
    else:
        protocol = json.loads((output / "protocol.json").read_text())
        if digest(bundle.read_bytes()) != protocol["bundle_sha256"]:
            raise ValueError("Replay bundle differs")
        expected = {"schema": "kbound-cct20-retrospective-finite-frame-audit-v1",
                    "status": "retrospective_preexisting_opened_outcomes", "draws_per_cell": 512,
                    "family_alpha_exact": "1/20", "cell_alpha_exact": "1/900",
                    "sampling": "uniform_with_replacement_on_prediction_disagreement",
                    "prospective_deployment_validated": False, "macro_f1_inference": False,
                    "scoring_object": SCORE, "label_object": LABELS, "label_contract_object": CONTRACT}
        if any(protocol.get(k) != v for k, v in expected.items()):
            raise ValueError("Replay protocol semantics differ")
        for name, expected_hash in protocol["executed_sources"].items():
            if digest((ROOT/name).read_bytes()) != expected_hash:
                raise ValueError("Replay executable source differs; use the archived source snapshot")
            if digest((output/"executed_sources"/Path(name).name).read_bytes()) != expected_hash:
                raise ValueError("Archived executed source differs")
    with zipfile.ZipFile(bundle) as archive:
        manifest_raw = archive.read("manifest.json")
        manifest = json.loads(manifest_raw)
        entries = [x for x in manifest["objects"] if re.fullmatch(
            r"upstream_artifacts.prediction_cells.items\[\d+\]", x["content_role"]) and x["bytes"] > 100000]
        if len(entries) != 45 or len({x["published_sha256"] for x in entries}) != 45:
            raise ValueError("Expected all 45 distinct published prediction cells")
        frames = []
        for entry in entries:
            cell = read_object(archive, entry["published_sha256"])
            # Probe rows used in original adaptation are excluded by the original
            # evaluation role, never by an observed outcome or new selection rule.
            rows = [r for r in cell["rows"] if r["role"] == "evaluation"]
            if not rows:
                raise ValueError("Empty evaluation frame")
            identity = f"seed{cell['checkpoint_seed']}_location{cell['location_id']}"
            plan = make_plan(ids=[r["image_id"] for r in rows],
                frozen=[r["frozen_prediction"] for r in rows], adapted=[r["adapted_prediction"] for r in rows],
                classes=list(range(16)), sample_size=512, alpha=Fraction(1,900),
                predictor_identity=entry["published_sha256"], stratum="disagreement", label_semantics="set_membership")
            if not replay:
                write(output / (identity + "_plan.json"), plan)
                draws = draw_sample(plan, output / (identity + "_draws.json"))
            else:
                saved = json.loads((output / (identity + "_plan.json")).read_text())
                if saved != plan:
                    raise ValueError("Replay frame or procedure differs from saved plan")
                draws = json.loads((output / (identity + "_draws.json")).read_text())
            frames.append((identity, cell, plan, draws))
        if len({identity for identity, *_ in frames}) != 45:
            raise ValueError("Duplicate checkpoint-location key")
        # All new draw inventories are persisted before the scorer opens labels.
        # These historical labels were already opened in the original study.
        annotations = read_object(archive, LABELS)
        contract = read_object(archive, CONTRACT)
        category_map = contract["frozen_category_mapping"]["category_id_to_output_index"]
        labels = defaultdict(set)
        for row in annotations["annotations"]:
            labels[row["image_id"]].add(category_map[str(row["category_id"])])
        labels = {key: sorted(value) for key, value in labels.items()}
        original = read_object(archive, SCORE)
        authorities = {(c["checkpoint_seed"], c["location_id"]): c for c in original["cells"]}
        results = []
        for identity, cell, plan, draws in sorted(frames):
            assessment = assess_sample(plan, draws, labels)
            truth = Fraction(sum(int(a in labels[i]) - int(f in labels[i]) for i, f, a in
                                 zip(plan["ids"], plan["frozen"], plan["adapted"])), len(plan["ids"]))
            original_cell = authorities[(cell["checkpoint_seed"], cell["location_id"])]
            if len(plan["ids"]) != original_cell["n_evaluation_images"] or abs(float(truth) - original_cell["adaptation_benefit"]) > 1e-12:
                raise ValueError("Original primary metric failed reconstruction")
            lo, hi = map(Fraction, assessment["benefit_interval_exact"])
            row = {"cell": identity, "assessment": assessment, "full_frame_benefit_exact": str(truth),
                   "full_frame_benefit_in_interval": lo <= truth <= hi,
                   "original_decision": original_cell["decision"],
                   "label_access": "retrospective previously opened outcomes; audit label budget is simulated"}
            if replay:
                if json.loads((output / (identity + "_result.json")).read_text()) != row:
                    raise ValueError("Replay result differs")
            else:
                write(output / (identity + "_result.json"), row)
                write(output / (identity + "_sampled_labels.json"), {plan["ids"][i]: labels[plan["ids"][i]] for i in draws["indices"]})
            results.append(row)
    summary = {
        "schema": "kbound-cct20-retrospective-finite-frame-summary-v1",
        "study_status": "retrospective_audit_assisted_fixed_frame_only", "cells": len(results),
        "family_alpha_exact": "1/20", "draws_per_cell": 512,
        "actions": dict(Counter(r["assessment"]["action"] for r in results)),
        "full_frame_containment": sum(r["full_frame_benefit_in_interval"] for r in results),
        "helpful_frames": sum(Fraction(r["full_frame_benefit_exact"]) > 0 for r in results),
        "harmful_frames": sum(Fraction(r["full_frame_benefit_exact"]) < 0 for r in results),
        "accepted_harmful": sum(r["assessment"]["action"] == "ADAPT" and Fraction(r["full_frame_benefit_exact"]) <= 0 for r in results),
        "helpful_selected": sum(r["assessment"]["action"] == "ADAPT" and Fraction(r["full_frame_benefit_exact"]) > 0 for r in results),
        "audit_draw_count": sum(r["assessment"]["draw_count"] for r in results),
        "unique_label_uses_summed_across_cells": sum(r["assessment"]["unique_labels_used"] for r in results),
        "prospective_deployment_validated": False,
        "inference_scope": "simultaneous design-conditional finite-frame intervals; no environment-population or macro-F1 guarantee",
        "bundle_manifest_sha256": digest(manifest_raw),
    }
    if replay:
        if json.loads((output/"summary.json").read_text()) != summary:
            raise ValueError("Replay summary differs")
    else:
        write(output/"summary.json", summary)
    print(json.dumps({"replay": replay, "seconds": time.monotonic()-started, **summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    run(args.bundle, args.output, args.replay)
