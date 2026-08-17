#!/usr/bin/env python3
"""Seal SHA256 hashes for sealable KBound nine-track evidence artifacts.

Usage:
  python docs/research/kbound/scripts/seal_nine_track_lock.py          # write seal
  python docs/research/kbound/scripts/seal_nine_track_lock.py --verify # check seal

Locked here means: frozen aggregate + hashed source artifacts (replayable).
Does NOT change scientific verdicts (beats-both vs no-harm). Does NOT promote
Office-Home LOO beats-both. Completed null diagnostics are sealed as diagnostics;
CIFAR SAR remains withheld.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "experiments/kbound/results/nine_track_lock_v1"
SEAL_JSON = OUT_DIR / "LOCK_SEAL.json"
SEAL_SHA = OUT_DIR / "LOCK_SEAL.sha256"
LOCK_YAML = ROOT / "research_lock/NINE_TRACK_LOCK_SEAL_v1.yaml"

# Canonical artifacts per track. Paths relative to repo root.
TRACKS = {
    "cifar10c_tent_eata": {
        "status": "locked",
        "verdict": "beats-both (CI utility); SAR withheld",
        "artifacts": [
            "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json",
            "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_FINDINGS.md",
        ],
    },
    "imagenetc_sar": {
        "status": "locked",
        "verdict": "beats-both (pooled paired CI)",
        "artifacts": [
            "experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed0.json",
            "experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed1.json",
            "experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed2.json",
            "experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed3.json",
            "experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed4.json",
        ],
    },
    "camelyon17_ood": {
        "status": "locked",
        "verdict": "no-harm (genuine OOD test; ties adapt)",
        "artifacts": [
            "audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json",
            "audits/integrity_2026-06-20/camelyon_reconciliation/VERDICT_phase1.md",
            "research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml",
        ],
    },
    "iwildcam_H_v2": {
        "status": "locked",
        "verdict": "no-harm (OOF lock; ties freeze)",
        "artifacts": [
            "research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json",
            "research_lock/IWILDCAM_PROTOCOL_H_v2.yaml",
            "experiments/kbound/results/iwildcam_protocol_H_v2/protocol_result.json",
        ],
    },
    "officehome_M_v2": {
        "status": "locked",
        "verdict": "no-harm (OOF lock; NOT LOO beats-both)",
        "artifacts": [
            "research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json",
            "experiments/kbound/results/officehome_protocol_M_v2/protocol_result.json",
        ],
        "caveat": "Paper promotes OOF no-harm only; do not promote LOO beats-both.",
    },
    "rxrx1_J": {
        "status": "locked",
        "verdict": "no-harm (ties freeze)",
        "artifacts": [
            "experiments/kbound/results/rxrx1_protocol_J_v1/analyze_F_results.json",
            "research_lock/GPU_EXPERIMENTS_PROTOCOL_v1.md",
        ],
    },
    "three_source_oof": {
        "status": "locked",
        "verdict": "beats-both (constructed routing mixture, not transfer)",
        "artifacts": [
            "experiments/kbound/results/mixed_protocol_oof_v2/mixed_protocol_oof_v2_result.json",
            "research_lock/KBOUND_MIXED_STREAM_v2.json",
        ],
    },
    "cifar10_1_K": {
        "status": "locked_diagnostic_fail",
        "verdict": "diagnostic fail (transfer bar); no claim",
        "artifacts": [
            "experiments/kbound/results/cifar101_multiseed_v1/seed0/result_manifest.json",
            "experiments/kbound/results/cifar101_multiseed_v1/seed1/result_manifest.json",
            "experiments/kbound/results/cifar101_multiseed_v1/seed2/result_manifest.json",
            "experiments/kbound/results/cifar101_multiseed_v1/seed3/result_manifest.json",
            "experiments/kbound/results/cifar101_multiseed_v1/seed4/result_manifest.json",
        ],
    },
    "pacs_multiseed": {
        "status": "locked_diagnostic_null",
        "verdict": "completed 3-seed action-safety diagnostic; no beats-both claim",
        "artifacts": [
            "experiments/kbound/results/win_hunt_v5/pacs_aggr/pacs_result.json",
            "experiments/kbound/results/pacs_seed1.json",
            "experiments/kbound/results/pacs_seed2.json",
            "experiments/kbound/results/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json",
        ],
    },
    "imagenet_r_D": {
        "status": "locked_diagnostic_null",
        "verdict": "completed 4-seed, 10-backbone diagnostic; 0/10 CI-supported beats-both",
        "artifacts": [
            "experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json",
            *[
                f"experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/per_condition_imagenet-r_{method}_seed{seed}.json"
                for seed in range(4)
                for method in (
                    "convnext_base", "convnext_tiny", "efficientnet_b0", "efficientnet_b3",
                    "resnet101", "resnet152", "resnext101_32x8d", "swin_b", "swin_t", "vit_b_16",
                )
            ],
        ],
    },
}

NOT_LOCKED = {}

# cifar10c_sar is PERMANENTLY WITHHELD: seed-0 aggregate is non-reproducing.
# The SAR arm is noted as withheld in kbound_short_body.tex and in cifar10c_tent_eata verdict.
# No lock record will be created for this track. This is a deliberate scientific decision,
# not an incomplete item. Updated: 2026-08-17.
PERMANENTLY_WITHHELD = {
    "cifar10c_sar": "seed-0 aggregate non-reproducing; withheld from paper per SUBMISSION_LEDGER §3; permanently closed 2026-08-17",
}



def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_seal() -> dict:
    tracks_out = {}
    missing = []
    for name, spec in TRACKS.items():
        files = {}
        for rel in spec["artifacts"]:
            p = ROOT / rel
            if not p.is_file():
                missing.append(rel)
                continue
            files[rel] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
        entry = {
            "status": spec["status"],
            "verdict": spec["verdict"],
            "files": files,
        }
        if "caveat" in spec:
            entry["caveat"] = spec["caveat"]
        tracks_out[name] = entry
    if missing:
        raise FileNotFoundError("Missing seal artifacts:\n  " + "\n  ".join(missing))
    return {
        "schema_version": 1,
        "seal_id": "NINE_TRACK_LOCK_SEAL_v1",
        "sealed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": (
            "Locked = frozen aggregate + hashed source artifacts. "
            "Does not change scientific verdicts. "
            "Office-Home LOO beats-both is not promoted."
        ),
        "tracks": tracks_out,
        "not_locked": NOT_LOCKED,
        "permanently_withheld": PERMANENTLY_WITHHELD,
    }


def write_yaml_sidecar(seal: dict) -> None:
    lines = [
        "# NINE_TRACK_LOCK_SEAL_v1 — sealed evidence inventory for kbound_short",
        f"# sealed_utc: {seal['sealed_utc']}",
        "# Auto-generated companion to experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json",
        "# Verify: python docs/research/kbound/scripts/seal_nine_track_lock.py --verify",
        "",
        "seal_id: NINE_TRACK_LOCK_SEAL_v1",
        "policy: >",
        "  Locked means frozen aggregate + hashed source artifacts (replayable).",
        "  Scientific verdicts unchanged. Office-Home LOO beats-both is NOT promoted.",
        "",
        "locked_tracks:",
    ]
    for name, t in seal["tracks"].items():
        lines.append(f"  - id: {name}")
        lines.append(f"    status: {t['status']}")
        lines.append(f"    verdict: {t['verdict']!r}")
        if "caveat" in t:
            lines.append(f"    caveat: {t['caveat']!r}")
        lines.append("    artifacts:")
        for rel, meta in t["files"].items():
            lines.append(f"      - path: {rel}")
            lines.append(f"        sha256: {meta['sha256']}")
            lines.append(f"        bytes: {meta['bytes']}")
    lines.append("")
    lines.append("not_locked:")
    for name, reason in seal["not_locked"].items():
        lines.append(f"  {name}: {reason!r}")
    lines.append("")
    lines.append("permanently_withheld:")
    for name, reason in seal["permanently_withheld"].items():
        lines.append(f"  {name}: {reason!r}")
    lines.append("")
    LOCK_YAML.write_text("\n".join(lines), encoding="utf-8")


def verify(seal: dict) -> list[str]:
    errors = []
    for name, t in seal["tracks"].items():
        for rel, meta in t["files"].items():
            p = ROOT / rel
            if not p.is_file():
                errors.append(f"{name}: missing {rel}")
                continue
            got = sha256_file(p)
            if got != meta["sha256"]:
                errors.append(f"{name}: hash mismatch {rel}\n  expected {meta['sha256']}\n  got      {got}")
            elif p.stat().st_size != meta["bytes"]:
                errors.append(f"{name}: size mismatch {rel}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="Verify existing seal hashes")
    args = ap.parse_args()

    if args.verify:
        if not SEAL_JSON.is_file():
            print(f"FAIL: seal missing: {SEAL_JSON}")
            return 1
        seal = json.loads(SEAL_JSON.read_text(encoding="utf-8"))
        errs = verify(seal)
        if errs:
            print("FAIL:")
            for e in errs:
                print(" ", e)
            return 1
        print(f"OK: verified {len(seal['tracks'])} tracks against {SEAL_JSON}")
        for name, t in seal["tracks"].items():
            print(f"  {name}: {t['status']} ({len(t['files'])} files)")
        print("not_locked:", ", ".join(seal["not_locked"]))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seal = build_seal()
    payload = json.dumps(seal, indent=2, sort_keys=True) + "\n"
    SEAL_JSON.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    SEAL_SHA.write_text(f"{digest}  LOCK_SEAL.json\n", encoding="utf-8")
    write_yaml_sidecar(seal)
    print(f"Wrote {SEAL_JSON}")
    print(f"Wrote {SEAL_SHA}")
    print(f"Wrote {LOCK_YAML}")
    for name, t in seal["tracks"].items():
        print(f"  sealed {name}: {t['status']} ({len(t['files'])} files)")
    print("not_locked:", ", ".join(seal["not_locked"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
