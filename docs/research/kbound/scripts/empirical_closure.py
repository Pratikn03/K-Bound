#!/usr/bin/env python3
"""Retired July empirical-artifact builder retained for forensic provenance.

The current release artifacts are generated from the source-hashed canonical panel
by ``scripts/sync_reconciled_panels.py``.  This older builder reads superseded raw
trees and metric contracts, so executing it would reintroduce stale CIFAR-10-C
counts and a non-release-eligible iWildCam row.  It now fails closed.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/research/kbound/paper/generated/empirical_audit"


def load(path: str):
    return json.loads((ROOT / path).read_text())


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def records(paths: list[str]):
    out = []
    for pattern in paths:
        for p in sorted(ROOT.glob(pattern)):
            d = json.loads(p.read_text())
            out.extend(d.get("records", []))
    return out


def from_records(name: str, rows: list[dict], source: str, seeds, command: str):
    decisions = [str(r.get("kga_decision", r.get("decision", ""))).upper() for r in rows]
    decisions = [d for d in decisions if d in {"ADAPT", "FREEZE", "ABSTAIN"}]
    n = len(decisions)
    counts = Counter(decisions)
    b_rows = [r for r in rows if r.get("B") is not None and str(r.get("kga_decision", "")).upper() in {"ADAPT", "FREEZE", "ABSTAIN"}]
    fa = sum(str(r.get("kga_decision", "")).upper() == "ADAPT" and float(r["B"]) <= 0 for r in b_rows)
    coverable = [r for r in rows if all(r.get(k) is not None for k in ("b_hat", "B", "eps_conformal"))]
    covered = sum(abs(float(r["b_hat"]) - float(r["B"])) <= float(r["eps_conformal"]) for r in coverable)
    return {
        "track": name, "n_decisions": n, "seeds": seeds, "source": source,
        "reproduction_command": command,
        "actions": {a.lower(): {"count": counts[a], "rate": counts[a] / n if n else None,
                                 "ci95_wilson": wilson(counts[a], n)}
                    for a in ("ADAPT", "FREEZE", "ABSTAIN")},
        "false_adapt_unconditional": {"count": fa, "n": len(b_rows),
                                        "rate": fa / len(b_rows) if b_rows else None,
                                        "ci95_wilson": wilson(fa, len(b_rows))},
        "interval_coverage_observed": {"count": covered, "n": len(coverable),
                                        "rate": covered / len(coverable) if coverable else None,
                                        "ci95_wilson": wilson(covered, len(coverable)),
                                        "status": "empirical_only" if coverable else "not_retained"},
        "theoretical_coverage": {"status": "conditional_on_declared_premise",
                                  "note": "Observed coverage does not prove exchangeability, drift correction, or risk alignment."},
    }


def summary_track(name, n, adapt, coverage, fa, source, seeds, command):
    vals = {"adapt": adapt, "freeze": None if coverage is None or adapt is None else max(0.0, coverage-adapt),
            "abstain": None if coverage is None else 1-coverage}
    actions = {}
    for a, rate in vals.items():
        actions[a] = {"count": None, "rate": rate, "ci95_wilson": None,
                      "status": "summary_rate_only" if rate is not None else "not_retained"}
    return {"track": name, "n_decisions": n, "seeds": seeds, "source": source,
            "reproduction_command": command, "actions": actions,
            "false_adapt_unconditional": {"count": None, "n": None, "rate": fa,
                "ci95_wilson": None, "status": "summary_rate_only" if fa is not None else "not_retained"},
            "interval_coverage_observed": {"count": None, "n": None, "rate": None,
                "ci95_wilson": None, "status": "not_retained"},
            "theoretical_coverage": {"status": "conditional_on_declared_premise",
                "note": "No theorem premise is inferred from this empirical summary."}}


def _legacy_main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = []
    # CIFAR-10-C SAR is deliberately excluded: the archived aggregate does not
    # reproduce from the current seed-0 replay (CIFAR10C_SAR_QUARANTINE.md).
    for method in ("tent", "eata"):
        pat = f"experiments/kbound/results/stress_grid_multiseed_v1/seed*/per_condition_cifar10c_{method}_seed*.json"
        tracks.append(from_records(f"CIFAR-10-C {method.upper()}", records([pat]), pat, [0,1,2,3,4],
            f".venv/bin/python docs/research/kbound/scripts/percondition_bootstrap.py --root experiments/kbound/results/stress_grid_multiseed_v1"))
    for method in ("tent", "eata", "sar"):
        pat = f"experiments/kbound/results/win_hunt_v5/imagenetc_aggr/per_condition_imagenetc_{method}_seed0.json"
        tracks.append(from_records(f"ImageNet-C {method.upper()}", records([pat]), pat, [0],
            f".venv/bin/python docs/research/kbound/scripts/percondition_bootstrap.py --root experiments/kbound/results/win_hunt_v5/imagenetc_aggr --pattern 'per_condition_imagenetc_*.json'"))
    for method in ("tent", "eata", "sar"):
        pat = f"experiments/kbound/results/win_hunt_v5/cifar101_aggr/seed*/per_condition_cifar101_{method}_seed*.json"
        tracks.append(from_records(f"CIFAR-10.1 {method.upper()}", records([pat]), pat, [0,1,2,3,4], "bash docs/research/kbound/scripts/kbtrain.sh cifar101"))

    pacs_aggregate_rel = "experiments/kbound/results/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json"
    pacs_aggregate_path = ROOT / pacs_aggregate_rel
    if pacs_aggregate_path.exists():
        pacs = load(pacs_aggregate_rel)
        for domain, d in pacs["per_domain"].items():
            n = sum(d["n_test_cells_per_seed"])
            tracks.append(summary_track(f"PACS {domain}", n, d["adapt_rate"]["mean"],
                d["coverage"]["mean"], d["FA_u"]["mean"], pacs_aggregate_rel,
                pacs["seeds"], "bash docs/research/kbound/runbooks/finish_empirical_training.sh run"))
    else:
        pacs = load("experiments/kbound/results/win_hunt_v5/pacs_aggr/pacs_result.json")
        for domain, d in pacs["per_domain"].items():
            tracks.append(summary_track(f"PACS {domain}", d["n_test_cells"], d.get("adapt_rate"), d.get("coverage"), d.get("FA_u"),
                "experiments/kbound/results/win_hunt_v5/pacs_aggr/pacs_result.json", [0],
                "bash docs/research/kbound/runbooks/finish_empirical_training.sh run"))

    # Action summaries come from the protocol artifacts; regret CIs remain sourced from the OOF lock.
    for name, path, seeds in (
        ("OfficeHome", "experiments/kbound/results/officehome_protocol_M_v2/protocol_result.json", [0,1]),
        ("iWildCam", "experiments/kbound/results/iwildcam_protocol_H_v2/protocol_result.json", [0,1]),
    ):
        d = load(path)["test_locked"]
        tracks.append(summary_track(name, d["n_test"], d["adapt_rate"], d["coverage"], d["false_adapt"],
            path, seeds, ".venv/bin/python docs/research/kbound/scripts/bootstrap_win_cis.py"))
    iw_stability_path = "experiments/kbound/results/multiseed/iwildcam/extracted_locked/multiseed_iwildcam_tent_online.json"
    if (ROOT / iw_stability_path).exists():
        iw_stability = load(iw_stability_path)
        iw_track = next(t for t in tracks if t["track"] == "iWildCam")
        iw_track["supplemental_multiseed_stability"] = {
            "seeds": iw_stability["seeds"],
            "conditions_per_seed": iw_stability["conditions_per_seed"],
            "regret_kga_mean_sd": iw_stability["regret_kga"],
            "regret_adapt_mean_sd": iw_stability["regret_adapt"],
            "regret_freeze_mean_sd": iw_stability["regret_freeze"],
            "false_adapt_max": iw_stability["FA_u_max"],
            "verdict": iw_stability["verdict"],
            "source": iw_stability_path,
            "scope": "stability check; not a replacement for the held-out OOF lock",
        }
    cam = load("audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json")["run_split_by_domain"]["OOD_test_only"]
    tracks.append(summary_track("Camelyon17", cam["n_test"], cam["adapt_rate"], cam["coverage"], cam["false_adapt"],
        "audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json", [2,3,4],
        ".venv/bin/python audits/integrity_2026-06-20/camelyon_reconciliation/camelyon_G_reconciliation.py"))

    rx = load("experiments/kbound/results/win_hunt_v5/rxrx1_aggr/result_4a2840ef.json")
    for cand, d in rx.get("routing_a_single_candidate", {}).items():
        k = d["kga"]; c = k["decision_counts"]; n = sum(c.values())
        tracks.append(summary_track(f"RxRx1 {cand}", n, c.get("ADAPT",0)/n, (c.get("ADAPT",0)+c.get("FREEZE",0))/n, 0.0,
            "experiments/kbound/results/win_hunt_v5/rxrx1_aggr/result_4a2840ef.json", [0], "bash docs/research/kbound/scripts/kbtrain.sh rxrx1-9plus"))

    im = load("experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json")
    for cand, d in im["candidates"].items():
        n = d["false_adapt_den"]
        im_rows = records([f"experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/per_condition_imagenet-r_{cand}_seed*.json"])
        counts = Counter(str(r.get("kga_decision", "")).upper() for r in im_rows)
        tracks.append(summary_track(f"ImageNet-R {cand}", n, counts.get("ADAPT",0)/n,
            (counts.get("ADAPT",0)+counts.get("FREEZE",0))/n, d["false_adapt_num"]/n,
            "experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/per_condition_imagenet-r_*_seed*.json", im["seeds"], "bash docs/research/kbound/runbooks/finish_empirical_training.sh run"))

    d33_path = "experiments/kbound/results/controlled_multimodal_d33/results.json"
    d33 = load(d33_path)
    d33_dec = d33["decisions"]
    tracks.append(summary_track("Controlled multimodal D33", d33["n_conditions"],
        d33_dec["ADAPT"] / d33["n_conditions"],
        (d33_dec["ADAPT"] + d33_dec["FREEZE"]) / d33["n_conditions"],
        d33["false_adapt_rate"], d33_path, [0],
        "python experiments/kbound/controlled_multimodal_d33.py"))

    payload = {"schema_version": 1, "interval_method": "95% Wilson binomial interval",
               "coverage_boundary": "empirical interval hits are descriptive; theorem coverage remains conditional on its premise",
               "tracks": tracks}
    (OUT / "decision_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    pacs_done = sorted({0} | {int(p.stem.removeprefix("pacs_seed")) for p in (ROOT / "experiments/kbound/results").glob("pacs_seed*.json")})
    pacs_aggregate = pacs_aggregate_path
    inr_seeds = im["seeds"]
    claims = [
        ["Three-way action behavior", "nine-track panel", "track-specific", "decision_metrics.json", "regime/action tables", "closed: rates and Wilson intervals reported"],
        ["False-adapt uncertainty", "nine-track panel", "track-specific", "decision_metrics.json", "decision metrics table", "closed where denominator exists"],
        ["Coverage uncertainty", "nine-track panel", "track-specific", "decision_metrics.json", "coverage audit", "closed where interval-hit records exist; historical natural logs marked not retained"],
        ["CIFAR-10-C SAR quarantine", "CIFAR-10-C", "none promoted", "CIFAR10C_SAR_QUARANTINE.md", "excluded from tables and claims", "withheld: replay mismatch; reinstatement gates not met"],
        ["iWildCam numerical/action evidence", "iWildCam", "none promoted", "claim_ledger.json (KB-CLAIM-021)", "excluded from numerical tables and routing claims", "withheld: archived metric contract is invalid; population-sealed official-metric rerun required"],
        ["PACS planned seeds", "PACS", ",".join(map(str, pacs_done)), "experiments/kbound/results/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json", "PACS row", "closed" if pacs_aggregate.exists() and {0,1,2}.issubset(pacs_done) else "pending seeds 1--2; seed-matched locked protocol"],
        ["ImageNet-R planned seeds", "ImageNet-R", ",".join(map(str,inr_seeds)), "imagenetr_protocol_d_multiseed_v1", "ImageNet-R row", "closed" if set(inr_seeds)=={0,1,2,3} else "pending seed 3"],
        ["Controlled multimodal routing", "MNIST two-view controlled", "fixed seed 0; 130 conditions", d33_path, "Appendix Protocol D33", "closed: KB-CLAIM-027 supported; controlled mechanism confirmation"],
    ]
    md = ["# Empirical claim matrix", "", "| Claim | Dataset | Seeds | Authoritative artifact | Paper consumer | Status |", "|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(row) + " |" for row in claims]
    (OUT / "claim_matrix.md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT/'decision_metrics.json'}")
    print(f"wrote {OUT/'claim_matrix.md'}")


def main() -> None:
    raise RuntimeError(
        "empirical_closure.py is retired because it reads superseded artifact trees. "
        "Run scripts/reconcile_result_panels.py followed by "
        "scripts/sync_reconciled_panels.py instead."
    )


if __name__ == "__main__":
    main()
