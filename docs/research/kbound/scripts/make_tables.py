#!/usr/bin/env python3
"""Emit paper result-table numbers from the canonical result manifest.

Falls back to canonical locked JSON artifacts when results_source.json lacks
locked_analysis / headtohead blocks (so PDF macros stay current before a full rerun).

    python docs/research/kbound/scripts/make_tables.py

Single source of truth -> docs/research/kbound/paper/generated/kbound_numbers.tex.
"""

import json
import os

HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ROOT = REPO_ROOT if os.path.isdir(os.path.join(REPO_ROOT, "docs/research/kbound")) else os.path.dirname(HERE)
KBOUND = os.path.join(ROOT, "docs/research/kbound") if ROOT == REPO_ROOT else ROOT
SRC = os.path.join(KBOUND, "paper/generated/kbound_result_manifest.json")
OUT = os.path.join(KBOUND, "paper/generated/kbound_numbers.tex")
LOCKED_DEFAULT = os.path.join(ROOT, "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json")
H2H_DEFAULT = os.path.join(
    ROOT,
    "experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json",
)
RECONCILED = os.path.join(ROOT, "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json")
# These generated macros appear in both text-mode tables and math-mode cells.
# ``\textnormal`` is safe in either context; a bare ``\mathrm`` is math-only
# and previously broke the maintained full-manuscript build.
WITHHELD = r"\textnormal{withheld}"
PENDING = r"\textnormal{pending}"


def f(x):
    return f"{x:.4f}"


def pct(x):
    return f"{x * 100:.0f}"


def zero_event_cp95(n):
    """Upper Clopper-Pearson bound for zero events, undefined at zero exposure."""
    return r"\textnormal{not defined}" if n <= 0 else f(1.0 - 0.05 ** (1.0 / n))


def _load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def _locked():
    d = _load_json(SRC)
    la = d.get("locked_analysis")
    if la:
        return la
    return _load_json(LOCKED_DEFAULT)


def _headtohead():
    d = _load_json(SRC)
    hh = d.get("headtohead")
    if hh:
        return hh
    raw = _load_json(H2H_DEFAULT)
    if not raw:
        return {}
    h = raw.get("headtohead", raw)
    return {
        "verdict": h.get("VERDICT", "—"),
        "kga_regret": float(raw.get("policy_mean_regret", {}).get("kga", 0)),
        "adapt_regret": float(raw.get("policy_mean_regret", {}).get("always_adapt", 0)),
        "freeze_regret": float(raw.get("policy_mean_regret", {}).get("always_freeze", 0)),
        "poem_regret": float(raw.get("policy_mean_regret", {}).get("poem", 0)),
        "aetta_regret": float(raw.get("policy_mean_regret", {}).get("aetta", 0)),
        "kga_fa": float(raw.get("policy_false_adapt_rate", {}).get("kga", 0)),
        "kga_decisive": float(raw.get("policy_decisive_rate", {}).get("kga", 0)),
    }


d = _load_json(SRC)
tracks = d.get("tracks", {})
canonical = _load_json(RECONCILED)
iwild_release_eligible = False
if canonical:
    source_manifest_sha256 = canonical.get("source_manifest_sha256")
    if not isinstance(source_manifest_sha256, str) or len(source_manifest_sha256) != 64:
        raise ValueError("canonical panel is missing a valid source_manifest_sha256")
    panels = canonical["panels"]
    iwild_release_eligible = panels["iwildcam"].get("release_promotion", {}).get("eligible", False)

    def as_track(score):
        regret = score["regret"]
        return {
            "regret": [regret["kga"], regret["always_adapt"], regret["always_freeze"]],
            "false_adapt": score["fa_u"],
        }

    tracks = {
        "officehome_M_v2": as_track(panels["officehome"]["primary"]["exact_rank_transfer_score"]),
        "iwildcam_H_v2": as_track(panels["iwildcam"]["primary"]["exact_rank_transfer_score"]),
        "cifar10c_tent": as_track(panels["cifar10c"]["panel"]["candidates"]["tent"]),
        "cifar10c_eata": as_track(panels["cifar10c"]["panel"]["candidates"]["eata"]),
        "imagenetc_sar": as_track(panels["imagenetc"]["panel"]["candidates"]["sar"]),
    }

    office_primary = panels["officehome"]["primary"]["exact_rank_transfer_score"]
    office_replication = panels["officehome"]["test_stream_seed_replication"]["exact_rank_transfer_score"]
    generated_macros = {}
    for prefix, score in (
        ("OH", office_primary),
        ("OHRep", office_replication),
    ):
        generated_macros.update(
            {
                f"{prefix}N": str(score["n"]),
                f"{prefix}AdaptCount": str(score["adapt_count"]),
                f"{prefix}FreezeCount": str(score["freeze_count"]),
                f"{prefix}AbstainCount": str(score["abstain_count"]),
            }
        )
    generated_macros.update(
        {
            "SourceManifestSHA": source_manifest_sha256,
            "OHRepKga": f(office_replication["regret"]["kga"]),
            "OHRepAdapt": f(office_replication["regret"]["always_adapt"]),
            "OHRepFreeze": f(office_replication["regret"]["always_freeze"]),
            "OHFaCUpper": zero_event_cp95(office_primary["adapt_count"]),
            "OHRepFaCUpper": zero_event_cp95(office_replication["adapt_count"]),
        }
    )
else:
    generated_macros = {}
ns = d.get("natural_shifts", {})
if tracks:
    ns = {
        "officehome_M_v2": dict(
            zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["officehome_M_v2"]["regret"])
        )
        | {"false_adapt": tracks["officehome_M_v2"]["false_adapt"]},
        "iwildcam_H_v2": dict(zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["iwildcam_H_v2"]["regret"]))
        | {"false_adapt": tracks["iwildcam_H_v2"]["false_adapt"]},
    }
oh = ns.get("officehome_M_v2", {})
iw = ns.get("iwildcam_H_v2", {})
M = {}
M.update(generated_macros)
if oh:
    M.update(
        {
            "OHadapt": f(oh["regret_adapt"]),
            "OHfreeze": f(oh["regret_freeze"]),
            "OHkga": f(oh["regret_kga"]),
            "OHfa": pct(oh["false_adapt"]),
        }
    )
if iw and iwild_release_eligible:
    M.update(
        {
            "iWadapt": f(iw["regret_adapt"]),
            "iWfreeze": f(iw["regret_freeze"]),
            "iWkga": f(iw["regret_kga"]),
            "iWfa": pct(iw["false_adapt"]),
        }
    )
else:
    # The archived iWildCam scorer used sklearn macro-F1, which includes
    # prediction-only classes. Keep every paper-facing macro non-numeric until
    # a pinned rerun uses the official WILDS label-present metric contract.
    M.update(
        {
            "iWN": WITHHELD,
            "iWAdaptCount": WITHHELD,
            "iWFreezeCount": WITHHELD,
            "iWAbstainCount": WITHHELD,
            "iWadapt": WITHHELD,
            "iWfreeze": WITHHELD,
            "iWkga": WITHHELD,
            "iWfa": WITHHELD,
        }
    )

cg = d.get("corruption_grids", {})
if tracks:
    cg = {
        "cifar10c_stress": {
            "candidates": {
                "tent": dict(zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["cifar10c_tent"]["regret"]))
                | {"false_adapt": tracks["cifar10c_tent"]["false_adapt"]},
                "eata": dict(zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["cifar10c_eata"]["regret"]))
                | {"false_adapt": tracks["cifar10c_eata"]["false_adapt"]},
            }
        },
        "imagenetc_sar": dict(zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["imagenetc_sar"]["regret"])),
    }
if "cifar10c_stress" in cg:
    c10 = cg["cifar10c_stress"]
    if "candidates" in c10:
        c10 = c10["candidates"]["tent"]
    M["CIFARkga"] = f(c10["regret_kga"])
    M["CIFARadapt"] = f(c10["regret_adapt"])
    M["CIFARfreeze"] = f(c10["regret_freeze"])
if "imagenetc_sar" in cg:
    ic = cg["imagenetc_sar"]
    M["ICkga"] = f(ic["regret_kga"])
    M["ICadapt"] = f(ic["regret_adapt"])
    M["ICfreeze"] = f(ic["regret_freeze"])

la = _locked()
manifest_candidates = cg.get("cifar10c_stress", {}).get("candidates", {})
for cand in ("tent", "eata"):
    c = manifest_candidates.get(cand) or la.get("candidates", {}).get(cand, {})
    if c:
        M[f"CIFAR{cand}Kga"] = f(c.get("regret_kga", c.get("kga_mean_regret")))
        M[f"CIFAR{cand}Adapt"] = f(c.get("regret_adapt", c.get("adapt_mean_regret")))
        M[f"CIFAR{cand}Freeze"] = f(c.get("regret_freeze", c.get("freeze_mean_regret")))
        M[f"CIFAR{cand}FA"] = pct(c.get("false_adapt", c.get("false_adapt_rate_pooled", 0)))

hh = _headtohead()
if hh:
    if hh.get("policy_synchronized") is False or hh.get("numeric_release_eligible") is False:
        M["HeadToHeadVerdict"] = "HISTORICAL ONLY"
        for macro in (
            "HeadToHeadKga",
            "HeadToHeadAdapt",
            "HeadToHeadFreeze",
            "HeadToHeadPoem",
            "HeadToHeadAetta",
            "HeadToHeadKgaFA",
            "HeadToHeadKgaDec",
        ):
            M[macro] = PENDING
    else:
        M["HeadToHeadVerdict"] = hh.get("verdict", "—")
        for k, macro in [
            ("kga_regret", "HeadToHeadKga"),
            ("adapt_regret", "HeadToHeadAdapt"),
            ("freeze_regret", "HeadToHeadFreeze"),
            ("poem_regret", "HeadToHeadPoem"),
            ("aetta_regret", "HeadToHeadAetta"),
        ]:
            if k in hh:
                M[macro] = f(hh[k])
        if "kga_fa" in hh:
            M["HeadToHeadKgaFA"] = f(hh["kga_fa"])
        if "kga_decisive" in hh:
            M["HeadToHeadKgaDec"] = f(hh["kga_decisive"])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    fh.write("% AUTO-GENERATED by scripts/make_tables.py. Do not edit by hand.\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("wrote", OUT)
for k, v in M.items():
    print(f"  \\{k} = {v}")
