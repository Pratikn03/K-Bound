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
LOCKED_DEFAULT = os.path.join(
    ROOT, "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json"
)
H2H_DEFAULT = os.path.join(
    ROOT,
    "experiments/kbound/results/mixed_headtohead_v1/"
    "HEADTOHEAD_RESULTS_cifar10c_tent_primary.json",
)

f = lambda x: f"{x:.4f}"
pct = lambda x: f"{x * 100:.0f}"


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
ns = d.get("natural_shifts", {})
if tracks:
    ns = {
        "officehome_M_v2": dict(zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["officehome_M_v2"]["regret"])) | {"false_adapt": tracks["officehome_M_v2"]["false_adapt"]},
        "iwildcam_H_v2": dict(zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["iwildcam_H_v2"]["regret"])) | {"false_adapt": tracks["iwildcam_H_v2"]["false_adapt"]},
    }
oh = ns.get("officehome_M_v2", {})
iw = ns.get("iwildcam_H_v2", {})
M = {}
if oh:
    M.update({
        "OHadapt": f(oh["regret_adapt"]),
        "OHfreeze": f(oh["regret_freeze"]),
        "OHkga": f(oh["regret_kga"]),
        "OHfa": pct(oh["false_adapt"]),
    })
if iw:
    M.update({
        "iWadapt": f(iw["regret_adapt"]),
        "iWfreeze": f(iw["regret_freeze"]),
        "iWkga": f(iw["regret_kga"]),
        "iWfa": pct(iw["false_adapt"]),
    })

cg = d.get("corruption_grids", {})
if tracks:
    cg = {
        "cifar10c_stress": {"candidates": {
            "tent": dict(zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["cifar10c_tent"]["regret"])) | {"false_adapt": tracks["cifar10c_tent"]["false_adapt"]},
            "eata": dict(zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["cifar10c_eata"]["regret"])) | {"false_adapt": tracks["cifar10c_eata"]["false_adapt"]},
        }},
        "imagenetc_sar": dict(zip(("regret_kga", "regret_adapt", "regret_freeze"), tracks["imagenetc_sar"]["regret"]))
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
print("wrote", os.path.relpath(OUT, ROOT))
for k, v in M.items():
    print(f"  \\{k} = {v}")
