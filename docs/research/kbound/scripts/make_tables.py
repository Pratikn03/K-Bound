#!/usr/bin/env python3
"""Emit paper result-table numbers from the canonical result manifest.

Falls back to canonical locked JSON artifacts when results_source.json lacks
locked_analysis / headtohead blocks (so PDF macros stay current before a full rerun).

    python docs/research/kbound/scripts/make_tables.py

Single source of truth -> docs/research/kbound/paper/generated/kbound_numbers.tex.
"""

import hashlib
import json
import math
import os

HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ROOT = REPO_ROOT if os.path.isdir(os.path.join(REPO_ROOT, "docs/research/kbound")) else os.path.dirname(HERE)
KBOUND = os.path.join(ROOT, "docs/research/kbound") if ROOT == REPO_ROOT else ROOT
SRC = os.path.join(KBOUND, "paper/generated/kbound_result_manifest.json")
OUT = os.path.join(KBOUND, "paper/generated/kbound_numbers.tex")
CCT_RELEASE = os.path.join(KBOUND, "paper/generated/cct20_release_manifest.json")
CCT_PRIMARY_TABLE = os.path.join(KBOUND, "paper/generated/cct20_primary_table.tex")
CCT_PRIMARY_DISPLAY_OUT = os.path.join(
    KBOUND,
    "paper/generated/cct20_primary_table_display.tex",
)
CCT_LOCATION_DISPLAY_OUT = os.path.join(
    KBOUND,
    "paper/generated/cct20_location_effects_display.tex",
)
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


def _write_cct20_location_display_table():
    """Emit a readable, rounded paper view without changing the sealed authority."""

    release = _load_json(CCT_RELEASE)
    rows = release.get("location_effects")
    if not isinstance(rows, list) or not rows:
        raise ValueError("CCT-20 release manifest is missing location_effects")

    rendered_rows = []
    for row in rows:
        actions = row["action_counts"]
        effects = row["effect_counts"]
        rendered_rows.append(
            f"{row['location_id']} & {row['n_evaluation_images_per_checkpoint']} & "
            f"{actions['ADAPT']}/{actions['FREEZE']}/{actions['ABSTAIN']} & "
            f"{effects['helpful']}/{effects['zero']}/{effects['harmful']} & "
            f"{f(float(row['mean_adaptation_benefit']))} & "
            f"{f(float(row['mean_versus_always_adapt']))} & "
            f"{f(float(row['mean_versus_always_freeze']))} \\\\"
        )

    lines = [
        "% AUTO-GENERATED by scripts/make_tables.py. Do not edit by hand.",
        "% Display-only CCT-20 table rounded to four decimals for legibility.",
        "% The sealed release manifest and cct20_location_effects.tex retain full precision.",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Location & $n$/checkpoint & A/F/U & Benefit $+/0/-$ & Adapt $-$ freeze & KGA $-$ adapt & KGA $-$ freeze \\",
        r"\midrule",
        *rendered_rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    with open(CCT_LOCATION_DISPLAY_OUT, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines))
    print("wrote", CCT_LOCATION_DISPLAY_OUT)


def _write_cct20_primary_display_table():
    """Clarify inference headings without rewriting the receipt-bound table.

    The sealed source table is copied verbatim below its header. Reference-test
    flags remain the recorded flags; new headings do not assert null invariance
    or finite-sample coverage. Fail closed if the source hash or schema changes.
    """

    release = _load_json(CCT_RELEASE)
    expected = release["generated_artifacts"]["cct20_primary_table_tex"]
    with open(CCT_PRIMARY_TABLE, "rb") as handle:
        source_bytes = handle.read()
    if len(source_bytes) != expected["bytes"] or hashlib.sha256(source_bytes).hexdigest() != expected["sha256"]:
        raise ValueError("CCT-20 primary table differs from its sealed manifest")
    source = source_bytes.decode("ascii")
    old_header = (
        r"Comparator & Baseline regret $-$ KGA regret & Bonferroni 97.5\% CI & "
        r"Exact $p$ & Holm $p$ & Reject at .05 \\"
    )
    new_header = (
        r"Comparator & Baseline regret $-$ KGA regret & Nominal 97.5\% CI & "
        r"Sign-flip $p$ & Holm $p$ & Holm flag (.05) \\"
    )
    if source.count(old_header) != 1 or source.count(r"\midrule") != 1:
        raise ValueError("Unexpected sealed CCT-20 primary table layout")
    # Strip only historical generator comments; numerical rows and flags stay
    # byte-for-byte identical to the sealed table, including decimal precision.
    display = "\n".join(line for line in source.splitlines() if not line.startswith("%")) + "\n"
    display = display.replace(old_header, new_header)
    if display.split(r"\midrule", 1)[1] != source.split(r"\midrule", 1)[1]:
        raise ValueError("CCT-20 display rendering changed a result row")
    provenance = (
        "% AUTO-GENERATED by scripts/make_tables.py. Do not edit by hand.\n"
        "% Presentation-only headings; sealed numerical rows and flags are unchanged.\n"
        f"% Source table SHA-256: {expected['sha256']}\n"
        "% Nominal bootstrap levels and sign-flip assumptions are defined in the manuscript.\n"
    )
    with open(CCT_PRIMARY_DISPLAY_OUT, "w", encoding="ascii") as handle:
        handle.write(provenance + display)
    print("wrote", CCT_PRIMARY_DISPLAY_OUT)



def _display_score_fields(score, *, aggregate=False):
    """Read exact recorded fields; never infer a decision unit from a score unit."""
    n_key = "n_domain_seed_units" if aggregate else "n"
    n = score[n_key]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("Display row must declare a positive evaluation-unit count")
    values = [score["regret"][key] for key in ("kga", "always_adapt", "always_freeze")]
    values.extend((score["fa_u"], score["decision_coverage"]))
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Display row has an invalid recorded score")
    if any(value > 1 for value in values[-2:]):
        raise ValueError("Display frequencies must lie in [0, 1]")
    if not aggregate:
        counts = [score[key] for key in ("adapt_count", "freeze_count", "abstain_count")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("Display row has invalid action counts")
        if sum(counts) != n:
            raise ValueError("Display action counts do not match the declared unit count")
        if not math.isclose((counts[0] + counts[1]) / n, values[-1], abs_tol=1e-12):
            raise ValueError("Display commitment rate disagrees with the recorded actions")
    return n, values


def _render_metric_table(rows, *, primary=False):
    """Format one explicitly named metric; no pooling or favorable-row filtering."""
    header = (
        r"Candidate & $n$ & KGA & Adapt & Freeze & A/F/U & $\mathrm{FA}_{\mathrm u}$ & Commitment rate \\"
        if primary else
        r"Protocol & $n$ & KGA & Adapt & Freeze & $\mathrm{FA}_{\mathrm u}$ & Commitment rate \\"
    )
    rendered = []
    for label, score, aggregate in rows:
        n, values = _display_score_fields(score, aggregate=aggregate)
        parts = [label, str(n), *(f(value) for value in values[:3])]
        if primary:
            parts.append("/".join(str(score[key]) for key in ("adapt_count", "freeze_count", "abstain_count")))
        parts.extend(f(value) for value in values[3:])
        rendered.append(" & ".join(parts) + r" \\")
    return "\n".join([
        "% AUTO-GENERATED by scripts/make_tables.py. Display only; authorities unchanged.",
        r"\begin{tabular}{@{}lrrrrcrr@{}}" if primary else r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule", header, r"\midrule", *rendered, r"\bottomrule", r"\end{tabular}", "",
    ])


def _write_metric_separated_display_tables():
    """Keep primary accuracy, diagnostic accuracy, and balanced accuracy distinct."""
    if not canonical:
        raise ValueError("Metric-separated tables require the reconciled canonical authority")
    panel = canonical["panels"]
    primary = [(candidate.upper() if candidate != "tent" else "Tent",
                panel["cifar10c"]["panel"]["candidates"][candidate], False)
               for candidate in ("tent", "eata", "sar")]
    accuracy = [
        ("Office-Home primary", panel["officehome"]["primary"]["exact_rank_transfer_score"], False),
        ("Office-Home stream seeds", panel["officehome"]["test_stream_seed_replication"]["exact_rank_transfer_score"], False),
        *[(f"ImageNet-C {candidate.upper() if candidate != 'tent' else 'Tent'}",
           panel["imagenetc"]["panel"]["candidates"][candidate], False)
          for candidate in ("tent", "eata", "sar")],
        ("PACS (aggregate)", panel["pacs"]["pooled_domain_seed_mean"], True),
        ("CIFAR-10.1", panel["cifar101"]["replay"]["exact_rank_transfer_score"], False),
    ]
    balanced = [
        ("ImageNet-R backbones", panel["imagenet_r"]["panel"]["architecture_panel_aggregate"], False),
        ("Camelyon17 OOD", panel["camelyon17"]["ood"]["replay"]["exact_rank_transfer_score"], False),
        *[(f"Camelyon17 B--v2 {candidate.upper() if candidate != 'tent' else 'Tent'}",
           panel["camelyon17"]["b_v2_diagnostic"]["panel"]["candidates"][candidate], False)
          for candidate in ("tent", "eata", "sar")],
        ("RxRx1 model seed 0", panel["rxrx1"]["primary_model_seed0"]["exact_rank_transfer_score"], False),
    ]
    # Render and validate every row before replacing any presentation output.
    outputs = {
        "kbound_primary_accuracy_table.tex": _render_metric_table(primary, primary=True),
        "kbound_auxiliary_accuracy_table.tex": _render_metric_table(accuracy),
        "kbound_auxiliary_balanced_accuracy_table.tex": _render_metric_table(balanced),
    }
    for name, source in outputs.items():
        path = os.path.join(KBOUND, "paper", "generated", name)
        with open(path, "w", encoding="ascii") as handle:
            handle.write(source)
        print("wrote", path)


def _write_cct20_safe_utility_display_table():
    """Expose the already sealed 95% endpoint without rescoring or changing its rule."""
    release = _load_json(CCT_RELEASE)
    safe = release["safe_utility"]
    expected_sign = "baseline_regret_minus_kga_regret; positive_favors_kga"
    if safe["contrast_sign"] != expected_sign:
        raise ValueError("CCT-20 safe-utility contrast convention changed")
    margin = safe["frozen_noninferiority_margin"]
    if margin != -0.005:
        raise ValueError("CCT-20 frozen noninferiority margin changed")
    rows = []
    endpoints = {}
    for key, label, threshold in (
        ("versus_always_adapt", "Always adapt", 0.0),
        ("versus_always_freeze", "Always freeze", margin),
    ):
        row = safe[key]
        point = row["point_estimate"]
        interval = row["pointwise_95_ci"]
        if len(interval) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in [point, *interval]
        ) or interval[0] > interval[1]:
            raise ValueError("CCT-20 safe-utility interval is malformed")
        endpoints[key] = interval[0]
        rows.append(
            f"{label} & {f(point)} & [{f(interval[0])}, {f(interval[1])}] & "
            f"$L>{f(threshold)}$ " + r"\\"
        )
    passed = endpoints["versus_always_adapt"] > 0 and endpoints["versus_always_freeze"] > margin
    if not isinstance(safe["passes"], bool) or passed is not safe["passes"]:
        raise ValueError("CCT-20 recorded safe-utility flag contradicts its strict rule")
    path = os.path.join(KBOUND, "paper", "generated", "cct20_safe_utility_display.tex")
    source = "\n".join([
        "% AUTO-GENERATED by scripts/make_tables.py. Sealed endpoint, display rounding only.",
        r"\begin{tabular}{@{}lrrl@{}}",
        r"\toprule",
        r"Comparator & Mean contrast & Nominal 95\% CI & Required lower bound \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", "",
    ])
    with open(path, "w", encoding="ascii") as handle:
        handle.write(source)
    print("wrote", path)

_write_metric_separated_display_tables()
_write_cct20_safe_utility_display_table()
_write_cct20_location_display_table()
_write_cct20_primary_display_table()
