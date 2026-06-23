#!/usr/bin/env python3
"""
foldin_multiseed_results.py
===========================
Fold a MULTISEED_ANALYSIS_RESULTS.json (emitted by
experiments/kbound/wilds/multiseed_paired_ci.py) into PAPER-READY artifacts:

  (a) a LaTeX table-row block  (--emit latex, default also written)
  (b) a short Markdown summary (--emit md)

For each method in {tent, eata, sar} it reports the paired difference of the
KGA gate vs. the two trivial policies (always-adapt, always-freeze), each with
its 95 % bootstrap CI, Holm-adjusted p-value, and the survives-Holm flag --
EXACTLY the fields the analysis script computes. Nothing is invented.

DESIGN PRINCIPLE -- NO PLACEHOLDER NUMBERS.
Every number printed is read from a real field in the input JSON. If a required
field is absent the script raises SchemaError and exits non-zero. It will never
emit a 0.0 / NaN / "TBD" stand-in for a missing measurement.

Two input schemas are accepted and auto-detected:

  * PRODUCTION schema (what multiseed_paired_ci.py writes for the GPU runs):
        top-level "comparisons": [ {candidate, trivial, label,
            mean_diff_kga_minus_trivial, ci95_lo, ci95_hi, p_raw, p_holm,
            survives_holm, kga_lower, ...}, ... ]
        top-level "candidates": { <method>: {...}, ... }

  * SMOKE schema (the synthetic verify_runner_pipeline.py report, used only to
    prove this integrator end-to-end):
        "datasets": { <dataset>: { "comparisons": [ {label, mean_diff,
            ci95:[lo,hi], p_raw, p_holm, survives_holm}, ... ] } }
    Rows from the smoke file are stamped SYNTHETIC in every artifact so they can
    never be mistaken for real measurements.

Usage:
    python3 scripts/foldin_multiseed_results.py \
        --in  experiments/kbound/results/<run>/MULTISEED_ANALYSIS_RESULTS.json \
        --dataset {imagenet-r,camelyon17} \
        [--emit latex|md|both] [--out-dir DIR]

Exit codes: 0 ok; 2 file missing/empty/unparseable; 3 schema/field missing.
Pure stdlib. No numpy, no torch.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import subprocess
import sys

METHODS = ("tent", "eata", "sar")
TRIVIALS = ("always-adapt", "always-freeze")

# Human-facing dataset labels for the paper.
DATASET_LABEL = {
    "imagenet-r": "ImageNet-R (Protocol D, multi-seed)",
    "camelyon17": "Camelyon17-WILDS (full-scale B)",
}
DATASET_ALIASES = {  # tolerate dataset strings stored inside the JSON
    "imagenet-r": "imagenet-r", "imagenetr": "imagenet-r", "imagenet_r": "imagenet-r",
    "camelyon17": "camelyon17", "camelyon": "camelyon17",
}


class SchemaError(Exception):
    """A required field/row was absent. We refuse to emit a placeholder."""


def _git_commit(repo_root: str) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, timeout=10)
        h = out.stdout.strip()
        return h if h else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _require(d: dict, key: str, where: str):
    if key not in d or d[key] is None:
        raise SchemaError(
            f"required field '{key}' missing in {where}. "
            f"Refusing to emit a placeholder number. Present keys: {sorted(d.keys())}")
    return d[key]


def _norm_dataset(s: str) -> str:
    s = (s or "").strip().lower()
    return DATASET_ALIASES.get(s, s)


def _comparisons_for_dataset(doc: dict, dataset: str, src: str):
    """Return (rows, is_synthetic) where rows is a list of comparison dicts in
    whatever schema the file uses. Raises SchemaError if none are found."""
    # --- SMOKE schema: nested under datasets.<dataset>.comparisons ---
    if isinstance(doc.get("datasets"), dict):
        is_synth = bool(doc.get("_synthetic_smoke", False))
        dsmap = doc["datasets"]
        # match by normalized key
        key = None
        for k in dsmap:
            if _norm_dataset(k) == dataset:
                key = k
                break
        if key is None:
            raise SchemaError(
                f"dataset '{dataset}' not present in {src} 'datasets' block "
                f"(have: {sorted(dsmap.keys())})")
        block = dsmap[key]
        rows = _require(block, "comparisons", f"{src}:datasets.{key}")
        if not rows:
            raise SchemaError(f"'comparisons' is empty for dataset '{dataset}' in {src}")
        return rows, is_synth
    # --- PRODUCTION schema: top-level comparisons ---
    if "comparisons" in doc:
        rows = doc["comparisons"]
        if not rows:
            raise SchemaError(f"top-level 'comparisons' is empty in {src}")
        # sanity: dataset stamp should match if present (warn, don't fabricate)
        stamped = _norm_dataset(doc.get("dataset", ""))
        if stamped and stamped != dataset:
            raise SchemaError(
                f"--dataset {dataset} does not match dataset stamped in file "
                f"('{doc.get('dataset')}'). Aborting rather than mislabel results.")
        return rows, bool(doc.get("_synthetic_smoke", False))
    raise SchemaError(
        f"{src} has neither a top-level 'comparisons' list (production schema) "
        f"nor a 'datasets.*.comparisons' block (smoke schema).")


def _extract(row: dict, src: str) -> dict:
    """Pull the five reported numbers out of ONE comparison row, supporting both
    field-name conventions. Missing -> SchemaError (never a default)."""
    label = _require(row, "label", f"{src} comparison row")

    # mean diff: production='mean_diff_kga_minus_trivial', smoke='mean_diff'
    if "mean_diff_kga_minus_trivial" in row:
        diff = row["mean_diff_kga_minus_trivial"]
    elif "mean_diff" in row:
        diff = row["mean_diff"]
    else:
        raise SchemaError(
            f"comparison '{label}' lacks a mean-diff field "
            f"('mean_diff_kga_minus_trivial' or 'mean_diff') in {src}")
    if diff is None:
        raise SchemaError(f"comparison '{label}' has null mean-diff in {src}")

    # CI: production=ci95_lo/ci95_hi, smoke=ci95:[lo,hi]
    if "ci95_lo" in row and "ci95_hi" in row:
        lo, hi = row["ci95_lo"], row["ci95_hi"]
    elif "ci95" in row and isinstance(row["ci95"], (list, tuple)) and len(row["ci95"]) == 2:
        lo, hi = row["ci95"][0], row["ci95"][1]
    else:
        raise SchemaError(
            f"comparison '{label}' lacks a 95% CI ('ci95_lo'/'ci95_hi' or "
            f"'ci95':[lo,hi]) in {src}")
    if lo is None or hi is None:
        raise SchemaError(f"comparison '{label}' has null CI bound in {src}")

    p_holm = _require(row, "p_holm", f"comparison '{label}' in {src}")
    survive = _require(row, "survives_holm", f"comparison '{label}' in {src}")
    # p_raw is informative but optional in the smoke schema; keep if present.
    p_raw = row.get("p_raw", None)

    return {
        "label": label,
        "diff": float(diff),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_holm": float(p_holm),
        "p_raw": (float(p_raw) if p_raw is not None else None),
        "survive": bool(survive),
    }


def _index_rows(rows, src):
    """Map (method, trivial) -> extracted numbers. Parses the trivial out of the
    'candidate'/'trivial' fields when present, else out of the label text."""
    idx = {}
    for r in rows:
        ext = _extract(r, src)
        cand = r.get("candidate")
        triv = r.get("trivial")
        if cand is None or triv is None:
            # derive from "<method> vs <trivial>"
            lab = ext["label"]
            if " vs " not in lab:
                raise SchemaError(
                    f"cannot parse method/trivial from label '{lab}' in {src}")
            cand, triv = (x.strip() for x in lab.split(" vs ", 1))
        idx[(cand.strip(), triv.strip())] = ext
    return idx


def collect(doc, dataset, src):
    """Build an ordered structure: for each METHOD, its two TRIVIAL comparisons.
    Raises if any of the 6 expected (method, trivial) rows is absent -- we never
    silently drop or zero-fill a cell."""
    rows, is_synth = _comparisons_for_dataset(doc, dataset, src)
    idx = _index_rows(rows, src)
    out = []
    missing = []
    for m in METHODS:
        cell = {"method": m, "rows": {}}
        for t in TRIVIALS:
            key = (m, t)
            if key not in idx:
                missing.append(f"{m} vs {t}")
                continue
            cell["rows"][t] = idx[key]
        out.append(cell)
    if missing:
        raise SchemaError(
            f"required comparison(s) absent from {src} for dataset '{dataset}': "
            f"{missing}. Expected all of {{tent,eata,sar}} x {{always-adapt,"
            f"always-freeze}}. Refusing to emit a partial/placeholder table.")
    return out, is_synth


# ----------------------------- rendering -------------------------------------

def _fmt(x, nd=4):
    return f"{x:+.{nd}f}"


def _sig(survive):
    return r"\checkmark" if survive else r"$-$"


def header_stamp(src_abs, commit, dataset, is_synth):
    ds_label = DATASET_LABEL.get(dataset, dataset)
    synth_tag = "  [SYNTHETIC SMOKE -- NOT A REAL MEASUREMENT]" if is_synth else ""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"source_file : {src_abs}{synth_tag}",
        f"git_commit  : {commit}",
        f"dataset     : {dataset} ({ds_label})",
        f"generated   : {ts} by scripts/foldin_multiseed_results.py",
    ]
    return lines


def render_latex(struct, stamp_lines, is_synth, dataset):
    L = []
    for ln in stamp_lines:
        L.append("% " + ln)
    if is_synth:
        L.append("% !!! SYNTHETIC SMOKE DATA -- do NOT paste into the paper as real results !!!")
    cap_synth = " (SYNTHETIC SMOKE)" if is_synth else ""
    ds_label = DATASET_LABEL.get(dataset, dataset)
    L.append(r"% --- paste rows into the natural-shift results table ---")
    L.append(r"% columns: Method & vs always-adapt (diff [95\% CI], Holm-$p$, surv.)"
             r" & vs always-freeze (diff [95\% CI], Holm-$p$, surv.)")
    L.append(r"\multicolumn{6}{l}{\textit{%s%s}}\\" % (ds_label, cap_synth))
    for cell in struct:
        m = cell["method"]
        a = cell["rows"]["always-adapt"]
        f = cell["rows"]["always-freeze"]
        row = (
            r"\textsc{%s} & $%s$ & $[%s,\,%s]$ & %.2e %s & $%s$ & $[%s,\,%s]$ & %.2e %s \\"
            % (
                m,
                _fmt(a["diff"]), _fmt(a["ci_lo"]), _fmt(a["ci_hi"]), a["p_holm"], _sig(a["survive"]),
                _fmt(f["diff"]), _fmt(f["ci_lo"]), _fmt(f["ci_hi"]), f["p_holm"], _sig(f["survive"]),
            )
        )
        L.append(row)
    return "\n".join(L) + "\n"


def render_md(struct, stamp_lines, is_synth, dataset):
    M = []
    title_synth = " — SYNTHETIC SMOKE (placeholder, not real)" if is_synth else ""
    M.append(f"## Multi-seed paired-CI fold-in: {dataset}{title_synth}\n")
    M.append("```")
    for ln in stamp_lines:
        M.append(ln)
    M.append("```")
    if is_synth:
        M.append("> WARNING: these rows come from the synthetic smoke report "
                 "(`_synthetic_smoke: true`). They prove the pipeline works; "
                 "they are NOT measured results and must not be cited.\n")
    M.append("KGA gate regret minus trivial-policy regret (lower = gate better). "
             "CI is the 95% paired bootstrap interval; *p* is Holm-adjusted; "
             "Survives = Holm-significant AND gate strictly lower.\n")
    M.append("| Method | vs | mean diff (KGA − trivial) | 95% CI | Holm *p* | Survives |")
    M.append("|---|---|---|---|---|---|")
    for cell in struct:
        m = cell["method"]
        for t in TRIVIALS:
            r = cell["rows"][t]
            surv = "yes" if r["survive"] else "no"
            M.append(
                f"| {m} | {t} | {_fmt(r['diff'])} | "
                f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | {r['p_holm']:.2e} | {surv} |")
    # compact survive-summary
    M.append("")
    survivors = [
        f"{cell['method']} vs {t}"
        for cell in struct for t in TRIVIALS if cell["rows"][t]["survive"]
    ]
    if survivors:
        M.append("**Survives Holm (gate strictly better):** " + ", ".join(survivors) + ".")
    else:
        M.append("**Survives Holm (gate strictly better):** none.")
    return "\n".join(M) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="path to MULTISEED_ANALYSIS_RESULTS.json (or the synthetic smoke report)")
    ap.add_argument("--dataset", required=True, choices=["imagenet-r", "camelyon17"])
    ap.add_argument("--emit", choices=["latex", "md", "both"], default="both")
    ap.add_argument("--out-dir", default="",
                    help="if set, also write foldin_<dataset>.tex / .md here")
    ap.add_argument("--repo-root", default="",
                    help="git repo root for the commit stamp (default: inferred)")
    a = ap.parse_args(argv)

    src_abs = os.path.abspath(a.inp)

    # --- load, with loud errors on missing / empty / unparseable ---
    if not os.path.exists(src_abs):
        sys.stderr.write(f"ERROR: input file does not exist: {src_abs}\n")
        return 2
    if os.path.getsize(src_abs) == 0:
        sys.stderr.write(f"ERROR: input file is empty (0 bytes): {src_abs}\n")
        return 2
    try:
        with open(src_abs) as fh:
            doc = json.load(fh)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"ERROR: input file is not valid JSON ({e}): {src_abs}\n")
        return 2
    if not isinstance(doc, dict):
        sys.stderr.write(f"ERROR: top-level JSON is not an object: {src_abs}\n")
        return 2

    dataset = _norm_dataset(a.dataset)
    repo_root = a.repo_root or os.path.dirname(os.path.dirname(src_abs))
    # prefer the actual repo containing this script for the commit hash
    script_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = _git_commit(script_repo if os.path.isdir(os.path.join(script_repo, ".git"))
                         else repo_root)

    try:
        struct, is_synth = collect(doc, dataset, src_abs)
    except SchemaError as e:
        sys.stderr.write(f"SCHEMA ERROR: {e}\n")
        return 3

    stamp = header_stamp(src_abs, commit, dataset, is_synth)

    blocks = []
    if a.emit in ("latex", "both"):
        tex = render_latex(struct, stamp, is_synth, dataset)
        blocks.append(("LATEX TABLE-ROW BLOCK", tex))
    if a.emit in ("md", "both"):
        md = render_md(struct, stamp, is_synth, dataset)
        blocks.append(("MARKDOWN SUMMARY", md))

    for title, body in blocks:
        print("=" * 78)
        print(title + ("   [SYNTHETIC]" if is_synth else ""))
        print("=" * 78)
        print(body)

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        suffix = ".SYNTHETIC" if is_synth else ""
        if a.emit in ("latex", "both"):
            p = os.path.join(a.out_dir, f"foldin_{dataset}{suffix}.tex")
            with open(p, "w") as fh:
                fh.write(render_latex(struct, stamp, is_synth, dataset))
            print("WROTE", os.path.abspath(p))
        if a.emit in ("md", "both"):
            p = os.path.join(a.out_dir, f"foldin_{dataset}{suffix}.md")
            with open(p, "w") as fh:
                fh.write(render_md(struct, stamp, is_synth, dataset))
            print("WROTE", os.path.abspath(p))

    return 0


if __name__ == "__main__":
    sys.exit(main())
