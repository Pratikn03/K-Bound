#!/usr/bin/env python3
"""Generate a fresh Table 27 candidate from hash-bound, recorded cell actions.

No fitting, calibration, inferred actions, aggregate-only fallback or promotion.
The published output is historical and is intentionally not overwritten.

Manifest schema: kbound_task1_table_authority_v1, with models (model_id, name,
source_seed, clean_accuracy, checkpoint {path, sha256}) and panels
helpful_only/mixed_loo/mixed_b6 (each {path, sha256}). Each panel file uses schema
kbound_task1_cell_records_v1, panel_id and records. A record has model_id, cell_id,
checkpoint_sha256, a0, aa, kga_action and, on mixed panels, aetta_action.
The manifest also requires expected_cells: an independently reviewed roster of
{model_id, cell_id} pairs per panel, not derived from the exported records.
Only equal-cell weights are supported. All paths are relative to the manifest.
Checkpoint hashing never deserializes it. File identity is not proof of training
provenance, outcome exclusion, native fidelity or prospective validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


PANELS = ("helpful_only", "mixed_loo", "mixed_b6")
ACTIONS = ("ADAPT", "FREEZE", "ABSTAIN")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
DATALESS = 0x40000000


def _materialized_file(path: Path) -> Path:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlinked input is not an authority: {path}")
    info = path.stat()
    if not path.is_file() or info.st_size == 0 or getattr(info, "st_flags", 0) & DATALESS:
        raise ValueError(f"non-materialized or empty input: {path}")
    return path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with _materialized_file(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def _bound(root: Path, reference: dict) -> tuple[Path, str]:
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("missing input path binding")
    digest = reference.get("sha256", "")
    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts or not HASH_RE.fullmatch(str(digest)):
        raise ValueError("input binding requires a relative path and SHA-256")
    path = _materialized_file(root / relative)
    if _digest(path) != digest:
        raise ValueError(f"input hash mismatch: {path}")
    return path, digest


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric data")
    if not 0 <= value <= 1:
        raise ValueError(f"{field} must be in [0, 1]")
    return float(value)


def _metrics(rows: list[dict], policies: tuple[str, ...]) -> dict:
    n = len(rows)
    def mean(values):
        return math.fsum(values) / n
    oracle = [max(r["a0"], r["aa"]) for r in rows]
    accuracy = {"always_freeze": mean([r["a0"] for r in rows]),
                "always_adapt": mean([r["aa"] for r in rows]), "oracle": mean(oracle)}
    regret = {name: accuracy["oracle"] - accuracy[name]
              for name in ("always_freeze", "always_adapt")}
    counts, errors = {}, {}
    for policy in policies:
        actions = [r[f"{policy}_action"] for r in rows]
        served = [r["aa"] if a == "ADAPT" else r["a0"] for r, a in zip(rows, actions)]
        accuracy[policy] = mean(served)
        regret[policy] = mean([o - a for o, a in zip(oracle, served)])
        counts[policy] = {action: actions.count(action) for action in ACTIONS}
        harmful = sum(a == "ADAPT" and r["aa"] < r["a0"] for r, a in zip(rows, actions))
        ties = sum(a == "ADAPT" and r["aa"] == r["a0"] for r, a in zip(rows, actions))
        nonpositive = harmful + ties
        exposed = counts[policy]["ADAPT"]
        errors[policy] = {"nonpositive": nonpositive, "harmful": harmful, "ties": ties,
                          "unconditional": nonpositive / n,
                          "conditional": nonpositive / exposed if exposed else None}
    return {"n": n, "accuracy": accuracy, "regret": regret, "actions": counts,
            "false_adapt": errors}


def build_table(manifest_path: Path) -> dict:
    manifest_path = _materialized_file(Path(manifest_path).absolute())
    raw_manifest = manifest_path.read_bytes()
    manifest = _json_bytes(raw_manifest)
    if not isinstance(manifest, dict) or manifest.get("schema") != "kbound_task1_table_authority_v1":
        raise ValueError("a reviewed per-cell table authority manifest is required")
    if set(manifest.get("panels", {})) != set(PANELS):
        raise ValueError("all three panel authorities are required; no numeric fallback")
    rosters = manifest.get("expected_cells")
    if not isinstance(rosters, dict) or set(rosters) != set(PANELS):
        raise ValueError("an independent expected roster is required for every panel")
    model_list = manifest.get("models")
    if not isinstance(model_list, list) or not model_list:
        raise ValueError("nonempty model identity list required")
    models, bindings = {}, {}
    for model in model_list:
        if not isinstance(model, dict):
            raise ValueError("invalid model identity")
        mid = model.get("model_id")
        if not isinstance(mid, str) or not mid or mid in models:
            raise ValueError("unique nonempty model IDs required")
        if not isinstance(model.get("name"), str) or not model["name"]:
            raise ValueError("model display name required")
        if type(model.get("source_seed")) is not int:
            raise ValueError("source seed must be explicit, not inferred from a stream seed")
        _number(model.get("clean_accuracy"), "clean_accuracy")
        _, digest = _bound(manifest_path.parent, model.get("checkpoint"))
        models[mid] = model
        bindings[f"checkpoint:{mid}"] = digest
    result, all_rows = {}, {}
    for panel in PANELS:
        expected = set()
        if not isinstance(rosters[panel], list) or not rosters[panel]:
            raise ValueError(f"nonempty expected roster required: {panel}")
        for cell in rosters[panel]:
            if not isinstance(cell, dict):
                raise ValueError(f"invalid expected roster entry: {panel}")
            mid, cid = cell.get("model_id"), cell.get("cell_id")
            if not isinstance(mid, str) or mid not in models or not isinstance(cid, str) or not cid:
                raise ValueError(f"unknown model or invalid cell in expected roster: {panel}")
            if (mid, cid) in expected:
                raise ValueError(f"duplicate expected roster cell: {panel}")
            expected.add((mid, cid))
        if {mid for mid, _ in expected} != set(models):
            raise ValueError(f"expected roster omits a declared model: {panel}")
        source, digest = _bound(manifest_path.parent, manifest["panels"][panel])
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"input changed during read: {source}")
        document = _json_bytes(raw)
        if (not isinstance(document, dict) or document.get("schema") != "kbound_task1_cell_records_v1"
                or document.get("panel_id") != panel):
            raise ValueError(f"panel identity mismatch: {panel}")
        rows = document.get("records")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"per-cell actions missing: {panel}; summary-only is not regenerable")
        policies = ("kga",) if panel == "helpful_only" else ("kga", "aetta")
        keyed, grouped = {}, {mid: [] for mid in models}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("cell must be an object")
            mid, cid = row.get("model_id"), row.get("cell_id")
            if not isinstance(mid, str) or mid not in models or not isinstance(cid, str) or not cid:
                raise ValueError("cell requires known model and nonempty cell ID")
            key = (mid, cid)
            if key in keyed:
                raise ValueError(f"duplicate cell: {key}")
            if row.get("checkpoint_sha256") != bindings[f"checkpoint:{mid}"]:
                raise ValueError(f"cell checkpoint identity mismatch: {key}")
            a0, aa = _number(row.get("a0"), "a0"), _number(row.get("aa"), "aa")
            if "weight" in row and row["weight"] != 1:
                raise ValueError("only equal-cell weighting is supported")
            if "B" in row and (not isinstance(row["B"], (int, float))
                               or not math.isclose(row["B"], aa - a0, rel_tol=0, abs_tol=1e-10)):
                raise ValueError(f"stale benefit: {key}")
            if panel == "helpful_only" and aa <= a0:
                raise ValueError("helpful-only panel contains a nonpositive benefit")
            for policy in policies:
                if row.get(f"{policy}_action") not in ACTIONS:
                    raise ValueError(f"recorded {policy} action missing or invalid: {key}")
            keyed[key] = row
            grouped[mid].append(row)
        if any(not group for group in grouped.values()):
            raise ValueError(f"model missing from panel: {panel}")
        if set(keyed) != expected:
            raise ValueError(f"export does not match expected roster: {panel}")
        result[panel] = {mid: _metrics(group, policies) for mid, group in grouped.items()}
        all_rows[panel] = keyed
        bindings[f"panel:{panel}"] = digest
    loo, b6 = all_rows["mixed_loo"], all_rows["mixed_b6"]
    if loo.keys() != b6.keys():
        raise ValueError("mixed LOO/B6 cell identities differ")
    for key in loo:
        if any(loo[key][field] != b6[key][field]
               for field in ("a0", "aa", "checkpoint_sha256", "aetta_action")):
            raise ValueError(f"paired scores/checkpoint/AETTA differ between mixed panels: {key}")
    return {"schema": "kbound_task1_record_derived_table_v1",
            "claim_scope": "RECORDED_ACTION_ARITHMETIC_NOT_PROTOCOL_VALIDATION",
            "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "input_sha256": bindings, "models": models, "panels": result,
            "expected_cells": rosters, "roster_match": True,
            "warnings": ["Hash binding does not establish outcome-blindness, native fidelity, "
                         "independent training or prospective validity; separate receipts are required."]}


def _tex(text: str) -> str:
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(c, c) for c in text)


def render_table(report: dict) -> str:
    lines = [r"% Record-derived candidate. Requires separate protocol and publication review.",
             r"\begin{table*}[t]", r"\centering\small",
             r"\caption{Recorded-action comparison across source models. Helpful-only, historical LOO, "
             r"and B6 panels are separate. Values are recomputed from hash-bound per-cell records "
             r"with equal cell weights. This table alone does not authenticate the decision procedure "
             r"or establish native fidelity, population coverage or prospective validity.}",
             r"\label{tab:task1_independent_models}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lrrrrrrrrrl}\toprule",
             r"Model / policy & $n$ & Clean & $a_0$ & $a_a$ & $R_{AF}$ & $R_{AA}$ & Accuracy & Regret & $FA_u$ & A/F/Abs \\",
             r"\midrule"]
    for panel, models in report["panels"].items():
        lines.append(r"\multicolumn{11}{l}{" + _tex(panel) + r"} \\")
        for mid, row in models.items():
            model = report["models"][mid]
            for policy, counts in row["actions"].items():
                values = [model["clean_accuracy"], row["accuracy"]["always_freeze"],
                          row["accuracy"]["always_adapt"], row["regret"]["always_freeze"],
                          row["regret"]["always_adapt"], row["accuracy"][policy], row["regret"][policy]]
                formatted = " & ".join(f"{value * 100:.2f}\\%" for value in values)
                actions = "/".join(str(counts[a]) for a in ACTIONS)
                name = _tex(f"{model['name']} (source seed {model['source_seed']}) / {policy}")
                lines.append(f"{name} & {row['n']} & {formatted} & "
                             f"{row['false_adapt'][policy]['unconditional']:.4f} & {actions} " + r"\\")
        lines.append(r"\midrule")
    return "\n".join(lines + [r"\bottomrule\end{tabular}}", r"\end{table*}", ""])


def generate(manifest_path: Path, out_dir: Path) -> dict:
    report = build_table(manifest_path)
    tex = render_table(report)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "task1_table.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out_dir / "tab_task1_independent_models.tex").write_text(tex)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Reviewed per-cell input bindings")
    parser.add_argument("--out-dir", type=Path, required=True, help="New, nonexistent candidate directory")
    args = parser.parse_args(argv)
    try:
        generate(args.manifest, args.out_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Table not generated: {exc}\n")
    print(f"Record-derived candidate written to {args.out_dir}; no manuscript promotion performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
