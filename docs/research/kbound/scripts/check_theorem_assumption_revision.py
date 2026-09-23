#!/usr/bin/env python3
"""Check active-source correspondence and review freshness, never proof truth.

The maintained drivers use literal braced TeX inputs. This is a bounded source
inventory, not a TeX interpreter: dynamic/unresolved inputs and include cycles
are errors, not silently ignored files. Compilation remains a separate check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REVISION = Path("docs/research/kbound/theorem_assumption_completion")
ENVIRONMENTS = "theorem|lemma|proposition|corollary"
BEGIN = re.compile(r"\\begin\{(" + ENVIRONMENTS + r")\}(?:\[([^\]]*)\])?")
INPUT = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path outside repository: {relative}")
    return path


def mask_comments(text: str) -> str:
    """Preserve character offsets for exact statement hashes and source lines."""
    chars = list(text)
    for match in re.finditer("%", text):
        index = match.start() - 1
        slashes = 0
        while index >= 0 and text[index] == "\\":
            slashes += 1
            index -= 1
        if slashes % 2 == 0:
            end = text.find("\n", match.start())
            if end == -1:
                end = len(text)
            chars[match.start() : end] = " " * (end - match.start())
    return "".join(chars)


def inventory(root: Path, drivers: list[str]) -> tuple[list[dict], list[dict]]:
    sources: dict[str, str] = {}
    statements: dict[tuple[str, int], dict] = {}

    def visit(path: Path, tex_root: Path, stack: tuple[Path, ...]) -> None:
        relative = str(path.relative_to(root.resolve()))
        if path in stack:
            raise ValueError(f"include cycle: {relative}")
        if not path.is_file():
            raise ValueError(f"unresolved input: {relative}")
        raw = path.read_text(encoding="utf-8")
        text = mask_comments(raw)
        sources[relative] = digest(path)
        # Reject bare or macro-generated input syntax rather than omit it.
        commands = list(re.finditer(r"\\(?:input|include)\b", text))
        includes = list(INPUT.finditer(text))
        if len(commands) != len(includes):
            raise ValueError(f"unsupported input syntax: {relative}")
        for match in BEGIN.finditer(text):
            end_token = "\\end{" + match[1] + "}"
            stop = text.find(end_token, match.end())
            if stop < 0 or BEGIN.search(text, match.end(), stop):
                raise ValueError(f"unclosed or nested result: {relative}:{match.start()}")
            end = stop + len(end_token)
            labels = re.findall(r"\\label\{([^{}]+)\}", text[match.start() : end])
            if not labels:
                raise ValueError(f"unlabeled result: {relative}:{match.start()}")
            proof_start = text.find(r"\begin{proof}", end)
            proof_stop = text.find(r"\end{proof}", proof_start)
            next_result = BEGIN.search(text, end)
            if proof_start < 0 or proof_stop < 0 or (next_result and next_result.start() < proof_start):
                raise ValueError(f"missing adjacent proof for {labels[0]} in {relative}")
            proof_end = proof_stop + len(r"\end{proof}")
            statements[(relative, match.start())] = {
                "id": labels[0],
                "labels": labels,
                "environment": match[1],
                "title": match[2] or "",
                "source": {
                    "path": relative,
                    "line": raw.count("\n", 0, match.start()) + 1,
                    "end_line": raw.count("\n", 0, end) + 1,
                    "sha256": sources[relative],
                    "statement_sha256": hashlib.sha256(raw[match.start() : end].encode()).hexdigest(),
                },
                "proof_authority": {
                    "path": relative,
                    "line": raw.count("\n", 0, proof_start) + 1,
                    "end_line": raw.count("\n", 0, proof_end) + 1,
                    "proof_sha256": hashlib.sha256(raw[proof_start:proof_end].encode()).hexdigest(),
                },
            }
        for include in includes:
            target = include[1]
            if any(char in target for char in ("\\", "#", "~")):
                raise ValueError(f"dynamic input in {relative}: {target}")
            if not Path(target).suffix:
                target += ".tex"
            child = local_path(root, str(tex_root.relative_to(root.resolve()) / target))
            visit(child, tex_root, stack + (path,))

    for driver in drivers:
        path = local_path(root, driver)
        visit(path, path.parent, ())
    rows = list(statements.values())
    seen: set[str] = set()
    for row in rows:
        for label in row["labels"]:
            if label in seen:
                raise ValueError(f"duplicate active theorem label: {label}")
            seen.add(label)
    return rows, [{"path": path, "sha256": sha} for path, sha in sorted(sources.items())]


def check_register(root: Path, register: dict, assumptions: dict | None = None) -> dict:
    errors: list[str] = []
    try:
        actual, sources = inventory(root, register["scope"]["drivers"])
    except (ValueError, OSError, KeyError) as exc:
        actual, sources = [], []
        errors.append(str(exc))
    actual_by_id = {row["id"]: row for row in actual}
    expected_count = register.get("scope", {}).get("count")
    if expected_count is not None and expected_count != len(actual):
        errors.append("stale named-statement count")
    counts = {env: sum(row["environment"] == env for row in actual) for env in ENVIRONMENTS.split("|")}
    expected_counts = register.get("scope", {}).get("count_by_environment")
    if expected_counts is not None and expected_counts != counts:
        errors.append("stale environment counts")
    registered: dict[str, dict] = {}
    bindings: dict[str, str] = {}
    for source in register.get("scope", {}).get("source_files", []):
        path = source["path"]
        if path in bindings:
            errors.append(f"duplicate source binding: {path}")
        bindings[path] = source["sha256"]
        try:
            if digest(local_path(root, path)) != source["sha256"]:
                errors.append(f"stale source binding: {path}")
        except (OSError, ValueError) as exc:
            errors.append(f"invalid source binding: {path}: {exc}")
    for row in register.get("statements", []):
        identifier = row["id"]
        if identifier in registered:
            errors.append(f"duplicate registration: {identifier}")
        registered[identifier] = row
        if identifier not in actual_by_id:
            errors.append(f"registered result absent from active source: {identifier}")
            continue
        active = actual_by_id[identifier]
        for field in ("labels", "environment", "title"):
            if row.get(field) != active[field]:
                errors.append(f"{identifier}: mismatched {field}")
        for field in ("path", "sha256", "statement_sha256", "line", "end_line"):
            if row["source"].get(field) != active["source"][field]:
                errors.append(f"{identifier}: stale statement bytes or source ({field})")
        for field in ("path", "proof_sha256", "line", "end_line"):
            if row.get("proof_authority", {}).get(field) != active["proof_authority"][field]:
                errors.append(f"{identifier}: mismatched proof bytes or source ({field})")
        for authority, label in ((row["source"], "statement"), (row.get("proof_authority", {}), "proof")):
            if authority.get("path") not in bindings:
                errors.append(f"{identifier}: unbound {label} reference")
        for field in ("assumptions", "formal_correspondence", "manual_review"):
            if not row.get(field):
                errors.append(f"{identifier}: missing review field {field}")
    for identifier in actual_by_id.keys() - registered.keys():
        errors.append(f"unregistered active result: {identifier}")

    def check_lean_refs(value):
        if isinstance(value, dict):
            if "name" in value and str(value.get("path", "")).endswith(".lean"):
                path = value["path"]
                if path not in bindings:
                    errors.append(f"unbound Lean reference: {path}: {value['name']}")
                try:
                    source = local_path(root, path).read_text()
                    name = value["name"].removeprefix("KBound.").split(".")[-1]
                    # Declaration existence only; kernel evidence is a separate audit.
                    if not re.search(r"\b(?:theorem|lemma|def)\s+" + re.escape(name) + r"\b", source):
                        errors.append(f"missing Lean declaration: {path}: {value['name']}")
                except (OSError, ValueError) as exc:
                    errors.append(f"invalid Lean reference: {exc}")
            for item in value.values():
                check_lean_refs(item)
        elif isinstance(value, list):
            for item in value:
                check_lean_refs(item)

    check_lean_refs(register)

    assumption_count = 0
    if assumptions is not None:
        assumption_ids = [row["id"] for row in assumptions.get("assumptions", [])]
        assumption_count = len(assumption_ids)
        if not assumption_ids or len(assumption_ids) != len(set(assumption_ids)):
            errors.append("missing or duplicate assumption ids")
        for study in assumptions.get("studies", []):
            for identifier in study.get("assumption_ids", []):
                if identifier not in assumption_ids:
                    errors.append(f"{study['id']}: unknown assumption {identifier}")

        def check_refs(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    try:
                        if digest(local_path(root, value["path"])) != value["sha256"]:
                            errors.append(f"stale assumption reference: {value['path']}")
                        if value.get("anchor") and value["anchor"] not in local_path(root, value["path"]).read_text():
                            errors.append(f"missing assumption anchor: {value['path']}: {value['anchor']}")
                    except (OSError, ValueError) as exc:
                        errors.append(f"invalid assumption reference: {exc}")
                for item in value.values():
                    check_refs(item)
            elif isinstance(value, list):
                for item in value:
                    check_refs(item)

        check_refs(assumptions)
    return {
        "schema": "kbound-theorem-assumption-correspondence/1",
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "named_statement_count": len(actual),
        "assumption_count": assumption_count,
        "active_source_files": sources,
        "mathematical_correctness_established": False,
        "empirical_assumptions_established": False,
        "meaning": "Source inventory, reference existence, and reviewed-byte freshness only; mathematical and empirical judgments are separate.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--register-directory", type=Path, default=REVISION,
                        help="Repository-relative directory of versioned theorem and assumption registers")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    register_path = local_path(root, str(args.register_directory / "theorem_register.json"))
    assumptions_path = local_path(root, str(args.register_directory / "assumption_register.json"))
    report = check_register(root, json.loads(register_path.read_text()), json.loads(assumptions_path.read_text()))
    report["registers"] = [
        {"path": str(path.relative_to(root)), "sha256": digest(path)} for path in (register_path, assumptions_path)
    ]
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(json.dumps({key: report[key] for key in ("status", "named_statement_count", "assumption_count", "errors")}))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
