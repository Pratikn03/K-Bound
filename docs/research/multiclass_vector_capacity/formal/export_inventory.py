#!/usr/bin/env python3
"""Build this isolated Lean library and export fail-closed theorem evidence.

The compiled namespace audit is authoritative for transitive axioms. The source
scan and exact command inventory are additional guards, not proof substitutes.
No dependencies are fetched: all pinned package directories must already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
MATHLIB_REV = "5e932f97dd25535344f80f9dd8da3aab83df0fe6"
PROOF_MODULES = {
    "Basic": "MulticlassVectorCapacity",
    "Benefit": "MulticlassVectorCapacity",
    "ObservableFiber": "MulticlassVectorCapacity",
    "SignCapacity": "MulticlassVectorCapacity",
    "Examples": "MulticlassVectorCapacity.ThreeClass",
    "EdgeCases": "MulticlassVectorCapacity.EdgeCases",
}


class AuditFailure(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resident_bytes(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise AuditFailure(f"Required file is not regular: {path}")
    if getattr(info, "st_flags", 0) & getattr(stat, "SF_DATALESS", 0x40000000):
        raise AuditFailure(f"Required file is cloud-only: {path}")
    return path.read_bytes()


def strip_comments(text: str) -> str:
    """Preserve line positions while removing nested Lean comments.

    This intentionally handles comment syntax only. Proof sources in this
    initial slice contain no quoted strings; additions with strings require a
    reviewed scanner extension rather than silently hiding interpolation code.
    """
    result = []
    i = 0
    depth = 0
    line_comment = False
    while i < len(text):
        if line_comment:
            if text[i] == "\n":
                line_comment = False
                result.append("\n")
            else:
                result.append(" ")
            i += 1
        elif depth:
            if text.startswith("/-", i):
                depth += 1
                result.extend("  ")
                i += 2
            elif text.startswith("-/", i):
                depth -= 1
                result.extend("  ")
                i += 2
            else:
                result.append("\n" if text[i] == "\n" else " ")
                i += 1
        elif text.startswith("--", i):
            line_comment = True
            result.extend("  ")
            i += 2
        elif text.startswith("/-", i):
            depth = 1
            result.extend("  ")
            i += 2
        else:
            if text[i] == '"':
                raise AuditFailure("Quoted strings in a proof source require scanner review")
            result.append(text[i])
            i += 1
    if depth:
        raise AuditFailure("Unterminated Lean block comment")
    return "".join(result)


def parse_messages(text: str, commands: dict[int, tuple[str, str]]):
    expected = {(kind, name) for kind, name in commands.values()}
    if len(expected) != len(commands):
        raise AuditFailure("Duplicate registered audit command")
    result = {}
    seen = set()
    namespace_count = None
    dependencies = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            message = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AuditFailure("Non-JSON compiler output") from exc
        if message.get("severity") != "information":
            raise AuditFailure(f"Non-informational compiler output: {message}")
        data = message.get("data")
        if not isinstance(data, str):
            raise AuditFailure("Compiler message has no text")
        marker = re.fullmatch(r"MVC_NAMESPACE_AUDIT_PASS ([1-9][0-9]*)", data)
        if marker:
            if namespace_count is not None:
                raise AuditFailure("Duplicate namespace audit marker")
            namespace_count = int(marker.group(1))
            continue
        dependency = re.fullmatch(r"MVC_DIRECT_DEPENDENCIES (\S+) (\[.*\])", data)
        if dependency:
            dependency_name = dependency.group(1)
            try:
                values = json.loads(dependency.group(2))
            except json.JSONDecodeError as exc:
                raise AuditFailure("Invalid compiled dependency array") from exc
            if (dependency_name in dependencies or not dependency_name.startswith("MulticlassVectorCapacity.")
                    or not isinstance(values, list)
                    or any(not isinstance(x, str) or not x.startswith("MulticlassVectorCapacity.") for x in values)
                    or len(values) != len(set(values))):
                raise AuditFailure("Invalid or duplicate compiled dependency report")
            dependencies[dependency_name] = sorted(values)
            continue
        line = message.get("pos", {}).get("line")
        if line not in commands:
            raise AuditFailure(f"Unregistered compiler message at line {line}")
        kind, name = commands[line]
        key = (kind, name)
        if key in seen:
            raise AuditFailure(f"Duplicate compiler report: {key}")
        seen.add(key)
        item = result.setdefault(name, {})
        if kind == "type":
            match = re.fullmatch(r"@?" + re.escape(name) + r"(?:\.\{[^}]*\})?\s*:\s*(.+)", data, re.S)
            if not match or "⋯" in data or "?m." in data or "..." in data:
                raise AuditFailure(f"Missing, mismatched, or elided type for {name}")
            item["printed_declaration"] = data
            item["printed_type"] = match.group(1)
        else:
            match = re.fullmatch(re.escape(f"'{name}' depends on axioms:") + r"\s*\[(.*?)\]", data, re.S)
            if match:
                without_universes = re.sub(r"\.\{[^}]*\}", "", match.group(1))
                axioms = [x.strip() for x in without_universes.split(",") if x.strip()]
                if len(axioms) != len(set(axioms)) or not set(axioms) <= ALLOWED_AXIOMS:
                    raise AuditFailure(f"Unapproved or duplicate axiom for {name}: {axioms}")
            elif data == f"'{name}' does not depend on any axioms":
                axioms = []
            else:
                raise AuditFailure(f"Unrecognized axiom report for {name}")
            item["transitive_axioms"] = sorted(axioms)
            item["raw_axiom_report"] = data
    if seen != expected or namespace_count is None:
        raise AuditFailure(f"Incomplete compiler audit; missing {sorted(expected - seen)}")
    if (len(dependencies) != namespace_count or not set(result) <= set(dependencies)
            or any(not set(values) <= set(dependencies) for values in dependencies.values())):
        raise AuditFailure("Compiled dependency inventory is incomplete")
    if any(set(item) != {"printed_declaration", "printed_type", "transitive_axioms", "raw_axiom_report"}
           for item in result.values()):
        raise AuditFailure("A declaration lacks its type or axiom report")
    for name, item in result.items():
        item["local_direct_dependencies"] = dependencies[name]
    return result, namespace_count


def source_inventory():
    declarations = {}
    proof_paths = []
    for module, prefix in PROOF_MODULES.items():
        relative = Path("MulticlassVectorCapacity") / f"{module}.lean"
        path = ROOT / relative
        clean = strip_comments(resident_bytes(path).decode())
        if re.search(r"\b(?:sorry|admit|axiom|unsafe|partial)\b", clean):
            raise AuditFailure(f"Disallowed proof-source token: {relative}")
        for match in re.finditer(r"^theorem ([A-Za-z][A-Za-z0-9_]*)\b", clean, re.M):
            name = f"{prefix}.{match.group(1)}"
            if name in declarations:
                raise AuditFailure(f"Duplicate theorem: {name}")
            declarations[name] = {"source_path": relative.as_posix(),
                                  "source_line": clean.count("\n", 0, match.start()) + 1,
                                  "module": f"MulticlassVectorCapacity.{module}"}
        proof_paths.append(path)
    # Any new proof module must be deliberately added to the audited boundary.
    allowed = {p.name for p in proof_paths} | {"Audit.lean", "Regression.lean"}
    if {p.name for p in (ROOT / "MulticlassVectorCapacity").glob("*.lean")} != allowed:
        raise AuditFailure("Unexpected or missing module in formal namespace directory")
    return declarations


def audit_commands():
    commands = {}
    source = resident_bytes(ROOT / "MulticlassVectorCapacity/Audit.lean").decode()
    for line, text in enumerate(source.splitlines(), 1):
        checked = re.fullmatch(r"#check @([A-Za-z0-9_.]+)", text)
        axioms = re.fullmatch(r"#print axioms ([A-Za-z0-9_.]+)", text)
        if checked or axioms:
            commands[line] = ("type" if checked else "axioms", (checked or axioms).group(1))
    return commands


def run_command(command: list[str], log: Path | None = None) -> str:
    process = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=300, check=False)
    if log is not None:
        log.write_text(process.stdout)
    if process.returncode:
        raise AuditFailure(f"Command failed ({process.returncode}): {command}; log={log}")
    return process.stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", default=shutil.which("lake"))
    parser.add_argument("--output-dir", default="verification")
    args = parser.parse_args(argv)
    if not args.lake:
        raise AuditFailure("Lake executable is required")
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise AuditFailure("Verification output must remain inside this isolated formal subtree")
    output.mkdir(parents=True, exist_ok=True)
    declared = source_inventory()
    commands = audit_commands()
    expected_pairs = {(kind, name) for name in declared for kind in ("type", "axioms")}
    if set(commands.values()) != expected_pairs or len(commands) != len(expected_pairs):
        raise AuditFailure("Source theorem set and paired audit commands differ")
    source_paths = [ROOT / item["source_path"] for item in declared.values()]
    source_paths += [ROOT / "MulticlassVectorCapacity/Audit.lean", ROOT / "MulticlassVectorCapacity/Regression.lean",
                     ROOT / "MulticlassVectorCapacity.lean", ROOT / "lakefile.lean", ROOT / "lean-toolchain",
                     ROOT / "lake-manifest.json", Path(__file__).resolve()]
    source_paths += sorted((ROOT / "tests").glob("*.py"))
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(resident_bytes(p)) for p in sorted(set(source_paths))}
    manifest = json.loads(resident_bytes(ROOT / "lake-manifest.json"))
    dependencies = []
    for package in manifest["packages"]:
        path = ROOT / manifest["packagesDir"] / package["name"]
        if not path.is_dir():
            raise AuditFailure(f"Pinned dependency is absent; no network fetch attempted: {package['name']}")
        actual = run_command(["git", "--no-optional-locks", "-C", str(path), "rev-parse", "HEAD"]).strip()
        if actual != package["rev"]:
            raise AuditFailure(f"Dependency revision mismatch: {package['name']}")
        dependencies.append({"name": package["name"], "revision": actual})
    if next(p["revision"] for p in dependencies if p["name"] == "mathlib") != MATHLIB_REV:
        raise AuditFailure("Wrong Mathlib pin")
    toolchain = resident_bytes(ROOT / "lean-toolchain").decode().strip()
    if toolchain != "leanprover/lean4:v4.29.1":
        raise AuditFailure("Wrong Lean toolchain pin")
    lean_version = run_command([args.lake, "--no-cache", "env", "lean", "--version"]).strip()
    if "version 4.29.1," not in lean_version:
        raise AuditFailure("Active Lean executable does not match pin")
    build_command = [args.lake, "--no-cache", "--wfail", "build"]
    build_log = output / "lake-build.log"
    run_command(build_command, build_log)
    axiom_command = [args.lake, "--no-cache", "env", "lean", "MulticlassVectorCapacity/Audit.lean", "--json"]
    axiom_log = output / "axiom-types.jsonl"
    text = run_command(axiom_command, axiom_log)
    parsed, namespace_count = parse_messages(text, commands)
    if set(parsed) != set(declared):
        raise AuditFailure("Compiled audit did not report the exact source theorem set")
    for relative, before in source_hashes.items():
        if sha256(resident_bytes(ROOT / relative)) != before:
            raise AuditFailure(f"Source changed during verification: {relative}")
    logs = {"build_log": {"path": build_log.relative_to(ROOT).as_posix(), "sha256": sha256(resident_bytes(build_log))},
            "axiom_log": {"path": axiom_log.relative_to(ROOT).as_posix(), "sha256": sha256(resident_bytes(axiom_log))}}
    theorems = []
    for name in sorted(declared):
        item = dict(declared[name], declaration=name, verification_status="VERIFIED", **parsed[name], **logs)
        item["source_sha256"] = source_hashes[item["source_path"]]
        olean = ROOT / ".lake/build/lib/lean" / Path(*item["module"].split(".")).with_suffix(".olean")
        item["olean_path"] = olean.relative_to(ROOT).as_posix()
        item["olean_sha256"] = sha256(resident_bytes(olean))
        theorems.append(item)
    payload = {"schema_version": 1, "namespace": "MulticlassVectorCapacity", "verification_status": "VERIFIED_FINITE_SLICE",
               "clean_checkout_portability": "NOT_RUN", "theorem_count": len(theorems),
               "compiled_namespace_declarations_checked": namespace_count,
               "allowed_axioms": sorted(ALLOWED_AXIOMS), "lean_toolchain": toolchain, "lean_version": lean_version,
               "dependency_scope": "Local namespace constants directly present in each compiled theorem type and proof expression; not a transitive or full external dependency graph.",
               "dependencies": dependencies, "source_hashes": source_hashes,
               "build_command": build_command, "axiom_command": axiom_command, **logs, "theorems": theorems,
               "excluded_claims": ["T2 general uniform rank/row-space criterion", "T3 admissible-supplement minimality",
                                   "T5 unconditional zero-crossing or decision-error claim", "T6–T9 label complexity and statistical controller claims",
                                   "novelty", "promotion into K-Bound"]}
    destination = output / "theorem_inventory.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["verification_status"], "theorems": len(theorems),
                      "compiled_namespace_declarations_checked": namespace_count,
                      "inventory": str(destination), "inventory_sha256": sha256(resident_bytes(destination))}))


if __name__ == "__main__":
    try:
        main()
    except (AuditFailure, OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"AUDIT FAILED: {error}") from error
