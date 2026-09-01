"""Check local source/type/statement bindings without claiming a release seal.

--record-reviewed-local-snapshot is an explicit authoring step after review.
The default only verifies; it never repairs a stale receipt or runs a compiler.
The formal inventory records the actual separate Lean invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOT = "reports/initial_local_snapshot.json"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
INITIAL_STATUSES = {
    "T1": "LEAN_VERIFIED_LOCAL_IDENTITY",
    "T2": "NARROWED_CONJECTURE_NOT_FORMALIZED",
    "T3": "ORIGINAL_UNSPECIFIED_ADMISSIBLE_EQUALITY_WITHDRAWN_EXACT_COUNTEREXAMPLE",
    "T4": "LEAN_VERIFIED_LOCAL_SCALAR_GUARD",
    "T5": "ORIGINAL_IMPLICATION_REFUTED_LEAN_COUNTEREXAMPLE",
    "T6": "CONJECTURE_NOT_FORMALIZED_OR_IMPLEMENTED",
    "T7": "CONJECTURE_NOT_FORMALIZED_OR_IMPLEMENTED",
    "T8": "EXACT_NONVACUITY_CANDIDATE_RATE_NOT_FORMALIZED",
    "T9": "CONJECTURE_NOT_FORMALIZED_OR_IMPLEMENTED",
}
# Deliberately reviewed milestone contract, not inferred from arbitrary ledger text.
# A later research milestone must update and review this contract explicitly.
INITIAL_DECLARATIONS = {
    "T1": {
        "cost_benefit_identity", "expectedCost_mem_unitInterval", "benefit_mem_Icc",
        "positive_benefit_iff_lower_cost", "negative_benefit_iff_higher_cost",
    },
    "T4": {
        "lowerBenefit_pos_iff_uniform_margin", "upperBenefit_neg_iff_uniform_margin",
        "empty_fiber_no_certificate", "adapt_decision_iff", "freeze_decision_iff",
        "abstain_decision_iff", "adapt_decision_sound", "freeze_decision_sound",
        "ThreeClass.identified_interval", "ThreeClass.not_point_identified", "ThreeClass.strict_adapt",
    },
    "T5": {
        "ThreeClass.surviving_null_contrast_without_sign_ambiguity",
        "ThreeClass.no_negative_world", "ThreeClass.candidate_not_pointwise_dominant",
    },
}


class VerificationFailure(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationFailure(message)


def payload_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_file(relative: str, base: Path = ROOT) -> Path:
    require(isinstance(relative, str) and bool(relative), "Missing local path")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Path must stay inside the isolated track")
    resolved = (base / path).resolve()
    require(resolved.is_relative_to(ROOT), "Local proof/source path escapes the isolated track")
    require(resolved.is_file(), f"Missing file: {relative}")
    return resolved


def resident_bytes(path: Path) -> bytes:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"Not a regular file: {path}")
    require(not getattr(info, "st_flags", 0) & getattr(stat, "SF_DATALESS", 0x40000000), f"Cloud-only file: {path}")
    return path.read_bytes()


def _import_exporter():
    specification = importlib.util.spec_from_file_location("mvc_inventory_exporter", ROOT / "formal/export_inventory.py")
    require(specification is not None and specification.loader is not None, "Cannot load formal inventory validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def check_formal_inventory(ledger: dict) -> dict:
    evidence = ledger["formal_evidence"]
    require(evidence["status"] == "VERIFIED_FINITE_SLICE_LOCAL", "Formal audit is not complete for this local slice")
    path = local_file(evidence["inventory"])
    inventory = json.loads(resident_bytes(path))
    require(inventory["verification_status"] == "VERIFIED_FINITE_SLICE", "Inventory is not verified")
    require(inventory["clean_checkout_portability"] == "NOT_RUN", "This initial receipt must not promote a fresh-checkout result")
    require(set(inventory["allowed_axioms"]) == ALLOWED_AXIOMS, "Wrong axiom allowlist")
    exporter = _import_exporter()
    source_theorems = exporter.source_inventory()
    commands = exporter.audit_commands()
    parsed, namespace_count = exporter.parse_messages(resident_bytes(local_file(inventory["axiom_log"]["path"], ROOT / "formal")).decode(), commands)
    require(namespace_count == inventory["compiled_namespace_declarations_checked"], "Namespace audit count drift")
    records = inventory["theorems"]
    by_name = {item["declaration"]: item for item in records}
    require(len(by_name) == len(records) == inventory["theorem_count"], "Duplicate or missing theorem entries")
    require(set(by_name) == set(parsed) == set(source_theorems), "Ledger/source/compiler theorem set mismatch")
    expected_sources = {item["source_path"] for item in source_theorems.values()} | {
        "MulticlassVectorCapacity/Audit.lean", "MulticlassVectorCapacity/Regression.lean",
        "MulticlassVectorCapacity.lean", "lakefile.lean", "lean-toolchain", "lake-manifest.json",
        "export_inventory.py",
    } | {path.relative_to(ROOT / "formal").as_posix() for path in (ROOT / "formal/tests").glob("*.py")}
    require(set(inventory["source_hashes"]) == expected_sources, "Formal source/configuration binding coverage is incomplete")
    require(inventory["lean_toolchain"] == resident_bytes(ROOT / "formal/lean-toolchain").decode().strip(), "Toolchain receipt differs from the pinned source")
    dependency_manifest = json.loads(resident_bytes(ROOT / "formal/lake-manifest.json"))
    expected_dependencies = {item["name"]: item["rev"] for item in dependency_manifest["packages"]}
    actual_dependencies = {item["name"]: item["revision"] for item in inventory["dependencies"]}
    require(len(actual_dependencies) == len(inventory["dependencies"]) and actual_dependencies == expected_dependencies, "Dependency receipt differs from the pinned manifest")
    for relative, expected in inventory["source_hashes"].items():
        require(digest(resident_bytes(local_file(relative, ROOT / "formal"))) == expected, f"Stale formal source: {relative}")
    for key in ("build_log", "axiom_log"):
        record = inventory[key]
        require(digest(resident_bytes(local_file(record["path"], ROOT / "formal"))) == record["sha256"], f"Stale formal {key}")
    seen_oleans = {}
    types = {}
    for name, item in by_name.items():
        require(item["verification_status"] == "VERIFIED", f"Unverified declaration: {name}")
        require(set(item["transitive_axioms"]) <= ALLOWED_AXIOMS, f"Unapproved axiom: {name}")
        for key in ("build_log", "axiom_log"):
            require(item[key] == inventory[key], f"Theorem log binding differs from the audited top-level receipt: {name}/{key}")
        for key in ("printed_type", "printed_declaration", "transitive_axioms", "raw_axiom_report", "local_direct_dependencies"):
            require(item[key] == parsed[name][key], f"Compiler report mismatch for {name}/{key}")
        require(item["source_path"] == source_theorems[name]["source_path"], f"Wrong source binding: {name}")
        require(item["source_line"] == source_theorems[name]["source_line"], f"Wrong source line: {name}")
        require(item["source_sha256"] == inventory["source_hashes"][item["source_path"]], f"Wrong source hash: {name}")
        require(item["module"] == source_theorems[name]["module"], f"Wrong compiled module binding: {name}")
        expected_olean = (Path(".lake/build/lib/lean") / Path(*source_theorems[name]["module"].split(".")).with_suffix(".olean")).as_posix()
        relative = item["olean_path"]
        require(relative == expected_olean, f"Not the compiled .olean for the source module: {name}")
        if relative not in seen_oleans:
            seen_oleans[relative] = digest(resident_bytes(local_file(relative, ROOT / "formal")))
        require(seen_oleans[relative] == item["olean_sha256"], f"Stale compiled module: {name}")
        types[name] = digest(item["printed_declaration"].encode())
    return {"inventory_sha256": digest(resident_bytes(path)), "theorems": len(records), "compiled_namespace_declarations_checked": namespace_count, "type_sha256": types, "olean_sha256": seen_oleans}


def check_claim_ledger(ledger: dict, compiled_names: set[str]) -> dict[str, str]:
    require(ledger["schema"] == "mvc.theorem-ledger.v1", "Unknown ledger schema")
    claims = ledger["claims"]
    require(len(claims) == 9 and {claim["id"] for claim in claims} == {f"T{i}" for i in range(1, 10)}, "Claim IDs are incomplete or duplicated")
    require(ledger["research_gates"]["integration_into_current_kbound"] == "NOT_PERMITTED", "No integration is authorized by this initial milestone")
    require(ledger["research_gates"]["complete_T1_to_T5_verified_novel_foundation"] == "NOT_MET", "Initial slice is not the novel foundation")
    require(ledger["research_gates"]["complete_T1_to_T9"] == "NOT_MET", "Initial slice is not the full program")
    result = {}
    for claim in claims:
        require(claim["promotion_status"] == "NOT_PROMOTED", f"Unsupported promotion of {claim['id']}")
        require(bool(claim["statement"]) and bool(claim["scope"]) and bool(claim["assumptions"]), f"Unscoped claim: {claim['id']}")
        require(claim["mathematical_status"] == INITIAL_STATUSES[claim["id"]], f"Unsupported initial milestone status for {claim['id']}")
        expected = {f"MulticlassVectorCapacity.{name}" for name in INITIAL_DECLARATIONS.get(claim["id"], set())}
        require(set(claim["lean_declarations"]) == expected and len(claim["lean_declarations"]) == len(expected), f"Incorrect exact theorem binding for {claim['id']}")
        require(set(claim["lean_declarations"]) <= compiled_names, f"Missing compiled declaration for {claim['id']}")
        result[claim["id"]] = digest(payload_bytes(claim))
    return result


def check_mutation_receipt(receipt: dict) -> dict:
    probes = receipt["probes"]
    require(isinstance(probes, list) and bool(probes), "Missing Lean mutation controls")
    require(len({probe["name"] for probe in probes}) == len(probes), "Duplicate Lean mutation control")
    positives = negatives = 0
    for probe in probes:
        require(isinstance(probe["source_text"], str) and bool(probe["source_text"]), "Missing durable mutation source")
        require(digest(probe["source_text"].encode()) == probe["source_sha256"], "Mutation source hash mismatch")
        require(type(probe["expected_exit"]) is int and probe["expected_exit"] in (0, 1), "Invalid expected compiler exit")
        require(probe["actual_exit"] == probe["expected_exit"], "Mutation compiler result did not match expectation")
        require(digest(resident_bytes(local_file(probe["log"], ROOT / "formal"))) == probe["log_sha256"], "Mutation log hash mismatch")
        if probe["expected_exit"] == 0:
            require(probe["status"] == "PASS", "Positive control did not pass")
            positives += 1
        else:
            require(probe["status"] == "REJECTED_AS_REQUIRED", "False control was not rejected")
            negatives += 1
    require(positives == receipt["positive_controls"] == 1 and negatives == receipt["rejected_mutations"] == 5, "Incomplete initial Lean mutation controls")
    return {"positive_controls": positives, "rejected_false_controls": negatives, "durable_sources_verified": len(probes)}


def source_files() -> list[Path]:
    answer = []
    for current, directories, files in os.walk(ROOT, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in {".lake", "__pycache__", ".git"})
        require(not any((Path(current) / name).is_symlink() for name in directories), "Unexpected source/report directory symlink")
        for name in sorted(files):
            path = Path(current) / name
            if path.relative_to(ROOT).as_posix() == SNAPSHOT or name == ".DS_Store" or name.startswith("._"):
                continue
            require(not path.is_symlink(), f"Unexpected source/report symlink: {path}")
            answer.append(path)
    return sorted(answer)


def current_snapshot() -> dict:
    ledger = json.loads(resident_bytes(ROOT / "theorem_ledger.json"))
    formal = check_formal_inventory(ledger)
    statements = check_claim_ledger(ledger, set(formal["type_sha256"]))
    mutations = check_mutation_receipt(json.loads(resident_bytes(local_file(ledger["formal_evidence"]["mutation_receipt"]))))
    source = ledger["supplied_specification"]
    spec = resident_bytes(local_file(source["path"]))
    require(len(spec) == source["stored_bytes"] and digest(spec) == source["stored_sha256"], "Supplied design copy changed")
    require(spec.endswith(b"\n") and digest(spec[:-1]) == source["original_attachment_sha256"], "Attachment normalization mismatch")
    hashes = {path.relative_to(ROOT).as_posix(): digest(resident_bytes(path)) for path in source_files()}
    return {
        "schema": "mvc.local-snapshot.v1",
        "status": "REVIEWED_LOCAL_SOURCE_AND_STATEMENT_BINDINGS_NOT_RELEASE_SEAL",
        "files": len(hashes),
        "sha256": hashes,
        "claim_record_sha256": statements,
        "formal": formal,
        "lean_mutation_controls": mutations,
        "fresh_checkout": "NOT_RUN",
        "git_source_freeze": "NOT_PERFORMED",
        "novelty": "UNRESOLVED_PARTIAL_COLLISION",
        "program_success_level": "NONE_OF_THE_DESIGN_SUCCESS_LEVELS_ESTABLISHED",
        "semantic_review": "Human review plus exact text/type/source binding, not automated natural-language theorem equivalence",
        "program_or_paper_promotion": "NOT_PERMITTED",
    }


def check_snapshot(current: dict, recorded: dict) -> None:
    require(current == recorded, "Local source, claim wording, compiled type, artifact, or scope changed; review and explicitly re-record, never auto-repair")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-reviewed-local-snapshot", action="store_true")
    args = parser.parse_args()
    current = current_snapshot()
    path = ROOT / SNAPSHOT
    if args.record_reviewed_local_snapshot:
        path.write_bytes(payload_bytes(current))
        mode = "record-reviewed-local-snapshot"
    else:
        check_snapshot(current, json.loads(resident_bytes(path)))
        mode = "check"
    print(json.dumps({"status": "PASS", "mode": mode, "files": current["files"], "claim_ids": len(current["claim_record_sha256"]), "lean_theorems": current["formal"]["theorems"], "scope": "local bindings only; not a clean-checkout gate, release seal, or novelty finding"}))


if __name__ == "__main__":
    try:
        main()
    except (VerificationFailure, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"LOCAL VERIFICATION FAILED: {exc}")
