"""Read-only five-artifact package checks, not a source seal or scientific verdict.

The reviewed, versioned contract is the trust anchor, not an authenticity signature.
Poppler is required. Static path checks do not protect against concurrent mutation.
Privacy checks cover extracted PDF text/metadata and OOXML; not arbitrary steganography.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from urllib.parse import unquote, urlsplit

PREFIX = "docs/research/kbound/"
REVISION_PATH = PREFIX + "paper/release/manuscript_revision.json"
SOURCE_PATHS = frozenset(PREFIX + p for p in (
    "kbound_abstract_core.tex", "kbound_full_report.tex", "kbound_main_with_appendix.tex",
    "kbound_submission_body.tex", "kbound_submission_supplement.tex",
    "paper/full_report/evidence_protocol_atlas.tex", "paper/full_report/formal_reproducibility_atlas.tex",
    "paper/full_report/theory_exposition.tex", "paper/generated/empirical_evidence_numbers.tex",
    "paper/sections/officehome_mechanism_check.tex", "paper/sections/proof_traceability.tex",
    "paper/sections/theory_certificate.tex",
))
ROLES = {
    "main_pdf": ("KBound_Main_22pages.pdf", "short_main", "pdf"),
    "combined_pdf": ("KBound_Main_Appendices_49pages.pdf", "main_with_appendix", "pdf"),
    "full_pdf": ("KBound_Full_Report_105pages.pdf", "full_report", "pdf"),
    "main_docx": ("KBound_Main_Editable.docx", "short_main", "docx"),
    "combined_docx": ("KBound_Main_Appendices_Editable.docx", "main_with_appendix", "docx"),
}
DOCUMENTS = {
    "short_main": (False, "kbound_short_main.pdf", "Named compact main paper"),
    "main_with_appendix": (False, "kbound_main_with_appendix.pdf", "Main paper with integrated appendices"),
    "full_report": (False, "kbound_full_report.pdf", "Named full technical report"),
    "short_supplement": (False, "kbound_short_supplement.pdf", "Named standalone supplement"),
    "tmlr": (True, "kbound_tmlr.pdf", "Anonymous integrated TMLR review manuscript"),
}
AUTHORITY_PATHS = (
    "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
    "experiments/kbound/results/reconciled_panels_v1/source_manifest.json",
    PREFIX + "paper/generated/cct20_release_manifest.json",
    PREFIX + "paper/generated/current_policy_interval_diagnostics.json",
    PREFIX + "paper/generated/cct20_numbers.tex",
    PREFIX + "paper/generated/cct20_reporting_numbers.tex",
)
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
PRIVATE = re.compile(r"(?:/(?:Users|Volumes|home|private|tmp)/|file:(?://)?|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\[A-Za-z0-9_.-]+\\)", re.I)
SECRET = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b")


class AuditError(ValueError):
    """Only controlled diagnostics, never parser-supplied document contents."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def keys(value, expected, context):
    require(type(value) is dict and set(value) == set(expected), f"{context}: invalid object keys")


def hash_string(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256 type or encoding")


def positive_int(value):
    require(type(value) is int and 0 < value < 100_000_000, "invalid positive integer")


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("non-finite JSON number")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def safe_path(path, *, directory=False):
    """Reject lexical traversal and every symlink component before resolution/read."""
    path = Path(path)
    require(".." not in path.parts, "parent traversal is not allowed")
    absolute = Path(os.path.abspath(path))
    for component in (*reversed(absolute.parents), absolute):
        require(not component.is_symlink(), "symlink path is not allowed")
    mode = absolute.stat().st_mode
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode), "expected directory" if directory else "expected regular file")
    return absolute


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def privacy(text):
    require(not PRIVATE.search(unquote(text)), "local/private path detected (value withheld)")
    require(not SECRET.search(text), "credential-like material detected (value withheld)")


def validate_contract(c):
    keys(c, {"schema", "revision_id", "scope", "revision_authority", "reviewed_manuscript_inputs", "artifacts"}, "contract")
    require(c["schema"] == "kbound_publication_package_v1", "unsupported publication schema")
    require(c["scope"] == "READ_ONLY_PRE_RELEASE_PACKAGE_NOT_SOURCE_SEAL_OR_CI", "wrong contract scope")
    require(type(c["revision_id"]) is str and re.fullmatch(r"KBOUND-REVISION-\d{4}-\d{2}-\d{2}", c["revision_id"]), "invalid revision ID")
    keys(c["revision_authority"], {"path", "sha256"}, "revision authority")
    require(c["revision_authority"]["path"] == REVISION_PATH, "non-allowlisted revision authority")
    hash_string(c["revision_authority"]["sha256"])
    keys(c["reviewed_manuscript_inputs"], SOURCE_PATHS, "reviewed manuscript inputs")
    for value in c["reviewed_manuscript_inputs"].values():
        hash_string(value)
    keys(c["artifacts"], ROLES, "artifact roles")
    for role, (filename, document, kind) in ROLES.items():
        row = c["artifacts"][role]
        counts = {"pages"} if kind == "pdf" else {"omml", "tables", "media"}
        keys(row, {"filename", "document", "kind", "sha256", "bytes"} | counts, role)
        require((row["filename"], row["document"], row["kind"]) == (filename, document, kind), f"{role}: role/path/type mismatch")
        hash_string(row["sha256"])
        for key in counts | {"bytes"}:
            positive_int(row[key])


def validate_revision(r, contract):
    keys(r, {"schema", "revision_id", "revision_date", "base_commit", "canonical_panel_sha256", "historical_evidence_manifest_sha256", "status", "documents"}, "manuscript revision")
    require(r["schema"] == "kbound_manuscript_revision_v1" and r["revision_id"] == contract["revision_id"], "revision identity mismatch")
    require(r["status"] == "MANUSCRIPT_REVISION_NOT_A_NEW_SOURCE_SEAL", "wrong revision scope")
    require(type(r["revision_date"]) is str and re.fullmatch(r"[A-Za-z]+ \d{1,2}, \d{4}", r["revision_date"]), "invalid revision date")
    require(type(r["base_commit"]) is str and re.fullmatch(r"[0-9a-f]{12,40}", r["base_commit"]), "invalid base commit")
    for key in ("canonical_panel_sha256", "historical_evidence_manifest_sha256"):
        hash_string(r[key])
    keys(r["documents"], DOCUMENTS, "revision documents")
    for key, (anonymous, filename, role) in DOCUMENTS.items():
        doc = r["documents"][key]
        keys(doc, {"anonymous", "output_filename", "role"}, "revision document")
        require(type(doc["anonymous"]) is bool and (doc["anonymous"], doc["output_filename"], doc["role"]) == (anonymous, filename, role), "invalid document identity")


def run_external(name, *args):
    result = subprocess.run([name, *map(str, args)], capture_output=True, text=True, timeout=60, check=False)
    # Do not return parser diagnostics containing potentially sensitive document text.
    require(result.returncode == 0 and not result.stderr.strip(), f"{name}: parser failure or warning")
    return result.stdout


def phrase(text):
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def identity(text, revision, document, *, include_hashes):
    expected = [revision["revision_id"], revision["revision_date"], revision["base_commit"], revision["documents"][document]["role"], "Pratik Niroula"]
    if include_hashes:
        expected += [revision["canonical_panel_sha256"], revision["historical_evidence_manifest_sha256"]]
    haystack = phrase(text)
    require(all(phrase(item) in haystack for item in expected), "document revision identity missing or mismatched")
    require("??" not in text, "unresolved reference marker")


def inspect_pdf(path, row, revision, tool):
    with path.open("rb") as f:
        require(f.read(5) == b"%PDF-", "not a PDF")
    info = tool("pdfinfo", str(path))
    for field, value in (("Encrypted", "no"), ("JavaScript", "no"), ("Form", "none")):
        require(re.search(rf"^{field}:\s+{value}\s*$", info, re.M), f"PDF {field} is unsupported")
    pages = re.findall(r"^Pages:\s+(\d+)\s*$", info, re.M)
    require(len(pages) == 1 and int(pages[0]) == row["pages"], "PDF page count mismatch")
    require(not tool("pdfinfo", "-js", str(path)).strip(), "PDF JavaScript detected")
    require(tool("pdfdetach", "-list", str(path)).strip() == "0 embedded files", "PDF embedded files detected")
    text = tool("pdftotext", "-raw", str(path), "-")
    for value in (text, info, tool("pdfinfo", "-meta", str(path)), tool("pdfinfo", "-url", str(path))):
        privacy(value)
    identity(text, revision, row["document"], include_hashes=True)
    return {"pages": int(pages[0])}


def inspect_docx(path, row, revision):
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)), "duplicate ZIP member")
        require(len(names) <= 1000 and sum(i.file_size for i in infos) <= 50_000_000, "oversized DOCX package")
        for i in infos:
            p = PurePosixPath(i.filename)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in i.filename and str(p) == i.filename and not i.is_dir(), "unsafe ZIP member path")
            require(not stat.S_ISLNK(i.external_attr >> 16) and not i.flag_bits & 1, "symlink/encrypted ZIP member")
            require(not any(x in i.filename.lower() for x in ("vbaproject", "embeddings/", "activex/", "customxml/")), "unsupported active/embedded DOCX part")
            require(i.filename.endswith((".xml", ".rels")) or (i.filename.startswith("word/media/") and i.filename.endswith(".png")), "unapproved non-XML DOCX part")
        require({"[Content_Types].xml", "_rels/.rels", "word/document.xml"} <= set(names), "incomplete Word package")
        require(z.testzip() is None, "DOCX CRC failure")
        roots = {}
        for name in names:
            if name.endswith((".xml", ".rels")):
                raw = z.read(name)
                require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(), "XML entities/DTD are unsupported")
                root = ET.fromstring(raw)
                privacy(raw.decode("utf-8"))
                for e in root.iter():
                    for value in (e.text or "", e.tail or "", *e.attrib.values()):
                        privacy(value)
                    local = e.tag.rsplit("}", 1)[-1]
                    require(local not in {"comment", "ins", "del", "moveFrom", "moveTo", "trackRevisions", "vanish", "webHidden", "altChunk", "object", "oleObject"} and not local.endswith("PrChange"), "comments/tracking/hidden/embedded content detected")
                    if local in {"instrText", "fldSimple"}:
                        require(not re.search(r"DDE|INCLUDETEXT|INCLUDEPICTURE|LINK", " ".join([e.text or "", *e.attrib.values()]), re.I), "active external Word field")
                roots[name] = root
        content_types = roots["[Content_Types].xml"]
        ct_namespace = "http://schemas.openxmlformats.org/package/2006/content-types"
        require(content_types.tag == f"{{{ct_namespace}}}Types", "invalid content types root/namespace")
        declarations = {"Default": {}, "Override": {}}
        for declaration in content_types:
            kind = declaration.tag.removeprefix(f"{{{ct_namespace}}}")
            require(declaration.tag == f"{{{ct_namespace}}}{kind}" and kind in declarations, "invalid content type declaration")
            identifier = "Extension" if kind == "Default" else "PartName"
            keys(declaration.attrib, {identifier, "ContentType"}, "content type declaration")
            name = declaration.attrib[identifier]
            mime = declaration.attrib["ContentType"]
            require(bool(name) and bool(mime) and "/" in mime and not any(c.isspace() for c in mime), "malformed content type declaration")
            require("macroEnabled" not in mime, "macro-enabled content type")
            if kind == "Default":
                require(re.fullmatch(r"[A-Za-z0-9]+", name), "invalid content type extension")
                name = name.casefold()
            else:
                part = PurePosixPath(name)
                require(name.startswith("/") and ".." not in part.parts and "\\" not in name and str(part) == name, "invalid content type part name")
            require(name not in declarations[kind], "duplicate content type declaration")
            declarations[kind][name] = mime
        require(declarations["Override"].get("/word/document.xml") == "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml", "missing exact Word main content type override")
        for name, root in roots.items():
            if not name.endswith(".rels"):
                continue
            require(root.tag == f"{{{REL}}}Relationships", "invalid relationship root")
            ids = set()
            for e in root:
                require(e.tag == f"{{{REL}}}Relationship", "invalid relationship element")
                require(e.get("Id") and e.get("Id") not in ids, "duplicate/missing relationship ID")
                ids.add(e.get("Id"))
                target = e.get("Target", "")
                require(bool(target), "missing relationship target")
                privacy(target)
                mode = e.get("TargetMode", "Internal")
                require(mode in {"Internal", "External"}, "invalid relationship target mode")
                if mode == "External":
                    url = urlsplit(target)
                    require(e.get("Type", "").endswith("/hyperlink") and url.scheme == "https" and url.hostname in {"doi.org", "proceedings.mlr.press"} and url.username is None and url.password is None and url.port in {None, 443}, "non-allowlisted external relationship")
                else:
                    require(not target.startswith("/") and "\\" not in target and not urlsplit(target).scheme, "invalid internal relationship")
                    base = "" if name == "_rels/.rels" else str(PurePosixPath(name).parent.parent)
                    resolved = posixpath.normpath(posixpath.join(base, unquote(target)))
                    require(not resolved.startswith("../") and resolved in names, "missing/escaping relationship target")
        root_rel = roots["_rels/.rels"]
        require(sum(e.get("Type", "").endswith("/officeDocument") and e.get("Target") == "word/document.xml" and e.get("TargetMode", "Internal") == "Internal" for e in root_rel) == 1, "missing unique Word document relationship")
        doc = roots["word/document.xml"]
        require(doc.tag == f"{{{W}}}document", "invalid Word document root")
        identity(" ".join(doc.itertext()), revision, row["document"], include_hashes=False)
        counts = {"omml": len(doc.findall(f".//{{{M}}}oMath")), "tables": len(doc.findall(f".//{{{W}}}tbl")), "media": sum(n.startswith("word/media/") for n in names)}
        require(all(counts[key] == row[key] for key in counts), "Word structure counts mismatch")
        return counts


def audit_package(publication_dir, repo, contract_path, *, run_tool=run_external):
    """Validate fixed allowlisted inputs only; caller separately checks numeric authorities."""
    result = {"schema": "kbound_publication_package_audit_v1", "status": "FAIL", "scope": "READ_ONLY_PRE_RELEASE_PACKAGE_NOT_SOURCE_SEAL_OR_CI", "artifacts": {}, "problems": []}
    try:
        directory = safe_path(publication_dir, directory=True)
        repo = safe_path(repo, directory=True)
        contract_path = safe_path(contract_path)
        require(contract_path.stat().st_size <= 100_000, "oversized contract")
        c = strict_json(contract_path)
        validate_contract(c)
        # Complete the static path/type/inventory preflight before reading documents.
        paths = {role: safe_path(directory / row[0]) for role, row in ROLES.items()}
        sources = {name: safe_path(repo / name) for name in SOURCE_PATHS}
        authority_path = safe_path(repo / REVISION_PATH)
        actual = set()
        for entry in directory.iterdir():
            safe_path(entry)
            if entry.name.startswith("._"):
                require(entry.name[2:] in {v[0] for v in ROLES.values()}, "AppleDouble file lacks an artifact counterpart")
                with entry.open("rb") as f:
                    header = f.read(8)
                require(header == bytes.fromhex("0005160700020000"), "non-AppleDouble metadata file")
            else:
                actual.add(entry.name)
        require(actual == {v[0] for v in ROLES.values()}, "publication inventory is not exactly five artifacts")
        inodes = [(p.stat().st_dev, p.stat().st_ino) for p in paths.values()]
        require(len(inodes) == len(set(inodes)), "aliased artifact files")
        require(digest(authority_path) == c["revision_authority"]["sha256"], "revision authority hash mismatch")
        revision = strict_json(authority_path)
        validate_revision(revision, c)
        for name, p in sources.items():
            require(digest(p) == c["reviewed_manuscript_inputs"][name], "reviewed manuscript hash mismatch: " + name)
        for role, p in paths.items():
            row = c["artifacts"][role]
            require(p.stat().st_size == row["bytes"] and digest(p) == row["sha256"], f"{role}: artifact size/hash mismatch")
        for role, p in paths.items():
            row = c["artifacts"][role]
            counts = inspect_pdf(p, row, revision, run_tool) if row["kind"] == "pdf" else inspect_docx(p, row, revision)
            result["artifacts"][role] = {"filename": row["filename"], "sha256": row["sha256"], "bytes": row["bytes"], "counts": counts}
        # Catch ordinary concurrent edits; this is not a locking or TOCTOU guarantee.
        for role, p in paths.items():
            require(digest(safe_path(p)) == c["artifacts"][role]["sha256"], "artifact changed during audit")
        for name, p in sources.items():
            require(digest(safe_path(p)) == c["reviewed_manuscript_inputs"][name], "source changed during audit")
        require(digest(safe_path(authority_path)) == c["revision_authority"]["sha256"], "revision changed during audit")
        result.update(status="PASS", revision_id=c["revision_id"], contract_sha256=digest(contract_path), reviewed_manuscript_inputs=12, revision_authority_sha256=c["revision_authority"]["sha256"])
    except AuditError as exc:
        result["problems"].append(str(exc))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ET.ParseError, zipfile.BadZipFile, NotImplementedError, subprocess.SubprocessError):
        # Deliberately no exception text: parsers may echo private values or file paths.
        result["problems"].append("Package validation failed; check inventory, strict contract/bindings, paths, document parsing/identity and privacy rules.")
    return result


def audit_numerical_authorities(repo, audit_authorities):
    """Only the six existing opened numerical authority inputs; never Markdown scans."""
    repo = safe_path(repo, directory=True)
    for name in AUTHORITY_PATHS:
        p = safe_path(repo / name)
        if name.endswith(".json"):
            strict_json(p)
    return audit_authorities(repo)
