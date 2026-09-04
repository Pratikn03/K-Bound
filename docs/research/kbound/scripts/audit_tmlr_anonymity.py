#!/usr/bin/env python3
"""Aggressively audit the anonymous K-Bound TMLR PDF for identity leaks.

The mandatory path uses only the Python standard library and Poppler command
line tools.  When pypdf is importable, the audit also walks decoded PDF objects,
outlines, annotations, actions, destinations, metadata, and attachments.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
KBOUND_ROOT = REPO_ROOT / "docs/research/kbound"
DEFAULT_PDF = KBOUND_ROOT / "release/current/kbound_tmlr.pdf"
DEFAULT_REPORT = KBOUND_ROOT / "paper/reports/TMLR_ANONYMITY_AUDIT.md"


@dataclass(frozen=True)
class LeakRule:
    name: str
    display: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Finding:
    rule: LeakRule
    surface: str
    snippet: str


@dataclass(frozen=True)
class CommandResult:
    tool: str
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def display_command(self) -> str:
        return shlex.join((self.tool, *self.args))

    @property
    def output_sha256(self) -> str:
        payload = (self.stdout + "\n" + self.stderr).encode("utf-8", errors="replace")
        return hashlib.sha256(payload).hexdigest()


FORBIDDEN_RULES = (
    LeakRule("named author", "Pratik Niroula", re.compile(r"\bpratik\s+niroula\b", re.I)),
    LeakRule("author given name", "Pratik", re.compile(r"\bpratik\b", re.I)),
    LeakRule("author surname", "Niroula", re.compile(r"\bniroula\b", re.I)),
    LeakRule(
        "institution",
        "Minnesota State University",
        re.compile(r"\bminnesota\s+state\s+university\b", re.I),
    ),
    LeakRule(
        "institution",
        "Minnesota State",
        re.compile(r"\bminnesota\s+state\b", re.I),
    ),
    LeakRule("institution location", "Mankato", re.compile(r"\bmankato\b", re.I)),
    LeakRule(
        "named author email",
        "pratik.niroula@mnsu.edu",
        re.compile(r"pratik\.niroula\s*@\s*mnsu\.edu", re.I),
    ),
    LeakRule("institution domain", "mnsu.edu", re.compile(r"\bmnsu\.edu\b", re.I)),
    LeakRule(
        "personal GitHub username",
        "Pratikn03",
        re.compile(r"\bpratikn03\b", re.I),
    ),
    LeakRule(
        "personal GitHub noreply identity",
        "122629934+Pratikn03",
        re.compile(r"122629934\s*\+\s*pratikn03", re.I),
    ),
    LeakRule(
        "local account name",
        "pratik_n",
        re.compile(r"(?<![A-Za-z0-9])pratik_n(?![A-Za-z0-9])", re.I),
    ),
    LeakRule(
        "local repository checkout name",
        "K-Bound-resident",
        re.compile(r"\bk[-_]bound[-_]resident\b", re.I),
    ),
    LeakRule(
        "local workspace directory name",
        "AutoML_" "Flagship_V8",
        re.compile(r"\bautoml_flagship_v8\b", re.I),
    ),
)

PERSONAL_REPOSITORY_URL = LeakRule(
    "personal repository or forge URL",
    "personal repository or forge URL",
    re.compile(r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/[^\s<>)\]}]+", re.I),
)
ABSOLUTE_LOCAL_PATH = LeakRule(
    "absolute local filesystem path",
    "absolute local filesystem path",
    re.compile(
        r"(?:/Users/[^\s<>'\"]+|/home/[^\s<>'\"]+|[A-Za-z]:\\Users\\[^\s<>'\"]+)",
        re.I,
    ),
)
EMAIL_ADDRESS = LeakRule(
    "email address in anonymous manuscript",
    "email address",
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
)
ACKNOWLEDGMENT = LeakRule(
    "acknowledgments section or statement",
    "acknowledgments statement",
    re.compile(r"\backnowledg(?:e)?ments?\b", re.I),
)
AUTHOR_CONTRIBUTION = LeakRule(
    "author-contribution statement",
    "author-contribution statement",
    re.compile(r"\b(?:author\s+contributions?|credit\s+authorship)\b", re.I),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tool(name: str, *, required: bool = True) -> str | None:
    override_name = f"KBOUND_TOOL_{name.upper()}"
    override = os.environ.get(override_name)
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise RuntimeError(f"{override_name} is not an executable file")

    found = shutil.which(name)
    if found:
        return found

    # Codex's bundled Poppler exposes pdfinfo on PATH through a wrapper while
    # the sibling utilities live in the native dependency directory.
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        candidate = Path(pdfinfo).resolve().parents[2] / "native/poppler/poppler/bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    if required:
        raise RuntimeError(f"missing required PDF tool: {name}")
    return None


def _run(tool: str, *args: str, required: bool = True) -> CommandResult | None:
    executable = _tool(tool, required=required)
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return CommandResult(tool, tuple(args), completed.returncode, completed.stdout, completed.stderr)


def _repo_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _parse_pdfinfo(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def _snippet(text: str, match: re.Match[str], *, radius: int = 72) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    compact = re.sub(r"\s+", " ", text[start:end]).strip()
    if start:
        compact = "..." + compact
    if end < len(text):
        compact += "..."
    return compact.replace("`", "'")


def _scan(text: str, surface: str, rules: Iterable[LeakRule]) -> list[Finding]:
    findings: list[Finding] = []
    for rule in rules:
        for match in rule.pattern.finditer(text):
            findings.append(Finding(rule, surface, _snippet(text, match)))
    return findings


def _walk_pypdf_object_graph(reader: Any) -> tuple[str, dict[str, Any]]:
    """Return decoded object text plus a conservative structural inventory."""

    from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, StreamObject

    chunks: list[str] = []
    seen: set[tuple[int, int]] = set()
    stream_count = 0
    decoded_stream_count = 0

    def walk(value: Any) -> None:
        nonlocal stream_count, decoded_stream_count
        if isinstance(value, IndirectObject):
            identity = (value.idnum, value.generation)
            if identity in seen:
                return
            seen.add(identity)
            try:
                walk(value.get_object())
            except Exception as exc:  # pragma: no cover - corrupt-object diagnostic
                chunks.append(f"pypdf-object-error:{type(exc).__name__}")
            return
        if isinstance(value, StreamObject):
            stream_count += 1
            try:
                data = value.get_data()
            except Exception as exc:  # pragma: no cover - corrupt-stream diagnostic
                chunks.append(f"pypdf-stream-error:{type(exc).__name__}")
            else:
                decoded_stream_count += 1
                if len(data) <= 16 * 1024 * 1024:
                    chunks.append(data.decode("latin-1", errors="ignore"))
            for key, item in value.items():
                chunks.append(str(key))
                walk(item)
            return
        if isinstance(value, DictionaryObject):
            for key, item in value.items():
                chunks.append(str(key))
                walk(item)
            return
        if isinstance(value, ArrayObject):
            for item in value:
                walk(item)
            return
        if isinstance(value, bytes):
            chunks.append(value.decode("latin-1", errors="ignore"))
            return
        if isinstance(value, str):
            chunks.append(value)

    walk(reader.trailer)
    return "\n".join(chunks), {
        "indirect_objects_walked": len(seen),
        "streams_seen": stream_count,
        "streams_decoded": decoded_stream_count,
    }


def _flatten_outline_titles(value: Any) -> list[str]:
    titles: list[str] = []
    if isinstance(value, list):
        for item in value:
            titles.extend(_flatten_outline_titles(item))
        return titles
    title = getattr(value, "title", None)
    if title is not None:
        titles.append(str(title))
    elif isinstance(value, dict) and "/Title" in value:
        titles.append(str(value["/Title"]))
    return titles


def _pypdf_inventory(pdf: Path) -> tuple[dict[str, Any], dict[str, str]] | None:
    try:
        import pypdf
    except ImportError:
        return None

    reader = pypdf.PdfReader(str(pdf), strict=False)
    metadata = {str(key): str(value) for key, value in (reader.metadata or {}).items()}
    outline_titles = _flatten_outline_titles(reader.outline)
    named_destinations = reader.named_destinations

    annotations: list[str] = []
    action_count = 0
    for page_number, page in enumerate(reader.pages, start=1):
        for reference in page.get("/Annots", []):
            annotation = reference.get_object()
            annotations.append(f"page={page_number} {annotation!s}")
            if annotation.get("/A") is not None or annotation.get("/AA") is not None:
                action_count += 1

    try:
        attachment_names = sorted(str(name) for name in reader.attachments)
    except Exception:  # pragma: no cover - malformed name-tree diagnostic
        attachment_names = ["<attachment inventory could not be decoded>"]

    object_text, walk_counts = _walk_pypdf_object_graph(reader)
    surfaces = {
        "pypdf metadata": "\n".join(f"{key}: {value}" for key, value in metadata.items()),
        "pypdf bookmarks": "\n".join(outline_titles),
        "pypdf annotations and actions": "\n".join(annotations),
        "decoded PDF object graph": object_text,
    }
    inventory: dict[str, Any] = {
        "pypdf_version": pypdf.__version__,
        "pages": len(reader.pages),
        "metadata": metadata,
        "bookmark_count": len(outline_titles),
        "named_destination_count": len(named_destinations),
        "annotation_count": len(annotations),
        "annotation_action_count": action_count,
        "attachment_count": len(attachment_names),
        "attachment_names": attachment_names,
        **walk_counts,
    }
    return inventory, surfaces


def _contexts(text: str, pattern: re.Pattern[str], limit: int = 4) -> list[str]:
    return [_snippet(text, match, radius=80) for match in list(pattern.finditer(text))[:limit]]


def _command_summary(result: CommandResult) -> str:
    payload = result.stdout + result.stderr
    if result.returncode != 0:
        return f"ERROR (exit {result.returncode}; {len(payload)} characters)"
    if result.tool == "pdfinfo" and "-dests" in result.args:
        count = sum(1 for line in result.stdout.splitlines() if re.match(r"^\s*\d+\s+\[", line))
        return f"OK; {count} named-destination rows"
    if result.tool == "pdfinfo" and "-url" in result.args:
        count = sum(1 for line in result.stdout.splitlines() if "http://" in line or "https://" in line)
        return f"OK; {count} external URL rows"
    if result.tool == "pdfinfo" and "-js" in result.args:
        return "OK; no JavaScript" if not result.stdout.strip() else "OK; JavaScript output present"
    if result.tool == "pdfdetach":
        return f"OK; {result.stdout.strip() or 'no attachment listing'}"
    if result.tool in {"pdftotext", "pdftohtml"}:
        return f"OK; {len(result.stdout)} extracted characters"
    if result.tool == "gs":
        return "OK; Ghostscript parsed every page without diagnostics" if not payload.strip() else "OK"
    return f"OK; {len(payload)} output characters"


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def audit(pdf: Path) -> tuple[bool, str]:
    pdf = pdf.resolve()
    display_pdf = _repo_display(pdf)
    if not pdf.is_file():
        raise RuntimeError(f"missing PDF: {pdf}")

    command_results: list[CommandResult] = []
    mandatory_specs = (
        ("pdfinfo", ("-isodates", display_pdf)),
        ("pdfinfo", ("-custom", display_pdf)),
        ("pdfinfo", ("-meta", display_pdf)),
        ("pdfinfo", ("-dests", display_pdf)),
        ("pdfinfo", ("-url", display_pdf)),
        ("pdfinfo", ("-js", display_pdf)),
        ("pdfdetach", ("-list", display_pdf)),
        ("pdftotext", ("-layout", display_pdf, "-")),
        ("pdftotext", ("-raw", display_pdf, "-")),
        ("pdftohtml", ("-xml", "-hidden", "-i", "-stdout", display_pdf)),
    )
    for tool, args in mandatory_specs:
        result = _run(tool, *args)
        assert result is not None
        command_results.append(result)

    gs_result = _run(
        "gs",
        "-q",
        "-dNOPAUSE",
        "-dBATCH",
        "-sDEVICE=nullpage",
        display_pdf,
        required=False,
    )
    if gs_result is not None:
        command_results.append(gs_result)

    by_signature = {(item.tool, item.args): item for item in command_results}

    def result(tool: str, *args: str) -> CommandResult:
        return by_signature[(tool, tuple(args))]

    info = result("pdfinfo", "-isodates", display_pdf)
    custom = result("pdfinfo", "-custom", display_pdf)
    meta = result("pdfinfo", "-meta", display_pdf)
    dests = result("pdfinfo", "-dests", display_pdf)
    urls = result("pdfinfo", "-url", display_pdf)
    javascript = result("pdfinfo", "-js", display_pdf)
    attachments = result("pdfdetach", "-list", display_pdf)
    layout = result("pdftotext", "-layout", display_pdf, "-")
    raw_text = result("pdftotext", "-raw", display_pdf, "-")
    hidden_xml = result("pdftohtml", "-xml", "-hidden", "-i", "-stdout", display_pdf)

    raw_pdf = pdf.read_bytes()
    raw_objects = raw_pdf.decode("latin-1", errors="ignore")
    surfaces = {
        "PDF metadata and document properties": info.stdout + "\n" + custom.stdout + "\n" + meta.stdout,
        "visible text (layout extraction)": layout.stdout,
        "visible text (raw reading-order extraction)": raw_text.stdout,
        "hidden-inclusive XML text extraction": hidden_xml.stdout,
        "bookmarks and named destinations": dests.stdout,
        "annotations and external URLs": urls.stdout,
        "JavaScript actions": javascript.stdout,
        "attachment inventory": attachments.stdout,
        "raw PDF objects": raw_objects,
    }

    pypdf_result = _pypdf_inventory(pdf)
    if pypdf_result is None:
        pypdf_inventory = None
    else:
        pypdf_inventory, pypdf_surfaces = pypdf_result
        surfaces.update(pypdf_surfaces)

    findings: list[Finding] = []
    identity_rules = (*FORBIDDEN_RULES, PERSONAL_REPOSITORY_URL, ABSOLUTE_LOCAL_PATH, EMAIL_ADDRESS)
    for surface, text in surfaces.items():
        findings.extend(_scan(text, surface, identity_rules))

    prose_only = layout.stdout + "\n" + raw_text.stdout
    findings.extend(_scan(prose_only, "manuscript prose", (ACKNOWLEDGMENT, AUTHOR_CONTRIBUTION)))

    # Keep the report readable when one visible string appears in every
    # independent extraction.  Surface-level duplication is retained because
    # it demonstrates which inspection modes caught the leak.
    deduped: list[Finding] = []
    seen_findings: set[tuple[str, str, str]] = set()
    for finding in findings:
        key = (finding.rule.name, finding.surface, finding.snippet)
        if key not in seen_findings:
            seen_findings.add(key)
            deduped.append(finding)
    findings = deduped

    failures: list[str] = []
    for command in command_results:
        if command.returncode != 0:
            failures.append(f"{command.display_command} exited {command.returncode}")

    info_fields = _parse_pdfinfo(info.stdout)
    author = info_fields.get("Author", "")
    if author and author.casefold() not in {"anonymous", "anonymous authors"}:
        failures.append(f"PDF Author metadata is non-anonymous: {author!r}")

    attachment_match = re.search(r"(\d+)\s+embedded files?", attachments.stdout, re.I)
    if attachment_match is None:
        failures.append("pdfdetach did not provide a parseable embedded-file count")
        attachment_count = -1
    else:
        attachment_count = int(attachment_match.group(1))
        if attachment_count:
            failures.append(f"PDF contains {attachment_count} embedded attachment(s)")

    if javascript.stdout.strip():
        failures.append("PDF contains JavaScript")
    if pypdf_inventory is not None and pypdf_inventory["attachment_count"]:
        failures.append(f"pypdf object walk found {pypdf_inventory['attachment_count']} embedded attachment(s)")
    if findings:
        failures.append(f"found {len(findings)} author-identifying or double-blind-prohibited match(es)")

    repository_pattern = re.compile(
        r"\b(?:repository|repositories|source\s+code|codebase|github|gitlab|bitbucket)\b", re.I
    )
    source_file_pattern = re.compile(
        r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+\.(?:tex|py|json|sh|txt|toml|ya?ml|csv|parquet|pt|pth)\b",
        re.I,
    )
    relative_path_pattern = re.compile(r"\b(?:docs|experiments|tests|src|paper)/[A-Za-z0-9_./-]+", re.I)
    supplement_pattern = re.compile(
        r"\b(?:supplement(?:al|ary)?\s+material|standalone\s+supplement|supplementary\s+file)\b",
        re.I,
    )
    review_signals = {
        "Generic repository/source-code wording": _contexts(prose_only, repository_pattern),
        "Displayed source or artifact filenames": _contexts(prose_only, source_file_pattern),
        "Displayed repository-relative paths": _contexts(prose_only, relative_path_pattern),
        "Supplemental-material references": _contexts(prose_only, supplement_pattern),
    }

    passed = not failures
    pdf_sha = _sha256_bytes(raw_pdf)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    lines = [
        "# TMLR Anonymity Audit",
        "",
        f"- Audit time (UTC): `{timestamp}`",
        f"- PDF: `{display_pdf}`",
        f"- PDF SHA-256: `{pdf_sha}`",
        f"- File size: `{len(raw_pdf)}` bytes",
        f"- Overall result: **{'PASS' if passed else 'FAIL'}**",
        "",
        "This is an identity-leak audit, not a scientific-content or layout review. It scans visible text, "
        "hidden-inclusive text, raw reading order, metadata, raw PDF objects, destinations/bookmarks, "
        "annotations/actions, URLs, JavaScript, attachments, and (when available) a decoded pypdf object walk.",
        "",
        "## Reproduction command",
        "",
        "Run from the repository root:",
        "",
        "```bash",
        f"python docs/research/kbound/scripts/audit_tmlr_anonymity.py --pdf {shlex.quote(display_pdf)} --report docs/research/kbound/paper/reports/TMLR_ANONYMITY_AUDIT.md",
        "```",
        "",
        "## Low-level commands and exact outcomes",
        "",
        "| Command | Exit | Result | Output SHA-256 |",
        "|---|---:|---|---|",
    ]
    for command in command_results:
        lines.append(
            "| `"
            + _md_cell(command.display_command)
            + "` | "
            + str(command.returncode)
            + " | "
            + _md_cell(_command_summary(command))
            + " | `"
            + command.output_sha256
            + "` |"
        )

    metadata_keys = (
        "Title",
        "Author",
        "Subject",
        "Keywords",
        "Creator",
        "Producer",
        "CreationDate",
        "ModDate",
        "Pages",
        "Encrypted",
        "Form",
        "JavaScript",
    )
    lines.extend(
        [
            "",
            "## Document properties",
            "",
            "| Field | Exact value |",
            "|---|---|",
        ]
    )
    for key in metadata_keys:
        if key in info_fields:
            value = info_fields[key] if info_fields[key] else "<empty>"
            lines.append(f"| {_md_cell(key)} | `{_md_cell(value)}` |")
    lines.extend(
        [
            "",
            f"- Metadata stream output: {'empty (no XMP metadata stream)' if not meta.stdout.strip() else 'present and scanned'}.",
            f"- Embedded attachments: `{attachment_count if attachment_count >= 0 else 'unparsed'}`.",
            f"- JavaScript: `{'absent' if not javascript.stdout.strip() else 'present'}`.",
            "",
            "## URLs and actions",
            "",
        ]
    )
    url_lines = [line.strip() for line in urls.stdout.splitlines() if "http://" in line or "https://" in line]
    if url_lines:
        for line in url_lines:
            safe_line = line.replace("`", "'")
            lines.append(f"- `{safe_line}`")
    else:
        lines.append("- No external URL annotations were reported by Poppler.")

    lines.extend(["", "## Object-level inventory", ""])
    if pypdf_inventory is None:
        lines.append(
            "- `pypdf` was not importable. The fail-closed Poppler, raw-object, metadata, URL, "
            "destination, hidden-text, JavaScript, and attachment checks still ran. Re-run with pypdf "
            "available for the additional decoded object graph inventory."
        )
    else:
        lines.extend(
            [
                f"- pypdf version: `{pypdf_inventory['pypdf_version']}`",
                f"- Pages: `{pypdf_inventory['pages']}`",
                f"- Bookmarks/outlines: `{pypdf_inventory['bookmark_count']}`",
                f"- Named destinations: `{pypdf_inventory['named_destination_count']}`",
                f"- Annotations: `{pypdf_inventory['annotation_count']}`",
                f"- Annotation actions: `{pypdf_inventory['annotation_action_count']}`",
                f"- Attachments: `{pypdf_inventory['attachment_count']}`",
                f"- Indirect objects walked: `{pypdf_inventory['indirect_objects_walked']}`",
                f"- Streams seen/decoded: `{pypdf_inventory['streams_seen']}/{pypdf_inventory['streams_decoded']}`",
            ]
        )

    lines.extend(["", "## Prohibited-identity findings", ""])
    if findings:
        for finding in findings[:80]:
            lines.append(
                f"- **{finding.rule.display}** ({finding.rule.name}) in {finding.surface}: `{finding.snippet}`"
            )
        if len(findings) > 80:
            lines.append(f"- ... {len(findings) - 80} additional matches omitted from this report.")
    else:
        lines.append(
            "- None. No named author, institution, author email/domain, personal GitHub identity, "
            "forge repository URL, email address, absolute home-directory path, acknowledgment, or "
            "author-contribution statement was found on any inspected surface."
        )

    lines.extend(
        [
            "",
            "## Non-identifying review signals",
            "",
            "These signals are reported for human review but do not fail the audit by themselves. "
            "Generic repository-relative artifact names and generic references to supplementary material "
            "do not reveal an author unless paired with an identity, public repository URL, username, or local path.",
            "",
        ]
    )
    for label, contexts in review_signals.items():
        lines.append(f"### {label}")
        lines.append("")
        if contexts:
            for context in contexts:
                lines.append(f"- `{context}`")
        else:
            lines.append("- None.")
        lines.append("")

    lines.extend(["## Failure reasons", ""])
    if failures:
        for failure in failures:
            lines.append(f"- {failure}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "PASS: the audited PDF contains no detected author-identifying text, metadata, hidden "
                "object content, personal URLs, local paths, attachments, JavaScript, acknowledgments, "
                "or author-contribution statement. The PDF Author field is empty."
                if passed
                else "FAIL: one or more anonymity gates above failed. Do not upload this PDF for double-blind review."
            ),
            "",
        ]
    )
    return passed, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    try:
        passed, report = audit(args.pdf)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: TMLR anonymity audit -> {args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
