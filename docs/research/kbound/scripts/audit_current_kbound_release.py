#!/usr/bin/env python3
"""Fail-closed audit for the four current K-Bound manuscript roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
KBOUND_ROOT = REPO_ROOT / "docs/research/kbound"
RELEASE_ID = "KBOUND-2026-09-03-R1"
PANEL_SHA256 = "35d4c165843de1ece3cb35ffb4ac50dbfcbfa33b646754e2b6d133fbdfa78c6e"
SOURCE_MANIFEST_SHA256 = "03b1d2b1e9e5ed1cf835126f871ed83eb8497a1ac81727b3955af0d69eb89742"


@dataclass(frozen=True)
class Document:
    filename: str
    role: str
    anonymous: bool


DOCUMENTS = (
    Document("kbound_short_main.pdf", "Named compact main paper", False),
    Document("kbound_short_supplement.pdf", "Named standalone supplement", False),
    Document("kbound_tmlr.pdf", "Anonymous integrated TMLR review manuscript", True),
    Document("kbound_full_report.pdf", "Named full technical report", False),
)

ANONYMOUS_FORBIDDEN = (
    "Pratik Niroula",
    "Pratik",
    "Niroula",
    "Minnesota State University",
    "Minnesota State",
    "Mankato",
    "pratik.niroula@mnsu.edu",
    "mnsu.edu",
    "/Users/pratik_n",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool(name: str) -> str:
    override = os.environ.get(f"KBOUND_TOOL_{name.upper()}")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise RuntimeError(f"KBOUND_TOOL_{name.upper()} is not an executable file")
    found = shutil.which(name)
    if found:
        return found
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        candidate = Path(pdfinfo).resolve().parents[2] / "native/poppler/poppler/bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(f"missing required PDF tool: {name}")


def _run_tool(name: str, *args: str) -> str:
    completed = subprocess.run(
        [_tool(name), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{name} failed: {detail}")
    return completed.stdout


def extract_pdf_text(path: Path) -> str:
    # Raw reading order keeps two-column paragraphs and wrapped hashes
    # contiguous; layout mode can interleave the opposite column between a
    # phrase's two lines and produce a false release-audit failure.
    return _run_tool("pdftotext", "-raw", str(path), "-")


def pdfinfo(path: Path) -> str:
    return _run_tool("pdfinfo", str(path))


def normalized(text: str) -> str:
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def phrase_key(text: str) -> str:
    """Normalize PDF line wrapping without weakening the required wording check.

    Poppler may split a long hash across lines and may either keep or remove a
    discretionary hyphen at a column boundary. Required release phrases are
    therefore compared as case-folded alphanumeric sequences. The original
    extracted text remains available for diagnostics and the separate exact
    duplicate/reference checks below.
    """

    return re.sub(r"[^0-9a-z]+", "", text.casefold())


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_equal(problems: list[str], label: str, actual: object, expected: object) -> None:
    if actual != expected:
        problems.append(f"{label}: expected {expected!r}, found {actual!r}")


def _require_close(
    problems: list[str], label: str, actual: float, expected: float, tolerance: float = 5e-5
) -> None:
    if abs(actual - expected) > tolerance:
        problems.append(f"{label}: expected {expected}, found {actual}")


def audit_authorities(repo_root: Path = REPO_ROOT) -> list[str]:
    problems: list[str] = []
    panel_path = repo_root / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
    source_path = repo_root / "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"
    cct_path = repo_root / "docs/research/kbound/paper/generated/cct20_release_manifest.json"
    diagnostics_path = (
        repo_root
        / "docs/research/kbound/paper/generated/current_policy_interval_diagnostics.json"
    )
    for authority in (panel_path, source_path, cct_path, diagnostics_path):
        if not authority.is_file():
            problems.append(f"missing authority: {authority}")
    if problems:
        return problems

    _require_equal(problems, "canonical panel SHA-256", sha256(panel_path), PANEL_SHA256)
    _require_equal(
        problems,
        "canonical source-manifest SHA-256",
        sha256(source_path),
        SOURCE_MANIFEST_SHA256,
    )

    panel = _load_json(panel_path)
    assert isinstance(panel, dict)
    candidates = panel["panels"]["cifar10c"]["panel"]["candidates"]
    expected_cifar = {
        "tent": {
            "n": 2160,
            "adapt_count": 1091,
            "freeze_count": 337,
            "abstain_count": 732,
            "false_adapt_count": 2,
            "regret": (0.0019, 0.0080, 0.1239),
            "inclusion": 1976,
            "false_freeze": (1, 337),
        },
        "eata": {
            "n": 2160,
            "adapt_count": 1207,
            "freeze_count": 105,
            "abstain_count": 848,
            "false_adapt_count": 0,
            "regret": (0.0016, 0.0033, 0.1313),
            "inclusion": 1950,
            "false_freeze": (0, 105),
        },
        "sar": {
            "n": 2160,
            "adapt_count": 1414,
            "freeze_count": 0,
            "abstain_count": 746,
            "false_adapt_count": 0,
            "regret": (0.0018, 0.0003, 0.1405),
            "inclusion": 1949,
            "false_freeze": (0, 0),
        },
    }
    diagnostics = _load_json(diagnostics_path)
    assert isinstance(diagnostics, dict)
    for candidate, expected in expected_cifar.items():
        row = candidates[candidate]
        for field in ("n", "adapt_count", "freeze_count", "abstain_count", "false_adapt_count"):
            _require_equal(problems, f"CIFAR {candidate} {field}", row[field], expected[field])
        regret = row["regret"]
        for key, value in zip(
            ("kga", "always_adapt", "always_freeze"), expected["regret"], strict=True
        ):
            _require_close(problems, f"CIFAR {candidate} regret {key}", regret[key], value)
        summary = diagnostics["candidates"][candidate]["summary"]
        _require_equal(
            problems,
            f"CIFAR {candidate} interval inclusion",
            summary["observed_inclusion"]["numerator"],
            expected["inclusion"],
        )
        _require_equal(
            problems,
            f"CIFAR {candidate} false FREEZE count",
            summary["false_freeze"]["conditional"]["numerator"],
            expected["false_freeze"][0],
        )
        _require_equal(
            problems,
            f"CIFAR {candidate} false FREEZE denominator",
            summary["false_freeze"]["conditional"]["denominator"],
            expected["false_freeze"][1],
        )

    cct = _load_json(cct_path)
    assert isinstance(cct, dict)
    design = cct["design"]
    mix = cct["adaptation_effect_mix"]
    actions = cct["action_exposure"]["counts"]
    _require_equal(problems, "CCT total cells", design["cell_count"], 45)
    _require_equal(problems, "CCT independent checkpoints", design["checkpoint_count"], 5)
    _require_equal(problems, "CCT target locations", design["location_cluster_count"], 9)
    _require_equal(problems, "CCT helpful cells", mix["helpful_cells_strictly_positive"], 1)
    _require_equal(problems, "CCT tied cells", mix["neutral_cells_exactly_zero"], 0)
    _require_equal(problems, "CCT harmful cells", mix["harmful_cells_strictly_negative"], 44)
    _require_equal(problems, "CCT ADAPT count", actions["ADAPT"], 0)
    _require_equal(problems, "CCT FREEZE count", actions["FREEZE"], 44)
    _require_equal(problems, "CCT ABSTAIN count", actions["ABSTAIN"], 1)
    effect_rows = sum(cct["adaptation_effect_cells_by_sign"].values(), [])
    false_freeze = sum(
        row["decision"] == "FREEZE" and row["adaptation_benefit"] >= 0
        for row in effect_rows
    )
    _require_equal(problems, "CCT false FREEZE count", false_freeze, 1)
    safe = cct["safe_utility"]
    _require_equal(problems, "CCT safe utility passed", safe["passes"], True)
    _require_close(
        problems,
        "CCT safe contrast versus always adapt",
        safe["versus_always_adapt"]["point_estimate"],
        0.18190227,
        5e-8,
    )
    for index, expected in enumerate((0.09374776, 0.29138626)):
        _require_close(
            problems,
            f"CCT nominal 95% safe interval endpoint {index}",
            safe["versus_always_adapt"]["pointwise_95_ci"][index],
            expected,
            5e-8,
        )
    _require_equal(
        problems,
        "CCT protocol strong success",
        cct["strong_success_checks"]["protocol_strong_success"],
        False,
    )
    _require_equal(problems, "CCT verdict", cct["verdict"]["code"], "SAFE_UTILITY_ONLY")

    numbers = (repo_root / "docs/research/kbound/paper/generated/cct20_numbers.tex").read_text(
        encoding="utf-8"
    )
    for value in (
        r"\newcommand{\CCTVsAdaptCILower}{0.08501071}",
        r"\newcommand{\CCTVsAdaptCIUpper}{0.31066374}",
        r"\newcommand{\CCTVsAdaptExactP}{0.00195312}",
        r"\newcommand{\CCTVsAdaptHolmP}{0.00390625}",
        r"\newcommand{\CCTVsFreezeHolmP}{1}",
    ):
        if value not in numbers:
            problems.append(f"CCT generated strong-audit value missing: {value}")
    reporting_numbers = (
        repo_root / "docs/research/kbound/paper/generated/cct20_reporting_numbers.tex"
    ).read_text(encoding="utf-8")
    for value in (
        r"\newcommand{\CCTFalseFreezeCount}{1}",
        r"\newcommand{\CCTFalseFreezeCountWord}{one}",
        r"\newcommand{\CCTFalseFreezeConditionalDenominator}{44}",
        r"\newcommand{\CCTFalseFreezeOverallDenominator}{45}",
    ):
        if value not in reporting_numbers:
            problems.append(f"CCT generated false-FREEZE value missing: {value}")
    return problems


def _require_pdf_phrase(problems: list[str], filename: str, text: str, phrase: str) -> None:
    if phrase_key(phrase) not in phrase_key(text):
        problems.append(f"{filename}: missing required phrase: {phrase}")


def audit_pdfs(pdf_dir: Path) -> list[str]:
    problems: list[str] = []
    texts: dict[str, str] = {}
    infos: dict[str, str] = {}
    for document in DOCUMENTS:
        path = pdf_dir / document.filename
        if not path.is_file():
            problems.append(f"missing current PDF: {path}")
            continue
        try:
            texts[document.filename] = extract_pdf_text(path)
            infos[document.filename] = pdfinfo(path)
        except RuntimeError as exc:
            problems.append(f"{document.filename}: {exc}")

    if problems:
        return problems

    common_phrases = (
        RELEASE_ID,
        PANEL_SHA256,
        SOURCE_MANIFEST_SHA256,
        "0/44/1",
        "one false FREEZE among 44 FREEZE",
        "1/45 overall",
        "All 45 served predictions used the frozen model",
        "does not demonstrate selective routing or strong success",
        "cell-outcome-unopened, internally sealed execution",
        "not globally label-unopened or publicly preregistered",
        "nominal pointwise 95% intervals",
        "nominal 97.5% Bonferroni intervals",
        "not alternate versions of the same interval",
    )
    for document in DOCUMENTS:
        text = texts[document.filename]
        for phrase in common_phrases:
            _require_pdf_phrase(problems, document.filename, text, phrase)
        _require_pdf_phrase(problems, document.filename, text, document.role)
        if "??" in text:
            problems.append(f"{document.filename}: unresolved ?? reference appears in extracted text")
        if "0/44/1, or 0/44/1" in normalized(text):
            problems.append(f"{document.filename}: duplicated CCT action wording remains")
        if document.anonymous:
            haystack = text + "\n" + infos[document.filename]
            for token in ANONYMOUS_FORBIDDEN:
                if token.casefold() in haystack.casefold():
                    problems.append(f"{document.filename}: anonymity leak: {token}")
            _require_pdf_phrase(problems, document.filename, text, "Anonymous build ID")
        else:
            _require_pdf_phrase(problems, document.filename, text, "Pratik Niroula")
            _require_pdf_phrase(problems, document.filename, text, "Source snapshot commit")

    for filename in ("kbound_short_main.pdf", "kbound_tmlr.pdf", "kbound_full_report.pdf"):
        _require_pdf_phrase(
            problems,
            filename,
            texts[filename],
            "Using labeled development and residual-calibration cells",
        )
        _require_pdf_phrase(
            problems,
            filename,
            texts[filename],
            "The scored cell contributes label-free evidence only",
        )
    return problems


def audit_release_directory(pdf_dir: Path) -> list[str]:
    expected = {document.filename for document in DOCUMENTS}
    actual = {path.name for path in pdf_dir.glob("*.pdf")}
    if actual != expected:
        return [
            f"current release PDF inventory mismatch: expected {sorted(expected)}, found {sorted(actual)}"
        ]
    return []


def audit_active_claims(kbound_root: Path = KBOUND_ROOT) -> list[str]:
    """Scan every active publication surface for promoted legacy claims.

    This is deliberately a curated allowlist rather than a recursive scan.  The
    tree contains experiment logs, superseded audits, and archived protocols
    whose job is to preserve negative history.  Those files are checked for an
    explicit superseded banner in :func:`audit_release_files`; the paths below
    are the sources that can reach a current manuscript, release note, generated
    claim matrix, or default packaging command.
    """

    problems: list[str] = []
    active_relatives = (
        "kbound_abstract.tex",
        "kbound_abstract_core.tex",
        "kbound_short_main.tex",
        "kbound_short_supplement.tex",
        "kbound_submission.tex",
        "kbound_submission_body.tex",
        "kbound_submission_supplement.tex",
        "kbound_tmlr.tex",
        "kbound_full_report.tex",
        "kbound_full_report_extensions.tex",
        "README.md",
        "CURRENT_RELEASE.md",
        "DOCS_INDEX.md",
        "RELEASE_CHECKLIST.md",
        "REPRODUCE.md",
        "KBOUND_SHORT_CLAIM_MANIFEST.md",
        "KBOUND_SHORT_RESULT_AUDIT.md",
        "KBOUND_SHORT_THEORY_AUDIT.md",
        "paper/reports/KBOUND_SOURCE_OUTPUT_MAP.md",
        "paper/RELEASE_TABLE_CROSSWALK.md",
        "paper/generated/empirical_audit/claim_matrix.md",
        "paper/release/current_release.json",
        "scripts/build_pdfs.sh",
        "scripts/publish_current_pdfs.py",
        "scripts/empirical_closure.py",
    )
    forbidden = (
        r"zero false ADAPT(?: actions?)? for (?:current )?(?:CIFAR(?:-10-C)? )?Tent",
        r"(?:valid|confirmed|successful) current iWildCam (?:result|claim)",
        r"iWildCam[^\n]{0,100}(?:no-harm|beats both|beats-both)",
        r"(?:universal natural-shift|uniformly) no-harm",
        r"uniformly no-harm on every natural shift",
        r"completed (?:real|physical)[ -]camera (?:result|study)",
        r"physical-camera[^\n]{0,80}(?:accuracy|regret|utility|error)\s*[=:]?\s*\d",
        r"(?:old )?36-cell ImageNet-C[^\n]{0,80}(?:current|authority)",
        r"official-setting SAR conclusion",
        r"mechanism-faithful SAR",
        r"(?:leave-one-out|jackknife\+?)[^\n]{0,100}current CIFAR",
        r"CIFAR(?:-10-C)? Tent[^\n]{0,100}0\.0016",
        r"CIFAR(?:-10-C)? EATA[^\n]{0,100}0\.0015",
        r"(?:population-risk|population risk) certificate",
        r"13\s*[-–]\s*24\s*[x×]\s+below either fixed policy",
        r"(?:current headline|headline result)[^\n]{0,100}pooled mixed-stream",
        r"(?:maintained headline theory|headline theorem)[^\n]{0,120}(?:one-bit|Le Cam|minimax|full-foundations)",
        r"KGA[^\n]{0,120}beats both[^\n]{0,120}(?:all|most) natural shifts",
        r"CCT(?:-20)?[^\n]{0,100}(?:learned|demonstrates) selective routing",
        r"CCT(?:-20)?[^\n]{0,100}(?:no directional error|zero directional error)",
        r"CCT(?:-20)?[^\n]{0,100}globally label-unopened",
        r"0/44/1,\s*or\s*0/44/1",
    )
    contextual_markers = (
        "historical",
        "superseded",
        "withdrawn",
        "invalid",
        "pending",
        "not completed",
        "no completed",
        "not globally",
        "does not",
        "cannot",
        "lacks",
        "not claim",
        "not a current",
        "forbidden",
    )
    for relative in active_relatives:
        path = kbound_root / relative
        if not path.is_file():
            problems.append(f"missing active source for legacy-claim scan: {path}")
            continue
        text = text_without_comments(path.read_text(encoding="utf-8"))
        for pattern in forbidden:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                context = text[max(0, match.start() - 220) : match.end() + 220].lower()
                if not any(marker in context for marker in contextual_markers):
                    line = text.count("\n", 0, match.start()) + 1
                    problems.append(f"{path.relative_to(REPO_ROOT)}:{line}: stale promoted claim: {match.group(0)}")
    return problems


def text_without_comments(text: str) -> str:
    text = re.sub(r"\\iffalse.*?\\fi", "", text, flags=re.DOTALL)
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())


def audit_release_files(kbound_root: Path = KBOUND_ROOT) -> list[str]:
    problems: list[str] = []
    for relative in (
        "CURRENT_RELEASE.md",
        "paper/release/current_release.json",
        "paper/generated/current_release_identity.tex",
        "paper/RELEASE_TABLE_CROSSWALK.md",
        "paper/reports/TMLR_ANONYMITY_AUDIT.md",
    ):
        if not (kbound_root / relative).is_file():
            problems.append(f"missing release file: {relative}")

    superseded_markers = {
        "CIFAR10C_SAR_QUARANTINE.md": "SUPERSEDED HISTORICAL AUDIT",
        "G8_EXACTRANK_REGEN.md": "SUPERSEDED HISTORICAL AUDIT",
        "RELATED_WORK_POSITIONING.md": "SUPERSEDED HISTORICAL POSITIONING",
        "MIXED_BENCHMARK_PROTOCOL.md": "SUPERSEDED HISTORICAL PROTOCOL",
    }
    for relative, marker in superseded_markers.items():
        candidate = kbound_root / relative
        if not candidate.is_file():
            problems.append(f"missing historical-scope file: {relative}")
        elif marker not in candidate.read_text(encoding="utf-8")[:1200]:
            problems.append(f"historical-scope file lacks superseded banner: {relative}")

    claim_matrix = kbound_root / "paper/generated/empirical_audit/claim_matrix.md"
    if not claim_matrix.is_file():
        problems.append("missing current empirical claim matrix")
    else:
        matrix_text = claim_matrix.read_text(encoding="utf-8")
        if "CIFAR-10-C SAR negative arm" not in matrix_text:
            problems.append("current empirical claim matrix omits the SAR negative arm")
        if "CIFAR-10-C SAR quarantine" in matrix_text:
            problems.append("current empirical claim matrix still presents SAR as quarantined")
    return problems


def run_audit(pdf_dir: Path, *, require_exact_inventory: bool) -> list[str]:
    problems = audit_authorities()
    problems.extend(audit_release_files())
    problems.extend(audit_active_claims())
    problems.extend(audit_pdfs(pdf_dir))
    if require_exact_inventory:
        problems.extend(audit_release_directory(pdf_dir))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, default=KBOUND_ROOT)
    parser.add_argument("--require-exact-inventory", action="store_true")
    args = parser.parse_args()
    problems = run_audit(args.pdf_dir.resolve(), require_exact_inventory=args.require_exact_inventory)
    if problems:
        print("K-Bound current release audit: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("K-Bound current release audit: PASS")
    print(f"Verified {len(DOCUMENTS)} role-distinct PDFs against frozen numerical authorities.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
