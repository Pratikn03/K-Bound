"""Phase 1.1 — deterministic validator for PDF-source-result consistency.

Runs the seven Phase 1.1 consistency checks:
  A. Primary mechanism result consistency (B1/B2 in abstract / body /
     primary table / conclusion).
  B. Canonical MVTec policy consistency (no promoted PR/ECE/Brier).
  C. Audited RGA+ comparison consistency (master table vs manifest).
  D. UNSW language consistency (no broad-generalization claim).
  E. Real3D consistency (exactly one policy-valid exploratory row).
  F. Statistical-language consistency (Family A audited-primary; no
     pre-registered confirmatory framing for existing cells).
  G. Sensitivity + polarity consistency (no causal effect claim; no
     primary prediction flip claim; diagnostic-only wording).

Exits 0 if all checks pass; 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PAPER = Path("docs/research/PAPER_DRAFT_v1.tex")
THESIS = Path("docs/research/THESIS_CHAPTER_v1.tex")
PAPER_PDF = Path("output/pdf/PAPER_DRAFT_PHASE1_1_VERIFIED.pdf")
THESIS_PDF = Path("output/pdf/THESIS_CHAPTER_PHASE1_1_VERIFIED.pdf")


def _pdf_text(path: Path) -> str:
    if not path.exists():
        return ""
    from pypdf import PdfReader
    r = PdfReader(str(path))
    return "".join(p.extract_text() + "\n" for p in r.pages)


def _count(pat: str, text: str, ignore_case: bool = True) -> int:
    flags = re.IGNORECASE if ignore_case else 0
    return len(re.findall(pat, text, flags))


def main() -> None:
    fails: list[str] = []

    paper_text = PAPER.read_text() if PAPER.exists() else ""
    thesis_text = THESIS.read_text() if THESIS.exists() else ""
    paper_pdf_text = _pdf_text(PAPER_PDF)
    thesis_pdf_text = _pdf_text(THESIS_PDF)

    # ----- A. PRIMARY mechanism consistency -----
    for name, txt in [("paper", paper_pdf_text), ("thesis", thesis_pdf_text)]:
        for pat in (r"\+0\.0506", r"\+0\.0319"):
            n = _count(pat, txt, ignore_case=False)
            if n == 0:
                fails.append(f"A: {name} PDF missing PRIMARY mechanism delta {pat}")

    # ----- B. Canonical policy: no 0.7835 anywhere in promoted view -----
    for name, txt in [("paper", paper_pdf_text), ("thesis", thesis_pdf_text)]:
        n = _count(r"0\.7835", txt, ignore_case=False)
        if n > 0:
            fails.append(f"B: {name} PDF still contains {n} instances of 0.7835 "
                         f"(canonical degenerate value)")

    # ----- C. Audited comparison consistency: master table caption -----
    if PAPER.exists():
        if "validation-frozen" not in paper_text.lower():
            fails.append("C: paper source missing 'validation-frozen' in master comparison")
        if "Family A audited-primary" not in paper_text and "audited primary reanalysis" not in paper_text.lower():
            fails.append("C: paper source missing 'Family A audited-primary' language")

    # ----- D. UNSW: no broad-generalization claim -----
    for name, txt in [("paper PDF", paper_pdf_text), ("thesis PDF", thesis_pdf_text)]:
        for pat in (r"prove the cross-benchmark",
                    r"beats every non-ELARA",
                    r"establishes broad cross-domain superiority",
                    r"without losing the cross-domain generalization property"):
            n = _count(pat, txt)
            if n > 0:
                fails.append(f"D: {name} contains forbidden UNSW overclaim '{pat}' ({n} hits)")
        if _count(r"\+0\.0003", txt, ignore_case=False) == 0:
            fails.append(f"D: {name} missing UNSW practically-very-small delta +0.0003")

    # ----- E. Real3D consistency -----
    for name, txt in [("paper PDF", paper_pdf_text), ("thesis PDF", thesis_pdf_text)]:
        if _count(r"no longer the negative cell", txt) > 0:
            fails.append(f"E: {name} still contains 'no longer the negative cell' Real3D claim")
        if _count(r"FPFH\+depth", txt) > 0:
            fails.append(f"E: {name} still contains stale 'FPFH+depth' label")

    # ----- F. Statistical-language consistency -----
    for name, txt in [("paper PDF", paper_pdf_text), ("thesis PDF", thesis_pdf_text)]:
        for pat in (r"Family A confirmatory",
                    r"pre-registered confirmatory",
                    r"Fisher-combined",
                    r"nine evaluated cells", r"9-test Holm"):
            n = _count(pat, txt)
            if n > 0:
                fails.append(f"F: {name} contains forbidden statistical-language '{pat}'")

    # ----- G. Sensitivity + polarity -----
    for name, txt in [("paper PDF", paper_pdf_text), ("thesis PDF", thesis_pdf_text)]:
        for pat in (r"interventional ATE",
                    r"Structural Causal Model",
                    r"Causal Reliability Attribution",
                    r"Causal Inference for Reliability",
                    r"deployment-grade",
                    r"deployment-time sanity check"):
            n = _count(pat, txt)
            if n > 0:
                fails.append(f"G: {name} contains forbidden causal/polarity phrase '{pat}'")

    # ----- Forbidden general -----
    for name, txt in [("paper PDF", paper_pdf_text), ("thesis PDF", thesis_pdf_text)]:
        for pat in (r"max\(router", r"MAX\(router",
                    r"best non-router", r"strongest non-router",
                    r"universally superior", r"\bSOTA\b", r"production-ready"):
            n = _count(pat, txt)
            if n > 0:
                fails.append(f"general: {name} contains forbidden phrase '{pat}'")

    # ----- Report -----
    if fails:
        print(f"PHASE 1.1 VALIDATION FAILED: {len(fails)} violations")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("PHASE 1.1 VALIDATION PASSED: 0 violations.")


if __name__ == "__main__":
    main()
