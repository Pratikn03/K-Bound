#!/usr/bin/env python3
"""Remove figure-document metadata without changing its rendered page.

Uses the existing pinned TeX/Poppler tools; no PDF runtime dependency is added.
Every candidate must pass privacy, text, geometry and exact pixel checks before
replacing an original. Already anonymous figures are byte-preserved.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.research.kbound.scripts.build_anonymous_supplement import verify_pdf_anonymity
from docs.research.kbound.scripts.release_privacy import PrivacyError

FIGURES = (
    "fig_certificate.pdf",
    "fig_cifar_tent_online.pdf",
    "fig_decision_flow.pdf",
    "fig_decisive_pareto_cifar10c.pdf",
    "fig_frontier_schematic.pdf",
)
WRAPPER = r"""\pdfoutput=1
\pdfdecimaldigits=5
\pdfinfoomitdate=1
\pdftrailerid{}
\pdfinfo{/Author () /Creator () /Producer () /Title ()}
\pdfximage{input.pdf}
\setbox0=\hbox{\pdfrefximage\pdflastximage}
\pdfpagewidth=\wd0
\pdfpageheight=\ht0
\pdfhorigin=0pt
\pdfvorigin=0pt
\shipout\box0
\end
"""


def run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=90).stdout


def pixels(path: Path, output: Path) -> tuple[tuple[int, int], bytes]:
    run(["pdftoppm", "-r", "144", "-singlefile", "-png", str(path), str(output)])
    with Image.open(output.with_suffix(".png")) as source:
        converted = source.convert("RGB")
        try:
            return converted.size, converted.tobytes()
        finally:
            converted.close()


def sanitize(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"figure must be a regular non-symlink file: {path}")
    original = path.read_bytes()
    try:
        verify_pdf_anonymity(path.name, original)
        return False
    except PrivacyError:
        pass
    # This wrapper supports only the verified static figure subset. pdfximage
    # omits annotations; text and pixel comparisons cannot detect that loss.
    # Decode PDF name escapes and fail closed on object streams (which could
    # hide page dictionaries), annotations, forms, or document actions.
    inspected = re.sub(rb"#([0-9A-Fa-f]{2})", lambda m: bytes([int(m[1], 16)]), original)
    if re.search(rb"/(?:ObjStm|Annots|AcroForm|OpenAction|AA)(?=[\x00\s/<>\[\](){}%]|$)", inspected):
        raise ValueError("figure annotations, actions, forms, or object streams are unsupported")
    with tempfile.TemporaryDirectory(prefix="kbound-figure-metadata-") as name:
        temporary = Path(name)
        source = temporary / "input.pdf"
        source.write_bytes(original)
        info = run(["pdfinfo", str(source)])
        fields = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in info.splitlines() if ":" in line)}
        if any(
            fields.get(k) != v
            for k, v in {"Pages": "1", "Page rot": "0", "Form": "none", "JavaScript": "no", "Encrypted": "no"}.items()
        ):
            raise ValueError("metadata normalization requires a single static unrotated figure page")
        if run(["pdfdetach", "-list", str(source)]).strip() != "0 embedded files":
            raise ValueError("figure must not contain attachments")
        (temporary / "wrapper.tex").write_text(WRAPPER)
        run(
            [
                "pdflatex",
                "-fmt=pdftex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                "wrapper.tex",
            ],
            cwd=temporary,
        )
        candidate = temporary / "wrapper.pdf"
        result = candidate.read_bytes()
        verify_pdf_anonymity(path.name, result)
        new_info = run(["pdfinfo", str(candidate)])
        for key in ("Pages", "Page rot"):
            match = re.search(rf"^{key}:\s*(.+)$", new_info, re.MULTILINE)
            if match is None or match.group(1).strip() != fields[key]:
                raise ValueError(f"figure {key} changed during metadata normalization")
        old_size = [float(v) for v in re.findall(r"[0-9.]+", fields["Page size"])[:2]]
        new_size_line = re.search(r"^Page size:\s*(.+)$", new_info, re.MULTILINE)
        if new_size_line is None:
            raise ValueError("normalized figure lacks page dimensions")
        new_size = [float(v) for v in re.findall(r"[0-9.]+", new_size_line.group(1))[:2]]
        # TeX dimensions are quantized to scaled points. Bound that rounding
        # within 0.001 reported PDF point and require exact 144-DPI pixel identity below.
        if len(old_size) != 2 or len(new_size) != 2 or any(abs(a - b) > 0.001 for a, b in zip(old_size, new_size)):
            raise ValueError("figure page dimensions changed beyond TeX rounding")
        if run(["pdftotext", str(source), "-"]) != run(["pdftotext", str(candidate), "-"]):
            raise ValueError("figure extracted text changed during metadata normalization")
        if pixels(source, temporary / "before") != pixels(candidate, temporary / "after"):
            raise ValueError("figure rendered pixels changed during metadata normalization")
        if path.is_symlink() or path.read_bytes() != original:
            raise ValueError("figure changed during metadata normalization")
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".figure-metadata-", delete=False) as handle:
            replacement = Path(handle.name)
            handle.write(result)
        try:
            os.chmod(replacement, path.stat().st_mode & 0o777)
            os.replace(replacement, path)
        finally:
            replacement.unlink(missing_ok=True)
    print(f"normalized {path.name}: {hashlib.sha256(original).hexdigest()} -> {hashlib.sha256(result).hexdigest()}")
    return True


def main() -> None:
    for filename in FIGURES:
        sanitize(ROOT / "docs/research/kbound/figures" / filename)


if __name__ == "__main__":
    main()
