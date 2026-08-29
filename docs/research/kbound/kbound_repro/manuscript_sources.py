r"""Single source of truth for maintained K-Bound manuscript inputs.

Release checks start from both maintained drivers and follow each live
``\input``/``\include`` closure.  This prevents a generated table or nested
section from bypassing claim validation while still excluding archived drafts
that are not compiled.  The same live-LaTeX projection removes comments and
``\iffalse`` audit blocks before both dependency discovery and claim scanning.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "ACTIVE_DRIVER_RELATIVE_PATHS",
    "ACTIVE_SOURCE_RELATIVE_PATHS",
    "active_source_paths",
    "live_latex",
]


ACTIVE_DRIVER_RELATIVE_PATHS = (
    "docs/research/kbound/kbound_submission.tex",
    "docs/research/kbound/kbound_tmlr.tex",
)


ACTIVE_SOURCE_RELATIVE_PATHS = (
    "docs/research/kbound/kbound_submission.tex",
    "docs/research/kbound/kbound_submission_body.tex",
    "docs/research/kbound/paper/sections/theory_core_main.tex",
    "docs/research/kbound/paper/sections/theory_algorithm_bridge.tex",
    "docs/research/kbound/paper/sections/theory_certificate.tex",
)


def active_source_paths(repo_root: str | Path) -> tuple[Path, ...]:
    """Return the ordered live TeX dependency closure under ``repo_root``.

    Literal inputs are resolved first relative to the including file, then the
    K-Bound manuscript root, then the repository root.  An unresolved literal
    is retained as a missing path so release callers can fail closed.
    """
    root = Path(repo_root).resolve()
    kbound_root = root / "docs/research/kbound"
    ordered: list[Path] = []
    seen: set[Path] = set()
    input_re = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")

    def candidates(including: Path, token: str) -> tuple[Path, ...]:
        raw = Path(token.strip())
        variants = (raw,) if raw.suffix else (raw.with_suffix(".tex"), raw)
        bases = (including.parent, kbound_root, root)
        return tuple((base / variant).resolve() for base in bases for variant in variants)

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        ordered.append(resolved)
        if not resolved.is_file():
            return
        text = live_latex(resolved.read_text(errors="ignore"))
        for match in input_re.finditer(text):
            token = match.group(1).strip()
            if not token or "\\" in token or "#" in token:
                continue
            options = candidates(resolved, token)
            target = next((candidate for candidate in options if candidate.is_file()), options[0])
            visit(target)

    for relative in ACTIVE_DRIVER_RELATIVE_PATHS:
        visit(root / relative)
    return tuple(ordered)


def live_latex(text: str) -> str:
    """Remove disabled blocks and comments before checking assertion text."""
    text = re.sub(r"\\iffalse.*?\\fi", "", text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        lines.append(re.split(r"(?<!\\)%", line, maxsplit=1)[0])
    return "\n".join(lines)
