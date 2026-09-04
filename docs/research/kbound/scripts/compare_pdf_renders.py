#!/usr/bin/env python3
"""Compare baseline and current K-Bound page renders pixel by pixel."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from docs.research.kbound.scripts.render_pdf_pages import (
        CURRENT_PDF_NAMES,
        require_binary,
    )
except ModuleNotFoundError:  # direct execution from the scripts directory
    from render_pdf_pages import CURRENT_PDF_NAMES, require_binary

ROOT = Path(__file__).resolve().parents[1]
PAGE_PATTERN = re.compile(r"^page-(\d+)\.png$")


@dataclass(frozen=True)
class PageComparison:
    page: int
    changed_pixels: int
    total_pixels: int

    @property
    def changed_fraction(self) -> float:
        return self.changed_pixels / self.total_pixels


def page_inventory(directory: Path) -> dict[int, Path]:
    pages: dict[int, Path] = {}
    for path in sorted(directory.glob("page-*.png")):
        match = PAGE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        page = int(match.group(1))
        if page in pages:
            raise ValueError(f"{directory}: duplicate rendered page {page}")
        pages[page] = path
    if not pages:
        raise ValueError(f"{directory}: no rendered pages found")
    return pages


def select_preview_pages(comparisons: list[PageComparison]) -> tuple[int, ...]:
    changed = [record for record in comparisons if record.changed_pixels]
    if not changed:
        return ()
    largest = max(changed, key=lambda record: (record.changed_fraction, -record.page))
    return tuple(dict.fromkeys((changed[0].page, largest.page, changed[-1].page)))


def _geometry(image: Path, magick: str) -> tuple[int, int]:
    result = subprocess.run(
        [magick, "identify", "-format", "%w %h", str(image)],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = (int(value) for value in result.stdout.split())
    return width, height


def _changed_pixels(before: Path, current: Path, magick: str) -> int:
    result = subprocess.run(
        [magick, "compare", "-metric", "AE", str(before), str(current), "null:"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"ImageMagick compare failed for {before} and {current}: {result.stderr.strip()}"
        )
    metric = result.stderr.strip().split()[0]
    try:
        return int(round(float(metric)))
    except ValueError as exc:
        raise RuntimeError(f"unrecognized ImageMagick AE metric: {result.stderr!r}") from exc


def _write_preview(
    before: Path,
    current: Path,
    destination: Path,
    magick: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kbound-render-diff-") as temp_dir:
        difference = Path(temp_dir) / "difference.png"
        result = subprocess.run(
            [
                magick,
                "compare",
                "-highlight-color",
                "#ff0055",
                "-lowlight-color",
                "#000000",
                str(before),
                str(current),
                str(difference),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(f"ImageMagick could not write {difference}: {result.stderr}")
        subprocess.run(
            [magick, str(before), str(current), str(difference), "+append", str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )


def compare_role(
    role: str,
    baseline_dir: Path,
    current_dir: Path,
    preview_root: Path,
    magick: str,
) -> dict[str, object]:
    baseline = page_inventory(baseline_dir)
    current = page_inventory(current_dir)
    shared_pages = sorted(set(baseline) & set(current))
    comparisons: list[PageComparison] = []
    for page in shared_pages:
        before_geometry = _geometry(baseline[page], magick)
        current_geometry = _geometry(current[page], magick)
        if before_geometry != current_geometry:
            raise RuntimeError(
                f"{role} page {page}: geometry changed from {before_geometry} to {current_geometry}"
            )
        width, height = current_geometry
        comparisons.append(
            PageComparison(
                page=page,
                changed_pixels=_changed_pixels(baseline[page], current[page], magick),
                total_pixels=width * height,
            )
        )

    previews: list[str] = []
    for page in select_preview_pages(comparisons):
        destination = preview_root / role / f"page-{page:03d}-before-current-diff.png"
        _write_preview(baseline[page], current[page], destination, magick)
        previews.append(destination.relative_to(preview_root.parent).as_posix())

    changed = [record for record in comparisons if record.changed_pixels]
    return {
        "role": role,
        "baseline_pages": len(baseline),
        "current_pages": len(current),
        "compared_pages": len(comparisons),
        "changed_pages": len(changed),
        "unchanged_pages": len(comparisons) - len(changed),
        "added_pages": sorted(set(current) - set(baseline)),
        "removed_pages": sorted(set(baseline) - set(current)),
        "changed_pixels": sum(record.changed_pixels for record in comparisons),
        "compared_pixels": sum(record.total_pixels for record in comparisons),
        "maximum_changed_fraction": max(
            (record.changed_fraction for record in comparisons), default=0.0
        ),
        "preview_pages": list(select_preview_pages(comparisons)),
        "preview_files": previews,
        "pages": [asdict(record) for record in comparisons],
    }


def _write_markdown(results: list[dict[str, object]], destination: Path) -> None:
    lines = [
        "# K-Bound baseline-to-current render comparison",
        "",
        "This report is generated from 192-DPI full-page PNG renders. Pixel differences are evidence of layout or content change, not by themselves defects.",
        "",
        "| Release role | Baseline pages | Current pages | Compared | Changed | Added | Removed | Max changed area |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for result in results:
        added = ", ".join(str(value) for value in result["added_pages"]) or "none"
        removed = ", ".join(str(value) for value in result["removed_pages"]) or "none"
        lines.append(
            "| {role} | {baseline_pages} | {current_pages} | {compared_pages} | "
            "{changed_pages} | {added} | {removed} | {maximum_changed_fraction:.2%} |".format(
                **result,
                added=added,
                removed=removed,
            )
        )
    lines.extend(
        [
            "",
            "## Deterministic preview selection",
            "",
            "For each role, previews contain the first changed page, the page with the largest changed-pixel fraction, and the last changed page (duplicates removed). Each image is baseline | current | highlighted pixel difference.",
            "",
        ]
    )
    for result in results:
        pages = ", ".join(str(page) for page in result["preview_pages"]) or "none"
        lines.append(f"- `{result['role']}`: pages {pages}.")
    lines.extend(
        [
            "",
            "The machine-readable per-page metrics are in `render_diff_summary.json`.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "paper/reports/baseline_renders",
    )
    parser.add_argument(
        "--current-root",
        type=Path,
        default=ROOT / "paper/reports/final_renders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "paper/reports/render_diffs",
    )
    args = parser.parse_args()
    magick = require_binary("magick")
    output_root = args.output_root.resolve()
    preview_root = output_root / "previews"
    if preview_root.exists():
        shutil.rmtree(preview_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for pdf_name in CURRENT_PDF_NAMES:
        role = Path(pdf_name).stem
        results.append(
            compare_role(
                role,
                args.baseline_root.resolve() / f"{role}.before",
                args.current_root.resolve() / role,
                preview_root,
                magick,
            )
        )
    payload = {"render_dpi": 192, "roles": results}
    (output_root / "render_diff_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(results, output_root / "VISUAL_RENDER_COMPARISON.md")
    print(output_root / "VISUAL_RENDER_COMPARISON.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
