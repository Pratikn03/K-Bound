#!/usr/bin/env python3
"""kbound_repro.release_checks -- authority-chain release gate.

Runs the fail-closed authority checks for the release build:

* validate the claim ledger against its schema;
* if a canonical result manifest is present, validate it and cross-check that
  every supported/no-harm empirical claim is backed (or long-paper-only), and
  that every row has a unique claim ID, matches the ledger status exactly, and
  no pending, withdrawn, or withheld claim leaks into numerical evidence;
* scan the *promoted* manuscript sources for forbidden wording of withdrawn
  claims (semantic guard, not just exact grep).

Exit codes:
    0  all checks passed
    1  a disagreement / forbidden-wording violation was found
    2  required evidence (canonical manifest) is incomplete/absent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from . import authority, manuscript_sources, paths, schema
else:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kbound_repro import authority, manuscript_sources, paths, schema


def _discover_promoted_tex(root: Path) -> list[str]:
    return [str(path) for path in manuscript_sources.active_source_paths(root)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--promoted-tex", nargs="*", default=None)
    ap.add_argument("--long-paper-only", nargs="*", default=[])
    ap.add_argument("--require-manifest", action="store_true",
                    help="fail closed (exit 2) if the canonical manifest is absent")
    args = ap.parse_args(argv)

    try:
        root = paths.find_repo_root()
    except Exception:
        root = Path.cwd()

    ledger_path = Path(args.ledger) if args.ledger else root / "docs/research/kbound/claim_ledger.json"
    if not ledger_path.exists():
        print(f"ERROR: claim ledger not found at {ledger_path}", file=sys.stderr)
        return 2
    ledger = authority.load_ledger(ledger_path)

    # 1) ledger schema
    try:
        schema.validate(ledger, "claim_ledger")
        print(f"OK: claim ledger validates ({len(ledger.get('claims', []))} claims).")
    except schema.SchemaError as exc:
        print(f"FAIL: claim ledger schema: {exc}", file=sys.stderr)
        return 1

    # 2) manifest (optional but required for a full release)
    manifest = None
    manifest_path = Path(args.manifest) if args.manifest else root / "docs/research/kbound/RESULT_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        try:
            schema.validate(manifest, "result_manifest")
            print(f"OK: result manifest validates ({len(manifest.get('results', []))} entries).")
        except schema.SchemaError as exc:
            print(f"FAIL: result manifest schema: {exc}", file=sys.stderr)
            return 1
    else:
        msg = f"canonical result manifest absent at {manifest_path}"
        if args.require_manifest:
            print(f"FAIL (fail-closed): {msg}. Promoted numerical results are unverifiable.",
                  file=sys.stderr)
            return 2
        print(f"WARN: {msg}; skipping numerical-backing checks.")

    # 3) consistency + disagreement
    tex = args.promoted_tex if args.promoted_tex is not None else _discover_promoted_tex(root)
    missing_tex = [t for t in tex if not Path(t).is_file()]
    if missing_tex:
        print("FAIL: maintained TeX dependency closure is incomplete:", file=sys.stderr)
        for path in missing_tex:
            print(f"  - {path}", file=sys.stderr)
        return 1
    manuscript_texts = {}
    for t in tex:
        try:
            manuscript_texts[str(Path(t).relative_to(root))] = manuscript_sources.live_latex(
                Path(t).read_text(errors="ignore")
            )
        except (OSError, ValueError):
            try:
                manuscript_texts[t] = manuscript_sources.live_latex(
                    Path(t).read_text(errors="ignore")
                )
            except OSError:
                pass

    problems = authority.detect_disagreements(
        ledger,
        manifest=manifest,
        manuscript_texts=manuscript_texts,
        long_paper_only=args.long_paper_only,
    )
    if problems:
        print(f"\nFAIL: {len(problems)} authority-chain disagreement(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"OK: authority chain consistent "
          f"(scanned {len(manuscript_texts)} promoted manuscript file(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
