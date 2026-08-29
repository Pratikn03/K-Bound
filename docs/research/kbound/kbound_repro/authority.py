"""kbound_repro.authority -- the single authority chain for K-Bound claims.

Enforces the dependency chain::

    raw per-condition records
      -> validated per-seed summaries
      -> validated multiseed aggregates
      -> ONE canonical result manifest        (only numerical source)
      -> generated claim matrix                (ledger x manifest)
      -> tables / figures / PDFs

Two authorities, kept separate:

* the **claim ledger** is the only authority for *allowed wording and status*;
* the **result manifest** is the only *numerical* source for promoted results.

This module (a) generates the empirical claim matrix from ledger + manifest,
(b) provides semantic (not merely exact-string) guards against forbidden
wording, and (c) detects disagreement between the TODO docs, ledger, manifest,
claim matrix and manuscript so the release build can fail closed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "load_ledger",
    "index_claims",
    "build_claim_matrix",
    "forbidden_patterns",
    "scan_text_for_forbidden",
    "scan_text_for_unreleased_curated",
    "consistency_problems",
    "detect_disagreements",
    "PROMOTED_STATUSES",
    "MANIFEST_ELIGIBLE_STATUSES",
    "WITHDRAWN_STATUS",
]

PROMOTED_STATUSES = frozenset({"supported", "no-harm"})
MANIFEST_ELIGIBLE_STATUSES = frozenset({"supported", "no-harm", "descriptive", "diagnostic"})
WITHDRAWN_STATUS = "withdrawn"

# Curated semantic variants for the highest-risk withdrawn / bounded claims.
# Patterns run against *normalized* text (see _normalize): lowercase, hyphens and
# underscores -> spaces, punctuation stripped, whitespace collapsed.  This catches
# variants that a plain grep for the ledger's exact forbidden string would miss.
_EXTRA_FORBIDDEN: dict[str, list[str]] = {
    "KB-CLAIM-003": [r"guaranteed safe in the wild", r"assumption free", r"fa c\s*(<=|le)\s*alpha"],
    "KB-CLAIM-004": [r"fa c\s*(<=|le|bounded by)\s*alpha", r"conditional false adapt[^|]{0,20}(<=|le|bounded)\s*alpha"],
    # Assertive forms only: bare "jackknife" is a legitimate method name (the LOO
    # jackknife quantile) and "jackknife+ is not claimed" is a disclaimer.
    "KB-CLAIM-012": [
        r"jackknife\s*\+[^|]{0,30}(guarantee|holds|certificate|valid|coverage)",
        r"distribution free (guarantee|certificate|coverage|without assumptions)",
    ],
    "KB-CLAIM-021": [
        r"iwildcam[^|]{0,100}(tie|ties|no harm|zero (observed )?(adapt|adaptation)|0 adapts?|regret|false adapt|beats both|stable)",
        r"(promoted|primary)[^|]{0,50}iwildcam",
    ],
    "KB-CLAIM-022": [r"beats both[^|]{0,60}camelyon", r"camelyon[^|]{0,60}beats both"],
    "KB-CLAIM-023": [r"\b13\s*x\b", r"\b24\s*x\b", r"beats both mixed"],
    "KB-CLAIM-024": [r"\b13\s*x\b", r"\b24\s*x\b", r"universal mixed deployment", r"natural shift win"],
    "KB-CLAIM-026": [r"beats poem on natural shifts"],
    "KB-CLAIM-027": [r"natural multimodal sota", r"universal fusion win"],
    "KB-CLAIM-050": [r"universal (accuracy )?improvement", r"always beats adapt"],
}


def _normalize(text: str) -> str:
    """Lowercase, de-hyphenate, and mark sentence boundaries with ``|``.

    The ``|`` sentinel lets proximity patterns (``beats both[^|]{0,60}camelyon``)
    match only *within a sentence*, so an allowed "beats both" in one sentence
    and an unrelated "Camelyon" in the next do not produce a false positive.
    """
    t = text.lower().replace("-", " ").replace("_", " ")
    t = re.sub(r"[.;:!?]+", " | ", t)          # sentence-boundary sentinel
    t = re.sub(r"[^a-z0-9x<=|+\s]", " ", t)    # keep digits, x, <=, |, +, spaces
    return re.sub(r"[ \t]+", " ", t).strip()


def load_ledger(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def index_claims(ledger: dict) -> dict[str, dict]:
    return {c["claim_id"]: c for c in ledger.get("claims", [])}


def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _manifest_claim_ids(manifest: dict) -> set[str]:
    return {r.get("claim_id") for r in manifest.get("results", []) if r.get("claim_id")}


def build_claim_matrix(
    ledger: dict,
    manifest: dict | None = None,
    *,
    long_paper_only: Iterable[str] = (),
) -> list[dict]:
    """Generate the empirical claim matrix from ledger (+ optional manifest).

    Each row records the claim's ledger status, whether the canonical manifest
    carries a numerical entry for it, whether it is marked long-paper-only, and
    -- for supported/no-harm *empirical* claims -- whether that support is
    actually backed by a manifest entry (the promotion rule).
    """
    long_only = set(long_paper_only)
    in_manifest = _manifest_claim_ids(manifest) if manifest else set()
    manifest_statuses = {
        row.get("claim_id"): row.get("status")
        for row in (manifest or {}).get("results", [])
        if row.get("claim_id")
    }
    rows = []
    for cid, c in index_claims(ledger).items():
        status = c.get("status")
        ctype = c.get("claim_type")
        empirical = ctype == "empirical"
        promoted = status in PROMOTED_STATUSES
        backed = cid in in_manifest
        status_agrees = backed and manifest_statuses.get(cid) == status
        needs_backing = empirical and promoted and cid not in long_only
        rows.append({
            "claim_id": cid,
            "claim_type": ctype,
            "status": status,
            "in_manifest": backed,
            "manifest_status_agrees": status_agrees,
            "long_paper_only": cid in long_only,
            "promoted": promoted,
            "requires_manifest_backing": needs_backing,
            "backing_ok": (backed and status_agrees if needs_backing else True),
        })
    return sorted(rows, key=lambda r: r["claim_id"])


def forbidden_patterns(ledger: dict) -> dict[str, list[str]]:
    """Return, per claim, the list of forbidden normalized patterns to guard.

    Combines the ledger's own ``forbidden_wording`` with the curated semantic
    variants in ``_EXTRA_FORBIDDEN``.
    """
    out: dict[str, list[str]] = {}
    for cid, c in index_claims(ledger).items():
        pats = []
        for w in _as_list(c.get("forbidden_wording")):
            nw = _normalize(w)
            if not nw:
                continue
            # Tolerant spacing so a literal '+' (e.g. "jackknife+") matches
            # "jackknife +" too, but bare "jackknife" (the LOO method name) does
            # NOT match the "jackknife+" forbidden phrase.
            pats.append(re.escape(nw).replace(r"\+", r"\s*\+"))
        pats += _EXTRA_FORBIDDEN.get(cid, [])
        if pats:
            out[cid] = pats
    return out


# Disclaimer / negation cues: when one of these sits close to a forbidden-phrase
# match, the sentence is *disclaiming* the claim (allowed) rather than asserting
# it.  Padded with spaces so "not" never matches inside "another".
_DISCLAIMER_CUES = (
    " not ", " no ", " never ", " without ", " cannot ", " neither ", " nor ",
    " descriptive", " descriptively", " not claimed", " rather than ",
    " impossible", " only asymptotically", " asymptotically ", " not an exact",
    " no supporting", " does not", " do not", " isn t", " avoid", " withheld",
)


def _is_disclaimed(norm: str, start: int, end: int, window: int = 95) -> bool:
    """True if a negation/disclaimer cue sits within ``window`` chars of a match."""
    ctx = " " + norm[max(0, start - window): end + window] + " "
    return any(cue in ctx for cue in _DISCLAIMER_CUES)


def scan_text_for_forbidden(
    text: str, ledger: dict, *, only_withdrawn: bool = False, negation_aware: bool = True
) -> list[dict]:
    """Scan ``text`` for *assertive* forbidden wording; return a list of hits.

    ``only_withdrawn=True`` restricts the scan to withdrawn claims (used to assert
    that withdrawn claims are absent from promoted manuscript wording).
    ``negation_aware`` (default) suppresses matches that occur inside a
    disclaimer ("does not claim ...", "jackknife+ is not claimed", "FA_c is
    descriptive") so a paper can correctly state what it does NOT claim.
    """
    norm = _normalize(text)
    claims = index_claims(ledger)
    hits = []
    for cid, pats in forbidden_patterns(ledger).items():
        if only_withdrawn and claims.get(cid, {}).get("status") != WITHDRAWN_STATUS:
            continue
        for pat in pats:
            for m in re.finditer(pat, norm):
                if negation_aware and _is_disclaimed(norm, m.start(), m.end()):
                    continue
                start = max(0, m.start() - 30)
                hits.append({
                    "claim_id": cid,
                    "status": claims.get(cid, {}).get("status"),
                    "pattern": pat,
                    "snippet": norm[start:m.end() + 30],
                })
                break  # one hit per pattern is enough
    return hits


def scan_text_for_unreleased_curated(text: str, ledger: dict) -> list[dict]:
    """Scan contextual high-risk patterns for withheld or pending claims.

    Ledger phrases such as ``beats both`` are too generic to apply globally to
    every withheld claim.  This guard therefore uses only the curated,
    claim-contextual patterns in ``_EXTRA_FORBIDDEN`` and remains
    negation/withholding aware.
    """
    norm = _normalize(text)
    claims = index_claims(ledger)
    hits = []
    for cid, patterns in _EXTRA_FORBIDDEN.items():
        status = claims.get(cid, {}).get("status")
        if status not in {"withheld", "pending"}:
            continue
        for pattern in patterns:
            for match in re.finditer(pattern, norm):
                if _is_disclaimed(norm, match.start(), match.end()):
                    continue
                start = max(0, match.start() - 30)
                hits.append({
                    "claim_id": cid,
                    "status": status,
                    "pattern": pattern,
                    "snippet": norm[start:match.end() + 30],
                })
                break

    # KB-CLAIM-021 is stricter than ordinary wording restrictions: its entire
    # iWildCam result/action row is numerically ineligible.  A nearby word such
    # as "diagnostic" or "withheld" therefore cannot license printing the
    # values themselves.  Inspect the live source following each iWildCam
    # mention, using only the current table row for tabular material and the
    # current paragraph for prose.
    if claims.get("KB-CLAIM-021", {}).get("status") in {"withheld", "pending"}:
        for mention in re.finditer(r"\biwildcam\b", text, flags=re.IGNORECASE):
            line_start = text.rfind("\n", 0, mention.start()) + 1
            line_end = text.find("\n", mention.start())
            line_end = len(text) if line_end < 0 else line_end
            full_line = text[line_start:line_end]
            if "&" in full_line and "\\\\" in full_line:
                segment = text[mention.start():line_end]
                value_patterns = (
                    r"&\s*\d[\d,]*(?:\.\d+)?\s*&",
                    r"\\iw(?:n|adaptcount|freezecount|abstaincount)\b",
                )
            else:
                paragraph_end = re.search(r"\n\s*\n", text[mention.start():])
                paragraph_end_pos = (
                    mention.start() + paragraph_end.start()
                    if paragraph_end
                    else min(len(text), mention.start() + 1400)
                )
                paragraph = text[mention.start():paragraph_end_pos]
                sentence_end = re.search(r"[.!?](?:\s|$)", paragraph)
                segment = paragraph[:sentence_end.end()] if sentence_end else paragraph
                value_patterns = (
                    r"\b(?:regrets?|actions?|decision coverage|benefit|population|candidate records?|"
                    r"images?|sign flips?|n\s*=)\b[\s\S]{0,220}(?<![a-z])(?:\$?\s*[+-]?\d)",
                    r"(?<![a-z])(?:\$?\s*[+-]?\d[\d,.]*)[\s\S]{0,70}"
                    r"\b(?:candidate records?|images?|sign flips?)\b",
                    r"\b(?:diagnostic (?:recomputation|replay)|archived population|"
                    r"correction changes benefit)\b[\s\S]{0,260}(?<![a-z])(?:\$?\s*[+-]?\d)",
                )
                # The third pattern is paragraph-scoped because an iWildCam
                # paragraph can introduce an implied diagnostic sentence after
                # naming the dataset once. The other patterns remain bounded to
                # the iWildCam sentence so later results for another track do
                # not become false positives.
                paragraph_pattern = value_patterns[-1]
                if re.search(paragraph_pattern, paragraph, flags=re.IGNORECASE):
                    segment = paragraph
                    value_patterns = (paragraph_pattern,)
            for pattern in value_patterns:
                match = re.search(pattern, segment, flags=re.IGNORECASE)
                if match:
                    snippet = re.sub(r"\s+", " ", segment[: min(len(segment), 260)]).strip()
                    hits.append({
                        "claim_id": "KB-CLAIM-021",
                        "status": claims["KB-CLAIM-021"]["status"],
                        "pattern": pattern,
                        "snippet": snippet,
                    })
                    break
    return hits


def consistency_problems(
    ledger: dict,
    manifest: dict | None,
    *,
    long_paper_only: Iterable[str] = (),
) -> list[str]:
    """Return a list of consistency problems (empty == consistent).

    * Every supported/no-harm empirical claim must be present in the canonical
      manifest OR explicitly marked long-paper-only.
    * The manifest may carry only numerically eligible ledger states
      (supported, no-harm, descriptive, or diagnostic), and the row status must
      exactly match the ledger.
    * A manifest claim_id must exist in the ledger.
    * A manifest claim_id may occur only once.
    """
    problems: list[str] = []
    ledger_ids = [c.get("claim_id") for c in ledger.get("claims", []) if c.get("claim_id")]
    duplicate_ledger_ids = sorted(cid for cid, count in Counter(ledger_ids).items() if count > 1)
    if duplicate_ledger_ids:
        problems.append(f"claim ledger contains duplicate claim IDs: {duplicate_ledger_ids}.")
    claims = index_claims(ledger)
    # Backing can only be verified against an EXISTING manifest. When the
    # canonical manifest is absent, that is reported once by the caller
    # (WARN / fail-closed), not as a per-claim disagreement here.
    if manifest is not None:
        matrix = build_claim_matrix(ledger, manifest, long_paper_only=long_paper_only)
        for row in matrix:
            if row["requires_manifest_backing"] and not row["in_manifest"]:
                problems.append(
                    f"{row['claim_id']}: supported empirical claim is neither in the canonical "
                    f"manifest nor marked long-paper-only."
                )
    if manifest:
        seen_claim_ids: set[str] = set()
        for result in manifest.get("results", []):
            cid = result.get("claim_id")
            if not cid:
                continue  # schema validation owns the required-field error
            if cid in seen_claim_ids:
                problems.append(f"manifest contains duplicate claim_id {cid}.")
                continue
            seen_claim_ids.add(cid)
            if cid not in claims:
                problems.append(f"manifest references unknown claim {cid} (absent from ledger).")
                continue

            ledger_status = claims[cid].get("status")
            manifest_status = result.get("status")
            if claims[cid].get("claim_type") != "empirical":
                problems.append(
                    f"{cid}: numerical manifest references non-empirical ledger claim type "
                    f"{claims[cid].get('claim_type')!r}."
                )
            elif ledger_status not in MANIFEST_ELIGIBLE_STATUSES:
                problems.append(
                    f"{cid}: {ledger_status} in ledger but present in the numerical manifest."
                )
            elif manifest_status != ledger_status:
                problems.append(
                    f"{cid}: manifest status {manifest_status!r} disagrees with "
                    f"ledger status {ledger_status!r}."
                )
    return problems


def detect_disagreements(
    ledger: dict,
    *,
    manifest: dict | None = None,
    manuscript_texts: dict[str, str] | None = None,
    todo_texts: dict[str, str] | None = None,
    long_paper_only: Iterable[str] = (),
) -> list[str]:
    """Aggregate disagreements across ledger / manifest / manuscript / TODO docs.

    Returns a flat list of human-readable problems; an empty list means the
    sources agree and the release build may proceed.
    """
    problems = list(consistency_problems(ledger, manifest, long_paper_only=long_paper_only))
    for name, text in (manuscript_texts or {}).items():
        for hit in scan_text_for_forbidden(text, ledger, only_withdrawn=True):
            problems.append(
                f"{name}: forbidden wording for withdrawn {hit['claim_id']} "
                f"('...{hit['snippet']}...')."
            )
        for hit in scan_text_for_unreleased_curated(text, ledger):
            problems.append(
                f"{name}: forbidden wording for unreleased {hit['claim_id']} "
                f"('...{hit['snippet']}...')."
            )
    # TODO docs must not resurrect a withdrawn claim as promoted/open work.
    for name, text in (todo_texts or {}).items():
        for hit in scan_text_for_forbidden(text, ledger, only_withdrawn=True):
            problems.append(
                f"{name}: TODO text echoes forbidden wording for withdrawn "
                f"{hit['claim_id']} ('...{hit['snippet']}...')."
            )
    return problems
