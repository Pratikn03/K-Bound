#!/usr/bin/env python3
"""Fail when maintained K-Bound manuscripts reintroduce known claim drift."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KBOUND = ROOT / "docs/research/kbound"
sys.path.insert(0, str(KBOUND))

from kbound_repro import authority, manuscript_sources  # noqa: E402

CANONICAL = ROOT / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
SOURCE_MANIFEST = ROOT / "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"
CURRENT_CLUSTER = ROOT / "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json"
GENERATED_MANIFEST = KBOUND / "paper/generated/kbound_result_manifest.json"
KBOUND_NUMBERS = KBOUND / "paper/generated/kbound_numbers.tex"
CURRENT_CLUSTER_TABLE = KBOUND / "paper/generated/current_policy_family_sensitivity.tex"
UNIFORM_VERDICTS = KBOUND / "paper/generated/uniform_verdicts.json"
DECISION_METRICS = KBOUND / "paper/generated/empirical_audit/decision_metrics.json"
CLAIM_MATRIX = KBOUND / "paper/generated/empirical_audit/claim_matrix.md"
CLAIM_LEDGER = KBOUND / "claim_ledger.json"
RESULT_MANIFEST = KBOUND / "RESULT_MANIFEST.json"
RESULTS_SOURCE = KBOUND / "results_source.json"
HISTORICAL_LEDGER = KBOUND / "SUBMISSION_LEDGER.md"
RESULT_AUDIT = KBOUND / "KBOUND_SHORT_RESULT_AUDIT.md"
CLAIM_MANIFEST = KBOUND / "KBOUND_SHORT_CLAIM_MANIFEST.md"
README = KBOUND / "README.md"
LONG_TMLR = KBOUND / "kbound_tmlr.tex"
STORAGE_MANIFEST = KBOUND / "STORAGE_MANIFEST.json"
LOCK_SEAL = ROOT / "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json"
CCT20_RELEASE_MANIFEST = KBOUND / "paper/generated/cct20_release_manifest.json"
CCT20_RELEASE_BUILDER = KBOUND / "scripts/build_cct20_release.py"
REQUIRED_CCT20_UPSTREAM_KEYS = {
    "checkpoint_audit",
    "development_gate",
    "development_trace_collection",
    "development_traces",
    "execution_dependencies",
    "execution_seal",
    "one_shot_score",
    "one_shot_scoring_marker",
    "prediction_actions",
    "prediction_cells",
    "prediction_collection",
    "release_generator",
    "two_way_inference",
}
CCT20_VERDICT_CLAIMS = {
    "CONFIRMATORY_STRONG_SUCCESS": (
        "The prospective CCT-20 result satisfies both simultaneous and exact-test criteria, "
        "passes action-exposure thresholds, and contains both helpful and harmful adaptation cases."
    ),
    "CONFIRMATORY_PRIMARY_SUCCESS_MIXED_EFFECTS_MISSING": (
        "The prospective CCT-20 primary contrasts pass the locked criteria, but the expanded "
        "mixed helpful/harmful evidence requirement is not met."
    ),
    "SAFE_UTILITY_ONLY": (
        "The prospective CCT-20 result passes only the locked safe-utility check; it does not "
        "establish the preregistered strong-success claim."
    ),
    "NO_CONFIRMATORY_SUCCESS": (
        "The prospective CCT-20 result does not satisfy the preregistered strong-success or "
        "safe-utility criteria; the complete result is reported without promotion."
    ),
}
REQUIRED_CCT20_NUMBER_MACROS = {
    "CCTAdaptCount",
    "CCTFreezeCount",
    "CCTAbstainCount",
    "CCTStrictDecisionCoverage",
    "CCTFalseAdaptCount",
    "CCTFalseAdaptRate",
    "CCTSafeUtilityMargin",
    "CCTSafeVsAdaptPoint",
    "CCTSafeVsAdaptCILower",
    "CCTSafeVsAdaptCIUpper",
    "CCTSafeVsFreezePoint",
    "CCTSafeVsFreezeCILower",
    "CCTSafeVsFreezeCIUpper",
    "CCTSafeUtilityPass",
    "CCTVerdict",
    "CCTManuscriptClaim",
    "CCTSecondaryMetricDisclosure",
    "CCTInferenceSHA",
}
ACTIVE_SOURCES = manuscript_sources.active_source_paths(ROOT)


live_latex = manuscript_sources.live_latex


def row_by_track(rows: list[dict], track: str) -> dict:
    matches = [row for row in rows if row.get("track") == track]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {track!r} row, found {len(matches)}")
    return matches[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_source_paths(value: object):
    if isinstance(value, dict):
        source = value.get("source")
        if isinstance(source, str):
            yield source
        for child in value.values():
            yield from iter_source_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_source_paths(child)


def _strip_tex_comments_for_claims(text: str) -> str:
    """Apply TeX comment semantics so comments cannot split forbidden wording."""

    pieces: list[str] = []
    for line in text.splitlines(keepends=True):
        comment_at: int | None = None
        for index, character in enumerate(line):
            if character != "%":
                continue
            preceding_backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                preceding_backslashes += 1
                cursor -= 1
            if preceding_backslashes % 2 == 0:
                comment_at = index
                break
        if comment_at is None:
            pieces.append(line)
        else:
            # An unescaped percent comments out the physical newline as well.
            pieces.append(line[:comment_at])
    return "".join(pieces)


def _normalize_claim_text(text: str) -> str:
    """Collapse harmless TeX/Unicode variation without changing claim polarity."""

    normalized = _strip_tex_comments_for_claims(text)
    normalized = normalized.replace("~", " ").replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]", "", normalized)
    formatting = r"(?:text(?:bf|it|tt|sc|rm|normal)|emph|mbox|textrm)"
    previous = None
    while normalized != previous:
        previous = normalized
        normalized = re.sub(
            rf"\\{formatting}\s*\{{([^{{}}]*)\}}",
            r"\1",
            normalized,
        )
    normalized = re.sub(
        r"\\(?:allowbreak|nobreak|protect|relax|/)(?![A-Za-z@])",
        "",
        normalized,
    )
    normalized = re.sub(r"\\(?:hspace|kern|mkern)\*?\s*\{[^{}]*\}", "", normalized)
    normalized = normalized.replace("{", " ").replace("}", " ")
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def _cct20_claim_contexts(text: str) -> str:
    """Return top-level manuscript sections that mention the CCT-20 campaign.

    Restricting these rules to CCT-bearing sections prevents a CCT result from
    accidentally changing the interpretation of the paper's general theorem
    statements about conformal coverage.
    """

    marker = re.compile(r"\b(?:cct\s*[- ]?20|caltech camera traps?\s*[- ]?20)\b", re.IGNORECASE)

    def mentions_cct(fragment: str) -> bool:
        return marker.search(_normalize_claim_text(fragment)) is not None

    section_starts = list(re.finditer(r"\\section\*?\s*\{", text))
    if not section_starts:
        return text if mentions_cct(text) else ""

    contexts: list[str] = []
    boundaries = [match.start() for match in section_starts] + [len(text)]
    preamble = text[: boundaries[0]]
    if mentions_cct(preamble):
        contexts.append(preamble)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        section = text[start:end]
        if mentions_cct(section):
            contexts.append(section)
    return "\n".join(contexts)


def _claim_clause(text: str, start: int, end: int) -> str:
    """Return the punctuation-bounded clause surrounding one phrase."""

    left = max(text.rfind(token, 0, start) for token in (".", "!", "?", ";", ",", ":"))
    right_candidates = [
        position for token in (".", "!", "?", ";", ",", ":") if (position := text.find(token, end)) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(text)
    for boundary in re.finditer(r"\b(?:but|yet|however|nevertheless|nonetheless)\b", text, flags=re.IGNORECASE):
        if boundary.end() <= start:
            left = max(left, boundary.end() - 1)
        elif boundary.start() >= end:
            right = min(right, boundary.start())
    return text[left + 1 : right]


def _has_unsafe_match(
    text: str,
    claim_pattern: str,
    *,
    safe_patterns: tuple[str, ...] = (),
) -> bool:
    for match in re.finditer(claim_pattern, text, flags=re.IGNORECASE):
        clause = _claim_clause(text, match.start(), match.end())
        if not any(re.search(pattern, clause, flags=re.IGNORECASE) for pattern in safe_patterns):
            return True
    return False


def _cct20_completed_result_claimed(text: str) -> bool:
    """Detect promotion of an observed CCT result, not a future protocol."""

    completed_patterns = (
        r"\\(?:input|include)\s*(?:\{\s*)?[^{}\s]*"
        r"cct20_(?:numbers|primary_table|location_effects)(?:\.tex)?(?:\s*\})?",
        r"\bcompleted\s+(?:cct\s*[- ]?20|caltech camera traps?\s*[- ]?20)\s+"
        r"(?:evaluation|experiment|analysis|run|result)",
        r"\b(?:cct\s*[- ]?20|caltech camera traps?\s*[- ]?20)\s+"
        r"(?:evaluation|experiment|analysis|run)\s+(?:is|was|has been)\s+completed\b",
        r"\b(?:we|this paper)\s+report(?:s|ed)?\s+(?:a\s+|the\s+)?(?:completed\s+)?"
        r"(?:cct\s*[- ]?20|caltech camera traps?\s*[- ]?20)\b",
        r"\b(?:cct\s*[- ]?20|caltech camera traps?\s*[- ]?20)\s+results?\s+"
        r"(?:show|demonstrate|yield|establish|confirm|were|are)\b",
        r"\b(?:cct\s*[- ]?20|caltech camera traps?\s*[- ]?20)\s+"
        r"(?:result|outcome|finding)\s+(?:is|was)\s+"
        r"(?:positive|negative|mixed|complete|a\s+(?:win|tie|loss)|reported)\b",
        r"\bwe\s+(?:evaluated|tested|ran)\s+(?:the\s+)?"
        r"(?:cct\s*[- ]?20|caltech camera traps?\s*[- ]?20)\b",
        r"\bon\s+(?:cct\s*[- ]?20|caltech camera traps?\s*[- ]?20)\b[^.]{0,180}"
        r"\b(?:was|were|achieved|yielded|observed|measured)\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in completed_patterns)


def _artifact_path(raw_path: str, repository_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repository_root / path


def _stable_json_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _cct20_tex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _expected_cct20_verdict_code(
    *,
    protocol_strong_success: bool,
    expanded_mixed_effects_success: bool,
    safe_utility_passes: bool,
) -> str:
    if expanded_mixed_effects_success:
        return "CONFIRMATORY_STRONG_SUCCESS"
    if protocol_strong_success:
        return "CONFIRMATORY_PRIMARY_SUCCESS_MIXED_EFFECTS_MISSING"
    if safe_utility_passes:
        return "SAFE_UTILITY_ONLY"
    return "NO_CONFIRMATORY_SUCCESS"


def _iter_hash_bindings(value: object, label: str = "upstream_artifacts"):
    """Yield nested ``{path, sha256}`` bindings from a release manifest."""

    if isinstance(value, dict):
        if "path" in value or "sha256" in value:
            yield label, value
            return
        for key, child in value.items():
            yield from _iter_hash_bindings(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_hash_bindings(child, f"{label}[{index}]")


def _validate_hash_binding(
    problems: list[str],
    *,
    label: str,
    binding: dict,
    repository_root: Path,
    require_canonical_absolute_path: bool = False,
) -> bool:
    raw_path = binding.get("path")
    expected_sha256 = binding.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or raw_path.startswith("$")
        or any(token in raw_path for token in ("*", "?", "["))
    ):
        problems.append(f"CCT-20 release {label} lacks a literal artifact path")
        return False
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        problems.append(f"CCT-20 release {label} lacks a lowercase SHA-256")
        return False

    path = _artifact_path(raw_path, repository_root)
    if require_canonical_absolute_path:
        if not Path(raw_path).is_absolute() or str(path.resolve()) != raw_path:
            problems.append(f"CCT-20 release {label} path is not canonical and absolute: {raw_path}")
            return False
    if path.is_symlink():
        problems.append(f"CCT-20 release {label} artifact path is a symlink: {raw_path}")
        return False
    if not path.is_file():
        problems.append(f"CCT-20 release {label} artifact is missing: {raw_path}")
        return False
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        problems.append(
            f"CCT-20 release {label} SHA-256 mismatch for {raw_path}: expected {expected_sha256}, got {observed_sha256}"
        )
        return False

    size_bytes = binding.get("size_bytes")
    bytes_alias = binding.get("bytes")
    if size_bytes is not None and bytes_alias is not None and size_bytes != bytes_alias:
        problems.append(f"CCT-20 release {label} has conflicting byte counts")
        return False
    expected_size = size_bytes if size_bytes is not None else bytes_alias
    if expected_size is not None:
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            problems.append(f"CCT-20 release {label} has invalid size_bytes")
            return False
        if path.stat().st_size != expected_size:
            problems.append(
                f"CCT-20 release {label} size mismatch for {raw_path}: "
                f"expected {expected_size}, got {path.stat().st_size}"
            )
            return False
    return True


def _load_json_object_for_release(problems: list[str], path: Path, *, label: str) -> dict | None:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        problems.append(f"CCT-20 release {label} is not readable strict JSON: {exc}")
        return None
    if not isinstance(value, dict):
        problems.append(f"CCT-20 release {label} must be a JSON object")
        return None
    return value


def _validate_received_identity(
    problems: list[str],
    *,
    label: str,
    binding: dict,
    repository_root: Path,
) -> None:
    """Verify the receipt half of one builder-emitted received identity."""

    receipt_path_raw = binding.get("receipt_path")
    receipt_sha256 = binding.get("receipt_sha256")
    canonical_document_sha256 = binding.get("canonical_document_sha256")
    has_receipt_field = "receipt_path" in binding or "receipt_sha256" in binding
    if not has_receipt_field and "canonical_document_sha256" not in binding:
        return
    if not has_receipt_field:
        if (
            not isinstance(canonical_document_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", canonical_document_sha256) is None
        ):
            problems.append(f"CCT-20 release {label} has an invalid canonical document SHA-256")
            return
        artifact_path = _artifact_path(str(binding.get("path", "")), repository_root)
        artifact_document = _load_json_object_for_release(
            problems,
            artifact_path,
            label=label,
        )
        if artifact_document is not None and _stable_json_sha256(artifact_document) != canonical_document_sha256:
            problems.append(f"CCT-20 release {label} canonical document SHA-256 mismatch")
        return
    if not all(
        isinstance(value, str) and value for value in (receipt_path_raw, receipt_sha256, canonical_document_sha256)
    ):
        problems.append(f"CCT-20 release {label} has an incomplete receipt identity")
        return
    if re.fullmatch(r"[0-9a-f]{64}", canonical_document_sha256) is None:
        problems.append(f"CCT-20 release {label} has an invalid canonical document SHA-256")
        return
    receipt_binding = {
        "path": receipt_path_raw,
        "sha256": receipt_sha256,
    }
    if not _validate_hash_binding(
        problems,
        label=f"{label}.receipt",
        binding=receipt_binding,
        repository_root=repository_root,
        require_canonical_absolute_path=True,
    ):
        return
    artifact_path = _artifact_path(str(binding.get("path", "")), repository_root)
    expected_receipt_path = artifact_path.with_name(artifact_path.name + ".receipt.json")
    receipt_path = _artifact_path(receipt_path_raw, repository_root)
    if receipt_path != expected_receipt_path:
        problems.append(f"CCT-20 release {label} receipt path is not adjacent to its artifact")
        return
    receipt = _load_json_object_for_release(
        problems,
        receipt_path,
        label=f"{label}.receipt",
    )
    if receipt is None:
        return
    expected_receipt = {
        "schema": "kbound_cct20_artifact_receipt_v1",
        "artifact_path": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size if artifact_path.is_file() else None,
        "artifact_sha256": binding.get("sha256"),
        "canonical_document_sha256": canonical_document_sha256,
    }
    for field, expected in expected_receipt.items():
        if receipt.get(field) != expected:
            problems.append(f"CCT-20 release {label} receipt {field} disagrees with its identity")
    artifact_document = _load_json_object_for_release(
        problems,
        artifact_path,
        label=label,
    )
    if artifact_document is not None:
        try:
            observed_canonical = _stable_json_sha256(artifact_document)
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            problems.append(f"CCT-20 release {label} is not canonicalizable: {exc}")
        else:
            if observed_canonical != canonical_document_sha256:
                problems.append(f"CCT-20 release {label} canonical document SHA-256 mismatch")


def _validate_counted_release_bundle(
    problems: list[str],
    *,
    label: str,
    bundle: object,
    expected_count: int,
    ordering_key,
) -> None:
    if not isinstance(bundle, dict):
        problems.append(f"CCT-20 release {label} ledger is missing")
        return
    items = bundle.get("items")
    if (
        bundle.get("count") != expected_count
        or not isinstance(items, list)
        or len(items) != expected_count
        or not all(isinstance(row, dict) for row in items)
    ):
        problems.append(f"CCT-20 release {label} ledger does not contain {expected_count} items")
        return
    try:
        observed_aggregate = _stable_json_sha256(items)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        problems.append(f"CCT-20 release {label} ledger is not canonicalizable: {exc}")
        return
    if bundle.get("aggregate_sha256") != observed_aggregate:
        problems.append(f"CCT-20 release {label} aggregate SHA-256 is stale")
    if items != sorted(items, key=ordering_key):
        problems.append(f"CCT-20 release {label} ledger ordering differs from the builder")
    paths = [row.get("path") for row in items]
    if len(set(paths)) != expected_count or any(not isinstance(path, str) for path in paths):
        problems.append(f"CCT-20 release {label} ledger has duplicate or invalid paths")


def _validate_execution_dependency_bundle(problems: list[str], bundle: object) -> None:
    if not isinstance(bundle, dict):
        problems.append("CCT-20 release execution dependency ledger is missing")
        return
    dataset_items = bundle.get("dataset_items")
    code_items = bundle.get("code_items")
    if not isinstance(dataset_items, list) or not isinstance(code_items, list):
        problems.append("CCT-20 release execution dependency items are missing")
        return
    if (
        bundle.get("dataset_count") != 4
        or len(dataset_items) != 4
        or bundle.get("code_count") != 138
        or len(code_items) != 138
        or bundle.get("total_count") != 142
    ):
        problems.append("CCT-20 release execution dependency counts differ from 4 + 138")
    if not all(isinstance(row, dict) for row in (*dataset_items, *code_items)):
        problems.append("CCT-20 release execution dependency ledger contains a non-object")
        return
    if dataset_items != sorted(dataset_items, key=lambda row: str(row.get("name", ""))):
        problems.append("CCT-20 release dataset dependency ordering differs from the builder")
    if code_items != sorted(code_items, key=lambda row: str(row.get("name", ""))):
        problems.append("CCT-20 release code dependency ordering differs from the builder")
    names = [str(row.get("name", "")) for row in (*dataset_items, *code_items)]
    if "" in names or len(set(names)) != len(names):
        problems.append("CCT-20 release execution dependency names are empty or duplicated")
    expected_aggregate = _stable_json_sha256(
        {
            "dataset_dependencies": dataset_items,
            "code_dependencies": code_items,
        }
    )
    if bundle.get("aggregate_sha256") != expected_aggregate:
        problems.append("CCT-20 release execution dependency aggregate SHA-256 is stale")


def _validate_one_shot_marker_binding(
    problems: list[str],
    *,
    upstream: dict,
    repository_root: Path,
) -> None:
    """Cross-bind the spent scoring marker to the exact release inputs."""

    required = {
        name: upstream.get(name)
        for name in (
            "execution_seal",
            "prediction_collection",
            "prediction_cells",
            "one_shot_score",
            "one_shot_scoring_marker",
        )
    }
    if not all(isinstance(value, dict) for value in required.values()):
        return
    marker_binding = required["one_shot_scoring_marker"]
    collection_binding = required["prediction_collection"]
    score_binding = required["one_shot_score"]
    marker = _load_json_object_for_release(
        problems,
        _artifact_path(str(marker_binding.get("path", "")), repository_root),
        label="one-shot scoring marker",
    )
    collection = _load_json_object_for_release(
        problems,
        _artifact_path(str(collection_binding.get("path", "")), repository_root),
        label="prediction collection",
    )
    score = _load_json_object_for_release(
        problems,
        _artifact_path(str(score_binding.get("path", "")), repository_root),
        label="one-shot score",
    )
    cell_rows = required["prediction_cells"].get("items")
    if marker is None or collection is None or score is None or not isinstance(cell_rows, list):
        return
    cell_hashes: list[str] = []
    for index, binding in enumerate(cell_rows):
        if not isinstance(binding, dict):
            return
        cell = _load_json_object_for_release(
            problems,
            _artifact_path(str(binding.get("path", "")), repository_root),
            label=f"prediction cell {index}",
        )
        if cell is None:
            return
        cell_sha256 = cell.get("cell_sha256")
        if not isinstance(cell_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", cell_sha256) is None:
            problems.append(f"CCT-20 release prediction cell {index} lacks cell_sha256")
            return
        cell_hashes.append(cell_sha256)
    expected_request = {
        "execution_seal_artifact_sha256": required["execution_seal"].get("sha256"),
        "prediction_collection_sha256": collection.get("collection_sha256"),
        "prediction_cell_sha256": sorted(cell_hashes),
        "output_path": str(_artifact_path(str(score_binding.get("path", "")), repository_root).resolve()),
        "expected_target_images": 23_275,
        "label_contract": "set_membership_top1_and_16_indicator_multilabel_macro_f1",
    }
    if (
        set(marker) != {"schema", "status", "request", "request_sha256"}
        or marker.get("schema") != "kbound_cct20_one_shot_score_marker_v1"
        or marker.get("status") != "SPENT_BEFORE_GROUND_TRUTH_LOAD"
        or marker.get("request") != expected_request
        or marker.get("request_sha256") != _stable_json_sha256(expected_request)
    ):
        problems.append("CCT-20 release one-shot scoring marker differs from the release chain")
    expected_score_fields = {
        "execution_seal_artifact_sha256": expected_request["execution_seal_artifact_sha256"],
        "prediction_collection_sha256": expected_request["prediction_collection_sha256"],
        "target_image_count": 23_275,
        "checkpoint_count": 5,
        "location_count": 9,
        "cell_count": 45,
    }
    for field, expected in expected_score_fields.items():
        if score.get(field) != expected:
            problems.append(f"CCT-20 release one-shot score has stale {field}")


def validate_cct20_release_manifest(
    manifest_path: Path = CCT20_RELEASE_MANIFEST,
    *,
    repository_root: Path = ROOT,
) -> tuple[dict | None, list[str]]:
    """Validate the manifest required by a promoted, completed CCT result."""

    problems: list[str] = []
    manifest_path = Path(manifest_path).expanduser().resolve()
    if manifest_path.name != "cct20_release_manifest.json":
        problems.append("completed CCT-20 release manifest has a non-canonical filename")
    try:
        document = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-standard JSON constant {token}")),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"completed CCT-20 result lacks a readable release manifest: {exc}"]
    if not isinstance(document, dict):
        return None, ["completed CCT-20 release manifest must be a JSON object"]
    if manifest_path.stat().st_mode & 0o222:
        problems.append("completed CCT-20 release manifest is writable, not immutable")
    if document.get("schema") != "kbound_cct20_release_manifest_v1":
        problems.append("completed CCT-20 release manifest has an unknown schema")
    if document.get("status") != "RELEASE_COMPLETE":
        problems.append("completed CCT-20 release manifest is not marked RELEASE_COMPLETE")
    if document.get("artifacts_complete") is not True:
        problems.append("completed CCT-20 release manifest does not mark artifacts complete")

    release_sha256 = document.get("release_sha256")
    unsigned = dict(document)
    unsigned.pop("release_sha256", None)
    try:
        expected_release_sha256 = _stable_json_sha256(unsigned)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        problems.append(f"completed CCT-20 release manifest is not canonicalizable: {exc}")
    else:
        if release_sha256 != expected_release_sha256:
            problems.append("completed CCT-20 release manifest has a stale release_sha256")

    expected_disclosure = (
        "outcome-unopened before model execution; aggregate target metadata had already been "
        "inspected during candidate ranking, so this is not described as literally label-unopened"
    )
    if document.get("prospective_disclosure") != expected_disclosure:
        problems.append("completed CCT-20 release manifest has a stale prospective disclosure")
    expected_signs = {
        "adaptation_benefit": "adapted_accuracy_minus_frozen_accuracy",
        "primary_contrast": "baseline_regret_minus_kga_regret; positive_favors_kga",
        "helpful": "adaptation_benefit > 0",
        "zero": "adaptation_benefit = 0",
        "harmful": "adaptation_benefit < 0",
        "target_selection_lock_nonpositive_boundary": (
            "adaptation_benefit <= 0; the release separates exact zero from strictly harmful"
        ),
    }
    if document.get("sign_conventions") != expected_signs:
        problems.append("completed CCT-20 release manifest has stale sign conventions")
    design = document.get("design")
    if not isinstance(design, dict) or any(
        design.get(field) != expected
        for field, expected in {
            "checkpoint_count": 5,
            "location_cluster_count": 9,
            "matrix_shape": [5, 9],
            "cell_count": 45,
            "cluster_unit_for_exact_test": "camera_location",
            "independent_checkpoint_tensor_identities_verified": True,
        }.items()
    ):
        problems.append("completed CCT-20 release manifest does not describe the locked 5 x 9 design")
    comparisons = document.get("primary_comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != {
        "versus_always_adapt",
        "versus_always_freeze",
    }:
        problems.append("completed CCT-20 release manifest lacks the two locked comparisons")
        comparisons = {}
    else:
        for comparison, comparator in (
            ("versus_always_adapt", "always_adapt"),
            ("versus_always_freeze", "always_freeze"),
        ):
            evidence = comparisons.get(comparison)
            pointwise = evidence.get("pointwise_95_ci") if isinstance(evidence, dict) else None
            simultaneous = (
                evidence.get("simultaneous_bonferroni_97_5_ci")
                if isinstance(evidence, dict)
                else None
            )
            if (
                not isinstance(evidence, dict)
                or evidence.get("comparator") != comparator
                or not _is_finite_number(evidence.get("point_estimate"))
                or not isinstance(pointwise, list)
                or len(pointwise) != 2
                or not all(_is_finite_number(value) for value in pointwise)
                or float(pointwise[0]) > float(pointwise[1])
                or evidence.get("simultaneous_bonferroni_confidence_level") != 0.975
                or not isinstance(simultaneous, list)
                or len(simultaneous) != 2
                or not all(_is_finite_number(value) for value in simultaneous)
                or float(simultaneous[0]) > float(simultaneous[1])
                or not _is_finite_number(evidence.get("exact_location_sign_flip_p_one_sided"))
                or not _is_finite_number(evidence.get("holm_adjusted_p"))
                or not isinstance(evidence.get("holm_reject_at_familywise_0_05"), bool)
            ):
                problems.append(
                    f"completed CCT-20 release manifest has invalid {comparator} primary evidence"
                )
                continue
            raw_p = float(evidence["exact_location_sign_flip_p_one_sided"])
            holm_p = float(evidence["holm_adjusted_p"])
            if not 0.0 <= raw_p <= holm_p <= 1.0:
                problems.append(
                    f"completed CCT-20 release manifest has invalid {comparator} raw/Holm p-values"
                )
    exposure = document.get("action_exposure")
    exposure_counts = exposure.get("counts") if isinstance(exposure, dict) else None
    if (
        not isinstance(exposure_counts, dict)
        or set(exposure_counts) != {"ADAPT", "FREEZE", "ABSTAIN"}
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in exposure_counts.values()
        )
        or sum(exposure_counts.values()) != 45
    ):
        problems.append("completed CCT-20 release manifest action exposure does not cover 45 cells")

    false_adapt = document.get("false_adapt_accounting")
    false_count = false_adapt.get("false_adapt_count") if isinstance(false_adapt, dict) else None
    false_rate = false_adapt.get("false_adapt_rate_unconditional") if isinstance(false_adapt, dict) else None
    conditional_rate = (
        false_adapt.get("false_adapt_rate_conditional") if isinstance(false_adapt, dict) else None
    )
    adapt_count = exposure_counts.get("ADAPT") if isinstance(exposure_counts, dict) else None
    if (
        not isinstance(false_adapt, dict)
        or false_adapt.get("event") != "decision == ADAPT and adaptation_benefit <= 0"
        or false_adapt.get("unit") != "checkpoint_x_location"
        or false_adapt.get("denominator_all_cells") != 45
        or false_adapt.get("adapt_count") != adapt_count
        or isinstance(false_count, bool)
        or not isinstance(false_count, int)
        or false_count < 0
        or not isinstance(adapt_count, int)
        or false_count > adapt_count
        or isinstance(false_rate, bool)
        or not isinstance(false_rate, (int, float))
        or not math.isfinite(float(false_rate))
        or not math.isclose(float(false_rate), false_count / 45.0, abs_tol=1e-15)
        or (
            adapt_count == 0
            and conditional_rate is not None
        )
        or (
            adapt_count > 0
            and (
                not _is_finite_number(conditional_rate)
                or not math.isclose(
                    float(conditional_rate),
                    false_count / adapt_count,
                    abs_tol=1e-15,
                )
            )
        )
    ):
        problems.append("completed CCT-20 release manifest has invalid false-adapt accounting")

    safe_utility = document.get("safe_utility")
    safe_comparisons = (
        ("versus_always_adapt", "always-adapt"),
        ("versus_always_freeze", "always-freeze"),
    )
    if (
        not isinstance(safe_utility, dict)
        or safe_utility.get("contrast_sign")
        != "baseline_regret_minus_kga_regret; positive_favors_kga"
        or not _is_finite_number(safe_utility.get("frozen_noninferiority_margin"))
        or not math.isclose(
            float(safe_utility.get("frozen_noninferiority_margin", math.nan)),
            -0.005,
            abs_tol=1e-15,
        )
        or not isinstance(safe_utility.get("passes"), bool)
    ):
        problems.append("completed CCT-20 release manifest has invalid safe-utility evidence")
    else:
        safe_evidence_valid = True
        for comparison, label in safe_comparisons:
            evidence = safe_utility.get(comparison)
            interval = evidence.get("pointwise_95_ci") if isinstance(evidence, dict) else None
            primary = comparisons.get(comparison)
            if (
                not isinstance(evidence, dict)
                or not _is_finite_number(evidence.get("point_estimate"))
                or not isinstance(interval, list)
                or len(interval) != 2
                or not all(_is_finite_number(value) for value in interval)
                or float(interval[0]) > float(interval[1])
                or not isinstance(primary, dict)
                or evidence.get("point_estimate") != primary.get("point_estimate")
                or interval != primary.get("pointwise_95_ci")
            ):
                problems.append(f"completed CCT-20 release manifest has invalid {label} safe-utility evidence")
                safe_evidence_valid = False
        if safe_evidence_valid:
            expected_safe_pass = (
                float(safe_utility["versus_always_freeze"]["pointwise_95_ci"][0]) > -0.005
                and float(safe_utility["versus_always_adapt"]["pointwise_95_ci"][0]) > 0.0
            )
            if safe_utility["passes"] is not expected_safe_pass:
                problems.append("completed CCT-20 release manifest has a stale safe-utility pass flag")

    secondary = document.get("secondary_outcome_reporting")
    expected_secondary_disclosure = (
        "Cell-level 16-indicator macro-F1 is archived as descriptive secondary evidence; "
        "no post-hoc aggregate or inference claim is made."
    )
    if (
        not isinstance(secondary, dict)
        or secondary.get("metric")
        != "16-indicator multilabel macro-F1 with top-1 as a one-hot prediction set"
        or secondary.get("scope") != "complete cell-level score artifact"
        or secondary.get("aggregate_claim") is not None
        or secondary.get("disclosure") != expected_secondary_disclosure
    ):
        problems.append("completed CCT-20 release manifest has stale secondary-outcome disclosure")

    verdict = document.get("verdict")
    verdict_code = verdict.get("code") if isinstance(verdict, dict) else None
    checks = document.get("strong_success_checks")
    protocol_strong = (
        checks.get("protocol_strong_success") if isinstance(checks, dict) else None
    )
    expanded_strong = (
        checks.get("expanded_empirical_bundle_including_mixed_effects")
        if isinstance(checks, dict)
        else None
    )
    safe_pass = safe_utility.get("passes") if isinstance(safe_utility, dict) else None
    if (
        verdict_code not in CCT20_VERDICT_CLAIMS
        or verdict.get("manuscript_claim") != CCT20_VERDICT_CLAIMS.get(verdict_code)
        or not isinstance(protocol_strong, bool)
        or not isinstance(expanded_strong, bool)
        or not isinstance(safe_pass, bool)
    ):
        problems.append("completed CCT-20 release manifest has an invalid paper verdict")
    else:
        expected_code = _expected_cct20_verdict_code(
            protocol_strong_success=protocol_strong,
            expanded_mixed_effects_success=expanded_strong,
            safe_utility_passes=safe_pass,
        )
        expected_flags = {
            "confirmatory_strong_claim_supported": expanded_strong,
            "primary_confirmatory_claim_supported": protocol_strong,
            "protocol_strong_success": protocol_strong,
            "expanded_mixed_effects_success": expanded_strong,
            "safe_utility_passes": safe_pass,
        }
        if expanded_strong and not protocol_strong:
            problems.append("completed CCT-20 release manifest has inconsistent strong-success checks")
        if verdict_code != expected_code or any(
            verdict.get(field) is not expected for field, expected in expected_flags.items()
        ):
            problems.append("completed CCT-20 release manifest verdict is inconsistent with its evidence")

    upstream = document.get("upstream_artifacts")
    if not isinstance(upstream, dict) or not REQUIRED_CCT20_UPSTREAM_KEYS <= set(upstream):
        problems.append("completed CCT-20 release manifest lacks builder-required upstream ledgers")
        upstream = {} if not isinstance(upstream, dict) else upstream
    _validate_execution_dependency_bundle(problems, upstream.get("execution_dependencies"))
    _validate_counted_release_bundle(
        problems,
        label="development traces",
        bundle=upstream.get("development_traces"),
        expected_count=55,
        ordering_key=lambda row: str(row.get("trace_id", "")),
    )
    _validate_counted_release_bundle(
        problems,
        label="prediction cells",
        bundle=upstream.get("prediction_cells"),
        expected_count=45,
        ordering_key=lambda row: str(row.get("path", "")),
    )
    _validate_counted_release_bundle(
        problems,
        label="prediction actions",
        bundle=upstream.get("prediction_actions"),
        expected_count=45,
        ordering_key=lambda row: (
            int(row.get("checkpoint_seed", -1)),
            str(row.get("location_id", "")),
        ),
    )
    _validate_one_shot_marker_binding(
        problems,
        upstream=upstream,
        repository_root=repository_root,
    )
    bindings = list(_iter_hash_bindings(upstream))
    if not bindings:
        problems.append("completed CCT-20 release manifest has no upstream artifact hashes")
    for label, binding in bindings:
        valid = _validate_hash_binding(
            problems,
            label=label,
            binding=binding,
            repository_root=repository_root,
            require_canonical_absolute_path=True,
        )
        if valid:
            _validate_received_identity(
                problems,
                label=label,
                binding=binding,
                repository_root=repository_root,
            )
    release_generator = upstream.get("release_generator")
    if isinstance(release_generator, dict):
        expected_builder = (repository_root / "docs/research/kbound/scripts/build_cct20_release.py").resolve()
        if Path(str(release_generator.get("path", ""))).resolve() != expected_builder:
            problems.append("completed CCT-20 release manifest names the wrong release generator")

    generated = document.get("generated_artifacts")
    generated_bindings = list(_iter_hash_bindings(generated, "generated_artifacts"))
    required_generated = {
        "cct20_numbers_tex",
        "cct20_primary_table_tex",
        "cct20_location_effects_tex",
    }
    if not isinstance(generated, dict) or set(generated) != required_generated:
        problems.append("completed CCT-20 release manifest lacks the required generated TeX artifacts")
    if not generated_bindings:
        problems.append("completed CCT-20 release manifest has no generated artifact hashes")
    for label, binding in generated_bindings:
        if _validate_hash_binding(
            problems,
            label=label,
            binding=binding,
            repository_root=repository_root,
            require_canonical_absolute_path=True,
        ):
            generated_path = _artifact_path(str(binding.get("path", "")), repository_root)
            expected_name = label.rsplit(".", maxsplit=1)[-1].removesuffix("_tex") + ".tex"
            if generated_path.name != expected_name:
                problems.append(f"CCT-20 release {label} uses the wrong generated filename")
            if generated_path.stat().st_mode & 0o222:
                problems.append(f"CCT-20 release {label} is writable, not immutable")
            if label.endswith(".cct20_numbers_tex"):
                try:
                    numbers_text = generated_path.read_text(encoding="ascii")
                except (OSError, UnicodeError) as exc:
                    problems.append(f"completed CCT-20 number macros are unreadable: {exc}")
                else:
                    missing_macros = sorted(
                        name
                        for name in REQUIRED_CCT20_NUMBER_MACROS
                        if re.search(rf"\\newcommand\s*\{{\\{re.escape(name)}\}}", numbers_text) is None
                    )
                    if missing_macros:
                        problems.append(
                            "completed CCT-20 number macros omit publication fields: "
                            + ", ".join(missing_macros)
                        )
                    expected_commands: dict[str, str] = {}
                    if isinstance(false_count, int) and not isinstance(false_count, bool):
                        expected_commands["CCTFalseAdaptCount"] = rf"\newcommand{{\CCTFalseAdaptCount}}{{{false_count}}}"
                    if isinstance(safe_pass, bool):
                        expected_commands["CCTSafeUtilityPass"] = (
                            rf"\newcommand{{\CCTSafeUtilityPass}}{{\textnormal{{"
                            f"{'yes' if safe_pass else 'no'}"
                            r"}}"
                        )
                    if isinstance(verdict, dict) and verdict_code in CCT20_VERDICT_CLAIMS:
                        expected_commands["CCTVerdict"] = (
                            rf"\newcommand{{\CCTVerdict}}{{\textnormal{{"
                            f"{_cct20_tex_text(verdict_code.replace('_', ' '))}"
                            r"}}"
                        )
                        expected_commands["CCTManuscriptClaim"] = (
                            rf"\newcommand{{\CCTManuscriptClaim}}{{\textnormal{{"
                            f"{_cct20_tex_text(verdict['manuscript_claim'])}"
                            r"}}"
                        )
                    if isinstance(secondary, dict) and isinstance(secondary.get("disclosure"), str):
                        expected_commands["CCTSecondaryMetricDisclosure"] = (
                            rf"\newcommand{{\CCTSecondaryMetricDisclosure}}{{\textnormal{{"
                            f"{_cct20_tex_text(secondary['disclosure'])}"
                            r"}}"
                        )
                    inference_binding = upstream.get("two_way_inference")
                    if isinstance(inference_binding, dict):
                        inference_sha = inference_binding.get("canonical_document_sha256")
                        if isinstance(inference_sha, str):
                            expected_commands["CCTInferenceSHA"] = (
                                rf"\newcommand{{\CCTInferenceSHA}}{{{inference_sha}}}"
                            )
                    stale_commands = sorted(
                        name
                        for name, command in expected_commands.items()
                        if command not in numbers_text
                    )
                    if stale_commands:
                        problems.append(
                            "completed CCT-20 number macros are inconsistent with the release: "
                            + ", ".join(stale_commands)
                        )

    receipt_path = manifest_path.with_name(manifest_path.name + ".receipt.json")
    receipt = _load_json_object_for_release(
        problems,
        receipt_path,
        label="manifest receipt",
    )
    if receipt is not None:
        if receipt_path.stat().st_mode & 0o222:
            problems.append("completed CCT-20 release manifest receipt is writable, not immutable")
        expected_receipt = {
            "schema": "kbound_cct20_artifact_receipt_v1",
            "artifact_path": str(manifest_path),
            "artifact_bytes": manifest_path.stat().st_size,
            "artifact_sha256": file_sha256(manifest_path),
            "canonical_document_sha256": _stable_json_sha256(document),
        }
        for field, expected in expected_receipt.items():
            if receipt.get(field) != expected:
                problems.append(f"completed CCT-20 release manifest receipt has stale {field}")
    return document, problems


def _has_verified_public_registry(document: dict | None, repository_root: Path) -> bool:
    """Accept public-preregistration language only with a hashed registry snapshot."""

    if not isinstance(document, dict) or document.get("schema") != "kbound_cct20_release_manifest_v1":
        return False
    registry = document.get("public_registry_evidence")
    if not isinstance(registry, dict):
        return False
    if registry.get("registered_before_target_execution") is not True:
        return False
    if not isinstance(registry.get("registry_id"), str) or not registry["registry_id"].strip():
        return False
    if not isinstance(registry.get("url"), str) or not registry["url"].startswith("https://"):
        return False
    snapshot = registry.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    binding_problems: list[str] = []
    return (
        _validate_hash_binding(
            binding_problems,
            label="public_registry_evidence.snapshot",
            binding=snapshot,
            repository_root=repository_root,
            require_canonical_absolute_path=True,
        )
        and not binding_problems
    )


def _validate_cct20_verdict_usage(
    context: str,
    normalized: str,
    release_document: dict,
) -> list[str]:
    """Bind completed-result prose to the exact generated release verdict."""

    problems: list[str] = []
    source = _strip_tex_comments_for_claims(context)
    for macro in ("CCTVerdict", "CCTManuscriptClaim"):
        if re.search(rf"\\{macro}(?![A-Za-z@])", source) is None:
            problems.append(f"completed CCT-20 manuscript must consume generated \\{macro}")

    verdict = release_document.get("verdict", {})
    code = verdict.get("code")
    if code not in CCT20_VERDICT_CLAIMS:
        return problems
    for other_code in CCT20_VERDICT_CLAIMS:
        if other_code == code:
            continue
        phrase = other_code.replace("_", " ").lower()
        if phrase in normalized:
            problems.append(
                f"CCT-20 manuscript names verdict {other_code}, but the release verdict is {code}"
            )

    cct_prefix = r"\b(?:cct\s*[- ]?20|caltech camera traps?\s*[- ]?20)\b"
    confirmatory_success = (
        rf"{cct_prefix}[^.{{}}]{{0,180}}\b(?:confirmatory|strong[- ]success|beats?\s+both)\b"
    )
    mixed_effects_present = (
        rf"{cct_prefix}[^.{{}}]{{0,180}}\b(?:both\s+helpful\s+and\s+harmful|"
        r"mixed\s+helpful\s*/?\s*harmful)\b[^.]{0,80}\b(?:present|observed|met|pass(?:es|ed)?)\b"
    )
    safe_utility_pass = (
        rf"{cct_prefix}[^.{{}}]{{0,180}}\bsafe[- ]utility\b[^.{{}}]{{0,80}}"
        r"\b(?:pass(?:es|ed)?|satisf(?:y|ies|ied))\b"
    )
    action_pair = r"both\s+(?:\\?adapt\s+and\s+\\?freeze|\\?freeze\s+and\s+\\?adapt)"
    kga_both_actions = (
        r"\bkga\b[^.{}]{0,120}\b(?:uses?|used|makes?|made|issues?|issued|selects?|selected|"
        r"produces?|produced|exercises?|exercised)\b[^.{}]{0,100}\b" + action_pair
    )
    observed_both_actions = (
        rf"(?:{cct_prefix}[^.{{}}]{{0,500}}{kga_both_actions}|"
        rf"{kga_both_actions}[^.{{}}]{{0,500}}{cct_prefix})"
    )
    kga_beats_both = (
        r"\bkga\b[^.{}]{0,120}\b(?:beats?|beat|outperforms?|outperformed|improves?|improved)\b"
        r"(?:\s+(?:on|over))?\s+both\s+(?:fixed\s+)?(?:policies|baselines|comparators)\b"
    )
    both_beaten_by_kga = (
        r"\bboth\s+(?:fixed\s+)?(?:policies|baselines|comparators)\b[^.{}]{0,80}"
        r"\b(?:are|were)\s+(?:beaten|outperformed|improved\s+on)\s+by\s+kga\b"
    )
    observed_beats_both = (
        rf"(?:{cct_prefix}[^.{{}}]{{0,500}}(?:{kga_beats_both}|{both_beaten_by_kga})|"
        rf"(?:{kga_beats_both}|{both_beaten_by_kga})[^.{{}}]{{0,500}}{cct_prefix})"
    )
    if code != "CONFIRMATORY_STRONG_SUCCESS" and _has_unsafe_match(
        normalized,
        confirmatory_success,
        safe_patterns=(
            r"\bdoes\s+not\s+establish\b",
            r"\bdid\s+not\s+(?:establish|support|satisfy|meet)\b",
            r"\bnot\s+evidence\s+of\b",
        ),
    ):
        problems.append(f"CCT-20 manuscript overstates the release verdict {code} as confirmatory success")
    if code == "CONFIRMATORY_PRIMARY_SUCCESS_MIXED_EFFECTS_MISSING" and re.search(
        mixed_effects_present, normalized, flags=re.IGNORECASE
    ):
        problems.append("CCT-20 manuscript claims mixed helpful/harmful evidence that the release verdict lacks")
    if code == "NO_CONFIRMATORY_SUCCESS" and re.search(
        safe_utility_pass, normalized, flags=re.IGNORECASE
    ):
        problems.append("CCT-20 manuscript claims safe utility although the release verdict rejects it")

    exposure = release_document.get("action_exposure", {})
    counts = exposure.get("counts") if isinstance(exposure, dict) else None
    adapt_count = counts.get("ADAPT") if isinstance(counts, dict) else None
    freeze_count = counts.get("FREEZE") if isinstance(counts, dict) else None
    if (
        isinstance(adapt_count, int)
        and not isinstance(adapt_count, bool)
        and isinstance(freeze_count, int)
        and not isinstance(freeze_count, bool)
        and (adapt_count == 0 or freeze_count == 0)
        and _has_unsafe_match(
            normalized,
            observed_both_actions,
            safe_patterns=(
                r"\b(?:does|did)\s+not\s+(?:show|establish|indicate|demonstrate|mean)\b",
                r"\bkga\b[^,;:.]{0,80}\b(?:does|did|has|had)\s+not\b",
                r"\bnot\s+both\s+(?:\\?adapt|\\?freeze)\b",
                r"\bneither\b",
            ),
        )
    ):
        problems.append(
            "CCT-20 manuscript claims both ADAPT and FREEZE exposure, but release action "
            f"counts are ADAPT={adapt_count}, FREEZE={freeze_count}"
        )

    comparisons = release_document.get("primary_comparisons", {})
    comparison_points = []
    if isinstance(comparisons, dict):
        for name in ("versus_always_adapt", "versus_always_freeze"):
            evidence = comparisons.get(name)
            comparison_points.append(
                evidence.get("point_estimate") if isinstance(evidence, dict) else None
            )
    both_point_estimates_positive = len(comparison_points) == 2 and all(
        _is_finite_number(value) and float(value) > 0.0 for value in comparison_points
    )
    if not both_point_estimates_positive and _has_unsafe_match(
        normalized,
        observed_beats_both,
        safe_patterns=(
            r"\b(?:does|did)\s+not\s+(?:beat|improve|outperform)\b",
            r"\bkga\b[^,;:.]{0,80}\b(?:does|did|has|had)\s+not\b",
            r"\bneither\b",
            r"\bnot\s+(?:evidence|proof)\s+of\b",
        ),
    ):
        problems.append(
            "CCT-20 manuscript claims improvement over both fixed policies, but the released "
            "primary point estimates do not improve on both"
        )
    return problems


def validate_cct20_claims(
    manuscript_text: str,
    *,
    release_manifest_path: Path = CCT20_RELEASE_MANIFEST,
    repository_root: Path = ROOT,
) -> list[str]:
    """Reject known overclaims in any live CCT-20 manuscript section.

    The release-manifest gate is dormant while CCT remains a protocol-only
    section.  It becomes mandatory as soon as the manuscript promotes an
    observed result, including by inputting any generated CCT result table.
    """

    context = _cct20_claim_contexts(manuscript_text)
    if not context:
        return []
    normalized = _normalize_claim_text(context)
    problems: list[str] = []

    rules = (
        (
            r"\b(?:label[- ]?un[- ]?opened|un[- ]?opened\s+(?:target\s+)?"
            r"(?:labels?|annotations?)|target\s+(?:labels?|annotations?)\s+"
            r"(?:were|remained|are|had\s+been)\s+(?:completely\s+|literally\s+)?"
            r"(?:un[- ]?opened|never\s+(?:opened|inspected|accessed|seen))|"
            r"(?:never|not)\s+(?:opened|inspected|accessed|seen)\s+(?:the\s+)?"
            r"target\s+(?:labels?|annotations?))\b",
            (
                r"\bnot\s+literally\s+label[- ]?un[- ]?opened\b",
                r"\bnot\s+(?:described|claimed|presented)\s+as\s+literally\s+"
                r"label[- ]?un[- ]?opened\b",
                r"\bnot\s+(?:a\s+)?label[- ]?un[- ]?opened\s+"
                r"(?:study|experiment|evaluation|claim)\b",
            ),
            "CCT-20 provenance overclaim: target metadata had been inspected, so literal label-unopened wording is forbidden",
        ),
        (
            r"(?:\b(?:45|forty[- ]five)\s+(?:statistically\s+)?independent\s+"
            r"(?:environments?|locations?|camera(?:\s+sites?)?|domains?|evaluation\s+units?|"
            r"observations?|replicates?|cells?)\b|"
            r"\b(?:45|forty[- ]five)\s+checkpoint[- ]by[- ]location\s+cells?\s+"
            r"(?:are|were|as|constitute[sd]?|provide[sd]?)\s+(?:statistically\s+)?"
            r"independent(?:\s+(?:evaluation\s+units?|observations?|replicates?|cells?))?\b)",
            (
                r"\b(?:not|rather\s+than)\s+(?:45|forty[- ]five)\s+"
                r"(?:statistically\s+)?independent\b",
                r"\b(?:45|forty[- ]five)\b[^,;:.]{0,80}\b(?:are|were|is)\s+not\s+independent\b",
                r"\b(?:do|does)\s+not\s+(?:treat|count|use)\s+(?:the\s+)?"
                r"(?:45|forty[- ]five)\b[^,;:.]{0,80}\bas\s+independent\b",
            ),
            "CCT-20 unit overclaim: 45 checkpoint-by-location cells are not 45 independent environments",
        ),
        (
            r"(?:\bholm(?:[- ](?:adjusted|corrected))?\s+"
            r"(?:(?:simultaneous|pointwise|(?:9[057](?:\.5)?)\s*%?)\s+){0,3}"
            r"(?:confidence\s+)?(?:intervals?|cis?|bands?|bounds?|error\s+bars?)\b|"
            r"\b(?:confidence\s+)?(?:intervals?|cis?|bands?|bounds?|error\s+bars?)\s+"
            r"(?:are\s+|were\s+)?holm[- ](?:adjusted|corrected)\b)",
            (
                r"\b(?:not|no)\s+holm(?:[- ](?:adjusted|corrected))?\s+"
                r"(?:confidence\s+)?(?:intervals?|cis?)\b",
                r"\bholm\b[^,;:.]{0,80}\b(?:does\s+not|is\s+not\s+used\s+to)\s+adjust\b",
                r"\bholm[- ](?:adjusted|corrected)\b[^,;:.]{0,80}"
                r"\b(?:was|were|is|are)\s+not\s+(?:used|reported|constructed)\b",
            ),
            "CCT-20 multiplicity overclaim: Holm adjusts p-values, not confidence intervals",
        ),
        (
            r"(?:\b(?:reproduce[sd]?|replicates?|matches?|is\s+comparable\s+to|is\s+equivalent\s+to)\s+"
            r"(?:the\s+)?(?:official\s+)?(?:cct\s*[- ]?20\s+)?(?:single[- ]label\s+)?"
            r"(?:classification\s+)?(?:leaderboard|benchmark)\b|"
            r"\b(?:official\s+)?(?:cct\s*[- ]?20\s+)?(?:single[- ]label\s+)?(?:classification\s+)?"
            r"(?:leaderboard|benchmark)\s+(?:reproduction|replication|comparability|equivalence)\b)",
            (
                r"\b(?:does\s+not|do\s+not|cannot|is\s+not|are\s+not|not\s+an?|no)\b"
                r"[^,;:.]{0,100}\b(?:reproduc|replic|match|comparab|equival|leaderboard|benchmark)",
                r"\b(?:reproduction|replication|comparability|equivalence)\s+"
                r"(?:is|are)\s+not\s+claimed\b",
            ),
            "CCT-20 task overclaim: set-valued scoring is not an official single-label leaderboard reproduction",
        ),
        (
            r"(?:\b(?:universal(?:ly)?|all)\s+natural[- ]shift\s+"
            r"(?:win|superiority|dominance|guarantee)\b|"
            r"\b(?:wins?|dominates?)\s+(?:universally\s+|across\s+all\s+)?natural\s+shifts?\b|"
            r"\bgeneralizes?\s+to\s+all\s+natural\s+shifts?\b|"
            r"\b(?:win|superiority|dominance|guarantee)\s+(?:on|for|across)\s+"
            r"(?:every|all)\s+natural\s+shifts?\b)",
            (
                r"\b(?:not|no|does\s+not\s+(?:establish|show|imply|support))\b"
                r"[^,;:.]{0,100}\b(?:universal|all\s+natural|every\s+natural|wins?|"
                r"dominates?|generalizes?)\b",
                r"\b(?:universal|all\s+natural|every\s+natural)[^,;:.]{0,80}"
                r"\b(?:is|are)\s+not\s+(?:claimed|established)\b",
            ),
            "CCT-20 scope overclaim: one locked experiment cannot establish a universal natural-shift win",
        ),
        (
            r"(?:\b(?:five|5)\s+(?:independently\s+trained\s+)?checkpoints?\b"
            r"[^,;:.]{0,120}\b(?:five|5)?\s*independent\s+"
            r"(?:studies|experiments|replications|evaluation\s+runs|experimental\s+runs)\b|"
            r"\b(?:five|5)\s+independent\s+(?:studies|experiments|replications|"
            r"evaluation\s+runs|experimental\s+runs)\b[^,;:.]{0,120}"
            r"\b(?:checkpoints?|models?)\b|"
            r"\b(?:five|5)\s+checkpoints?\s+(?:constitute|count\s+as|provide|supply)\s+"
            r"(?:five|5)\s+independent\s+(?:runs|replicates|trials)\b)",
            (
                r"\b(?:not|do\s+not\s+(?:constitute|represent|provide))\b[^,;:.]{0,100}"
                r"\b(?:independent\s+)?(?:studies|experiments|replications)\b",
                r"\bindependent\s+(?:studies|experiments|replications)\s+"
                r"(?:are|were)\s+not\s+claimed\b",
            ),
            "CCT-20 replication overclaim: five checkpoints are repeated models, not five independent studies",
        ),
        (
            r"(?:\b(?:formal|exact|finite[- ]sample|guaranteed)\s+"
            r"(?:trans[- ]location\s+|cross[- ]location\s+)?(?:conformal\s+)?"
            r"(?:coverage|guarantees?)\b|"
            r"\bconformal\s+coverage\b[^,;:.]{0,100}\b(?:across|on|for)\s+"
            r"trans[- ]locations?\b|"
            r"\bfinite[- ]sample\s+guarantees?\b[^,;:.]{0,100}"
            r"\b(?:across|on|for)\s+(?:unseen\s+|trans[- ]?)locations?\b)",
            (
                r"\b(?:not|no|does\s+not\s+(?:establish|provide|imply|guarantee))\b"
                r"[^,;:.]{0,120}\b(?:formal|exact|finite[- ]sample|guaranteed|conformal)\b",
                r"\b(?:formal|exact|finite[- ]sample|guaranteed)[^,;:.]{0,100}"
                r"\b(?:is|are)\s+not\s+claimed\b",
                r"\bconformal\s+coverage\b[^,;:.]{0,100}\b(?:is|are)\s+not\s+claimed\b",
            ),
            "CCT-20 coverage overclaim: cis-heavy calibration does not establish formal trans-location conformal coverage",
        ),
    )
    for pattern, safe_patterns, message in rules:
        if _has_unsafe_match(normalized, pattern, safe_patterns=safe_patterns):
            problems.append(message)

    public_preregistration_claimed = _has_unsafe_match(
        normalized,
        r"\b(?:publicly\s+pre[- ]?registered|pre[- ]?registered\s+"
        r"(?:publicly|in\s+(?:a\s+)?(?:public\s+registry|osf))|"
        r"registered\s+publicly|public\s+pre[- ]?registration)\b",
        safe_patterns=(
            r"\b(?:not|was\s+not|is\s+not|never)\s+publicly\s+pre[- ]?registered\b",
            r"\bpublicly\s+pre[- ]?registered\b[^,;:.]{0,60}\b(?:is|was)\s+not\s+claimed\b",
            r"\b(?:public\s+pre[- ]?registration|registered\s+publicly)\b"
            r"[^,;:.]{0,60}\b(?:is|was)\s+not\s+claimed\b",
        ),
    )
    completed_result_claimed = _cct20_completed_result_claimed(normalized)
    release_document: dict | None = None
    if completed_result_claimed or public_preregistration_claimed:
        release_document, release_problems = validate_cct20_release_manifest(
            release_manifest_path,
            repository_root=repository_root,
        )
        problems.extend(release_problems)
    if completed_result_claimed and isinstance(release_document, dict):
        problems.extend(_validate_cct20_verdict_usage(context, normalized, release_document))
    if public_preregistration_claimed and not _has_verified_public_registry(release_document, repository_root):
        problems.append(
            "CCT-20 registration overclaim: publicly preregistered requires a hashed public-registry record"
        )
    return problems


def validate_storage_manifest(problems: list[str], generated: dict) -> tuple[int, int]:
    """Verify local evidence, lock-seal agreement, and promoted-source coverage."""

    try:
        storage = json.loads(STORAGE_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"storage manifest is unreadable: {exc}")
        return 0, 0

    def verify_present(*, label: str, location: object, expected_size: object, expected_sha256: object) -> bool:
        if not isinstance(location, str) or not location or location.startswith("$"):
            problems.append(f"{label} does not name an explicit repository-relative path")
            return False
        parsed_location = Path(location)
        if (
            parsed_location.is_absolute()
            or ".." in parsed_location.parts
            or any(token in location for token in ("*", "?", "["))
        ):
            problems.append(f"{label} is not a literal repository-relative path: {location!r}")
            return False
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            problems.append(f"{label} has invalid size_bytes: {expected_size!r}")
            return False
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
        ):
            problems.append(f"{label} has invalid SHA-256 metadata")
            return False

        path = ROOT / location
        try:
            path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            problems.append(f"{label} resolves outside the repository: {location}")
            return False
        if not path.is_file():
            problems.append(f"{label} is missing from disk: {location}")
            return False

        observed_size = path.stat().st_size
        if observed_size != expected_size:
            problems.append(f"{label} size mismatch for {location}: expected {expected_size}, got {observed_size}")
        observed_sha256 = file_sha256(path)
        if observed_sha256 != expected_sha256:
            problems.append(
                f"{label} SHA-256 mismatch for {location}: expected {expected_sha256}, got {observed_sha256}"
            )
        return observed_size == expected_size and observed_sha256 == expected_sha256

    artifacts = storage.get("artifacts")
    if not isinstance(artifacts, list):
        problems.append("storage manifest artifacts must be a list")
        artifacts = []
    direct_records: dict[str, tuple[int, str]] = {}
    for index, row in enumerate(artifacts):
        if not isinstance(row, dict):
            problems.append(f"storage manifest artifact row {index} is not an object")
            continue
        location = row.get("expected_location")
        expected_size = row.get("size_bytes")
        expected_sha256 = row.get("sha256")
        if row.get("tracked") is True and (expected_size is None or expected_sha256 is None):
            problems.append(f"tracked storage artifact {location!r} must record both size_bytes and SHA-256")
            continue
        if expected_size is None and expected_sha256 is None:
            continue
        if expected_size is None or expected_sha256 is None:
            problems.append(f"storage artifact {location!r} must record both size_bytes and SHA-256")
            continue
        label = f"storage artifact row {index}"
        verify_present(
            label=label,
            location=location,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        if isinstance(location, str):
            record = (expected_size, expected_sha256)
            if location in direct_records:
                problems.append(f"duplicate storage artifact path: {location}")
            else:
                direct_records[location] = record

    sealed = storage.get("sealed_evidence_checksums")
    if not isinstance(sealed, dict):
        problems.append("storage manifest sealed_evidence_checksums must be an object")
        sealed = {}
    sealed_records: dict[str, tuple[int, str]] = {}
    status_counts = {"present": 0, "absent": 0}
    for location, row in sealed.items():
        if not isinstance(row, dict):
            problems.append(f"sealed evidence row is not an object: {location}")
            continue
        status = str(row.get("status", "")).lower()
        if status not in status_counts:
            problems.append(f"sealed evidence has invalid status {status!r}: {location}")
            continue
        status_counts[status] += 1
        path = ROOT / location
        if status == "absent":
            if path.exists():
                problems.append(f"sealed evidence marked absent is present: {location}")
            continue
        expected_size = row.get("size_bytes")
        expected_sha256 = row.get("sha256")
        verify_present(
            label="sealed evidence",
            location=location,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        sealed_records[location] = (expected_size, expected_sha256)

    summary = storage.get("sealed_evidence_summary")
    if not isinstance(summary, dict):
        problems.append("storage manifest sealed_evidence_summary must be an object")
    else:
        expected_summary = {
            "files": len(sealed),
            "present": status_counts["present"],
            "absent": status_counts["absent"],
        }
        for key, expected_value in expected_summary.items():
            if summary.get(key) != expected_value:
                problems.append(
                    f"storage manifest sealed summary {key} mismatch: "
                    f"expected {expected_value}, got {summary.get(key)!r}"
                )

    unsealed = storage.get("unsealed_present_artifacts", [])
    if not isinstance(unsealed, list):
        problems.append("storage manifest unsealed_present_artifacts must be a list")
    else:
        for index, row in enumerate(unsealed):
            if not isinstance(row, dict) or row.get("status") != "present_unsealed":
                problems.append(f"invalid unsealed-present artifact row {index}")
                continue
            verify_present(
                label=f"unsealed-present artifact row {index}",
                location=row.get("path"),
                expected_size=row.get("current_bytes"),
                expected_sha256=row.get("current_sha256"),
            )

    try:
        lock = json.loads(LOCK_SEAL.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"nine-track lock seal is unreadable: {exc}")
        lock = {}
    locked_records: dict[str, tuple[int, str]] = {}
    tracks = lock.get("tracks", {}) if isinstance(lock, dict) else {}
    if not isinstance(tracks, dict):
        problems.append("nine-track lock seal tracks must be an object")
        tracks = {}
    for track, track_row in tracks.items():
        files = track_row.get("files", {}) if isinstance(track_row, dict) else {}
        if not isinstance(files, dict):
            problems.append(f"nine-track lock seal files must be an object: {track}")
            continue
        for location, row in files.items():
            if not isinstance(row, dict):
                problems.append(f"invalid nine-track lock entry: {track}/{location}")
                continue
            record = (row.get("bytes"), row.get("sha256"))
            previous = locked_records.get(location)
            if previous is not None and previous != record:
                problems.append(f"nine-track lock has conflicting duplicate metadata: {location}")
            locked_records[location] = record
    for location, record in locked_records.items():
        if location not in sealed_records:
            problems.append(f"nine-track locked evidence is absent from storage manifest: {location}")
        elif sealed_records[location] != record:
            problems.append(f"storage manifest disagrees with nine-track lock metadata: {location}")

    for location in sorted(set(direct_records) & set(sealed_records)):
        if direct_records[location] != sealed_records[location]:
            problems.append(f"direct and sealed storage metadata disagree for duplicate path: {location}")

    covered_paths = set(direct_records) | set(sealed_records)
    for source in sorted(set(iter_source_paths(generated))):
        if source not in covered_paths:
            problems.append(f"promoted result source lacks storage-manifest coverage: {source}")

    return len(direct_records), len(sealed_records)


def main() -> int:
    problems: list[str] = []
    missing = [str(path.relative_to(ROOT)) for path in ACTIVE_SOURCES if not path.is_file()]
    if missing:
        problems.extend(f"missing active source: {path}" for path in missing)
        corpus = ""
    else:
        corpus = "\n".join(live_latex(path.read_text()) for path in ACTIVE_SOURCES)
    normalized_corpus = " ".join(corpus.split())
    problems.extend(validate_cct20_claims(corpus))

    if not LONG_TMLR.is_file():
        problems.append(f"missing maintained long driver: {LONG_TMLR.relative_to(ROOT)}")
    else:
        long_driver = LONG_TMLR.read_text()
        long_driver_live = live_latex(long_driver)
        if not any(
            token in long_driver_live
            for token in (
                r"\input{kbound_submission_body}",
                r"\input{kbound_submission_body.tex}",
            )
        ):
            problems.append("maintained long driver must input the synchronized kbound_submission_body")
        stale_long_tokens = {
            "kbound_short_body": "stale pre-Phase-1 body",
            "kbound_short_appendix": "stale pre-Phase-1 appendix",
            "SUPERSEDED HISTORICAL": "superseded-release banner",
        }
        for token, reason in stale_long_tokens.items():
            if token in long_driver:
                problems.append(f"maintained long driver retains {reason}: {token}")

    forbidden = {
        r"The sign of $\Delta$ is identifiable iff": "overbroad sign-identifiability headline",
        r"$\varepsilon=\beta$": "epsilon equated with beta",
        r"KGA estimates $\beta$": "KGA described as estimating beta",
        r"KGA computes $M$": "KGA described as computing population margin M",
        r"at least $0.90$ by construction": "empirical coverage promoted to exact coverage",
        r"1{,}113": "stale CIFAR-10-C Tent adapt count",
        r"1{,}244": "stale CIFAR-10-C EATA adapt count",
        r"66.8\%": "stale CIFAR-10-C aggregate coverage",
        r"5.00\times": "stale canonical regret ratio",
        r"9{,}504": "stale pooled decision denominator",
        r"0.001585": "stale earlier-policy Tent value",
        r"cluster-robust for Tent": "obsolete current-policy Tent cluster claim",
        r"Verdict: WIN": "obsolete current-policy POEM/AETTA win",
        "four one-sided natural tracks": "withheld iWildCam implicitly counted as no-harm",
        "four one-sided tracks": "withheld iWildCam implicitly counted as no-harm",
    }
    for token, reason in forbidden.items():
        if token in corpus:
            problems.append(f"{reason}: {token}")

    required = {
        r"\Delta=R_T(f_0)-R_T(f_a)": "benefit convention",
        r"if and only if $|M|>\beta$": "strict-commitment frontier",
        r"does not numerically apply $|M|>\beta$": "population/KGA separation",
        r"not a universal accuracy booster": "claim-scope limitation",
        r"leave-one-condition-out cross-fitted empirical residual calibration": "stress-grid scope",
    }
    for token, reason in required.items():
        if token not in corpus and token not in normalized_corpus:
            problems.append(f"missing {reason}: {token}")

    for path in (KBOUND / "kbound_submission_body.tex",):
        if path.is_file():
            text = live_latex(path.read_text())
            if "candidate TTA" not in text or "transductive" not in text:
                problems.append(f"missing transductive candidate-TTA disclosure: {path.relative_to(ROOT)}")
            if "evaluation-batch BatchNorm statistics" not in text:
                problems.append(f"missing evaluation-batch BatchNorm disclosure: {path.relative_to(ROOT)}")
            if path.name == "kbound_submission_body.tex" and text.count(r"\SourceManifestSHA") < 2:
                problems.append(
                    "compact manuscript must obtain both printed provenance hashes from "
                    "the generated SourceManifestSHA macro"
                )

    data: dict | None = None
    cluster_data: dict | None = None
    if not CANONICAL.is_file():
        problems.append(f"missing canonical panel: {CANONICAL.relative_to(ROOT)}")
    else:
        data = json.loads(CANONICAL.read_text())
        if not SOURCE_MANIFEST.is_file():
            problems.append(f"missing canonical source manifest: {SOURCE_MANIFEST.relative_to(ROOT)}")
        else:
            observed_source_manifest_sha256 = file_sha256(SOURCE_MANIFEST)
            if data.get("source_manifest_sha256") != observed_source_manifest_sha256:
                problems.append("canonical panel source-manifest hash disagrees with the live source manifest")
            if KBOUND_NUMBERS.is_file():
                expected_macro = rf"\newcommand{{\SourceManifestSHA}}{{{observed_source_manifest_sha256}}}"
                if expected_macro not in KBOUND_NUMBERS.read_text():
                    problems.append("generated LaTeX exposes a stale source-manifest hash")
            else:
                problems.append(f"missing generated numbers: {KBOUND_NUMBERS.relative_to(ROOT)}")
        if data.get("source_file_count") != 106:
            problems.append(
                f"canonical source artifact count changed: expected 106, got {data.get('source_file_count')}"
            )
        panel = data["panels"]["cifar10c"]["panel"]
        candidates = panel["candidates"]
        expected = {
            "tent": (1107, 359, 694),
            "eata": (1241, 132, 787),
            "sar": (1446, 0, 714),
        }
        for candidate, counts in expected.items():
            row = candidates[candidate]
            observed = (row["adapt_count"], row["freeze_count"], row["abstain_count"])
            if observed != counts:
                problems.append(f"canonical {candidate} action counts changed: expected {counts}, got {observed}")
        aggregate = panel["architecture_panel_aggregate"]
        if aggregate["false_adapt_count"] != 0:
            problems.append("canonical CIFAR-10-C aggregate no longer has zero observed false adapt")
        iwild = data["panels"]["iwildcam"]
        historical = iwild.get("historical_reconciliation", {})
        if historical.get("status") != "superseded_not_promotable":
            problems.append("iWildCam historical beats-both result is not marked superseded")
        if historical.get("historical_claim", {}).get("beats_both") is not True:
            problems.append("iWildCam historical reconciliation lost the archived positive flag")
        if historical.get("corrected_claim", {}).get("point_beats_both") is not False:
            problems.append("iWildCam corrected replay was accidentally promoted to beats-both")

    if not CURRENT_CLUSTER.is_file():
        problems.append(f"missing current-policy family sensitivity: {CURRENT_CLUSTER.relative_to(ROOT)}")
    else:
        cluster_data = json.loads(CURRENT_CLUSTER.read_text())
        if cluster_data.get("schema") != "kbound-current-policy-cluster-inference-v2":
            problems.append("current-policy family sensitivity does not use the v2 schema")
        if cluster_data.get("contrast_convention") != ("baseline_regret_minus_kga_regret; positive values favor KGA"):
            problems.append("current-policy family sensitivity uses the wrong contrast convention")
        analysis_path = ROOT / cluster_data.get("analysis_script", "")
        if not analysis_path.is_file() or file_sha256(analysis_path) != cluster_data.get("analysis_script_sha256"):
            problems.append("current-policy family sensitivity analysis-script binding is stale")
        for name, binding in cluster_data.get("live_code_bindings", {}).items():
            bound_path = ROOT / binding.get("path", "")
            if not bound_path.is_file() or file_sha256(bound_path) != binding.get("sha256"):
                problems.append(f"current-policy family sensitivity {name} binding is stale")
        family = cluster_data.get("preregistered_six_comparison_holm", {})
        if family.get("family_size") != 6 or family.get("alpha") != 0.05:
            problems.append("current-policy family sensitivity lacks the preregistered six-way Holm family")
        for candidate in ("tent", "eata", "sar"):
            gate = cluster_data.get("candidates", {}).get(candidate, {}).get("gate", {})
            if gate.get("preregistered_six_comparison_cluster_sensitivity_pass") is not False:
                problems.append(
                    f"current-policy family sensitivity incorrectly passes preregistered gate for {candidate}"
                )
        tent = cluster_data.get("candidates", {}).get("tent", {})
        if tent and not tent.get("gate", {}).get("both_pointwise_95pct_cluster_bootstrap_intervals_positive"):
            problems.append("Tent family sensitivity lost its two positive ordinary intervals")
        for baseline in ("always_adapt", "always_freeze"):
            p_value = (
                tent.get("comparisons", {}).get(baseline, {}).get("p_value_holm_preregistered_six_comparison_family")
            )
            if p_value != 0.09375:
                problems.append(f"Tent preregistered six-way Holm p-value changed for {baseline}: {p_value!r}")

    release_paths = (
        GENERATED_MANIFEST,
        CURRENT_CLUSTER_TABLE,
        UNIFORM_VERDICTS,
        DECISION_METRICS,
        CLAIM_MATRIX,
        CLAIM_LEDGER,
        RESULT_MANIFEST,
        RESULTS_SOURCE,
        HISTORICAL_LEDGER,
        RESULT_AUDIT,
        CLAIM_MANIFEST,
        README,
        STORAGE_MANIFEST,
    )
    for path in release_paths:
        if not path.is_file():
            problems.append(f"missing release consistency surface: {path.relative_to(ROOT)}")

    if data is not None and cluster_data is not None and all(path.is_file() for path in release_paths):
        expected_counts = {
            "tent": {"ADAPT": 1107, "FREEZE": 359, "ABSTAIN": 694},
            "eata": {"ADAPT": 1241, "FREEZE": 132, "ABSTAIN": 787},
        }

        generated = json.loads(GENERATED_MANIFEST.read_text())
        direct_storage_count, sealed_storage_count = validate_storage_manifest(problems, generated)
        generated_tracks = generated["tracks"]
        accounting = generated["decision_accounting_summary"]["rows"]
        for candidate, label in (("tent", "CIFAR-10-C Tent"), ("eata", "CIFAR-10-C EATA")):
            if generated_tracks[f"cifar10c_{candidate}"]["decision_counts"] != expected_counts[candidate]:
                problems.append(f"generated paper manifest has stale {candidate} decision counts")
            sensitivity = generated_tracks[f"cifar10c_{candidate}"].get("current_policy_family_sensitivity", {})
            if sensitivity.get("preregistered_six_comparison_holm_rejects_both") is not False:
                problems.append(f"generated paper manifest overstates preregistered family inference for {candidate}")
            summary = row_by_track(accounting, label)
            observed = {action: summary[action] for action in ("ADAPT", "FREEZE", "ABSTAIN")}
            if observed != expected_counts[candidate]:
                problems.append(f"generated decision-accounting summary has stale {candidate} counts")
        generated_iwild = generated_tracks["iwildcam_H_v2"]
        if generated_iwild.get("numeric_release_eligible") is not False:
            problems.append("generated paper manifest does not withhold iWildCam numerics")
        if (
            generated_iwild.get("n_test") is not None
            or generated_iwild.get("regret") is not None
            or any(value is not None for value in generated_iwild.get("decision_counts", {}).values())
        ):
            problems.append("generated paper manifest exposes iWildCam sample, performance, or action values")
        if any(generated_iwild.get(field) is not None for field in ("ci_vs_adapt", "ci_vs_freeze", "seal")):
            problems.append("generated paper manifest exposes an iWildCam interval or current seal")
        accounting_iwild = row_by_track(accounting, "iWildCam H v2")
        if accounting_iwild.get("numeric_release_eligible") is not False or any(
            accounting_iwild.get(field) is not None for field in ("n", "ADAPT", "FREEZE", "ABSTAIN", "FA_u")
        ):
            problems.append("generated decision-accounting summary exposes iWildCam values")

        uniform = json.loads(UNIFORM_VERDICTS.read_text())
        for candidate, label in (("tent", "CIFAR-10-C Tent"), ("eata", "CIFAR-10-C EATA")):
            row = row_by_track(uniform["wave"], label)
            if row.get("decision_counts") != expected_counts[candidate]:
                problems.append(f"uniform verdicts have stale {candidate} decision counts")
            if row.get("survives_preregistered_six_comparison_holm") is not False:
                problems.append(f"uniform verdicts overstate preregistered family inference for {candidate}")
        uniform_iwild = row_by_track(uniform["wave"], "iWildCam H v2")
        if uniform_iwild.get("numeric_release_eligible") is not False or any(
            uniform_iwild.get(field) is not None for field in ("regret_kga", "regret_adapt", "regret_freeze", "FA_u")
        ):
            problems.append("uniform verdicts expose iWildCam performance values")
        if "n=" in uniform_iwild.get("unit", ""):
            problems.append("uniform verdicts expose the withheld iWildCam sample count")

        decision_metrics = json.loads(DECISION_METRICS.read_text())
        for candidate, label in (("tent", "CIFAR-10-C TENT"), ("eata", "CIFAR-10-C EATA")):
            row = row_by_track(decision_metrics["tracks"], label)
            observed = {action.upper(): row["actions"][action]["count"] for action in ("adapt", "freeze", "abstain")}
            if observed != expected_counts[candidate]:
                problems.append(f"decision metrics have stale {candidate} action counts")
            if row["false_adapt_conditional"]["n_adapt_decisions"] != expected_counts[candidate]["ADAPT"]:
                problems.append(f"decision metrics have stale {candidate} conditional denominator")
        metrics_iwild = row_by_track(decision_metrics["tracks"], "iWildCam")
        if (
            metrics_iwild.get("numeric_release_eligible") is not False
            or metrics_iwild.get("n_decisions") is not None
            or any(
                metrics_iwild["actions"][action]["count"] is not None
                or metrics_iwild["actions"][action]["rate"] is not None
                for action in ("adapt", "freeze", "abstain")
            )
        ):
            problems.append("decision metrics expose iWildCam action values")

        claim_matrix_text = CLAIM_MATRIX.read_text()
        required_iwild_matrix_row = (
            "| iWildCam numerical/action evidence | iWildCam | none promoted | "
            "claim_ledger.json (KB-CLAIM-021) | excluded from numerical tables and routing claims | "
            "withheld: archived metric contract is invalid; population-sealed official-metric "
            "rerun required |"
        )
        if required_iwild_matrix_row not in claim_matrix_text:
            problems.append("generated empirical claim matrix does not withhold iWildCam evidence")

        claim_ledger = json.loads(CLAIM_LEDGER.read_text())
        iwild_claims = [row for row in claim_ledger["claims"] if row.get("claim_id") == "KB-CLAIM-021"]
        if len(iwild_claims) != 1 or iwild_claims[0].get("status") != "withheld":
            problems.append("claim ledger does not mark KB-CLAIM-021 withheld")
        cifar_claims = [row for row in claim_ledger["claims"] if row.get("claim_id") == "KB-CLAIM-010"]
        if len(cifar_claims) != 1:
            problems.append("claim ledger does not contain exactly one KB-CLAIM-010")
        else:
            cluster_claim = cifar_claims[0].get("current_policy_family_sensitivity", {})
            if cluster_claim.get("status") != "retrospective_current_policy_family_sensitivity":
                problems.append("claim ledger does not expose the current family sensitivity")
            if "current-policy cluster-robust win" not in cifar_claims[0].get("forbidden_wording", []):
                problems.append("claim ledger does not forbid a current-policy cluster-robust win")
        for source in ACTIVE_SOURCES:
            for hit in authority.scan_text_for_unreleased_curated(
                live_latex(source.read_text(errors="ignore")), claim_ledger
            ):
                problems.append(f"{source.relative_to(ROOT)} promotes unreleased {hit['claim_id']}: {hit['snippet']}")
        result_manifest = json.loads(RESULT_MANIFEST.read_text())
        if any(row.get("claim_id") == "KB-CLAIM-021" for row in result_manifest["results"]):
            problems.append("promoted result manifest still contains withheld KB-CLAIM-021")
        result_cifar = next(
            (row for row in result_manifest["results"] if row.get("claim_id") == "KB-CLAIM-010"),
            None,
        )
        if result_cifar is None:
            problems.append("promoted result manifest omits KB-CLAIM-010")
        else:
            sensitivity = result_cifar.get("metrics", {}).get("current_policy_family_sensitivity", {})
            if sensitivity.get("confirmatory") is not False:
                problems.append("promoted result manifest overstates the family sensitivity")
            tent_sensitivity = sensitivity.get("candidates", {}).get("tent", {})
            if tent_sensitivity.get("preregistered_six_comparison_holm_rejects_both") is not False:
                problems.append("promoted result manifest incorrectly passes Tent's preregistered gate")
        if SOURCE_MANIFEST.is_file():
            recorded_source_hash = result_manifest.get("reconciliation_source", {}).get("source_manifest_sha256")
            if recorded_source_hash != file_sha256(SOURCE_MANIFEST):
                problems.append("promoted result manifest exposes a stale source-manifest hash")
        results_source = json.loads(RESULTS_SOURCE.read_text())
        source_iwild = results_source["tracks"]["iwildcam_H_v2"]
        if (
            source_iwild.get("n_test") is not None
            or source_iwild.get("regret") is not None
            or any(source_iwild.get(field) is not None for field in ("ci_vs_adapt", "ci_vs_freeze", "seal"))
        ):
            problems.append("legacy compatibility view exposes iWildCam sample/performance values or a current seal")

        historical_text = HISTORICAL_LEDGER.read_text()
        if "HISTORICAL VALUES BELOW ARE NOT RELEASE VALUES" not in historical_text:
            problems.append("historical submission ledger lacks a current-value demotion banner")
        result_audit_text = RESULT_AUDIT.read_text()
        if "| iWildCam | withheld | withheld | withheld | withheld | withheld | withheld |" not in result_audit_text:
            problems.append("current result audit does not withhold the iWildCam numerical row")
        if "preregistered six-comparison Holm p-values are 0.09375" not in result_audit_text:
            problems.append("current result audit omits the failed preregistered Tent Holm result")
        if (
            "earlier KGA policy" not in result_audit_text
            or "confidence intervals are unadjusted" not in result_audit_text
        ):
            problems.append("current result audit omits the historical POEM/AETTA policy and Holm scope")

        claim_manifest_text = CLAIM_MANIFEST.read_text()
        if "preregistered six-comparison Holm fails" not in claim_manifest_text:
            problems.append("claim manifest omits the failed preregistered cluster gate")
        if "Holm applies only to archived p-values" not in claim_manifest_text:
            problems.append("claim manifest omits the historical POEM/AETTA Holm scope")

        readme_text = README.read_text()
        if "preregistered six-comparison Holm gate fails" not in readme_text:
            problems.append("README omits the failed preregistered cluster gate")
        if "iWildCam numerical/action row is withheld" not in readme_text:
            problems.append("README does not clearly withhold iWildCam numerical/action evidence")

    if problems:
        print("Manuscript claim validation: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Manuscript claim validation: PASS")
    print(
        f"Checked {len(ACTIVE_SOURCES)} maintained LaTeX sources plus the synchronized long driver, "
        "the canonical panel, "
        f"{len(release_paths)} release consistency surfaces, {direct_storage_count} direct storage hashes, "
        f"and {sealed_storage_count} sealed evidence hashes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
