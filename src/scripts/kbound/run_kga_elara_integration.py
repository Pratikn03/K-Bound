#!/usr/bin/env python3
"""Run the versioned, claim-safe KGA-over-ELARA research integration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kga.integrations.claims import assess_promotion  # noqa: E402
from kga.integrations.elara import (  # noqa: E402
    ELARAKGAGuard,
    EvaluationMode,
    FrozenLinearBenefitEstimator,
    evaluate_result,
)

PROTOCOL_SCHEMA = "kga_elara_integration_protocol_v1"
RESULT_SCHEMA = "kga_elara_integrated_results_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _portable_path(path_value: str | Path) -> str:
    path = Path(path_value).expanduser().resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _portable_inventory(tracks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **track,
            "cache": _portable_path(str(track["cache"])),
            "files": [_portable_path(str(path)) for path in track["files"]],
        }
        for track in tracks
    ]


def _load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = yaml.safe_load(raw)
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"protocol schema must be {PROTOCOL_SCHEMA}")
    if not isinstance(protocol.get("tracks"), list) or not protocol["tracks"]:
        raise ValueError("protocol must declare at least one track")
    EvaluationMode(str(protocol.get("mode", "")))
    alpha = float(protocol.get("alpha", 0.0))
    if not 0.0 < alpha < 1.0:
        raise ValueError("protocol alpha must be in (0, 1)")
    return protocol, hashlib.sha256(raw).hexdigest()


def _inventory(protocol: dict) -> tuple[list[dict[str, object]], list[str], list[Path]]:
    tracks: list[dict[str, object]] = []
    missing: list[str] = []
    all_files: list[Path] = []
    for declaration in protocol["tracks"]:
        name = str(declaration["name"])
        cache = _resolve(str(declaration["cache"]))
        pattern = str(declaration.get("pattern", "*.npz"))
        files = sorted(cache.glob(pattern)) if cache.is_dir() else []
        tracks.append(
            {
                "name": name,
                "cache": str(cache),
                "pattern": pattern,
                "matched_files": len(files),
                "files": [str(path) for path in files],
            }
        )
        all_files.extend(files)
        if not files:
            missing.append(name)
    return tracks, missing, all_files


def _load_estimator(protocol: dict) -> FrozenLinearBenefitEstimator | None:
    if str(protocol["mode"]) != EvaluationMode.LABEL_FREE.value:
        return None
    declaration = protocol.get("frozen_estimator")
    if not isinstance(declaration, dict) or "path" not in declaration:
        raise ValueError("label_free protocol requires frozen_estimator.path")
    path = _resolve(str(declaration["path"]))
    with np.load(path, allow_pickle=False) as data:
        feature_names = tuple(str(item) for item in data["feature_names"].tolist())
        protocol_hash = str(data["protocol_hash"].item())
        return FrozenLinearBenefitEstimator(
            feature_names=feature_names,
            weights=data["weights"],
            intercept=float(data["intercept"].item()),
            residuals=data["residuals"],
            protocol_hash=protocol_hash,
        )


def _probe_indices(n: int, k: int, seed: int) -> np.ndarray:
    if k <= 0 or k > n:
        raise ValueError(f"probe_k must be in [1, {n}], got {k}")
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def _mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row["evaluation"][key]) for row in rows if np.isfinite(float(row["evaluation"][key]))]
    return float(np.mean(values)) if values else None


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "n_categories": 0,
            "mean_auroc_frozen": None,
            "mean_auroc_candidate": None,
            "mean_auroc_kga": None,
            "regret_always_freeze": None,
            "regret_always_adapt": None,
            "regret_kga": None,
            "coverage": 0.0,
            "false_adapt_rate_unconditional": 0.0,
            "false_adapt_rate_conditional": 0.0,
            "labels_used_for_decision": 0,
            "decisions": {"ADAPT": 0, "FREEZE": 0, "ABSTAIN": 0},
        }
    decisions = Counter(str(row["decision"]["decision"]) for row in rows)
    n = len(rows)
    n_adapt = decisions["ADAPT"]
    false_adapts = sum(bool(row["evaluation"]["false_adapt"]) for row in rows)
    return {
        "n_categories": n,
        "mean_auroc_frozen": _mean(rows, "auroc_frozen"),
        "mean_auroc_candidate": _mean(rows, "auroc_candidate"),
        "mean_auroc_kga": _mean(rows, "auroc_kga"),
        "regret_always_freeze": _mean(rows, "regret_frozen"),
        "regret_always_adapt": _mean(rows, "regret_candidate"),
        "regret_kga": _mean(rows, "regret_kga"),
        "coverage": float(sum(bool(row["evaluation"]["covered"]) for row in rows) / n),
        "false_adapt_rate_unconditional": float(false_adapts / n),
        "false_adapt_rate_conditional": float(false_adapts / n_adapt) if n_adapt else 0.0,
        "labels_used_for_decision": int(sum(int(row["decision"]["labels_used_for_decision"]) for row in rows)),
        "decisions": {key: int(decisions[key]) for key in ("ADAPT", "FREEZE", "ABSTAIN")},
    }


def _json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _fmt(value: object, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _tex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def _write_table(path: Path, tracks: list[dict[str, object]], mode: str, eligible: bool) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\caption{KGA over ELARA-U on opened multimodal caches. This is a retrospective audit, not a label-free or headline claim. AUROC is averaged over valid categories.}",
        r"\label{tab:kga-elara-integrated}",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Track & $n$ & freeze & ELARA & KGA & A/F/U & status \\",
        r"\midrule",
    ]
    for track in tracks:
        aggregate = track["aggregate"]
        decisions = aggregate["decisions"]
        decision_text = f"{decisions['ADAPT']}/{decisions['FREEZE']}/{decisions['ABSTAIN']}"
        lines.append(
            f"{_tex_escape(str(track['name']))} & {track['n_valid_categories']} & "
            f"{_fmt(aggregate['mean_auroc_frozen'])} & {_fmt(aggregate['mean_auroc_candidate'])} & "
            f"{_fmt(aggregate['mean_auroc_kga'])} & {decision_text} & audit only \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\par\\vspace{{2pt}}\\scriptsize Mode: {_tex_escape(mode)}; promotion eligible: {'yes' if eligible else 'no'}.",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_findings(path: Path, results: dict) -> None:
    aggregate = results["aggregate"]
    reasons = ", ".join(results["claim_eligibility"]["reasons"])
    text = (
        "# KGA-over-ELARA Integrated Audit\n\n"
        "**RETROSPECTIVE - NOT A LABEL-FREE OR HEADLINE CLAIM.**\n\n"
        f"- Mode: `{results['mode']}`\n"
        f"- Valid categories: {aggregate['n_categories']}\n"
        f"- Mean AUROC: freeze {_fmt(aggregate['mean_auroc_frozen'], 4)}, "
        f"ELARA {_fmt(aggregate['mean_auroc_candidate'], 4)}, KGA {_fmt(aggregate['mean_auroc_kga'], 4)}\n"
        f"- Coverage: {_fmt(aggregate['coverage'], 4)}\n"
        f"- False-adapt (unconditional): {_fmt(aggregate['false_adapt_rate_unconditional'], 4)}\n"
        f"- Promotion eligible: {results['claim_eligibility']['eligible']}\n"
        f"- Ineligibility reasons: {reasons}\n"
    )
    path.write_text(text, encoding="utf-8")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_protocol(
    protocol_path: Path,
    output_dir: Path | None = None,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Run one declared KGA-ELARA protocol or return its dry-run inventory."""

    protocol_path = Path(protocol_path).resolve()
    protocol, protocol_hash = _load_protocol(protocol_path)
    inventory, missing_tracks, files = _inventory(protocol)
    serialized_inventory = _portable_inventory(inventory)
    if dry_run:
        return {
            "dry_run": True,
            "protocol": _portable_path(protocol_path),
            "protocol_sha256": protocol_hash,
            "tracks": serialized_inventory,
            "missing_tracks": missing_tracks,
            "n_matched_files": len(files),
        }

    output = Path(output_dir).resolve() if output_dir is not None else _resolve(str(protocol["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and not overwrite:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("refusing to overwrite results from a different protocol hash")

    mode = EvaluationMode(str(protocol["mode"]))
    estimator = _load_estimator(protocol)
    guard = ELARAKGAGuard(alpha=float(protocol["alpha"]), router_action=str(protocol["router_action"]))
    track_results: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    invalid_categories: list[dict[str, str]] = []
    probe_k = int(protocol.get("probe_k", 0))
    probe_seed = int(protocol.get("probe_seed", 20260615))

    for track in inventory:
        rows: list[dict[str, object]] = []
        invalid: list[dict[str, str]] = []
        for file_index, file_name in enumerate(track["files"]):
            path = Path(str(file_name))
            try:
                with np.load(path, allow_pickle=False) as data:
                    required = {"Sval", "yval", "Stest", "ytest"}
                    missing_keys = sorted(required - set(data.files))
                    if missing_keys:
                        raise ValueError(f"missing cache keys: {missing_keys}")
                    s_val = data["Sval"]
                    y_val = data["yval"]
                    s_test = data["Stest"]
                    y_test = data["ytest"]
                kwargs: dict[str, object] = {}
                if mode is EvaluationMode.RETROSPECTIVE_AUDIT:
                    kwargs["y_test"] = y_test
                elif mode is EvaluationMode.TARGET_LABEL_LIGHT:
                    kwargs["y_test"] = y_test
                    kwargs["probe_indices"] = _probe_indices(len(y_test), probe_k, probe_seed + file_index)
                else:
                    kwargs["estimator"] = estimator
                decision = guard.decide(
                    s_val=s_val,
                    y_val=y_val,
                    s_test=s_test,
                    mode=mode,
                    **kwargs,
                )
                evaluation = evaluate_result(decision, y_test)
                row = {
                    "file": path.name,
                    "input_sha256": _sha256(path),
                    "decision": decision.to_record(),
                    "evaluation": evaluation,
                }
                rows.append(row)
                all_rows.append(row)
            except Exception as exc:
                failure = {"track": str(track["name"]), "file": path.name, "error": repr(exc)}
                invalid.append(failure)
                invalid_categories.append(failure)
        track_results.append(
            {
                "name": track["name"],
                "cache": _portable_path(str(track["cache"])),
                "pattern": track["pattern"],
                "n_matched_files": track["matched_files"],
                "n_valid_categories": len(rows),
                "invalid_categories": invalid,
                "aggregate": _aggregate(rows),
                "categories": rows,
            }
        )

    aggregate = _aggregate(all_rows)
    complete = not missing_tracks and not invalid_categories
    promotion_input = {
        "mode": mode.value,
        "frozen_estimator_verified": estimator is not None,
        "held_out_natural_datasets": 0,
        "frozen_before_scoring": bool(protocol.get("integrity", {}).get("frozen_before_target_scoring", False)),
        "independent_splits": 0,
        "regret_kga": aggregate["regret_kga"] if aggregate["regret_kga"] is not None else 1.0,
        "regret_always_adapt": aggregate["regret_always_adapt"] if aggregate["regret_always_adapt"] is not None else 0.0,
        "regret_always_freeze": aggregate["regret_always_freeze"] if aggregate["regret_always_freeze"] is not None else 0.0,
        "false_adapt_rate": aggregate["false_adapt_rate_unconditional"],
        "alpha": float(protocol["alpha"]),
        "coverage": aggregate["coverage"],
        "confidence_intervals_complete": False,
        "strong_baselines_complete": False,
        "required_tracks_complete": complete,
        "integrity_failures": ["opened_retrospective_data"] if protocol["status"] == "RETROSPECTIVE_OPENED_DATA" else [],
    }
    claim_eligibility = assess_promotion(promotion_input)
    results = {
        "schema": RESULT_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol": _portable_path(protocol_path),
        "protocol_sha256": protocol_hash,
        "protocol_status": protocol["status"],
        "claim_scope": protocol["claim_scope"],
        "mode": mode.value,
        "alpha": float(protocol["alpha"]),
        "tracks": track_results,
        "aggregate": aggregate,
        "missing_tracks": missing_tracks,
        "invalid_categories": invalid_categories,
        "claim_eligibility": claim_eligibility,
    }
    manifest = {
        "schema": "kga_elara_run_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": _portable_path(protocol_path),
        "protocol_sha256": protocol_hash,
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in ("numpy", "scipy", "scikit-learn", "PyYAML")},
        "inputs": [{"path": _portable_path(path), "sha256": _sha256(path)} for path in files],
    }
    _json_dump(output / "results.json", results)
    _write_table(output / "results_table.tex", track_results, mode.value, bool(claim_eligibility["eligible"]))
    _write_findings(output / "FINDINGS.md", results)
    _json_dump(manifest_path, manifest)
    return {
        "dry_run": False,
        "output_dir": _portable_path(output),
        "tracks": serialized_inventory,
        "missing_tracks": missing_tracks,
        "n_matched_files": len(files),
        "n_valid_categories": len(all_rows),
        "claim_eligibility": claim_eligibility,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_protocol(
        args.protocol,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
