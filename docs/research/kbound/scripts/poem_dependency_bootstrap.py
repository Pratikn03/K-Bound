"""Run pinned POEM with an explicitly authenticated SAR model import.

This dependency repair is not a native benchmark attestation. CUDA, weights,
other source files, data, protocol and environment need their own verification.
No CUDA-to-MPS rewriting, model substitution or upstream file edits occur here.
The optional single-corruption mode is a reviewed native-algorithm protocol
adapter: only the authenticated driver's initial corruption-list literal changes
in memory. Without that option, the original driver bytes execute unchanged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

POEM_COMMIT = "8911b8514346706f916b609a1d9a3652197a05bf"
SAR_COMMIT = "20f6e24b17525f34503510afccedc0629b67b7c4"
MAIN_SHA256 = "0885d42a2657d182d89a425df6a6406a1e57aba1ca3eb385cffbfdea5ead356f"
MODEL_SHA256 = "266528c80d4232ccf284d325a5c173e6ee97a6b682cd83fafbfad37be9e0171e"
NATIVE_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)
DERIVATION_RECEIPT = "POEM_PROTOCOL_DERIVATION_RECEIPT.json"


def authenticated_bytes(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return payload


def bind_model(path: Path, expected: str) -> None:
    if "models" in sys.modules or "models.Res" in sys.modules:
        raise RuntimeError("models package already loaded; use a fresh process")
    payload = authenticated_bytes(path, expected)
    package = types.ModuleType("models")
    package.__path__ = []  # Never search an ignored local compatibility shim.
    model = types.ModuleType("models.Res")
    model.__file__ = str(path)
    model.__package__ = "models"
    sys.modules["models"] = package
    sys.modules["models.Res"] = model
    try:
        exec(compile(payload, str(path), "exec"), model.__dict__)
        package.Res = model
    except BaseException:
        sys.modules.pop("models.Res", None)
        sys.modules.pop("models", None)
        raise


def _unique_long_options(argv: list[str], allowed: set[str]) -> None:
    """Reject duplicate options and abbreviations that upstream argparse accepts."""
    seen: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        if option not in allowed:
            raise ValueError(f"unsupported or abbreviated option in protocol adapter: {option}")
        if option in seen:
            raise ValueError(f"duplicate option in protocol adapter: {option}")
        seen.add(option)


def _single_corruption_output(payload: bytes, upstream: list[str], corruption: str) -> Path:
    if corruption not in NATIVE_CORRUPTIONS:
        raise ValueError(f"unknown native corruption: {corruption}")
    tree = ast.parse(payload)
    native_options = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }
    _unique_long_options(upstream, native_options)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--exp_type", required=True, choices=["normal"])
    parser.add_argument("--corruption", required=True, choices=[corruption])
    parser.add_argument("--output", required=True, type=Path)
    selected, _ = parser.parse_known_args(upstream)
    return selected.output.absolute()


def derive_single_corruption(payload: bytes, corruption: str) -> tuple[bytes, dict]:
    """Replace only the unique native list literal, preserving all other bytes."""
    if corruption not in NATIVE_CORRUPTIONS:
        raise ValueError(f"unknown native corruption: {corruption}")
    tree = ast.parse(payload)
    guards = [
        node for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
        and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    ]
    assignments = [
        node for guard in guards for node in guard.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "common_corruptions" for target in node.targets)
    ]
    if len(guards) != 1 or len(assignments) != 1:
        raise ValueError("expected one direct common_corruptions initializer in the native main guard")
    assignment = assignments[0]
    literal = assignment.value
    if (
        len(assignment.targets) != 1 or not isinstance(literal, ast.List)
        or any(not isinstance(element, ast.Constant) or not isinstance(element.value, str) for element in literal.elts)
        or tuple(element.value for element in literal.elts) != NATIVE_CORRUPTIONS
    ):
        raise ValueError("common_corruptions initializer does not match the native list literal")
    # AST columns are UTF-8 byte offsets; do not slice a decoded Unicode string.
    lines = payload.splitlines(keepends=True)
    start = sum(map(len, lines[:literal.lineno - 1])) + literal.col_offset
    end = sum(map(len, lines[:literal.end_lineno - 1])) + literal.end_col_offset
    replacement = repr([corruption]).encode("ascii")
    derived = payload[:start] + replacement + payload[end:]
    ast.parse(derived)
    return derived, {
        "kind": "single_corruption_literal",
        "exp_type": "normal",
        "corruption": corruption,
        "literal_byte_span": [start, end],
        "original_literal": payload[start:end].decode("utf-8"),
        "replacement_literal": replacement.decode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--poem-source", required=True, type=Path)
    parser.add_argument("--sar-source", required=True, type=Path)
    parser.add_argument("--single-corruption", action="append", metavar="NAME",
                        help="Reviewed normal-mode protocol adapter; records a literal-only source derivation")
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    upstream = args.upstream_args
    if upstream[:1] == ["--"]:
        upstream = upstream[1:]
    if args.single_corruption is not None:
        if len(args.single_corruption) != 1:
            parser.error("--single-corruption must appear exactly once")
        wrapper = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
        _unique_long_options(wrapper, {"--poem-source", "--sar-source", "--single-corruption"})
    poem, sar = args.poem_source.resolve(), args.sar_source.resolve()
    for source, expected in ((poem, POEM_COMMIT), (sar, SAR_COMMIT)):
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise ValueError(f"Pinned revision mismatch: {source}")
    entry = poem / "main.py"
    payload = authenticated_bytes(entry, MAIN_SHA256)
    if args.single_corruption is not None:
        corruption = args.single_corruption[0]
        output = _single_corruption_output(payload, upstream, corruption)
        derived, override = derive_single_corruption(payload, corruption)
        # Authenticate the dependency before creating any derivation artifacts;
        # bind_model checks it again immediately before executing its source.
        authenticated_bytes(sar / "models/Res.py", MODEL_SHA256)
        receipt = {
            "schema": "kbound_poem_protocol_derivation_v1",
            "status": "DERIVATION_RECORDED_BEFORE_EXECUTION",
            "scope": "reviewed native-algorithm protocol adapter; not unchanged native-driver execution",
            "benchmark_attestation": False,
            "upstream_entrypoint": str(entry),
            "poem_commit": POEM_COMMIT,
            "sar_commit": SAR_COMMIT,
            "model_source_sha256": MODEL_SHA256,
            "original_main_sha256": hashlib.sha256(payload).hexdigest(),
            "derived_main_sha256": hashlib.sha256(derived).hexdigest(),
            "override": override,
            "upstream_argv": upstream,
        }
        # Exclusive directory creation rejects previous runs and concurrent writers.
        output.mkdir(parents=True, exist_ok=False)
        with (output / DERIVATION_RECEIPT).open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        payload = derived
    bind_model(sar / "models/Res.py", MODEL_SHA256)
    sys.path.insert(0, str(poem))
    sys.argv = [str(entry), *upstream]
    exec(compile(payload, str(entry), "exec"), {
        "__name__": "__main__", "__file__": str(entry), "__package__": None,
    })


if __name__ == "__main__":
    main()
