#!/usr/bin/env python3
"""Item 14: tolerance-diff an external reproducer's numbers against the paper's promoted values.

The reproducer follows REVIEWER_REPRO_PACKET.md Part B and sends back a JSON of the metrics they
measured. This script compares against the promoted values parsed from
paper/generated/kbound_numbers.tex (the same macros the paper compiles from) and exits nonzero on
any out-of-tolerance mismatch, printing a per-metric report for the sign-off form (Part D).

Their JSON schema (exactly all declared keys, finite JSON numbers only):
  {"cifar_tent_kga_regret": 0.0016, "cifar_tent_fa_u": 0.0, ...}
Key -> macro mapping below; extend MAPPING as the packet grows.
"""
import argparse, json, re, sys, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
NUMBERS_TEX = os.path.join(HERE, "..", "paper", "generated", "kbound_numbers.tex")

# reproducer key -> (tex macro, absolute tolerance)
MAPPING = {
    "cifar_tent_kga_regret":  ("CIFARtentKga",   0.0005),
    "cifar_tent_adapt_regret":("CIFARtentAdapt", 0.0005),
    "cifar_tent_freeze_regret":("CIFARtentFreeze",0.0010),
    "cifar_tent_fa_u":        ("CIFARtentFA",    0.0001),
    "cifar_eata_kga_regret":  ("CIFAReataKga",   0.0005),
    "cifar_eata_adapt_regret":("CIFAReataAdapt", 0.0005),
    "cifar_eata_freeze_regret":("CIFAReataFreeze",0.0010),
    "cifar_eata_fa_u":        ("CIFAReataFA",    0.0001),
    "headtohead_kga_regret":  ("HeadToHeadKga",  0.0005),
    "headtohead_poem_regret": ("HeadToHeadPoem", 0.0005),
    "headtohead_aetta_regret":("HeadToHeadAetta",0.0005),
}


def parse_macros(path: str) -> dict:
    txt = open(path).read()
    out = {}
    for name, body in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", txt):
        m = re.search(r"-?\d+\.?\d*", body.replace("$", ""))
        if m:
            out[name] = float(m.group())
    return out


def load_results(path: str) -> dict:
    def reject_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with open(path, encoding="utf-8") as handle:
        result = json.load(handle, parse_constant=reject_constant,
                           object_pairs_hook=unique_pairs)
    if not isinstance(result, dict) or set(result) != set(MAPPING):
        raise ValueError("metric keys must match the complete declared set exactly")
    for key, value in result.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"metric {key} must be a finite JSON number")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--their-results", required=True)
    ap.add_argument("--numbers-tex", default=NUMBERS_TEX)
    args = ap.parse_args()

    try:
        macros = parse_macros(args.numbers_tex)
        theirs = load_results(args.their_results)
    except (OSError, ValueError, OverflowError) as exc:
        print(f"[verify] INVALID RESULTS — {exc}")
        sys.exit(2)

    failures, checked = [], 0
    for key, value in sorted(theirs.items()):
        macro, tol = MAPPING[key]
        if macro not in macros:
            print(f"  !  {key}: macro \\{macro} not found in kbound_numbers.tex")
            failures.append(key)
            continue
        ref = macros[macro]
        ok = abs(float(value) - ref) <= tol
        checked += 1
        print(f"  {'OK ' if ok else 'FAIL'} {key}: theirs={value} paper={ref} tol={tol}")
        if not ok:
            failures.append(key)

    print(f"\n[verify] {checked} metrics checked, {len(failures)} failures.")
    if failures:
        print("[verify] MISMATCH — do not sign off; investigate before any paper edit.")
        sys.exit(1)
    print("[verify] Within tolerance — attach this output to REVIEWER_REPRO_PACKET.md Part D.")


if __name__ == "__main__":
    main()
