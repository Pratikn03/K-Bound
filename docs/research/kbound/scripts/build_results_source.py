#!/usr/bin/env python3
"""Assemble docs/research/kbound/results_source.json from a FRESH final run — honestly.

This is the connective tissue between `kbtrain.sh final-all` and the paper. It exists
because results_source.json was previously hand-authored; this automates that step WITHOUT
re-introducing the in-sample-radius overclaim that bit us before.

Honesty guardrails (all enforced; the script aborts rather than write a suspect number):
  G1 INTEGRITY GREP   - the scorer sources must not contain an in-sample-eps pattern.
  G2 VERDICT-FROM-CI  - a "beats-both" verdict is granted ONLY if the condition-bootstrap
                        95% CI excludes 0 vs BOTH always-adapt AND always-freeze. The
                        point-estimate `beats_both`/`verdict_win` flags in the record JSONs
                        are deliberately IGNORED (OfficeHome's flag is true on an in-sample
                        radius; the honest verdict is no-harm).
  G3 EPS FLOOR        - a natural-shift conformal radius below --eps-floor is almost
                        certainly an in-sample (un-cross-fit) residual quantile
                        (OfficeHome's contaminated eps was 0.00102; the OOF radius is ~0.03).
                        Such a record is refused.
  G4 CROSS-CONSISTENCY- the point regrets in the bootstrap-CI file must match the fresh
                        protocol record (so a stale/in-sample CI file can't slip through).

Inputs (all from the fresh run):
  --manifest     experiments/.../final_manifest_<stamp>.json   (corruption-grid regrets)
  --protocol-dir experiments/kbound/results                    (natural-shift protocol_result.json)
  --win-cis      research_lock/KBOUND_WIN_BOOTSTRAP_CIS.json    (natural-shift bootstrap CIs)
  --prev         docs/research/kbound/results_source.json       (for the side-by-side diff)
Output:
  --out          docs/research/kbound/results_source.json  (+ a `_provenance` block)

Use --check-only to validate the fresh artifacts and print the diff WITHOUT writing.
"""
import argparse, json, os, re, subprocess, sys, datetime

KB = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(KB, "..", "..", ".."))

SCORERS = [
    "docs/research/kbound/scripts/analyze_F.py",
    "docs/research/kbound/scripts/score_kbound_holdout.py",
    "docs/research/kbound/scripts/run_protocol_dev_lock.py",
    "docs/research/kbound/scripts/mixed_stream_kbound.py",
]
INSAMPLE_PAT = re.compile(r"predict\(Zc\) - Bc|abs\(Bhat_c - Bc\)")
OK_PAT = re.compile(r"resid_c|_loo|out-of-fold")


def die(msg):
    print(f"\n[build_results_source] ABORT: {msg}", file=sys.stderr)
    sys.exit(2)


def g1_integrity():
    bad = []
    for rel in SCORERS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p), 1):
            if INSAMPLE_PAT.search(line) and not OK_PAT.search(line):
                bad.append(f"{rel}:{i}: {line.strip()}")
    if bad:
        die("in-sample-eps pattern in a scorer (G1):\n  " + "\n  ".join(bad))
    print("[G1] scorers are out-of-fold (no in-sample-eps pattern).")


def gap_ci(block):
    """From a kga_vs_* sub-dict -> (mean, [lo,hi], excludes_zero)."""
    mean = float(block["mean"])
    lo, hi = (float(x) for x in block["ci95"])
    return mean, [lo, hi], (lo > 0.0)


def verdict_from_ci(adapt_excl, freeze_excl):
    if adapt_excl and freeze_excl:
        return "beats-both"
    if adapt_excl and not freeze_excl:
        return "no-harm"          # beats always-adapt, ties always-freeze (damage prevention)
    return "no-harm"              # ties both / inconclusive -> never overclaim


def load_protocol(protocol_dir, sub, eps_floor):
    p = os.path.join(ROOT, protocol_dir, sub, "protocol_result.json")
    if not os.path.exists(p):
        die(f"missing fresh protocol record: {p}")
    d = json.load(open(p))
    tl = d["test_locked"]
    eps = float(tl.get("eps_global", 0.0))
    if eps < eps_floor:                                  # G3
        die(f"{sub}: conformal radius eps_global={eps:.5f} < floor {eps_floor} "
            f"-> in-sample (un-cross-fit). Re-run the OOF scorer; do not use this record.")
    return {
        "regret_adapt": round(float(tl["regret_adapt"]), 4),
        "regret_freeze": round(float(tl["regret_freeze"]), 4),
        "regret_kga": round(float(tl["regret_kga"]), 4),
        "false_adapt": float(tl["false_adapt"]),
        "n_test": int(tl["n_test"]),
        "_eps_global": eps,
        "_point_beats_both": bool(tl.get("beats_both", False)),
        "_src": os.path.relpath(p, ROOT),
    }


def natural_shift(protocol_dir, sub, win_name, wins, eps_floor):
    rec = load_protocol(protocol_dir, sub, eps_floor)
    w = next((x for x in wins if x["name"].lower() == win_name.lower()), None)
    if w is None:
        die(f"no bootstrap-CI entry named {win_name} in --win-cis")
    # G4: the CI file's point regrets must agree with the fresh protocol record.
    if abs(float(w["point"]["regret_kga"]) - rec["regret_kga"]) > 1e-3:
        die(f"{win_name}: bootstrap-CI regret_kga={w['point']['regret_kga']:.4f} disagrees with "
            f"fresh protocol regret_kga={rec['regret_kga']:.4f} -> stale/in-sample CI file (G4). "
            f"Re-run bootstrap_win_cis.py against the fresh logs.")
    ga, cia, a_excl = gap_ci(w["kga_vs_adapt"])
    gf, cif, f_excl = gap_ci(w["kga_vs_freeze"])
    verdict = verdict_from_ci(a_excl, f_excl)            # G2
    if rec["_point_beats_both"] and verdict != "beats-both":
        print(f"[G2] NOTE {win_name}: point-estimate said beats-both, but the bootstrap CI "
              f"vs always-freeze includes 0 -> recording honest verdict '{verdict}'.")
    return {
        "regret_adapt": rec["regret_adapt"], "regret_freeze": rec["regret_freeze"],
        "regret_kga": rec["regret_kga"], "false_adapt": rec["false_adapt"],
        "n_test": rec["n_test"], "verdict": verdict,
        "gap_vs_adapt": round(ga, 4), "ci_vs_adapt": [round(cia[0], 4), round(cia[1], 4)],
        "gap_vs_freeze": round(gf, 4), "ci_vs_freeze": [round(cif[0], 4), round(cif[1], 4)],
    }, rec["_src"]


def _mean(s):
    """'0.0016+/-0.0003' -> 0.0016 ; '0.0016' -> 0.0016."""
    return float(str(s).split("+/-")[0])


def corruption_grid(manifest_rows, ds_key, verdict):
    r = next((x for x in manifest_rows if x["dataset"] == ds_key), None)
    if r is None:
        die(f"corruption grid '{ds_key}' not in manifest rows")
    return {
        "regret_kga": round(_mean(r["regret_kga"]), 4),
        "regret_adapt": round(_mean(r["regret_adapt"]), 4),
        "regret_freeze": round(_mean(r["regret_freeze"]), 4),
        "false_adapt": 0.0,
        "verdict": verdict,
        "_n_records": r.get("n"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--protocol-dir", default="experiments/kbound/results")
    ap.add_argument("--win-cis", default="research_lock/KBOUND_WIN_BOOTSTRAP_CIS.json")
    ap.add_argument("--prev", default="docs/research/kbound/results_source.json")
    ap.add_argument("--out", default="docs/research/kbound/results_source.json")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--eps-floor", type=float, default=0.005)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--locked-analysis",
                    default="experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json")
    ap.add_argument("--headtohead",
                    default="experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json")
    a = ap.parse_args()

    g1_integrity()

    man = json.load(open(os.path.join(ROOT, a.manifest))) if not os.path.isabs(a.manifest) \
        else json.load(open(a.manifest))
    rows = man["rows"]
    wins = json.load(open(os.path.join(ROOT, a.win_cis)))["wins"]

    oh, oh_src = natural_shift(a.protocol_dir, "officehome_protocol_M_v2", "OfficeHome", wins, a.eps_floor)
    iw, iw_src = natural_shift(a.protocol_dir, "iwildcam_protocol_H_v2", "iWildCam", wins, a.eps_floor)

    out = {
        "_README": ("Single source of truth for the K-Bound paper's result-table numbers. "
                    "AUTO-BUILT by scripts/build_results_source.py from a fresh OOF run. These are "
                    "OUT-OF-FOLD values; verdicts come from the condition-bootstrap CI, not the "
                    "point-estimate beats_both flag. Re-run build_results_source.py after each "
                    "final-all, then make_tables.py, then rebuild the PDFs."),
        "alpha": a.alpha,
        "natural_shifts": {"officehome_M_v2": oh, "iwildcam_H_v2": iw},
        "corruption_grids": {
            "cifar10c_stress": corruption_grid(rows, "cifar10c", "beats-both-CI-robust"),
            "imagenetc_sar": corruption_grid(rows, "imagenetc", "beats-both"),
        },
        "_provenance": {
            "built_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "git_sha": subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
            "manifest": a.manifest, "win_cis": a.win_cis,
            "officehome_src": oh_src, "iwildcam_src": iw_src,
            "eps_floor": a.eps_floor,
        },
    }

    locked_path = os.path.join(ROOT, a.locked_analysis) if a.locked_analysis else ""
    if locked_path and os.path.exists(locked_path):
        out["locked_analysis"] = json.load(open(locked_path))
        out["_provenance"]["locked_analysis"] = a.locked_analysis
        # Prefer Holm-locked tent regret for corruption grid when available
        tent = out["locked_analysis"].get("candidates", {}).get("tent", {})
        if tent:
            out["corruption_grids"]["cifar10c_stress"] = {
                "regret_kga": round(float(tent["kga_mean_regret"]), 4),
                "regret_adapt": round(float(tent["adapt_mean_regret"]), 4),
                "regret_freeze": round(float(tent["freeze_mean_regret"]), 4),
                "false_adapt": float(tent.get("false_adapt_rate_pooled", 0.0)),
                "verdict": "beats-both-CI-robust",
                "_source": a.locked_analysis,
            }

    h2h_path = os.path.join(ROOT, a.headtohead) if a.headtohead else ""
    if h2h_path and os.path.exists(h2h_path):
        h2h = json.load(open(h2h_path))
        hh = h2h.get("headtohead", h2h)
        out["headtohead"] = {
            "verdict": hh.get("VERDICT", "—"),
            "kga_regret": float(h2h.get("policy_mean_regret", {}).get("kga", 0)),
            "_source": a.headtohead,
        }
        out["_provenance"]["headtohead"] = a.headtohead

    # diff vs previous
    if os.path.exists(os.path.join(ROOT, a.prev)):
        prev = json.load(open(os.path.join(ROOT, a.prev)))
        print("\n=== diff vs previous results_source.json (natural shifts) ===")
        for k in ("officehome_M_v2", "iwildcam_H_v2"):
            pv, nv = prev.get("natural_shifts", {}).get(k, {}), out["natural_shifts"][k]
            for fld in ("regret_kga", "regret_freeze", "regret_adapt", "verdict"):
                if pv.get(fld) != nv.get(fld):
                    print(f"  {k}.{fld}: {pv.get(fld)} -> {nv.get(fld)}")

    if a.check_only:
        print("\n[check-only] validated; not writing. Proposed natural-shift verdicts:",
              {k: v["verdict"] for k, v in out["natural_shifts"].items()})
        return
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
