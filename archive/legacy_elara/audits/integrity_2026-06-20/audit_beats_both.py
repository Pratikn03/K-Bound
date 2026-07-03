#!/usr/bin/env python3
"""Integrity audit: recompute beats_both with the pre-registered FA<=alpha gate.

For EVERY *.json under the repo we locate any node carrying a boolean key whose
name starts with 'beats_both', extract the underlying regret triple and the
false-adapt info from the SAME node, and re-derive the correct verdict:

  beats_both_correct = (router_regret < freeze_regret)
                   AND (router_regret < best_fixed_adapt_regret)
                   AND (false_adapt_rate <= ALPHA)

We do NOT mutate anything here -- this is read-only. It prints every node and
flags STORED-vs-CORRECT mismatches, with special attention to stored True that
should be False (the integrity bug). Writes a machine-readable audit JSON.
"""
import json, os, sys, math

REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"
ALPHA = 0.10
EPS = 1e-9
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache",
             ".ruff_cache", ".pytest_cache", ".torch_cache", ".tex_build_universal"}

def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))

def get_regret(d):
    """Return (router, freeze, adapt, src) or None. Looks in d['regret_vs_oracle']
    or top-level regret_* keys."""
    rv = d.get("regret_vs_oracle")
    if isinstance(rv, dict):
        router = rv.get("K_Bound", rv.get("router"))
        freeze = rv.get("always_freeze")
        adapt = rv.get("best_fixed_always_adapt", rv.get("always_adapt"))
        if is_num(router) and is_num(freeze) and is_num(adapt):
            return float(router), float(freeze), float(adapt), "regret_vs_oracle"
    # flat keys
    router = d.get("regret_kga", d.get("regret_router"))
    freeze = d.get("regret_freeze")
    adapt = d.get("regret_adapt", d.get("regret_best_fixed_adapt"))
    if is_num(router) and is_num(freeze) and is_num(adapt):
        return float(router), float(freeze), float(adapt), "flat_regret_keys"
    return None

def get_fa(d):
    """Return (fa_rate, src) or (None, reason). Tries rate fields first, then count/adapt."""
    for k in ("false_adapt_rate_among_adapt", "false_adapt_rate", "false_adapt_rate_B<0", "fa_rate"):
        if k in d:
            v = d[k]
            if v is None:
                return 0.0, f"{k}=null(no adapts->0)"
            if is_num(v):
                return float(v), k
    # 'false_adapt' may be a rate (<=1 typical) or a count
    if "false_adapt" in d and is_num(d["false_adapt"]):
        fa = float(d["false_adapt"])
        ac = d.get("adapt_count", d.get("n_adapt"))
        if is_num(ac) and ac > 0 and fa > 1:
            return fa / float(ac), "false_adapt_count/adapt_count"
        return fa, "false_adapt(as_rate?)"
    # counts
    fac = d.get("false_adapt_count", d.get("route_b_false_adapt"))
    ac = d.get("adapt_count", d.get("n_adapt"))
    if is_num(fac):
        if is_num(ac) and ac > 0:
            return float(fac) / float(ac), "false_adapt_count/adapt_count"
        if fac == 0:
            return 0.0, "false_adapt_count=0"
        # decision_counts ADAPT
        dc = d.get("decision_counts")
        if isinstance(dc, dict) and is_num(dc.get("ADAPT")) and dc["ADAPT"] > 0:
            return float(fac) / float(dc["ADAPT"]), "false_adapt_count/decision_counts.ADAPT"
        return None, f"fa_count={fac} but no adapt denom"
    return None, "no_fa_field"

def walk(node, path, filerec):
    if isinstance(node, dict):
        for k, v in node.items():
            if k.startswith("beats_both") and isinstance(v, bool):
                reg = get_regret(node)
                fa_rate, fa_src = get_fa(node)
                entry = {"path": path + "/" + k, "stored": v,
                         "regret": None, "fa_rate": fa_rate, "fa_src": fa_src}
                regret_beats = None
                gate = None
                corrected = None
                if reg:
                    router, freeze, adapt, rsrc = reg
                    entry["regret"] = {"router": router, "freeze": freeze, "adapt": adapt, "src": rsrc}
                    regret_beats = (router < freeze - EPS) and (router < adapt - EPS)
                if fa_rate is not None:
                    gate = (fa_rate <= ALPHA + 1e-12)
                if regret_beats is not None and gate is not None:
                    corrected = bool(regret_beats and gate)
                entry["regret_beats"] = regret_beats
                entry["gate_fa_le_alpha"] = gate
                entry["corrected"] = corrected
                entry["mismatch"] = (corrected is not None and corrected != v)
                entry["bug_true_to_false"] = (v is True and corrected is False)
                # capture raw fa-related keys present in this node for manual verification
                entry["raw_fa_keys"] = {k2: node[k2] for k2 in node
                    if ("false_adapt" in k2 or k2 in ("adapt_count", "n_adapt")) and is_num(node.get(k2))}
                entry["depth"] = path.count("/")
                filerec.append(entry)
            walk(v, path + "/" + str(k), filerec)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + f"[{i}]", filerec)

def main():
    results = {}
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".json"):
                continue
            if fn.startswith("._"):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp) as f:
                    data = json.load(f)
            except Exception:
                continue
            rec = []
            walk(data, "", rec)
            if rec:
                results[os.path.relpath(fp, REPO)] = rec
    # summary
    n_nodes = sum(len(v) for v in results.values())
    bugs = [(f, e) for f, v in results.items() for e in v if e["bug_true_to_false"]]
    mismatches = [(f, e) for f, v in results.items() for e in v if e["mismatch"]]
    stored_true = [(f, e) for f, v in results.items() for e in v if e["stored"] is True]
    print(f"Files with beats_both nodes: {len(results)}")
    print(f"Total beats_both* nodes: {n_nodes}")
    print(f"Stored TRUE nodes: {len(stored_true)}")
    print(f"Mismatches (corrected != stored, where computable): {len(mismatches)}")
    print(f"INTEGRITY BUGS (stored True -> corrected False): {len(bugs)}")
    print("=" * 100)

    # Per-file rollup: how many stored-True / bugs per file, and whether a TOP-LEVEL
    # (depth<=1) beats_both key exists (that's the file's headline verdict).
    print("\n### PER-FILE ROLLUP (sorted by #bugs desc) ###")
    rows = []
    for f, v in results.items():
        st = sum(1 for e in v if e["stored"] is True)
        bg = sum(1 for e in v if e["bug_true_to_false"])
        top = [e for e in v if e["depth"] <= 1 and e["path"].count("/") <= 1]
        top_keys = sorted({e["path"] for e in top})
        rows.append((bg, st, len(v), f, top_keys))
    for bg, st, tot, f, top_keys in sorted(rows, key=lambda x: (-x[0], -x[1], x[3])):
        print(f"  bugs={bg:3d} storedTrue={st:3d} nodes={tot:3d}  {f}")
        if top_keys:
            print(f"        TOP-LEVEL verdict keys: {top_keys}")

    # Focused dump: TOP-LEVEL stored-True nodes (these are the files' headline verdicts)
    print("\n" + "=" * 100)
    print("### TOP-LEVEL (depth<=1) STORED-TRUE NODES — headline verdicts ###")
    tl = [(f, e) for f, e in stored_true if e["depth"] <= 1]
    for f, e in sorted(tl, key=lambda x: x[0]):
        r = e["regret"]
        rstr = (f"router={r['router']:.5f} freeze={r['freeze']:.5f} adapt={r['adapt']:.5f}"
                if r else "regret=NOT_FOUND_in_node")
        fastr = (f"{e['fa_rate']:.4f}" if e["fa_rate"] is not None else "NA") + f"({e['fa_src']})"
        print(f"\nFILE: {f}\n  node: {e['path']}  raw_fa={e['raw_fa_keys']}\n  {rstr}\n  fa_rate={fastr}"
              f"\n  regret_beats={e['regret_beats']} gate={e['gate_fa_le_alpha']} "
              f"=> corrected={e['corrected']}  [stored={e['stored']}]"
              + ("  <<< BUG (flip True->False)" if e["bug_true_to_false"] else
                 ("  <<< MISMATCH" if e["mismatch"] else "  (ok)")))

    out = os.path.join(REPO, "audits/integrity_2026-06-20/audit_beats_both_raw.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    bugout = os.path.join(REPO, "audits/integrity_2026-06-20/audit_bugs_true_to_false.json")
    with open(bugout, "w") as f:
        json.dump([{"file": ff, **ee} for ff, ee in bugs], f, indent=2)
    print("\nWrote", out, "and", bugout)

if __name__ == "__main__":
    main()
