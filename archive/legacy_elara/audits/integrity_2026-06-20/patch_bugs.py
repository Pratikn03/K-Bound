#!/usr/bin/env python3
"""Surgically correct the stored beats_both verdicts that ignore the FA<=alpha gate.

ONLY the ungated stored-verdict nodes are touched (nodes whose `beats_both` is the
stored verdict with NO sibling gate field such as fa_ok / verdict_win / candidate_win).
Win-finder rows (candidate_win) and protocol dev_rows (verdict_win) are left intact:
their final verdict already enforces FA<=alpha by design.

For each target node the corrected verdict is RE-DERIVED from that node's own numbers:
  corrected = (router<freeze) AND (router<best_fixed_adapt) AND (fa_rate<=ALPHA)
We preserve beats_both_raw, add beats_both_corrected, flip beats_both to the gated
value, and add beats_both_correction_note. Backups already exist under backups/.
Idempotent: if beats_both_raw already present, the node is skipped.
"""
import json, os
REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"
ALPHA = 0.10
EPS = 1e-9
NOTE = ("Integrity pass 2026-06-20: original stored `beats_both` was regret-only and "
        "did NOT enforce the pre-registered false-adapt budget FA<=alpha (alpha=0.1). "
        "Re-derived from this node's own regret_vs_oracle + false-adapt rate: "
        "beats_both := (router_regret<freeze_regret) AND (router_regret<best_fixed_adapt_regret) "
        "AND (FA_rate<=alpha). See audits/integrity_2026-06-20/.")

# (relative file, list of dict keys to the node holding beats_both)
TARGETS = [
    ("experiments/kbound/results/iwildcam_full_val/result_f08e751c.json",
     ["routing_b_multicandidate"]),
    ("experiments/kbound/results/iwildcam_full_idval/result_489da28f.json",
     ["routing_a_single_candidate", "sar_online", "kga"]),
    ("experiments/kbound/results/iwildcam_full_idval/result_489da28f.json",
     ["routing_a_single_candidate", "tent_online", "kga"]),
    ("experiments/kbound/results/imagenetc_1pct/decisive_tta_results.json",
     ["benchmarks", "imagenetc", "methods", "eata", "metrics"]),
    ("experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json",
     ["routing_a_single_candidate", "sar_online", "kga"]),
]

def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def get_regret(d):
    rv = d.get("regret_vs_oracle", {})
    router = rv.get("K_Bound", rv.get("router"))
    freeze = rv.get("always_freeze")
    adapt = rv.get("best_fixed_always_adapt", rv.get("always_adapt"))
    return router, freeze, adapt

def get_fa(d):
    for k in ("false_adapt_rate_among_adapt", "false_adapt_rate", "false_adapt_rate_B<0"):
        if k in d:
            return (0.0 if d[k] is None else float(d[k])), k
    if "false_adapt" in d and is_num(d["false_adapt"]):
        return float(d["false_adapt"]), "false_adapt"
    return None, None

def navigate(root, keys):
    node = root
    for k in keys:
        node = node[k]
    return node

def main():
    changes = []
    # group targets by file so we load/write each file once
    byfile = {}
    for rel, keys in TARGETS:
        byfile.setdefault(rel, []).append(keys)
    for rel, keylists in byfile.items():
        fp = os.path.join(REPO, rel)
        with open(fp) as f:
            data = json.load(f)
        touched = False
        for keys in keylists:
            node = navigate(data, keys)
            assert isinstance(node, dict), f"{rel}:{keys} not a dict"
            assert "beats_both" in node, f"{rel}:{keys} has no beats_both"
            for gate in ("fa_ok", "verdict_win", "candidate_win"):
                assert gate not in node, f"REFUSE: {rel}:{keys} has gate '{gate}' (already gated by design)"
            if "beats_both_raw" in node:
                print(f"SKIP (already patched): {rel} :: {'/'.join(keys)}")
                continue
            stored = node["beats_both"]
            router, freeze, adapt = get_regret(node)
            fa_rate, fa_src = get_fa(node)
            assert is_num(router) and is_num(freeze) and is_num(adapt), f"{rel}:{keys} missing regret"
            assert fa_rate is not None, f"{rel}:{keys} missing FA"
            regret_beats = (router < freeze - EPS) and (router < adapt - EPS)
            gate_ok = fa_rate <= ALPHA + 1e-12
            corrected = bool(regret_beats and gate_ok)
            node["beats_both_raw"] = stored
            node["beats_both_corrected"] = corrected
            node["beats_both"] = corrected
            node["beats_both_correction_note"] = NOTE
            touched = True
            changes.append({"file": rel, "node": "/".join(keys), "stored_raw": stored,
                            "router": router, "freeze": freeze, "best_fixed_adapt": adapt,
                            "fa_rate": fa_rate, "fa_src": fa_src,
                            "regret_beats": regret_beats, "gate_fa_le_alpha": gate_ok,
                            "corrected": corrected})
            print(f"PATCH {rel}\n   node={'/'.join(keys)}  raw={stored} -> corrected={corrected}"
                  f"  (router={router:.6f} freeze={freeze:.6f} adapt={adapt:.6f} "
                  f"FA={fa_rate} via {fa_src}; regret_beats={regret_beats} gate={gate_ok})")
        if touched:
            with open(fp, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  wrote {rel}")
    with open(os.path.join(REPO, "audits/integrity_2026-06-20/patch_changes.json"), "w") as f:
        json.dump(changes, f, indent=2)
    print(f"\nTotal nodes patched: {len(changes)}")

if __name__ == "__main__":
    main()
