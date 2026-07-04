#!/usr/bin/env python3
"""Post-patch integrity proof.

Re-walk every *.json. For each node where stored beats_both is True but the gated
re-derivation is False, classify it as GATED (the node carries a sibling gate field
fa_ok/verdict_win/candidate_win/verdict_win -> the file's real verdict already
enforces FA<=alpha) or UNGATED (no such gate -> a real integrity bug).

PASS criterion: zero UNGATED stored-True-but-False nodes remain. Also verifies no
node has candidate_win/verdict_win == True while FA>alpha (gate consistency)."""
import json, os, math
REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"
ALPHA = 0.10; EPS = 1e-9
SKIP = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache",
        ".pytest_cache", ".torch_cache", ".tex_build_universal", "audits"}
GATES = ("fa_ok", "verdict_win", "candidate_win")

def is_num(x): return isinstance(x, (int, float)) and not isinstance(x, bool)
def regret(d):
    rv = d.get("regret_vs_oracle", {})
    r = rv.get("K_Bound", rv.get("router", d.get("regret_kga", d.get("regret_router"))))
    fr = rv.get("always_freeze", d.get("regret_freeze"))
    ad = rv.get("best_fixed_always_adapt", rv.get("always_adapt", d.get("regret_adapt")))
    return r, fr, ad
def fa(d):
    for k in ("false_adapt_rate_among_adapt","false_adapt_rate","false_adapt_rate_B<0"):
        if k in d: return 0.0 if d[k] is None else float(d[k])
    if "false_adapt" in d and is_num(d["false_adapt"]): return float(d["false_adapt"])
    return None

ungated_bugs=[]; gated=[]; gate_inconsistencies=[]
for dp,dn,fn in os.walk(REPO):
    dn[:]=[d for d in dn if d not in SKIP]
    for name in fn:
        if not name.endswith(".json") or name.startswith("._"): continue
        fp=os.path.join(dp,name)
        try: data=json.load(open(fp))
        except Exception: continue
        stack=[data]
        while stack:
            nd=stack.pop()
            if isinstance(nd,dict):
                # gate consistency: a True gated verdict must have FA<=alpha
                for g in ("candidate_win","verdict_win"):
                    if nd.get(g) is True:
                        f=fa(nd)
                        if f is not None and f>ALPHA+1e-12:
                            gate_inconsistencies.append((os.path.relpath(fp,REPO),g,f))
                if isinstance(nd.get("beats_both"),bool) and nd["beats_both"] is True:
                    r,fr,ad=regret(nd); f=fa(nd)
                    if is_num(r) and is_num(fr) and is_num(ad) and f is not None:
                        corrected=(r<fr-EPS) and (r<ad-EPS) and (f<=ALPHA+1e-12)
                        if not corrected:
                            has_gate=any(g in nd for g in GATES) or ("beats_both_raw" in nd)
                            rec=(os.path.relpath(fp,REPO),f)
                            (gated if has_gate else ungated_bugs).append(rec)
                stack.extend(nd.values())
            elif isinstance(nd,list): stack.extend(nd)

print(f"Remaining stored-True-but-gated-False nodes WITH a gate (ok, by design): {len(gated)}")
print(f"Remaining UNGATED stored-True-but-False nodes (integrity bugs): {len(ungated_bugs)}")
for f,faj in ungated_bugs: print("   UNGATED BUG:",f,"FA=",faj)
print(f"Gate-consistency violations (candidate_win/verdict_win True but FA>alpha): {len(gate_inconsistencies)}")
for f,g,v in gate_inconsistencies: print("   INCONSISTENT:",f,g,"FA=",v)
print("\nRESULT:", "PASS - no ungated integrity bug remains" if not ungated_bugs and not gate_inconsistencies
      else "FAIL - see above")
