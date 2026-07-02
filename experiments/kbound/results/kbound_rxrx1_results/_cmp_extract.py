import json, sys
import numpy as np
WILDS = "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/wilds"
THEORY = "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/theory_validation"
sys.path[:0] = [WILDS, THEORY]
import analysis as an
import tta_methods as tm

def agg(d, name, status):
    recs = d.get("records", []); conds = d.get("conditions", [])
    B = np.array([r["B"] for r in recs], float)
    cands = sorted(set(r["candidate"] for r in recs))
    per_cand = {c: float(np.mean([r["aa"] for r in recs if r["candidate"] == c])) for c in cands}
    c_a0 = np.array([c["a0"] for c in conds], float)
    oracle = np.array([c["oracle"] for c in conds], float)
    realized = np.array([c["realized"] for c in conds], float)
    dec = [c["route"].get("decision") for c in conds]
    det = an.detectability_analysis(recs, tm.EVIDENCE_NAMES) if len(recs) >= 4 else {}
    return dict(name=name, status=status, n_cells=len(conds), n_records=len(recs),
                mean_B=float(B.mean()), base_harmful=float(np.mean(B < -0.02)),
                helpful_rate=float(np.mean(B > 0.02)), marginal_rate=float(np.mean(np.abs(B) <= 0.02)),
                always_freeze=float(c_a0.mean()), oracle=float(oracle.mean()),
                router_realized=float(realized.mean()),
                best_fixed_adapt=float(max(per_cand.values())), worst_fixed_adapt=float(min(per_cand.values())),
                worst_cand=min(per_cand, key=per_cand.get), per_cand={k: round(v, 3) for k, v in per_cand.items()},
                abstain_rate=float(np.mean([x == "ABSTAIN" for x in dec])),
                detect_verdict=det.get("detectability_verdict"),
                best_harm_auc=det.get("best_single_feature_harm_AUC"))

RX = json.load(open("/Users/pratik_n/kbound_rxrx1_results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json"))
IRp = json.load(open("/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/imagenetr_kbound_light_mps_internal/_partial.json"))
IR1 = json.load(open("/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/imagenetr_kbound_1pct_mps_internal/result_604f04ba.json"))
out = {"rxrx1": agg(RX, "RxRx1", "complete 48/48"),
       "imagenetr_light": agg(IRp, "ImageNet-R (light)", "partial 33/48"),
       "imagenetr_1pct": agg(IR1, "ImageNet-R (1pct)", "complete")}
print(json.dumps(out, indent=2))
