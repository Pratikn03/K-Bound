import json, sys
import numpy as np
WILDS = "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/wilds"
THEORY = "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/theory_validation"
sys.path[:0] = [WILDS, THEORY]
import analysis as an
import tta_methods as tm

def agg(d, name):
    recs = d.get("records", []); conds = d.get("conditions", [])
    B = np.array([r["B"] for r in recs], float)
    cands = sorted(set(r["candidate"] for r in recs))
    per_cand = {c: float(np.mean([r["aa"] for r in recs if r["candidate"] == c])) for c in cands}
    c_a0 = np.array([c["a0"] for c in conds], float)
    oracle = np.array([c["oracle"] for c in conds], float)
    realized = np.array([c["realized"] for c in conds], float)
    dec = [c["route"].get("decision") for c in conds]
    det = an.detectability_analysis(recs, tm.EVIDENCE_NAMES) if len(recs) >= 4 else {}
    cfg = d.get("config", {})
    ks = d.get("kbound_summary", {})
    return dict(name=name, schema=d.get("schema"),
                cfg={k: cfg.get(k) for k in ("n_eval","batch_regimes","seeds","compositions","aggressiveness","episodic_steps","episodic_batch","tau_star","max_classes")},
                n_cells=len(conds), n_records=len(recs),
                classification=ks.get("classification"),
                mean_B=float(B.mean()), base_harmful=float(np.mean(B < -0.02)),
                helpful_rate=float(np.mean(B > 0.02)), marginal_rate=float(np.mean(np.abs(B) <= 0.02)),
                always_freeze=float(c_a0.mean()), oracle=float(oracle.mean()),
                router_realized=float(realized.mean()),
                best_fixed_adapt=float(max(per_cand.values())), worst_fixed_adapt=float(min(per_cand.values())),
                worst_cand=min(per_cand, key=per_cand.get), best_cand=max(per_cand, key=per_cand.get),
                per_cand={k: round(v, 4) for k, v in per_cand.items()},
                abstain_rate=float(np.mean([x == "ABSTAIN" for x in dec])),
                detect_verdict=det.get("detectability_verdict"),
                best_harm_auc=det.get("best_single_feature_harm_AUC"),
                top_feats=sorted(((k, v["harm_AUC"]) for k, v in det.get("per_feature", {}).items() if v.get("harm_AUC") is not None), key=lambda kv: -kv[1])[:3])

RX = json.load(open("/Users/pratik_n/kbound_rxrx1_results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json"))
IR = json.load(open("/Users/pratik_n/kbound_inr_results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json"))
out = {"rxrx1": agg(RX, "RxRx1"), "imagenetr": agg(IR, "ImageNet-R")}
json.dump(out, open("/Users/pratik_n/kbound_rxrx1_results/_cmp_numbers.json", "w"), indent=2)
print(json.dumps(out, indent=2))
