# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). KB_REPO_ROOT is discovered from this
# --- file's own location; override with $KBOUND_REPO_ROOT.
import os as _kb_os
from pathlib import Path as _KbPath


def _kb_repo_root() -> str:
    override = _kb_os.environ.get("KBOUND_REPO_ROOT", "").strip()
    if override:
        return str(_KbPath(override).expanduser().resolve())
    here = _KbPath(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
    raise RuntimeError(f"repository root not found above {here}; set KBOUND_REPO_ROOT")


KB_REPO_ROOT = _kb_repo_root()

# --- external (git-excluded) data volume: ONE documented variable, no default.
def _kb_external_root() -> str:
    value = _kb_os.environ.get("KBOUND_EXTERNAL_ROOT", "").strip()
    if not value:
        raise RuntimeError(
            "KBOUND_EXTERNAL_ROOT is not set. This script needs data that is deliberately "
            "not in the git release (raw datasets, checkpoints, caches). Point "
            "KBOUND_EXTERNAL_ROOT at the volume holding them; the expected layout is "
            "documented in docs/research/kbound/kbound_repro/paths.py (EXTERNAL_LAYOUT) "
            "and acquisition is in DATA.md. There is no default on purpose: this used to "
            "be one author's external SSD, and defaulting to $HOME would write gigabytes "
            "somewhere you did not choose."
        )
    return str(_KbPath(value).expanduser().resolve())


KB_EXTERNAL_ROOT = _kb_external_root()

import json, sys
import numpy as np
WILDS = KB_REPO_ROOT + "/experiments/kbound/wilds"
THEORY = KB_REPO_ROOT + "/experiments/kbound/theory_validation"
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

RX = json.load(open(KB_EXTERNAL_ROOT + "/kbound_rxrx1_results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json"))
IR = json.load(open(KB_EXTERNAL_ROOT + "/kbound_inr_results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json"))
out = {"rxrx1": agg(RX, "RxRx1"), "imagenetr": agg(IR, "ImageNet-R")}
json.dump(out, open(KB_EXTERNAL_ROOT + "/kbound_rxrx1_results/_cmp_numbers.json", "w"), indent=2)
print(json.dumps(out, indent=2))
