"""Build the ELARA-U score archive: per-(task, detector) val/test scores + labels.

The backbone of the canonical pipeline (research_lock/ELARA_U_PROTOCOL_v1.yaml).
Runs the detector zoo once and persists per-sample scores so the router,
abstention, calibration, and leave-family-out evaluation can iterate cheaply
without re-running detectors. Writes one npz per task to
experiments/elara_u/score_archive/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reuse the validated task loader + detector zoo from the seed harness.
from scripts.elara_u.gate_u_seed_eval import RNG, detector_zoo, load_tasks

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "experiments/elara_u/score_archive"


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def build_task(X, y):
    """Return (Sval[n,d], yval, Stest[m,d], ytest, det_names, val_auc) for one task."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=RNG, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=RNG, stratify=ytmp)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    names, sval, stest, vauc = [], [], [], []
    for name, ctor in detector_zoo().items():
        try:
            m = ctor().fit(Xtr)
            ref = m.decision_function(Xtr)
            v = _zsig(m.decision_function(Xva), ref)
            t = _zsig(m.decision_function(Xte), ref)
        except Exception as e:
            print(f"    [{name}] failed: {e}")
            v = np.full(len(yva), 0.5)
            t = np.full(len(yte), 0.5)
        names.append(name)
        sval.append(v)
        stest.append(t)
        vauc.append(float(roc_auc_score(yva, v)) if len(np.unique(yva)) > 1 else 0.5)
    return (np.column_stack(sval), yva, np.column_stack(stest), yte,
            np.array(names), np.array(vauc))


def main() -> int:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    manifest = {}
    for name, dom, X, y in tasks:
        t0 = time.time()
        Sval, yval, Stest, ytest, det_names, vauc = build_task(X, y)
        np.savez(ARCHIVE / f"{name}.npz", Sval=Sval, yval=yval, Stest=Stest, ytest=ytest,
                 det_names=det_names, val_auc=vauc, domain=dom)
        manifest[name] = {"domain": dom, "n_val": int(len(yval)), "n_test": int(len(ytest)),
                          "anomaly_rate": round(float(y.mean()), 4),
                          "detectors": det_names.tolist()}
        print(f"[{name:20}] {dom:8} n_val={len(yval)} n_test={len(ytest)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    (ROOT / "experiments/elara_u/score_archive_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\narchive: {len(manifest)} tasks -> {ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
