"""Validation v2 (adversary-hardened): multi-seed, asymmetric abstain regions, full precision.
Tests: (F),(A) identities; forced-abstention >=1/2; and whether (A)'s sign robustly separates
ImageNet-C-like (harmful mass) from Camelyon-like (helpful-dominated) or is a knife-edge."""
# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). This file previously hard-coded a
# --- Cowork *session sandbox* mount, which is worse than a
# --- home directory: it is valid only inside one ephemeral container.
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

import numpy as np, json, os
beta=0.10
def scen(rng, p_help,p_harm, abst_skew):
    # abst_skew in [-1,1]: >0 => abstain region net-helpful (more Delta>0), <0 => net-harmful
    n=300000; u=rng.random(n); M=np.empty(n); g=rng.uniform(-beta,beta,n)
    p_ab=1-p_help-p_harm
    h=u<p_help; hm=(u>=p_help)&(u<p_help+p_harm); ab=u>=p_help+p_harm
    M[h]=rng.uniform(beta+0.05,0.45,h.sum()); M[hm]=rng.uniform(-0.45,-beta-0.05,hm.sum())
    # asymmetric abstain: shift M within (-beta,beta) so mean sign of Delta is skewed
    ab_n=ab.sum(); mm=rng.uniform(-beta+1e-3,beta-1e-3,ab_n)+abst_skew*0.5*beta
    M[ab]=np.clip(mm,-beta+1e-3,beta-1e-3)
    D=M+g; oracle=np.sign(D)
    R=lambda a: np.mean(np.abs(D)*(a!=oracle))
    star=np.where(M>beta,1.0,np.where(M<-beta,-1.0,-1.0))
    AA,AF,ST=R(np.ones(n)),R(-np.ones(n)),R(star)
    F=np.mean(np.abs(D)*(M>beta)); A=(np.mean(np.abs(D)*(M<-beta))
        +np.mean(np.abs(D)*((np.abs(M)<=beta)&(D<0)))-np.mean(np.abs(D)*((np.abs(M)<=beta)&(D>0))))
    return dict(AA=AA,AF=AF,ST=ST,Fd=AF-ST,Ff=F,Ad=AA-ST,Af=A)
def multi(p_help,p_harm,abst_skew,seeds=25):
    rows=[scen(np.random.default_rng(1000+s),p_help,p_harm,abst_skew) for s in range(seeds)]
    A=np.array([r['Ad'] for r in rows]); F=np.array([r['Fd'] for r in rows])
    beats_adapt=A>1e-6
    return dict(A_mean=float(A.mean()),A_std=float(A.std()),A_min=float(A.min()),A_max=float(A.max()),
        F_mean=float(F.mean()), beats_adapt_frac=float(beats_adapt.mean()),
        Fid_ok=bool(max(abs(r['Fd']-r['Ff']) for r in rows)<1e-9),
        Aid_ok=bool(max(abs(r['Ad']-r['Af']) for r in rows)<1e-9))
out={"beta":beta,
 "imagenetC_like(harmful mass, abst net-harmful)": multi(0.55,0.25,-0.3),
 "camelyon_like(helpful-dom, no harmful, abst NET-HELPFUL)": multi(0.80,0.00,+0.6),
 "camelyon_like(helpful-dom, tiny harmful, abst ~neutral)": multi(0.78,0.02,0.0),
 "forced_abstention": "structural: |M|<beta admits gamma=+/-beta with opposite sign(Delta) at matched (Z,M) => any commit wrong w.p.>=1/2"}
json.dump(out,open(KB_REPO_ROOT + "/docs/research/kbound/theory_v2/minimax_frontier/results_v2.json","w"),indent=2)
print(json.dumps(out,indent=2))
