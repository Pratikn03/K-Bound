import json, glob, math, numpy as np, os
R=os.path.expanduser("~/Documents/AutoML_Flagship_V8/experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed")
A=0.10; rng=np.random.default_rng(20260720)
def cexact(r): r=np.sort(np.abs(r)); n=len(r); k=min(n,int(math.ceil((n+1)*(1-A)))); return float(r[k-1])
def load(f): d=json.load(open(f)); return d['records'] if 'records' in d else d
def pairs(cand):
    fs=sorted(glob.glob(f"{R}/per_condition_imagenetc_{cand}_seed*.json")); B=[];DEC=[]
    for f in fs:
        r=load(f); b=np.array([x['B'] for x in r]); bh=np.array([x.get('b_hat') for x in r])
        eps=cexact(np.abs(bh-b))
        DEC+=['ADAPT' if x-eps>0 else ('FREEZE' if x+eps<0 else 'ABSTAIN') for x in bh]; B+=list(b)
    B=np.array(B); act=np.where(np.array(DEC)=='ADAPT','ADAPT','FREEZE'); orc=np.where(B>0,'ADAPT','FREEZE')
    rk=np.abs(B)*(act!=orc); ra=np.abs(B)*('ADAPT'!=orc); rf=np.abs(B)*('FREEZE'!=orc)
    return rk,ra,rf
def ci(cand):
    rk,ra,rf=pairs(cand); n=len(rk); gf=rk-rf; ga=rk-ra  # KGA minus fixed policy (want <0)
    def boot(g):
        idx=rng.integers(0,n,(5000,n)); ms=g[idx].mean(1); return float(np.percentile(ms,2.5)),float(np.percentile(ms,97.5))
    print(f"{cand.upper()} EXACT-rank: KGA-adapt gap95={boot(ga)} ; KGA-freeze gap95={boot(gf)}")
    lo_f,hi_f=boot(gf); lo_a,hi_a=boot(ga)
    print(f"   CI beats-both (both upper<0): {hi_a<0 and hi_f<0}")
for c in ("sar","eata","tent"): ci(c)
