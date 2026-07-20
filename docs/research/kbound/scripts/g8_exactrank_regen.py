import json, glob, math, re, numpy as np, os
R=os.path.expanduser("~/Documents/AutoML_Flagship_V8/experiments/kbound/results")
ALPHA=0.10
def cexact(res):
    r=np.sort(np.abs(np.asarray(res,float))); n=len(r); k=min(n,int(math.ceil((n+1)*(1-ALPHA)))); return float(r[k-1])
def load(fp):
    d=json.load(open(fp)); return d['records'] if isinstance(d,dict) and 'records' in d else d
def decide(bh,e): return ['ADAPT' if b-e>0 else ('FREEZE' if b+e<0 else 'ABSTAIN') for b in bh]
def regret(dec,B):
    B=np.asarray(B,float); orc=np.where(B>0,'ADAPT','FREEZE'); act=np.where(np.array(dec)=='ADAPT','ADAPT','FREEZE')
    return float(np.mean(np.abs(B)*(act!=orc)))
def pick_seeds(files):
    by={}
    for f in files:
        m=re.search(r'seed(\d)',os.path.basename(f))
        if m: by.setdefault(int(m.group(1)),f)   # first hit per seed
    return [by[s] for s in sorted(by)]
def track(name, globs):
    files=[]; 
    for g in globs: files+=glob.glob(g,recursive=True)
    fs=pick_seeds(files)
    if not fs: print(f"[{name}] none"); return
    poolFAi=poolFAe=poolN=0; rk=[]; ra=[]; rf=[]
    for fp in fs:
        r=load(fp); B=np.array([x['B'] for x in r],float); bh=np.array([x.get('b_hat',x.get('Bhat')) for x in r],float)
        rho=np.abs(bh-B); ei=float(np.quantile(rho,1-ALPHA)); ee=cexact(rho)
        de=decide(bh,ee)
        poolFAe+=sum((d=='ADAPT') and (b<=0) for d,b in zip(de,B)); poolN+=len(B)
        poolFAi+=sum((d=='ADAPT') and (b<=0) for d,b in zip(decide(bh,ei),B))
        rk.append(regret(de,B)); ra.append(regret(['ADAPT']*len(B),B)); rf.append(regret(['FREEZE']*len(B),B))
    print(f"\n=== {name}: {len(fs)} seeds, n/seed={len(B)} ===")
    print(f"  POOLED FA_u  interp={poolFAi/poolN:.4f}  exact={poolFAe/poolN:.4f}  (alpha={ALPHA})  -> {'OK <=a' if poolFAe/poolN<=ALPHA else 'BREACH'}")
    print(f"  regret exact: KGA={np.mean(rk):.4f}  adapt={np.mean(ra):.4f}  freeze={np.mean(rf):.4f}")
    print(f"  BEATS-BOTH (exact-rank): {np.mean(rk)<np.mean(ra)-1e-9 and np.mean(rk)<np.mean(rf)-1e-9}")
IC=R+"/win_hunt_v5_imagenetc_ms"
track("ImageNet-C SAR", [IC+"/seed*/per_condition_imagenetc_sar_seed*.json", IC+"/pooled_5seed/per_condition_imagenetc_sar_seed*.json"])
track("ImageNet-C EATA",[IC+"/seed*/per_condition_imagenetc_eata_seed*.json", IC+"/pooled_5seed/per_condition_imagenetc_eata_seed*.json"])
track("ImageNet-C TENT",[IC+"/seed*/per_condition_imagenetc_tent_seed*.json", IC+"/pooled_5seed/per_condition_imagenetc_tent_seed*.json"])
SG=R+"/stress_grid_multiseed_v1"
track("CIFAR-10-C TENT",[SG+"/seed*/per_condition_cifar10c_tent_seed*.json"])
track("CIFAR-10-C EATA",[SG+"/seed*/per_condition_cifar10c_eata_seed*.json"])
