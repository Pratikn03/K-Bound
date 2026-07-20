import json, glob, math, numpy as np, os
R=os.path.expanduser("~/Documents/AutoML_Flagship_V8/experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed")
A=0.10
def cexact(r): r=np.sort(np.abs(r)); n=len(r); k=min(n,int(math.ceil((n+1)*(1-A)))); return float(r[k-1])
def load(f): d=json.load(open(f)); return d['records'] if 'records' in d else d
def poolB(cand, use_exact):
    fs=sorted(glob.glob(f"{R}/per_condition_imagenetc_{cand}_seed*.json"))
    allB=[]; allDec=[]
    for f in fs:
        r=load(f); B=np.array([x['B'] for x in r]); bh=np.array([x.get('b_hat') for x in r])
        rho=np.abs(bh-B); eps= cexact(rho) if use_exact else float(np.quantile(rho,1-A))
        dec=['ADAPT' if b-eps>0 else ('FREEZE' if b+eps<0 else 'ABSTAIN') for b in bh]
        allB+=list(B); allDec+=dec
    B=np.array(allB); act=np.where(np.array(allDec)=='ADAPT','ADAPT','FREEZE'); orc=np.where(B>0,'ADAPT','FREEZE')
    rk=float(np.mean(np.abs(B)*(act!=orc))); ra=float(np.mean(np.abs(B)*('ADAPT'!=orc))); rf=float(np.mean(np.abs(B)*('FREEZE'!=orc)))
    fa=float(np.mean([(d=='ADAPT') and (b<=0) for d,b in zip(allDec,B)]))
    return rk,ra,rf,fa
for cand,tgt in [("sar",0.01075),("eata",0.00026),("tent",0.01387)]:
    i=poolB(cand,False); e=poolB(cand,True)
    print(f"{cand.upper()}: panel_KGA={tgt}")
    print(f"   INTERP(Method B): KGA={i[0]:.4f} adapt={i[1]:.4f} freeze={i[2]:.4f} FA_u={i[3]:.4f}  {'REPRODUCES panel' if abs(i[0]-tgt)<0.002 else 'no'}")
    print(f"   EXACT (Method B): KGA={e[0]:.4f} adapt={e[1]:.4f} freeze={e[2]:.4f} FA_u={e[3]:.4f}  BB={e[0]<e[1]-1e-9 and e[0]<e[2]-1e-9}")
