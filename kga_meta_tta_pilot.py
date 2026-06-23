"""KGA-Meta-TTA pilot - CIFAR-100-C continual go/no-go. Mac-ready: CUDA->MPS->CPU."""
import argparse, copy, json, time
import numpy as np
import torch, torch.nn as nn
from robustbench.data import load_cifar100c
from robustbench.utils import load_model

CORRUPTIONS = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur',
               'motion_blur','zoom_blur','snow','frost','fog','brightness','contrast',
               'elastic_transform','pixelate','jpeg_compression']

def pick_device(o=''):
    if o: return o
    if torch.cuda.is_available(): return 'cuda'
    if getattr(torch.backends,'mps',None) is not None and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'
DEVICE = pick_device()

def entropy(x): return -(x.softmax(1)*x.log_softmax(1)).sum(1)

def collect_bn_affine(model):
    model.train(); model.requires_grad_(False); params=[]
    for m in model.modules():
        if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d,nn.GroupNorm,nn.LayerNorm)):
            m.requires_grad_(True); params+=[p for p in (m.weight,m.bias) if p is not None]
            if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
                m.track_running_stats=False; m.running_mean=None; m.running_var=None
    return params

class Tent(nn.Module):
    def __init__(self,model,lr=1e-3,steps=1):
        super().__init__(); self.model=model; self.params=collect_bn_affine(model)
        self.opt=torch.optim.Adam(self.params,lr=lr); self.steps=steps
    @torch.enable_grad()
    def adapt(self,x):
        for _ in range(self.steps):
            loss=entropy(self.model(x)).mean(); self.opt.zero_grad(); loss.backward(); self.opt.step()
    def forward(self,x):
        self.adapt(x)
        with torch.no_grad(): return self.model(x)

class EataLite(Tent):
    def __init__(self,model,lr=1e-3,steps=1,e_margin=None):
        super().__init__(model,lr,steps); self.e_margin=e_margin if e_margin is not None else 0.4*np.log(100)
    @torch.enable_grad()
    def adapt(self,x):
        for _ in range(self.steps):
            out=self.model(x); ent=entropy(out); keep=ent<self.e_margin
            if keep.any():
                loss=ent[keep].mean(); self.opt.zero_grad(); loss.backward(); self.opt.step()

def fresh_pool(model_name,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    base=load_model(model_name,dataset='cifar100',threat_model='corruptions').to(DEVICE).eval()
    return {'frozen':copy.deepcopy(base).eval(),'tent':Tent(copy.deepcopy(base).to(DEVICE)),
            'eata':EataLite(copy.deepcopy(base).to(DEVICE))}

@torch.no_grad()
def acc_of(l,y): return (l.argmax(1)==y).float().mean().item()

def kga_gate(pl,py,floor=0.0):
    best,blb='frozen',-1.0; n=len(py)
    for nm,lg in pl.items():
        a=acc_of(lg,py); eps=(a*(1-a)/max(n,1))**0.5+floor; lb=a-eps
        if lb>blb: blb,best=lb,nm
    if blb<acc_of(pl['frozen'],py) and best!='frozen': best='frozen'
    return best

def run_seed(mn,seed,corrs,ne,sev,bs,pk):
    pool=fresh_pool(mn,seed); pp={k:[] for k in ['frozen','tent','eata','kga_meta','oracle']}
    for c in corrs:
        x,y=load_cifar100c(n_examples=ne,severity=sev,data_dir='./data',shuffle=True,corruptions=[c])
        x,y=x.to(DEVICE),y.to(DEVICE); lo={}
        for nm,m in pool.items():
            o=[]
            for i in range(0,len(x),bs): o.append(m(x[i:i+bs]))
            lo[nm]=torch.cat(o,0)
        if pk>0:
            idx=torch.randperm(len(y))[:pk]; ch=kga_gate({k:v[idx] for k,v in lo.items()},y[idx])
            mask=torch.ones(len(y),dtype=torch.bool); mask[idx]=False
        else:
            mg={k:v.softmax(1).max(1).values.mean().item() for k,v in lo.items()}; ch=max(mg,key=mg.get)
            mask=torch.ones(len(y),dtype=torch.bool)
        yt=y[mask]
        for nm in ['frozen','tent','eata']: pp[nm].append(1-acc_of(lo[nm][mask],yt))
        pp['kga_meta'].append(1-acc_of(lo[ch][mask],yt))
        pp['oracle'].append(1-max(acc_of(lo[k][mask],yt) for k in ['frozen','tent','eata']))
    return {k:np.array(v) for k,v in pp.items()}

def pboot(a,b,nb=10000,seed=0):
    rng=np.random.default_rng(seed); d=b-a; N=len(d); df=np.array([d[rng.integers(0,N,N)].mean() for _ in range(nb)])
    return float((df>0).mean()),[float(np.percentile(df,2.5)),float(np.percentile(df,97.5))]

def main():
    global DEVICE
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='Hendrycks2020AugMix_WRN'); ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--n_examples',type=int,default=2000); ap.add_argument('--severity',type=int,default=5)
    ap.add_argument('--batch',type=int,default=200); ap.add_argument('--probe_k',type=int,default=64)
    ap.add_argument('--device',default='',choices=['','cpu','mps','cuda']); ap.add_argument('--out',default='kga_meta_pilot_results.json')
    a=ap.parse_args(); DEVICE=pick_device(a.device); print('device =',DEVICE,flush=True)
    t0=time.time(); cells={k:[] for k in ['frozen','tent','eata','kga_meta','oracle']}
    for s in a.seeds:
        print(f'[seed {s}] {len(CORRUPTIONS)} corruptions on {DEVICE} ...',flush=True)
        r=run_seed(a.model,s,CORRUPTIONS,a.n_examples,a.severity,a.batch,a.probe_k)
        for k in cells: cells[k].append(r[k])
        print('  mean err: '+'  '.join(f'{k}={r[k].mean():.4f}' for k in cells),flush=True)
    flat={k:np.concatenate(v) for k,v in cells.items()}; means={k:float(flat[k].mean()) for k in flat}
    sg={k:means[k] for k in ['frozen','tent','eata']}; bs=min(sg,key=sg.get); P,ci=pboot(flat['kga_meta'],flat[bs])
    verdict='CONVERTS' if (means['kga_meta']<means[bs] and ci[0]>0) else ('PARTIAL' if means['kga_meta']<=min(sg.values())+1e-9 else 'FAILS')
    out={'mean_error':means,'best_single':bs,'P_kga_beats_bestsingle':P,'ci95_gap':ci,'oracle':means['oracle'],'verdict':verdict,'wall_sec':round(time.time()-t0,1)}
    json.dump(out,open(a.out,'w'),indent=2); print('\n==== VERDICT:',verdict,'===='); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
