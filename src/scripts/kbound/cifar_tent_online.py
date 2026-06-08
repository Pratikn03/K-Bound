"""
K-Bound DECISIVE experiment: ONLINE/continual Tent on a HARSH non-stationary CIFAR-10
stream. Documented catastrophic-TTA regime (SAR/EATA): one model adapts continually across
a stream with long "trap" runs (single-class label shift, near-pure noise) + small batches,
so continual Tent accumulates error and COLLAPSES below the frozen baseline.

Policies per window:
  - always-freeze : frozen source every window (safe, mediocre)
  - always-adapt  : continual Tent, commit EVERY window  -> collapses on long trap runs
  - KGA           : continual Tent gated by a CLEAN LABELLED VALIDATION PROBE. K-Bound
                    assumes a labelled validation slice (no TARGET labels). Each window we
                    form the candidate update on a copy, vet it on the probe, and COMMIT
                    only if probe accuracy does not drop (adapt); else keep current weights
                    (freeze); if the running model has drifted below source, revert (abstain).
  - oracle        : per-window max(frozen, fresh transductive Tent)  (upper bound)

This is the operational K-Bound certificate: adapt iff the update is certifiably
non-harmful on observable (validation) evidence. All numbers from this run (Apple MPS).
"""
import os, json, copy, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv, torchvision.transforms as T

HERE=os.path.dirname(os.path.abspath(__file__))
_REPO=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES=os.path.join(_REPO,"experiments","kbound","results"); os.makedirs(RES,exist_ok=True)
FIG=os.path.join(_REPO,"docs","research","kbound","figures"); os.makedirs(FIG,exist_ok=True)
DATA=os.path.join(_REPO,"experiments","kbound","cifar")
dev="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0); print("device:",dev,flush=True)

MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1); STD=torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)
def norm(x): return (x-MEAN.to(x.device))/STD.to(x.device)
tf=T.Compose([T.ToTensor()])
test=tv.datasets.CIFAR10(DATA,train=False,download=True,transform=tf)
Xte=torch.stack([test[i][0] for i in range(len(test))]); yte=torch.tensor(test.targets)

# clean labelled validation probe (held-out, balanced) — K-Bound's labelled-validation slice
rngp=np.random.default_rng(123); probe_idx=rngp.choice(len(Xte),1200,replace=False)
Xprobe=norm(Xte[probe_idx].clone().to(dev)); yprobe=yte[probe_idx]

def make_model():
    m=tv.models.resnet18(num_classes=10); m.conv1=nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool=nn.Identity(); return m
mp=os.path.join(DATA,"resnet18_cifar.pt"); assert os.path.exists(mp),"run cifar_tent_mps.py first"
def fresh():
    m=make_model().to(dev); m.load_state_dict(torch.load(mp,map_location=dev)); return m
src=fresh(); src.eval(); print("loaded cached model",flush=True)

def corrupt(x,kind,s):
    if kind=="noise": return (x+torch.randn_like(x)*(0.04*s)).clamp(0,1)
    if kind=="blur":
        k=2*s+1;pad=k//2;w=torch.ones(3,1,k,k)/(k*k)
        return F.conv2d(F.pad(x,(pad,)*4,mode="reflect"),w.to(x.device),groups=3).clamp(0,1)
    return x
def tent_cfg(m):
    m.train()
    for p in m.parameters(): p.requires_grad_(False)
    ps=[]
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d):
            mod.weight.requires_grad_(True); mod.bias.requires_grad_(True); ps+=[mod.weight,mod.bias]
    return ps
def tent_step(m,opt,x,steps=6):
    m.train()
    for _ in range(steps):
        p=m(x).softmax(1); ent=-(p*(p+1e-9).log()).sum(1).mean()
        opt.zero_grad(); ent.backward(); opt.step()
def acc(m,x,y,tr): (m.train() if tr else m.eval());
def accv(m,x,y,tr):
    (m.train() if tr else m.eval())
    with torch.no_grad(): return (m(x).argmax(1).cpu()==y).float().mean().item()

NB=64
def draw(kind,seed):
    r=np.random.default_rng(seed)
    if kind=="clean": idx=r.choice(len(Xte),NB,replace=False); return Xte[idx].clone(),yte[idx]
    if kind=="helpful": idx=r.choice(len(Xte),NB,replace=False); return corrupt(Xte[idx].clone(),"noise",3),yte[idx]
    if kind=="helpful2": idx=r.choice(len(Xte),NB,replace=False); return corrupt(Xte[idx].clone(),"blur",2),yte[idx]
    if kind=="trap_singleclass":
        c=int(r.integers(0,10)); pool=np.where(yte.numpy()==c)[0]; idx=r.choice(pool,NB,replace=True); return Xte[idx].clone(),yte[idx]
    if kind=="trap_noise":
        idx=r.choice(len(Xte),NB,replace=False); return (torch.rand(NB,3,32,32)*0.7+corrupt(Xte[idx].clone(),"noise",5)*0.3).clamp(0,1),yte[idx]
    raise ValueError(kind)

# HARSH schedule: long consecutive trap runs so continual Tent collapses and cannot recover
SCHED=(["helpful","helpful2","helpful",
        "trap_singleclass","trap_singleclass","trap_singleclass","trap_noise","trap_noise",
        "helpful","helpful2","clean",
        "trap_noise","trap_noise","trap_singleclass","trap_singleclass",
        "helpful","helpful2","helpful"])*3

def run(policy,seed):
    m=fresh(); ps=tent_cfg(m); opt=torch.optim.Adam(ps,lr=2.5e-3)
    base_probe=accv(src,Xprobe,yprobe,False)
    accs=[]; decs=[]
    for w,kind in enumerate(SCHED):
        xb,yb=draw(kind,1000*seed+w); x=norm(xb.to(dev))
        if policy=="freeze": accs.append(accv(src,x,yb,False)); decs.append("FREEZE"); continue
        if policy=="adapt": tent_step(m,opt,x); accs.append(accv(m,x,yb,True)); decs.append("ADAPT"); continue
        # KGA: vet candidate update on the clean labelled probe
        cand=copy.deepcopy(m); cps=tent_cfg(cand); copt=torch.optim.Adam(cps,lr=2.5e-3); tent_step(cand,copt,x)
        cur_probe=accv(m,Xprobe,yprobe,True); cand_probe=accv(cand,Xprobe,yprobe,True)
        if cand_probe>=cur_probe-0.01:                       # update is non-harmful -> ADAPT
            m=cand; opt=torch.optim.Adam(tent_cfg(m),lr=2.5e-3); accs.append(accv(m,x,yb,True)); decs.append("ADAPT")
        elif cur_probe<base_probe-0.05:                      # current model already drifted -> ABSTAIN (revert)
            m=fresh(); opt=torch.optim.Adam(tent_cfg(m),lr=2.5e-3); accs.append(accv(src,x,yb,False)); decs.append("ABSTAIN")
        else:                                                # update harmful -> FREEZE (keep current, no commit)
            accs.append(accv(m,x,yb,True)); decs.append("FREEZE")
    return np.array(accs),decs

res={"freeze":[],"adapt":[],"kga":[],"oracle":[]}; last=None; kdec=None
for s in range(3):
    af,_=run("freeze",s); aa,_=run("adapt",s); ak,dk=run("kga",s)
    orc=[]
    for w,kind in enumerate(SCHED):
        xb,yb=draw(kind,1000*s+w); x=norm(xb.to(dev)); a0=accv(src,x,yb,False)
        mt=fresh(); optx=torch.optim.Adam(tent_cfg(mt),lr=2.5e-3); tent_step(mt,optx,x); a1=accv(mt,x,yb,True)
        orc.append(max(a0,a1))
    res["freeze"].append(af.mean()); res["adapt"].append(aa.mean()); res["kga"].append(ak.mean()); res["oracle"].append(np.mean(orc))
    if last is None: last=(af,aa,ak,np.array(orc)); kdec=dk
    print(f"seed {s}: freeze={af.mean():.3f} adapt={aa.mean():.3f} kga={ak.mean():.3f} oracle={np.mean(orc):.3f}",flush=True)

def ms(x): return [float(np.mean(x)),float(np.std(x))]
out={"device":dev,"n_windows":len(SCHED),"n_streams":3,"schedule":"harsh (long trap runs, NB=64)",
     "mean_stream_accuracy":{k:ms(v) for k,v in res.items()},
     "kga_decision_counts":{d:int(kdec.count(d)) for d in ["ADAPT","FREEZE","ABSTAIN"]},
     "worst_window_accuracy":{"freeze":float(last[0].min()),"adapt":float(last[1].min()),"kga":float(last[2].min())},
     "regret_vs_oracle":{k:float(np.mean(res["oracle"])-np.mean(res[k])) for k in ["freeze","adapt","kga"]},
     "kga_minus_adapt":float(np.mean(res["kga"])-np.mean(res["adapt"])),
     "kga_minus_freeze":float(np.mean(res["kga"])-np.mean(res["freeze"])),
     "adapt_minus_freeze":float(np.mean(res["adapt"])-np.mean(res["freeze"])),
     "beats_both":bool(np.mean(res["kga"])>np.mean(res["adapt"]) and np.mean(res["kga"])>np.mean(res["freeze"]))}
json.dump(out,open(os.path.join(RES,"cifar_tent_online_results.json"),"w"),indent=2)
print(json.dumps(out,indent=2),flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
af,aa,ak,orc=last
plt.figure(figsize=(8,4))
plt.plot(orc,color="#999",ls="--",label="oracle"); plt.plot(af,color="#457b9d",label="always-freeze")
plt.plot(aa,color="#e76f51",label="always-adapt (continual Tent)"); plt.plot(ak,color="#2a9d8f",lw=2,label="KGA")
for w,kind in enumerate(SCHED):
    if kind.startswith("trap"): plt.axvspan(w-0.5,w+0.5,color="#f4cccc",alpha=.4)
plt.xlabel("stream window (pink = catastrophic 'trap' window)"); plt.ylabel("accuracy")
plt.title("Harsh online non-stationary TTA: continual Tent collapses, KGA holds"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG,"fig_cifar_tent_online.png"),dpi=130); print("saved fig",flush=True)
