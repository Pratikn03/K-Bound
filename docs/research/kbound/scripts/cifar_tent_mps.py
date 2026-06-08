"""
K-Bound DECISIVE experiment: catastrophic-harm test-time adaptation on CIFAR-10.
Runs on Apple Silicon GPU (MPS). Real ResNet, real Tent (entropy minimization on
BatchNorm affine params). Builds helpful / harmful / unknowable conditions via
corruptions + label shift, then tests whether KGA (adapt/freeze/abstain) beats
BOTH always-adapt (Tent) and always-freeze.

All numbers produced by this run. Self-contained (downloads CIFAR-10 ~170MB once).
"""
import os, json, copy, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv, torchvision.transforms as T
from sklearn.ensemble import GradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES  = os.path.join(_REPO,"experiments","kbound","results"); FIG = os.path.join(_REPO,"docs","research","kbound","figures")
DATA = os.path.join(_REPO,"experiments","kbound","cifar")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
print("device:", dev)

MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
STD =torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)

# ---------- data ----------
tf=T.Compose([T.ToTensor()])
train=tv.datasets.CIFAR10(DATA,train=True,download=True,transform=tf)
test =tv.datasets.CIFAR10(DATA,train=False,download=True,transform=tf)
Xtr=torch.stack([train[i][0] for i in range(len(train))]); ytr=torch.tensor(train.targets)
Xte=torch.stack([test[i][0]  for i in range(len(test))]);  yte=torch.tensor(test.targets)
print("data:",Xtr.shape,Xte.shape)

def norm(x): return (x-MEAN.to(x.device))/STD.to(x.device)

# ---------- model: ResNet-18 for 32x32 ----------
def make_model():
    m=tv.models.resnet18(num_classes=10)
    m.conv1=nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool=nn.Identity()
    return m

mpath=os.path.join(DATA,"resnet18_cifar.pt")
model=make_model().to(dev)
if os.path.exists(mpath):
    model.load_state_dict(torch.load(mpath,map_location=dev)); print("loaded cached model")
else:
    opt=torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=0.1,total_steps=8*(len(Xtr)//256+1))
    aug=T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip()])
    model.train()
    for ep in range(8):
        perm=torch.randperm(len(Xtr))
        tot=0;corr=0
        for i in range(0,len(Xtr),256):
            idx=perm[i:i+256]; xb=aug(Xtr[idx]).to(dev); yb=ytr[idx].to(dev)
            out=model(norm(xb)); loss=F.cross_entropy(out,yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            corr+=(out.argmax(1)==yb).sum().item(); tot+=len(yb)
        print(f"epoch {ep+1}/8 train_acc={corr/tot:.3f}")
    torch.save(model.state_dict(),mpath)

model.eval()
with torch.no_grad():
    cl=[];
    for i in range(0,len(Xte),512):
        cl.append(model(norm(Xte[i:i+512].to(dev))).argmax(1).cpu())
    clean_acc=(torch.cat(cl)==yte).float().mean().item()
print("clean test acc:",round(clean_acc,3))

# ---------- corruptions (on-the-fly) ----------
def corrupt(x, kind, sev):
    s=sev
    if kind=="gauss_noise": return (x+torch.randn_like(x)*(0.04*s)).clamp(0,1)
    if kind=="blur":
        k=2*s+1; pad=k//2
        w=torch.ones(3,1,k,k)/(k*k)
        return F.conv2d(F.pad(x,(pad,)*4,mode="reflect"),w.to(x.device),groups=3).clamp(0,1)
    if kind=="bright": return (x+0.12*s).clamp(0,1)
    if kind=="contrast":
        m=x.mean(dim=(2,3),keepdim=True); return ((x-m)*(1-0.18*s)+m).clamp(0,1)
    if kind=="pixelate":
        sz=max(8,32-6*s); xs=F.interpolate(x,size=sz,mode="bilinear",align_corners=False)
        return F.interpolate(xs,size=32,mode="nearest")
    return x

# ---------- Tent ----------
def tent_adapt(base, x, steps=10, lr=1e-3):
    m=copy.deepcopy(base); m.train()
    for p in m.parameters(): p.requires_grad_(False)
    ps=[]
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d):
            mod.track_running_stats=False; mod.running_mean=None; mod.running_var=None
            mod.weight.requires_grad_(True); mod.bias.requires_grad_(True); ps+=[mod.weight,mod.bias]
    opt=torch.optim.Adam(ps,lr=lr)
    for _ in range(steps):
        out=m(x); p=out.softmax(1); ent=-(p*(p+1e-9).log()).sum(1).mean()
        opt.zero_grad(); ent.backward(); opt.step()
    return m

def evidence(base,x):
    base.eval()
    with torch.no_grad():
        out=base(x); p=out.softmax(1)
        ent=(-(p*(p+1e-9).log()).sum(1)).mean().item()
        conf=p.max(1).values.mean().item()
        preddist=p.mean(0); pbal=(-(preddist*(preddist+1e-9).log()).sum()).item()/math.log(10)
    return out,p,[ent,conf,pbal]

def acc_of(m,x,y,train_mode=False):
    if train_mode: m.train()
    else: m.eval()
    with torch.no_grad(): pred=m(x).argmax(1).cpu()
    return (pred==y).float().mean().item()

# ---------- build conditions ----------
rng=np.random.default_rng(0)
conds=[]
NB=400  # images per condition
def sample_balanced(n):
    idx=rng.choice(len(Xte),n,replace=False); return idx
def sample_labelshift(n,major):
    # heavy imbalance toward class 'major'
    maj=np.where(yte.numpy()==major)[0]; oth=np.where(yte.numpy()!=major)[0]
    nm=int(n*0.85); idx=np.concatenate([rng.choice(maj,nm),rng.choice(oth,n-nm)]); rng.shuffle(idx); return idx

specs=[]
for kind in ["gauss_noise","blur","bright","contrast","pixelate"]:
    for sev in [1,3,5]:
        specs.append(("corrupt",kind,sev))
for major in range(6):  # label-shift harmful conditions
    specs.append(("labelshift",major,5))
specs.append(("clean",None,0))

rows=[]
for rep in range(2):
  for (typ,a,b) in specs:
    if typ=="labelshift": idx=sample_labelshift(NB,a); xk=corrupt(Xte[idx].clone(),"gauss_noise",b)
    elif typ=="clean": idx=sample_balanced(NB); xk=Xte[idx].clone()
    else: idx=sample_balanced(NB); xk=corrupt(Xte[idx].clone(),a,b)
    y=yte[idx]; x=norm(xk.to(dev))
    out,p,Z=evidence(model,x)
    a0=acc_of(model,x,y,train_mode=False)
    mt=tent_adapt(model,x,steps=10,lr=1e-3)
    aa=acc_of(mt,x,y,train_mode=True)
    # post-Tent predicted balance (label-free)
    mt.train()
    with torch.no_grad():
        pt=mt(x).softmax(1).mean(0); pbal_t=(-(pt*(pt+1e-9).log()).sum()).item()/math.log(10)
    Zfull=Z+[pbal_t, Z[2]-pbal_t]
    rows.append((f"{typ}:{a}:{b}", Zfull, aa-a0, a0, aa))
    print(f"{typ:10} {str(a):10} sev{b}  frozen={a0:.3f} tent={aa:.3f}  B={aa-a0:+.3f}")

# ---------- KGA trichotomy (kfold + conformal) ----------
names=["entropy","confidence","pred_balance","pred_balance_tent","balance_drop"]
X=np.array([r[1] for r in rows]); B=np.array([r[2] for r in rows])
a0=np.array([r[3] for r in rows]); aa=np.array([r[4] for r in rows]); N=len(rows)
def kfold_bhat(X,B,k=5,seed=0):
    rngp=np.random.default_rng(seed); idx=rngp.permutation(N); folds=np.array_split(idx,k); bh=np.zeros(N)
    for i in range(k):
        te=folds[i]; tr=np.setdiff1d(idx,te)
        m=GradientBoostingRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,random_state=0)
        m.fit(X[tr],B[tr]); bh[te]=m.predict(X[te])
    return bh
Bhat=kfold_bhat(X,B); eps=float(np.quantile(np.abs(Bhat-B),0.90))
dec=np.where(Bhat-eps>0,"ADAPT",np.where(Bhat+eps<0,"FREEZE","ABSTAIN")); adapt=dec=="ADAPT"
pol=np.where(adapt,aa,a0); oracle=np.maximum(a0,aa)
naive=Bhat>0; naive_pol=np.where(naive,aa,a0)  # naive: adapt if estimated>0 (no abstain)
out={
 "device":dev,"clean_acc":clean_acc,"n_conditions":N,"eps":eps,
 "base_rate_harmful_B<0":float(np.mean(B<0)),
 "mean_true_B":float(B.mean()),
 "decision_counts":{d:int((dec==d).sum()) for d in ["ADAPT","FREEZE","ABSTAIN"]},
 "adapt_precision_B>0":float(np.mean(B[adapt]>0)) if adapt.any() else None,
 "false_adapt_rate_B<0":float(np.mean(B[adapt]<0)) if adapt.any() else None,
 "mean_acc":{"always_adapt(Tent)":float(aa.mean()),"always_freeze":float(a0.mean()),
             "naive_gating":float(naive_pol.mean()),"K_Bound":float(pol.mean()),"oracle":float(oracle.mean())},
 "regret_vs_oracle":{"always_adapt":float((oracle-aa).mean()),"always_freeze":float((oracle-a0).mean()),
             "naive_gating":float((oracle-naive_pol).mean()),"K_Bound":float((oracle-pol).mean())},
 "worst_case_acc(min over conditions)":{"always_adapt":float(aa.min()),"always_freeze":float(a0.min()),"K_Bound":float(pol.min())},
}
json.dump(out,open(os.path.join(RES,"cifar_tent_results.json"),"w"),indent=2)
print("\n==== SUMMARY ===="); print(json.dumps(out["mean_acc"],indent=2)); print(json.dumps(out["regret_vs_oracle"],indent=2))
print("false_adapt_rate:",out["false_adapt_rate_B<0"],"| harmful base rate:",out["base_rate_harmful_B<0"])

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(6,4)); labs=["always-adapt\n(Tent)","always-freeze","naive\ngating","K-Bound","oracle"]
vals=[aa.mean(),a0.mean(),naive_pol.mean(),pol.mean(),oracle.mean()]
plt.bar(labs,vals,color=["#e76f51","#457b9d","#e9c46a","#2a9d8f","#999999"])
for i,v in enumerate(vals): plt.text(i,v+.005,f"{v:.3f}",ha="center",fontsize=9)
plt.ylabel("mean accuracy over conditions"); plt.title(f"CIFAR-10 TTA (M5 GPU): K-Bound vs baselines\nclean acc {clean_acc:.2f}, harmful base rate {out['base_rate_harmful_B<0']:.2f}")
plt.ylim(0,1.0); plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_cifar_tent.png"),dpi=130)
print("saved fig + json")
