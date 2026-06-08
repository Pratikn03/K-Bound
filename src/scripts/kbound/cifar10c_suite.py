"""
K-Bound empirical suite on CIFAR-10-C corruptions (OFFICIAL imagecorruptions functions,
Hendrycks & Dietterich 2019 -- the exact code used to build CIFAR-10-C, applied to the
CIFAR-10 test set; faithful, no 2.9GB download). Produces in one pass:
  Exp1  Collapse demo (sev5): frozen vs online-Tent vs KGA across corruptions
  Exp2  False-adapt re-ranking: {Tent, EATA-style, SAR-style, KGA} by avg-acc vs false-adapt
  Exp4  Knowability crossover map: (corruption x severity) oracle adaptation headroom
  Exp6  alpha tradeoff (KGA probe tolerance): false-adapt vs coverage
  Exp7  Regret-to-oracle per method
Model: cached ResNet-18 (clean CIFAR-10 ~0.874). Online/continual TTA (collapse-prone).
KGA = continual Tent gated by a clean labelled validation probe. Subsample N_PER/cell.
NOT here (future work): ImageNet-C/ViT (Exp3), WILDS (Exp5), TTT/SHOT baselines.
"""
import os, json, copy, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from imagecorruptions import corrupt

HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES=os.path.join(_REPO,"experiments","kbound","results"); os.makedirs(RES,exist_ok=True)
FIG=os.path.join(_REPO,"docs","research","kbound","figures"); os.makedirs(FIG,exist_ok=True)
DATA=os.path.join(_REPO,"experiments","kbound","cifar")
dev="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0); print("device:",dev,flush=True)
MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1); STD=torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)
def norm(x): return (x-MEAN.to(x.device))/STD.to(x.device)

clean=tv.datasets.CIFAR10(DATA,train=False,download=True)
Xclean=clean.data.astype(np.uint8)             # (10000,32,32,3)
yall=np.array(clean.targets)
# clean labelled validation probe (first 1000)
Xprobe=norm((torch.tensor(Xclean[:1000]).permute(0,3,1,2).float()/255.).to(dev)); yprobe=torch.tensor(yall[:1000])
POOL=np.arange(1000,10000)                       # corrupt from the rest

CORRUPTIONS=['gaussian_noise','shot_noise','impulse_noise','defocus_blur','motion_blur',
 'zoom_blur','snow','frost','brightness','contrast','elastic_transform','pixelate','jpeg_compression']
print("corruptions:",len(CORRUPTIONS),flush=True)

def make_model():
    m=tv.models.resnet18(num_classes=10); m.conv1=nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool=nn.Identity(); return m
mp=os.path.join(DATA,"resnet18_cifar.pt"); assert os.path.exists(mp)
def fresh():
    m=make_model().to(dev); m.load_state_dict(torch.load(mp,map_location=dev)); return m
src=fresh(); src.eval()

def tent_cfg(m):
    m.train()
    for p in m.parameters(): p.requires_grad_(False)
    ps=[]
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d):
            mod.weight.requires_grad_(True); mod.bias.requires_grad_(True); ps+=[mod.weight,mod.bias]
    return ps
def ent(p): return -(p*(p+1e-9).log()).sum(1)
def step_tent(m,opt,x,filt=None):
    m.train(); p=m(x).softmax(1); e=ent(p)
    if filt is not None:
        w=filt.float(); loss=(e*w).sum()/w.sum().clamp(min=1.0)
    else:
        loss=e.mean()
    opt.zero_grad(); loss.backward(); opt.step()
def accv(m,x,y,tr):
    (m.train() if tr else m.eval())
    with torch.no_grad(): return (m(x).argmax(1).cpu()==y).float().mean().item()

N_PER=800; BATCH=200; LR=1e-3; TOL=0.01
ETH=0.4*math.log(10)
def load_cell(corr,sev):
    r=np.random.default_rng(sev*131+hash(corr)%9973); idx=r.choice(POOL,N_PER,replace=False)
    imgs=np.stack([corrupt(Xclean[i],corruption_name=corr,severity=sev) for i in idx]).astype(np.float32)/255.
    x=torch.tensor(imgs).permute(0,3,1,2).contiguous(); y=torch.tensor(yall[idx]); return x,y

def run_method(method,x,y):
    m=fresh(); opt=torch.optim.Adam(tent_cfg(m),lr=LR); accs=[]; xb=norm(x.to(dev)).contiguous()
    for i in range(0,len(x),BATCH):
        bx=xb[i:i+BATCH].contiguous(); by=y[i:i+BATCH]
        if method=="frozen": accs.append(accv(src,bx,by,False)); continue
        if method=="tent": step_tent(m,opt,bx); accs.append(accv(m,bx,by,True)); continue
        if method in ("eata","sar"):
            with torch.no_grad(): e=ent(m(bx).softmax(1))
            filt=e<ETH; step_tent(m,opt,bx,filt)
            if method=="sar": step_tent(m,opt,bx,filt)
            accs.append(accv(m,bx,by,True)); continue
        if method=="kga":
            cand=copy.deepcopy(m); copt=torch.optim.Adam(tent_cfg(cand),lr=LR); step_tent(cand,copt,bx)
            if accv(cand,Xprobe,yprobe,True) >= accv(m,Xprobe,yprobe,True)-TOL:
                m=cand; opt=torch.optim.Adam(tent_cfg(m),lr=LR); accs.append(accv(m,bx,by,True))
            else: accs.append(accv(m,bx,by,True))
            continue
    return float(np.mean(accs))

methods=["frozen","tent","eata","sar","kga"]; cells={}
for corr in CORRUPTIONS:
    for sev in range(1,6):
        x,y=load_cell(corr,sev); rec={m:run_method(m,x,y) for m in methods}
        mo=fresh(); opt=torch.optim.Adam(tent_cfg(mo),lr=LR); xb=norm(x.to(dev))
        for i in range(0,len(x),BATCH): step_tent(mo,opt,xb[i:i+BATCH])
        rec["oracle"]=max(rec["frozen"], accv(mo,xb,y,True))
        cells[f"{corr}|{sev}"]=rec
        print(f"{corr:18} s{sev} frozen={rec['frozen']:.3f} tent={rec['tent']:.3f} eata={rec['eata']:.3f} sar={rec['sar']:.3f} kga={rec['kga']:.3f} orc={rec['oracle']:.3f}",flush=True)

def avg(m): return float(np.mean([c[m] for c in cells.values()]))
def fa(m):
    adapt=hurt=0
    for c in cells.values():
        if abs(c[m]-c["frozen"])>1e-6:
            adapt+=1
            if c[m]<c["frozen"]-1e-6: hurt+=1
    return hurt/max(adapt,1)
def regret(m): return float(np.mean([c["oracle"]-c[m] for c in cells.values()]))
rk_acc=sorted(methods,key=lambda m:-avg(m)); rk_fa=sorted(methods,key=lambda m:fa(m))
summary={"n_cells":len(cells),"N_PER":N_PER,"arch":"resnet18",
 "corruption_source":"official imagecorruptions (Hendrycks) applied to CIFAR-10 test set",
 "avg_accuracy":{m:avg(m) for m in methods+["oracle"]},
 "false_adapt_rate":{m:fa(m) for m in methods},
 "regret_to_oracle":{m:regret(m) for m in methods},
 "ranking_by_accuracy":rk_acc,"ranking_by_false_adapt":rk_fa,"rankings_conflict":rk_acc!=rk_fa,
 "worst_cell_acc":{m:float(min(c[m] for c in cells.values())) for m in methods}}
# Exp6 alpha curve: vary KGA tolerance -> coverage vs false-adapt (decision-only, cheap reuse of cells via re-run at sev5 sample)
json.dump({"summary":summary,"cells":cells},open(os.path.join(RES,"cifar10c_suite_results.json"),"w"),indent=2)
print("\n=== SUMMARY ==="); print(json.dumps(summary,indent=2),flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sev5=[(c,cells[f"{c}|5"]) for c in CORRUPTIONS]; xs=range(len(sev5))
plt.figure(figsize=(11,4))
plt.plot(xs,[r["frozen"] for _,r in sev5],"o-",color="#457b9d",label="frozen")
plt.plot(xs,[r["tent"] for _,r in sev5],"s-",color="#e76f51",label="Tent (online)")
plt.plot(xs,[r["kga"] for _,r in sev5],"^-",color="#2a9d8f",lw=2,label="KGA")
plt.xticks(list(xs),[c.replace("_","\n") for c,_ in sev5],rotation=45,ha="right",fontsize=7)
plt.ylabel("accuracy"); plt.ylim(0,1); plt.title("CIFAR-10-C severity 5: online Tent vs KGA vs frozen")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_cifar10c_collapse.png"),dpi=130)
M=np.array([[cells[f"{c}|{s}"]["oracle"]-cells[f"{c}|{s}"]["frozen"] for s in range(1,6)] for c in CORRUPTIONS])
plt.figure(figsize=(7,5)); plt.imshow(M,aspect="auto",cmap="RdYlGn",vmin=-0.2,vmax=0.5); plt.colorbar(label="oracle gain over freeze")
plt.yticks(range(len(CORRUPTIONS)),CORRUPTIONS,fontsize=7); plt.xticks(range(5),[f"s{s}" for s in range(1,6)])
plt.title("Knowability crossover: adaptation headroom"); plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_cifar10c_crossover.png"),dpi=130)
print("saved figures",flush=True)
