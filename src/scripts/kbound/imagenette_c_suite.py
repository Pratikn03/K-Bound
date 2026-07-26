"""
K-Bound at ImageNet scale + architecture breadth (Exp3-style). ImageNet-C is only on
Zenodo, which is undownloadable from this host (320 B/s). We instead use IMAGENETTE
(10-class ImageNet subset, fast.ai S3, 224px real photos) + the OFFICIAL ImageNet-C
corruption functions (imagecorruptions), with PRETRAINED ResNet-50 and ViT-B/16.
This tests whether the helpful/harmful behaviour persists at realistic resolution and
across a CNN and a transformer. Honestly labeled: Imagenette, not full 1000-class ImageNet-C.
Online TTA; KGA = clean-probe-gated continual Tent. All numbers from this run.
"""
import os, json, math, tarfile, urllib.request, glob, hashlib
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torchvision as tv, torchvision.transforms as T
from imagecorruptions import corrupt

HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES=os.path.join(_REPO,"experiments","kbound","results"); os.makedirs(RES,exist_ok=True)
FIG=os.path.join(_REPO,"docs","research","kbound","figures"); os.makedirs(FIG,exist_ok=True)
DATA=os.path.join(_REPO,"experiments","kbound","imagenette")
dev="mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
print("device:",dev,flush=True)

# ---- get Imagenette (fast S3) ----
root=os.path.join(DATA,"imagenette2-320")
if not os.path.isdir(root):
    os.makedirs(DATA,exist_ok=True); tgz=os.path.join(DATA,"imagenette2-320.tgz")
    if not os.path.exists(tgz):
        print("downloading imagenette2-320 (~325MB)...",flush=True)
        urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",tgz)
    print("extracting...",flush=True)
    with tarfile.open(tgz) as t: t.extractall(DATA)
# wnid -> ImageNet-1k index
WNID2IDX={"n01440764":0,"n02102040":217,"n02979186":482,"n03000684":491,"n03028079":497,
 "n03394916":566,"n03417042":569,"n03425413":571,"n03445777":574,"n03888257":701}
valdir=os.path.join(root,"val")
items=[]
for wnid,idx in WNID2IDX.items():
    for f in glob.glob(os.path.join(valdir,wnid,"*.JPEG")): items.append((f,idx))
print("val images:",len(items),flush=True)
rng=np.random.default_rng(0); rng.shuffle(items)

IM_MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); IM_STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
def to_uint8(path):
    im=Image.open(path).convert("RGB").resize((224,224),Image.BILINEAR); return np.array(im,dtype=np.uint8)
def normb(x): return (x-IM_MEAN.to(x.device))/IM_STD.to(x.device)

# preload a working pool + clean probe (uncorrupted)
POOL=items[300:]; PROBE=items[:300]
probe_imgs=np.stack([to_uint8(p) for p,_ in PROBE]).astype(np.float32)/255.
Xprobe=normb(torch.tensor(probe_imgs).permute(0,3,1,2).contiguous().to(dev)); yprobe=torch.tensor([i for _,i in PROBE])

def load_model(arch):
    if arch=="resnet50": m=tv.models.resnet50(weights="IMAGENET1K_V2")
    else: m=tv.models.vit_b_16(weights="IMAGENET1K_V1")
    return m.to(dev).eval()

def tent_params(m):
    ps=[]
    for mod in m.modules():
        if isinstance(mod,(nn.BatchNorm2d,nn.LayerNorm)):
            for p in mod.parameters(): p.requires_grad_(True); ps.append(p)
    return ps
def configure(m):
    for p in m.parameters(): p.requires_grad_(False)
    return tent_params(m)
def ent(p): return -(p*(p+1e-9).log()).sum(1)
def step(m,opt,x):
    m.train(); p=m(x).softmax(1); loss=ent(p).mean(); opt.zero_grad(); loss.backward(); opt.step()
def acc(m,x,y,tr):
    (m.train() if tr else m.eval())
    with torch.no_grad(): return (m(x).argmax(1).cpu()==y).float().mean().item()
def snapshot(ps): return [p.detach().clone() for p in ps]
def restore(ps,snap):
    with torch.no_grad():
        for p,s in zip(ps,snap): p.copy_(s)

CORRUPTIONS=["gaussian_noise","contrast","brightness","defocus_blur"]
SEV=[1,3,5]; N_PER=150; BATCH=75; LR=1e-3; TOL=0.01

def stable_seed(*parts, mod=2**31 - 1):
    """Process-stable seed (fix-queue item 30 / F2-8). See cifar10c_suite.stable_seed:
    Python salts hash() on str/tuple per process unless PYTHONHASHSEED is set, and
    nothing in this repo sets it, so `hash((corr,sev))` drew a different subsample
    every run."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.blake2b(key, digest_size=8).hexdigest(), 16) % mod


def load_cell(corr,sev):
    # fix-queue item 30: was np.random.default_rng(hash((corr,sev))%9991)
    sub=POOL[:N_PER]; r=np.random.default_rng(stable_seed("imagenette_c_suite", corr, sev))
    imgs=np.stack([corrupt(to_uint8(p),corruption_name=corr,severity=sev) for p,_ in sub]).astype(np.float32)/255.
    x=torch.tensor(imgs).permute(0,3,1,2).contiguous(); y=torch.tensor([i for _,i in sub]); return x,y

def run(arch,method,x,y,src):
    m=load_model(arch); ps=configure(m); opt=torch.optim.Adam(ps,lr=LR); accs=[]
    xb=normb(x.to(dev)).contiguous()
    for i in range(0,len(x),BATCH):
        bx=xb[i:i+BATCH].contiguous(); by=y[i:i+BATCH]
        if method=="frozen": accs.append(acc(src,bx,by,False)); continue
        if method=="tent": step(m,opt,bx); accs.append(acc(m,bx,by,True)); continue
        if method=="kga":
            # vet on clean probe: trial-update, compare probe acc post vs pre, revert if it drops
            snap=snapshot(ps); step(m,opt,bx)
            post=acc(m,Xprobe,yprobe,True); restore(ps,snap); pre=acc(m,Xprobe,yprobe,True)
            if post>=pre-TOL:
                step(m,opt,bx); accs.append(acc(m,bx,by,True))   # commit update
            else:
                accs.append(acc(m,bx,by,True))                    # freeze (snapshot kept)
            continue
    return float(np.mean(accs))

results={}
for arch in ["resnet50","vit_b_16"]:
    src=load_model(arch)
    cells={}
    for corr in CORRUPTIONS:
        for sev in SEV:
            x,y=load_cell(corr,sev)
            rec={mth:run(arch,mth,x,y,src) for mth in ["frozen","tent","kga"]}
            # oracle: max(frozen, fresh transductive tent over the cell)
            mo=load_model(arch); po=configure(mo); oo=torch.optim.Adam(po,lr=LR); xb=normb(x.to(dev)).contiguous()
            for i in range(0,len(x),BATCH): step(mo,oo,xb[i:i+BATCH].contiguous())
            rec["oracle"]=max(rec["frozen"],acc(mo,xb,y,True))
            cells[f"{corr}|{sev}"]=rec
            print(f"{arch:10} {corr:16} s{sev} frozen={rec['frozen']:.3f} tent={rec['tent']:.3f} kga={rec['kga']:.3f} orc={rec['oracle']:.3f}",flush=True)
    def avg(m): return float(np.mean([c[m] for c in cells.values()]))
    def fa(m):
        a=h=0
        for c in cells.values():
            if abs(c[m]-c["frozen"])>1e-6:
                a+=1; h+= int(c[m]<c["frozen"]-1e-6)
        return h/max(a,1)
    def reg(m): return float(np.mean([c["oracle"]-c[m] for c in cells.values()]))
    results[arch]={"avg_accuracy":{m:avg(m) for m in ["frozen","tent","kga","oracle"]},
                   "false_adapt_rate":{m:fa(m) for m in ["tent","kga"]},
                   "regret_to_oracle":{m:reg(m) for m in ["frozen","tent","kga"]},
                   "cells":cells}
json.dump(results,open(os.path.join(RES,"imagenette_c_results.json"),"w"),indent=2)
print("\n=== SUMMARY ==="); print(json.dumps({a:{k:results[a][k] for k in ['avg_accuracy','false_adapt_rate','regret_to_oracle']} for a in results},indent=2),flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,arch in zip(axes,["resnet50","vit_b_16"]):
    cells=results[arch]["cells"]; s5=[(c,cells[f"{c}|5"]) for c in CORRUPTIONS]; xs=range(len(s5))
    ax.plot(xs,[r["frozen"] for _,r in s5],"o-",color="#457b9d",label="frozen")
    ax.plot(xs,[r["tent"] for _,r in s5],"s-",color="#e76f51",label="Tent")
    ax.plot(xs,[r["kga"] for _,r in s5],"^-",color="#2a9d8f",lw=2,label="KGA")
    ax.set_xticks(list(xs)); ax.set_xticklabels([c.replace("_","\n") for c,_ in s5],rotation=45,ha="right",fontsize=7)
    ax.set_title(arch); ax.set_ylim(0,1); ax.set_ylabel("accuracy")
axes[0].legend(fontsize=8); plt.suptitle("Imagenette + ImageNet-C corruptions (sev5): ResNet-50 vs ViT-B/16")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_imagenette_c.png"),dpi=130); print("saved fig",flush=True)
