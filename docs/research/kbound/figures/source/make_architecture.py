"""Generate Figure 1: K-Bound system architecture (clean, vector, ELARA-free).
Writes fig_architecture.pdf + .png to ../ and ../final/. Pure matplotlib (no extra deps).
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(HERE)                 # docs/research/kbound/figures
FINAL = os.path.join(FIG, "final"); os.makedirs(FINAL, exist_ok=True)

PUR="#5b2a86"; PUR_F="#f4f0fb"; GREY_E="#c9ccd6"; GREY_F="#f6f7f9"
GRN_E="#2a9d8f"; GRN_F="#e7f5f1"; BLU_E="#2f5d8a"; BLU_F="#e9eef6"; ABS_E="#8a8f9a"; ABS_F="#eef0f2"
INK="#222633"; MUT="#6b7280"

fig, ax = plt.subplots(figsize=(13.0, 8.7)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")

def box(x,y,w,h,fc,ec,lw=1.4,r=0.025):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.0,rounding_size={r*100}",
                                fc=fc,ec=ec,lw=lw,mutation_aspect=h/w if w else 1))
def txt(x,y,s,size=10.5,w="normal",c=INK,st="normal",ha="center",va="center"):
    ax.text(x,y,s,fontsize=size,fontweight=w,color=c,style=st,ha=ha,va=va)
def arrow(x1,y1,x2,y2,c=MUT,lw=1.6):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,color=c,lw=lw,
                                 shrinkA=0,shrinkB=0))

# ---- title ----
txt(50,97,"K-Bound — System Architecture",20,"bold")
txt(50,93.2,"Knowability-guided decision-making for label-free test-time adaptation",11.5,c=MUT)
txt(50,90.2,"Should the system adapt, freeze, or abstain?",11,c=PUR,st="italic")

# ---- inputs ----
txt(3,86,"INPUTS",10,"bold",c=MUT,ha="left")
ins=[("Labelled source + validation","used for calibration / training"),
     ("Unlabelled target batch","deployment data under shift"),
     (r"Frozen model $f_0$","safe baseline"),
     (r"Adapt candidate $f_a$","Tent / EATA / SAR / other TTA")]
ix=2.0; iw=23.0; gap=(96-4*iw)/3
for i,(t,s) in enumerate(ins):
    x=ix+i*(iw+gap); box(x,78.5,iw,6.6,"#ffffff",GREY_E,1.2); txt(x+iw/2,82.1,t,10,"bold"); txt(x+iw/2,79.7,s,8.4,c=MUT)
    arrow(x+iw/2,78.3,x+iw/2,73.4)

# ---- KGA container ----
box(2,50.5,96,22.0,PUR_F,PUR,2.0,r=0.018)
txt(50,70.2,"KGA — Knowability-Guided Adaptation",13,"bold",c=PUR)
steps=[("(1) Label-free evidence  $Z=\\phi(\\cdot)$","entropy · confidence · drift ·\ndisagreement · update statistics"),
       ("(2) Benefit estimate + certificate","$\\hat{\\Delta}(Z)\\;\\pm\\;\\epsilon(Z)$\nconformal / Bernstein / e-value"),
       ("(3) Decision rule","adapt if $\\hat{\\Delta}-\\epsilon>0$;  freeze if $\\hat{\\Delta}+\\epsilon<0$;\nabstain otherwise")]
sw=29.5; sx=4.5; sgap=(91-3*sw)/2
for i,(t,s) in enumerate(steps):
    x=sx+i*(sw+sgap); box(x,53.0,sw,13.5,"#ffffff",GREY_E,1.2)
    txt(x+sw/2,63.3,t,10.2,"bold"); txt(x+sw/2,57.7,s,8.6,c=MUT)
    if i<2: arrow(x+sw+0.6,59.8,x+sw+sgap-0.6,59.8,c=PUR)

# ---- KGA -> outcomes ----
outs=[("ADAPT","knowably helpful",r"deploy $f_a$",GRN_E,GRN_F),
      ("FREEZE","knowably harmful",r"keep $f_0$",BLU_E,BLU_F),
      ("ABSTAIN","unknowable","safe default / review",ABS_E,ABS_F)]
ow=29.5; ox=3.0; ogap=(94-3*ow)/2
cx=[ox+ow/2, ox+(ow+ogap)+ow/2, ox+2*(ow+ogap)+ow/2]
for c in cx: arrow(50,50.3,c,46.6,c=MUT)
for i,(t,s1,s2,ec,fc) in enumerate(outs):
    x=ox+i*(ow+ogap); box(x,37.5,ow,9.0,fc,ec,1.8)
    txt(x+ow/2,43.7,t,15,"bold",c=ec); txt(x+ow/2,41.0,s1,9.2,c=INK)
    ax.plot([x+ow/2-7,x+ow/2+7],[40.0,40.0],color=ec,lw=0.8,alpha=0.5)
    txt(x+ow/2,38.7,s2,8.8,c=MUT)

# ---- theory foundation ----
txt(3,33.5,"THEORY FOUNDATION",10,"bold",c=MUT,ha="left")
box(2,24.5,96,7.5,GREY_F,GREY_E,1.2)
th=["Impossibility /\nnon-identifiability","Finite-sample\ncertificate",
    "Provable\npositive regime","Disagreement-region\nsign characterization"]
for i,t in enumerate(th):
    x=2+ (i+0.5)*(96/4); txt(x,28.2,t,9.4,"bold",c=PUR)
    if i: ax.plot([2+i*(96/4)]*2,[25.3,31.2],color=GREY_E,lw=1)

# ---- validated on ----
txt(3,20.5,"VALIDATED ON",10,"bold",c=MUT,ha="left")
box(2,11.5,96,7.0,GREY_F,GREY_E,1.2)
val=["anomaly routing\n(123 tasks)","regression\ncovariate shift","CIFAR-10-C\nstress grid","online\nnon-stationary TTA"]
for i,t in enumerate(val):
    x=2+(i+0.5)*(96/4); txt(x,15.0,t,9.2,c=INK)
    if i: ax.plot([2+i*(96/4)]*2,[12.3,17.7],color=GREY_E,lw=1)

plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
for out in (os.path.join(FIG,"fig_architecture.pdf"), os.path.join(FIG,"fig_architecture.png"),
            os.path.join(FINAL,"fig_architecture.pdf"), os.path.join(FINAL,"fig_architecture.png")):
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
print("wrote fig_architecture.pdf/.png to figures/ and figures/final/")
