#!/usr/bin/env python3
"""Emit the ready-to-paste LaTeX tables for fix-queue items 2, 3 and 5.

Writes latex_item2.tex, latex_item3.tex, latex_item5.tex next to this script.
Every number is read from NUMBERS_PACK.json -- nothing is typed by hand.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open("/home/claude/kb_fixes/NUMBERS_PACK.json"))
E = {e["id"]: e for e in P["entries"]}


def f4(x):
    return f"{x:.4f}"


# ------------------------------------------------------------------ item 2
rows = []
for s in range(5):
    v = E[f"item2.imagenetc_sar.seed{s}.exact_rank"]["value"]
    r = v["regret"]
    tie = r"\;$\equiv$" if v["bit_identical_tie_with_always_freeze"] else ""
    bb = r"\checkmark" if v["beats_both"] else "---"
    rows.append(f"{s} & ${f4(r[0])}$ & ${f4(r[1])}$ & ${f4(r[2])}${tie} & "
                f"${v['fa_u']:.3f}$ & {v['ADAPT']} & {v['FREEZE']} & {v['ABSTAIN']} & {bb} \\\\")
pv = E["item2.imagenetc_sar.pooled.exact_rank.regret"]["value"]
pa = E["item2.imagenetc_sar.pooled.exact_rank.actions"]["value"]
bbn = E["item2.imagenetc_sar.seeds_beating_both"]["value"]
item2 = r"""%% fix-queue item 2 -- regenerated under the PROMOTED exact-rank rule
%% source: experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/
%%         per_condition_imagenetc_sar_seed{0..4}.json
%% eps = rho_(k), k = min(n, ceil((n+1)(1-alpha))), fitted per seed, alpha = 0.10
\begin{table}[t]\centering\small
\caption{\textbf{ImageNet-C SAR, per seed, under the exact split-conformal rank rule}
($n{=}27$ conditions per seed, $\alpha{=}0.10$). Regret is against the per-cell oracle
$\max(a_0,a_a)$. ``$\equiv$'' marks a seed on which \textsc{KGA} never adapts, so its
regret is bit-identical to always-freeze. Point estimates improve \emph{both}
fixed-policy regrets on """ + str(bbn["n_beats_both"]) + r"""/5 seeds (seeds """ + \
    ", ".join(map(str, bbn["seeds_beating_both"])) + r"""); on seeds """ + \
    ", ".join(map(str, bbn["tie_seeds"])) + r""" \textsc{KGA} abstains throughout.
The pooled win is driven by seeds """ + ", ".join(map(str, bbn["seeds_beating_both"])) + r""".}
\label{tab:imagenetc-perseed}
\begin{tabular}{lcccccccc}
\toprule
seed & \textsc{KGA} & always-adapt & always-freeze & $\mathrm{FA}_{\mathrm u}$
& \textsc{ad} & \textsc{fr} & \textsc{ab} & beats both \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
pooled & $\mathbf{""" + f4(pv[0]) + r"""}$ & $""" + f4(pv[1]) + r"""$ & $""" + f4(pv[2]) + \
    r"""$ & $""" + f"{pa['FA_u']:.3f}" + r"""$ & """ + str(pa["ADAPT"]) + r""" & """ + \
    str(pa["FREEZE"]) + r""" & """ + str(pa["ABSTAIN"]) + r""" & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
"""

# ------------------------------------------------------------------ item 3
def ci(v, star=False):
    st = r"^{\ast}" if star else ""
    return f"$[{v[0]:+.4f},\\,{v[1]:+.4f}]{st}$"


sa = E["item3.imagenetc_sar.ci.seedavg27.exact_rank"]["value"]
iid = E["item3.imagenetc_sar.ci.iid135_as_coded.exact_rank"]["value"]
csd = E["item3.imagenetc_sar.ci.cluster_by_seed.exact_rank"]["value"]
ccf = E["item3.imagenetc_sar.ci.cluster_by_corruption_family.exact_rank"]["value"]
loo = E["item3.imagenetc_sar.ci.after_item4_loo_radius"]["value"]
item3 = r"""%% fix-queue item 3 -- ImageNet-C SAR gap CIs at the correct unit of analysis
%% 20000 paired percentile-bootstrap replicates, rng seed 20260720
%% "seed-averaged, 27 conditions" is the design kbound_short.tex:797-802 describes and
%% the one _locked_analysis_script.py:54 uses for the CIFAR-10-C rows.
\begin{table}[t]\centering\small
\caption{\textbf{ImageNet-C SAR: the beats-both interval depends on the resampling unit.}
Paired percentile bootstrap, $20{,}000$ replicates, promoted exact-rank radius. The
manuscript describes a seed-averaged condition-level design; the shipped script resampled
the $135$ cell--seed rows i.i.d. At the condition level the gap to always-adapt is
\emph{not} interval-supported. $^{\ast}$ marks an interval excluding zero.}
\label{tab:imagenetc-ci-unit}
\begin{tabular}{lccc}
\toprule
resampling unit & units & \textsc{KGA} $-$ always-adapt & \textsc{KGA} $-$ always-freeze \\
\midrule
$135$ cell--seed rows, i.i.d.\ (as coded) & 135 & """ + ci(iid["adapt_gap_ci95"], True) + r""" & """ + ci(iid["freeze_gap_ci95"], True) + r""" \\
\textbf{$27$ conditions, seed-averaged} (as described) & 27 & """ + ci(sa["adapt_gap_ci95"]) + r""" & """ + ci(sa["freeze_gap_ci95"], True) + r""" \\
cluster $=$ seed & 5 & """ + ci(csd["adapt_gap_ci95"], True) + r""" & """ + ci(csd["freeze_gap_ci95"]) + r""" \\
cluster $=$ corruption family & 3 & """ + ci(ccf["adapt_gap_ci95"], True) + r""" & """ + ci(ccf["freeze_gap_ci95"], True) + r""" \\
\bottomrule
\end{tabular}
\end{table}
%% If the leave-one-out-of-pool radius of item 4 is also adopted, the freeze-side interval
%% no longer excludes zero either:
%%   27 conditions, seed-averaged : adapt """ + str([round(x, 4) for x in loo["seedavg27_adapt_gap_ci95"]]) + r""" ; freeze """ + str([round(x, 4) for x in loo["seedavg27_freeze_gap_ci95"]]) + r"""
%%   135 rows i.i.d.              : freeze """ + str([round(x, 4) for x in loo["iid135_freeze_gap_ci95"]]) + r"""
"""

# ------------------------------------------------------------------ item 5
pr = E["item5.promoted_row_accounting"]["value"]
lines = []
for r in pr:
    if r.get("n") is None or r.get("fa_u") is None:
        continue
    ad = r.get("adapt")
    fr = r.get("freeze")
    ab = r.get("abstain")
    cp = r.get("cp95_upper_fa_c")
    cps = f"${cp:.4f}$" if cp is not None else "undef."
    fa = r.get("fa_num")
    mark = r"$^{\dagger}$" if r.get("guarantee_untested_lt10_adapts") else ""
    name = (r["track"].replace("(promoted panel row)", "").replace("(5 seeds x 432)", "")
            .replace("(5 seeds x 27)", "").replace("(promoted diagnostic-fail row, n=48)", "")
            .replace("(promoted panel row, n=18)", "").replace("&", r"\&").strip())
    lines.append(f"{name}{mark} & {r['n']} & {ad if ad is not None else '---'} & "
                 f"{fr if fr is not None else '---'} & {ab if ab is not None else '---'} & "
                 f"{fa if fa is not None else '---'} & "
                 f"${r['fa_u']:.4f}$ & {cps} \\\\")
ident = E["item5.structural_identity"]["value"]["ceilings_by_n"]
item5 = r"""%% fix-queue item 5 -- action composition and a Clopper-Pearson bound for every panel row
%% counts recomputed cell-by-cell where per-cell artifacts exist; otherwise taken from the
%% promoted summary artifact as n_test x adapt_rate.  CP upper bound: scipy.stats.beta.ppf(0.95, k+1, n-k).
\begin{table}[t]\centering\small
\caption{\textbf{Action composition and the strength of the false-adapt evidence, per track.}
$\mathrm{FA}_{\mathrm u}$ is the marginal false-adapt rate the certificate bounds;
CP$_{95}$ is the one-sided Clopper--Pearson upper bound on the \emph{conditional} rate
$\mathrm{FA}_{\mathrm c}=\Pr[\Delta\le 0\mid\textsc{adapt}]$, i.e.\ how much the observed
zero actually constrains the guarantee. $^{\dagger}$ fewer than $10$ \textsc{adapt}
decisions: guarantee untested. Only CIFAR-10-C exercises the guarantee with real power.}
\label{tab:decision-accounting}
\begin{tabular}{lccccccc}
\toprule
track & $N$ & \textsc{ad} & \textsc{fr} & \textsc{ab} & false & $\mathrm{FA}_{\mathrm u}$ & CP$_{95}$ upper on $\mathrm{FA}_{\mathrm c}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize With in-sample rank calibration the miscoverage count is identically
$N-k$, so $\mathrm{FA}_{\mathrm u}\le (N-k)/N$ holds for \emph{any} data: the ceiling is
$""" + f"{ident['432']['exact_fa_u_ceiling']:.4f}" + r"""$ at $n{=}432$ and
$""" + f"{ident['27']['exact_fa_u_ceiling']:.4f}" + r"""$ at $n{=}27$, both below
$\alpha{=}0.10$. The informative statistic is therefore
$\mathrm{FA}_{\mathrm u}{=}0$ \emph{versus that ceiling}, not
``$\mathrm{FA}_{\mathrm u}\le\alpha$''.}
\end{table}
"""

for name, body in [("latex_item2.tex", item2), ("latex_item3.tex", item3),
                   ("latex_item5.tex", item5)]:
    open(os.path.join(HERE, name), "w").write(body)
    print("wrote", name)
    print(body)
    print("-" * 90)
