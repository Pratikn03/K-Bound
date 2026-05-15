# Phase A draft prose — to insert into PAPER_DRAFT_v1.tex after experiments finish

This file holds the interpretation paragraphs that should be added to the paper
once the multi-category MVTec 3D-AD experiment completes. The numbers below come
from the existing label-aligned (`rga_*`) tau-sweep and component-ablation
artifacts and should not change unless those experiments are re-run.

---

## §IX.D τ-sweep interpretation (insert after Table~\ref{tab:tau-sweep})

The threshold sweep resolves the dormancy observed in §IX.A and §IX.B. The
reliability gate is essentially inactive on clean data at the configured
$\tau=0.66$, with only 3.2\% of test samples triggering the reliability-aware
path; raising $\tau$ to 0.80 raises the adaptation rate to 67.2\% but
introduces a small clean ROC-AUC penalty (-0.0001, 95\% CI $[-0.0003,
-0.0001]$). Under all-domain \texttt{max\_attack} and \texttt{zero\_attack},
the gate fires for 100\% of samples at any $\tau \geq 0.66$, and the resulting
ROC-AUC gain is stable from $\tau=0.66$ through $\tau=0.90$ ($+0.0304$ for
\texttt{max\_attack}, $+0.0640$ for \texttt{zero\_attack}, both with 95\%
confidence intervals that exclude zero). This supports two design claims: the
conservative gate threshold $\tau=0.66$ preserves clean ranking, and the gain
under coherent stress is robust to the exact threshold rather than an artifact
of hyperparameter tuning.

## §IX.D component-ablation interpretation (insert after Table~\ref{tab:component-ablation})

The component ablation isolates which reliability signal carries the
stress-test gain. Removing the KS-drift component (\texttt{no\_ks}) eliminates
the gain entirely: the ROC-AUC delta collapses to $0.0000$ across all attacks
because the reliability score never falls below $\tau$ and the gate never
fires. Removing the gate (\texttt{no\_gate}) reproduces the full result, since
the attack conditions already trigger the gate for every sample — the gate
matters for clean preservation, not for stressed performance. Removing
sharpness (\texttt{no\_sharpness}) is approximately neutral. Removing ECE
(\texttt{no\_ece}) actually \emph{improves} the adversarial gain by
$+0.0124$ ROC-AUC on \texttt{max\_attack} (95\% CI $[0.0067, 0.0790]$) and
$+0.0051$ on \texttt{zero\_attack}. This is a meaningful negative result for
the calibration component: the validation-ECE term, while motivated by
calibration theory~\cite{guo2017calibration}, contributes noise rather than
signal under coherent score collapse. A leaner reliability variant with only
KS-drift (and optionally sharpness) is therefore a defensible candidate for
the next configuration; the calibration story is preserved as a clean-data
property, not a stress-test mechanism.

## §VIII MVTec 3D-AD headline (rewrite once 8-category run finishes)

The current §VII.A caption reads "Paired MVTec 3D-AD bagel-subset smoke
benchmark" and the surrounding prose says "deliberately conservative". After
the eight-category extraction completes and the experiment is re-run, replace
that wording with:

> Table~\ref{tab:mvtec-clean-results} reports the multi-category paired
> MVTec 3D-AD benchmark. Each composite is naturally co-observed RGB and depth
> evidence from the same object scan; no label alignment is required. Across
> the configured object categories, [fill in actual numbers from new
> mvtec3d_clean_results.tex] ...

The earlier wording about "not yet competitive with tree and MLP fusion on
this small subset" should also be revised to reflect the new evidence.
