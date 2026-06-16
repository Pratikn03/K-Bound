# D33 controlled multimodal corruption test: STRONG (significant beats-both) — with honest scope

*Pre-registered: `research_lock/CONTROLLED_MULTIMODAL_PROTOCOL_D33_v1.yaml` (sealed before results).
MNIST two-view (A=left half, B=right half); B corrupted by Gaussian noise across 13 severities x 10
batches = 130 conditions; α=0.10. Report as-is.*

## Result
| policy | mean accuracy (130 conditions) |
|---|---|
| always-fuse (adapt) | 0.5832 |
| always-single A (freeze) | 0.8536 |
| **KGA-routed** | **0.8568** |
| oracle | 0.8573 |
- KGA − always-fuse = **+0.2736, P(KGA>fuse)=1.000**
- KGA − always-single = **+0.0032, P(KGA>single)=1.000**
- false-adapt = **0 / 9 commits = 0.000** (≤ α). Decisions: ADAPT 9, FREEZE 119, ABSTAIN 2.
- **VERDICT: STRONG (significant beats-both).**

## What it shows (the useful part)
When the named condition genuinely holds — one modality detectably degrades, so fusion flips
helpful↔harmful — **KGA significantly beats both trivial policies with zero false-adapt.** It adapts
on the (clean-B) conditions where fusion helps and freezes on the (corrupted-B) conditions where
fusion hurts. So **the method is not the bottleneck**: the certificate routes correctly and dominates
both baselines (never worse than the better trivial policy in any condition; strictly better where
each fails). This is the multimodal echo of the CIFAR-10-C stress-grid win, on real image data with a
real detectable modality failure.

## What it does NOT show (the honest limits)
- **Controlled construction, not a natural benchmark.** I injected the corruption to make the named
  condition hold. So this does NOT demonstrate that any *natural* benchmark exhibits the condition —
  that remains the open problem (the on-disk 3D-ADAM hunt tied; external acquisition is pending).
- **It does not raise the real-world rank to ~80.** ~80 requires the natural-shift win; this is a
  mechanism confirmation, not that.
- **The magnitude over always-single is small (+0.003) and grid-dependent.** Most of KGA's value is the
  safety half (avoiding fusion collapse); the gain over the safe single baseline is concentrated in the
  few clean conditions. The grid-independent, robust claim is *dominance* (KGA ≥ both everywhere,
  strictly better where each fails, 0 false-adapt, P=1.0), not the size of the mean gap.

## Bottom line
This isolates the open problem cleanly: **the KGA mechanism works when the named condition holds and is
detectable (proven here, significantly).** What's unresolved is purely empirical — finding a *natural*
multimodal benchmark that exhibits the condition with a fusion gain large enough to matter. That's the
external-acquisition project scoped in EXTERNAL_BENCHMARK_SCOPING.md.
