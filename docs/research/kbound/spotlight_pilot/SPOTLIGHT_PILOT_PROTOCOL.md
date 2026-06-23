# Pre-registration — KGA-Meta-TTA spotlight go/no-go (Protocol S1)

**Sealed:** 2026-06-20, before any result. **Status:** pre-registered, not yet run.
**Purpose:** decide *cheaply and honestly* whether the K-Bound knowability gate, used as the
engine of a meta-TTA method, can BEAT the best single TTA method — the only thing that turns
this from an ~80 safety-certificate accept into a top-tier "beats-SOTA" contribution.

This is the **go/no-go before spending GPU/cluster compute** on ImageNet-C/ViT.

---

## Hypothesis (stated before running)

A KGA-gated meta-policy that routes per batch over a pool of TTA adapters achieves **lower mean
error on continual CIFAR-100-C** than the best single adapter in the pool, **without tuning the
gate on test corruptions**, across ≥3 seeds.

If true on CIFAR-100-C → scale to ImageNet-C/ViT (the spotlight target).
If false → honest null; K-Bound stays a safety-certificate paper (~80), not a beats-SOTA paper.

## Method under test — KGA-Meta-TTA

- **Pool** `C = {frozen, Tent, EATA, SAR}` (pilot uses frozen + Tent + EATA-lite; official
  EATA/SAR/CoTTA/RoTTA plug in for the publishable version).
- **Adapters adapt continually** across the corruption stream (no reset between corruptions).
- **Gate (multi-candidate KGA, target-label-light operating point):** at each corruption block,
  a small labeled probe of `k` target examples (drawn once at block start) gives each candidate
  an accuracy estimate `â_c` with a conformal radius `ε`. Deploy the candidate with the highest
  certified lower bound `â_c − ε`; if none exceeds `frozen`, **freeze** (deploy frozen). This is
  exactly K-Bound's adapt/freeze/abstain routing — never worse than frozen up to ε, best
  certified adapter otherwise.
- **The gate's selection rule + k + ε are fixed on a DEV corruption set** (CIFAR-10-C or a
  held-out subset of CIFAR-100-C corruptions) **before** touching the reported corruptions.

## Benchmark & baselines

- **Data:** CIFAR-100-C, severity 5, the standard 15 corruptions, **continual** protocol
  (corruptions streamed in fixed order, model state carried over). RobustBench loaders + a
  RobustBench-pretrained WRN-28-10 → numbers are literature-comparable.
- **Baselines (each run identically):** source/frozen, Tent, EATA, SAR (+ CoTTA/RoTTA from
  official repos for the full version), an **always-best-single** oracle-of-policies, and the
  **per-condition oracle** (upper bound).

## Primary endpoint (fixed now)

Mean top-1 **error** over the 15 corruptions, averaged over seeds {0,1,2}.
**WIN** ⇔ `mean_err(KGA-Meta) < mean_err(best single adapter)` AND the paired-bootstrap 95% CI of
that gap (resampling corruption×seed cells) **excludes 0**.

## Pre-registered outcomes (all publishable)

1. **CONVERTS** — KGA-Meta significantly beats the best single adapter → the bet is alive;
   pre-register and run the ImageNet-C / ViT-B scale-up (the spotlight result).
2. **PARTIAL** — beats some adapters / ties the best → a principled *safety+selection* result,
   not a SOTA beat; strengthens the ~80 paper, no spotlight claim.
3. **FAILS** — no improvement over best single → honest null; the gate is a guard, not a
   method; K-Bound stays the safety-certificate paper. Reported as-is.

## Forbidden (anti-cherry-pick contract)

- No tuning of the gate threshold / k / ε on the reported corruptions (DEV-only).
- No seed selection; report all 3.
- No reporting a favorable subset of corruptions; the endpoint is the 15-corruption mean.
- The `beats-both`/`WIN` flag is the pre-registered CI test above, not a point estimate.

## Compute

Pilot: 1 free GPU (Colab T4 / Kaggle P100), ~30–90 min for 3 seeds × 15 corruptions × ~4 policies.
Scale-up (only if CONVERTS): ImageNet-C/ViT needs a real GPU box — NAIRR/ACCESS or NVIDIA Academic
grant, or a host lab's cluster. Not runnable on a MacBook.
