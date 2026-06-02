# ELARA Dataset Use Matrix

**Date:** 2026-06-02

This file records what every local dataset is used for, what it is not used
for, and how much scientific weight it can carry. The purpose is to prevent
development datasets, diagnostic datasets, and official transfer datasets from
being mixed together.

## Rules

1. **Primary evidence is real-dataset evidence.** The current paper/system
   claim should be built from real captured datasets first: MVTec 3D-AD,
   3D-ADAM, Real-IAD, Real-IAD D3, MulSen-AD, Real3D-AD, UNSW-NB15, and
   BIDMC-healthcare. Synthetic or proxy artifacts can support mechanism
   checks, smoke tests, and failure analysis, but they do not carry the main
   transfer or production-readiness claim.
2. **Official transfer evidence requires a fresh or prelocked holdout.** Once a
   dataset is opened and used for method choices, it becomes development
   evidence unless a still-unopened split/category was pre-registered.
3. **Derived-view datasets cannot carry independent multimodal claims.** If one
   "modality" is computed from the other, the result is diagnostic only.
4. **Label-aligned composites are mechanism tests, not natural multimodal
   transfer.** They are useful for stress testing the fusion layer, but they do
   not represent co-observed multimodal incidents.
5. **A failed dataset is still useful.** Failed transfer and negative CIs
   define the operating boundary and are kept as audit evidence.

## Local Raw Data Inventory

Approximate sizes are from `du -sh data/raw/*` on 2026-06-02. Download caches
are listed separately because they are not evidence-bearing datasets.

| Dataset / directory | Approx. local size | Modalities / views | Current use in this work | Claim ceiling |
|---|---:|---|---|---|
| `data/raw/mvtec3d` / MVTec 3D-AD | 26 GB | RGB + depth/XYZ | Primary naturally paired vision benchmark; M1/Gate D and T5 strong-baseline work; controlled degradation replication | Bounded in-domain and stress evidence; not external transfer; not one-class state-of-the-art |
| `data/raw/3d_adam_anomalib` / 3D-ADAM | 6.5 GB | RGB + depth | Sealed M2 external transfer and v3 stress-regime transfer; also D13 development evidence | Opened/spent as official M2; clean Gate E failed/tied, stress-regime evidence remains useful |
| `data/raw/realiad` / Real-IAD | 4.0 GB | RGB industrial images with Real-IAD metadata | D13 prelocked natural positive-transfer attempt | Official D13 fail: beats CW, fails SAR; now opened for development only |
| `data/raw/realiad_d3` / Real-IAD D3 | 259 GB | RGB + pseudo-3D + 3D point cloud | D16 natural-degradation/headroom audit on 19-category holdout | Supportive natural-degradation evidence; all-test CW positive; primary stress CI crosses zero, so no Gate S pass |
| `data/raw/mulsen_ad` / MulSen-AD | 19 GB | RGB + infrared + point cloud | D13 opened-development replication; promising because modalities are genuinely complementary | Development only unless a prelocked unopened split/category is defined before evaluation |
| `data/raw/eyecandies` / Eyecandies | 25 GB | Synthetic RGB + depth + normals | Family D clean-transfer study and failure record | Valid transfer dataset, but failed; now development/negative evidence, not final positive transfer |
| `data/raw/real3d` / Real3D-AD | 10 GB | 3D point cloud, with local exploratory paired inputs | Earlier mechanism/exploratory benchmark cells | Tier-B/exploratory only; not v3 headline transfer |
| `data/raw/mvtec_loco` / MVTec LOCO-AD | 12 GB | RGB plus derived edge proxy | Family A secondary benchmark | Diagnostic only; derived-view proxy cannot prove independent-modality generalization |
| `data/raw/visa` / VisA | 4.3 GB | RGB plus derived edge/noise-floor proxy | Family A/C secondary and noise-floor checks | Diagnostic only; not independent multimodal evidence |
| `data/raw/cyber` / UNSW-NB15 | 606 MB | Flow / connection / context event views | Non-vision structured event-view benchmark; held-out attack checks | Shows fusion machinery outside vision; small effect; not co-observed sensor multimodality |
| `data/raw/healthcare` / BIDMC-healthcare inputs | 354 MB | Clinical time-series / structured views | M3 development, patient-stratified and deployment-audit gap checks | Development evidence; not regulatory or confirmatory deployment validation |
| `data/raw/fraud` | 144 MB | Tabular fraud records | Component domain for ELARA-Bench-LA label-aligned fusion | Mechanism/stress only; not natural multimodal transfer |
| `data/raw/behavior` | 1.3 MB | Tabular behavioral records | Component domain for ELARA-Bench-LA label-aligned fusion | Mechanism/stress only |
| `data/raw/nlp` | 107 MB | Text / fake-news records | Component domain for ELARA-Bench-LA label-aligned fusion | Mechanism/stress only |
| `data/raw/vision` | 256 KB | Small image scaffold | Utility/smoke/scaffolding | No research claim |
| `data/raw/_downloads_mulsen` | 8.4 GB | Download staging/cache | Acquisition cache | Not evidence |
| `data/raw/_downloads_realiad` | 1.4 MB | Download staging/cache | Acquisition cache | Not evidence |

## Pending or Scaffolded Datasets

These are mentioned as future candidates or scaffolds, but they are not current
official evidence.

| Candidate | Intended use | Current status |
|---|---|---|
| CICIDS-2017 + authentication logs | Fourth paired benchmark candidate for cyber/event transfer | Scaffolded only; not acquired/integrated as official evidence |
| MIMIC-IV + clinical notes | Healthcare multimodal paired candidate | Credentialed access and governance required; not official evidence |
| MVTec AD-2D + depth pseudo-channel | Controlled derived/pseudo-depth candidate | Scaffolded only; would be diagnostic, not independent RGB-D transfer |

## Current Evidence Ordering

1. **Highest claim weight:** MVTec 3D-AD in-domain supervised-paired evidence
   and 3D-ADAM stress-regime transfer.
2. **Supportive but not gate-passing:** Real-IAD D3 D16 natural degradation.
3. **Development/negative transfer evidence:** Real-IAD, MulSen-AD,
   Eyecandies, Real3D-AD.
4. **Diagnostic only:** VisA, MVTec LOCO-AD, UNSW-NB15, healthcare, and
   ELARA-Bench-LA components.

Primary paper/system evidence should cite the real datasets in items 1-3
before any synthetic/proxy material. Eyecandies remains visible as a failed
synthetic transfer record, not as a source of positive primary evidence.

## Practical Use Going Forward

- Use **MulSen-AD** and **Real-IAD D3** to design better natural-degradation and
  reliability-routing methods, but do not call tuned results on those same
  opened data official.
- Use **a fresh unopened RGB+X dataset or a prelocked unopened split/category**
  for the next official transfer attempt.
- Keep **VisA/LOCO/ELARA-Bench-LA** in the paper only as diagnostic or
  mechanism evidence.
- Keep **Eyecandies and Real-IAD failures** visible; they are important negative
  evidence, not wasted work.
