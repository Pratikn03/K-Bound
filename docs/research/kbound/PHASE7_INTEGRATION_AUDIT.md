# PHASE 7 — CLAIM-BY-CLAIM MANUSCRIPT-INTEGRATION AUDIT (K-Bound short paper)

Read-only audit. Generated 2026-07-21. Manuscript audited: `docs/research/kbound/kbound_short.tex`
(1211 lines; `\input{kbound_short_appendix.tex}` at line 1207) + `kbound_short_appendix.tex` (337 lines).
Canonical priority: `SUBMISSION_LEDGER.md` (sec 3 nine-track / sec 4 gaps / sec 5 distinctions) >
`claim_ledger.json` > `paper/generated/kbound_result_manifest.json` (regenerated 2026-07-21) > raw
`experiments/kbound/results/`. Numbers read/replayed with `jq` over the raw JSON tree.

## VERDICT SUMMARY
Manuscript is broadly faithful: all six sec-5 distinction gates HOLD, every withdrawn-claim
forbidden phrase is ABSENT, and the four already-fixed items (CIFAR-10-C SAR withheld, ImageNet-C
SAR 0.0264/0.0529/0.0319 FA_u=0, PACS photo, Office-Home 0.0157) read consistently. Of the audited
quantitative rows: **MATCH = 20, MISMATCH = 3, UNVERIFIABLE = 1**.
Three defects, all NEW (not in the ledger gap list G1–G11):
1. **RxRx1 always-adapt regret printed as 0.2587 (lines 902 + 940); every source says 0.2531** — hard numeric error.
2. **iWildCam evidence tier "5-seed real-ckpt confirmed" (line 900) contradicts the ledger ("5-seed pending"), the manifest (single-run), and the paper's own line 865-866 ("single-run … future work").**
3. **Uniform-panel CIFAR-10-C adapt/freeze (line 897: 0.0080/0.1239, 0.1313) disagree with the primary table, head-to-head macros, manifest, `uniform_verdicts.json`, and ledger-sec3 (all 0.0079/0.1241, 0.1314).** Cosmetic (KGA + beats-both unaffected) but internally inconsistent.

---

## (a) CLAIM-BY-CLAIM TABLE
Triple = KGA / always-adapt / always-freeze. All manuscript refs are `kbound_short.tex:<line>`.

### Uniform nine-track panel (Table `tab:uniform-panel`)
| # | Track | Printed value / tier (file:line) | Ledger (sec3) | Manifest | Raw | STATUS |
|---|-------|----------------------------------|---------------|----------|-----|--------|
| 1 | CIFAR-10-C | Tent 0.0016/**0.0080**/**0.1239**; EATA 0.0013/0.0033/**0.1313**; SAR withheld; FA_u=0 (897) | Tent .0016/.0079/.1241; EATA …/.1314 | tent [.0015736,.0079234,.1240979]; eata […,.1313789] | LOCKED_ANALYSIS: tent .0016259/.0079757/.1239368; eata …/.1313370 | **MISMATCH (internal)** — panel = raw/G8 rounding; rest of paper + manifest + ledger = .0079/.1241,.1314 |
| 2 | ImageNet-C SAR | 0.0264/0.0529/0.0319 pooled; FA_u=0.000 (exact SC); locked 5-seed (898) | .0107/.0529/.0319 FA_u=.007 **[ledger self-flags STALE]**; G8=0.0264/0.0529/0.0319 | imagenetc_sar [.026422,.052933,.031894] FA_u 0, 135 cells, exact-rank, regen 2026-07-21 | win_hunt_v5_imagenetc_ms 5-seed | **MATCH** (= manifest+G8+already-fixed; supersedes stale ledger-sec3 KGA/.007) |
| 3 | Camelyon17 OOD | 0.0000/0.0000/0.1381; FA_u=0 (899) | .0000/.0000/.1381 FA_u=0 | camelyon17_ood [0,0,.1381] FA_u 0 n=18 | — | **MATCH** |
| 4 | iWildCam H v2 | 0.0041/0.1028/0.0041; FA_u=0; **tier "5-seed real-ckpt confirmed"** (900) | .0041/.1028/.0041 FA_u=0; **"fresh 5-seed pending (rerun queued)"** | iwildcam_H_v2 […] cal_seeds[0] test_seeds[1] (single-run) | — | numbers **MATCH**; **TIER MISMATCH** (see defect 2) |
| 5 | Office-Home M v2 | 0.0157/0.0468/0.0158; FA_u=0; tier "OOF-lock design value; 5-seed no-harm confirmed" (901) | .0157/.0468/.0158 FA_u=0 | officehome_M_v2 [.0157143,.0468132,.0158242] seeds[0,1] | multiseed OH 5-seed = no-harm | **MATCH** (tier = already-fixed #4; line 865-866 still lists OH as "future work" = stale) |
| 6 | RxRx1 J | 0.0000/**0.2587**/0.0000; FA_u=0 (902) | .0000/**.2531**/.0000 FA_u=0 | rxrx1_J [0,**0.2531**,0] | uniform_verdicts .2531; archived drafts .2531 | **MISMATCH** (0.2587 in no artifact) |
| 7 | PACS | primary replay FA_u=0 all four targets; photo-null (0.056) not reproduced; diag 1/3 seeds (903) | 3 safe/1 null; diagnostic | pacs {1 seed, diagnostic} | pacs_result.json: 4 targets FA_u=0, photo verdict NULL | **MATCH** (= already-fixed #4) |
| 8 | ImageNet-R D | no CI-robust routing utility; diag 3/4 seeds (904) | no CI-robust BB; diag 3/4 | imagenet_r_D diagnostic | uniform_verdicts diag | **MATCH** (qualitative) |
| 9 | CIFAR-10.1 K | 0.0021/0.0190/0.0017; FA_u=0.167, FA_c=0.444; corrected label (905) | fails bar FA_u=.167,FA_c=.444 | cifar10_1_K [.0021,.019,.0017] FA_u .1667 FA_c .4444 | — | **MATCH** (label-corrected) |

### Primary numeric table (Table `tab:primary-numeric`)
| # | Track | Printed (file:line) | Manifest | STATUS |
|---|-------|---------------------|----------|--------|
| 10 | CIFAR-10-C Tent | .0016/.0079/.1241 (936) | [.0015736,.0079234,.1240979] | **MATCH** (≠ panel line 897) |
| 11 | CIFAR-10-C EATA | .0013/.0033/.1314 (937) | [.0012676,.0032683,.1313789] | **MATCH** (≠ panel line 897) |
| 12 | ImageNet-C SAR | .0264/.0529/.0319 (938) | [.026422,.052933,.031894] | **MATCH** |
| 13 | Camelyon17 OOD | .0000/.0000/.1381 (939) | [0,0,.1381] | **MATCH** |
| 14 | RxRx1 J | .0000/**.2587**/.0000 (940) | [0,**.2531**,0] | **MISMATCH** (same error as #6) |
| 15 | Three-source OOF | .0059/.0632/.0342 (941) | three_source_oof [.0059117,.0632323,.0342043] | **MATCH** |

### Other quantitative tables
| # | Item | Printed (file:line) | Source | STATUS |
|---|------|---------------------|--------|--------|
| 16 | ImageNet-C faithful SAR row | K-Bound 0.427 / 0.0264; adapt 0.385/0.0529; freeze 0.406/0.0319 (819-821, text 786-790) | manifest imagenetc_sar 0.0264/0.0529/0.0319 | **MATCH** (point est. exact) |
| 16b | ImageNet-C SAR gap CIs | freeze [-0.0088,-0.0027]; adapt [-0.052,-0.003] (789-790) | manifest [-0.0086,-0.0026]; [-0.0518,-0.0038] | **MATCH-approx** (4th-decimal bootstrap noise; both exclude 0) |
| 17 | Multi-seed Camelyon Tent | 0.020±0.023/0.138/0.020; FA_u 0.00; stable no-harm (877) | raw `wilds_kbound` (9 cond/seed×4): kga .0201/adapt .1380/freeze .0201 | **MATCH** (see provenance note) |
| 18 | Multi-seed Camelyon EATA | 0.039±0.025/0.042/0.042; FA_u 0.00; inconclusive (878) | wilds_kbound: kga .0393/adapt .0417/freeze .0424 | **MATCH** |
| 19 | Multi-seed Camelyon SAR | 0.041±0.017/0.000/0.065; FA_u 0.11; over-freezes (879) | wilds_kbound: kga .041/adapt .0002/freeze .0654; FA per-seed max 1/9=.111 | **MATCH** (FA_u=0.11 = per-seed max per caption 869-870) |
| 20 | Head-to-head KGA vs POEM/AETTA | gaps adapt -0.0064, freeze -0.1225, POEM -0.0072[-0.0088,-0.0057], AETTA -0.0058[-0.0071,-0.0044]; FA_u=0 (750-753) | manifest headtohead kga .0015736 / poem .0088046 / aetta .0073299 / adapt .0079234 / freeze .1240979 | **MATCH** (gap arithmetic exact; KB-CLAIM-026 supported) |
| 21 | Gates + α-sweep ablations | KGA cert 0.0017/FA_u 0.000 (707,1011); no-radius FA_u **0.049** (706) vs **0.051** (986,1013) | manifest: "sensitivity_ablations: no immutable final exact-rank artifacts found" | **UNVERIFIABLE** (no artifact-backed lock; + minor 0.049/0.051 internal inconsistency) |

---

## (b) NEW mismatches / blurred distinctions (NOT already in ledger gaps G1–G11)

1. **[HARD NUMERIC] RxRx1 always-adapt regret = 0.2587** at `kbound_short.tex:902` (uniform panel)
   and `:940` (primary numeric table). Correct value **0.2531** in `kbound_result_manifest.json:158`,
   `uniform_verdicts.json:176`, `SUBMISSION_LEDGER.md:70`, and both archived 2026-07-15 drafts.
   `0.2587` occurs in NO result artifact (only unrelated Lean/mathlib C build noise). Transcription
   regression introduced after the 2026-07-15 draft.

2. **[TIER OVERSTATEMENT] iWildCam "5-seed real-ckpt confirmed"** at `kbound_short.tex:900`.
   Contradicts (i) ledger sec3:68 "fresh 5-seed pending (real-ckpt rerun queued)"; (ii) manifest
   `iwildcam_H_v2` = single calibration seed [0] / single test seed [1]; (iii) the paper's own
   line 865-866 "Per-seed multi-seed replays for iWildCam, Office-Home, and RxRx1 are single-run …
   and remain future work"; (iv) the only iWildCam multi-seed stability artifact
   (`multiseed_natural_forest_payload.json`) uses 2 seeds. iWildCam numbers themselves (0.0041/0.1028/0.0041)
   are fine — only the tier label is unsupported.

3. **[INTERNAL INCONSISTENCY] CIFAR-10-C Tent/EATA adapt+freeze differ between the two panels.**
   Uniform panel `:897` prints 0.0080/0.1239 (Tent) and 0.1313 (EATA freeze) — matching only the raw
   `LOCKED_ANALYSIS_RESULTS.json` and `G8_EXACTRANK_REGEN.md`. The primary table `:936-937`, the
   head-to-head macros (`kbound_numbers.tex`: adapt 0.0079 / freeze 0.1241), the manifest,
   `uniform_verdicts.json`, and ledger-sec3 all use 0.0079/0.1241 and 0.1314. Note: the manifest
   itself differs from its own cited raw source (`LOCKED_ANALYSIS_RESULTS.json`) at the 4th decimal
   (~5e-5). Cosmetic only (KGA 0.0016/0.0013 identical; beats-both & FA_u=0 unaffected; fixed-policy
   regrets are labeled "secondary"), but line 897 is the single out-of-step location.

### Provenance / repo-hygiene notes (not manuscript contradictions)
- **Camelyon multi-seed source split.** The manuscript's `tab:multiseed` (877-879) is reproduced by
  raw `experiments/kbound/results/wilds_kbound/` (9 conditions/seed × 4 seeds). But the *generated*
  `figures/tab_multiseed_natural.tex` + `multiseed_natural_forest_payload.json` were built from a
  DIFFERENT committed Camelyon set (`natural_win_v1_camelyon`, 36 conditions/seed), which flips the
  regime: it labels Tent "unstable/other" (helpful-dominated, KGA loses to adapt, FA_max 0.028) and
  SAR "stable no-harm" (FA 0) — the opposite of the manuscript's Tent-stable / SAR-over-freezes story.
  No in-paper conflict today (the manuscript embeds a hand-written table and includes the single-seed
  `fig_natural_forest.png` at line 848, NOT `fig_natural_forest_multiseed.png`), but if that generated
  table/figure is ever `\input`/included it will contradict `tab:multiseed`. Recommend deleting or
  regenerating the stale generated multiseed artifact from `wilds_kbound`.
- **Ablation artifacts.** `kbound_short.tex:979-980` cites a "locked exact-rank ablation artifact with
  input hashes," but `kbound_result_manifest.json` `withheld_or_pending.sensitivity_ablations` =
  "no immutable final exact-rank artifacts found." Tables `tab:abl-*` are internally consistent but not
  artifact-backed in the manifest.

---

## (c) PRIORITIZED FIX QUEUE
1. **P0** — Fix RxRx1 `0.2587 → 0.2531` at `kbound_short.tex:902` and `:940`.
2. **P1** — iWildCam tier at `:900`: change "5-seed real-ckpt confirmed" to match reality/ledger
   (single-run OOF lock; 5-seed pending), OR reconcile ledger sec3:68 + line 865-866 if a real
   iWildCam 5-seed run now exists. Also update line 865-866, which still lists Office-Home as
   "single-run … future work" despite line 901's (already-fixed-endorsed) "5-seed no-harm confirmed".
3. **P2** — Reconcile uniform-panel CIFAR-10-C adapt/freeze at `:897` to the paper-wide value
   0.0079/0.1241 (Tent) and 0.1314 (EATA freeze); align the manifest with its cited raw source.
4. **P3** — Delete/regenerate the stale generated `tab_multiseed_natural.tex` /
   `multiseed_natural_forest_payload.json` (built from `natural_win_v1`, verdict-flipped vs the paper).
5. **P3** — Reconcile no-radius FA_u 0.049 (`:706`) vs 0.051 (`:986`,`:1013`); commit an immutable
   ablation artifact to back `tab:abl-*` (manifest currently says none exists).

---

## WORDING GUARDS (withdrawn-claim forbidden phrases — all ABSENT from both .tex)
- **KB-CLAIM-022** "beats both Camelyon17": ABSENT (occurs only in `claim_ledger.json:172`). PASS.
- **KB-CLAIM-023/024** "13x","24x","beats both mixed": ABSENT. PASS. Three-source labeled "constructed
  routing mixture, not a natural-transfer result" (`:929`, `:1098-1099`).
- **KB-CLAIM-004** "FA_c ≤ alpha guaranteed": ABSENT; paper states "FA_c … not claimed" (`:958`),
  "do not interpret FA_c as a certificate" (`:841`), "no conditional-error guarantee" (`:57`). PASS.
- **KB-CLAIM-050** "universal improvement / always beats adapt": ABSENT; "not a universal accuracy
  booster" (`:41`), "does not claim universal improvement" (`:88`), "What we do not claim: Universal
  accuracy gains" (`:1146`). PASS.
- **KB-CLAIM-012** "jackknife+ / distribution-free without assumptions": paper uses "jackknife+" only
  in the NEGATED sense ("jackknife+ is not claimed", appendix `:390`; "distribution-free only
  asymptotically", `:326-327`). PASS.
- The only "beats both" in the body is `:787` (ImageNet-C SAR — a legitimate mixed-regime track). PASS.

## DISTINCTION GATES (sec 5) — ALL HOLD
- safety/validity ≠ accuracy: `:41`, `:89`, `:1091`, `:1146`.
- theorem-guarantee ≠ empirical-coverage: `:56-57`, `:957`, `:1101`; guarantee table `:956-966`.
- mixed-regime beats-both ≠ one-sided no-harm: `:41`, `:87-89`, `:111-112`, `:787`, `:1091`.
- natural benchmark ≠ constructed mixture: `:96-97`, `:125`, `:922-929`, `:1098-1099`.
- official method ≠ protocol-matched port: `:553`, `:724-727`, `:731`, `:760-775`, `:1140`, `:1190`.
- locked/reconciled ≠ diagnostic/incomplete: `:843-845`, panel tier column `:895-905`, `:131-132`, `:959`.

## ALREADY-FIXED ITEMS — CONFIRMED CONSISTENT
- **CIFAR-10-C SAR WITHHELD**: `:897` "SAR withheld (seed-0 aggregate non-reproducing)"; ledger sec3;
  manifest `withheld_or_pending.cifar10c_sar`. Not printed as a promoted triple anywhere. CONSISTENT.
- **ImageNet-C SAR 0.0264/0.0529/0.0319, FA_u=0 (exact split-conformal, 5-seed, 135 cells)**: `:898`,
  `:938`, `:819-821` all agree; = manifest (regen 2026-07-21). Supersedes stale ledger-sec3:66. CONSISTENT.
- **PACS photo 0.056 withdrawn → FA_u=0 on all four targets**: `:903`, `:122`; = `pacs_result.json`
  (4 targets FA_u=0, photo verdict NULL). CONSISTENT.
- **Office-Home 0.0157 OOF-lock design value + 5-seed no-harm confirmation**: `:901` = manifest
  `officehome_M_v2`; 5-seed stability artifact present. CONSISTENT (note line 865-866 stale re OH).
