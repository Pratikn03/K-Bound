# Phase 4-5 Evidence Matrix (2026-07-21). Two audit subagents (grid=complete, natural=partial).
Replay rule: exact split-conformal eps=sorted(rho)[k-1], k=ceil((n+1)(1-a)), a=0.10; FA_u=mean(ADAPT & B<=0).

## GRID + DIAGNOSTIC (Agent A, complete)
Track            | seeds | replay KGA/adapt/freeze / FA_u | matches ms | status
CIFAR-10-C Tent  | 5/5   | .0016/.0080/.1239 / 0          | 3dp yes (4dp refreshed) | COMPLETE
CIFAR-10-C EATA  | 5/5   | .0013/.0033/.1313 / 0          | 3dp yes (4dp refreshed) | COMPLETE
CIFAR-10-C SAR   | seed0 non-repro | .0016/.0112/.1286 (seed0 harmful .528 vs others ~.09, 180x outlier) | -- | STALE/WITHHELD (G2) -> panel now WITHHELD
ImageNet-C SAR   | 5/5   | .0264/.0529/.0319 / 0.000 EXACT | YES exact | COMPLETE (headline verified from raw)
PACS             | 1/3   | aggregate only; photo FA_u raw=0.0 (ms says .056) | NO (.056 not in raw) | NON-REPLAYABLE + PARTIAL
ImageNet-R D     | 3/4   | 10/10 no-beats-both, FA_u=0 (safer under exact) | YES diag | COMPLETE-diag + PARTIAL
CIFAR-10.1 K     | 5/5   | regrets exact; FA_u fails transfer bar | YES diag | COMPLETE-diag

## NATURAL-SHIFT (Agent B, PARTIAL — connection dropped)
Camelyon17 OOD   | eata_online test seeds{2,3,4} n=18 -> freeze .13813 adapt 0 frac_harm 0 = 0.0000/0.0000/0.1381 EXACT | COMPLETE
Office-Home M v2 | promoted 0.0157 NOT FOUND in any raw artifact (only manuscript/audit table) | NON-TRACEABLE -> needs traced OOF artifact or note
iWildCam / RxRx1 | (agent cut off before finishing; RxRx1 fresh 0.0/0.2587/0.0 real ckpt confirmed earlier; iWildCam episodic run in progress)

## GAPS
G1 CONFIRMED STALE: paper/generated/kbound_result_manifest.json imagenetc_sar = single-seed OLD (regret .0108/.0625/.0319, seeds[0]); also pooled_5seed/percondition_bootstrap_pooled.json holds interpolated .0107. Raw 5-seed tree is intact + replays exact. -> regenerate manifest to 5-seed exact-rank.
G2 CONFIRMED: CIFAR-10-C SAR seed0 does NOT reproduce (harmful .528 vs .074-.116 archived); flips verdict. FIXED: SAR pulled from CIFAR-10-C panel, marked withheld.
G9 (Camelyon withdrawn beats-both reachability): NOT finished by agent B — re-run needed.

## FIX QUEUE (from evidence matrix)
[DONE] CIFAR-10-C SAR withheld in panel; Tent/EATA 4dp refreshed to raw.
[TODO-local] regenerate kbound_result_manifest.json (ImageNet-C 5-seed exact) + pooled bootstrap JSON.  [G1]
[TODO-local] PACS photo FA_u=0.056 not in raw -> mark non-replayable or correct to 0.0.
[TODO-local] Office-Home 0.0157 -> attach a traced OOF artifact or annotate as design-based.
[TODO] finish G9 Camelyon withdrawn-BB reachability sweep; mark SUPERSEDED where found.
