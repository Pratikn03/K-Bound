# Independent review of Task 3

Verdict: **PASS for metadata-only eligibility and the no-launch conclusion;
minor metadata corrections recommended before final release.** No candidate
should be launched. No scientific-freshness claim needs to be weakened further.
This review changed only this file and did not open data, pixels, raw labels,
archives, protected containers or checkpoint tensors.

## Scope and checks

Reviewed `IMPLEMENTATION_PLAN.md` Task 3, `data_task_report.md`, all candidate
rows and gates in `data_eligibility.json`, and `data_eligibility.md`. Recomputed
hashes only for the 20 already-consulted protocol/metadata/status/derived-summary
files. Inspected the metadata-only authoring script without executing it, the
So2Sat screen/access records, DomainNet proposal access status, prior-target
provenance audit and PovertyMap STOP receipt. Independently read public text
from the official WILDS, DomainNet and MLCommons documentation; no linked data
or example-image payload was opened.

- All 12 candidate IDs are unique; every row has explicit blockers, a false
  qualification flag and `launch_allowed=false`; no qualifying IDs are listed.
  Every candidate source ID resolves to the consulted-source inventory.
- So2Sat's stored screen is failed: seven direct cities of each direction were
  required, one ADAPT and zero FREEZE cities were observed; 19-city/95-cell
  completion and zero target-read checks remain true. Target inputs are empty,
  target pixels/labels read are both zero in that scoped receipt. The report
  correctly avoids treating these counters as a universal custody proof or
  preparation-time zero calibration reads as current nonaccess.
- DomainNet's exact proposal records clipart/painting OPENED and four remaining
  domains UNKNOWN. The report preserves those values and does not convert
  archive presence into freshness. The prior provenance audit records opened
  Camelyon17, iWildCam, RxRx1, FMoW, Office-Home and PACS tracks, with no verified
  unopened tracks.
- PovertyMap's source receipt says `dev-screen-stop`, screen `STOP`, and held-out
  val/test not run per preregistration. The report correctly distinguishes this
  historical no-run statement from verified current freshness, and keeps the
  existing screen binding. CCT-20 is explicitly already evaluated.
- Recomputed exact-rank examples: n=6 gives k=7, n=8 gives k=9, n=9 gives k=9
  at alpha=.10. Nine is the minimum finite-rank count, not a power or validity
  guarantee. Repeated seeds/images do not become additional independent domains.
  New episode-level claims require a different declared population and design.
- Unknown counts remain null. New eligible verified counts of zero mean zero
  established by this audit, not a claim that the dataset contains no eligible
  units. CLEAR/Dollar Street remain unknown-access candidates with bounded
  search scope, not proven absent datasets or declared fresh targets.
- Gates include explicit role separation, metric/pair binding, maintained loss
  weight 5, executable preregistration, effect/precision rationale, multiplicity,
  local cost pilot and review before launch. Since no candidate qualifies, not
  executing a new sample-size pilot is the correct Task 3 outcome.
- Resource observations are dated availability measurements, explicitly not a
  runtime pilot. The inspected script reads only allowlisted documentary inputs,
  capped public text responses, single-level names, lstat/statvfs and sysctl
  metadata. It performs no data/model imports, training, scoring or target reads.
  This verifies the script's scope, not every action in the user's history.

The licensing language is conservative and consistent with checked primary
notices: [WILDS](https://wilds.stanford.edu/datasets/) lists the stated iWildCam,
Camelyon17 and RxRx1 terms; [DomainNet](https://ai.bu.edu/DomainNet/) provides a
noncommercial research/education fair-use notice rather than blanket ownership
or redistribution permission; [MLCommons](https://mlcommons.org/datasets/dollar-street/)
lists CC-BY/CC-BY-SA 4.0 and 63 countries. The report does not equate usable
access, scientific freshness and redistribution permission. Unverified terms
for already-blocked candidates stay explicitly unverified.

## Minor actionable issues

1. **P3 — Give numeric group counts explicit count units.** In the JSON,
   Dollar Street's `groups.unit` is “household nested within country” while
   `documented_total=63` counts countries, not households. FMoW's unit combines
   region and time while five counts broad regions. So2Sat's 42 counts training
   cities and excludes ten additional target cities. The prose qualifies these
   correctly, but the generic numeric field and table column could be misread
   by a downstream consumer. Add `documented_total_unit` and population scope
   (or separate country/household, region/time, training/target fields), keeping
   unknown counts null. This does not alter any eligibility verdict.
2. **P3 — Preserve snapshot semantics during final source binding.** Nineteen
   of 20 consulted local hashes still match. `manuscript_supplement` now differs
   after the root's concurrent integration (194,706 bytes recorded; 195,139 at
   review). This is expected source drift, not evidence corruption. Either keep
   the report explicitly as the dated pre-integration snapshot and package that
   source version, or review the final supplement and add a clearly dated final
   binding. Do not silently imply all 20 hashes identify the final manuscript.
   The immutable screen/access/protocol authorities matched exactly.

No high- or medium-priority issue was found. Neither minor issue authorizes
opening a blocked target, relaxing a screen, or declaring prospective closure.
