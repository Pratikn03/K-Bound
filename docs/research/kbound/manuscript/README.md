# K-Bound Manuscript Directory

> **Status (2026-09-19):** Parallel book-style exposition. Not the maintained submission driver.
> The primary submission manuscript is `kbound_submission.tex` / `kbound_tmlr.tex`.

This directory contains two extended expository fragments used during theory development:

| File | Content |
|------|---------|
| `theory_spine/theory_beta_impossible.tex` | Impossibility frontier — why $\beta^\star$ cannot be identified from unlabeled observables alone |
| `theory_spine/theory_beta_estimable.tex` | Estimable-budget characterization under Assumption E2 (exchangeable episodes) |

## Open conjecture

`theory_beta_estimable.tex` contains one open conjecture (`epi:conj-open`, §"What is left open"):

> Is there a structural assumption, weaker than E2 and *falsifiable from retrospective labels on the history alone*, under which $\beta^\star$ for a novel episode is identified?

This is correctly labelled as an open conjecture in the text ("we neither exhibit such an assumption nor prove that none exists"). It is **not** a gap — it is a disclosed limitation of the current theory.

**Note on `conj:gen`:** The earlier label `conj:gen` (label-free bracketing conjecture) appeared in draft `.tex.bak` files and archived drafts under `archive/paper_drafts_2026-07-15/`. It was resolved via the one-bit dichotomy theorem (`thm:conj1-dichotomy`) and is marked resolved in all current active manuscript files. There is no `conj:gen` open item in this directory.
