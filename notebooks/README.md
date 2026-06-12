# Notebooks — K-Bound (10-notebook suite)

Reviewer-facing, **executed** reproduction of *When Is Label-Free Adaptation Knowable?*
(KGA: Knowability-Guided Adaptation). Read in order; each notebook finds the repo root
automatically and loads only **real** result artifacts. Outputs are saved in the files,
so they read without re-running.

| # | Notebook | Contents |
|---|---|---|
| 00 | `00_KBound_Reproduction.ipynb` | One-page overview + live trichotomy + all results + figures |
| 01 | `01_Problem_and_Theory.ipynb` | The problem, adapt/freeze/abstain, Thms 1–5 + validators, architecture |
| 02 | `02_Knowability_Trichotomy.ipynb` | Live 123-task suite: evidence `Z`, certificate, decisions, safety, ablations |
| 03 | `03_Harmful_Mixed_Rigor.ipynb` | Harmful-fusion + mixed regime + 8-seed paired t-tests |
| 04 | `04_Regression_and_Witness.ipynb` | Regression covariate-shift + clean non-identifiability witness (live) |
| 05 | `05_TTA_CIFAR_and_Online.ipynb` | CIFAR-10-C, decisive CIFAR+Tent, online continual-Tent collapse |
| 06 | `06_Evidence_and_Drift.ipynb` | The label-free evidence `Z` (drift/KS/disagreement) computed live + importance |
| 07 | `07_Certificate_and_Calibration.ipynb` | Conformal / empirical-Bernstein / e-value certificates + α–coverage calibration |
| 08 | `08_ELARA_Multimodal_Instantiation.ipynb` | ELARA/RGA worked case (MVTec-3D D23, T1–T9/GDR map) |
| 09 | `09_Conclusions_and_Reproducibility.ipynb` | Summary, cross-experiment regret, repro commands, open items |

Full pipeline + paper: `PYTHON=.venv/bin/python bash scripts/rebuild_kbound.sh`.
Dashboard: `docs/research/kbound/kbound_dashboard.html` · Paper: `docs/research/kbound/K-Bound_paper.pdf`.

**Honest scope:** per-corruption CIFAR-10-C is helpful-dominated (always-adapt strong; KGA
matches it safely); KGA's distinctive value is safety + cross-regime robustness. Full
1000-class ImageNet-C is pending (host unreachable). Conjecture 1 (label-free bracketing) is open.

---

`legacy_elara/` holds the 25 superseded ELARA-era EDA notebooks (kept for provenance; not part
of the K-Bound paper).
