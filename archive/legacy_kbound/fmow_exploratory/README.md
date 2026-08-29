# Archived FMoW exploratory helpers

These three scripts are retained only as historical development material. They
are not part of the active K-Bound FMoW pipeline and must not be used to create
publication evidence.

- `download_fmow.py` is a small, unpinned convenience downloader.
- `find_fmow_regime_mix.py` has a misleading historical name: it evaluates the
  frozen source model by region but does not run Tent or EATA and therefore
  cannot establish an adaptation regime.
- `train_fmow_f0.py` is an early single-run trainer. It has no immutable
  protocol seal, exact environment receipt, resume contract, or independent
  checkpoint panel.

The maintained data path is `experiments/kbound/wilds/fmow_data.py`; the
maintained experiment entry point is
`experiments/kbound/wilds/run_geoshift_kbound.py`. Any future FMoW training must
use a new sealed protocol rather than reviving these helpers.
