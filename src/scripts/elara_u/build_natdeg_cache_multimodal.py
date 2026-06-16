"""Build REAL modality-degradation D23 caches for MVTec-3D and 3D-ADAM (Task L3.2).

Drop-in producer for the ``--natural`` runner: writes per-category caches in the same
``{Sval, yval, Stest, ytest, valauc}`` format that
``src/scripts/elara_u/multimodal_reliability_test.py`` consumes, so the reliability
gate can be tested under *genuine* modality degradation with NO synthetic injection:

    PYTHONPATH=src python src/scripts/elara_u/build_natdeg_cache_multimodal.py \
        --dataset mvtec3d --degradation missing_returns \
        --cache experiments/fusion/mvtec3d_natdeg_missing_returns_score_cache
    PYTHONPATH=src python src/scripts/elara_u/multimodal_reliability_test.py --natural \
        --cache experiments/fusion/mvtec3d_natdeg_missing_returns_score_cache --glob '*.npz' \
        --tag MVTec-3D-NatDeg-missing_returns \
        --out experiments/elara_u/multimodal_reliability_results_mvtec3d_natdeg.json

INTEGRITY (the whole point of L3.2 -- read before editing):
  The degradation is a REAL, physically-grounded sensor artifact applied to the raw
  modality IMAGES of the TEST split only -- the *other* modality and the *entire*
  validation split stay clean -- and the degraded modality is then RE-SCORED with the
  same one-class PatchCore detector used to build the clean caches. The degradation
  therefore propagates through the detector exactly as a real faulty sensor would.
  This is fundamentally different from multimodal_reliability_test.py's
  ``St[:, best] = rng.uniform(0, 1)``, which injects uniform noise directly into the
  SCORES. We never inject score-space noise here, and we never fabricate scores:
  if the raw data is absent, the script prints a clear ``DATA NEEDED`` message and
  exits non-zero -- it does not invent a cache.

Artifacts (per L3.2 "real depth-sensor artifacts" -- ALL DETERMINISTIC, no PRNG, derived
from the real modality data, never random score-space noise):
  missing_returns  -- structured-light/ToF dropout DERIVED FROM THE DATA: removes the
                      most physically dropout-prone REAL returns -- grazing-angle / steep
                      geometry (high local gradient) and valid pixels bordering the
                      sensor's own EXISTING no-return holes -- by setting them to 0
                      ('no return'). Severity = fraction of valid returns removed. The
                      pixels dropped are chosen by the real surface geometry, not at
                      random, so this is a sensor-physics artifact model, not injected noise.
  quantization     -- coarse ADC / bit-depth: real geometry collapsed to a few levels
                      (pure information loss matching a cheaper depth sensor)
  low_illumination -- low-exposure RGB capture: deterministic luminance roll-off
                      (scale + gamma; a real photometric transform, not added noise)

Honest scope: these are deterministic, physically-grounded artifact MODELS applied to the
real modality inputs and propagated through the real one-class PatchCore detector -- they
are NOT a separately captured set of degraded acquisitions. The purest "natural subset"
endpoint (selecting genuinely degraded samples by a measured property, zero operators) and
the already-built Real-IAD-D3 as-is cache (build_realiad_natdeg_cache.py) are the
complementary natural-degradation evidence; see PHASE2_RUNBOOK.md.

Heavy work (PatchCore scoring via torch) runs only on the first GPU pass AFTER the
RxRx1 job frees the device; all torch/tifffile/PIL imports are lazy, so importing this
module is numpy-only. ``--dry-run`` exercises the artifact transforms on tiny synthetic
arrays and reports raw presence, WITHOUT any scoring (CPU-only).

Mirrors the IO of gpu_build_mvtec3d_cache.py (same loaders, _zsig, stratified split).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RNG = 0

DATASETS = {
    # geom="normal": organized point-cloud xyz -> surface-normal image (MVTec-3D)
    # geom="depth" : replicated-channel depth tiff -> contrast-stretched depth (3D-ADAM)
    "mvtec3d": {"raw": ROOT / "data/raw/mvtec3d", "geom": "normal"},
    "3d_adam": {"raw": ROOT / "data/raw/3d_adam_anomalib", "geom": "depth"},
}
# Which modality each real artifact targets. geom = depth/point-cloud, rgb = colour.
DEG_MODALITY = {"missing_returns": "geom", "quantization": "geom", "low_illumination": "rgb"}


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


# --------------------------------------------------------------------------------------
# REAL sensor-artifact transforms. Operate on raw modality arrays (depth HxW, xyz HxWx3,
# or rgb HxWx3). Structured + physically motivated -- NOT score-space random noise.
# --------------------------------------------------------------------------------------
def _valid_mask(arr):
    """HxW bool: pixels that carry a real sensor return. depth HxW or xyz/normal HxWx3;
    0 (or all-zero vector) is the sensor's native 'no return' code."""
    a = np.asarray(arr, float)
    return a > 0 if a.ndim == 2 else np.any(a != 0, axis=-1)


def _grad_mag(arr):
    """HxW local gradient magnitude of the geometry -- a grazing-angle / steepness proxy.
    Structured-light/ToF sensors physically lose returns where the surface is steep, so
    high gradient marks the real pixels most prone to dropout. Summed over channels."""
    a = np.asarray(arr, float)
    if a.ndim == 2:
        gy, gx = np.gradient(a)
        return np.hypot(gx, gy)
    g = np.zeros(a.shape[:2], float)
    for c in range(a.shape[-1]):
        gy, gx = np.gradient(a[..., c])
        g += np.hypot(gx, gy)
    return g


def _dilate(mask, iters):
    """4-neighbour binary dilation in pure numpy (grow the existing no-return regions)."""
    m = np.asarray(mask, bool).copy()
    for _ in range(max(0, int(iters))):
        d = m.copy()
        d[1:, :] |= m[:-1, :]
        d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        m = d
    return m


def degrade_missing_returns(arr, severity, rng=None):
    """REAL structured-light/ToF dropout DERIVED FROM THE DATA (deterministic; ``rng``
    ignored). Sets to 0 ('no return') the ``severity`` fraction of valid returns that a
    real depth sensor is physically most likely to lose: grazing-angle / steep geometry
    (high local gradient) and valid pixels bordering the sensor's EXISTING no-return
    holes (dilated). No PRNG and no added values -- only genuine returns are removed,
    chosen by the real surface geometry, the way a faulty depth sensor loses them."""
    out = np.array(arr, float, copy=True)
    valid = _valid_mask(out)
    nvalid = int(valid.sum())
    if nvalid == 0:
        return out
    s = float(np.clip(severity, 0.0, 0.95))
    k = int(round(s * nvalid))
    if k <= 0:
        return out
    # susceptibility = normalized steepness + indicator of bordering an existing hole.
    g = _grad_mag(out)
    gmax = float(g[valid].max())
    steep = g / gmax if gmax > 0 else np.zeros_like(g)
    border = _dilate(~valid, max(1, int(round(3 * s)))) & valid
    susc = np.where(valid, steep + border.astype(float), -np.inf)
    # drop the k most dropout-prone valid pixels (deterministic top-k by susceptibility).
    drop_idx = np.argsort(susc, axis=None)[::-1][:k]
    drop = np.zeros(out.shape[:2], bool)
    drop.flat[drop_idx] = True
    out[drop] = 0.0
    return out


def degrade_quantization(arr, severity, rng=None):
    """Coarse ADC/bit-depth: collapse valid values to N levels. Higher severity = fewer
    levels (levels = max(2, round(64*(1-severity)))). Preserves 0 ('no-return') pixels."""
    out = np.array(arr, float, copy=True)
    flat = out.reshape(-1, out.shape[-1]) if out.ndim == 3 else out.reshape(-1, 1)
    levels = max(2, int(round(64 * (1.0 - float(np.clip(severity, 0.0, 0.98))))))
    for c in range(flat.shape[1]):
        col = flat[:, c]
        m = col > 0
        if int(m.sum()) < 10:
            continue
        lo, hi = np.percentile(col[m], 1), np.percentile(col[m], 99)
        q = np.round((np.clip(col, lo, hi) - lo) / (hi - lo + 1e-6) * (levels - 1)) / (levels - 1)
        col[m] = q[m] * (hi - lo) + lo
        flat[:, c] = col
    return flat.reshape(out.shape)


def degrade_low_illumination(arr, severity, rng=None):
    """Low-exposure RGB capture: deterministic luminance roll-off (scale + gamma).
    factor = 1 - 0.9*severity, gamma = 1 + 1.5*severity. No added random noise."""
    a = np.array(arr, float, copy=True)
    s = float(np.clip(severity, 0.0, 1.0))
    factor, gamma = max(0.05, 1.0 - 0.9 * s), 1.0 + 1.5 * s
    return np.clip(255.0 * np.power(np.clip(a / 255.0, 0, 1) * factor, gamma), 0, 255)


_TRANSFORMS = {
    "missing_returns": degrade_missing_returns,
    "quantization": degrade_quantization,
    "low_illumination": degrade_low_illumination,
}


# --------------------------------------------------------------------------------------
# Loaders (lazy heavy imports) -- mirror gpu_build_mvtec3d_cache.py.
# --------------------------------------------------------------------------------------
def _load_rgb_arr(p):
    from PIL import Image

    im = Image.open(p)
    return np.asarray(im.convert("RGB"), dtype=np.float32)


def _rgb_arr_to_img(a):
    from PIL import Image

    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def _load_geom_raw(p, geom):
    import tifffile

    a = np.asarray(tifffile.imread(p), dtype=np.float32)
    if geom == "depth":
        return a[..., 0] if a.ndim == 3 else a  # single-channel depth (3D-ADAM)
    return a  # organized xyz HxWx3 (MVTec-3D)


def _geom_raw_to_img(a, geom):
    if geom == "depth":
        from PIL import Image

        d = np.asarray(a, float)
        m = d > 0
        if int(m.sum()) > 10:
            lo, hi = np.percentile(d[m], 2), np.percentile(d[m], 98)
            d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
        return Image.fromarray((d * 255).astype(np.uint8)).convert("RGB")
    from uais.fusion.attention.realiad_3d_detector import xyz_to_normal_image

    return xyz_to_normal_image(np.asarray(a, np.float32))


def _samples(cat_dir: Path, split: str):
    """Return [(rgb_path, geom_path, label)] for a split; label 1 = defect."""
    out = []
    base = cat_dir / split
    if not base.exists():
        return out
    for defect_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("._")):
        label = 0 if defect_dir.name == "good" else 1
        rgb_dir, xyz_dir = defect_dir / "rgb", defect_dir / "xyz"
        if not rgb_dir.exists():
            continue
        for rgb in sorted(rgb_dir.glob("*.png")):
            xyz = xyz_dir / (rgb.stem + ".tiff")
            if xyz.exists():
                out.append((rgb, xyz, label))
    return out


def build_category(cat, ds, degradation, modality, severity, coreset, cap=130):
    """Score one category with the chosen modality degraded on TEST only. GPU/heavy."""
    from sklearn.metrics import roc_auc_score
    from uais.fusion.attention.realiad_3d_detector import score_one_class_patchcore

    geom = ds["geom"]
    cat_dir = ds["raw"] / cat
    train = _samples(cat_dir, "train")
    pool = _samples(cat_dir, "validation") + _samples(cat_dir, "test")
    if len([s for s in pool if s[2] == 1]) < 5 or len([s for s in pool if s[2] == 0]) < 5 or not train:
        return None
    rng = np.random.default_rng(RNG)
    pool = [pool[i] for i in rng.permutation(len(pool))[:cap]]
    y = np.array([s[2] for s in pool])
    idx = np.arange(len(pool))
    val_idx = []
    for lab in (0, 1):
        li = idx[y == lab]
        val_idx += list(li[: len(li) // 2])
    val_mask = np.zeros(len(pool), bool)
    val_mask[val_idx] = True
    yval, ytest = y[val_mask], y[~val_mask]
    if len(np.unique(yval)) < 2 or len(np.unique(ytest)) < 2:
        return None

    transform = _TRANSFORMS[degradation]
    Sval_cols, Stest_cols, vauc = [], [], []
    for m_name, col in (("rgb", 0), ("geom", 1)):
        load_raw = (lambda p: _load_rgb_arr(p)) if m_name == "rgb" else (lambda p: _load_geom_raw(p, geom))
        to_img = (lambda a: _rgb_arr_to_img(a)) if m_name == "rgb" else (lambda a: _geom_raw_to_img(a, geom))
        bank = [to_img(load_raw(s[col])) for s in train]
        ref = score_one_class_patchcore(bank[: min(40, len(bank))], bank, coreset_size=coreset)
        evs = []
        for i, s in enumerate(pool):
            arr = load_raw(s[col])
            if m_name == modality and not val_mask[i]:  # degrade test split of target modality
                arr = transform(arr, severity)  # deterministic, data-derived (no PRNG)
            evs.append(to_img(arr))
        sc = _zsig(score_one_class_patchcore(bank, evs, coreset_size=coreset), ref)
        Sval_cols.append(sc[val_mask])
        Stest_cols.append(sc[~val_mask])
        vauc.append(float(roc_auc_score(yval, sc[val_mask])) if len(np.unique(yval)) > 1 else 0.5)
    return np.column_stack(Sval_cols), yval, np.column_stack(Stest_cols), ytest, np.array(vauc, float)


def _check_raw_or_die(ds, dataset):
    raw = ds["raw"]
    cats = []
    if raw.exists():
        cats = sorted(p.name for p in raw.iterdir() if p.is_dir() and not p.name.startswith("._"))
    if not cats:
        print(
            "DATA NEEDED: raw multimodal data for "
            f"'{dataset}' not found at {raw}\n"
            "  Expected layout: <raw>/<category>/{train,validation,test}/<defect>/{rgb,xyz}/\n"
            "  MVTec-3D : https://www.mvtec.com/company/research/datasets/mvtec-3d-ad (~26 GB)\n"
            "  3D-ADAM  : anomalib-format RGB+depth export (~6.5 GB)\n"
            "  Stage the dataset, then re-run WITHOUT --dry-run on a GPU machine.",
            flush=True,
        )
        raise SystemExit(2)
    return cats


def _dry_run(ds, dataset, degradation, modality, severity):
    """CPU-only: exercise the artifact transform on tiny synthetic arrays and report raw
    presence. No torch, no scoring -- validates the operator + output schema, not the
    dataset (the real build still fails loudly via _check_raw_or_die if raw is absent)."""
    geom = ds["geom"]
    ramp = (np.linspace(0, 1, 48, dtype=np.float32) * 10.0)
    synth = {
        "rgb": np.full((48, 64, 3), 200.0, np.float32),
        "geom": (np.tile(ramp[:, None], (1, 64)) if geom == "depth"
                 else np.tile((ramp / 10.0)[:, None, None], (1, 64, 3))),
    }[modality]
    out = _TRANSFORMS[degradation](synth, severity)            # deterministic, no PRNG
    assert out.shape == synth.shape, "transform must preserve shape"
    if degradation == "missing_returns":
        effect = f"zeroed {(out == 0).mean() * 100:.0f}% of pixels (data-derived no-return)"
        assert (out == 0).any(), "missing_returns produced no dropout"
        assert (out != 0).any(), "missing_returns zeroed everything"
    elif degradation == "quantization":
        effect = f"unique levels {len(np.unique(synth)):d} -> {len(np.unique(out)):d}"
        assert len(np.unique(out)) <= len(np.unique(synth)), "quantization must not add levels"
    else:
        effect = f"mean luminance {synth.mean():.0f} -> {out.mean():.0f} (darker)"
        assert out.mean() < synth.mean(), "low_illumination did not darken"
    raw = ds["raw"]
    raw_cats = (sorted(p.name for p in raw.iterdir() if p.is_dir() and not p.name.startswith("._"))
                if raw.exists() else [])
    print("=== DRY RUN (no scoring, no torch, CPU-only) ===")
    print(f"dataset={dataset} geom={geom} degradation={degradation} modality={modality} severity={severity}")
    print(f"artifact on synthetic {modality} array {synth.shape}: {effect}")
    print(f"degraded modality column: {'0 (rgb)' if modality == 'rgb' else '1 (geom)'}")
    print("per-category output schema: npz {Sval[n,2], yval[n], Stest[m,2], ytest[m], valauc[2]} "
          "-- consumed by multimodal_reliability_test.py --natural; degraded modality is the "
          "TEST column only, validation + other modality stay clean")
    if raw_cats:
        print(f"raw OK: {len(raw_cats)} categories under {raw} -> ready to build on a free GPU")
    else:
        print(f"raw ABSENT at {raw} -> the real build will print DATA NEEDED and exit (no fabrication)")
    print("OK -- run WITHOUT --dry-run on a free GPU to build the real cache.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build REAL modality-degradation D23 caches (L3.2).")
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--degradation", choices=sorted(_TRANSFORMS), required=True)
    ap.add_argument("--modality", choices=["auto", "rgb", "geom"], default="auto",
                    help="degraded modality; 'auto' = the artifact's natural target")
    ap.add_argument("--severity", type=float, default=0.5, help="artifact strength in [0,1]")
    ap.add_argument("--coreset", type=int, default=4096)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--raw", default=None, help="override raw root")
    ap.add_argument("--cache", default=None, help="output score-cache dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU-only: validate raw + artifact transform on synthetic arrays, no scoring")
    args = ap.parse_args()

    ds = dict(DATASETS[args.dataset])
    if args.raw:
        ds["raw"] = Path(args.raw)
    modality = DEG_MODALITY[args.degradation] if args.modality == "auto" else args.modality

    if args.dry_run:
        return _dry_run(ds, args.dataset, args.degradation, modality, args.severity)

    cats = _check_raw_or_die(ds, args.dataset)
    cache = Path(args.cache) if args.cache else (
        ROOT / f"experiments/fusion/{args.dataset}_natdeg_{args.degradation}_score_cache")
    cache.mkdir(parents=True, exist_ok=True)
    cats = args.categories or cats
    n = 0
    for cat in cats:
        try:
            res = build_category(cat, ds, args.degradation, modality, args.severity, args.coreset)
        except Exception as e:  # keep the build robust; never fabricate a cache
            print(f"[{cat}] FAILED: {type(e).__name__}: {e}")
            continue
        if res is None:
            print(f"[{cat}] skipped (insufficient labelled split)")
            continue
        Sval, yval, Stest, ytest, vauc = res
        np.savez(cache / f"{cat}.npz", Sval=Sval, yval=yval, Stest=Stest, ytest=ytest, valauc=vauc)
        n += 1
        print(f"[{cat}] cached val={len(yval)} test={len(ytest)} valauc(rgb,geom)={np.round(vauc, 3)}", flush=True)
    print(f"\nwrote {n} {args.dataset} natural-degradation ({args.degradation}) caches to {cache}")
    print("next: multimodal_reliability_test.py --natural --cache "
          f"{cache} --glob '*.npz' --tag {args.dataset}-NatDeg-{args.degradation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
