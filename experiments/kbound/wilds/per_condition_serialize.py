"""
per_condition_serialize.py - torch-free per-condition serialization for the K-Bound
WILDS runners (Camelyon17, ImageNet-R, ...), matching the canonical schema produced by
docs/research/kbound/scripts/cifar_tent_mps_v2.py (the stress_grid_multiseed_v1 contract).

WHY THIS EXISTS
  run_camelyon17_kbound.py and run_imagenetr_kbound.py historically wrote a single
  monolithic result_*.json (records[] + conditions[] inline).  The multi-seed paired-CI
  analysis (experiments/kbound/results/stress_grid_multiseed_v1/_locked_analysis_script.py)
  expects, per (method, seed), a file:
      per_condition_<dataset>_<method>_seed<S>.json
  with one object per condition carrying: B, a0, a_adapted, regime, oracle_action, Z,
  Z_names, b_hat, eps_conformal, kga_decision, ...  This module produces exactly that,
  so the GPU multi-seed runs are directly consumable by the locked analysis machinery.

INTEGRITY
  - This module ONLY reshapes already-measured records into the per-condition schema and
    runs the single-candidate KGA certificate over them.  It fabricates nothing.
  - decide_benefit() prefers the label-disjoint cross-fitted diagnostic
    (sklearn GradientBoostingRegressor + disjoint exact-rank calibration). A clearly labelled numpy
    k-NN-in-Z estimator is available only as an explicit diagnostic fallback; it is
    never publication-eligible. It retains the legacy leave-one-out exact-rank
    radius for plumbing diagnostics and stamps `kga_backend` accordingly.

The function is intentionally pure-Python/numpy so it can be unit-tested and smoke-tested
without torch.
"""
from __future__ import annotations
import os
import sys
import json
import math
import hashlib
import re
import shutil
import tempfile
from collections import Counter
import numpy as np

# ---- the ONE K-Bound decision path (fix-queue items 4 + 15) -----------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 3))
_KB_SCRIPTS = os.path.join(_REPO_ROOT, "docs", "research", "kbound", "scripts")
if _KB_SCRIPTS not in sys.path:
    sys.path.insert(0, _KB_SCRIPTS)
import kbound_decide as _kb  # noqa: E402

CALIBRATION = "crossfit_split"

# Canonical evidence names (mirror tta_methods.EVIDENCE_NAMES; duplicated here so this
# module stays importable without torch).  Asserted equal to tta_methods.EVIDENCE_NAMES
# inside the runners at wiring time.
EVIDENCE_NAMES = [
    "pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf",
    "post_pbal", "pbal_drop", "entropy_drop", "frac_highconf",
    "marginal_KL", "update_norm",
]

ALPHA = 0.10


# --------------------------------------------------------------------------- #
# (a) single-candidate KGA benefit certificate                                #
# --------------------------------------------------------------------------- #
def _decide_kga_sklearn(Z, B, alpha, sample_ids=None):
    """Production path: THE shipped decision path (``kbound_decide`` -> ``kga``).

    FIX-QUEUE ITEMS 4 + 15.  The old body was inlined decide_kga fork #8::

        ... 250-tree GBR LOO loop ...
        eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
        dec = np.where(Bhat - eps > 0, "ADAPT", ...)

    -- an interpolated in-pool quantile, and a ninth private copy of the decision
    rule.  Both are gone.  ``eps`` is now an ndarray of per-cell
    label-disjoint cross-fitted exact-rank radii.
    """
    return _kb.decide_kga_crossfit(Z, B, alpha=alpha, sample_ids=sample_ids)


def _decide_kga_numpy(Z, B, alpha, k=8):
    """sklearn-free diagnostic: leave-one-out k-NN benefit estimate and radius.

    This is NOT the label-disjoint production estimator/calibration design; it exists so the
    serialization / aggregation / decision-rule plumbing can be exercised in a
    torch+sklearn-free sandbox.  Callers stamp `kga_backend="numpy_knn_fallback"`.

    FIX-QUEUE ITEMS 4 + 15: the estimator is local, while the legacy diagnostic
    radius and decision rule delegate to ``kbound_decide`` (exact-rank,
    leave-one-out-of-pool).  This backend can never support promotion.
    """
    Z = np.asarray(Z, float); B = np.asarray(B, float); N = len(B)
    # standardize Z columns for a scale-free distance
    mu = Z.mean(0); sd = Z.std(0); sd[sd == 0] = 1.0
    Zs = (Z - mu) / sd
    Bhat = np.zeros(N)
    kk = max(1, min(k, N - 1))
    for i in range(N):
        d = np.sqrt(((Zs - Zs[i]) ** 2).sum(1))
        d[i] = np.inf
        nn = np.argsort(d)[:kk]
        Bhat[i] = float(np.mean(B[nn]))
    # Explicit diagnostic fallback: it cannot implement the sklearn cross-fit
    # estimator, so retain the legacy LOO radius only under a non-promotable tag.
    eps, dec = _kb.decide_from_records(Bhat, B, alpha=alpha, calibration="loo")
    return Bhat, eps, np.asarray(dec)


def _minimum_total_for_exact_rank(alpha):
    """Historical LOO minimum; retained for API compatibility in old tests."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}")
    return int(math.ceil((1.0 / float(alpha)) - 1e-12))


def decide_benefit(
    Z,
    B,
    alpha=ALPHA,
    prefer="auto",
    allow_diagnostic_fallback=False,
    sample_ids=None,
):
    """Single-candidate KGA benefit certificate over a (method, seed) cell.

    Returns (Bhat: np.ndarray, eps: np.ndarray, dec: np.ndarray[str], backend: str).
    The backend is the diagnostic ``sklearn_gradient_boost_crossfit_split`` label, an explicit
    ``numpy_knn_fallback[_diagnostic]`` label, or an ``infeasible_*`` reason when
    the certificate cannot be formed and every decision is forced to ABSTAIN.

    prefer="auto" uses sklearn. It falls back only for an import/dependency
    failure and only when ``allow_diagnostic_fallback=True``; scientific/model
    exceptions are never swallowed.
    prefer="numpy" forces the fallback (used by the CPU verification harness).
    prefer="sklearn" forces the production path (raises if sklearn missing).
    """
    Z = np.asarray(Z, float); B = np.asarray(B, float)
    minimum_total = _kb.minimum_crossfit_size(alpha)
    if len(B) < minimum_total or len(np.unique(B)) < 2:
        # No finite exact-rank radius exists below the minimum total cell count;
        # a constant-benefit pool is also non-identifiable for the estimator.
        Bhat = np.full(len(B), float(np.mean(B)) if len(B) else 0.0)
        reason = (
            "infeasible_undersized_exact_rank"
            if len(B) < minimum_total
            else "infeasible_degenerate_benefit"
        )
        return Bhat, np.full(len(B), np.inf), np.full(len(B), "ABSTAIN"), reason
    if prefer == "numpy":
        bh, eps, dec = _decide_kga_numpy(Z, B, alpha)
        return bh, eps, dec, "numpy_knn_fallback"
    if prefer == "sklearn":
        bh, eps, dec = _decide_kga_sklearn(Z, B, alpha, sample_ids=sample_ids)
        return bh, eps, dec, "sklearn_gradient_boost_crossfit_split"
    # auto
    try:
        bh, eps, dec = _decide_kga_sklearn(Z, B, alpha, sample_ids=sample_ids)
        return bh, eps, dec, "sklearn_gradient_boost_crossfit_split"
    except (ImportError, ModuleNotFoundError):
        if not allow_diagnostic_fallback:
            raise
        bh, eps, dec = _decide_kga_numpy(Z, B, alpha)
        return bh, eps, dec, "numpy_knn_fallback_diagnostic"


# --------------------------------------------------------------------------- #
# per-condition record assembly + file writing                                #
# --------------------------------------------------------------------------- #
def _condition_key_camelyon(r):
    """Stable, order-preserving, SEED-INDEPENDENT condition string for Camelyon17.

    The seed is a file-level field (one per_condition_*_seed<S>.json per seed); the
    multi-seed paired-CI analysis pairs conditions ACROSS seeds, so the condition key
    must be identical for the same cell across seeds and therefore must NOT embed the
    seed (mirrors the stress_grid contract, where `condition` carries severity/regime
    but not the seed).

    `mode` (online/episodic) IS included: the WILDS runners pool {online,episodic}
    under one method, so without it the two would collide to the same key and the
    paired-CI condition arrays would be ambiguous. Each (cell, mode) is a distinct
    condition; this keeps every key unique within a (method, seed) file."""
    mode = r.get("mode", "")
    base = f"{r['domain']}|{r['comp']}|{r['regime']}|{r['aggr']}"
    return f"{base}|{mode}" if mode else base


def _condition_key_imagenetr(r):
    mode = r.get("mode", "")
    base = f"{r.get('domain', 'imagenet_r')}|{r['comp']}|{r['regime']}|{r['aggr']}"
    return f"{base}|{mode}" if mode else base


def _condition_key_officehome(r):
    """Seed-independent Office-Home cell key (domain × split × composition × regime)."""
    split = r.get("split") or "test"
    return f"{r['domain']}|{split}|{r['comp']}|{r['regime']}"


def _condition_key_iwildcam(r):
    """Seed-independent iWildCam cell key (location/domain × grid axes × mode)."""
    mode = r.get("mode", "")
    loc = r.get("location", r.get("domain", "loc"))
    base = f"{loc}|{r['comp']}|{r['regime']}|{r.get('aggr', '')}"
    return f"{base}|{mode}" if mode else base


def _condition_key_rxrx1(r):
    """Seed-independent RxRx1 cell key (composition × regime × aggressiveness × mode)."""
    mode = r.get("mode", "")
    base = f"{r.get('domain', 'rxrx1')}|{r['comp']}|{r['regime']}|{r.get('aggr', '')}"
    return f"{base}|{mode}" if mode else base


CONDITION_KEYS = {
    "camelyon17": _condition_key_camelyon,
    "imagenet-r": _condition_key_imagenetr,
    "imagenet_r": _condition_key_imagenetr,
    "officehome": _condition_key_officehome,
    "office-home": _condition_key_officehome,
    "iwildcam": _condition_key_iwildcam,
    "wilds-iwildcam": _condition_key_iwildcam,
    "rxrx1": _condition_key_rxrx1,
    "wilds-rxrx1": _condition_key_rxrx1,
}


def build_per_condition_records(records, method, seed, dataset, alpha=ALPHA,
                                z_names=None, prefer="auto", condition_key_fn=None,
                                method_field="method", allow_diagnostic_fallback=False):
    """Build the per-condition record list for one (method, seed) cell.

    `records` is the runner's flat records[] list (each is a dict with keys
    seed, method, comp, regime, aggr, a0, aa, B, Z, regime_label, ...).
    `method_field` selects which record field names the "method" axis (default
    "method"; the ImageNet-R diverse-backbone panel passes "candidate" so each
    frozen backbone becomes its own per-condition file).
    Returns (per_cond_list, backend_label).
    """
    if condition_key_fn is None:
        condition_key_fn = CONDITION_KEYS.get(dataset, _condition_key_imagenetr)
    z_names = z_names or EVIDENCE_NAMES
    rs = [r for r in records if r.get(method_field) == method and int(r.get("seed", -1)) == int(seed)]
    # A repeated scientific cell is never an extra replication.  In particular,
    # concatenating two copies of the same monolithic result used to leave an
    # identical twin in every LOO training pool, making the held-out prediction
    # nearly exact.  Refuse the input before any labels reach the estimator.
    condition_keys = [condition_key_fn(r) for r in rs]
    duplicates = sorted(key for key, count in Counter(condition_keys).items() if count > 1)
    if duplicates:
        preview = ", ".join(repr(key) for key in duplicates[:5])
        raise ValueError(
            "duplicate scientific condition key(s) for "
            f"dataset={dataset!r}, {method_field}={method!r}, seed={int(seed)}: "
            f"{preview}. Duplicate rows/runs are not independent evidence."
        )
    # preserve runner order (this defines the cross-seed condition order the locked
    # analysis asserts must match across seeds)
    Z = np.array([r["Z"] for r in rs], float) if rs else np.zeros((0, len(z_names)))
    B = np.array([r["B"] for r in rs], float) if rs else np.zeros((0,))
    Bhat, eps, dec, backend = decide_benefit(
        Z,
        B,
        alpha=alpha,
        prefer=prefer,
        allow_diagnostic_fallback=allow_diagnostic_fallback,
        sample_ids=condition_keys,
    )
    # fix-queue item 4: eps is now ONE RADIUS PER CELL (the scored cell is excluded
    # from its own calibration pool), so `eps_conformal` is serialised per record
    # rather than the single file-level scalar the old code broadcast.
    eps_vec = np.broadcast_to(np.asarray(eps, float), (len(rs),)) if len(rs) else np.zeros(0)
    per_cond = []
    for i, r in enumerate(rs):
        b_hat_i = float(Bhat[i])
        eps_raw_i = float(eps_vec[i])
        if np.isnan(eps_raw_i) or eps_raw_i < 0:
            raise ValueError(f"invalid conformal radius {eps_raw_i!r} for index {i}")
        calibration_feasible_i = bool(np.isfinite(eps_raw_i))
        if calibration_feasible_i:
            eps_i = eps_raw_i
            lb_i = b_hat_i - eps_i
            ub_i = b_hat_i + eps_i
            benefit_ci_i = [float(lb_i), float(ub_i)]
            gamma_ci_i = [float(0.5 * lb_i), float(0.5 * ub_i)]
            radius_status_i = "FINITE"
            if lb_i > 0:
                zone_i = "CERTIFIED_ADAPT"
            elif ub_i < 0:
                zone_i = "CERTIFIED_FREEZE"
            else:
                zone_i = "BLIND"
        else:
            # Exact-rank split conformal is mathematically infeasible when the
            # residual pool is too small for the requested alpha.  JSON has no
            # portable Infinity value: represent the absent radius/interval as
            # null and preserve the reason in explicit status fields.
            eps_i = None
            benefit_ci_i = None
            gamma_ci_i = None
            radius_status_i = "INFEASIBLE"
            zone_i = "INFEASIBLE"
            if str(dec[i]).upper() != "ABSTAIN":
                raise ValueError(
                    "non-finite conformal radius must force ABSTAIN; "
                    f"got decision={dec[i]!r} for index {i}"
                )
        a0_i = float(r["a0"]); aa_i = float(r["aa"])
        kga_acc_i = aa_i if dec[i] == "ADAPT" else a0_i
        per_cond.append({
            "seed": int(seed), "benchmark": dataset, "method": method,
            "condition": condition_key_fn(r),
            "source_file_sha256": r.get("_source_sha256"),
            "source_record_index": r.get("_source_record_index"),
            "source_record_sha256": r.get("_source_record_sha256"),
            "B": float(r["B"]),
            "a0": a0_i,
            "a_adapted": aa_i,
            "a_kbound": float(kga_acc_i),
            "a_oracle": float(max(a0_i, aa_i)),
            "regime": r.get("regime_label", "marginal"),
            "oracle_action": ("ADAPT" if aa_i > a0_i else "FREEZE"),
            "Z": [float(z) for z in r["Z"]],
            "Z_names": list(z_names),
            # Wave-5 pass-through: the runners now attach the panel agreement
            # matrix + disagreement count (panel_capture.py). None if absent.
            "n_D": r.get("n_D"),
            "c_ij": r.get("c_ij"),
            "Z_ev2": r.get("Z_ev2"),
            "Z_ev2_names": r.get("Z_ev2_names"),
            "tau_hat": 0.0,                  # single-candidate route => degenerate
            "tau_star": 0.0,
            "b_hat": b_hat_i,
            "eps_conformal": eps_i,
            "calibration_feasible": calibration_feasible_i,
            "radius_status": radius_status_i,
            "radius_infeasible_reason": (
                None
                if calibration_feasible_i
                else (
                    "constant/degenerate benefit pool cannot identify a routing boundary"
                    if backend == "infeasible_degenerate_benefit"
                    else "too few cells for disjoint estimator-fit, calibration, and score sets"
                )
            ),
            "benefit_ci": benefit_ci_i,
            "zone": zone_i,
            "gamma_hat": float(0.5 * b_hat_i),
            "gamma_ci": gamma_ci_i,
            "kga_decision": str(dec[i]),
        })
    return per_cond, backend


def _atomic_json_dump(payload, path):
    """Write one JSON document atomically in the destination directory."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".per-condition-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            os.fchmod(handle.fileno(), 0o644)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _match_imagenetr_cell(r, c):
    return (int(r.get("seed", -1)) == int(c["seed"])
            and r.get("comp") == c["comp"]
            and r.get("regime") == c["regime"]
            and r.get("aggr") == c["aggr"])


def build_panel_records(*_args, **_kwargs):
    """Retired: the old panel path selected a candidate with target labels.

    Selecting ``max(aa)`` on the scored target condition and then copying that
    candidate's evidence into a routing artifact is target-label leakage.  Keep
    the symbol only to make old callers fail loudly instead of silently changing
    scientific meaning.
    """

    raise RuntimeError(
        "panel serializer retired: it selected the target-label-best candidate; "
        "use a candidate fixed before scoring and serialize_run instead"
    )


def serialize_panel_run(*_args, **_kwargs):
    """Retired fail-closed alias; see :func:`build_panel_records`."""

    return build_panel_records()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_filename_token(value, field):
    token = str(value)
    if not token or re.fullmatch(r"[A-Za-z0-9_.-]+", token) is None:
        raise ValueError(f"{field} contains an unsafe filename token: {value!r}")
    return token


def serialize_run(records, dataset, out_dir, seeds=None, methods=None, alpha=ALPHA,
                  z_names=None, prefer="auto", condition_key_fn=None, extra_top=None,
                  method_field="method", seed_metadata=None,
                  allow_diagnostic_fallback=False):
    """Write per_condition_<dataset>_<method>_seed<S>.json for every (method, seed) cell
    present in `records`, matching the cifar_tent_mps_v2 / stress_grid serialization
    contract.  `method_field` selects the record field used for the method axis
    (default "method"; ImageNet-R diverse-backbone panel passes "candidate").
    Returns a manifest dict {written: [...], cells: {...}, kga_backend: ...}.
    """
    os.makedirs(out_dir, exist_ok=True)
    if methods is None:
        methods = sorted({r[method_field] for r in records})
    if seeds is None:
        seeds = sorted({int(r["seed"]) for r in records})
    methods = list(methods)
    seeds = [int(seed) for seed in seeds]
    if not methods or not seeds:
        raise ValueError("serializer requires non-empty methods and seeds")
    if len(methods) != len(set(methods)):
        raise ValueError("serializer method list contains duplicates")
    if len(seeds) != len(set(seeds)):
        raise ValueError("serializer seed list contains duplicates")
    observed_methods = {record.get(method_field) for record in records}
    observed_seeds = {int(record["seed"]) for record in records}
    if observed_methods != set(methods) or observed_seeds != set(seeds):
        raise ValueError(
            "records do not match the exact requested method x seed grid; "
            f"methods expected={sorted(map(str, methods))}, observed={sorted(map(str, observed_methods))}; "
            f"seeds expected={sorted(seeds)}, observed={sorted(observed_seeds)}"
        )

    dataset_token = _safe_filename_token(dataset, "dataset")
    method_tokens = {method: _safe_filename_token(method, "method") for method in methods}
    payloads = {}
    cells = {}
    backend_seen = set()
    reference_conditions = None
    for method in methods:
        for seed in seeds:
            per_cond, backend = build_per_condition_records(
                records, method, seed, dataset, alpha=alpha, z_names=z_names,
                prefer=prefer, condition_key_fn=condition_key_fn, method_field=method_field,
                allow_diagnostic_fallback=allow_diagnostic_fallback)
            if not per_cond:
                raise ValueError(
                    "missing method x seed cell: "
                    f"dataset={dataset!r}, {method_field}={method!r}, seed={seed}"
                )
            conditions = [record["condition"] for record in per_cond]
            if reference_conditions is None:
                reference_conditions = conditions
            elif conditions != reference_conditions:
                raise ValueError(
                    "method x seed cells do not have the same ordered scientific-condition grid; "
                    f"first={reference_conditions[:3]!r}, current={conditions[:3]!r}, "
                    f"method={method!r}, seed={seed}"
                )
            backend_seen.add(backend)
            fname = f"per_condition_{dataset_token}_{method_tokens[method]}_seed{seed}.json"
            payload = {
                "seed": int(seed), "benchmark": dataset, "method": method,
                "alpha": float(alpha), "n_conditions": len(per_cond),
                "kga_backend": backend,
                "estimator_publication_eligible": False,
                "estimator_claim_scope": (
                    "development diagnostic; confirmation requires a separately serialized "
                    "validation-locked estimator and unopened test score"
                ),
                "n_calibration_infeasible": sum(
                    not record["calibration_feasible"] for record in per_cond
                ),
                "per_condition_fields_absent": [
                    k for k in ("n_D", "c_ij")
                    if all(pcr.get(k) is None for pcr in per_cond)],
                "per_condition_fields_absent_reason":
                    "n_D/c_ij are attached by the runner via panel_capture.py "
                    "(Wave-5); absent means the runner predates the patch or the "
                    "panel capture was skipped.",
                "per_condition_field_notes": {
                    "tau_hat_tau_star":
                        "single-candidate route (frozen vs adapted), so tau is degenerate "
                        "and set to 0.0.",
                    "zone":
                        "computed from benefit_ci: lb>0 => CERTIFIED_ADAPT, ub<0 => "
                        "CERTIFIED_FREEZE, overlap => BLIND; INFEASIBLE means the "
                        "exact-rank radius does not exist at the requested alpha.",
                    "radius_status":
                        "FINITE carries a numeric eps_conformal; INFEASIBLE carries "
                        "JSON null for eps_conformal/benefit_ci/gamma_ci and forces ABSTAIN.",
                    "gamma_hat":
                        "single-candidate proxy (b_hat/2) with gamma_ci = benefit_ci/2.",
                    "kga_backend":
                        "sklearn_gradient_boost_crossfit_split = label-disjoint development diagnostic; "
                        "numpy_knn_fallback[_diagnostic] = non-publishable verification estimator "
                        "(decision rule + conformal radius identical, estimator differs).",
                },
                "records": per_cond,
            }
            if extra_top:
                reserved = set(payload).intersection(extra_top)
                if reserved:
                    raise ValueError(
                        "extra_top may not overwrite serializer-owned field(s): "
                        + ", ".join(sorted(reserved))
                    )
                payload.update(extra_top)
            if seed_metadata:
                metadata = seed_metadata.get(int(seed), {})
                reserved = set(payload).intersection(metadata)
                if reserved:
                    raise ValueError(
                        "seed_metadata may not overwrite serializer-owned field(s): "
                        + ", ".join(sorted(reserved))
                    )
                payload.update(metadata)
            payloads[fname] = payload
            cells[f"{method}_seed{seed}"] = {"file": fname, "n_conditions": len(per_cond)}

    expected_count = len(methods) * len(seeds)
    if len(payloads) != expected_count:
        raise ValueError(
            f"serializer built {len(payloads)} files for an expected {expected_count}-cell grid"
        )
    prefix = f"per_condition_{dataset_token}_"
    commit_name = f"per_condition_{dataset_token}_manifest.json"
    stale = sorted(
        name for name in os.listdir(out_dir)
        if (name.startswith(prefix) and name.endswith(".json")) or name == commit_name
    )
    if stale:
        raise ValueError(
            "refusing to mix or overwrite a prior serialization generation; "
            f"use a fresh output directory (found {stale[:5]})"
        )

    generation_material = {
        "schema": "kbound_per_condition_generation_v1",
        "dataset": dataset,
        "methods": methods,
        "seeds": seeds,
        "payloads": payloads,
    }
    generation_id = hashlib.sha256(
        json.dumps(
            generation_material,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    for payload in payloads.values():
        payload["serialization_generation_id"] = generation_id

    stage = tempfile.mkdtemp(prefix=".per-condition-generation-", dir=out_dir)
    written = []
    manifest_path = os.path.join(out_dir, commit_name)
    try:
        staged_files = {}
        for fname, payload in payloads.items():
            staged_path = os.path.join(stage, fname)
            _atomic_json_dump(payload, staged_path)
            staged_files[fname] = {
                "sha256": _file_sha256(staged_path),
                "n_conditions": int(payload["n_conditions"]),
            }
        staged_manifest = os.path.join(stage, commit_name)
        _atomic_json_dump(
            {
                "schema": "kbound_per_condition_generation_v1",
                "generation_id": generation_id,
                "generation_committed": True,
                "dataset": dataset,
                "methods": methods,
                "seeds": seeds,
                "expected_cells": expected_count,
                "files": staged_files,
            },
            staged_manifest,
        )
        for fname in payloads:
            destination = os.path.join(out_dir, fname)
            os.replace(os.path.join(stage, fname), destination)
            written.append(destination)
        # The generation manifest is the commit marker and is always published last.
        os.replace(staged_manifest, manifest_path)
        directory_fd = os.open(out_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {
        "written": written,
        "manifest": manifest_path,
        "generation_id": generation_id,
        "cells": cells,
        "kga_backend": sorted(backend_seen),
        "dataset": dataset,
        "methods": list(methods),
        "seeds": [int(s) for s in seeds],
        "out_dir": out_dir,
    }
