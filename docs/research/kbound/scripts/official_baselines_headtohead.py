#!/usr/bin/env python3
"""
Head-to-head harness: KGA vs POEM / AETTA on the locked CIFAR-10-C stress stream.

The scoring + statistics layer is the reusable part: it takes a per-condition DECISION
sequence from ANY policy and computes regret-to-oracle, FA_u, decisive rate, and the KGA
regret-gap with a paired condition-bootstrap CI (Holm-corrected across the baseline family).

External decisions alone do not close the official-reproduction requirement.
Native outputs require the separate authenticated provenance audit, the same
task-compatible panel, and reviewed conversion before promotion. Unauthenticated
decision maps may be scored only with an explicit unverified label:

    python3 official_baselines_headtohead.py \
        --decisions poem=/path/poem_decisions.json aetta=/path/aetta_decisions.json

where each JSON maps  {condition_string: "adapt"|"freeze"|"abstain"}  over the SAME 432
conditions (same order as the logged stream).  Without --decisions, the harness falls back to
protocol-matched *ports* (clearly labelled) so the pipeline is runnable end-to-end today.

No fabrication: if an external decisions file is missing a condition, it errors out.

For a completed saved-row checkpoint, use an explicit candidate, independently
recorded expected input digest, and a new output directory:

    python3 official_baselines_headtohead.py --candidate sar \
        --checkpoint /path/checkpoint.json --checkpoint-sha256 EXPECTED_SHA256 \
        --out-dir /path/new-replay

Checkpoint mode is a retrospective saved-feature diagnostic. It records exact
paired-oracle metrics and provenance without bootstrap significance claims.
"""
import argparse, json, os, hashlib, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for import_root in (REPO, HERE):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
from kga.crossfit import controlled_grid_crossfit
from official_baseline_provenance import OFFICIAL_LABEL, decisions_sha256, validate_promotable_audit
RES = os.path.join(REPO, "experiments", "kbound", "results")
GBR = dict(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
Zi = dict(pre_entropy=0, pre_conf=1, post_entropy=3, post_conf=4, entropy_drop=7)
# Match the absolute outcome-arithmetic tolerance in kga.experiment_contract.
STREAM_OUTCOME_ABS_TOL = 1e-10

def load(cand, stream_path=None):
    """Parse/hash one byte snapshot and validate the saved legacy outcome fields."""
    f = stream_path or os.path.join(RES, f"per_condition_cifar10c_{cand}_seed0.json")
    try:
        with open(f, "rb") as stream:
            snapshot = stream.read()
        digest = hashlib.sha256(snapshot).hexdigest()
        raw = json.loads(snapshot, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        records = raw.get("records") if isinstance(raw, dict) else None
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a nonempty list")

        def finite_number(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

        conditions, features, frozen, adapted = [], [], [], []
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("records must be objects")
            condition = row.get("condition")
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError("condition IDs must be nonempty strings")
            z = row.get("Z")
            if not isinstance(z, list) or len(z) != 11 or not all(finite_number(value) for value in z):
                raise ValueError("Z must contain exactly 11 finite numeric features")
            a0, aa, oracle = row.get("a0"), row.get("a_adapted"), row.get("a_oracle")
            if not all(finite_number(value) and 0 <= value <= 1 for value in (a0, aa, oracle)):
                raise ValueError("a0/a_adapted/a_oracle must be finite numbers in [0, 1]")
            benefit = row.get("B")
            if not finite_number(benefit) or not math.isclose(
                benefit, aa - a0, rel_tol=0.0, abs_tol=STREAM_OUTCOME_ABS_TOL
            ):
                raise ValueError("B must equal a_adapted - a0 within absolute tolerance 1e-10")
            if not math.isclose(oracle, max(a0, aa), rel_tol=0.0, abs_tol=STREAM_OUTCOME_ABS_TOL):
                raise ValueError("a_oracle must equal max(a0, a_adapted) within absolute tolerance 1e-10")
            conditions.append(condition); features.append(z); frozen.append(a0); adapted.append(aa)
        if len(set(conditions)) != len(conditions):
            raise ValueError("condition IDs must be unique")
    except (OSError, UnicodeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"invalid legacy stream {f}: {exc}") from exc
    Z = np.asarray(features, dtype=float)
    a0 = np.asarray(frozen, dtype=float); aa = np.asarray(adapted, dtype=float)
    return conditions, Z, aa - a0, a0, aa, np.maximum(a0, aa), digest

# ---- policies: return a per-condition array of 'adapt'/'freeze'/'abstain' ----
def always(v, n): return np.array([v]*n, dtype=object)

def kga_exact_rank(Z, B, alpha=0.10, k=8, *, sample_ids=None, return_details=False):
    """Replay with disjoint fit/calibration/score folds and fixed zero boundary.

    This replaces the historical pooled-residual construction. Existing saved
    results retain their historical status and must not be relabeled repaired.
    """
    ids = sample_ids if sample_ids is not None else [f"row-{i}" for i in range(len(B))]
    result = controlled_grid_crossfit(Z, B, sample_ids=ids, alpha=alpha, n_folds=k, **GBR)
    if return_details:
        return result
    return np.asarray([str(action).lower() for action in result.action], dtype=object)

def poem_port(Z):   # POEM-style committal gate (port): commit while adaptation lowers entropy
    d = np.where(Z[:,Zi["entropy_drop"]] > 0, "adapt", "freeze"); return d.astype(object)
def aetta_port(Z):  # AETTA-style accuracy-proxy gate (port): adapt if adapted confidence rose
    d = np.where(Z[:,Zi["post_conf"]] > Z[:,Zi["pre_conf"]], "adapt", "freeze"); return d.astype(object)

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def _invalid_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")

def load_checkpoint(path, candidate, expected_sha256):
    """Authenticate and validate saved rows; derive outcomes from a0/aa only."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_sha256
    ):
        raise SystemExit("checkpoint-sha256 must be an explicit 64-character SHA-256 digest")
    try:
        with open(path, "rb") as stream:
            payload = stream.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_sha256.lower():
            raise SystemExit("checkpoint SHA-256 mismatch; no replay was written")
        raw = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        if not isinstance(raw, dict) or not isinstance(raw.get("rows"), dict):
            raise ValueError("checkpoint must contain a rows object")
        records = raw["rows"].get(candidate)
        if not isinstance(records, list) or not records:
            raise ValueError(f"checkpoint has no nonempty rows for candidate {candidate!r}")

        def finite_number(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

        conditions, features, frozen, adapted = [], [], [], []
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("checkpoint rows must be objects")
            condition = row.get("condition")
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError("checkpoint condition IDs must be nonempty strings")
            z = row.get("Z")
            if not isinstance(z, list) or len(z) != 11 or not all(finite_number(value) for value in z):
                raise ValueError("checkpoint Z must contain 11 finite numeric features")
            a0, aa = row.get("a0"), row.get("aa")
            if not all(finite_number(value) and 0 <= value <= 1 for value in (a0, aa)):
                raise ValueError("checkpoint accuracy a0/aa must be finite numbers in [0, 1]")
            conditions.append(condition); features.append(z); frozen.append(a0); adapted.append(aa)
        if len(set(conditions)) != len(conditions):
            raise ValueError("checkpoint condition IDs must be unique")
        if "cells_done" in raw or "cells_total" in raw:
            if any(type(raw.get(key)) is not int or raw[key] != len(records)
                   for key in ("cells_done", "cells_total")):
                raise ValueError("checkpoint completion counts do not match the complete candidate panel")
        if "done" in raw:
            done = raw["done"]
            if (not isinstance(done, list) or not all(isinstance(value, str) for value in done)
                    or len(done) != len(conditions) or set(done) != set(conditions)):
                raise ValueError("checkpoint done IDs do not match the complete candidate panel")
    except (OSError, UnicodeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"invalid checkpoint {path}: {exc}") from exc
    Z = np.asarray(features, dtype=float)
    a0 = np.asarray(frozen, dtype=float); aa = np.asarray(adapted, dtype=float)
    metadata = {key: raw[key] for key in ("sar_bn_protocol", "cells_done", "cells_total", "updated") if key in raw}
    return conditions, Z, aa - a0, a0, aa, np.maximum(a0, aa), digest, metadata

def load_external(path, cond, provenance_audit_path=None, stream_sha256=None):
    try:
        with open(path, encoding="utf-8") as stream:
            raw = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_constant,
            )
    except (OSError, json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"invalid external decisions JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"external decisions must be a JSON object: {path}")
    if "decisions" in raw:
        m = raw["decisions"]
        official = raw.get("official_label_allowed", False)
        if not isinstance(official, bool):
            raise SystemExit(f"official_label_allowed must be boolean in {path}")
        label = raw.get("label", "external_protocol_adapter_unverified")
    else:
        m = raw
        official = False
        label = "legacy_external_decisions_unverified"
    if not isinstance(m, dict):
        raise SystemExit(f"external decisions payload is not an object: {path}")
    miss = [c for c in cond if c not in m]
    if miss:
        raise SystemExit(f"external decisions missing {len(miss)} conditions e.g. {miss[:2]} in {path}")
    extra = sorted(set(m) - set(cond))
    if extra:
        raise SystemExit(f"external decisions contain {len(extra)} out-of-stream conditions e.g. {extra[:2]}")
    values = np.array([m[c] for c in cond], dtype=object)
    bad = sorted(set(values) - {"adapt", "freeze", "abstain"})
    if bad:
        raise SystemExit(f"external decisions contain invalid actions: {bad}")
    if official:
        if provenance_audit_path is None:
            raise SystemExit(f"official external decisions require a separate provenance audit: {path}")
        if raw.get("schema_version") != 3:
            raise SystemExit(f"official external decisions require schema_version 3: {path}")
        if label != OFFICIAL_LABEL:
            raise SystemExit(f"official external decisions use an invalid label: {path}")
        wrapped_stream_sha = raw.get("locked_stream_sha256")
        if stream_sha256 is not None and wrapped_stream_sha != stream_sha256:
            raise SystemExit(f"official external decisions do not bind the loaded stream: {path}")
        if raw.get("decisions_sha256") != decisions_sha256(m):
            raise SystemExit(f"official external decisions payload hash mismatch: {path}")
        try:
            audit_sha = validate_promotable_audit(
                provenance_audit_path,
                method=str(raw.get("method", "")),
                decisions=m,
                source_log_sha256=str(raw.get("source_log_sha256", "")),
                locked_stream_sha256=str(wrapped_stream_sha or ""),
                environment_receipt_sha256=str(raw.get("environment_receipt_sha256", "")),
                toolchain_receipt_sha256=str(raw.get("toolchain_receipt_sha256", "")),
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"official provenance audit validation failed for {path}: {exc}") from exc
        if raw.get("provenance_audit_sha256") != audit_sha:
            raise SystemExit(f"official provenance audit hash mismatch: {path}")
    return values, official, label

# ---- scoring ----
def regret_pc(dec, a0, aa, ao): return ao - np.where(dec=="adapt", aa, a0)
def summ(dec, B, a0, aa, ao):
    adapt = dec=="adapt"
    return dict(regret=round(float(regret_pc(dec,a0,aa,ao).mean()),4),
                FA_u=round(float(np.mean(adapt & (B<=0))),4),
                decisive=round(float(np.mean(dec!="abstain")),3),
                adapt_rate=round(float(adapt.mean()),3))

def exact_metrics(dec, B, a0, aa, ao):
    """Full-precision paired-outcome metrics for an explicit checkpoint replay."""
    adapt = dec == "adapt"
    selected = np.where(adapt, aa, a0)
    false_adapt = adapt & (B <= 0)
    return dict(mean_accuracy=float(selected.mean()), regret_exact=float((ao - selected).mean()),
                worst_accuracy=float(selected.min()), n_false_adapt=int(false_adapt.sum()),
                FA_u_exact=float(false_adapt.mean()),
                FA_c=float(np.mean(B[adapt] <= 0)) if adapt.any() else None,
                decision_counts={action: int(np.sum(dec == action)) for action in ("adapt", "freeze", "abstain")})

def checkpoint_records(cond, B, a0, aa, ao, gate, point):
    """Serialize the one shared predictor and both selected-outcome paths."""
    def finite(value):
        return float(value) if math.isfinite(float(value)) else None

    records = []
    for i, condition in enumerate(cond):
        action = str(gate.action[i]).lower()
        point_acc = float(aa[i] if point[i] == "adapt" else a0[i])
        kga_acc = float(aa[i] if action == "adapt" else a0[i])
        records.append(dict(condition=condition, a0=float(a0[i]), aa=float(aa[i]), B=float(B[i]),
                            a_oracle=float(ao[i]), prediction=finite(gate.prediction[i]),
                            radius=finite(gate.radius[i]),
                            radius_status="finite" if np.isfinite(gate.radius[i]) else "positive_infinity",
                            lower=finite(gate.prediction[i] - gate.radius[i]),
                            upper=finite(gate.prediction[i] + gate.radius[i]), threshold=0.0,
                            point_action=str(point[i]), KGA_action=action,
                            point_accuracy=point_acc, KGA_accuracy=kga_acc,
                            point_regret=float(ao[i] - point_acc), KGA_regret=float(ao[i] - kga_acc)))
    return records

def paired_boot(rk, rb, nb=5000, seed=0):   # gap = baseline_regret - kga_regret (positive => KGA better)
    rng=np.random.default_rng(seed); n=len(rk); gaps=np.empty(nb)
    for i in range(nb):
        idx=rng.integers(0,n,n); gaps[i]=rb[idx].mean()-rk[idx].mean()
    lo,hi=np.percentile(gaps,[2.5,97.5])
    return dict(gap=round(float((rb-rk).mean()),4), ci95=[round(float(lo),4),round(float(hi),4)],
                p_better=round(float(np.mean(gaps>0)),4), ci_excludes_zero=bool(lo>0))

def holm(pvals):   # returns reject/keep at 0.05 family-wise
    order=sorted(range(len(pvals)), key=lambda i:pvals[i]); m=len(pvals); rej=[False]*m
    for rank,i in enumerate(order):
        if pvals[i] <= 0.05/(m-rank): rej[i]=True
        else: break
    return rej

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate", default="tent")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--decisions", nargs="*", default=[], help="name=path.json; external decisions are unverified without authenticated provenance")
    ap.add_argument("--provenance-audits", nargs="*", default=[], help="name=audit.json for authenticated external decisions")
    ap.add_argument("--out", default=None, help="historical CIFAR output path")
    ap.add_argument("--stream", help="explicit locked CIFAR record stream; cannot be combined with --checkpoint")
    ap.add_argument("--checkpoint", help="explicit saved-row checkpoint; no dataset or model execution")
    ap.add_argument("--checkpoint-sha256", help="required expected SHA-256 for --checkpoint")
    ap.add_argument("--out-dir", help="previously absent directory required with --checkpoint")
    a=ap.parse_args()
    checkpoint_metadata = None
    if a.checkpoint:
        if a.stream:
            raise SystemExit("--stream cannot be combined with --checkpoint")
        if not a.checkpoint_sha256:
            raise SystemExit("--checkpoint requires --checkpoint-sha256")
        if not a.out_dir or a.out is not None:
            raise SystemExit("--checkpoint requires --out-dir and cannot be combined with --out")
        if os.path.lexists(a.out_dir):
            raise SystemExit("checkpoint replay requires a fresh --out-dir; path already exists")
        cond,Z,B,a0,aa,ao,stream_sha,checkpoint_metadata = load_checkpoint(
            a.checkpoint, a.candidate, a.checkpoint_sha256)
        stream_path = os.path.abspath(a.checkpoint)
        sha = stream_sha[:12]
    else:
        if a.checkpoint_sha256 or a.out_dir:
            raise SystemExit("--checkpoint-sha256 and --out-dir require --checkpoint")
        stream_path = os.path.abspath(a.stream) if a.stream else os.path.join(
            RES, f"per_condition_cifar10c_{a.candidate}_seed0.json")
        cond,Z,B,a0,aa,ao,stream_sha = load(a.candidate, stream_path=stream_path)
        sha = stream_sha[:12]
        a.out = a.out or os.path.join(RES,"official_headtohead.json")
    n=len(B)
    audits = dict(kv.split("=",1) for kv in a.provenance_audits)
    ext = {kv.split("=",1)[0]: load_external(kv.split("=",1)[1], cond,
           provenance_audit_path=audits.get(kv.split("=",1)[0]), stream_sha256=stream_sha) for kv in a.decisions}
    if set(ext) - {"poem", "aetta"} or set(audits) - set(ext):
        raise SystemExit("unknown external method or unmatched provenance audit")

    gate = kga_exact_rank(Z,B,a.alpha,sample_ids=cond,return_details=True)
    kga = np.asarray([str(action).lower() for action in gate.action],dtype=object)
    rk = regret_pc(kga,a0,aa,ao)
    policies = {"always_adapt":always("adapt",n), "always_freeze":always("freeze",n),
                "oracle":np.where(B>0,"adapt","freeze").astype(object)}
    # The point gate shares KGA's exact fitted predictions, not another fit.
    policies["benefit_regression"] = np.where(
        np.isfinite(gate.prediction),
        np.where(gate.prediction > 0, "adapt", "freeze"), "abstain").astype(object)
    # official if provided, else labelled ports
    for method, fallback in (("poem", poem_port), ("aetta", aetta_port)):
        if method in ext:
            decisions, official, label = ext[method]
            policies[method.upper() + ("_official" if official else "_external_unverified")] = decisions
        else:
            policies[method.upper() + "_port"] = fallback(Z)

    if a.checkpoint:
        rows={name: {**summ(actions,B,a0,aa,ao), **exact_metrics(actions,B,a0,aa,ao)}
              for name, actions in {"KGA": kga, **policies}.items()}
    else:
        rows={"KGA":{**summ(kga,B,a0,aa,ao),"gap_vs_KGA":"---","holm_beats":"---"}}
        compare=[k for k in policies if k not in ("oracle",)]
        boots={k:paired_boot(rk, regret_pc(policies[k],a0,aa,ao)) for k in compare}
        rej = holm([1-boots[k]["p_better"] for k in compare])  # crude p ~ 1-p_better
        for k in policies:
            s=summ(policies[k],B,a0,aa,ao)
            s["holm_beats"]= (dict(zip(compare,rej)).get(k, False)) if k in compare else "---"
            s["gap_vs_KGA"]= boots[k] if k in compare else "---"
            rows[k]=s
    out=dict(candidate=a.candidate, alpha=a.alpha, n_conditions=n, input_sha12=sha,
             official=[k for k in ("poem","aetta") if k in ext and ext[k][1]],
             input_sha256=stream_sha,
             note="External files are not official evidence without authenticated provenance; fallback policies are local ports.",
             gate=gate.to_dict(), decision_threshold=0.0,
             evaluation_scope="retrospective_saved_feature_cell_outcome_disjoint",
             conditions=cond,
             policy_actions={"KGA":kga.tolist(), **{name:actions.tolist() for name,actions in policies.items()}},
             rows=rows)
    if a.checkpoint:
        with open(stream_path, "rb") as stream:
            final_sha = hashlib.sha256(stream.read()).hexdigest()
        if final_sha != stream_sha:
            raise SystemExit("checkpoint changed during replay; no output was written")
        with open(__file__, "rb") as stream:
            wrapper_sha = hashlib.sha256(stream.read()).hexdigest()
        receipt = dict(schema="kbound-checkpoint-replay-receipt-v1", scope=out["evaluation_scope"],
                       source_checkpoint=stream_path, candidate=a.candidate,
                       expected_input_sha256=a.checkpoint_sha256.lower(),
                       input_sha256_before=stream_sha, input_sha256_after=final_sha,
                       checkpoint_metadata=checkpoint_metadata, python_executable=sys.executable,
                       implementation_sha256={"wrapper": wrapper_sha, **gate.protocol["implementation_sha256"]},
                       software_versions=gate.protocol["software_versions"], argv=sys.argv,
                       limitations="Opened, dependent saved-feature diagnostic; no official/native, prospective, exchangeability, or environment-held-out claim.")
        out.update(source_checkpoint=stream_path, source_receipt="replay_receipt.json",
                   statistical_inference="not_computed_for_retrospective_checkpoint_replay",
                   oracle_mean_accuracy=float(ao.mean()),
                   per_condition=checkpoint_records(cond,B,a0,aa,ao,gate,policies["benefit_regression"]))
        # Serialize before creating anything, then reserve a directory exclusively.
        result_json = json.dumps(out, indent=2, allow_nan=False) + "\n"
        receipt_json = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        try:
            os.makedirs(a.out_dir, exist_ok=False)
        except FileExistsError as exc:
            raise SystemExit("checkpoint replay requires a fresh --out-dir; path already exists") from exc
        for filename, payload in (("headtohead.json", result_json), ("replay_receipt.json", receipt_json)):
            with open(os.path.join(a.out_dir, filename), "x", encoding="utf-8") as stream:
                stream.write(payload)
        print("wrote", a.out_dir)
        print(json.dumps(rows, indent=2, allow_nan=False))
        return
    with open(stream_path, "rb") as stream:
        final_sha = hashlib.sha256(stream.read()).hexdigest()
    if final_sha != stream_sha:
        raise SystemExit("selected stream changed during scoring; no output was written")
    json.dump(out, open(a.out,"w"), indent=2); print("wrote", a.out)
    print(f"\n{'policy':16s} {'regret':>8s} {'FA_u':>6s} {'decisive':>8s} {'gap[CI]':>22s} holm")
    for k,s in rows.items():
        g=s["gap_vs_KGA"]
        gs = f"{g['gap']:.4f} {g['ci95']}" if isinstance(g,dict) else "---"
        dec = str(s.get('decisive',''))
        print(f"{k:16s} {s['regret']:>8.4f} {s['FA_u']:>6.3f} {dec:>8s} {gs:>24s} {s['holm_beats']}")

if __name__=="__main__": main()
