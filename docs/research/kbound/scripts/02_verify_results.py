"""Audit every K-Bound result file: confirm it exists and print its headline numbers.

Run:  python scripts/02_verify_results.py
Reads the organized results/ groups and the experiment_registry.csv, prints the key
metric for each experiment, and flags anything marked verify_before_claim.
Pure-stdlib (json/csv/os) — runs anywhere.
"""
import json, os, csv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # docs/research/kbound
R = os.path.join(HERE, "results")

# result file -> list of dotted keys to surface
HEADLINES = {
    "main/knowability_results.json": ["metrics.adapt_precision_(B>0|ADAPT)", "metrics.coverage",
                                      "policy.mean_auc_always_adapt", "policy.mean_auc_trichotomy"],
    "main/kbound_harmful_results.json": ["mean_auc.K-Bound_trichotomy", "mean_auc.always_adapt(elara_fuse)",
                                         "regret_vs_oracle.K-Bound", "regret_vs_oracle.always_adapt"],
    "main/mixed_regime_results.json": ["mean_auc_policies.K_Bound", "mean_auc_policies.always_freeze",
                                       "mean_auc_policies.always_adapt", "safety.adapt_precision_B>0"],
    "main/rigor_multiseed.json": ["paired_ttest_KBound_vs_always_freeze.p",
                                  "paired_ttest_KBound_vs_always_adapt.p"],
    "ablations/ablations.json": ["baseline.regret", "evidence_drop.disagree.regret"],
    "regression/regression_covariate.json": ["mean_MSE.K_Bound", "mean_MSE.oracle", "mean_MSE.always_adapt(IW)"],
    "witness/witness_clean.json": ["all_Z_features_p>0.05", "abstain_rate", "forced_commit_regret_vs_oracle"],
    "tta/cifar_tent_results.json": ["base_rate_harmful_B<0", "regret_vs_oracle.K_Bound",
                                    "regret_vs_oracle.always_adapt"],
    "tta/tta_collapse_results.json": ["regret_vs_oracle.K_Bound", "regret_vs_oracle.always_freeze"],
}

def get(d, dotted):
    cur = d
    for k in dotted.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return "—"
    return cur

def main():
    print("=" * 72)
    print("K-Bound result audit")
    print("=" * 72)
    missing = 0
    for rel, keys in HEADLINES.items():
        path = os.path.join(R, rel)
        if not os.path.exists(path):
            print(f"[MISSING] {rel}"); missing += 1; continue
        d = json.load(open(path))
        print(f"\n• {rel}")
        for k in keys:
            print(f"    {k} = {get(d, k)}")

    # registry claim-status flags
    reg = os.path.join(HERE, "experiments", "experiment_registry.csv")
    if os.path.exists(reg):
        print("\n" + "-" * 72)
        print("Claim-status flags (from experiment_registry.csv):")
        for row in csv.DictReader(open(reg)):
            flag = row.get("claim_status", "")
            if flag and flag != "used":
                print(f"    [{flag.upper()}] {row['name']} -> {row['result_file']}")
    print("\nDone." + (f"  ({missing} files missing)" if missing else "  All result files present."))

if __name__ == "__main__":
    main()
