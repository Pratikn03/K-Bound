"""One-touch reproduction plan for the K-Bound paper.

By default prints the ordered pipeline. With --run it executes the numpy-only stages
(environment check, the AD/regression/witness/ablation experiments, manifest rebuild,
result audit). CIFAR/ImageNet TTA stages need a GPU/MPS box and are listed but not
auto-run here (see README_DECISIVE.md and scripts/cifar_tent_mps_v2.py).

Usage:
    python scripts/99_reproduce_kbound.py            # print the plan
    python scripts/99_reproduce_kbound.py --run      # run the CPU/numpy stages
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

# (stage label, script, needs_gpu)
PIPELINE = [
    ("Verify environment",            "00_verify_environment.py",      False),
    ("Clean 123-task suite (E1)",     "knowability_experiment.py",     False),
    ("Harmful fusion regime (E2)",    "kbound_harmful_regime.py",      False),
    ("Mixed regime (E3)",             "mixed_regime_experiment.py",    False),
    ("Witness/regression/ablations/rigor (E4-E7)", "kbound_full_experiments.py", False),
    ("TTA collapse probe (E10)",      "tta_collapse_experiment.py",    False),
    ("CIFAR-10/100-C + Tent/EATA/SAR (E8) [GPU]", "cifar_tent_mps_v2.py", True),
    ("CIFAR online Tent (E9) [GPU]",  "cifar_tent_online.py",          True),
    ("Rebuild result manifest",       "01_build_manifests.py",         False),
    ("Audit all results",             "02_verify_results.py",          False),
    ("Make tables",                   "03_make_tables.py",             False),
    ("Make figures",                  "04_make_figures.py",            False),
]

def main():
    run = "--run" in sys.argv
    print("K-Bound reproduction pipeline\n" + "=" * 60)
    for i, (label, script, gpu) in enumerate(PIPELINE):
        tag = " [needs GPU/MPS]" if gpu else ""
        print(f"{i:2d}. {label}{tag}\n      -> {script}")
    if not run:
        print("\n(Plan only. Re-run with --run to execute the CPU/numpy stages.)")
        return
    for label, script, gpu in PIPELINE:
        if gpu:
            print(f"\n[skip GPU] {label} -> run on your M5: python scripts/{script}")
            continue
        path = os.path.join(HERE, script)
        if not os.path.exists(path):
            print(f"\n[skip] {script} not found"); continue
        print(f"\n>>> {label}: python scripts/{script}")
        subprocess.run([sys.executable, path], check=False)

if __name__ == "__main__":
    main()
