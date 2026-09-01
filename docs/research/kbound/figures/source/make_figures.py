"""List maintained and historical figure generators.

The maintained population frontier and Figure 3 interval rule are data-free
illustrations. Other entries below are historical experiment generators, not
instructions to rerun experiments for a manuscript build. This dispatcher prints
the mapping and executes no generator.
"""
MAP = {
    "fig_frontier_schematic.png": "make_submission_figures.py --frontier-only",
    "fig_certificate.png": "plot_kga_interval_rule.py",
    "fig_kbound_harmful.png": "kbound_harmful_regime.py",
    "fig_mixed_policies.png": "mixed_regime_experiment.py",
    "fig_witness_clean.png": "kbound_full_experiments.py",
    "fig_cifar_tent.png": "cifar_tent_mps.py",
    "fig_tta_collapse.png": "tta_collapse_experiment.py",
}
if __name__ == "__main__":
    print("Generator inventory (frontier and interval illustrations are maintained; other entries are historical):")
    for fig, scr in MAP.items():
        print(f"  {fig:28s} <- scripts/{scr}")
