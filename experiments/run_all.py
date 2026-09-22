"""Run every experiment in sequence, reproducing the full set of report figures/tables.

Run as ``python -m experiments.run_all`` from the repo root.
"""

from __future__ import annotations

from experiments import (
    baseline_accuracy,
    commuting_gradient_check,
    entanglement_structure,
    shot_efficiency,
    three_arm_comparison,
    trainability_scaling,
)


def run():
    print("\n" + "#" * 78)
    print("# 1. Baseline accuracy (per-sample SGD vs batch GD)")
    print("#" * 78)
    baseline_accuracy.run()

    print("\n" + "#" * 78)
    print("# 2. Commuting-generator gradient check")
    print("#" * 78)
    commuting_gradient_check.run()

    print("\n" + "#" * 78)
    print("# 3. Shot efficiency (honest comparison)")
    print("#" * 78)
    shot_efficiency.run()

    print("\n" + "#" * 78)
    print("# 4. Three-arm training comparison")
    print("#" * 78)
    three_arm_comparison.run()

    print("\n" + "#" * 78)
    print("# 5. Entanglement structure experiment")
    print("#" * 78)
    entanglement_structure.run()

    print("\n" + "#" * 78)
    print("# 6. Trainability scaling (n=2..14)")
    print("#" * 78)
    trainability_scaling.run()


if __name__ == "__main__":
    run()
