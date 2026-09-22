"""Figure generation for the report.

Produces the 7 canonical figures used in the report (down from the original
notebook's 9): the fudged ``ratio_plot.png`` (idealized, never-real shot-cost
model) is dropped entirely, and ``honest_comparison.png`` is retired as a
duplicate name for what ``scaling_plot.png`` now shows -- see the README for
details on both changes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_shot_efficiency(
    cg_history,
    ps_history,
    n_qubits: int,
    out_path: str | Path = "figures/scaling_plot.png",
) -> Path:
    """Plot test accuracy vs cumulative shots for the commuting-gen and parameter-shift methods.

    This is the HONEST comparison (two genuinely different gradient methods,
    real per-method shot costs) -- it replaces the original notebook's
    ``scaling_plot.png``, which was built from a fudged/idealized cost model
    that never computed a second gradient method at all. See the README's
    "Bug found during the port" note.
    """
    cg_x, cg_y = zip(*cg_history)
    ps_x, ps_y = zip(*ps_history)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ps_x, ps_y, "C0-o", ms=4, label=f"Parameter-shift  ({2 * n_qubits}xM shots/epoch)")
    ax.plot(cg_x, cg_y, "C1-s", ms=4, label="Commuting-gen    (1xM shots/epoch)")
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative shots (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Same gradient, same learning — commuting-gen uses {2 * n_qubits}x fewer shots")
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_loss_curves_3arm(
    losses_a,
    losses_b,
    losses_c,
    n_qubits: int,
    out_path: str | Path = "figures/loss_curves_3arm.png",
) -> Path:
    """Plot test-MSE loss curves for the three training arms (cell 17)."""
    n_epochs = len(losses_a)
    epochs = range(1, n_epochs + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, losses_a, "C1-", lw=2, label="A: Commuting-gen  (our method, 1×M shots/sample)")
    ax.plot(epochs, losses_b, "C0--", lw=2, label=f"B: Parameter-shift  ({2 * n_qubits}×M shots/sample)")
    ax.plot(epochs, losses_c, "C2:", lw=2, label="C: PennyLane GradientDescentOptimizer  (autograd)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE loss")
    ax.set_title(f"Loss curves — three training arms, {n_epochs} epochs, same circuit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_trainability(
    trainability_results,
    x_val: float,
    n_list: list[int],
    out_path: str | Path = "figures/trainability_plot.png",
) -> Path:
    """Plot empirical vs theoretical gradient variance against qubit count, log-log.

    ``trainability_results`` is a list of ``(n, empirical_var, theory_var, ratio)``.
    """
    import numpy as np

    ns_emp = [r[0] for r in trainability_results]
    var_emp = [r[1] for r in trainability_results]

    n_fine = np.linspace(min(n_list), max(n_list), 200)
    theory_curve = np.sin(x_val) ** 2 / (2 * n_fine**2)

    theory_at_n2 = np.sin(x_val) ** 2 / (2 * 2**2)
    generic_scale = theory_at_n2 * (2**2)
    generic_curve = generic_scale * (2.0 ** (-n_fine))

    fig = plt.figure(figsize=(8, 5))
    plt.plot(ns_emp, var_emp, "o", color="#2980b9", markersize=8, label="empirical (autograd, 500 samples)")
    plt.plot(n_fine, theory_curve, "-", color="#2980b9", linewidth=2, label="commuting model ~ 1/(2n^2)")
    plt.plot(n_fine, generic_curve, "--", color="#c0392b", linewidth=2, label="generic circuit ~ 2^(-n)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of qubits n")
    plt.ylabel("Gradient variance  Var[dC/dtheta_0]")
    plt.title("Trainability: polynomial vs exponential gradient decay")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
