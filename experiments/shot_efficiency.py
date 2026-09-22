"""Honest shot-efficiency comparison: commuting-gen vs parameter-shift.

This is the corrected counterpart to the original notebook's cells 6-7 (which
used a fudged/idealized "commuting" shot cost that was never backed by a real
second gradient method). Here, two genuinely different gradient methods are
run against the SAME circuit, data, and initialisation, with REAL per-method
shot costs, and produces the canonical ``figures/scaling_plot.png``.
"""

from __future__ import annotations

from pennylane import numpy as np

from qml_commuting_gradients.circuits import make_cg_circuit
from qml_commuting_gradients.config import CG_WITH_BIAS, SHOTS_PER_CIRCUIT
from qml_commuting_gradients.data import load_iris_binary
from qml_commuting_gradients.gradients import commuting_gradient, parameter_shift_gradient
from qml_commuting_gradients.plotting import plot_shot_efficiency
from qml_commuting_gradients.training import train_cg_with_bias

N_QUBITS = 4


def run():
    cg_circuit, cg_grad_circuit, H_cg = make_cg_circuit(N_QUBITS)
    split = load_iris_binary((0, np.pi / 2))

    grad_fn_cg = lambda x, theta: commuting_gradient(cg_grad_circuit, x, theta, N_QUBITS)
    grad_fn_ps = lambda x, theta: parameter_shift_gradient(cg_circuit, x, theta)

    M = SHOTS_PER_CIRCUIT
    _, _, cg_history = train_cg_with_bias(
        cg_circuit, grad_fn_cg, split.X_train, split.y_train, split.X_test, split.y_test,
        N_QUBITS, CG_WITH_BIAS, shots_per_epoch=M,
    )
    _, _, ps_history = train_cg_with_bias(
        cg_circuit, grad_fn_ps, split.X_train, split.y_train, split.X_test, split.y_test,
        N_QUBITS, CG_WITH_BIAS, shots_per_epoch=2 * N_QUBITS * M,
    )

    print(f"Commuting-gen    (1 call)      final acc={cg_history[-1][1]:.3f}  shots={cg_history[-1][0]:,}")
    print(f"Parameter-shift  (2n calls)    final acc={ps_history[-1][1]:.3f}  shots={ps_history[-1][0]:,}")

    ratio = ps_history[-1][0] // cg_history[-1][0]
    print(f"\nShot ratio PS/CG = {ratio}:1  (expected {2 * N_QUBITS}:1)")

    identical = all(abs(p[1] - c[1]) < 1e-9 for p, c in zip(ps_history, cg_history))
    print("Trajectories identical:", identical)

    out_path = plot_shot_efficiency(cg_history, ps_history, N_QUBITS)
    print(f"\nSaved: {out_path}")

    return cg_history, ps_history


if __name__ == "__main__":
    run()
