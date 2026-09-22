"""Regression test for the honest shot-efficiency comparison (fixes scaling_plot.png)."""

from pennylane import numpy as np

from qml_commuting_gradients.circuits import make_cg_circuit
from qml_commuting_gradients.config import CG_WITH_BIAS, SHOTS_PER_CIRCUIT
from qml_commuting_gradients.data import load_iris_binary
from qml_commuting_gradients.gradients import commuting_gradient, parameter_shift_gradient
from qml_commuting_gradients.training import train_cg_with_bias

N_QUBITS = 4


def test_commuting_and_parameter_shift_produce_identical_trajectories():
    cg_circuit, cg_grad_circuit, _ = make_cg_circuit(N_QUBITS)
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

    assert cg_history[-1][1] == 1.0
    assert ps_history[-1][1] == 1.0
    assert cg_history[-1][0] == 60_000
    assert ps_history[-1][0] == 480_000
    assert all(abs(p[1] - c[1]) < 1e-9 for p, c in zip(ps_history, cg_history))
