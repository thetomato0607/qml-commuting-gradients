"""Regression tests: every gradient method must match PennyLane autograd to the
exact tolerance recorded in the original notebook."""

# AI-assisted (Claude, 1987c21): written with Claude; expected values are the original
# notebook's recorded outputs.

import pennylane as qml
from pennylane import numpy as np

from qml_commuting_gradients.circuits import (
    make_cg_circuit,
    make_cnot_entangled_circuit,
    make_czz_circuit,
    make_ring_cnot_circuit,
)
from qml_commuting_gradients.gradients import (
    commuting_gradient,
    czz_gradient,
    naive_commuting_gradient,
    parameter_shift_gradient,
)

N_QUBITS = 4


def test_parameter_shift_matches_autograd_on_ring_cnot_circuit():
    """Parameter-shift is exact on the RY + CNOT-ring classifier."""
    circuit = make_ring_cnot_circuit(N_QUBITS)
    x = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=False)
    params = np.array([0.5, 1.0, 1.5, 2.0], requires_grad=True)

    manual = parameter_shift_gradient(circuit, x, params)
    exact = qml.grad(circuit, argnums=1)(x, params)
    assert np.max(np.abs(manual - exact)) < 1e-10


def test_parameter_shift_matches_autograd_on_cg_circuit():
    """Parameter-shift reproduces the notebook's recorded gradient on the RZ-only circuit."""
    cg_circuit, _, _ = make_cg_circuit(N_QUBITS)
    np.random.seed(0)
    x_mock = np.array([0.3, 0.8, 1.1, 0.5], requires_grad=False)
    p_test = np.random.uniform(0, 2 * np.pi, size=N_QUBITS, requires_grad=True)

    manual = parameter_shift_gradient(cg_circuit, x_mock, p_test)
    exact = qml.grad(cg_circuit, argnums=1)(x_mock, p_test)
    expected = np.array([0.02230574, 0.17506638, 0.13406939, 0.03335438])
    assert np.max(np.abs(manual - expected)) < 1e-6
    assert np.max(np.abs(manual - exact)) < 1e-10


def test_commuting_gradient_matches_autograd():
    """The one-call commuting-generator gradient equals autograd to machine precision."""
    cg_circuit, cg_grad_circuit, _ = make_cg_circuit(N_QUBITS)
    np.random.seed(0)
    x_mock = np.array([0.3, 0.8, 1.1, 0.5], requires_grad=False)
    p_test = np.random.uniform(0, 2 * np.pi, size=N_QUBITS, requires_grad=True)

    manual = commuting_gradient(cg_grad_circuit, x_mock, p_test, N_QUBITS)
    exact = qml.grad(cg_circuit, argnums=1)(x_mock, p_test)
    diff = float(np.max(np.abs(manual - exact)))
    assert diff < 1e-10, f"expected ~6.94e-17, got {diff:.2e}"


def test_czz_gradient_matches_autograd_and_circuit_is_entangled():
    """The grouped-measurement gradient stays exact once IsingZZ entanglement is added."""
    cg_circuit, _, _ = make_cg_circuit(N_QUBITS)
    czz_circuit, czz_grad_circuit, _ = make_czz_circuit(N_QUBITS)
    np.random.seed(0)
    x_mock = np.array([0.3, 0.8, 1.1, 0.5], requires_grad=False)
    np.random.seed(0)
    params_czz = np.random.uniform(0, 2 * np.pi, size=2 * N_QUBITS, requires_grad=True)

    manual = czz_gradient(czz_grad_circuit, x_mock, params_czz, N_QUBITS)
    exact = qml.grad(czz_circuit, argnums=1)(x_mock, params_czz)
    diff = float(np.max(np.abs(manual - exact)))
    assert diff < 1e-10, f"expected ~1.11e-16, got {diff:.2e}"


def test_naive_commuting_gradient_fails_on_cnot_circuit():
    """Deliberate negative result: CNOT breaks the commuting structure."""
    cnot_circuit, cnot_grad_circuit, _ = make_cnot_entangled_circuit(N_QUBITS)
    np.random.seed(0)
    x_mock = np.array([0.3, 0.8, 1.1, 0.5], requires_grad=False)
    p_test = np.random.uniform(0, 2 * np.pi, size=N_QUBITS, requires_grad=True)

    naive = naive_commuting_gradient(cnot_grad_circuit, x_mock, p_test, N_QUBITS)
    exact = qml.grad(cnot_circuit, argnums=1)(x_mock, p_test)
    diff = float(np.max(np.abs(naive - exact)))
    assert diff > 1e-3, "the naive formula is expected to FAIL on a non-commuting circuit"
    assert abs(diff - 4.15e-02) < 5e-4
