"""Gradient correctness check for the commuting-generator (RZ-only) circuit.

Validates that the single-circuit-call commuting-gradient formula matches
PennyLane's automatic differentiation to machine precision.
"""

# AI-assisted (Claude, 1987c21): ported with Claude from notebooks/00_original_reference.ipynb.

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

from qml_commuting_gradients.circuits import make_cg_circuit
from qml_commuting_gradients.gradients import commuting_gradient

N_QUBITS = 4


def run():
    """Compare the commuting-generator gradient with autograd at a seeded random point.

    Returns:
        ``(max_abs_diff, x, params, cg_circuit, cg_grad_circuit, H)`` so that
        ``entanglement_structure`` can reuse the same inputs and circuits.
    """
    cg_circuit, cg_grad_circuit, H_cg = make_cg_circuit(N_QUBITS)

    np.random.seed(0)
    x_mock = np.array([0.3, 0.8, 1.1, 0.5], requires_grad=False)
    p_test = np.random.uniform(0, 2 * np.pi, size=N_QUBITS, requires_grad=True)

    autograd_grad = qml.grad(cg_circuit, argnums=1)(x_mock, p_test)
    manual_grad = commuting_gradient(cg_grad_circuit, x_mock, p_test, N_QUBITS)
    diff = float(np.max(np.abs(autograd_grad - manual_grad)))

    print("PennyLane autograd:   ", autograd_grad)
    print("Commuting-gen formula:", manual_grad)
    print(f"Max |difference|:      {diff:.2e}  {'PASS' if diff < 1e-10 else 'FAIL'}")

    return diff, x_mock, p_test, cg_circuit, cg_grad_circuit, H_cg


if __name__ == "__main__":
    run()
