"""Gradient computation methods: parameter-shift and commuting-generator.

Every trainable array passed into these functions (``params``/``theta``) must
be constructed with ``pennylane.numpy`` (imported as ``np`` below) so that
``.copy()`` and arithmetic preserve the array semantics PennyLane's QNodes
expect. Data samples (``x``) may remain plain NumPy, exactly as in the
original notebook.
"""

# AI-assisted (Claude, 1987c21): ported with Claude from notebooks/00_original_reference.ipynb.

from __future__ import annotations

from pennylane import numpy as np


def parameter_shift_gradient(circuit, x, params, shift: float = np.pi / 2) -> np.ndarray:
    """Parameter-shift gradient of ``circuit(x, params)`` w.r.t. ``params``.

    Unifies the notebook's three separate parameter-shift implementations
    (a flat 1D version, a layered 2D version using ``np.ndindex``, and one
    hardcoded to a specific circuit) into a single function: ``np.ndindex``
    already generalizes over any parameter array shape, flat or structured.
    """
    grad = np.zeros_like(params)
    for idx in np.ndindex(params.shape):
        params_plus = params.copy()
        params_plus[idx] += shift
        params_minus = params.copy()
        params_minus[idx] -= shift
        grad[idx] = 0.5 * (circuit(x, params_plus) - circuit(x, params_minus))
    return grad


def commuting_gradient(cg_grad_circuit, x, params, n_qubits: int) -> np.ndarray:
    """Commuting-generator gradient for the RZ-only circuit: -<Y_j> / n, all j from one call."""
    y_vals = cg_grad_circuit(x, params)
    return -np.array(y_vals) / n_qubits


def czz_gradient(czz_grad_circuit, x, params, n_qubits: int) -> np.ndarray:
    """Commuting-generator gradient for the RZ + IsingZZ circuit.

    Requires ``n`` circuit calls (one per qubit, grouped by co-measurable
    Pauli observables), returning the full ``2n``-length ``[theta_grad |
    phi_grad]`` vector.
    """
    outputs = [czz_grad_circuit(x, params, m) for m in range(n_qubits)]
    theta_grad = np.zeros(n_qubits)
    phi_grad = np.zeros(n_qubits)
    for j in range(n_qubits):
        a_j, b_j, _ = outputs[j]
        _, _, c_jp1 = outputs[(j + 1) % n_qubits]
        theta_grad[j] = -float(a_j) / n_qubits
        phi_grad[j] = -(float(b_j) + float(c_jp1)) / n_qubits
    return np.concatenate([theta_grad, phi_grad])


def naive_commuting_gradient(cnot_grad_circuit, x, theta, n_qubits: int) -> np.ndarray:
    """Naive -(1/n)<Y_j> formula applied to the RZ + CNOT circuit, where it does NOT hold.

    Kept as the deliberate negative result: CNOT breaks the commuting
    structure, so this formula disagrees with the true (autograd) gradient.
    """
    y_vals = cnot_grad_circuit(x, theta)
    return -np.array(y_vals, dtype=float) / n_qubits
