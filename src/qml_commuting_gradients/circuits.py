"""Circuit factories for the commuting-generator gradient experiments.

Every function here builds and returns fresh PennyLane QNode(s) bound to a new
``default.qubit`` device, so callers can request circuits at any qubit count
without clobbering global state (the one pattern the original notebook itself
used correctly for the trainability sweep, generalized here to every circuit
family).

All trainable parameters passed into these QNodes must be constructed with
``pennylane.numpy`` (imported as ``np`` below), not plain NumPy, so that
``requires_grad=True`` and ``qml.grad`` work as expected.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np


def make_ring_cnot_circuit(n_qubits: int, device: qml.Device | None = None) -> qml.QNode:
    """RY data encoding + RY trainable layer + CNOT entangling ring, measuring <Z_0>.

    This is the baseline classifier circuit used for the plain parameter-shift
    experiments (per-sample and batch gradient descent).
    """
    dev = device or qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, params):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_cg_circuit(n_qubits: int, device: qml.Device | None = None) -> tuple[qml.QNode, qml.QNode, qml.Hamiltonian]:
    """RY data encoding + RZ trainable layer (no entanglement); commuting generators.

    Returns ``(cg_circuit, cg_grad_circuit, H)`` where:
      - ``cg_circuit(x, params)``: forward pass, C = (1/n) sum_j sin(x_j) cos(theta_j)
      - ``cg_grad_circuit(x, params)``: returns [<Y_j> for all j] from ONE execution
      - ``H``: the measured observable, (1/n) sum_j X_j

    Generalizes over ``n_qubits`` so the same builder serves both the 4-qubit
    classifier and the trainability sweep up to n=14.
    """
    dev = device or qml.device("default.qubit", wires=n_qubits)
    H = qml.Hamiltonian(
        [1 / n_qubits] * n_qubits,
        [qml.PauliX(j) for j in range(n_qubits)],
    )

    @qml.qnode(dev)
    def cg_circuit(x, params):
        for j in range(n_qubits):
            qml.RY(x[j], wires=j)
        for j in range(n_qubits):
            qml.RZ(params[j], wires=j)
        return qml.expval(H)

    @qml.qnode(dev)
    def cg_grad_circuit(x, params):
        for j in range(n_qubits):
            qml.RY(x[j], wires=j)
        for j in range(n_qubits):
            qml.RZ(params[j], wires=j)
        return [qml.expval(qml.PauliY(j)) for j in range(n_qubits)]

    return cg_circuit, cg_grad_circuit, H


def make_czz_circuit(n_qubits: int, device: qml.Device | None = None) -> tuple[qml.QNode, qml.QNode, qml.Hamiltonian]:
    """RY encoding + RZ(theta) + IsingZZ(phi) ring; entangled but still commuting.

    ``params`` is the concatenation ``[theta (n) | phi (n)]``, length 2n.

    Returns ``(czz_circuit, czz_grad_circuit, H)`` where ``czz_grad_circuit(x,
    params, m)`` measures the co-measurable group for qubit m: [<Y_m>, <Y_m
    Z_{m+1}>, <Z_{m-1} Y_m>].
    """
    dev = device or qml.device("default.qubit", wires=n_qubits)
    H = qml.Hamiltonian(
        [1 / n_qubits] * n_qubits,
        [qml.PauliX(j) for j in range(n_qubits)],
    )

    @qml.qnode(dev)
    def czz_circuit(x, params):
        theta = params[:n_qubits]
        phi = params[n_qubits:]
        for j in range(n_qubits):
            qml.RY(x[j], wires=j)
        for j in range(n_qubits):
            qml.RZ(theta[j], wires=j)
        for j in range(n_qubits):
            qml.IsingZZ(phi[j], wires=[j, (j + 1) % n_qubits])
        return qml.expval(H)

    @qml.qnode(dev)
    def czz_grad_circuit(x, params, m):
        theta = params[:n_qubits]
        phi = params[n_qubits:]
        for j in range(n_qubits):
            qml.RY(x[j], wires=j)
        for j in range(n_qubits):
            qml.RZ(theta[j], wires=j)
        for j in range(n_qubits):
            qml.IsingZZ(phi[j], wires=[j, (j + 1) % n_qubits])
        mp1 = (m + 1) % n_qubits
        mm1 = (m - 1) % n_qubits
        return [
            qml.expval(qml.PauliY(m)),
            qml.expval(qml.PauliY(m) @ qml.PauliZ(mp1)),
            qml.expval(qml.PauliZ(mm1) @ qml.PauliY(m)),
        ]

    return czz_circuit, czz_grad_circuit, H


def make_density_probe(n_qubits: int, device: qml.Device | None = None) -> qml.QNode:
    """RY + RZ(theta) + IsingZZ(phi) circuit returning the reduced density matrix of qubit 0.

    Used to certify genuine entanglement (purity < 1) in the czz circuit.
    """
    dev = device or qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def czz_density(x, params):
        theta = params[:n_qubits]
        phi = params[n_qubits:]
        for j in range(n_qubits):
            qml.RY(x[j], wires=j)
        for j in range(n_qubits):
            qml.RZ(theta[j], wires=j)
        for j in range(n_qubits):
            qml.IsingZZ(phi[j], wires=[j, (j + 1) % n_qubits])
        return qml.density_matrix(wires=[0])

    return czz_density


def make_cnot_entangled_circuit(n_qubits: int, device: qml.Device | None = None) -> tuple[qml.QNode, qml.QNode, qml.Hamiltonian]:
    """RY encoding + RZ(theta) trainable layer + CNOT ring; entangled and NOT commuting.

    Deliberately breaks the commuting-generator structure: the naive
    ``-(1/n)<Y_j>`` gradient formula (see ``gradients.naive_commuting_gradient``)
    does not equal the true gradient for this circuit.
    """
    dev = device or qml.device("default.qubit", wires=n_qubits)
    H = qml.Hamiltonian(
        [1 / n_qubits] * n_qubits,
        [qml.PauliX(j) for j in range(n_qubits)],
    )

    @qml.qnode(dev)
    def cnot_circuit(x, theta):
        for j in range(n_qubits):
            qml.RY(x[j], wires=j)
        for j in range(n_qubits):
            qml.RZ(theta[j], wires=j)
        for j in range(n_qubits):
            qml.CNOT(wires=[j, (j + 1) % n_qubits])
        return qml.expval(H)

    @qml.qnode(dev)
    def cnot_grad_circuit(x, theta):
        for j in range(n_qubits):
            qml.RY(x[j], wires=j)
        for j in range(n_qubits):
            qml.RZ(theta[j], wires=j)
        for j in range(n_qubits):
            qml.CNOT(wires=[j, (j + 1) % n_qubits])
        return [qml.expval(qml.PauliY(j)) for j in range(n_qubits)]

    return cnot_circuit, cnot_grad_circuit, H
