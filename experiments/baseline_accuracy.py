"""Baseline classifier: per-sample SGD (negative baseline) vs batch gradient descent.

Reproduces the original notebook's first training experiments on the 4-qubit
RY+RY+CNOT-ring classifier, using the plain parameter-shift gradient.
"""

# AI-assisted (Claude, 1987c21): ported with Claude from notebooks/00_original_reference.ipynb.

from __future__ import annotations

from pennylane import numpy as np

from qml_commuting_gradients.circuits import make_ring_cnot_circuit
from qml_commuting_gradients.config import BATCH_GD, PER_SAMPLE_SGD
from qml_commuting_gradients.data import load_iris_binary
from qml_commuting_gradients.gradients import parameter_shift_gradient
from qml_commuting_gradients.training import train_batch_gd, train_per_sample_sgd

N_QUBITS = 4


def run():
    """Train both baselines on Iris and print test accuracy per epoch.

    Returns:
        ``(sgd_history, batch_history)``, each a list of ``(epoch, test_accuracy)``.
    """
    circuit = make_ring_cnot_circuit(N_QUBITS)
    split = load_iris_binary((0, np.pi))
    grad_fn = lambda x, p: parameter_shift_gradient(circuit, x, p)

    print("Per-sample SGD (negative baseline -- why NOT to do this):")
    _, sgd_history = train_per_sample_sgd(
        circuit, grad_fn, split.X_train, split.y_train, split.X_test, split.y_test, N_QUBITS, PER_SAMPLE_SGD
    )
    for epoch, acc in sgd_history:
        if epoch % 5 == 0:
            print(f"  epoch {epoch:2d}  test accuracy: {acc:.3f}")

    print("\nBatch gradient descent (canonical baseline):")
    _, batch_history = train_batch_gd(
        circuit, grad_fn, split.X_train, split.y_train, split.X_test, split.y_test, N_QUBITS, BATCH_GD
    )
    for epoch, acc in batch_history:
        if epoch % 10 == 0:
            print(f"  epoch {epoch:2d}  test accuracy: {acc:.3f}")

    return sgd_history, batch_history


if __name__ == "__main__":
    run()
