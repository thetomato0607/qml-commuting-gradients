"""Regression tests for the per-sample vs batch gradient descent baselines.

These are slow (real circuit executions over 30-60 epochs x 70 samples), but
the point is to catch a real training-loop regression, not skip it.

Expected values below are the notebook's ":.3f"-formatted printed accuracies
(e.g. 22/30 test samples correct prints as "0.733"); comparisons use a 1e-3
tolerance to match that print precision rather than asserting exact float
equality against a rounded literal.
"""

# AI-assisted (Claude, 1987c21): written with Claude; expected values are the original
# notebook's recorded outputs.

from pennylane import numpy as np

from qml_commuting_gradients.circuits import make_ring_cnot_circuit
from qml_commuting_gradients.config import BATCH_GD, PER_SAMPLE_SGD
from qml_commuting_gradients.data import load_iris_binary
from qml_commuting_gradients.gradients import parameter_shift_gradient
from qml_commuting_gradients.training import train_batch_gd, train_per_sample_sgd

N_QUBITS = 4
TOL = 1e-3


def _grad_fn(circuit):
    return lambda x, p: parameter_shift_gradient(circuit, x, p)


def test_per_sample_sgd_degrades_to_expected_accuracy():
    """Per-sample SGD falls from 0.733 to 0.567 test accuracy, as in the notebook."""
    circuit = make_ring_cnot_circuit(N_QUBITS)
    split = load_iris_binary((0, np.pi))
    _, history = train_per_sample_sgd(
        circuit, _grad_fn(circuit), split.X_train, split.y_train, split.X_test, split.y_test, N_QUBITS, PER_SAMPLE_SGD
    )
    by_epoch = dict(history)
    assert abs(by_epoch[0] - 0.733) < TOL
    assert abs(by_epoch[30] - 0.567) < TOL


def test_batch_gd_reaches_expected_accuracy():
    """Batch GD follows the notebook's 0.733 -> 0.767 -> 0.900 accuracy trajectory."""
    circuit = make_ring_cnot_circuit(N_QUBITS)
    split = load_iris_binary((0, np.pi))
    _, history = train_batch_gd(
        circuit, _grad_fn(circuit), split.X_train, split.y_train, split.X_test, split.y_test, N_QUBITS, BATCH_GD
    )
    by_epoch = dict(history)
    assert abs(by_epoch[0] - 0.733) < TOL
    assert abs(by_epoch[20] - 0.767) < TOL
    assert abs(by_epoch[30] - 0.900) < TOL
    assert abs(by_epoch[60] - 0.900) < TOL
