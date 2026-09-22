"""Training loops and evaluation helpers.

Every training function here takes ``circuit`` and ``grad_fn`` as explicit
arguments rather than closing over module-level globals (the original
notebook redefined ``circuit``/gradient functions repeatedly across cells).
``grad_fn`` always has the signature ``grad_fn(x, params) -> array``, already
bound to whichever circuit and gradient method the caller wants (e.g. via
``functools.partial`` or a lambda over ``gradients.parameter_shift_gradient``
or ``gradients.commuting_gradient``), matching how the original notebook's own
gradient functions closed over a fixed circuit.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

from .config import TrainConfig


def predict(x, params, circuit) -> int:
    """Class prediction: sign of the circuit's expectation value -> -1 or +1."""
    return 1 if circuit(x, params) >= 0 else -1


def accuracy(X, Y, params, circuit) -> float:
    preds = [predict(x, params, circuit) for x in X]
    return float(np.mean([p == y for p, y in zip(preds, Y)]))


def predict_with_bias(x, theta, b, circuit) -> int:
    """Class prediction with a learned threshold b, for the commuting-generator circuit."""
    return 1 if float(circuit(x, theta)) >= b else -1


def accuracy_with_bias(X, Y, theta, b, circuit) -> float:
    preds = [predict_with_bias(x, theta, b, circuit) for x in X]
    return float(np.mean([p == y for p, y in zip(preds, Y)]))


def test_mse(theta, b, X, Y, circuit) -> float:
    """Mean squared error of circuit(x, theta) - b against the +-1 labels."""
    preds = np.array([float(circuit(x, theta)) for x in X])
    return float(np.mean((preds - float(b) - np.array(Y, dtype=float)) ** 2))


def train_per_sample_sgd(circuit, grad_fn, X_train, y_train, X_test, y_test, n_qubits: int, config: TrainConfig):
    """Per-sample SGD baseline: update params after every single training example.

    Deliberately kept as a documented NEGATIVE result -- this update rule
    degrades test accuracy (0.733 -> 0.567) rather than improving it, which is
    why ``train_batch_gd`` (averaging the gradient over the full batch before
    each update) is used everywhere else in this repo. Not dropped as scratch
    exploration: it demonstrates *why* the batch approach is necessary.
    """
    np.random.seed(config.seed)
    params = np.random.uniform(0, 2 * np.pi, size=n_qubits, requires_grad=True)
    history = [(0, accuracy(X_test, y_test, params, circuit))]
    for epoch in range(config.epochs):
        for x, y in zip(X_train, y_train):
            c = circuit(x, params)
            grad = grad_fn(x, params)
            params = params - config.lr * 2 * (c - y) * grad
        history.append((epoch + 1, accuracy(X_test, y_test, params, circuit)))
    return params, history


def total_gradient(X, Y, params, circuit, grad_fn) -> np.ndarray:
    """Loss gradient averaged over the full training set (batch gradient)."""
    grad_sum = np.zeros(len(params))
    for x, y in zip(X, Y):
        c = circuit(x, params)
        grad_sum += 2 * (c - y) * grad_fn(x, params)
    return grad_sum / len(X)


def train_batch_gd(circuit, grad_fn, X_train, y_train, X_test, y_test, n_qubits: int, config: TrainConfig):
    """Batch gradient descent: one averaged update per epoch. Reaches 0.900 test accuracy."""
    np.random.seed(config.seed)
    params = np.random.uniform(0, 2 * np.pi, size=n_qubits, requires_grad=True)
    history = [(0, accuracy(X_test, y_test, params, circuit))]
    for epoch in range(config.epochs):
        grad = total_gradient(X_train, y_train, params, circuit, grad_fn)
        params = params - config.lr * grad
        history.append((epoch + 1, accuracy(X_test, y_test, params, circuit)))
    return params, history


def train_cg_with_bias(
    circuit,
    grad_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    n_qubits: int,
    config: TrainConfig,
    shots_per_epoch: int = 0,
):
    """Bias-augmented commuting-generator training, with optional shot-cost tracking.

    Generalizes both the single-method bias-training loop and the head-to-head
    shot-efficiency comparison: pass ``grad_fn`` bound to the commuting-gen or
    parameter-shift method, and ``shots_per_epoch`` set to that method's real
    per-epoch shot cost, to reproduce the honest shot-efficiency comparison.
    Returns ``(theta, b, history)`` where ``history`` is a list of
    ``(cumulative_shots, test_accuracy)`` tuples, one per epoch (plus an
    initial pre-training entry at shots=0).
    """
    np.random.seed(config.seed)
    noise = np.random.uniform(-0.1, 0.1, size=n_qubits)
    theta = np.pi / 4 + noise
    b = 1.0
    history = [(0, accuracy_with_bias(X_test, y_test, theta, b, circuit))]
    total_shots = 0
    for _ in range(config.epochs):
        dtheta = np.zeros(n_qubits)
        db_sum = 0.0
        for x, y in zip(X_train, y_train):
            c = float(circuit(x, theta))
            residual = c - b - y
            dtheta += 2 * residual * grad_fn(x, theta)
            db_sum += -2 * residual
        theta = theta - config.lr_theta * dtheta / len(X_train)
        b = b - config.lr_b * db_sum / len(X_train)
        total_shots += shots_per_epoch
        history.append((total_shots, accuracy_with_bias(X_test, y_test, theta, b, circuit)))
    return theta, b, history


def train_three_arms(circuit, grad_fn, X_train, y_train, X_test, y_test, n_qubits: int, config: TrainConfig):
    """One arm (A or B) of the three-arm comparison: tracks per-epoch test MSE.

    Pass ``grad_fn`` bound to the commuting-gen method for arm A, or to
    parameter-shift for arm B; both start from the identical seeded init.
    """
    np.random.seed(config.seed)
    theta = np.pi / 4 + np.random.uniform(-0.1, 0.1, size=n_qubits)
    b = 1.0
    losses = []
    for _ in range(config.epochs):
        dth = np.zeros(n_qubits)
        db = 0.0
        for x, y in zip(X_train, y_train):
            c = float(circuit(x, theta))
            r = c - b - float(y)
            dth += 2 * r * grad_fn(x, theta)
            db -= 2 * r
        theta = theta - config.lr_theta * dth / len(X_train)
        b = b - config.lr_b * db / len(X_train)
        losses.append(test_mse(theta, b, X_test, y_test, circuit))
    return theta, b, losses


def train_arm_c_pennylane_optimizer(circuit, X_train, y_train, X_test, y_test, n_qubits: int, config: TrainConfig):
    """Arm C: PennyLane's built-in GradientDescentOptimizer via autograd, no manual gradient."""
    np.random.seed(config.seed)
    theta = np.array(np.pi / 4 + np.random.uniform(-0.1, 0.1, size=n_qubits), requires_grad=True)
    b = np.array(1.0, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=config.lr_theta)
    losses = []

    def cost(th, bias):
        loss = np.array(0.0)
        for x, y in zip(X_train, y_train):
            loss = loss + (circuit(x, th) - bias - float(y)) ** 2
        return loss / len(X_train)

    for _ in range(config.epochs):
        (theta, b), _ = opt.step_and_cost(cost, theta, b)
        losses.append(test_mse(theta, b, X_test, y_test, circuit))
    return theta, b, losses
