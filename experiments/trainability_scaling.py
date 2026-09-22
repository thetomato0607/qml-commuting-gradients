"""Trainability check: gradient variance under random initialisation, n=2..14.

Analytic result for the RZ-only model C = (1/n) sum_j sin(x_j) cos(theta_j):
    dC/dtheta_j = -(1/n) sin(x_j) sin(theta_j)
Under theta_j ~ U[0, 2*pi]: Var[dC/dtheta_j] = sin^2(x_j) / (2 n^2) -- a
polynomial (not exponential/barren-plateau) decay. Verified here with the
real circuit + PennyLane autograd, not the analytic formula plugged into
itself.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

from qml_commuting_gradients.circuits import make_cg_circuit
from qml_commuting_gradients.config import (
    TRAINABILITY_N_LIST,
    TRAINABILITY_N_SAMPLES,
    TRAINABILITY_SEED,
    TRAINABILITY_X_VAL,
)
from qml_commuting_gradients.plotting import plot_trainability


def run():
    np.random.seed(TRAINABILITY_SEED)
    trainability_results = []

    print(f"{'n':>3} | {'empirical':>10} | {'theory':>10} | {'ratio':>6}")
    print("-" * 40)
    for n in TRAINABILITY_N_LIST:
        circuit_n, _, _ = make_cg_circuit(n)
        grad_fn = qml.grad(circuit_n, argnums=1)
        x = np.array([TRAINABILITY_X_VAL] * n, requires_grad=False)

        grad0_samples = []
        for _ in range(TRAINABILITY_N_SAMPLES):
            theta = np.random.uniform(0, 2 * np.pi, size=n, requires_grad=True)
            g = grad_fn(x, theta)
            grad0_samples.append(float(g[0]))

        empirical_var = float(np.var(grad0_samples))
        theory_var = np.sin(TRAINABILITY_X_VAL) ** 2 / (2 * n**2)
        ratio = empirical_var / theory_var

        trainability_results.append((n, empirical_var, theory_var, ratio))
        print(f"{n:3d} | {empirical_var:10.3e} | {theory_var:10.3e} | {ratio:6.3f}")

    out_path = plot_trainability(trainability_results, TRAINABILITY_X_VAL, TRAINABILITY_N_LIST)
    print(f"\nSaved: {out_path}")

    return trainability_results


if __name__ == "__main__":
    run()
