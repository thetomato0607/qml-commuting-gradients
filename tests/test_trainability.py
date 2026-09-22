"""Regression test for the gradient-variance trainability sweep, n=2..14."""

import pennylane as qml
from pennylane import numpy as np

from qml_commuting_gradients.circuits import make_cg_circuit
from qml_commuting_gradients.config import (
    TRAINABILITY_N_LIST,
    TRAINABILITY_N_SAMPLES,
    TRAINABILITY_SEED,
    TRAINABILITY_X_VAL,
)

# (n, empirical_var, theory_var) recorded in the original notebook / report,
# printed there with "%10.3e" (4 significant figures) -- comparisons below use
# a relative tolerance to match that print precision, not exact equality.
EXPECTED = {
    2: 7.568e-02,
    4: 1.850e-02,
    6: 8.354e-03,
    8: 4.698e-03,
    10: 3.130e-03,
    12: 2.069e-03,
    14: 1.602e-03,
}


def test_empirical_variance_matches_recorded_values_and_theory():
    np.random.seed(TRAINABILITY_SEED)
    for n in TRAINABILITY_N_LIST:
        circuit_n, _, _ = make_cg_circuit(n)
        grad_fn = qml.grad(circuit_n, argnums=1)
        x = np.array([TRAINABILITY_X_VAL] * n, requires_grad=False)

        samples = []
        for _ in range(TRAINABILITY_N_SAMPLES):
            theta = np.random.uniform(0, 2 * np.pi, size=n, requires_grad=True)
            samples.append(float(grad_fn(x, theta)[0]))

        empirical_var = float(np.var(samples))
        theory_var = np.sin(TRAINABILITY_X_VAL) ** 2 / (2 * n**2)

        rel_diff = abs(empirical_var - EXPECTED[n]) / EXPECTED[n]
        assert rel_diff < 1e-3, f"n={n}: {empirical_var} vs recorded {EXPECTED[n]} ({rel_diff:.1%} off)"
        ratio = empirical_var / theory_var
        assert 0.9 < ratio < 1.1, f"n={n}: empirical/theory ratio {ratio} outside sampling-error band"
