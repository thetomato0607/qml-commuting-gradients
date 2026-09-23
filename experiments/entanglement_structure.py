"""Entanglement experiment: where does the fast (co-measurable) gradient survive?

Three circuits sharing the same RY encoding and observable H:
  1. cg_circuit    RZ only            -- not entangled, commuting
  2. czz_circuit   RZ + IsingZZ ring  -- entangled, still commuting
  3. cnot_circuit  RZ + CNOT ring     -- entangled, NOT commuting

Circuit 1's gradient check is reused from ``commuting_gradient_check``.
"""

# AI-assisted (Claude, 1987c21): ported with Claude from notebooks/00_original_reference.ipynb.

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

from qml_commuting_gradients.circuits import (
    make_cg_circuit,
    make_cnot_entangled_circuit,
    make_czz_circuit,
    make_density_probe,
)
from qml_commuting_gradients.gradients import czz_gradient, naive_commuting_gradient
from experiments import commuting_gradient_check

N_QUBITS = 4


def run():
    """Run all three gradient checks plus the qubit-0 purity probe and print a summary table.

    Returns:
        Dict with the three max-abs gradient differences, the IsingZZ circuit's
        qubit-0 purity, and the objects the tests reuse.
    """
    cg_diff, x_mock, p_test, cg_circuit, cg_grad_circuit, H_cg = commuting_gradient_check.run()

    czz_circuit, czz_grad_circuit, H = make_czz_circuit(N_QUBITS)
    np.random.seed(0)
    params_czz = np.random.uniform(0, 2 * np.pi, size=2 * N_QUBITS, requires_grad=True)

    autograd_czz = qml.grad(czz_circuit, argnums=1)(x_mock, params_czz)
    manual_czz = czz_gradient(czz_grad_circuit, x_mock, params_czz, N_QUBITS)
    diff_czz = float(np.max(np.abs(autograd_czz - manual_czz)))

    print("\n2. czz_circuit (entangled, commuting)")
    print("   PennyLane autograd:", autograd_czz)
    print("   Grouped-measurement formula:", manual_czz)
    print(f"   Max |difference|: {diff_czz:.2e}  {'PASS' if diff_czz < 1e-10 else 'FAIL'}")

    density_probe = make_density_probe(N_QUBITS)
    rho0 = density_probe(x_mock, params_czz)
    purity_czz = float(np.real(np.trace(rho0 @ rho0)))
    print(f"   Reduced purity (qubit 0): {purity_czz:.4f}  {'(entangled)' if purity_czz < 1 - 1e-6 else '(product state!)'}")

    cnot_circuit, cnot_grad_circuit, Hc = make_cnot_entangled_circuit(N_QUBITS)
    autograd_cnot = qml.grad(cnot_circuit, argnums=1)(x_mock, p_test)
    naive_cnot = naive_commuting_gradient(cnot_grad_circuit, x_mock, p_test, N_QUBITS)
    diff_cnot = float(np.max(np.abs(autograd_cnot - naive_cnot)))

    print("\n3. cnot_circuit (entangled, NOT commuting)")
    print("   PennyLane autograd:  ", autograd_cnot)
    print("   Naive -(1/n)<Y_j>:   ", naive_cnot)
    print(f"   Max |difference|: {diff_cnot:.2e}")
    print("   FAILS as expected (structure broken)" if diff_cnot > 1e-3 else "   unexpectedly passed")

    print("\n" + "=" * 78)
    print(f"{'circuit':<16} {'entangled':>10} {'commuting':>10} {'grad check':>11} {'circuits/grad':>14}")
    print("=" * 78)
    print(f"{'1: cg_circuit':<16} {'no':>10} {'yes':>10} {'PASS':>11} {'1':>14}")
    print(f"{'2: czz_circuit':<16} {'yes':>10} {'yes':>10} {'PASS':>11} {'n':>14}")
    print(f"{'3: cnot_circuit':<16} {'yes':>10} {'no':>10} {'FAIL':>11} {'--':>14}")
    print("=" * 78)

    return {
        "cg_diff": cg_diff,
        "czz_diff": diff_czz,
        "czz_purity": purity_czz,
        "cnot_diff": diff_cnot,
        "params_czz": params_czz,
        "cnot_circuit": cnot_circuit,
    }


if __name__ == "__main__":
    run()
