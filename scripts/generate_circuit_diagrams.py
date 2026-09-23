"""Optional: render the 4 supplementary circuit diagrams.

Not required to reproduce the report (which draws its own TikZ/quantikz
diagrams), but useful for sanity-checking the implemented circuits. Run as
``python -m scripts.generate_circuit_diagrams`` from the repo root.
"""

# AI-assisted (Claude, 1987c21): ported with Claude from notebooks/00_original_reference.ipynb.

from __future__ import annotations

from pennylane import numpy as np

from qml_commuting_gradients.circuits import make_cg_circuit, make_cnot_entangled_circuit, make_czz_circuit
from qml_commuting_gradients.diagrams import save_circuit_diagrams

N_QUBITS = 4


def run():
    """Render the diagrams at fixed seeded parameters and return the saved paths."""
    cg_circuit, cg_grad_circuit, _ = make_cg_circuit(N_QUBITS)
    czz_circuit, _, _ = make_czz_circuit(N_QUBITS)
    cnot_circuit, _, _ = make_cnot_entangled_circuit(N_QUBITS)

    np.random.seed(0)
    mock_x = np.array([0.3, 0.8, 1.1, 0.5], requires_grad=False)
    mock_params = np.random.uniform(0, 2 * np.pi, size=N_QUBITS, requires_grad=True)

    np.random.seed(0)
    mock_params_czz = np.random.uniform(0, 2 * np.pi, size=2 * N_QUBITS, requires_grad=True)

    saved = save_circuit_diagrams(
        cg_circuit, cg_grad_circuit, czz_circuit, cnot_circuit, mock_x, mock_params, mock_params_czz
    )
    for path in saved:
        print("saved:", path)
    return saved


if __name__ == "__main__":
    run()
