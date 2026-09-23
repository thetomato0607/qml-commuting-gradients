"""Three-arm training comparison: commuting-gen vs parameter-shift vs PennyLane's own optimizer.

All three arms train the same circuit on the same data from the identical
seeded initialisation, tracked over 100 epochs by test MSE.
"""

# AI-assisted (Claude, 1987c21): ported with Claude from notebooks/00_original_reference.ipynb.

from __future__ import annotations

from pennylane import numpy as np

from qml_commuting_gradients.circuits import make_cg_circuit
from qml_commuting_gradients.config import THREE_ARM
from qml_commuting_gradients.data import load_iris_binary
from qml_commuting_gradients.gradients import commuting_gradient, parameter_shift_gradient
from qml_commuting_gradients.plotting import plot_loss_curves_3arm
from qml_commuting_gradients.training import (
    accuracy_with_bias,
    predict_with_bias,
    train_arm_c_pennylane_optimizer,
    train_three_arms,
)

N_QUBITS = 4


def run():
    """Train the three arms, save ``loss_curves_3arm.png`` and print final MSE and accuracy.

    Returns:
        ``(losses_a, losses_b, losses_c)``, the per-epoch test MSE of each arm.
    """
    cg_circuit, cg_grad_circuit, H_cg = make_cg_circuit(N_QUBITS)
    split = load_iris_binary((0, np.pi / 2))

    grad_fn_cg = lambda x, theta: commuting_gradient(cg_grad_circuit, x, theta, N_QUBITS)
    grad_fn_ps = lambda x, theta: parameter_shift_gradient(cg_circuit, x, theta)

    print("A - Commuting-gen gradient  (1 circuit call / sample)")
    theta_a, b_a, losses_a = train_three_arms(
        cg_circuit, grad_fn_cg, split.X_train, split.y_train, split.X_test, split.y_test, N_QUBITS, THREE_ARM
    )

    print("B - Parameter-shift gradient  (2n circuit calls / sample)")
    theta_b, b_b, losses_b = train_three_arms(
        cg_circuit, grad_fn_ps, split.X_train, split.y_train, split.X_test, split.y_test, N_QUBITS, THREE_ARM
    )

    print("C - PennyLane GradientDescentOptimizer  (autograd)")
    theta_c, b_c, losses_c = train_arm_c_pennylane_optimizer(
        cg_circuit, split.X_train, split.y_train, split.X_test, split.y_test, N_QUBITS, THREE_ARM
    )

    out_path = plot_loss_curves_3arm(losses_a, losses_b, losses_c, N_QUBITS)
    print(f"\nSaved: {out_path}")

    print("=" * 52)
    print(f"{'Arm':<37} {'MSE':>7}  {'Acc':>5}")
    print("=" * 52)
    for tag, losses, theta, b in [
        ("A: Commuting-gen  (our method)", losses_a, theta_a, b_a),
        ("B: Parameter-shift  (baseline)", losses_b, theta_b, b_b),
        ("C: PennyLane autograd", losses_c, theta_c, float(b_c)),
    ]:
        acc = accuracy_with_bias(split.X_test, split.y_test, theta, b, cg_circuit)
        print(f"{tag:<37} {losses[-1]:>7.4f}  {acc:>5.3f}")
    print("=" * 52)

    return losses_a, losses_b, losses_c


if __name__ == "__main__":
    run()
