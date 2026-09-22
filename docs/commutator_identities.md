# Why the commuting-generator trick works: commutator identities

This is a pedagogical walkthrough, not runnable pipeline code — it's the
matrix-algebra argument behind `gradients.commuting_gradient`, shown with
explicit 2x2 Pauli matrices so the identities are visible directly.

```python
import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def comm(A, B):
    return A @ B - B @ A
```

**1. Trainable generators commute with each other.** For `Z_0 = Z ⊗ I` and
`Z_1 = I ⊗ Z` (Pauli-Z acting on different qubits), `[Z_0, Z_1] = 0`. This is
what allows every `Y_j` to be measured simultaneously in a single circuit
execution — the generators live on different tensor factors and never
interact.

**2. The generator `Z_j` does NOT commute with the observable term `X_j` on
the same qubit:**

```
[Z, X] = -2i Y
```

**3. That non-commutativity IS the gradient.** From the commuting-generator
gradient formula,

```
dC/dtheta_j = (1/2) <i [Z_j, H]>_psi
            = (1/2) <i * (-2i Y_j) / n>_psi
            = -<Y_j>_psi / n
```

So measuring `Y_j` on every qubit gives the full gradient vector from one
circuit call — exactly the formula implemented in
`qml_commuting_gradients.gradients.commuting_gradient`.
