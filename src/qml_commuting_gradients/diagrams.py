"""Optional supplementary circuit-diagram generation.

Renders the actual PennyLane QNodes with ``qml.draw_mpl`` -- useful for
sanity-checking the implemented circuits against the report's hand-drawn
TikZ/quantikz diagrams, but not embedded in the report itself, so this is a
standalone/optional script, not part of the main experiment pipeline.

Every dependency is passed in explicitly (qnodes + mock inputs) rather than
assumed to already exist in scope, unlike the original notebook cell, which
relied on other cells having been executed first.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pennylane as qml


def save_circuit_diagrams(
    cg_circuit,
    cg_grad_circuit,
    czz_circuit,
    cnot_circuit,
    mock_x,
    mock_params,
    mock_params_czz,
    out_dir: str | Path = "figures",
) -> list[Path]:
    """Render and save the 4 supplementary circuit diagrams. Returns the saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = [
        ("circuit1_rz.png", cg_circuit, (mock_x, mock_params), "Circuit 1: commuting (RZ only)"),
        (
            "circuit1_gradpass.png",
            cg_grad_circuit,
            (mock_x, mock_params),
            "Circuit 1 gradient pass: Y measured on every qubit",
        ),
        (
            "circuit2_isingzz.png",
            czz_circuit,
            (mock_x, mock_params_czz),
            "Circuit 2: commuting with entanglement (RZ + IsingZZ ring)",
        ),
        ("circuit3_cnot.png", cnot_circuit, (mock_x, mock_params), "Circuit 3: non-commuting (RZ + CNOT ring)"),
    ]

    saved = []
    for fname, qnode, args, title in figs:
        fig, ax = qml.draw_mpl(qnode, decimals=None, style="pennylane")(*args)
        ax.set_title(title, fontsize=13, pad=12)
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved
