"""Integration test for the full entanglement-structure experiment."""

# AI-assisted (Claude, 1987c21): written with Claude; expected values are the original
# notebook's recorded outputs.

from experiments import entanglement_structure


def test_entanglement_experiment_matches_report_values():
    """Commuting circuits match autograd, IsingZZ entangles, and CNOT breaks the formula."""
    results = entanglement_structure.run()
    assert results["cg_diff"] < 1e-10
    assert results["czz_diff"] < 1e-10
    assert 0.98 < results["czz_purity"] < 1.0
    assert results["cnot_diff"] > 1e-3
