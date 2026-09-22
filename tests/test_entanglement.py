"""Integration test for the full entanglement-structure experiment."""

from experiments import entanglement_structure


def test_entanglement_experiment_matches_report_values():
    results = entanglement_structure.run()
    assert results["cg_diff"] < 1e-10
    assert results["czz_diff"] < 1e-10
    assert 0.98 < results["czz_purity"] < 1.0
    assert results["cnot_diff"] > 1e-3
