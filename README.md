# qml-commuting-gradients

Backpropagation-scaling gradients for parameterised quantum circuits via
commuting generators. Implements and validates the commuting-generator
gradient trick (a single circuit execution yields the full gradient vector
when gate generators mutually commute), compares it against manual
parameter-shift and PennyLane's built-in optimizer, tests it under
entanglement (IsingZZ vs CNOT), and empirically verifies polynomial (not
exponential/barren-plateau) gradient-variance scaling up to 14 qubits.

This repo is a refactor of a research notebook (preserved at
`notebooks/00_original_reference.ipynb`) into a reproducible package. See
"Deviations from the original notebook" below for the two intentional changes
made during the port. The accompanying written report is not included in this
repository.

## Setup

Requires Python 3.11 or newer (`requires-python` in `pyproject.toml`); the lock
file pins every dependency.

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements-lock.txt
pip install -e .
```

## Run everything

```bash
python -m experiments.run_all
```

Or run individual sections:

```bash
python -m experiments.baseline_accuracy          # per-sample SGD vs batch GD
python -m experiments.commuting_gradient_check   # gradient vs autograd, machine precision
python -m experiments.shot_efficiency            # honest shot-cost comparison -> scaling_plot.png
python -m experiments.three_arm_comparison       # commuting-gen vs param-shift vs PennyLane optimizer
python -m experiments.entanglement_structure     # RZ-only vs IsingZZ vs CNOT
python -m experiments.trainability_scaling       # gradient variance, n=2..14
python -m scripts.generate_circuit_diagrams      # optional: qml.draw_mpl circuit figures
```

Figures are written to `figures/`.

## Tests

```bash
pytest
```

Encodes every numeric result in the accompanying report (gradient-check values,
accuracy trajectories, the shot-cost table, the entanglement table, the
trainability sweep) as regression tests against the original notebook's
recorded outputs.

## Repository layout

```
src/qml_commuting_gradients/   core library: data loading, circuits, gradients, training, plotting
experiments/                    thin orchestration scripts, one per report section
scripts/                        optional supplementary tooling (circuit diagrams)
tests/                          regression tests against the notebook's recorded outputs
docs/                           pedagogical notes not part of the runnable pipeline
figures/                        generated report figures
notebooks/                      the original notebook, preserved for provenance
```

## Results summary

| Experiment | Result | Reproduce | Regression test |
|---|---|---|---|
| Gradient correctness | Commuting-gen formula matches autograd to 6.94e-17 | `python -m experiments.commuting_gradient_check` | `tests/test_gradients.py` |
| Baseline training | Batch GD reaches 0.900 test accuracy (per-sample SGD degrades to 0.567 -- kept as a documented negative baseline) | `python -m experiments.baseline_accuracy` | `tests/test_baselines.py` |
| Shot efficiency | Commuting-gen and parameter-shift reach identical 1.000 accuracy; 8:1 shot-cost ratio (60,000 vs 480,000 shots) | `python -m experiments.shot_efficiency` | `tests/test_shot_efficiency.py` |
| Entanglement | IsingZZ preserves exact gradients (1.11e-16) at n circuits/gradient while genuinely entangling (purity 0.9917); CNOT breaks the structure (4.15e-02 error) | `python -m experiments.entanglement_structure` | `tests/test_entanglement.py` |
| Trainability | Gradient variance follows Var = sin^2(x)/(2n^2), confirmed to n=14, no exponential concentration | `python -m experiments.trainability_scaling` | `tests/test_trainability.py` |

## Deviations from the original notebook

**Bug found and fixed: `scaling_plot.png`.** The original notebook's
`scaling_plot.png` (cells 6-7) was generated from a fudged/idealized shot-cost
model: both curves used the identical parameter-shift gradient computation,
just relabeled "Parameter-shift" and "Commuting" with a fabricated commuting
shot count. The notebook's own markdown later retracts this ("the cells above
were fudging the shot count"). The real, validated comparison — two genuinely
different gradient methods, real per-method shot costs, numerically identical
trajectories confirmed to <1e-9 — was cell 16, saved separately as
`honest_comparison.png`, and never referenced in the report despite being the
correct result. This repo's `scaling_plot.png` is now generated from that
honest computation (`experiments/shot_efficiency.py`), matching what the
report has always claimed the figure shows.

**Figure count: 9 -> 7.** `honest_comparison.png` is retired as a separate
filename now that `scaling_plot.png` carries the same (correct) data, and
`ratio_plot.png` (cells 8-10, built on the same fudged cost model, never cited
in the report) is dropped entirely.

**Kept intentionally:** the per-sample SGD baseline (which performs *worse*
than batch gradient descent, 0.733 -> 0.567) is preserved as an explicitly
labeled negative-result experiment in `experiments/baseline_accuracy.py`,
since it demonstrates why batch gradient descent is used everywhere else.

## Feedback

Questions and suggestions are welcome: open an issue on GitHub or email yuilonlam0607@gmail.com.

## License

MIT — see [LICENSE](LICENSE).

## AI assistance

Parts of this repository were written or changed with Claude, Anthropic's AI assistant. Affected code is marked in place with comments of the form `AI-assisted (Claude, <commit>)`; list them with `git grep -n "AI-assisted"`.

- `1987c21`: the package, experiment scripts, tests and packaging were ported with Claude from the original notebook, `notebooks/00_original_reference.ipynb`.
- `af2e63d`: docstrings and explanatory comments across the code.
- The commit after `af2e63d`: LICENSE, README fixes, and removal of a machine-specific `-e c:\users\...` line from `requirements-lock.txt`.
