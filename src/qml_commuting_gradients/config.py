"""Experiment hyperparameter configs.

Replaces the notebook's scattered/reused globals (``lr``, ``epochs``, ``seed``,
...) which took different values across five different training sections.
Each config below reproduces one section's values exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for a single training run."""

    seed: int
    epochs: int
    lr: float = 0.0
    lr_theta: float = 0.0
    lr_b: float = 0.0


PER_SAMPLE_SGD = TrainConfig(seed=42, epochs=30, lr=0.3)
BATCH_GD = TrainConfig(seed=42, epochs=60, lr=0.1)
CG_WITH_BIAS = TrainConfig(seed=42, epochs=60, lr_theta=0.3, lr_b=0.03)
THREE_ARM = TrainConfig(seed=42, epochs=100, lr_theta=0.3, lr_b=0.03)

TRAINABILITY_SEED = 1
TRAINABILITY_N_SAMPLES = 500
TRAINABILITY_X_VAL = 0.9
TRAINABILITY_N_LIST = [2, 4, 6, 8, 10, 12, 14]

SHOTS_PER_CIRCUIT = 1000
