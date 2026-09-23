"""Iris data loading and preprocessing.

Unifies the two near-identical Iris-loading blocks from the original notebook
(binary Iris scaled to ``(0, pi)`` and, separately, to ``(0, pi/2)``) into a
single parameterized loader.
"""

# AI-assisted (Claude, 1987c21): ported with Claude from notebooks/00_original_reference.ipynb.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class IrisSplit:
    """Train/test split of the binary (classes 0 vs 1) Iris dataset."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_iris_binary(
    feature_range: tuple[float, float],
    test_size: float = 0.3,
    random_state: int = 42,
) -> IrisSplit:
    """Load Iris, keep classes 0/1 relabelled to +-1, and scale features into feature_range.

    ``feature_range=(0, np.pi)`` reproduces the original wide-range encoding;
    ``feature_range=(0, np.pi / 2)`` reproduces the commuting-generator encoding,
    chosen so that sin(x) stays monotone across the encoded range.
    """
    X, y = load_iris(return_X_y=True)
    mask = y < 2
    X, y = X[mask], y[mask]
    y = np.where(y == 0, -1, 1)

    scaler = MinMaxScaler(feature_range=feature_range)
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return IrisSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
