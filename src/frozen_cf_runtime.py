"""Frozen official CF anchor and the unchanged production feature projection."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FrozenCFAnchor:
    weight: np.ndarray
    bias: float
    scaler: object

    def score_samples(self, raw_features):
        x = np.asarray(raw_features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 384 or not np.isfinite(x).all():
            raise ValueError('Official CF head expects finite raw 384-dimensional features')
        return x @ self.weight + self.bias
