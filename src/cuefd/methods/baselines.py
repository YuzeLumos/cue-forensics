from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Scorer(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Scorer": ...

    def score_samples(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class BaselineSpec:
    name: str
    display_name: str
    estimator: Scorer


class SklearnDecisionScorer:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SklearnDecisionScorer":
        self.estimator.fit(x, y)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(x)
            return np.asarray(scores, dtype="float64").reshape(-1)
        probs = self.estimator.predict_proba(x)
        return np.asarray(probs[:, 1], dtype="float64")


class HardNegativeLogisticScorer:
    """Logistic regression with extra weight on real samples in the high-score tail."""

    def __init__(
        self,
        random_state: int = 2026,
        tail_fraction: float = 0.15,
        tail_weight: float = 5.0,
        iterations: int = 2,
        c_value: float = 1.0,
    ) -> None:
        self.random_state = random_state
        self.tail_fraction = tail_fraction
        self.tail_weight = tail_weight
        self.iterations = iterations
        self.c_value = c_value
        self.scaler = StandardScaler()
        self.estimator: LogisticRegression | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HardNegativeLogisticScorer":
        z = self.scaler.fit_transform(np.asarray(x, dtype="float64"))
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("HardNegativeLogisticScorer requires both real=0 and fake=1 labels.")
        weights = balanced_sample_weights(y)
        for _ in range(max(1, self.iterations)):
            estimator = LogisticRegression(
                C=self.c_value,
                max_iter=3000,
                random_state=self.random_state,
            )
            estimator.fit(z, y, sample_weight=weights)
            scores = np.asarray(estimator.decision_function(z), dtype="float64").reshape(-1)
            hard_real = high_score_real_tail(scores, y, self.tail_fraction)
            weights = balanced_sample_weights(y)
            weights[hard_real] *= self.tail_weight
            self.estimator = estimator
        if self.estimator is None:
            raise RuntimeError("HardNegativeLogisticScorer failed to fit an estimator.")
        self.estimator.fit(z, y, sample_weight=weights)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("HardNegativeLogisticScorer is not fitted.")
        z = self.scaler.transform(np.asarray(x, dtype="float64"))
        return np.asarray(self.estimator.decision_function(z), dtype="float64").reshape(-1)


class NearestCentroidScorer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.centroids: dict[int, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroidScorer":
        z = self.scaler.fit_transform(x)
        self.centroids = {
            int(label): z[y == label].mean(axis=0)
            for label in np.unique(y)
        }
        if 0 not in self.centroids or 1 not in self.centroids:
            raise ValueError("NearestCentroidScorer requires both real=0 and fake=1 in training labels.")
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(x)
        d_real = np.linalg.norm(z - self.centroids[0], axis=1)
        d_fake = np.linalg.norm(z - self.centroids[1], axis=1)
        return d_real - d_fake


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    weights = np.ones(len(y), dtype="float64")
    for label in np.unique(y):
        count = int(np.sum(y == label))
        if count:
            weights[y == label] = len(y) / (len(np.unique(y)) * count)
    return weights


def high_score_real_tail(scores: np.ndarray, y: np.ndarray, tail_fraction: float) -> np.ndarray:
    scores = np.asarray(scores, dtype="float64")
    y = np.asarray(y, dtype=int)
    real_mask = y == 0
    if real_mask.sum() == 0:
        return np.zeros(len(y), dtype=bool)
    fraction = float(np.clip(tail_fraction, 0.0, 1.0))
    if fraction <= 0:
        return np.zeros(len(y), dtype=bool)
    cutoff = np.quantile(scores[real_mask], 1.0 - fraction)
    return real_mask & (scores >= cutoff)


def default_baselines(random_state: int = 2026) -> list[BaselineSpec]:
    return [
        BaselineSpec(
            name="logreg",
            display_name="Logistic Regression",
            estimator=SklearnDecisionScorer(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=2000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                )
            ),
        ),
        BaselineSpec(
            name="hardneg_logreg",
            display_name="Hard-Negative Logistic",
            estimator=HardNegativeLogisticScorer(
                random_state=random_state,
                tail_fraction=0.15,
                tail_weight=5.0,
                iterations=2,
                c_value=1.0,
            ),
        ),
        BaselineSpec(
            name="lda_shrinkage",
            display_name="LDA Shrinkage",
            estimator=SklearnDecisionScorer(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                    ]
                )
            ),
        ),
        BaselineSpec(
            name="nearest_centroid",
            display_name="Nearest Centroid",
            estimator=NearestCentroidScorer(),
        ),
    ]
