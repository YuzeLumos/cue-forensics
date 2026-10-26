from __future__ import annotations

import copy
import io
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from cuefd.methods.baselines import HardNegativeLogisticScorer
from cuefd.methods.cuefd_v2 import (
    CueFDV2State,
    ProjectedMomentLedger,
    anchored_residual_scores,
    fit_cuefd_v2,
    freeze_kappa,
)


@dataclass
class FixedAnchorProjection:
    anchor: HardNegativeLogisticScorer
    pca: PCA
    projection_dim: int
    seed: int

    @classmethod
    def fit(
        cls,
        anchor: HardNegativeLogisticScorer,
        x: np.ndarray,
        projection_dim: int,
        seed: int,
    ) -> "FixedAnchorProjection":
        if anchor.estimator is None:
            raise RuntimeError("Cannot fit P from an unfitted shared anchor.")
        standardized = anchor.scaler.transform(np.asarray(x, dtype=np.float64))
        dimension = min(int(projection_dim), standardized.shape[0] - 1, standardized.shape[1])
        if dimension <= 0:
            raise ValueError("Fixed Q00 projection has nonpositive dimension.")
        pca = PCA(n_components=dimension, random_state=int(seed))
        pca.fit(standardized)
        return cls(anchor=anchor, pca=pca, projection_dim=dimension, seed=int(seed))

    def transform(self, x: np.ndarray) -> np.ndarray:
        standardized = self.anchor.scaler.transform(np.asarray(x, dtype=np.float64))
        projected = self.pca.transform(standardized)
        if projected.shape[1] != self.projection_dim or not np.isfinite(projected).all():
            raise RuntimeError("Fixed Q00 projection output gate failed.")
        return np.asarray(projected, dtype=np.float64)


@dataclass
class CandidateResource:
    rank: int
    jitter: float | None
    linear_residual: float
    orthogonality_residual: float
    projected_statistical_state_bytes: int
    method_state_bytes: int


@dataclass
class CueFDV21Candidate:
    projection: FixedAnchorProjection
    ledger: ProjectedMomentLedger
    initial_state: CueFDV2State
    current_state: CueFDV2State
    kappa: float
    window_cap: float
    geometry_lambda: float

    @classmethod
    def initialize(
        cls,
        *,
        projection: FixedAnchorProjection,
        anchor_x: np.ndarray,
        anchor_y: np.ndarray,
        validation_x: np.ndarray,
        window_cap: float,
        geometry_lambda: float,
    ) -> "CueFDV21Candidate":
        projected_anchor = projection.transform(anchor_x)
        anchor_y = np.asarray(anchor_y, dtype=np.int64)
        ledger = ProjectedMomentLedger(dim=projection.projection_dim)
        for label in (0, 1):
            selected = projected_anchor[anchor_y == label]
            if len(selected) == 0:
                raise RuntimeError("Initial Q00 moments lack one of the labels.")
            ledger.add_samples(selected, label, "anonymous_window_00")
        initial = fit_cuefd_v2(
            ledger,
            window_cap=float(window_cap),
            geometry_lambda=float(geometry_lambda),
            rank_mode="adaptive",
        )
        projected_validation = projection.transform(validation_x)
        anchor_validation = projection.anchor.score_samples(validation_x)
        initial_validation = initial.score(projected_validation)
        kappa = freeze_kappa(anchor_validation, initial_validation)
        return cls(
            projection=projection,
            ledger=ledger,
            initial_state=initial,
            current_state=copy.deepcopy(initial),
            kappa=kappa,
            window_cap=float(window_cap),
            geometry_lambda=float(geometry_lambda),
        )

    def clone(self) -> "CueFDV21Candidate":
        return copy.deepcopy(self)

    def score(self, x: np.ndarray) -> np.ndarray:
        projected = self.projection.transform(x)
        anchor_scores = self.projection.anchor.score_samples(x)
        initial_scores = self.initial_state.score(projected)
        current_scores = self.current_state.score(projected)
        return anchored_residual_scores(
            anchor_scores, current_scores, initial_scores, self.kappa
        )

    def update(self, x: np.ndarray, y: np.ndarray, window: str) -> CandidateResource:
        projected = self.projection.transform(x)
        y = np.asarray(y, dtype=np.int64)
        for label in (0, 1):
            selected = projected[y == label]
            if len(selected) == 0:
                raise RuntimeError("Candidate update lacks one of the labels.")
            self.ledger.add_samples(selected, label, str(window))
        self.current_state = fit_cuefd_v2(
            self.ledger,
            window_cap=self.window_cap,
            geometry_lambda=self.geometry_lambda,
            rank_mode="adaptive",
        )
        return self.resource()

    def resource(self) -> CandidateResource:
        projected_bytes, method_bytes = serialized_candidate_state_bytes(self)
        return CandidateResource(
            rank=int(self.current_state.geometry.rank),
            jitter=self.current_state.head.jitter,
            linear_residual=float(self.current_state.head.relative_residual),
            orthogonality_residual=float(
                self.current_state.geometry.orthogonality_residual
            ),
            projected_statistical_state_bytes=projected_bytes,
            method_state_bytes=method_bytes,
        )


def _npz_size(arrays: dict[str, np.ndarray]) -> int:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return int(buffer.tell())


def serialized_candidate_state_bytes(candidate: CueFDV21Candidate) -> tuple[int, int]:
    ledger_arrays: dict[str, np.ndarray] = {}
    for index, key in enumerate(sorted(candidate.ledger.cells)):
        cell = candidate.ledger.cells[key]
        ledger_arrays[f"cell_{index:02d}_label"] = np.asarray(cell.label, dtype=np.int64)
        ledger_arrays[f"cell_{index:02d}_window"] = np.asarray(cell.window, dtype="<U32")
        ledger_arrays[f"cell_{index:02d}_n"] = np.asarray(cell.n, dtype=np.int64)
        ledger_arrays[f"cell_{index:02d}_sum"] = np.asarray(cell.sum, dtype=np.float64)
        ledger_arrays[f"cell_{index:02d}_cross"] = np.asarray(cell.cross, dtype=np.float64)
    projected_bytes = _npz_size(ledger_arrays)
    if candidate.projection.anchor.estimator is None:
        raise RuntimeError("Candidate anchor estimator is missing.")
    method_arrays = {
        **ledger_arrays,
        "anchor_mean": np.asarray(candidate.projection.anchor.scaler.mean_, dtype=np.float64),
        "anchor_scale": np.asarray(candidate.projection.anchor.scaler.scale_, dtype=np.float64),
        "anchor_coef": np.asarray(candidate.projection.anchor.estimator.coef_, dtype=np.float64),
        "anchor_intercept": np.asarray(
            candidate.projection.anchor.estimator.intercept_, dtype=np.float64
        ),
        "pca_mean": np.asarray(candidate.projection.pca.mean_, dtype=np.float64),
        "pca_components": np.asarray(
            candidate.projection.pca.components_, dtype=np.float64
        ),
        "geometry_basis": np.asarray(
            candidate.current_state.geometry.basis, dtype=np.float64
        ),
        "geometry_eigenvalues": np.asarray(
            candidate.current_state.geometry.eigenvalues, dtype=np.float64
        ),
        "head_coefficients": np.asarray(
            candidate.current_state.head.coefficients, dtype=np.float64
        ),
        "head_intercept": np.asarray(
            candidate.current_state.head.intercept, dtype=np.float64
        ),
        "kappa": np.asarray(candidate.kappa, dtype=np.float64),
        "window_cap": np.asarray(candidate.window_cap, dtype=np.float64),
        "geometry_lambda": np.asarray(
            candidate.geometry_lambda, dtype=np.float64
        ),
    }
    return projected_bytes, _npz_size(method_arrays)
