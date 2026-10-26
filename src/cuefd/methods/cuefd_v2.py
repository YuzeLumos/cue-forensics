from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np


RankMode = Literal["adaptive", "fixed64", "fixed1"]


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return (matrix + matrix.T) * 0.5


@dataclass
class MomentCell:
    label: int
    window: str
    n: int
    sum: np.ndarray
    cross: np.ndarray

    @classmethod
    def from_samples(
        cls,
        label: int,
        window: str,
        samples: np.ndarray,
    ) -> "MomentCell":
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError("Moment samples must be a two-dimensional array.")
        return cls(
            label=int(label),
            window=str(window),
            n=int(len(samples)),
            sum=samples.sum(axis=0, dtype=np.float64),
            cross=symmetrize(samples.T @ samples),
        )

    @property
    def dim(self) -> int:
        return int(self.sum.shape[0])

    def copy(self) -> "MomentCell":
        return MomentCell(
            label=self.label,
            window=self.window,
            n=self.n,
            sum=self.sum.copy(),
            cross=self.cross.copy(),
        )


@dataclass
class ProjectedMomentLedger:
    dim: int
    cells: dict[tuple[int, str], MomentCell] = field(default_factory=dict)

    def add_samples(self, samples: np.ndarray, label: int, window: str) -> None:
        self.add_cell(MomentCell.from_samples(label, window, samples))

    def add_cell(self, incoming: MomentCell) -> None:
        self._validate_cell(incoming)
        if incoming.n == 0:
            return
        key = (incoming.label, incoming.window)
        current = self.cells.get(key)
        if current is None:
            self.cells[key] = incoming.copy()
            return
        current.n += incoming.n
        current.sum += incoming.sum
        current.cross = symmetrize(current.cross + incoming.cross)

    def subtract_cell(self, outgoing: MomentCell) -> None:
        self._validate_cell(outgoing)
        key = (outgoing.label, outgoing.window)
        current = self.cells.get(key)
        if current is None or outgoing.n > current.n:
            raise ValueError(f"Cannot subtract unavailable moment cell {key}.")
        current.n -= outgoing.n
        current.sum -= outgoing.sum
        current.cross = symmetrize(current.cross - outgoing.cross)
        if current.n == 0:
            if not np.allclose(current.sum, 0.0, atol=1e-10, rtol=0.0):
                raise ValueError(f"Moment subtraction left a nonzero sum for {key}.")
            if not np.allclose(current.cross, 0.0, atol=1e-9, rtol=0.0):
                raise ValueError(f"Moment subtraction left a nonzero cross for {key}.")
            del self.cells[key]

    def merge(self, other: "ProjectedMomentLedger") -> "ProjectedMomentLedger":
        if self.dim != other.dim:
            raise ValueError("Cannot merge ledgers with different dimensions.")
        merged = self.copy()
        for cell in other.cells.values():
            merged.add_cell(cell)
        return merged

    def copy(self) -> "ProjectedMomentLedger":
        return ProjectedMomentLedger(
            dim=self.dim,
            cells={key: value.copy() for key, value in self.cells.items()},
        )

    def total_rows(self) -> int:
        return int(sum(cell.n for cell in self.cells.values()))

    def state_bytes(self) -> int:
        return int(
            sum(
                np.dtype(np.int64).itemsize
                + cell.sum.nbytes
                + cell.cross.nbytes
                for cell in self.cells.values()
            )
        )

    def _validate_cell(self, cell: MomentCell) -> None:
        if cell.n < 0:
            raise ValueError("Moment count must be nonnegative.")
        if cell.label not in (0, 1):
            raise ValueError("CUE-FD v2 requires labels encoded as 0/1.")
        if cell.sum.shape != (self.dim,):
            raise ValueError("Moment sum dimension mismatch.")
        if cell.cross.shape != (self.dim, self.dim):
            raise ValueError("Moment cross dimension mismatch.")
        if not np.isfinite(cell.sum).all() or not np.isfinite(cell.cross).all():
            raise ValueError("Moment state contains non-finite values.")


@dataclass(frozen=True)
class WeightedCell:
    label: int
    window: str
    n: float
    sum: np.ndarray
    cross: np.ndarray
    alpha: float

    @property
    def mean(self) -> np.ndarray:
        return self.sum / self.n


@dataclass
class GeometrySolution:
    basis: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    A: np.ndarray
    W: np.ndarray
    G: np.ndarray
    orthogonality_residual: float
    rank_mode: RankMode


@dataclass
class MomentHead:
    basis: np.ndarray
    coefficients: np.ndarray
    intercept: float
    geometry_lambda: float
    jitter: float | None
    relative_residual: float
    rank: int
    tau_x: float

    def score(self, projected: np.ndarray) -> np.ndarray:
        projected = np.asarray(projected, dtype=np.float64)
        if projected.ndim != 2:
            raise ValueError("Projected samples must be a two-dimensional array.")
        if self.rank == 0:
            return np.zeros(len(projected), dtype=np.float64)
        return (
            (projected @ self.basis) @ self.coefficients + self.intercept
        ).astype(np.float64)


@dataclass
class CueFDV2State:
    ledger: ProjectedMomentLedger
    geometry: GeometrySolution
    head: MomentHead
    window_cap: float
    geometry_lambda: float

    def score(self, projected: np.ndarray) -> np.ndarray:
        return self.head.score(projected)

    def state_bytes(self) -> int:
        return int(
            self.ledger.state_bytes()
            + self.geometry.basis.nbytes
            + self.geometry.eigenvalues.nbytes
            + self.head.coefficients.nbytes
        )


def weighted_cells(
    ledger: ProjectedMomentLedger,
    window_cap: float,
) -> list[WeightedCell]:
    if window_cap <= 0:
        raise ValueError("Window cap must be positive.")
    result: list[WeightedCell] = []
    for key in sorted(ledger.cells):
        cell = ledger.cells[key]
        if cell.n <= 0:
            continue
        alpha = 1.0 if np.isinf(window_cap) else min(1.0, window_cap / cell.n)
        result.append(
            WeightedCell(
                label=cell.label,
                window=cell.window,
                n=float(alpha * cell.n),
                sum=alpha * cell.sum,
                cross=symmetrize(alpha * cell.cross),
                alpha=float(alpha),
            )
        )
    if not result:
        raise ValueError("CUE-FD v2 needs at least one nonempty moment cell.")
    return result


def _aggregate(cells: Iterable[WeightedCell], dim: int) -> tuple[float, np.ndarray, np.ndarray]:
    cells = list(cells)
    if not cells:
        return 0.0, np.zeros(dim, dtype=np.float64), np.zeros((dim, dim), dtype=np.float64)
    n = float(sum(cell.n for cell in cells))
    total_sum = np.sum([cell.sum for cell in cells], axis=0, dtype=np.float64)
    total_cross = np.sum([cell.cross for cell in cells], axis=0, dtype=np.float64)
    return n, total_sum, symmetrize(total_cross)


def solve_aw_geometry(
    ledger: ProjectedMomentLedger,
    window_cap: float,
    eta: float = 2.0,
    eps: float = 1e-3,
    relative_rank_threshold: float = 1e-6,
    max_rank: int = 64,
    rank_mode: RankMode = "adaptive",
) -> GeometrySolution:
    cells = weighted_cells(ledger, window_cap)
    dim = ledger.dim
    total_n, total_sum, _ = _aggregate(cells, dim)
    if total_n <= 0:
        raise ValueError("Effective moment count is zero.")
    global_mean = total_sum / total_n

    labels = sorted({cell.label for cell in cells})
    if labels != [0, 1]:
        raise ValueError("A+W geometry requires both labels.")
    A = np.zeros((dim, dim), dtype=np.float64)
    W = np.zeros((dim, dim), dtype=np.float64)
    for label in labels:
        label_cells = [cell for cell in cells if cell.label == label]
        label_n, label_sum, label_cross = _aggregate(label_cells, dim)
        if label_n <= 0:
            raise ValueError(f"Label {label} has zero effective weight.")
        difference = (label_sum / label_n - global_mean).reshape(-1, 1)
        A += (label_n / total_n) * (difference @ difference.T)
        W += (label_cross - np.outer(label_sum, label_sum) / label_n) / total_n

    for window in sorted({cell.window for cell in cells}):
        by_label = {
            label: [cell for cell in cells if cell.window == window and cell.label == label]
            for label in labels
        }
        if not all(by_label[label] for label in labels):
            continue
        negative_n, negative_sum, _ = _aggregate(by_label[0], dim)
        positive_n, positive_sum, _ = _aggregate(by_label[1], dim)
        window_n = negative_n + positive_n
        difference = (
            positive_sum / positive_n - negative_sum / negative_n
        ).reshape(-1, 1)
        A += (window_n / total_n) * (difference @ difference.T)

    A = symmetrize(A)
    W = symmetrize(W)
    G = symmetrize(float(eta) * W + float(eps) * np.eye(dim, dtype=np.float64))
    try:
        cholesky = np.linalg.cholesky(G)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("A+W geometry is not positive definite.") from error
    left = np.linalg.solve(cholesky, A)
    whitened = symmetrize(np.linalg.solve(cholesky, left.T).T)
    eigenvalues, eigenvectors = np.linalg.eigh(whitened)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)

    if rank_mode == "adaptive":
        positive = eigenvalues[eigenvalues > 0.0]
        if len(positive) == 0:
            rank = 0
        else:
            leading = float(positive[0])
            numeric_floor = (
                dim
                * np.finfo(np.float64).eps
                * max(leading, 1.0)
            )
            if leading <= numeric_floor:
                rank = 0
            else:
                threshold = max(relative_rank_threshold * leading, numeric_floor)
                rank = min(max_rank, int(np.sum(eigenvalues > threshold)))
    elif rank_mode == "fixed64":
        rank = min(max_rank, dim)
    elif rank_mode == "fixed1":
        rank = min(1, dim)
    else:
        raise ValueError(f"Unknown rank mode: {rank_mode}")

    if rank == 0:
        basis = np.zeros((dim, 0), dtype=np.float64)
        orthogonality = 0.0
    else:
        basis = np.linalg.solve(cholesky.T, eigenvectors[:, :rank])
        gram = symmetrize(basis.T @ G @ basis)
        orthogonality = float(
            np.linalg.norm(gram - np.eye(rank), ord="fro")
            / max(1.0, np.sqrt(rank))
        )
    return GeometrySolution(
        basis=basis,
        eigenvalues=eigenvalues,
        rank=int(rank),
        A=A,
        W=W,
        G=G,
        orthogonality_residual=orthogonality,
        rank_mode=rank_mode,
    )


def fit_moment_head(
    ledger: ProjectedMomentLedger,
    geometry: GeometrySolution,
    window_cap: float,
    geometry_lambda: float,
    jitter_schedule: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3),
    residual_tolerance: float = 1e-8,
) -> MomentHead:
    if geometry_lambda < 0:
        raise ValueError("Geometry strength must be nonnegative.")
    if geometry.rank == 0:
        return MomentHead(
            basis=geometry.basis,
            coefficients=np.zeros(0, dtype=np.float64),
            intercept=0.0,
            geometry_lambda=float(geometry_lambda),
            jitter=None,
            relative_residual=0.0,
            rank=0,
            tau_x=1e-12,
        )

    cells = weighted_cells(ledger, window_cap)
    dim = ledger.dim
    m, total_sum, total_cross = _aggregate(cells, dim)
    signed_count = float(sum(cell.n * (-1.0 if cell.label == 0 else 1.0) for cell in cells))
    signed_sum = np.sum(
        [(-1.0 if cell.label == 0 else 1.0) * cell.sum for cell in cells],
        axis=0,
        dtype=np.float64,
    )
    mean = total_sum / m
    mean_label = signed_count / m
    covariance = symmetrize(total_cross / m - np.outer(mean, mean))
    cross_xy = signed_sum / m - mean * mean_label

    basis = geometry.basis
    reduced_covariance = symmetrize(basis.T @ covariance @ basis)
    right_hand_side = basis.T @ cross_xy
    tau_x = max(float(np.trace(reduced_covariance) / geometry.rank), 1e-12)
    identity = np.eye(geometry.rank, dtype=np.float64)
    chosen_jitter: float | None = None
    solution: np.ndarray | None = None
    relative_residual = np.inf
    for jitter in jitter_schedule:
        system = symmetrize(
            reduced_covariance
            + (float(geometry_lambda) + float(jitter)) * tau_x * identity
        )
        try:
            cholesky = np.linalg.cholesky(system)
        except np.linalg.LinAlgError:
            continue
        candidate = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, right_hand_side),
        )
        denominator = max(float(np.linalg.norm(right_hand_side)), 1e-12)
        residual = float(
            np.linalg.norm(system @ candidate - right_hand_side) / denominator
        )
        if np.isfinite(residual) and residual <= residual_tolerance:
            chosen_jitter = float(jitter)
            solution = np.asarray(candidate, dtype=np.float64)
            relative_residual = residual
            break
    if solution is None or chosen_jitter is None:
        raise RuntimeError(
            "All deterministic Cholesky jitter attempts failed the residual gate."
        )
    intercept = float(mean_label - solution @ (basis.T @ mean))
    return MomentHead(
        basis=basis,
        coefficients=solution,
        intercept=intercept,
        geometry_lambda=float(geometry_lambda),
        jitter=chosen_jitter,
        relative_residual=relative_residual,
        rank=geometry.rank,
        tau_x=tau_x,
    )


def fit_cuefd_v2(
    ledger: ProjectedMomentLedger,
    window_cap: float,
    geometry_lambda: float,
    rank_mode: RankMode = "adaptive",
) -> CueFDV2State:
    geometry = solve_aw_geometry(
        ledger,
        window_cap=window_cap,
        rank_mode=rank_mode,
    )
    head = fit_moment_head(
        ledger,
        geometry=geometry,
        window_cap=window_cap,
        geometry_lambda=geometry_lambda,
    )
    return CueFDV2State(
        ledger=ledger.copy(),
        geometry=geometry,
        head=head,
        window_cap=float(window_cap),
        geometry_lambda=float(geometry_lambda),
    )


def freeze_kappa(anchor_scores: np.ndarray, initial_head_scores: np.ndarray) -> float:
    anchor_scores = np.asarray(anchor_scores, dtype=np.float64).reshape(-1)
    initial_head_scores = np.asarray(initial_head_scores, dtype=np.float64).reshape(-1)
    if anchor_scores.shape != initial_head_scores.shape:
        raise ValueError("Anchor and initial-head score arrays must align.")
    centered_anchor = anchor_scores - anchor_scores.mean()
    centered_head = initial_head_scores - initial_head_scores.mean()
    variance = float(np.mean(centered_head * centered_head))
    if variance <= 1e-12:
        raise ValueError("Invalid v2 config: Var(h0) <= 1e-12.")
    covariance = float(np.mean(centered_anchor * centered_head))
    kappa = covariance / max(variance, 1e-12)
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("Invalid v2 config: kappa must be positive.")
    return float(kappa)


def anchored_residual_scores(
    anchor_scores: np.ndarray,
    current_head_scores: np.ndarray,
    initial_head_scores: np.ndarray,
    kappa: float,
) -> np.ndarray:
    anchor_scores = np.asarray(anchor_scores, dtype=np.float64).reshape(-1)
    current_head_scores = np.asarray(current_head_scores, dtype=np.float64).reshape(-1)
    initial_head_scores = np.asarray(initial_head_scores, dtype=np.float64).reshape(-1)
    if not (
        anchor_scores.shape
        == current_head_scores.shape
        == initial_head_scores.shape
    ):
        raise ValueError("Anchored residual score arrays must align.")
    if kappa <= 0:
        raise ValueError("Anchored residual kappa must be positive.")
    return anchor_scores + float(kappa) * (
        current_head_scores - initial_head_scores
    )


def score_sha256(scores: np.ndarray) -> str:
    scores = np.asarray(scores, dtype="<f8").reshape(-1)
    return hashlib.sha256(scores.tobytes(order="C")).hexdigest()


def validate_nested_support(
    k16_identities: Iterable[str],
    k64_identities: Iterable[str],
    query_identities: Iterable[str],
) -> None:
    k16_ordered = list(map(str, k16_identities))
    k64_ordered = list(map(str, k64_identities))
    query_ordered = list(map(str, query_identities))
    if len(k16_ordered) != 16 or len(k64_ordered) != 64:
        raise ValueError("Nested support must contain exactly K=16 and K=64 identities.")
    if len(set(k16_ordered)) != len(k16_ordered) or len(set(k64_ordered)) != len(k64_ordered):
        raise ValueError("Nested support contains duplicate identities.")
    if len(set(query_ordered)) != len(query_ordered):
        raise ValueError("Query contains duplicate identities.")
    if k64_ordered[:16] != k16_ordered:
        raise ValueError("K=16 support is not the strict ordered prefix of K=64 support.")
    if set(k64_ordered) & set(query_ordered):
        raise ValueError("Support/query identity overlap.")
