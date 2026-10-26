from __future__ import annotations

import numpy as np
import pytest

from cuefd.methods.cuefd_v2 import (
    MomentCell,
    ProjectedMomentLedger,
    anchored_residual_scores,
    fit_cuefd_v2,
    fit_moment_head,
    freeze_kappa,
    score_sha256,
    solve_aw_geometry,
    validate_nested_support,
    weighted_cells,
)


def _rank_one_ledger(seed: int = 7) -> ProjectedMomentLedger:
    rng = np.random.default_rng(seed)
    ledger = ProjectedMomentLedger(dim=4)
    for window in ("w0", "w1", "w2"):
        shared = rng.normal(0.0, 0.05, size=(64, 4))
        shared[:, 1:] = 0.0
        ledger.add_samples(shared - np.array([1.0, 0.0, 0.0, 0.0]), 0, window)
        ledger.add_samples(shared + np.array([1.0, 0.0, 0.0, 0.0]), 1, window)
    return ledger


def test_single_window_rank_one_sanity() -> None:
    rng = np.random.default_rng(11)
    ledger = ProjectedMomentLedger(dim=5)
    shared = rng.normal(0.0, 0.03, size=(128, 5))
    shared[:, 1:] = 0.0
    ledger.add_samples(shared - np.array([0.8, 0.0, 0.0, 0.0, 0.0]), 0, "w0")
    ledger.add_samples(shared + np.array([0.8, 0.0, 0.0, 0.0, 0.0]), 1, "w0")
    solution = solve_aw_geometry(ledger, window_cap=np.inf)
    assert solution.rank == 1
    assert solution.orthogonality_residual < 1e-10


def test_rank_zero_boundary() -> None:
    samples = np.tile(np.array([[1.0, -2.0, 0.5]]), (32, 1))
    ledger = ProjectedMomentLedger(dim=3)
    ledger.add_samples(samples, 0, "w0")
    ledger.add_samples(samples, 1, "w0")
    state = fit_cuefd_v2(ledger, window_cap=64, geometry_lambda=1.0)
    assert state.geometry.rank == 0
    assert np.array_equal(state.score(samples), np.zeros(len(samples)))


def test_float64_cholesky_residual_gate() -> None:
    ledger = _rank_one_ledger()
    state = fit_cuefd_v2(ledger, window_cap=64, geometry_lambda=0.1)
    assert state.head.rank == 1
    assert state.head.jitter in (1e-6, 1e-5, 1e-4, 1e-3)
    assert state.head.relative_residual <= 1e-8
    assert state.head.coefficients.dtype == np.float64


def test_add_subtract_merge_endpoint_consistency() -> None:
    rng = np.random.default_rng(23)
    blocks = [
        MomentCell.from_samples(label, window, rng.normal(size=(17, 6)))
        for window in ("w0", "w1")
        for label in (0, 1)
    ]
    direct = ProjectedMomentLedger(dim=6)
    for block in blocks:
        direct.add_cell(block)

    left = ProjectedMomentLedger(dim=6)
    right = ProjectedMomentLedger(dim=6)
    for block in blocks[:2]:
        left.add_cell(block)
    for block in blocks[2:]:
        right.add_cell(block)
    merged = left.merge(right)
    for key in direct.cells:
        assert merged.cells[key].n == direct.cells[key].n
        assert np.array_equal(merged.cells[key].sum, direct.cells[key].sum)
        assert np.array_equal(merged.cells[key].cross, direct.cells[key].cross)

    restored = merged.copy()
    for block in blocks[2:]:
        restored.subtract_cell(block)
    for key in left.cells:
        assert restored.cells[key].n == left.cells[key].n
        assert np.allclose(restored.cells[key].sum, left.cells[key].sum)
        assert np.allclose(restored.cells[key].cross, left.cells[key].cross)


def test_time_zero_anchor_score_hash_exact() -> None:
    rng = np.random.default_rng(29)
    anchor = rng.normal(size=128)
    initial = rng.normal(size=128)
    current = initial.copy()
    combined = anchored_residual_scores(anchor, current, initial, kappa=0.75)
    assert np.array_equal(combined, anchor)
    assert score_sha256(combined) == score_sha256(anchor)


def test_kappa_validity_gate() -> None:
    head = np.linspace(-2.0, 2.0, 101)
    anchor = 3.0 * head + 0.5
    assert freeze_kappa(anchor, head) == pytest.approx(3.0)
    with pytest.raises(ValueError, match=r"Var\(h0\)"):
        freeze_kappa(anchor, np.zeros_like(head))
    with pytest.raises(ValueError, match="positive"):
        freeze_kappa(-anchor, head)


def test_nested_support_and_identity_disjoint() -> None:
    k64 = [f"id-{index}" for index in range(64)]
    k16 = k64[:16]
    query = [f"query-{index}" for index in range(100)]
    validate_nested_support(k16, k64, query)
    reordered = k64.copy()
    reordered[0], reordered[16] = reordered[16], reordered[0]
    with pytest.raises(ValueError, match="strict ordered prefix"):
        validate_nested_support(k16, reordered, query)
    with pytest.raises(ValueError, match="overlap"):
        validate_nested_support(k16, k64, query + [k64[-1]])


def test_window_cap_uses_cell_count_without_sample_duplication() -> None:
    rng = np.random.default_rng(31)
    ledger = ProjectedMomentLedger(dim=3)
    ledger.add_samples(rng.normal(size=(400, 3)), 0, "q00")
    ledger.add_samples(rng.normal(size=(400, 3)), 1, "q00")
    ledger.add_samples(rng.normal(size=(16, 3)), 0, "q01")
    ledger.add_samples(rng.normal(size=(16, 3)), 1, "q01")
    cells = {(cell.label, cell.window): cell for cell in weighted_cells(ledger, 64)}
    assert cells[(0, "q00")].alpha == pytest.approx(64 / 400)
    assert cells[(1, "q00")].n == pytest.approx(64)
    assert cells[(0, "q01")].alpha == pytest.approx(1.0)
    assert cells[(1, "q01")].n == pytest.approx(16)
    assert ledger.total_rows() == 832


def test_aw_geometry_matches_direct_weighted_moment_formula() -> None:
    samples = {
        (0, "w0"): np.array([[-2.0, 0.0], [-1.0, 1.0], [-1.5, -0.5]]),
        (1, "w0"): np.array([[1.0, 0.0], [2.0, 1.0]]),
        (0, "w1"): np.array([[-0.5, 2.0], [-1.0, 1.5]]),
        (1, "w1"): np.array([[0.5, 2.5], [1.0, 2.0], [1.5, 1.5], [2.0, 2.0]]),
    }
    ledger = ProjectedMomentLedger(dim=2)
    for (label, window), values in samples.items():
        ledger.add_samples(values, label, window)
    cap = 2.0
    solution = solve_aw_geometry(ledger, window_cap=cap)

    weighted = {}
    for key, values in samples.items():
        alpha = min(1.0, cap / len(values))
        weighted[key] = (alpha * len(values), alpha * values.sum(axis=0), alpha * values.T @ values)
    total_n = sum(item[0] for item in weighted.values())
    total_sum = sum((item[1] for item in weighted.values()), np.zeros(2))
    global_mean = total_sum / total_n
    expected_a = np.zeros((2, 2))
    expected_w = np.zeros((2, 2))
    for label in (0, 1):
        cells = [item for (cell_label, _), item in weighted.items() if cell_label == label]
        n = sum(item[0] for item in cells)
        summed = sum((item[1] for item in cells), np.zeros(2))
        cross = sum((item[2] for item in cells), np.zeros((2, 2)))
        difference = summed / n - global_mean
        expected_a += (n / total_n) * np.outer(difference, difference)
        expected_w += (cross - np.outer(summed, summed) / n) / total_n
    for window in ("w0", "w1"):
        negative = weighted[(0, window)]
        positive = weighted[(1, window)]
        difference = positive[1] / positive[0] - negative[1] / negative[0]
        expected_a += ((positive[0] + negative[0]) / total_n) * np.outer(difference, difference)
    assert np.allclose(solution.A, expected_a, atol=1e-12, rtol=1e-12)
    assert np.allclose(solution.W, expected_w, atol=1e-12, rtol=1e-12)
    assert np.allclose(solution.G, 2.0 * expected_w + 1e-3 * np.eye(2), atol=1e-12)


def test_moment_head_matches_direct_geometry_ridge_equations() -> None:
    rng = np.random.default_rng(37)
    ledger = ProjectedMomentLedger(dim=4)
    for window, count in (("q00", 80), ("q01", 16)):
        for label, shift in ((0, -0.7), (1, 0.7)):
            values = rng.normal(size=(count, 4))
            values[:, 0] += shift
            ledger.add_samples(values, label, window)
    geometry = solve_aw_geometry(ledger, window_cap=64)
    head = fit_moment_head(ledger, geometry, window_cap=64, geometry_lambda=1.0)
    cells = weighted_cells(ledger, 64)
    m = sum(cell.n for cell in cells)
    summed = sum((cell.sum for cell in cells), np.zeros(4))
    cross = sum((cell.cross for cell in cells), np.zeros((4, 4)))
    signed_count = sum(cell.n * (-1.0 if cell.label == 0 else 1.0) for cell in cells)
    signed_sum = sum(
        ((-1.0 if cell.label == 0 else 1.0) * cell.sum for cell in cells),
        np.zeros(4),
    )
    mean = summed / m
    mean_label = signed_count / m
    covariance = cross / m - np.outer(mean, mean)
    cross_xy = signed_sum / m - mean * mean_label
    reduced = geometry.basis.T @ covariance @ geometry.basis
    tau_x = max(float(np.trace(reduced) / geometry.rank), 1e-12)
    system = reduced + (1.0 + head.jitter) * tau_x * np.eye(geometry.rank)
    expected_coefficients = np.linalg.solve(system, geometry.basis.T @ cross_xy)
    expected_intercept = mean_label - expected_coefficients @ (geometry.basis.T @ mean)
    assert np.allclose(head.coefficients, expected_coefficients, atol=1e-11, rtol=1e-11)
    assert head.intercept == pytest.approx(expected_intercept, abs=1e-11, rel=1e-11)
    assert head.tau_x == pytest.approx(tau_x, abs=1e-12, rel=1e-12)


def test_failed_cholesky_residual_gate_stops_without_fallback() -> None:
    ledger = _rank_one_ledger()
    geometry = solve_aw_geometry(ledger, window_cap=64)
    with pytest.raises(RuntimeError, match="All deterministic Cholesky jitter attempts failed"):
        fit_moment_head(
            ledger,
            geometry,
            window_cap=64,
            geometry_lambda=0.1,
            residual_tolerance=-1.0,
        )


def test_projected_statistical_state_is_well_below_raw_v1_state() -> None:
    rng = np.random.default_rng(41)
    ledger = ProjectedMomentLedger(dim=256)
    for window in ("q00", "q10", "q11"):
        for label in (0, 1):
            ledger.add_samples(rng.normal(size=(16, 256)), label, window)
    state = fit_cuefd_v2(ledger, window_cap=64, geometry_lambda=1.0)
    assert state.state_bytes() < 10 * 1024 * 1024
    assert state.state_bytes() < 70_860_000
