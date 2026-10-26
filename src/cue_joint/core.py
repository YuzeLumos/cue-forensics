"""Moment-only engineering integration of the frozen joint-within estimator.

No tuning, controller policy, feature fitting, or evaluation metrics live here.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

D = 256
J = D + 1
SCHEMA = 'cue-joint-within-core-v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _readonly(value, shape):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Expected finite shape {shape}, got {a.shape}')
    # Immutable bytes backing prevents callers from re-enabling writes.
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(shape)


def _array(a):
    return dict(shape=list(a.shape), data=base64.b64encode(a.astype('<f8').tobytes()).decode())


def _restore(record):
    return np.frombuffer(base64.b64decode(record['data'], validate=True), dtype='<f8').reshape(record['shape'])


@dataclass(frozen=True)
class JointMomentCell:
    label: int
    n: int
    sum: np.ndarray
    cross: np.ndarray
    sample_ids: tuple[str, ...]

    def __post_init__(self):
        if self.label not in (0, 1) or not isinstance(self.n, (int, np.integer)) or self.n < 1:
            raise ValueError('Cells require binary label and positive integer count')
        ids = tuple(self.sample_ids)
        if len(ids) != self.n or len(set(ids)) != self.n or any(not isinstance(s, str) or not s for s in ids):
            raise ValueError('Every moment row requires a unique nonempty provenance identity')
        s = _readonly(self.sum, (J,)); q = _readonly(self.cross, (J, J))
        if not np.allclose(q, q.T, rtol=1e-12, atol=1e-12):
            raise ValueError('Second moment must be symmetric')
        object.__setattr__(self, 'sum', s); object.__setattr__(self, 'cross', q)
        object.__setattr__(self, 'n', int(self.n)); object.__setattr__(self, 'label', int(self.label))
        object.__setattr__(self, 'sample_ids', ids)

    @classmethod
    def from_samples(cls, label, joint_samples, sample_ids):
        u = np.asarray(joint_samples, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != J or not np.isfinite(u).all():
            raise ValueError('Joint samples must be finite [z,anchor/kappa] rows')
        return cls(label, len(u), u.sum(axis=0), u.T @ u, tuple(sample_ids))

    def scatter(self):
        s = self.cross - np.outer(self.sum, self.sum) / self.n
        return (s + s.T) / 2

    def record(self):
        return dict(label=self.label, n=self.n, sum=_array(self.sum), cross=_array(self.cross), sample_ids=list(self.sample_ids))

    @classmethod
    def from_record(cls, r):
        return cls(r['label'], r['n'], _restore(r['sum']), _restore(r['cross']), tuple(r['sample_ids']))


@dataclass(frozen=True)
class JointWindow:
    window_id: str
    cells: tuple[JointMomentCell, JointMomentCell]

    def __post_init__(self):
        if not isinstance(self.window_id, str) or not self.window_id:
            raise ValueError('A nonempty stable window ID is required')
        cells = tuple(sorted(self.cells, key=lambda c: c.label))
        if len(cells) != 2 or tuple(c.label for c in cells) != (0, 1):
            raise ValueError('A window must contain exactly one cell per binary class')
        if cells[0].n != cells[1].n:
            raise ValueError('Unequal class counts are outside the validated balanced formula')
        if set(cells[0].sample_ids) & set(cells[1].sample_ids):
            raise ValueError('One identity cannot enter both classes')
        object.__setattr__(self, 'cells', cells)

    @property
    def sample_ids(self):
        return frozenset(s for c in self.cells for s in c.sample_ids)

    def record(self):
        return dict(window_id=self.window_id, cells=[c.record() for c in self.cells])

    @classmethod
    def from_record(cls, r):
        return cls(r['window_id'], tuple(JointMomentCell.from_record(c) for c in r['cells']))

    @classmethod
    def from_samples(cls, window_id, z, anchor, labels, sample_ids, kappa):
        z = np.asarray(z, dtype=np.float64); anchor = np.asarray(anchor, dtype=np.float64)
        labels = np.asarray(labels); ids = np.asarray(sample_ids, dtype=object)
        if z.ndim != 2 or z.shape[1] != D or anchor.shape != (len(z),) or labels.shape != (len(z),) or ids.shape != (len(z),):
            raise ValueError('Mismatched sample shapes')
        if not np.isfinite(kappa) or kappa <= 0 or not np.isin(labels, [0, 1]).all():
            raise ValueError('Positive kappa and binary labels required')
        u = np.column_stack((z, anchor / kappa))
        return cls(window_id, tuple(JointMomentCell.from_samples(y, u[labels == y], ids[labels == y]) for y in (0, 1)))


@dataclass(frozen=True)
class JointHead:
    delta_weight: np.ndarray
    delta_bias: float
    diagnostics: Mapping

    def __post_init__(self):
        object.__setattr__(self, 'delta_weight', _readonly(self.delta_weight, (D,)))
        object.__setattr__(self, 'diagnostics', MappingProxyType(dict(self.diagnostics)))


class JointWithinCore:
    """History and current support occupy disjoint, explicit window stores.

    R is a fixed orthonormal basis in projected z coordinates. R=None is I.
    Empty R (256 by 0) permits only the fitted constant correction. Absence
    of current support always gives exactly zero correction, regardless of R.
    """
    def __init__(self, *, metric, kappa, lambda_prior, basis=None, bindings=None):
        self._metric = _readonly(metric, (D, D))
        if not np.allclose(self._metric, self._metric.T, rtol=1e-12, atol=1e-12):
            raise ValueError('Prior metric must be symmetric')
        np.linalg.cholesky(self._metric)
        r = np.eye(D) if basis is None else np.asarray(basis)
        if r.ndim != 2 or r.shape[0] != D or r.shape[1] > D:
            raise ValueError('Basis must have shape (256,r), 0<=r<=256')
        self._basis = _readonly(r, r.shape)
        if not np.allclose(r.T @ r, np.eye(r.shape[1]), atol=1e-10, rtol=1e-10):
            raise ValueError('Constraint basis must have orthonormal columns')
        if not np.isfinite(kappa) or kappa <= 0 or not np.isfinite(lambda_prior) or lambda_prior <= 0:
            raise ValueError('Positive finite frozen scales required')
        self._kappa = float(kappa); self._lambda_prior = float(lambda_prior)
        self._bindings = MappingProxyType(dict(bindings or {}))
        self._history = {}; self._current = None; self._head = None

    @property
    def kappa(self): return self._kappa

    @property
    def lambda_prior(self): return self._lambda_prior

    @property
    def basis(self): return self._basis

    @property
    def metric(self): return self._metric

    @property
    def bindings(self): return self._bindings

    @property
    def history(self): return MappingProxyType(self._history)

    @property
    def current(self): return self._current

    def _reject_overlap(self, window, others):
        if not isinstance(window, JointWindow): raise TypeError('Expected JointWindow')
        for other in others:
            if other is not None and (window.window_id == other.window_id or window.sample_ids & other.sample_ids):
                raise ValueError('Historical/current windows and provenance identities must be disjoint')

    def add_history_window(self, window):
        self._reject_overlap(window, [*self._history.values(), self._current])
        self._history[window.window_id] = window; self._head = None

    def remove_history_window(self, window_id):
        result = self._history.pop(window_id); self._head = None
        return result

    def forget_core_window(self, window_id):
        """Delete retained core moments only; controller audit history is untouched."""
        if self._current is not None and self._current.window_id == window_id:
            result = self._current; self.replace_current_window(None); return result
        return self.remove_history_window(window_id)

    def replace_current_window(self, window):
        if window is not None: self._reject_overlap(window, self._history.values())
        self._current = window; self._head = None

    def stage_update(self, update):
        """Return a solved clone, or raise with the original core untouched."""
        staged = type(self).from_bytes(self.to_bytes())
        update(staged)
        if not (staged.kappa == self.kappa and staged.lambda_prior == self.lambda_prior and dict(staged.bindings) == dict(self.bindings) and np.array_equal(staged.metric, self.metric) and np.array_equal(staged.basis, self.basis)):
            raise ValueError('Window updates cannot change frozen metric, basis, scales or bindings')
        staged.solve()
        return staged

    def advance_window(self, new_current, *, archive_current):
        """Explicit transition: archive old current once, or discard it.

        Validation/solve occur on a clone before the window stores commit.
        """
        if not isinstance(archive_current, bool): raise TypeError('Explicit boolean archive_current required')
        def transition(staged):
            previous = staged.current
            staged.replace_current_window(None)
            if archive_current and previous is not None: staged.add_history_window(previous)
            staged.replace_current_window(new_current)
        staged = self.stage_update(transition)
        self._history, self._current, self._head = staged._history, staged._current, staged._head

    def historical_scatter(self):
        scatter = np.zeros((J, J)); df = 0
        for key in sorted(self._history):
            for cell in self._history[key].cells:
                scatter += cell.scatter(); df += cell.n - 1
        return (scatter + scatter.T) / 2, df

    def solve(self):
        if self._head is not None: return self._head
        if self._current is None:
            self._head = JointHead(np.zeros(D), 0.0, dict(empty_current=True, current_n=0))
            return self._head
        cells = self._current.cells; n = sum(c.n for c in cells)
        mu = sum((c.sum for c in cells), np.zeros(J)) / n
        d = (cells[1].sum / cells[1].n - cells[0].sum / cells[0].n) / 2
        ss = sum((c.scatter() for c in cells), np.zeros((J, J)))
        sh, nu = self.historical_scatter()
        r = self._basis; reduced_dim = r.shape[1]
        T = np.zeros((J, reduced_dim + 1)); T[:D, :reduced_dim] = r; T[D, reduced_dim] = 1
        # Equivalent to mapping each joint cell with diag(R,1) before centering.
        W = T.T @ (ss + sh) @ T / (n + nu); W = (W + W.T) / 2
        contrast = T.T @ d; mean = T.T @ mu
        cz = contrast[:-1]; cg = contrast[-1]
        A = W[:-1, :-1] + np.outer(cz, cz) + self._lambda_prior / n * (r.T @ self._metric @ r)
        rhs = (1 - cg) * cz - W[:-1, -1]
        if reduced_dim:
            L = np.linalg.cholesky(A); reduced = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        else: reduced = np.zeros(0)
        dw = r @ reduced
        db = float(-mean[-1] - mean[:-1] @ reduced)  # balanced t mean equals zero
        residual = float(np.linalg.norm(A @ reduced - rhs) / max(np.linalg.norm(rhs), 1e-12))
        self._head = JointHead(dw, db, dict(empty_current=False, current_n=n, history_df=nu, moment_denominator=n + nu, reduced_dimension=reduced_dim, stationarity=residual, mean_score_error=float(abs(mu[-1] + mu[:-1] @ dw + db))))
        return self._head

    def score_projected(self, z, anchor):
        z = np.asarray(z, dtype=np.float64); anchor = np.asarray(anchor, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != D or anchor.shape != (len(z),) or not np.isfinite(z).all() or not np.isfinite(anchor).all():
            raise ValueError('Invalid projected score inputs')
        head = self.solve()
        return anchor + self._kappa * (z @ head.delta_weight + head.delta_bias)

    def _record(self):
        return dict(schema=SCHEMA, metric=_array(self._metric), basis=_array(self._basis), kappa=self._kappa, lambda_prior=self._lambda_prior, bindings=dict(self._bindings), history=[self._history[k].record() for k in sorted(self._history)], current=None if self._current is None else self._current.record())

    def to_bytes(self):
        record = self._record(); record['state_sha256'] = hashlib.sha256(_canonical(record)).hexdigest()
        return _canonical(record)

    @classmethod
    def from_bytes(cls, payload):
        r = json.loads(payload); digest = r.pop('state_sha256')
        if r.get('schema') != SCHEMA or hashlib.sha256(_canonical(r)).hexdigest() != digest:
            raise ValueError('Core schema or state checksum mismatch')
        core = cls(metric=_restore(r['metric']), basis=_restore(r['basis']), kappa=r['kappa'], lambda_prior=r['lambda_prior'], bindings=r['bindings'])
        for window in r['history']: core.add_history_window(JointWindow.from_record(window))
        if r['current'] is not None: core.replace_current_window(JointWindow.from_record(r['current']))
        if core.to_bytes() != payload: raise ValueError('Noncanonical state restore')
        return core

    def resource_counts(self):
        windows = [*self._history.values()] + ([] if self._current is None else [self._current])
        cells = [c for w in windows for c in w.cells]
        moments = sum(c.sum.nbytes + c.cross.nbytes + 8 for c in cells)
        fixed = self._metric.nbytes + self._basis.nbytes + 16
        cached = 0 if self._head is None else self._head.delta_weight.nbytes + 8
        return dict(historical_windows=len(self._history), current_windows=int(self._current is not None), cells=len(cells), history_rows=sum(c.n for w in self._history.values() for c in w.cells), current_rows=0 if self._current is None else sum(c.n for c in self._current.cells), moment_numeric_bytes=moments, fixed_numeric_bytes=fixed, cached_head_numeric_bytes=cached, total_numeric_payload_bytes=moments + fixed + cached, serialized_bytes=len(self.to_bytes()), provenance_utf8_bytes=sum(len(s.encode()) for c in cells for s in c.sample_ids), excluded='Shared feature encoder/projection/base, Python object overhead, solver temporaries, sensor bank/controller. Serialized size includes provenance metadata; cached derived head is reconstructed, not serialized.')


class JointWithinCandidate:
    """Internal adapter to the existing EvidenceLifecycleModel candidate API.

    It exposes the frozen projection for the existing evidence scorer. Actual
    sensor-bank construction/binding is external and deliberately not inferred.
    """
    def __init__(self, core, projection):
        self.core = core; self.projection = projection

    def score(self, raw_features):
        return self.core.score_projected(self.projection.transform(raw_features), self.projection.anchor.score_samples(raw_features))


def fixed_basis_from_sensor_directions(pca_components, directions):
    """ker((C B).T); constrains PCA coefficients, not raw-Pperp-then-PCA.

    All fixed directions are excluded independently of controller state.
    Rank uses the standard floating-point SVD tolerance, not a fitted threshold.
    """
    C = np.asarray(pca_components, dtype=float); B = np.asarray(directions, dtype=float)
    if C.shape != (D, 1102) or B.ndim != 2 or B.shape[0] != 1102 or not np.isfinite(C).all() or not np.isfinite(B).all():
        raise ValueError('Expected C=(256,1102) and B=(1102,sensors)')
    if B.shape[1] == 0: return np.eye(D)
    design = (C @ B).T
    _, singular, vt = np.linalg.svd(design, full_matrices=True)
    tolerance = max(design.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0)
    rank = int(np.count_nonzero(singular > tolerance))
    return vt[rank:].T.copy()
