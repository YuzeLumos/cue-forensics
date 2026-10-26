"""Reuse the existing capsule/bank/controller/decision implementation unchanged."""
from dataclasses import dataclass
import numpy as np
from cuefd.methods.evidence_lifecycle_v2 import (
    EvidenceCapsule, EvidenceBank, EvidenceBankScorer, EvidenceBankController,
    EvidenceLifecycleModel, EpochalWarrant,
)
from .core import JointWithinCandidate


def lift_existing_projected_sensor(projection, v, m, beta, *, bindings):
    C = np.asarray(projection.pca.components_, dtype=float)
    v = np.asarray(v, dtype=float)
    if C.shape != (256,1102) or v.shape != (256,) or not np.isfinite(v).all() or v @ v <= 0:
        raise ValueError('Invalid frozen projected sensor')
    if projection.pca.whiten:
        raise ValueError('This exact sensor lift requires the existing non-whitened PCA')
    if not np.allclose(C @ C.T, np.eye(256), atol=1e-10, rtol=1e-10):
        raise ValueError('PCA rows must be orthonormal for the prescribed center lift')
    direction = C.T @ v
    center = projection.pca.mean_ + C.T @ (float(m) * v / (v @ v))
    return EvidenceCapsule(v=direction, mbar=center, alpha=float(beta), bindings=bindings)


def existing_jpeg_controller(bank):
    return EvidenceBankController(bank, {'jpeg': EpochalWarrant(capsule_id=bank.capsules['jpeg'].capsule_id, discount=.25, off_threshold=.2, on_threshold=.8)})


@dataclass(frozen=True)
class _LiveState:
    core: object
    controller: EvidenceBankController


class AtomicJointLifecycle:
    """Single-writer facade: audit the old core, then stage next core and commit.

    No change to controller policy. Callers must not mutate live core/controller
    concurrently. Core forgetting deliberately does not replay audit history.
    """
    def __init__(self, core, projection, bank, controller):
        if bank.bank_id != controller.bank.bank_id: raise ValueError('Controller bank mismatch')
        B = np.column_stack([bank.capsules[name].v for name in bank.names])
        excluded = (projection.pca.components_ @ B).T @ core.basis
        if np.linalg.norm(excluded) > 1e-9 * max(1., np.linalg.norm(B)):
            raise ValueError('Every fixed bank direction must be excluded from the adaptive core')
        self.projection = projection; self.bank = bank
        self._state = _LiveState(core, controller)

    @property
    def core(self): return self._state.core

    @property
    def controller(self): return self._state.controller

    def predict(self, raw_features):
        state = self._state
        return EvidenceLifecycleModel(JointWithinCandidate(state.core, self.projection), self.bank, state.controller).predict(raw_features)

    def audit_and_update(self, audit_index, raw_features, labels, update_core):
        old = self._state
        pre_core, psi = EvidenceBankScorer(self.bank).components(JointWithinCandidate(old.core, self.projection), raw_features)
        controller = EvidenceBankController.from_bytes(old.controller.to_bytes(), self.bank)
        event = controller.audit(audit_index, pre_core, psi, labels)
        core = old.core.stage_update(update_core)
        # An immutable state reference is published only after BOTH operations succeed.
        self._state = _LiveState(core, controller)
        return event

    def forget_core_window(self, window_id):
        old = self._state
        core = old.core.stage_update(lambda c: c.forget_core_window(window_id))
        self._state = _LiveState(core, old.controller)
