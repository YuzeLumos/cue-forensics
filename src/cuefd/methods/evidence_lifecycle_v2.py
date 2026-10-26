from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


_CAPSULE_SCHEMA = "cuefd-immutable-blur-evidence-capsule-v1"
_WARRANT_EVENT_SCHEMA = "cuefd-epochal-warrant-event-v2"
_WARRANT_STATE_SCHEMA = "cuefd-epochal-warrant-state-v2"
_EVIDENCE_BANK_SCHEMA = "cuefd-immutable-evidence-bank-v1"
_BANK_EVENT_SCHEMA = "cuefd-evidence-bank-event-v1"
_BANK_STATE_SCHEMA = "cuefd-evidence-bank-controller-state-v1"
_STRENGTH_CALIBRATION_SCHEMA = "cuefd-evidence-strength-calibration-v1"
_VALID_STATES = frozenset({"PROVISIONAL", "ACTIVE", "DORMANT"})


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data_b64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _record_array(record: dict[str, Any]) -> np.ndarray:
    value = np.frombuffer(
        base64.b64decode(record["data_b64"]),
        dtype=np.dtype(record["dtype"]),
    ).reshape(record["shape"]).copy()
    value.setflags(write=False)
    return value


def _readonly_scores(value: np.ndarray, name: str) -> np.ndarray:
    scores = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError(f"{name} must be a nonempty finite score vector")
    scores.setflags(write=False)
    return scores


@dataclass(frozen=True)
class EvidenceCapsule:
    """Immutable evidence capsule preserving the byte schema used by v1."""

    v: np.ndarray
    mbar: np.ndarray
    alpha: float
    bindings: Mapping[str, str]
    capsule_id: str = field(init=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.v, dtype=np.float64).copy()
        mbar = np.asarray(self.mbar, dtype=np.float64).copy()
        if (
            v.shape != (1102,)
            or mbar.shape != (1102,)
            or not np.isfinite(v).all()
            or not np.isfinite(mbar).all()
            or not np.isfinite(self.alpha)
        ):
            raise ValueError("invalid evidence capsule")
        v.setflags(write=False)
        mbar.setflags(write=False)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "mbar", mbar)
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))
        object.__setattr__(self, "capsule_id", _sha256_bytes(_canonical(self._payload(False))))

    def _payload(self, include_id: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": _CAPSULE_SCHEMA,
            "v": _array_record(self.v),
            "mbar": _array_record(self.mbar),
            "alpha": float(self.alpha),
            "bindings": dict(sorted(self.bindings.items())),
        }
        if include_id:
            payload["capsule_id"] = self.capsule_id
        return payload

    def to_bytes(self) -> bytes:
        return _canonical(self._payload(include_id=True))

    @classmethod
    def from_bytes(cls, payload: bytes) -> "EvidenceCapsule":
        record = json.loads(payload)
        if record.get("schema") != _CAPSULE_SCHEMA:
            raise ValueError("unsupported evidence capsule schema")
        value = cls(
            v=_record_array(record["v"]),
            mbar=_record_array(record["mbar"]),
            alpha=float(record["alpha"]),
            bindings=dict(record["bindings"]),
        )
        if value.capsule_id != record.get("capsule_id") or value.to_bytes() != payload:
            raise ValueError("capsule byte/hash restore mismatch")
        return value

    def evidence(
        self,
        x: np.ndarray,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
    ) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        mean = np.asarray(scaler_mean, dtype=np.float64)
        scale = np.asarray(scaler_scale, dtype=np.float64)
        if matrix.shape[-1] != 1102 or mean.shape != (1102,) or scale.shape != (1102,):
            raise ValueError("capsule/scaler feature dimension mismatch")
        if not np.isfinite(matrix).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale == 0.0):
            raise ValueError("invalid capsule evidence inputs")
        standardized = (matrix - mean) / scale
        return float(self.alpha) * ((standardized - self.mbar) @ self.v)


class LifecycleScorer:
    """Internal on/off counterfactual primitive.

    Deployment code should use :class:`WarrantedLifecycleModel`, whose public
    prediction entry point returns an ``EpistemicDecision`` rather than a raw
    scalar score.
    """

    def __init__(self, capsule: EvidenceCapsule):
        self.capsule = capsule

    def score(self, candidate: Any, x: np.ndarray, gamma: float) -> np.ndarray:
        reference = candidate.score(x)
        if float(gamma) == 1.0:
            return reference
        evidence = self.capsule.evidence(
            x,
            candidate.projection.anchor.scaler.mean_,
            candidate.projection.anchor.scaler.scale_,
        )
        return reference + (float(gamma) - 1.0) * evidence


def _validated_binary_scores(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != values.shape or len(y) == 0 or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("labels and scores must contain both binary classes")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return y, values


def fp95_count(labels: np.ndarray, scores: np.ndarray) -> int:
    """Return the exact false-positive count at at least 95% TPR."""

    y, values = _validated_binary_scores(labels, scores)
    positive = values[y == 1]
    negative = values[y == 0]
    required_true_positives = int(np.ceil(0.95 * len(positive)))
    threshold = float(np.sort(positive)[-required_true_positives])
    return int(np.count_nonzero(negative >= threshold))


def fpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    y, values = _validated_binary_scores(labels, scores)
    return float(fp95_count(y, values) / np.count_nonzero(y == 0))


def _numeric_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    record = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data_sha256": _sha256_bytes(array.tobytes(order="C")),
    }
    return _sha256_bytes(_canonical(record))


def calibrate_evidence_capsule(
    capsule: EvidenceCapsule,
    scores_off: np.ndarray,
    contribution: np.ndarray,
    labels: np.ndarray,
    *,
    candidate_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    stream_risk_limit_count: int = 1,
    canary_scores_off: np.ndarray | None = None,
    canary_contribution: np.ndarray | None = None,
    canary_labels: np.ndarray | None = None,
    canary_risk_limit_count: int = 1,
) -> tuple[EvidenceCapsule, dict[str, Any]]:
    """Calibrate evidence strength once, before the capsule receives its final ID.

    Selection maximizes audited BCE gain over a fixed multiplier set while
    respecting exact FPR95 count guards on the audit stream and, when supplied,
    a stable canary.  The resulting multiplier is absorbed into ``alpha``;
    retirement and reactivation therefore reuse one byte-immutable capsule.
    """

    y, off = _validated_binary_scores(labels, scores_off)
    raw = _readonly_scores(contribution, "contribution")
    if raw.shape != off.shape:
        raise ValueError("calibration contribution shape mismatch")
    multipliers = tuple(sorted({float(value) for value in candidate_multipliers}))
    if not multipliers or any(not np.isfinite(value) or value <= 0.0 for value in multipliers):
        raise ValueError("candidate multipliers must be finite and positive")
    if int(stream_risk_limit_count) < 0 or int(canary_risk_limit_count) < 0:
        raise ValueError("risk limits must be nonnegative counts")

    canary_supplied = any(
        value is not None
        for value in (canary_scores_off, canary_contribution, canary_labels)
    )
    if canary_supplied and any(
        value is None
        for value in (canary_scores_off, canary_contribution, canary_labels)
    ):
        raise ValueError("canary scores, contribution, and labels must be supplied together")
    if canary_supplied:
        canary_y, canary_off = _validated_binary_scores(canary_labels, canary_scores_off)
        canary_raw = _readonly_scores(canary_contribution, "canary_contribution")
        if canary_raw.shape != canary_off.shape:
            raise ValueError("canary contribution shape mismatch")
    else:
        canary_y = canary_off = canary_raw = None

    off_loss = np.logaddexp(0.0, off) - y * off
    stream_fp_off = fp95_count(y, off)
    if canary_supplied:
        canary_fp_off = fp95_count(canary_y, canary_off)
    else:
        canary_fp_off = None

    candidates: list[dict[str, Any]] = []
    for multiplier in multipliers:
        scores_on = off + multiplier * raw
        gain = float(np.mean(off_loss - (np.logaddexp(0.0, scores_on) - y * scores_on)))
        stream_delta = int(fp95_count(y, scores_on) - stream_fp_off)
        if canary_supplied:
            canary_scores_on = canary_off + multiplier * canary_raw
            canary_delta = int(fp95_count(canary_y, canary_scores_on) - canary_fp_off)
        else:
            canary_delta = None
        admissible = bool(
            gain > 0.0
            and stream_delta <= int(stream_risk_limit_count)
            and (canary_delta is None or canary_delta <= int(canary_risk_limit_count))
        )
        candidates.append(
            {
                "multiplier": multiplier,
                "audited_bce_gain": gain,
                "stream_fpr95_delta_count": stream_delta,
                "canary_fpr95_delta_count": canary_delta,
                "admissible": admissible,
            }
        )

    admissible = [record for record in candidates if record["admissible"]]
    if not admissible:
        raise RuntimeError("no evidence strength passes gain and risk gates")
    selected = max(
        admissible,
        key=lambda record: (record["audited_bce_gain"], -record["multiplier"]),
    )
    source_record: dict[str, Any] = {
        "schema": _STRENGTH_CALIBRATION_SCHEMA,
        "source_capsule_id": capsule.capsule_id,
        "candidate_multipliers": list(multipliers),
        "stream_risk_limit_count": int(stream_risk_limit_count),
        "canary_risk_limit_count": int(canary_risk_limit_count) if canary_supplied else None,
        "scores_off_sha256": _numeric_array_sha256(off),
        "contribution_sha256": _numeric_array_sha256(raw),
        "labels_sha256": _numeric_array_sha256(y),
        "canary_scores_off_sha256": _numeric_array_sha256(canary_off) if canary_supplied else None,
        "canary_contribution_sha256": _numeric_array_sha256(canary_raw) if canary_supplied else None,
        "canary_labels_sha256": _numeric_array_sha256(canary_y) if canary_supplied else None,
        "candidates": candidates,
        "selected_multiplier": float(selected["multiplier"]),
    }
    calibration_id = _sha256_bytes(_canonical(source_record))
    bindings = dict(capsule.bindings)
    bindings["strength_calibration"] = calibration_id
    calibrated = EvidenceCapsule(
        v=capsule.v,
        mbar=capsule.mbar,
        alpha=float(capsule.alpha) * float(selected["multiplier"]),
        bindings=bindings,
    )
    report = {
        **source_record,
        "calibration_id": calibration_id,
        "calibrated_capsule_id": calibrated.capsule_id,
        "selected_audited_bce_gain": float(selected["audited_bce_gain"]),
        "selected_stream_fpr95_delta_count": int(selected["stream_fpr95_delta_count"]),
        "selected_canary_fpr95_delta_count": selected["canary_fpr95_delta_count"],
    }
    return calibrated, report


@dataclass(frozen=True)
class EpistemicDecision:
    state: str
    forensic_score: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    accusation_score: np.ndarray

    def __post_init__(self) -> None:
        if self.state not in _VALID_STATES:
            raise ValueError("invalid epistemic state")
        arrays = {
            "forensic_score": self.forensic_score,
            "lower": self.lower,
            "upper": self.upper,
            "accusation_score": self.accusation_score,
        }
        normalized = {name: _readonly_scores(value, name) for name, value in arrays.items()}
        shapes = {value.shape for value in normalized.values()}
        if len(shapes) != 1:
            raise ValueError("epistemic decision score shapes differ")
        if np.any(normalized["lower"] > normalized["upper"]):
            raise ValueError("invalid epistemic interval")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @property
    def interval(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower, self.upper


def make_epistemic_decision(
    state: str,
    scores_on: np.ndarray,
    scores_off: np.ndarray,
) -> EpistemicDecision:
    on = _readonly_scores(scores_on, "scores_on")
    off = _readonly_scores(scores_off, "scores_off")
    if on.shape != off.shape:
        raise ValueError("on/off score shapes differ")
    if state == "ACTIVE":
        forensic = on
        lower = on
        upper = on
        accusation = on
    elif state in {"DORMANT", "PROVISIONAL"}:
        forensic = off
        lower = np.minimum(on, off)
        upper = np.maximum(on, off)
        accusation = lower
    else:
        raise ValueError("invalid epistemic state")
    return EpistemicDecision(
        state=state,
        forensic_score=forensic,
        lower=lower,
        upper=upper,
        accusation_score=accusation,
    )


def _normal_posterior_update(
    prior_mean: float,
    prior_variance: float,
    observed_mean: float,
    observed_variance: float,
    discount: float,
) -> tuple[float, float, float]:
    prior_precision = float(discount / prior_variance)
    observation_precision = float(1.0 / observed_variance)
    new_variance = float(1.0 / (prior_precision + observation_precision))
    new_mean = float(
        new_variance
        * (prior_precision * prior_mean + observation_precision * observed_mean)
    )
    belief = float(NormalDist().cdf(new_mean / np.sqrt(new_variance)))
    return new_mean, new_variance, belief


@dataclass
class EpochalWarrant:
    capsule_id: str
    allow_reactivation: bool = True
    posterior_mean: float = 0.0
    posterior_variance: float = 1.0
    state: str = "PROVISIONAL"
    discount: float = 0.25
    on_threshold: float = 0.8
    off_threshold: float = 0.2
    epoch_id: int = 0
    epoch_reset_pending: bool = False
    ledger: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state not in _VALID_STATES:
            raise ValueError("invalid warrant state")
        if not self.capsule_id or not np.isfinite(self.posterior_mean) or not np.isfinite(self.posterior_variance) or self.posterior_variance <= 0.0:
            raise ValueError("invalid warrant posterior")
        if not (0.0 < self.discount <= 1.0 and 0.0 < self.off_threshold < self.on_threshold < 1.0):
            raise ValueError("invalid warrant thresholds or discount")
        if int(self.epoch_id) < 0:
            raise ValueError("invalid warrant epoch")

    @property
    def gamma(self) -> float:
        return 1.0 if self.state == "ACTIVE" else 0.0

    def decision(
        self,
        scores_on: np.ndarray,
        scores_off: np.ndarray,
    ) -> EpistemicDecision:
        return make_epistemic_decision(self.state, scores_on, scores_off)

    def audit(
        self,
        audit_index: int,
        scores_on: np.ndarray,
        scores_off: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        on = np.asarray(scores_on, dtype=np.float64).reshape(-1)
        off = np.asarray(scores_off, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if on.shape != off.shape or on.shape != y.shape:
            raise ValueError("invalid audit array shapes")
        _validated_binary_scores(y, on)
        _validated_binary_scores(y, off)
        if self.ledger and int(audit_index) <= int(self.ledger[-1]["audit_index"]):
            raise ValueError("audit indices must increase")

        prior_reset = bool(self.epoch_reset_pending)
        if prior_reset:
            prior_mean = 0.0
            prior_variance = 1.0
            self.epoch_id += 1
            self.epoch_reset_pending = False
        else:
            prior_mean = float(self.posterior_mean)
            prior_variance = float(self.posterior_variance)

        gain = (
            np.logaddexp(0.0, off)
            - y * off
            - np.logaddexp(0.0, on)
            + y * on
        )
        observed_mean = float(np.mean(gain))
        observed_variance = float(np.var(gain, ddof=1) / len(gain) + 1e-12)
        new_mean, new_variance, belief = _normal_posterior_update(
            prior_mean,
            prior_variance,
            observed_mean,
            observed_variance,
            self.discount,
        )

        fp_on = fp95_count(y, on)
        fp_off = fp95_count(y, off)
        real_count = int(np.count_nonzero(y == 0))
        risk_delta_count = int(fp_on - fp_off)
        risk_safe = bool(risk_delta_count <= 1)
        prior_state = self.state
        gamma_before = self.gamma
        action = "HOLD"
        if prior_state == "PROVISIONAL" and belief > self.on_threshold and risk_safe:
            self.state, action = "ACTIVE", "BIRTH"
        elif prior_state == "ACTIVE" and (belief < self.off_threshold or not risk_safe):
            self.state, action = "DORMANT", "RETIRE"
            self.epoch_reset_pending = True
        elif (
            prior_state == "DORMANT"
            and self.allow_reactivation
            and belief > self.on_threshold
            and risk_safe
        ):
            self.state, action = "ACTIVE", "REACTIVATE"

        self.posterior_mean = new_mean
        self.posterior_variance = new_variance
        event: dict[str, Any] = {
            "schema": _WARRANT_EVENT_SCHEMA,
            "audit_index": int(audit_index),
            "capsule_id": self.capsule_id,
            "epoch_id": int(self.epoch_id),
            "prior_reset": prior_reset,
            "prior_mean": prior_mean,
            "prior_variance": prior_variance,
            "prior_state": prior_state,
            "action": action,
            "state": self.state,
            "gamma_before": gamma_before,
            "gamma_after": self.gamma,
            "observed_gain": observed_mean,
            "observed_variance": observed_variance,
            "posterior_mean": new_mean,
            "posterior_variance": new_variance,
            "belief": belief,
            "fp95_on_count": fp_on,
            "fp95_off_count": fp_off,
            "risk_delta_count": risk_delta_count,
            "risk_limit_count": 1,
            "risk_delta": float(risk_delta_count / real_count),
            "risk_limit": float(1.0 / real_count),
            "risk_safe": risk_safe,
            "epoch_reset_pending_after": bool(self.epoch_reset_pending),
            "prev_hash": self.ledger[-1]["event_hash"] if self.ledger else "0" * 64,
        }
        event["event_hash"] = _sha256_bytes(_canonical(event))
        self.ledger.append(event)
        return dict(event)

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": _WARRANT_STATE_SCHEMA,
                "capsule_id": self.capsule_id,
                "allow_reactivation": self.allow_reactivation,
                "posterior_mean": self.posterior_mean,
                "posterior_variance": self.posterior_variance,
                "state": self.state,
                "discount": self.discount,
                "on_threshold": self.on_threshold,
                "off_threshold": self.off_threshold,
                "epoch_id": self.epoch_id,
                "epoch_reset_pending": self.epoch_reset_pending,
                "ledger": self.ledger,
            }
        )

    @classmethod
    def from_bytes(cls, payload: bytes, capsule_id: str) -> "EpochalWarrant":
        record = json.loads(payload)
        if record.get("schema") != _WARRANT_STATE_SCHEMA:
            raise ValueError("unsupported warrant schema")
        if record.get("capsule_id") != capsule_id:
            raise ValueError("capsule identity mismatch")
        value = cls(
            capsule_id=capsule_id,
            allow_reactivation=bool(record["allow_reactivation"]),
            posterior_mean=float(record["posterior_mean"]),
            posterior_variance=float(record["posterior_variance"]),
            state=str(record["state"]),
            discount=float(record["discount"]),
            on_threshold=float(record["on_threshold"]),
            off_threshold=float(record["off_threshold"]),
            epoch_id=int(record["epoch_id"]),
            epoch_reset_pending=bool(record["epoch_reset_pending"]),
            ledger=list(record["ledger"]),
        )
        value.verify_chain()
        if value.to_bytes() != payload:
            raise ValueError("warrant byte restore mismatch")
        return value

    def verify_chain(self) -> None:
        previous = "0" * 64
        previous_index: int | None = None
        for event in self.ledger:
            if event.get("schema") != _WARRANT_EVENT_SCHEMA:
                raise ValueError("ledger event schema mismatch")
            if event.get("capsule_id") != self.capsule_id:
                raise ValueError("ledger capsule mismatch")
            index = int(event["audit_index"])
            if previous_index is not None and index <= previous_index:
                raise ValueError("ledger audit order mismatch")
            if event.get("prev_hash") != previous:
                raise ValueError("ledger previous hash mismatch")
            without_hash = {key: value for key, value in event.items() if key != "event_hash"}
            if _sha256_bytes(_canonical(without_hash)) != event.get("event_hash"):
                raise ValueError("ledger event hash mismatch")
            previous = str(event["event_hash"])
            previous_index = index


class WarrantedLifecycleModel:
    """Deployment-safe facade: prediction always crosses the epistemic firewall."""

    def __init__(
        self,
        candidate: Any,
        capsule: EvidenceCapsule,
        warrant: EpochalWarrant,
    ) -> None:
        if warrant.capsule_id != capsule.capsule_id:
            raise ValueError("candidate warrant/capsule identity mismatch")
        self._candidate = candidate
        self._scorer = LifecycleScorer(capsule)
        self.warrant = warrant

    def predict(self, x: np.ndarray) -> EpistemicDecision:
        scores_on = self._scorer.score(self._candidate, x, 1.0)
        scores_off = self._scorer.score(self._candidate, x, 0.0)
        return self.warrant.decision(scores_on, scores_off)


@dataclass(frozen=True)
class EvidenceBank:
    """Immutable named collection of evidence capsules.

    Names are stable, human-readable handles. Evidence identity remains the
    capsule's content-derived ``capsule_id``. The existing capsule schema is
    embedded byte-for-byte, so legacy blur capsules remain directly reusable.
    """

    capsules: Mapping[str, EvidenceCapsule]
    bank_id: str = field(init=False)

    def __post_init__(self) -> None:
        normalized = dict(self.capsules)
        if not normalized or any(not isinstance(name, str) or not name for name in normalized):
            raise ValueError("evidence bank requires nonempty names")
        if any(not isinstance(capsule, EvidenceCapsule) for capsule in normalized.values()):
            raise TypeError("evidence bank values must be EvidenceCapsule instances")
        capsule_ids = [capsule.capsule_id for capsule in normalized.values()]
        if len(set(capsule_ids)) != len(capsule_ids):
            raise ValueError("evidence bank capsule identities must be unique")
        ordered = dict(sorted(normalized.items()))
        object.__setattr__(self, "capsules", MappingProxyType(ordered))
        object.__setattr__(self, "bank_id", _sha256_bytes(_canonical(self._payload(False))))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.capsules)

    def _payload(self, include_id: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": _EVIDENCE_BANK_SCHEMA,
            "capsules": [
                {
                    "name": name,
                    "capsule_id": capsule.capsule_id,
                    "payload_b64": base64.b64encode(capsule.to_bytes()).decode("ascii"),
                }
                for name, capsule in self.capsules.items()
            ],
        }
        if include_id:
            payload["bank_id"] = self.bank_id
        return payload

    def to_bytes(self) -> bytes:
        return _canonical(self._payload(True))

    @classmethod
    def from_bytes(cls, payload: bytes) -> "EvidenceBank":
        record = json.loads(payload)
        if record.get("schema") != _EVIDENCE_BANK_SCHEMA:
            raise ValueError("unsupported evidence bank schema")
        capsules: dict[str, EvidenceCapsule] = {}
        for item in record.get("capsules", []):
            capsule = EvidenceCapsule.from_bytes(base64.b64decode(item["payload_b64"]))
            if capsule.capsule_id != item.get("capsule_id"):
                raise ValueError("evidence bank capsule identity mismatch")
            name = str(item["name"])
            if name in capsules:
                raise ValueError("duplicate evidence bank name")
            capsules[name] = capsule
        value = cls(capsules)
        if value.bank_id != record.get("bank_id") or value.to_bytes() != payload:
            raise ValueError("evidence bank byte/hash restore mismatch")
        return value

    def contributions(
        self,
        x: np.ndarray,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        values = {
            name: _readonly_scores(
                capsule.evidence(x, scaler_mean, scaler_scale),
                f"contribution[{name}]",
            )
            for name, capsule in self.capsules.items()
        }
        return MappingProxyType(values)


class EvidenceBankScorer:
    """Compute a core score and every named evidence contribution once."""

    def __init__(self, bank: EvidenceBank):
        self.bank = bank

    def components(
        self,
        candidate: Any,
        x: np.ndarray,
    ) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
        core = _readonly_scores(candidate.score(x), "core_scores")
        contributions = self.bank.contributions(
            x,
            candidate.projection.anchor.scaler.mean_,
            candidate.projection.anchor.scaler.scale_,
        )
        if any(values.shape != core.shape for values in contributions.values()):
            raise ValueError("evidence bank contribution shape mismatch")
        return core, contributions


@dataclass(frozen=True)
class EvidenceDecision:
    bank_id: str
    states: Mapping[str, str]
    forensic_score: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    accusation_score: np.ndarray

    def __post_init__(self) -> None:
        states = dict(sorted(self.states.items()))
        if not self.bank_id or not states or any(state not in _VALID_STATES for state in states.values()):
            raise ValueError("invalid evidence decision identity or states")
        object.__setattr__(self, "states", MappingProxyType(states))
        arrays = {
            "forensic_score": self.forensic_score,
            "lower": self.lower,
            "upper": self.upper,
            "accusation_score": self.accusation_score,
        }
        normalized = {name: _readonly_scores(value, name) for name, value in arrays.items()}
        if len({value.shape for value in normalized.values()}) != 1:
            raise ValueError("evidence decision score shapes differ")
        if np.any(normalized["lower"] > normalized["upper"]):
            raise ValueError("invalid evidence decision interval")
        if not np.array_equal(normalized["accusation_score"], normalized["lower"]):
            raise ValueError("accusation score must use the conservative lower envelope")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @property
    def interval(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower, self.upper


def _validated_contributions(
    names: tuple[str, ...],
    contributions: Mapping[str, np.ndarray],
    expected_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    if set(contributions) != set(names):
        raise ValueError("evidence contribution names do not match the bank")
    normalized = {
        name: _readonly_scores(contributions[name], f"contribution[{name}]")
        for name in names
    }
    if any(values.shape != expected_shape for values in normalized.values()):
        raise ValueError("evidence contribution shape mismatch")
    return normalized


@dataclass
class EvidenceBankController:
    """Atomic, identity-blind controller for independent evidence warrants."""

    bank: EvidenceBank
    warrants: Mapping[str, EpochalWarrant] | None = None
    ledger: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.warrants is None:
            normalized = {
                name: EpochalWarrant(capsule_id=capsule.capsule_id)
                for name, capsule in self.bank.capsules.items()
            }
        else:
            normalized = dict(self.warrants)
        if set(normalized) != set(self.bank.names):
            raise ValueError("warrant names do not match the evidence bank")
        for name, warrant in normalized.items():
            if not isinstance(warrant, EpochalWarrant):
                raise TypeError("bank warrants must be EpochalWarrant instances")
            if warrant.capsule_id != self.bank.capsules[name].capsule_id:
                raise ValueError("bank warrant/capsule identity mismatch")
            warrant.verify_chain()
        self.warrants = dict(sorted(normalized.items()))
        self.verify_chain()

    @property
    def states(self) -> Mapping[str, str]:
        return MappingProxyType({name: warrant.state for name, warrant in self.warrants.items()})

    @property
    def gammas(self) -> Mapping[str, float]:
        return MappingProxyType({name: warrant.gamma for name, warrant in self.warrants.items()})

    def decision(
        self,
        base_scores: np.ndarray,
        contributions: Mapping[str, np.ndarray],
    ) -> EvidenceDecision:
        base = _readonly_scores(base_scores, "base_scores")
        values = _validated_contributions(self.bank.names, contributions, base.shape)
        forensic = base.copy()
        lower = base.copy()
        upper = base.copy()
        for name in self.bank.names:
            contribution = values[name]
            if self.warrants[name].state == "ACTIVE":
                forensic += contribution
                lower += contribution
                upper += contribution
            else:
                lower += np.minimum(0.0, contribution)
                upper += np.maximum(0.0, contribution)
        return EvidenceDecision(
            bank_id=self.bank.bank_id,
            states=self.states,
            forensic_score=forensic,
            lower=lower,
            upper=upper,
            accusation_score=lower,
        )

    def audit(
        self,
        audit_index: int,
        base_scores: np.ndarray,
        contributions: Mapping[str, np.ndarray],
        labels: np.ndarray,
    ) -> dict[str, Any]:
        base = _readonly_scores(base_scores, "base_scores")
        values = _validated_contributions(self.bank.names, contributions, base.shape)
        y, _ = _validated_binary_scores(labels, base)
        if self.ledger and int(audit_index) <= int(self.ledger[-1]["audit_index"]):
            raise ValueError("bank audit indices must increase")

        states_before = {name: warrant.state for name, warrant in self.warrants.items()}
        pre_score = base.copy()
        for name in self.bank.names:
            if self.warrants[name].state == "ACTIVE":
                pre_score += values[name]

        # Work on byte-exact clones. No live warrant is mutated until every
        # counterfactual audit succeeds, providing transaction-like semantics.
        staged: dict[str, EpochalWarrant] = {}
        events: dict[str, dict[str, Any]] = {}
        for name in self.bank.names:
            warrant = self.warrants[name]
            clone = EpochalWarrant.from_bytes(warrant.to_bytes(), warrant.capsule_id)
            off = pre_score - warrant.gamma * values[name]
            on = off + values[name]
            events[name] = clone.audit(audit_index, on, off, y)
            staged[name] = clone

        self.warrants = staged
        event: dict[str, Any] = {
            "schema": _BANK_EVENT_SCHEMA,
            "audit_index": int(audit_index),
            "bank_id": self.bank.bank_id,
            "states_before": states_before,
            "states_after": {name: warrant.state for name, warrant in staged.items()},
            "events": events,
            "prev_hash": self.ledger[-1]["event_hash"] if self.ledger else "0" * 64,
        }
        event["event_hash"] = _sha256_bytes(_canonical(event))
        self.ledger.append(event)
        return json.loads(_canonical(event))

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": _BANK_STATE_SCHEMA,
                "bank_id": self.bank.bank_id,
                "warrants": {
                    name: base64.b64encode(warrant.to_bytes()).decode("ascii")
                    for name, warrant in self.warrants.items()
                },
                "ledger": self.ledger,
            }
        )

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        bank: EvidenceBank,
    ) -> "EvidenceBankController":
        record = json.loads(payload)
        if record.get("schema") != _BANK_STATE_SCHEMA or record.get("bank_id") != bank.bank_id:
            raise ValueError("evidence bank controller identity/schema mismatch")
        warrants = {
            name: EpochalWarrant.from_bytes(
                base64.b64decode(encoded),
                bank.capsules[name].capsule_id,
            )
            for name, encoded in record.get("warrants", {}).items()
        }
        value = cls(bank=bank, warrants=warrants, ledger=list(record.get("ledger", [])))
        if value.to_bytes() != payload:
            raise ValueError("evidence bank controller byte restore mismatch")
        return value

    def verify_chain(self) -> None:
        previous = "0" * 64
        previous_index: int | None = None
        for event in self.ledger:
            if event.get("schema") != _BANK_EVENT_SCHEMA or event.get("bank_id") != self.bank.bank_id:
                raise ValueError("bank ledger identity/schema mismatch")
            index = int(event["audit_index"])
            if previous_index is not None and index <= previous_index:
                raise ValueError("bank ledger audit order mismatch")
            if event.get("prev_hash") != previous:
                raise ValueError("bank ledger previous hash mismatch")
            without_hash = {key: value for key, value in event.items() if key != "event_hash"}
            if _sha256_bytes(_canonical(without_hash)) != event.get("event_hash"):
                raise ValueError("bank ledger event hash mismatch")
            previous = str(event["event_hash"])
            previous_index = index


class EvidenceLifecycleModel:
    """Deployment facade for a multi-evidence bank; intentionally has no score API."""

    def __init__(
        self,
        candidate: Any,
        bank: EvidenceBank,
        controller: EvidenceBankController,
    ) -> None:
        if controller.bank.bank_id != bank.bank_id:
            raise ValueError("model controller/evidence bank identity mismatch")
        self._candidate = candidate
        self._scorer = EvidenceBankScorer(bank)
        self.controller = controller

    def predict(self, x: np.ndarray) -> EvidenceDecision:
        base, contributions = self._scorer.components(self._candidate, x)
        return self.controller.decision(base, contributions)


__all__ = [
    "EvidenceBank",
    "EvidenceBankController",
    "EvidenceBankScorer",
    "EvidenceCapsule",
    "EvidenceDecision",
    "EvidenceLifecycleModel",
    "EpochalWarrant",
    "EpistemicDecision",
    "WarrantedLifecycleModel",
    "calibrate_evidence_capsule",
    "make_epistemic_decision",
    "fpr95",
    "fp95_count",
]
