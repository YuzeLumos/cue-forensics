from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from cuefd.methods.evidence_lifecycle_v2 import (
    EpochalWarrant,
    EvidenceBank,
    EvidenceBankController,
    EvidenceCapsule,
    EvidenceLifecycleModel,
    LifecycleScorer,
    WarrantedLifecycleModel,
    _normal_posterior_update,
    calibrate_evidence_capsule,
    fp95_count,
    fpr95,
    make_epistemic_decision,
)


def _capsule() -> EvidenceCapsule:
    v = np.zeros(1102, dtype=np.float64)
    v[0] = 1.0
    return EvidenceCapsule(
        v=v,
        mbar=np.zeros(1102, dtype=np.float64),
        alpha=2.0,
        bindings={"A0": "a" * 64, "core": "b" * 64},
    )


def _named_capsule(index: int, name: str) -> EvidenceCapsule:
    v = np.zeros(1102, dtype=np.float64)
    v[index] = 1.0
    return EvidenceCapsule(
        v=v,
        mbar=np.zeros(1102, dtype=np.float64),
        alpha=1.0,
        bindings={"channel": name, "construction": chr(97 + index) * 64},
    )


def _bank() -> EvidenceBank:
    return EvidenceBank(
        {
            "blur": _named_capsule(0, "blur"),
            "codec": _named_capsule(1, "codec"),
            "resampling": _named_capsule(2, "resampling"),
        }
    )


def _synthetic_scores(kind: str, n_per_label: int = 64):
    labels = np.asarray([0] * n_per_label + [1] * n_per_label, dtype=np.int64)
    off = np.zeros(len(labels), dtype=np.float64)
    if kind == "positive":
        on = (2.0 * labels - 1.0) * 4.0
    elif kind == "strong_positive":
        off = -(2.0 * labels - 1.0) * 4.0
        on = (2.0 * labels - 1.0) * 4.0
    elif kind == "negative":
        on = -(2.0 * labels - 1.0) * 4.0
    elif kind == "neutral":
        on = off.copy()
    else:
        raise ValueError(kind)
    return on, off, labels


def test_capsule_v1_schema_roundtrip_and_immutability():
    capsule = _capsule()
    payload = capsule.to_bytes()
    record = json.loads(payload)
    assert record["schema"] == "cuefd-immutable-blur-evidence-capsule-v1"
    restored = EvidenceCapsule.from_bytes(payload)
    assert restored.capsule_id == capsule.capsule_id
    assert restored.to_bytes() == payload
    assert not capsule.v.flags.writeable
    with pytest.raises(ValueError):
        capsule.v[0] = 2.0
    with pytest.raises(TypeError):
        capsule.bindings["core"] = "c" * 64


def test_gamma_one_returns_candidate_score_directly():
    class Candidate:
        def __init__(self):
            self.result = np.asarray([1.25, -0.5], dtype=np.float64)

        def score(self, x):
            return self.result

    candidate = Candidate()
    result = LifecycleScorer(_capsule()).score(candidate, np.zeros((2, 1102)), 1.0)
    assert result is candidate.result


def test_deployment_facade_cannot_return_a_bare_score():
    class Scaler:
        mean_ = np.zeros(1102, dtype=np.float64)
        scale_ = np.ones(1102, dtype=np.float64)

    class Anchor:
        scaler = Scaler()

    class Projection:
        anchor = Anchor()

    class Candidate:
        projection = Projection()

        def score(self, x):
            return np.asarray(x, dtype=np.float64)[:, 0]

    capsule = _capsule()
    warrant = EpochalWarrant(capsule_id=capsule.capsule_id, state="DORMANT")
    model = WarrantedLifecycleModel(Candidate(), capsule, warrant)
    x = np.zeros((2, 1102), dtype=np.float64)
    x[:, 0] = [1.0, -1.0]
    decision = model.predict(x)
    assert not hasattr(model, "score")
    assert decision.state == "DORMANT"
    assert np.all(decision.accusation_score <= decision.forensic_score)
    assert np.all(decision.accusation_score <= x[:, 0])


@pytest.mark.parametrize("state", ["PROVISIONAL", "DORMANT"])
def test_epistemic_nonactive_decision_is_lower_bounded(state: str):
    on = np.asarray([3.0, -2.0, 0.5, 0.0])
    off = np.asarray([1.0, -1.0, 0.5, 2.0])
    decision = make_epistemic_decision(state, on, off)
    expected_lower = np.minimum(on, off)
    assert np.array_equal(decision.forensic_score, off)
    assert np.array_equal(decision.lower, expected_lower)
    assert np.array_equal(decision.upper, np.maximum(on, off))
    assert np.array_equal(decision.accusation_score, expected_lower)
    assert np.all(decision.accusation_score <= on)
    assert np.all(decision.accusation_score <= off)
    assert not decision.accusation_score.flags.writeable


def test_epistemic_active_decision_is_all_on():
    on = np.asarray([3.0, -2.0, 0.5])
    off = np.asarray([1.0, -1.0, 2.0])
    decision = make_epistemic_decision("ACTIVE", on, off)
    for values in (
        decision.forensic_score,
        decision.lower,
        decision.upper,
        decision.accusation_score,
    ):
        assert np.array_equal(values, on)


def test_epoch_reset_reproduces_four_frozen_beliefs():
    # Firefly N then A-prime, starting a new epoch after retirement.
    mean, variance, firefly_n = _normal_posterior_update(
        0.0, 1.0, 0.0005732678385521549, 3.34120872382642e-05, 0.25
    )
    _, _, firefly_a_prime = _normal_posterior_update(
        mean, variance, 0.010056194603047446, 5.1025737201609836e-05, 0.25
    )
    # Ideogram N then A-prime under the same frozen warrant parameters.
    mean, variance, ideogram_n = _normal_posterior_update(
        0.0, 1.0, -0.0031400673185372923, 8.348025528534285e-05, 0.25
    )
    _, _, ideogram_a_prime = _normal_posterior_update(
        mean, variance, 0.00842191603078811, 6.216010970742194e-05, 0.25
    )
    assert firefly_n == pytest.approx(0.539500496725058, abs=1e-12)
    assert firefly_a_prime == pytest.approx(0.8894636814204144, abs=1e-12)
    assert ideogram_n == pytest.approx(0.36554696374372747, abs=1e-12)
    assert ideogram_a_prime == pytest.approx(0.8193088836999223, abs=1e-12)


def test_epochal_action_chain_and_reset_timing():
    warrant = EpochalWarrant(capsule_id=_capsule().capsule_id)
    event1 = warrant.audit(1, *_synthetic_scores("positive"))
    event2 = warrant.audit(2, *_synthetic_scores("negative"))
    event3 = warrant.audit(3, *_synthetic_scores("neutral"))
    event4 = warrant.audit(4, *_synthetic_scores("strong_positive"))
    assert [event["action"] for event in warrant.ledger] == [
        "BIRTH",
        "RETIRE",
        "HOLD",
        "REACTIVATE",
    ]
    assert event1["epoch_id"] == 0 and not event1["prior_reset"]
    assert event2["epoch_reset_pending_after"]
    assert event3["epoch_id"] == 1 and event3["prior_reset"]
    assert event3["prior_mean"] == 0.0 and event3["prior_variance"] == 1.0
    assert not event4["prior_reset"]
    warrant.verify_chain()


def test_restore_and_hash_chain_tamper_detection():
    warrant = EpochalWarrant(capsule_id=_capsule().capsule_id)
    warrant.audit(1, *_synthetic_scores("positive"))
    warrant.audit(2, *_synthetic_scores("negative"))
    payload = warrant.to_bytes()
    restored = EpochalWarrant.from_bytes(payload, warrant.capsule_id)
    assert restored.to_bytes() == payload
    restored.verify_chain()

    tampered = json.loads(payload)
    tampered["ledger"][0]["belief"] = 0.123
    tampered_payload = json.dumps(
        tampered,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    with pytest.raises(ValueError, match="hash"):
        EpochalWarrant.from_bytes(tampered_payload, warrant.capsule_id)


def test_api_is_identity_blind_and_fp95_is_count_exact():
    assert tuple(inspect.signature(EpochalWarrant.audit).parameters) == (
        "self",
        "audit_index",
        "scores_on",
        "scores_off",
        "labels",
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.asarray([0.8, 0.2, 0.1, 0.9, 0.85, 0.7], dtype=np.float64)
    assert fp95_count(labels, scores) == 1
    assert fpr95(labels, scores) == pytest.approx(1.0 / 3.0)


def test_strength_calibration_is_absorbed_into_a_new_immutable_capsule():
    capsule = _capsule()
    original = capsule.to_bytes()
    labels = np.asarray([0] * 64 + [1] * 64, dtype=np.int64)
    off = np.zeros(len(labels), dtype=np.float64)
    contribution = (2.0 * labels - 1.0) * 0.5
    calibrated, report = calibrate_evidence_capsule(
        capsule,
        off,
        contribution,
        labels,
        candidate_multipliers=(0.5, 1.0, 2.0, 3.0),
    )
    assert report["selected_multiplier"] == 3.0
    assert calibrated.alpha == pytest.approx(3.0 * capsule.alpha)
    assert calibrated.capsule_id != capsule.capsule_id
    assert capsule.to_bytes() == original
    assert EvidenceCapsule.from_bytes(calibrated.to_bytes()).to_bytes() == calibrated.to_bytes()


def test_strength_calibration_respects_exact_canary_risk_guard():
    capsule = _capsule()
    labels = np.asarray([0] * 20 + [1] * 20, dtype=np.int64)
    off = np.concatenate([np.zeros(20), np.full(20, 2.0)])
    contribution = np.concatenate([
        np.asarray([3.0] + [0.0] * 19),
        np.ones(20),
    ])
    calibrated, report = calibrate_evidence_capsule(
        capsule,
        off,
        contribution,
        labels,
        candidate_multipliers=(0.1, 1.0),
        stream_risk_limit_count=0,
        canary_scores_off=off,
        canary_contribution=contribution,
        canary_labels=labels,
        canary_risk_limit_count=0,
    )
    assert report["selected_multiplier"] == 0.1
    assert calibrated.alpha == pytest.approx(0.1 * capsule.alpha)
    rejected = next(item for item in report["candidates"] if item["multiplier"] == 1.0)
    assert rejected["admissible"] is False
    assert rejected["canary_fpr95_delta_count"] > 0


def test_evidence_bank_roundtrip_preserves_legacy_capsule_bytes():
    bank = _bank()
    payloads = {name: capsule.to_bytes() for name, capsule in bank.capsules.items()}
    payload = bank.to_bytes()
    restored = EvidenceBank.from_bytes(payload)
    assert restored.to_bytes() == payload
    assert restored.bank_id == bank.bank_id
    assert restored.names == ("blur", "codec", "resampling")
    assert {name: capsule.to_bytes() for name, capsule in restored.capsules.items()} == payloads
    with pytest.raises(TypeError):
        restored.capsules["other"] = _capsule()


def test_multi_evidence_decision_uses_exact_additive_envelope():
    bank = _bank()
    warrants = {
        "blur": EpochalWarrant(bank.capsules["blur"].capsule_id, state="ACTIVE"),
        "codec": EpochalWarrant(bank.capsules["codec"].capsule_id, state="DORMANT"),
        "resampling": EpochalWarrant(bank.capsules["resampling"].capsule_id),
    }
    controller = EvidenceBankController(bank, warrants)
    base = np.asarray([1.0, -1.0])
    contributions = {
        "blur": np.asarray([2.0, -2.0]),
        "codec": np.asarray([3.0, -3.0]),
        "resampling": np.asarray([-4.0, 4.0]),
    }
    decision = controller.decision(base, contributions)
    assert np.array_equal(decision.forensic_score, base + contributions["blur"])
    assert np.array_equal(
        decision.lower,
        base
        + contributions["blur"]
        + np.minimum(0.0, contributions["codec"])
        + np.minimum(0.0, contributions["resampling"]),
    )
    assert np.array_equal(
        decision.upper,
        base
        + contributions["blur"]
        + np.maximum(0.0, contributions["codec"])
        + np.maximum(0.0, contributions["resampling"]),
    )
    assert np.array_equal(decision.accusation_score, decision.lower)


def test_multi_evidence_audit_is_atomic_and_independent():
    bank = _bank()
    controller = EvidenceBankController(bank)
    labels = np.asarray([0] * 64 + [1] * 64, dtype=np.int64)
    positive = (2.0 * labels - 1.0) * 4.0
    negative = -positive
    neutral = np.zeros(len(labels), dtype=np.float64)
    base = np.zeros(len(labels), dtype=np.float64)

    event1 = controller.audit(
        1,
        base,
        {"blur": positive, "codec": positive, "resampling": positive},
        labels,
    )
    event2 = controller.audit(
        2,
        base,
        {"blur": positive, "codec": positive, "resampling": negative * 4.0},
        labels,
    )
    event3 = controller.audit(
        3,
        base,
        {"blur": negative * 4.0, "codec": positive, "resampling": neutral},
        labels,
    )
    event4 = controller.audit(
        4,
        base,
        {"blur": neutral, "codec": positive, "resampling": positive * 4.0},
        labels,
    )
    event5 = controller.audit(
        5,
        base,
        {"blur": positive * 4.0, "codec": positive, "resampling": positive},
        labels,
    )
    assert event1["states_after"] == {"blur": "ACTIVE", "codec": "ACTIVE", "resampling": "ACTIVE"}
    assert event2["events"]["resampling"]["action"] == "RETIRE"
    assert event3["events"]["blur"]["action"] == "RETIRE"
    assert event4["events"]["resampling"]["action"] == "REACTIVATE"
    assert event5["events"]["blur"]["action"] == "REACTIVATE"
    assert all(event["events"]["codec"]["action"] in {"BIRTH", "HOLD"} for event in [event1, event2, event3, event4, event5])
    assert controller.states == {"blur": "ACTIVE", "codec": "ACTIVE", "resampling": "ACTIVE"}
    controller.verify_chain()

    payload = controller.to_bytes()
    restored = EvidenceBankController.from_bytes(payload, bank)
    assert restored.to_bytes() == payload
    assert restored.states == controller.states

    before = controller.to_bytes()
    with pytest.raises(ValueError):
        controller.audit(
            6,
            base,
            {"blur": positive, "codec": positive[:-1], "resampling": positive},
            labels,
        )
    assert controller.to_bytes() == before


def test_multi_evidence_public_api_is_identity_blind_and_query_does_not_audit():
    assert tuple(inspect.signature(EvidenceBankController.audit).parameters) == (
        "self",
        "audit_index",
        "base_scores",
        "contributions",
        "labels",
    )

    class Scaler:
        mean_ = np.zeros(1102, dtype=np.float64)
        scale_ = np.ones(1102, dtype=np.float64)

    class Anchor:
        scaler = Scaler()

    class Projection:
        anchor = Anchor()

    class Candidate:
        projection = Projection()

        def score(self, x):
            return np.zeros(len(x), dtype=np.float64)

    bank = _bank()
    controller = EvidenceBankController(bank)
    model = EvidenceLifecycleModel(Candidate(), bank, controller)
    before = controller.to_bytes()
    decision = model.predict(np.zeros((4, 1102), dtype=np.float64))
    assert not hasattr(model, "score")
    assert set(decision.states) == set(bank.names)
    assert controller.to_bytes() == before
