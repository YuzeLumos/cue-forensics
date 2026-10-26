import numpy as np
import pytest

from cue_joint.core import JointWithinCore, JointWindow
from cue_joint.lifecycle import AtomicJointLifecycle, EvidenceBank, existing_jpeg_controller
from cuefd.methods.evidence_lifecycle_v2 import EvidenceCapsule

class Scaler:
    mean_=np.zeros(1102)
    scale_=np.ones(1102)

class Anchor:
    scaler=Scaler()
    def score_samples(self,x):return x[:,1]

class PCA:
    components_=np.eye(256,1102)

class Projection:
    pca=PCA()
    anchor=Anchor()
    def transform(self,x):return x[:,:256]

def test_final_lifecycle_does_not_commit_audit_when_core_update_fails():
    rng=np.random.default_rng(781)
    raw=rng.normal(size=(8,1102));labels=np.repeat([0,1],4)
    core=JointWithinCore(metric=np.eye(256),kappa=2.,lambda_prior=1.,basis=np.eye(256)[:,:4])
    window=JointWindow.from_samples("support",raw[:,:256],raw[:,1],labels,[f"support/{i}" for i in range(8)],2.)
    core.replace_current_window(window)
    direction=np.zeros(1102);direction[10]=1.
    bank=EvidenceBank({"jpeg":EvidenceCapsule(v=direction,mbar=np.zeros(1102),alpha=1.,bindings={})})
    model=AtomicJointLifecycle(core,Projection(),bank,existing_jpeg_controller(bank))
    before_core=model.core.to_bytes();before_controller=model.controller.to_bytes()
    def fail_after_mutation(staged):
        staged.replace_current_window(None)
        raise RuntimeError("deliberate failed core transition")
    with pytest.raises(RuntimeError,match="failed core"):
        model.audit_and_update(0,raw,labels,fail_after_mutation)
    assert model.core.to_bytes()==before_core
    assert model.controller.to_bytes()==before_controller
