"""Complete restore bound to an existing trusted base asset; no base copy."""
import base64,hashlib,json,pickle
from pathlib import Path
from .core import JointWithinCore
from .lifecycle import AtomicJointLifecycle,EvidenceBank,EvidenceBankController


def _sha(data): return hashlib.sha256(data).hexdigest()
def _canonical(record): return json.dumps(record,sort_keys=True,separators=(',',':')).encode()
def _projection_id(projection): return _sha(pickle.dumps(projection,protocol=5))


def _bound_base(core,bank,base_path,projection=None):
    raw=Path(base_path).read_bytes();identity=_sha(raw)
    if core.bindings.get('base_sha256')!=identity or any(c.bindings.get('base_sha256')!=identity for c in bank.capsules.values()):
        raise ValueError('Core/capsule/base asset binding mismatch')
    # Only the existing trusted local base is unpickled, after identity validation.
    base=pickle.loads(raw)
    if float(base.kappa)!=core.kappa:
        raise ValueError('Core kappa differs from the bound base')
    projection_id=_projection_id(base.projection)
    if projection is not None and _projection_id(projection)!=projection_id:
        raise ValueError('Live projection/scaler/anchor differs from the bound base')
    return base,identity,projection_id


def snapshot_system(lifecycle,base_path):
    state=lifecycle._state
    _,base_id,projection_id=_bound_base(state.core,lifecycle.bank,base_path,lifecycle.projection)
    payloads={'core':state.core.to_bytes(),'bank':lifecycle.bank.to_bytes(),'controller':state.controller.to_bytes()}
    record=dict(schema='cue-joint-within-system-v1',base_sha256=base_id,projection_scaler_anchor_sha256=projection_id,parts={k:dict(sha256=_sha(v),payload=base64.b64encode(v).decode()) for k,v in payloads.items()})
    record['sha256']=_sha(_canonical(record))
    return _canonical(record)


def restore_system(payload,base_path):
    record=json.loads(payload);checksum=record.pop('sha256')
    if record['schema']!='cue-joint-within-system-v1' or _sha(_canonical(record))!=checksum:
        raise ValueError('System snapshot schema/checksum mismatch')
    parts={}
    for name,r in record['parts'].items():
        value=base64.b64decode(r['payload'],validate=True)
        if _sha(value)!=r['sha256']: raise ValueError('Snapshot component checksum mismatch')
        parts[name]=value
    core=JointWithinCore.from_bytes(parts['core']);bank=EvidenceBank.from_bytes(parts['bank'])
    base,base_id,projection_id=_bound_base(core,bank,base_path)
    if base_id!=record['base_sha256'] or projection_id!=record['projection_scaler_anchor_sha256']:
        raise ValueError('Snapshot base/projection/scaler/anchor binding mismatch')
    controller=EvidenceBankController.from_bytes(parts['controller'],bank)
    return AtomicJointLifecycle(core,base.projection,bank,controller)
