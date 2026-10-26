from pathlib import Path
import json
import shutil
import numpy as np
import pytest

from cue_joint.cli import main
from cue_joint.core import JointWithinCore, JointWindow
from cue_joint.io import sha256, verified_asset

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"

@pytest.fixture
def windows():
    rng=np.random.default_rng(781)
    def make(name):
        return JointWindow.from_samples(name,rng.normal(size=(8,256)),rng.normal(size=8),
            [0,0,0,0,1,1,1,1],[f"{name}/{i}" for i in range(8)],2.)
    return [make(name) for name in ("current","history1","history2")]

def core():
    return JointWithinCore(metric=np.eye(256),kappa=2.,lambda_prior=1.,basis=np.eye(256)[:,:6])

def test_final_core_empty_restore_and_delete_rebuild(windows):
    c,h1,h2=windows; rng=np.random.default_rng(93)
    query=rng.normal(size=(11,256));anchor=rng.normal(size=11)
    model=core();assert np.array_equal(model.score_projected(query,anchor),anchor)
    model.replace_current_window(c);model.add_history_window(h1);model.add_history_window(h2)
    restored=JointWithinCore.from_bytes(model.to_bytes())
    assert restored.to_bytes()==model.to_bytes()
    assert np.array_equal(restored.score_projected(query,anchor),model.score_projected(query,anchor))
    restored.forget_core_window("history1")
    fresh=core();fresh.replace_current_window(c);fresh.add_history_window(h2)
    assert np.array_equal(fresh.score_projected(query,anchor),restored.score_projected(query,anchor))

def test_duplicate_window_and_failed_update_are_atomic(windows):
    c,h1,_=windows;model=core();model.replace_current_window(c);model.add_history_window(h1)
    before=model.to_bytes()
    with pytest.raises(ValueError):model.stage_update(lambda staged:staged.add_history_window(h1))
    assert model.to_bytes()==before
    def broken(staged):
        staged.forget_core_window("history1")
        raise RuntimeError("intentional rollback probe")
    with pytest.raises(RuntimeError):model.stage_update(broken)
    assert model.to_bytes()==before

def test_cli_demo_score_and_output_collision_do_not_change_existing_files(tmp_path):
    demo=tmp_path/"demo"
    assert main(["demo","--output-dir",str(demo)])==0
    previous={p.name:sha256(p) for p in demo.iterdir()}
    assert main(["demo","--output-dir",str(demo)])==2
    assert {p.name:sha256(p) for p in demo.iterdir()}==previous
    score=tmp_path/"score"
    assert main(["score","--state",str(demo/"state.json"),"--input",str(demo/"query.npz"),"--output-dir",str(score)])==0
    with np.load(score/"scores.npz") as actual,np.load(demo/"scores.npz") as expected:
        assert np.array_equal(actual["scores"],expected["scores"])

def test_cli_window_init_replace_advance_and_forget(tmp_path):
    demo=tmp_path/"demo";assert main(["demo","--output-dir",str(demo)])==0
    empty=tmp_path/"empty"
    assert main(["init","--input",str(demo/"config.npz"),"--output-dir",str(empty)])==0
    assert main(["window","replace-current","--state",str(empty/"state.json"),"--input",str(demo/"support.npz"),"--window-id","current0","--output-dir",str(tmp_path/"updated")])==0
    next_support=tmp_path/"next.npz"
    with np.load(demo/"support.npz") as data:
        arrays={k:data[k].copy() for k in data.files}
    arrays["sample_ids"]=np.array([f"new/{i}" for i in range(8)])
    np.savez_compressed(next_support,**arrays)
    assert main(["window","advance","--archive-current","--state",str(tmp_path/"updated/state.json"),"--input",str(next_support),"--window-id","current1","--output-dir",str(tmp_path/"advanced")])==0
    state=JointWithinCore.from_bytes((tmp_path/"advanced/state.json").read_bytes())
    assert list(state.history)==["current0"] and state.current.window_id=="current1"
    assert main(["window","forget","--state",str(tmp_path/"advanced/state.json"),"--window-id","current0","--output-dir",str(tmp_path/"forgotten")])==0
    assert not JointWithinCore.from_bytes((tmp_path/"forgotten/state.json").read_bytes()).history

def test_bad_input_never_publishes_partial_output(tmp_path):
    data=tmp_path/"bad.npz";np.savez(data,metric=np.eye(2),kappa=2.,lambda_prior=1.)
    assert main(["init","--input",str(data),"--output-dir",str(tmp_path/"bad-output")])==2
    assert not (tmp_path/"bad-output").exists()
    assert not list(tmp_path.glob(".cue-joint-*"))

def test_frozen_h0_h32_snapshots_match_original_saved_heads_after_relocation(tmp_path):
    moved=tmp_path/"research copy with spaces"
    (moved/"assets").mkdir(parents=True)
    shutil.copy2(REPO/"SOURCE_MANIFEST.json",moved/"SOURCE_MANIFEST.json")
    shutil.copytree(ASSETS/"dino",moved/"assets/dino")
    out=tmp_path/"frozen"
    assert main(["frozen-demo","--assets-dir",str(moved/"assets"),"--output-dir",str(out)])==0
    report=json.loads((out/"report.json").read_text())
    for arm in ("H0","H32"):
        # Conda MKL and a fresh PyPI/OpenBLAS runtime may round differently.
        assert report["arms"][arm]["weight_max_abs_error"]<1e-12
        assert report["arms"][arm]["bias_abs_error"]<1e-12
    assert main(["dino-replay","--assets-dir",str(moved/"assets"),"--input",str(out/"encoded_probe.npz"),"--output-dir",str(tmp_path/"dino-scored")])==0
    for arm in ("H0","H32"):
        with np.load(out/f"{arm}_scores.npz") as a,np.load(tmp_path/f"dino-scored/{arm}_scores.npz") as b:
            assert np.array_equal(a["forensic_score"],b["forensic_score"])

def test_asset_tamper_rejected_before_pickle_loading(tmp_path):
    moved=tmp_path/"tamper";(moved/"assets/dino").mkdir(parents=True)
    shutil.copy2(REPO/"SOURCE_MANIFEST.json",moved/"SOURCE_MANIFEST.json")
    (moved/"assets/dino/base.pkl").write_bytes(b"not a pickle")
    with pytest.raises(ValueError,match="checksum"):
        verified_asset(moved/"assets","dino/base.pkl")

@pytest.mark.parametrize("benchmark",["UFD","GenImage","AIGC-DB","AIGIBench","WildFake","Chameleon"])
def test_six_formal_cf_states_have_working_feature_scoring_entry(tmp_path,benchmark):
    encoded=tmp_path/"cf_features.npz"
    np.savez_compressed(encoded,features=np.random.default_rng(11).normal(size=(3,384)))
    out=tmp_path/benchmark
    assert main(["cf-replay","--assets-dir",str(ASSETS),"--benchmark",benchmark,"--input",str(encoded),"--output-dir",str(out)])==0
    with np.load(out/"scores.npz") as data:
        assert data["scores"].shape==(3,) and np.isfinite(data["scores"]).all()

def test_all_copied_source_and_model_bindings_match():
    entries=json.loads((REPO/"SOURCE_MANIFEST.json").read_text(encoding="utf-8"))["entries"]
    for entry in entries:
        assert sha256(REPO/entry["path"])==entry["sha256"]
