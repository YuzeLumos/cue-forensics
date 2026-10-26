"""CPU wrappers for the unchanged final estimator and its frozen research states."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from .core import JointWithinCore, JointWindow
from .io import load_npz, new_output_dir, sha256, verified_asset, write_json

BENCHMARKS = ("UFD", "GenImage", "AIGC-DB", "AIGIBench", "WildFake", "Chameleon")

def read_core(path):
    return JointWithinCore.from_bytes(Path(path).read_bytes())

def save_core(path, core):
    Path(path).write_bytes(core.to_bytes())

def core_info(core):
    return dict(schema="cue-joint-within-core-v1", kappa=core.kappa,
                lambda_prior=core.lambda_prior, history_windows=list(core.history),
                current_window=None if core.current is None else core.current.window_id,
                resources=core.resource_counts(), diagnostics=dict(core.solve().diagnostics))

def input_window(args, core):
    data = load_npz(args.input, ["projected_features", "anchor_scores", "labels", "sample_ids"])
    if data["sample_ids"].dtype.kind not in "US":
        raise ValueError("sample_ids must be a Unicode/string array, without pickle")
    return JointWindow.from_samples(args.window_id, data["projected_features"],
        data["anchor_scores"], data["labels"], data["sample_ids"], core.kappa)

def restore_frozen(assets, arm):
    from .snapshot import restore_system
    base = verified_asset(assets, "dino/base.pkl")
    checkpoint = verified_asset(assets, f"dino/integrated_{arm}_K4.snapshot.json")
    return restore_system(checkpoint.read_bytes(), base), checkpoint

def command_demo(args):
    rng = np.random.default_rng(args.seed)
    config = dict(metric=np.eye(256), kappa=2., lambda_prior=1., basis=np.eye(256)[:, :8])
    core = JointWithinCore(**config)
    def samples(name):
        z = rng.normal(size=(8,256)); labels = np.repeat([0,1],4)
        anchor = rng.normal(size=8)
        ids = np.array([f"synthetic/{name}/{i}" for i in range(8)])
        return JointWindow.from_samples(name,z,anchor,labels,ids,core.kappa), dict(
            projected_features=z,anchor_scores=anchor,labels=labels,sample_ids=ids)
    historical,_ = samples("history0")
    current,support = samples("current0")
    core.add_history_window(historical)
    core.replace_current_window(current)
    query = dict(projected_features=rng.normal(size=(6,256)),anchor_scores=rng.normal(size=6))
    expected = core.score_projected(**dict(z=query["projected_features"],anchor=query["anchor_scores"]))
    restored = JointWithinCore.from_bytes(core.to_bytes())
    error = float(np.max(np.abs(expected-restored.score_projected(query["projected_features"],query["anchor_scores"]))))
    with new_output_dir(args.output_dir) as output:
        save_core(output/"state.json", core)
        np.savez_compressed(output/"config.npz", **config)
        np.savez_compressed(output/"support.npz", **support)
        np.savez_compressed(output/"query.npz", **query)
        np.savez_compressed(output/"scores.npz",scores=expected)
        write_json(output/"report.json",dict(purpose="synthetic CPU API smoke; no paper performance claim",seed=args.seed,restore_max_abs_error=error,**core_info(core)))

def command_init(args):
    data = load_npz(args.input,["metric","kappa","lambda_prior"])
    core = JointWithinCore(metric=data["metric"],kappa=float(data["kappa"]),
                          lambda_prior=float(data["lambda_prior"]),basis=data.get("basis"))
    with new_output_dir(args.output_dir) as output:
        save_core(output/"state.json",core)
        write_json(output/"report.json",core_info(core))

def command_score(args):
    core = read_core(args.state)
    query = load_npz(args.input,["projected_features","anchor_scores"])
    score = core.score_projected(query["projected_features"],query["anchor_scores"])
    with new_output_dir(args.output_dir) as output:
        np.savez_compressed(output/"scores.npz",scores=score)
        write_json(output/"report.json",dict(rows=len(score),input_sha256=sha256(args.input),state_sha256=sha256(args.state)))

def command_window(args):
    core = read_core(args.state)
    if args.action == "forget":
        staged = core.stage_update(lambda c:c.forget_core_window(args.window_id))
    else:
        window = input_window(args,core)
        if args.action == "add-history":
            staged = core.stage_update(lambda c:c.add_history_window(window))
        elif args.action == "replace-current":
            staged = core.stage_update(lambda c:c.replace_current_window(window))
        else:
            if args.archive_current is None:
                raise ValueError("advance requires --archive-current or --discard-current")
            core.advance_window(window,archive_current=args.archive_current)
            staged = core
    with new_output_dir(args.output_dir) as output:
        save_core(output/"state.json",staged)
        write_json(output/"report.json",dict(action=args.action,**core_info(staged)))

def command_inspect(args):
    print(json.dumps(core_info(read_core(args.state)),indent=2,sort_keys=True))

def command_frozen_demo(args):
    rng = np.random.default_rng(args.seed)
    # These are numerical probes in ENCODED DINO feature space, not images.
    raw = rng.normal(size=(4,1102))
    with new_output_dir(args.output_dir) as output:
        reports = {}
        heads_path = verified_asset(args.assets_dir,"dino/heads.npz")
        index_path = verified_asset(args.assets_dir,"dino/HEAD_INDEX.csv")
        heads = load_npz(heads_path,["delta_weight","delta_bias"])
        with index_path.open(encoding="utf-8") as handle:
            index = list(csv.DictReader(handle))
        for arm in ("H0","H32"):
            model,checkpoint = restore_frozen(args.assets_dir,arm)
            head = model.core.solve()
            rows = [r for r in index if r["model"]==f"integrated_{arm}" and r["world_index"]=="0" and r["seed"]=="20260918" and r["k"]=="4"]
            if len(rows)!=1: raise ValueError("Frozen head index is ambiguous")
            i=int(rows[0]["head_index"])
            error=float(np.max(np.abs(head.delta_weight-heads["delta_weight"][i])))
            bias_error=float(abs(head.delta_bias-heads["delta_bias"][i]))
            if error>1e-12 or bias_error>1e-12: raise ValueError("Snapshot differs from saved formal head")
            decision=model.predict(raw)
            np.savez_compressed(output/f"{arm}_scores.npz",forensic_score=decision.forensic_score,lower=decision.lower,upper=decision.upper,accusation_score=decision.accusation_score)
            save_core(output/f"{arm}_core.json",model.core)
            reports[arm]=dict(checkpoint_sha256=sha256(checkpoint),head_index=i,weight_max_abs_error=error,bias_abs_error=bias_error,**core_info(model.core))
        np.savez_compressed(output/"encoded_probe.npz",features=raw)
        write_json(output/"report.json",dict(purpose="real frozen checkpoint restore and saved-head consistency; synthetic encoded-feature probes; not a performance replication",arms=reports))

def command_dino_replay(args):
    raw=load_npz(args.input,["features"])["features"]
    if raw.ndim!=2 or raw.shape[1]!=1102 or len(raw)==0 or not np.isfinite(raw).all():
        raise ValueError("features must be finite nonempty DINO encoded rows [N,1102]")
    with new_output_dir(args.output_dir) as output:
        metadata={}
        for arm in ("H0","H32"):
            model,checkpoint=restore_frozen(args.assets_dir,arm)
            decision=model.predict(raw)
            np.savez_compressed(output/f"{arm}_scores.npz",forensic_score=decision.forensic_score,lower=decision.lower,upper=decision.upper,accusation_score=decision.accusation_score)
            metadata[arm]=dict(checkpoint_sha256=sha256(checkpoint),controller_states=dict(decision.states))
        write_json(output/"report.json",dict(scope="two saved world0/seed20260918/K4 heads, no refitting or metric claims",rows=len(raw),input_sha256=sha256(args.input),arms=metadata))

def command_cf_replay(args):
    # Only a manifest-verified shipped base is unpickled. Feature encoding is external.
    base_path=verified_asset(args.assets_dir,"cf/base.pkl")
    state_path=verified_asset(args.assets_dir,f"cf/states/{args.benchmark}_core.json")
    raw=load_npz(args.input,["features"])["features"]
    if raw.ndim!=2 or raw.shape[1]!=384 or len(raw)==0 or not np.isfinite(raw).all():
        raise ValueError("features must be finite nonempty official CF pre-logit rows [N,384]")
    with base_path.open("rb") as handle: base=pickle.load(handle)
    core=read_core(state_path)
    z=base.projection.transform(raw);anchor=base.projection.anchor.score_samples(raw)
    score=core.score_projected(z,anchor)
    with new_output_dir(args.output_dir) as output:
        np.savez_compressed(output/"scores.npz",scores=score,anchor_scores=anchor)
        write_json(output/"report.json",dict(scope="saved six-benchmark head inference; no fitting, support selection or full benchmark replication",benchmark=args.benchmark,rows=len(raw),input_sha256=sha256(args.input),base_sha256=sha256(base_path),core_sha256=sha256(state_path)))

def parser():
    p=argparse.ArgumentParser(prog="cue-joint",description="Final CUE joint-moment CPU core and frozen-state tools")
    p.add_argument("--threads",type=int,default=2,help="CPU numerical thread limit (default: 2)")
    sub=p.add_subparsers(dest="command",required=True)
    def output(command): command.add_argument("--output-dir",type=Path,required=True,help="A NEW output directory; existing runs are never overwritten")
    def input_(command): command.add_argument("--input",type=Path,required=True)
    def assets(command): command.add_argument("--assets-dir",type=Path,required=True,help="Repository assets directory, with SOURCE_MANIFEST.json in its parent")
    demo=sub.add_parser("demo",help="Synthetic CPU core/update/restore example")
    demo.add_argument("--seed",type=int,default=20261026);output(demo);demo.set_defaults(function=command_demo)
    init=sub.add_parser("init",help="Initialize empty core from explicit NPZ metric and frozen scales")
    input_(init);output(init);init.set_defaults(function=command_init)
    score=sub.add_parser("score",help="Score projected features and frozen anchor logits from a core state")
    input_(score);score.add_argument("--state",type=Path,required=True);output(score);score.set_defaults(function=command_score)
    window=sub.add_parser("window",help="Transactional add/replace/advance/forget on explicit state")
    window.add_argument("action",choices=["add-history","replace-current","advance","forget"])
    window.add_argument("--state",type=Path,required=True);window.add_argument("--input",type=Path)
    window.add_argument("--window-id",required=True)
    archive=window.add_mutually_exclusive_group();archive.add_argument("--archive-current",dest="archive_current",action="store_true");archive.add_argument("--discard-current",dest="archive_current",action="store_false")
    window.set_defaults(archive_current=None,function=command_window);output(window)
    inspect=sub.add_parser("inspect",help="Read core configuration, windows and resource counts")
    inspect.add_argument("--state",type=Path,required=True);inspect.set_defaults(function=command_inspect)
    frozen=sub.add_parser("frozen-demo",help="Restore real H0/H32 snapshots; compare saved heads on synthetic encoded probes")
    frozen.add_argument("--seed",type=int,default=20261026);assets(frozen);output(frozen);frozen.set_defaults(function=command_frozen_demo)
    dino=sub.add_parser("dino-replay",help="Score encoded DINO features using two saved world0/K4 H0/H32 snapshots")
    assets(dino);input_(dino);output(dino);dino.set_defaults(function=command_dino_replay)
    cf=sub.add_parser("cf-replay",help="Score official CF features using a saved six-benchmark head")
    assets(cf);input_(cf);output(cf);cf.add_argument("--benchmark",choices=BENCHMARKS,required=True);cf.set_defaults(function=command_cf_replay)
    return p

def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.threads<1: p.error("--threads must be positive")
    if args.command=="window" and args.action!="forget" and args.input is None:
        p.error("window updates require --input")
    try:
        with threadpool_limits(limits=args.threads): args.function(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"cue-joint: {exc}",file=sys.stderr)
        return 2
    return 0
