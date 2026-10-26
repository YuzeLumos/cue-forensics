"""Run after installing the candidate, as documented in the README."""
import os
from pathlib import Path
import subprocess
import sys

def test_installed_entrypoint_runs_outside_checkout_without_pythonpath(tmp_path):
    env=dict(os.environ)
    env.pop("PYTHONPATH",None)
    result=subprocess.run([sys.executable,"-I","-B","-m","cue_joint","--help"],
        cwd=tmp_path,env=env,text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert "Final CUE joint-moment" in result.stdout
    output=tmp_path/"outside-checkout-run"
    result=subprocess.run([sys.executable,"-I","-B","-m","cue_joint","demo","--output-dir",str(output)],
        cwd=tmp_path,env=env,text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert (output/"state.json").is_file()
