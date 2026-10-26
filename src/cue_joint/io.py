"""Portable explicit input/output boundaries; no author-specific storage paths."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def load_npz(path, required):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError("Missing NPZ arrays: " + ", ".join(sorted(missing)))
        return {name: archive[name].copy() for name in archive.files}

@contextmanager
def new_output_dir(path):
    """Stage outputs and publish one fresh directory; existing runs are immutable."""
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"Output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".cue-joint-", dir=target.parent))
    try:
        yield staged
        if target.exists():
            raise FileExistsError(f"Output directory already exists: {target}")
        os.rename(staged, target)
    finally:
        if staged.exists():
            if staged.resolve().parent != target.parent or not staged.name.startswith(".cue-joint-"):
                raise RuntimeError("Refusing to remove an unexpected staging directory")
            shutil.rmtree(staged)

def verified_asset(assets_dir, relative):
    """Validate a checked-in artifact against its portable source manifest."""
    assets_dir = Path(assets_dir).resolve()
    candidate = (assets_dir / relative).resolve()
    if not candidate.is_relative_to(assets_dir):
        raise ValueError("Asset path escapes assets directory")
    manifest = json.loads((assets_dir.parent / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    key = "assets/" + Path(relative).as_posix()
    matches = [x for x in manifest["entries"] if x["path"] == key]
    if len(matches) != 1 or sha256(candidate) != matches[0]["sha256"]:
        raise ValueError(f"Source-manifest checksum mismatch: {key}")
    return candidate
