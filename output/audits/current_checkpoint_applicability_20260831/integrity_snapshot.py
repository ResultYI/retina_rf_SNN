#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: D:/anaconda/python.exe -B integrity_snapshot.py from the repository root.
# This records hashes to stdout only; it does not write files.
from __future__ import annotations
from pathlib import Path
import hashlib,json,datetime,subprocess
root=Path.cwd()
lineage=root/"output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
sources=sorted(p for folder in ("models","training","evaluation","data") for p in (root/folder).rglob("*.py"))
checkpoints=sorted(lineage.glob("cells/*/model-trained.pt"))
artifacts=sorted(p for p in lineage.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
def hashes(paths):
    result={}
    for p in paths:
        with p.open("rb") as f: result[p.relative_to(root).as_posix()]=hashlib.file_digest(f,"sha256").hexdigest()
    return result
print(json.dumps({"utc":datetime.datetime.now(datetime.UTC).isoformat(),"source_sha256":hashes(sources),"checkpoint_sha256":hashes(checkpoints),"existing_lineage_artifact_sha256":hashes(artifacts),"git_status":subprocess.check_output(["git","status","--short","--untracked-files=no"],text=True)}))
