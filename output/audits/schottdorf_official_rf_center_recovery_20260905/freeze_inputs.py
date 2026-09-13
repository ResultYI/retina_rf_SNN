# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic"]
# ///
# How to run: D:/anaconda/python.exe -B freeze_inputs.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from data.schottdorf_lee_catalog import mc_pc_recordings

REFERENCE = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
LN = ROOT / "output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830"
OFFICIAL = ROOT / "data/real/schottdorf_lee_2021_repository"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    assert not (OUT / "evidence_manifest.json").exists()
    cells = json.loads((REFERENCE / "results.json").read_text())["cells"]
    catalog = mc_pc_recordings(OFFICIAL / "data")
    rows = []
    paths = set()
    for cell in cells:
        cid = cell["cell_id"]
        recordings = [r for r in catalog if r.cell_id == cid]
        assert [r.recording_id for r in recordings] == cell["recording_ids"]
        checkpoint_path = LN / "cells" / cid.replace("#", "_") / "ln-trained.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        assert checkpoint["cell_id"] == cid and checkpoint["best_step"] == checkpoint["refit_steps"]
        rows.append({"cell_id": cid, "recording_ids": cell["recording_ids"], "retinal_class": cell["retinal_class"],
                     "polarity": cell["polarity"], "LN_checkpoint": str(checkpoint_path.relative_to(ROOT)),
                     "LN_checkpoint_sha256": digest(checkpoint_path), "LN_center_pixels": checkpoint["model"]["center_xy"].tolist()})
        paths.add(checkpoint_path)
        paths.add(checkpoint_path.with_name("results.json"))
    assert len(rows) == 22 and sum(len(r["recording_ids"]) for r in rows) == 37
    roots = [REFERENCE, OFFICIAL, ROOT / "output/audits/macaque_spatial_alignment_20260905",
             ROOT / "output/audits/macaque_fixed_alignment_experiment_20260905",
             ROOT / "output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905"]
    for root in roots:
        paths.update(p for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts)
    paths.update(p for name in ("models", "training", "evaluation", "data", "baselines") for p in (ROOT / name).rglob("*.py"))
    paths.add(ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg")
    manifest = {"task": "official RF-center recovery", "new_training_runs": 0,
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status_before": subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True),
                "official_snapshot_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OFFICIAL, text=True).strip(),
                "cells": rows, "biological_cell_count": 22, "recording_count": 37,
                "frozen_input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}}
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen 22 biological cells,37 recordings,{len(paths)} input hashes; training=0")


if __name__ == "__main__":
    main()
