#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "matplotlib", "numpy"]
# ///
# How to run: D:/anaconda/python.exe -B -u run.py; frozen inference only.
from __future__ import annotations

import json
from pathlib import Path

import torch

from common import ROOT, OUT, SOURCE, TEMPORAL, ILLUSION, DIAGNOSTIC, ORIGINAL, CellIdentity, sha
from temporal_replay import run_temporal
from illusion_replay import run_illusion


def main() -> None:
    assert not (OUT / "verification.json").exists()
    torch.set_num_threads(2)
    training = json.loads((SOURCE / "results.json").read_text())
    provenance = json.loads((SOURCE / "comparison.json").read_text())["source_sha256"]
    metadata: list[CellIdentity] = [{k: r[k] for k in ("cell_id", "retinal_class", "polarity")} for r in training["cells"]]
    assert len(metadata) == 22
    checkpoints = [SOURCE / "cells" / c["cell_id"].replace("#", "_") / "model-trained.pt" for c in metadata]
    core = tuple((ROOT / "models/mechanistic_retina").glob("*.py"))
    for p in (*core, *checkpoints):
        assert sha(p) == provenance[str(p)]
    temporal_verification = json.loads((TEMPORAL / "verification.json").read_text())
    illusion_verification = json.loads((ILLUSION / "verification.json").read_text())
    assert sha(TEMPORAL / "inputs.pt") == temporal_verification["inputs_sha256"]
    for p in (DIAGNOSTIC / "inputs.pt", ORIGINAL / "input-tensors.pt", ORIGINAL / "stimuli.pt",
              DIAGNOSTIC / "protocol.json", ORIGINAL / "stimulus-contract.json"):
        assert sha(p) == illusion_verification["source_sha256"][str(p)]
    for p in (TEMPORAL / "probe.py", TEMPORAL / "report.py", ROOT / "evaluation/mechanistic_retina/temporal_center_surround.py"):
        assert sha(p) == temporal_verification["source_sha256"][str(p)]
    sources = (*core, *checkpoints, *OUT.glob("*.py"), SOURCE / "results.json", SOURCE / "comparison.json",
               TEMPORAL / "inputs.pt", TEMPORAL / "responses.pt", TEMPORAL / "protocol.json",
               TEMPORAL / "results.json", TEMPORAL / "probe.py", TEMPORAL / "report.py",
               ROOT / "evaluation/mechanistic_retina/temporal_center_surround.py",
               ILLUSION / "response-tensors.pt", ILLUSION / "per-cell-responses.csv",
               DIAGNOSTIC / "inputs.pt", DIAGNOSTIC / "protocol.json", ORIGINAL / "stimuli.pt",
               ORIGINAL / "input-tensors.pt", ORIGINAL / "stimulus-contract.json", ORIGINAL / "metrics.py", ORIGINAL / "stimuli.py")
    hashes = {str(p): sha(p) for p in sources}
    (OUT / "input-manifest.json").write_text(json.dumps({"source_sha256": hashes,
        "checkpoint_count": 22, "checkpoint_and_core_match_training_manifest": True,
        "temporal_inputs_match_previous_sha256": True, "illusion_inputs_match_previous_sha256": True,
        "training": False, "new_probe_families": 0}, indent=2))
    checks = run_temporal(metadata) + run_illusion(metadata)
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    verification = {"checks": checks, "unique_cells": 22, "checkpoint_count": 22,
        "checkpoint_and_core_match_training_manifest": True, "all_source_hashes_unchanged": True,
        "original_stimulus_tensors_loaded": True, "new_probe_families": 0, "RMS_normalization": False,
        "training": False, "optimizer_created": False, "training_checkpoint_writes": 0,
        "source_sha256": hashes, "temporal_history": "same all-zero history from frozen protocol; no saved history tensor in old temporal inputs",
        "illusion_history": "loaded old original_history and diagnostic_history tensors exactly"}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False))
    print("COMPLETE: frozen replay and verification 22/22 for both applications", flush=True)


if __name__ == "__main__":
    main()
