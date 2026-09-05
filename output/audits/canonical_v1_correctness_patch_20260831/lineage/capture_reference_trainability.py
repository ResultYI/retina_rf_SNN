#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0"]
# ///
# How to run: D:/anaconda/python.exe -B capture_reference_trainability.py; no forward calls.
# Supplement the completed reference from hash-verified prepatch source, without replacing it.
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Final

import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[3]
SNAPSHOT: Final = OUT.parent / "source_before"


def main() -> None:
    torch.set_num_threads(2)
    initial = json.loads((OUT / "reference_manifest.json").read_text(encoding="utf-8"))
    source_hashes = {}
    for path in sorted((SNAPSHOT / "models").rglob("*.py")):
        relative = str(path.relative_to(SNAPSHOT))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == initial["production_sources"][relative], relative
        source_hashes[relative] = digest
    sys.path.insert(0, str(ROOT))
    import models
    models.__path__ = [str(SNAPSHOT / "models")]
    from models.mechanistic_retina.contracts import MechanisticRetinaConfig
    from models.mechanistic_retina.model import build_mechanistic_retina
    paths = sorted(p for p in (OUT / "reference").glob("*.json") if not p.name.startswith("integrity_"))
    assert len(paths) == 22
    cells = {}
    for path in paths:
        reference = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_path = ROOT / reference["checkpoint"]
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == reference["checkpoint_sha256"]
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_mechanistic_retina(MechanisticRetinaConfig(**cp["model_config"]), cp["cone_positions_degs"],
            cp["cell_positions_degs"], tuple(cp["cell_types"]), tuple(cp["polarities"]))
        loaded = model.load_state_dict(cp["model"], strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        assert sorted(flags) == reference["parameter_names"]
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == reference["trainable_parameter_count"]
        for name, value in model.state_dict().items():
            assert hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest() == reference["state_identity"][name]["raw_sha256"]
        cells[reference["cell_id"]] = flags
    imported = {name: module.__file__ for name, module in sys.modules.items() if name.startswith("models.mechanistic_retina")}
    assert all(Path(path).is_relative_to(SNAPSHOT) for path in imported.values())
    destination = OUT / "reference_trainability.json"
    assert not destination.exists(), "Supplement overwrite prohibited"
    destination.write_text(json.dumps({"status": "PASS", "forward_calls": 0, "checkpoint_conversions": 0,
        "prepatch_source_hashes": source_hashes, "imported_modules": imported, "cells": cells}, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoints": len(cells), "forward_calls": 0, "prepatch_sources_hash_verified": len(source_hashes),
        "parameter_tensors_per_cell": len(next(iter(cells.values()))), "status": "PASS"}))


if __name__ == "__main__":
    main()
