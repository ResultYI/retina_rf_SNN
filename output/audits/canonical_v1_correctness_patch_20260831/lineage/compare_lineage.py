#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy"]
# ///
# How to run: use the frozen D:/anaconda/python.exe -B runtime from repository root.
# compare_lineage.py reference|candidate START STOP; ranges index sorted checkpoint paths.
# Reference outputs are immutable evidence; candidate mode stops on any changed tensor byte.
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Final

import numpy as np
import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina

CHECKPOINTS: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
INPUTS: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    phase, start, stop = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    assert phase in {"reference", "candidate"}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    paths = sorted(CHECKPOINTS.glob("cells/*/model-trained.pt"))
    assert len(paths) == 22
    source_paths = sorted((ROOT / "models").rglob("*.py"))
    source_before = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    frozen_paths = paths + [INPUTS / name / "inputs.pt" for name in ("temporal", "illusion")]
    frozen_before = {str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}
    manifest = OUT / "reference_manifest.json"
    if not manifest.exists():
        assert phase == "reference" and start == 0
        manifest.write_text(json.dumps({"production_sources": source_before, "frozen_files": frozen_before,
            "input_policy": "Saved illusion history; zero temporal history matching frozen production protocol",
            "normal_only": True, "torch_version": torch.__version__, "threads": torch.get_num_threads()}, indent=2), encoding="utf-8")
    initial = json.loads(manifest.read_text(encoding="utf-8"))
    assert frozen_before == initial["frozen_files"]
    if phase == "reference":
        assert source_before == initial["production_sources"], "STOP: production changed before reference completed"
    temporal = torch.load(INPUTS / "temporal/inputs.pt", map_location="cpu", weights_only=True)
    illusion = torch.load(INPUTS / "illusion/inputs.pt", map_location="cpu", weights_only=True)
    destination = OUT / phase
    destination.mkdir(exist_ok=True)
    completed = []
    for index in range(start, stop):
        began = time.perf_counter()
        path = paths[index]
        cp = torch.load(path, map_location="cpu", weights_only=True)
        assert cp["stage"] == "trained" and len(cp["cell_types"]) == 1
        assert cp["model_config"]["architecture_mode"] == "mechanism_identifiable"
        model = build_mechanistic_retina(MechanisticRetinaConfig(**cp["model_config"]), cp["cone_positions_degs"],
            cp["cell_positions_degs"], tuple(cp["cell_types"]), tuple(cp["polarities"]))
        loaded = model.load_state_dict(cp["model"], strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        model.eval()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        assert set(before) == set(cp["model"])
        assert all(tensor_sha(value) == tensor_sha(cp["model"][key]) for key, value in before.items())
        parameter_names = set(dict(model.named_parameters()))
        persistent_buffers = set(before) - parameter_names
        state_identity = {key: {"shape": list(value.shape), "dtype": str(value.dtype), "raw_sha256": tensor_sha(value),
            "role": "parameter" if key in parameter_names else "buffer"} for key, value in before.items()}
        cell_name = cp["cell_id"].replace("#", "_")
        record = {"index": index, "cell_id": cp["cell_id"], "checkpoint": str(path.relative_to(ROOT)),
            "checkpoint_sha256": sha(path), "strict_load": True, "raw_config": json.loads(json.dumps(cp["model_config"])),
            "state_identity": state_identity, "parameter_names": sorted(parameter_names),
            "persistent_buffer_names": sorted(persistent_buffers), "all_buffer_names": sorted(dict(model.named_buffers())),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "applications": {}}
        reference_record = None
        if phase == "candidate":
            reference_record = json.loads((OUT / "reference" / f"{cell_name}.json").read_text(encoding="utf-8"))
            for name in ("checkpoint_sha256", "raw_config", "state_identity", "parameter_names", "persistent_buffer_names", "all_buffer_names", "trainable_parameter_count"):
                assert record[name] == reference_record[name], (cp["cell_id"], "identity changed", name)
            expected_flags = json.loads((OUT / "reference_trainability.json").read_text(encoding="utf-8"))["cells"][cp["cell_id"]]
            record["parameter_requires_grad"] = {name: p.requires_grad for name, p in model.named_parameters()}
            assert record["parameter_requires_grad"] == expected_flags, (cp["cell_id"], "parameter trainability changed")
        group = ("MC" if cp["cell_types"][0] == "midget" else "PC") + "_" + cp["polarities"][0]
        temporal_drive = temporal[group]["drive"]
        banks = {"temporal": (temporal_drive, torch.zeros(temporal_drive.shape[:2] + (1,), dtype=temporal_drive.dtype)),
            "illusion": (illusion["cone_drive"], illusion["history"])}
        for application, (drive, history) in banks.items():
            captured: list[torch.Tensor] = []

            def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                assert len(inputs) == 1
                captured.append(inputs[0].detach().clone())

            hook = model.amacrine.register_forward_pre_hook(capture)
            try:
                with torch.no_grad():
                    result = model.forward_sequence(drive, observed_counts=history)
            finally:
                hook.remove()
            assert len(captured) == 1 and torch.equal(captured[0], result.bc_broad_presynaptic)
            tensors = {field.name: getattr(result, field.name).detach().contiguous().numpy() for field in fields(result)}
            tensors["ac_input_hook"] = captured[0].contiguous().numpy()
            assert all(bool(np.isfinite(value).all()) for value in tensors.values())
            tensor_meta = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                "raw_sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in tensors.items()}
            archive_name = f"{cell_name}__{application}.npz"
            if phase == "reference":
                assert not (destination / archive_name).exists(), "Reference overwrite prohibited"
                np.savez_compressed(destination / archive_name, **tensors)
                record["applications"][application] = {"archive_sha256": sha(destination / archive_name), "tensors": tensor_meta,
                    "drive_raw_sha256": tensor_sha(drive), "history_raw_sha256": tensor_sha(history), "ac_input_equals_bc_broad": True}
            else:
                comparisons = {}
                with np.load(OUT / "reference" / archive_name, allow_pickle=False) as baseline:
                    assert set(baseline.files) == set(tensors)
                    assert sha(OUT / "reference" / archive_name) == reference_record["applications"][application]["archive_sha256"]
                    for name, value in tensors.items():
                        old = baseline[name]
                        identical = old.shape == value.shape and old.dtype == value.dtype and old.tobytes() == value.tobytes()
                        maximum = float(np.max(np.abs(old.astype(np.float64) - value.astype(np.float64))))
                        comparisons[name] = {"bitwise_identical": identical, "max_abs_error": maximum,
                            "reference_raw_sha256": hashlib.sha256(old.tobytes()).hexdigest(), "candidate_raw_sha256": tensor_meta[name]["raw_sha256"]}
                        if not identical:
                            failure = {"status": "FAIL_STOP", "cell_id": cp["cell_id"], "application": application,
                                "tensor": name, **comparisons[name]}
                            (destination / "FAIL_STOP.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
                            raise SystemExit(json.dumps(failure))
                assert tensor_sha(drive) == reference_record["applications"][application]["drive_raw_sha256"]
                assert tensor_sha(history) == reference_record["applications"][application]["history_raw_sha256"]
                record["applications"][application] = {"tensors": tensor_meta, "comparisons": comparisons,
                    "ac_input_equals_bc_broad": True, "same_inputs": True}
        assert all(tensor_sha(value) == tensor_sha(before[key]) for key, value in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        record.update({"state_unchanged_by_forward": True, "no_gradients": True, "no_training": True,
            "elapsed_seconds": time.perf_counter() - began})
        (destination / f"{cell_name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        completed.append(cp["cell_id"])
        print(json.dumps({"phase": phase, "index": index, "cell_id": cp["cell_id"], "applications": 2,
            "tensor_count": sum(len(a["tensors"]) for a in record["applications"].values()),
            "seconds": round(record["elapsed_seconds"], 2), "pass": True}), flush=True)
    source_after = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    frozen_after = {str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}
    assert source_before == source_after, "STOP: production changed during capture"
    assert frozen_before == frozen_after
    (destination / f"integrity_{start}_{stop}.json").write_text(json.dumps({"production_sources_before": source_before,
        "production_sources_after": source_after, "frozen_files_before": frozen_before, "frozen_files_after": frozen_after,
        "completed": completed, "pass": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
