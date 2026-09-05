#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
# How to run: D:/anaconda/python.exe -B summarize_lineage.py after all candidate ranges finish.
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    initial = json.loads((OUT / "reference_manifest.json").read_text(encoding="utf-8"))
    paths = sorted(p for p in (OUT / "reference").glob("*.json") if not p.name.startswith("integrity_"))
    assert len(paths) == 22
    trainability = json.loads((OUT / "reference_trainability.json").read_text(encoding="utf-8"))
    assert trainability["status"] == "PASS" and trainability["forward_calls"] == 0
    for name, digest in trainability["prepatch_source_hashes"].items():
        assert digest == initial["production_sources"][name]
        assert sha(OUT.parent / "source_before" / name) == digest
    assert all(Path(path).is_relative_to(OUT.parent / "source_before") for path in trainability["imported_modules"].values())
    cells, tensor_count, element_count = [], 0, 0
    archive_bytes = 0
    all_fields = set()
    for path in paths:
        before = json.loads(path.read_text(encoding="utf-8"))
        after = json.loads((OUT / "candidate" / path.name).read_text(encoding="utf-8"))
        for key in ("cell_id", "checkpoint_sha256", "raw_config", "state_identity", "parameter_names", "persistent_buffer_names", "all_buffer_names", "trainable_parameter_count"):
            assert before[key] == after[key], (before["cell_id"], key)
        assert before["strict_load"] and after["strict_load"]
        assert after["parameter_requires_grad"] == trainability["cells"][before["cell_id"]]
        assert before["state_unchanged_by_forward"] and after["state_unchanged_by_forward"]
        assert before["no_gradients"] and after["no_gradients"]
        assert set(before["applications"]) == set(after["applications"]) == {"temporal", "illusion"}
        for application, bank in before["applications"].items():
            archive = OUT / "reference" / f"{path.stem}__{application}.npz"
            assert sha(archive) == bank["archive_sha256"]
            archive_bytes += archive.stat().st_size
            candidate = after["applications"][application]
            assert bank["ac_input_equals_bc_broad"] and candidate["ac_input_equals_bc_broad"] and candidate["same_inputs"]
            assert set(bank["tensors"]) == set(candidate["tensors"]) == set(candidate["comparisons"])
            with np.load(archive, allow_pickle=False) as saved:
                assert set(saved.files) == set(bank["tensors"])
                for name, tensor_meta in bank["tensors"].items():
                    array = saved[name]
                    assert hashlib.sha256(array.tobytes()).hexdigest() == tensor_meta["raw_sha256"]
                    comparison = candidate["comparisons"][name]
                    assert candidate["tensors"][name] == tensor_meta
                    assert comparison["reference_raw_sha256"] == comparison["candidate_raw_sha256"] == tensor_meta["raw_sha256"]
                    assert comparison["bitwise_identical"] and comparison["max_abs_error"] == 0.0
                    tensor_count += 1
                    element_count += int(array.size)
                    all_fields.add(name)
        cells.append({"cell_id": before["cell_id"], "strict_load": True, "state_identity_unchanged": True,
            "trainable_parameter_count": before["trainable_parameter_count"], "all_tensors_bitwise_identical": True,
            "all_buffer_names_unchanged": before["all_buffer_names"] == after["all_buffer_names"],
            "reference_seconds": before["elapsed_seconds"], "candidate_seconds": after["elapsed_seconds"]})
    assert tensor_count == 22 * 2 * 27
    assert not (OUT / "candidate/FAIL_STOP.json").exists()
    reference_integrity = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "reference").glob("integrity_*.json"))]
    assert set(cell for record in reference_integrity for cell in record["completed"]) == {c["cell_id"] for c in cells}
    for record in reference_integrity:
        assert record["pass"]
        assert record["production_sources_before"] == record["production_sources_after"] == initial["production_sources"]
        assert record["frozen_files_before"] == record["frozen_files_after"] == initial["frozen_files"]
    integrity_files = sorted((OUT / "candidate").glob("integrity_*.json"))
    integrity = [json.loads(p.read_text(encoding="utf-8")) for p in integrity_files]
    assert set(cell for record in integrity for cell in record["completed"]) == {c["cell_id"] for c in cells}
    candidate_source = integrity[0]["production_sources_before"]
    for record in integrity:
        assert record["pass"]
        assert record["production_sources_before"] == record["production_sources_after"] == candidate_source
        assert record["frozen_files_before"] == record["frozen_files_after"] == initial["frozen_files"]
    current_sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "models").rglob("*.py"))}
    assert current_sources == candidate_source, "Production changed after candidate capture"
    assert all(sha(ROOT / name) == digest for name, digest in initial["frozen_files"].items())
    changed_sources = sorted(name for name in set(initial["production_sources"]) | set(candidate_source)
        if initial["production_sources"].get(name) != candidate_source.get(name))
    summary = {"status": "PASS", "checkpoints": 22, "strict_load_pass": 22, "state_identity_unchanged": 22, "per_parameter_trainability_unchanged": 22,
        "applications": ["temporal", "illusion"], "mode": "normal", "reference_archives": 44,
        "tensor_comparisons": tensor_count, "scalar_elements_compared": element_count, "max_abs_error": 0.0,
        "all_bitwise_identical": True, "output_fields": sorted(all_fields), "reference_archive_bytes": archive_bytes,
        "frozen_checkpoint_and_input_files_unchanged": len(initial["frozen_files"]),
        "changed_production_sources": changed_sources, "all_buffer_names_unchanged": all(c["all_buffer_names_unchanged"] for c in cells),
        "no_training": True, "no_checkpoint_conversion": True, "reference_seconds": sum(c["reference_seconds"] for c in cells),
        "candidate_seconds": sum(c["candidate_seconds"] for c in cells), "cells": cells}
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
