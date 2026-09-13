# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "opencv-python", "pydantic"]
# ///
# How to run: D:/anaconda/python.exe -B -u gate_replay.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Final

import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
PRIMARY: Final = ROOT / "output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905"
sys.path.insert(0, str(PRIMARY))
sys.path.insert(0, str(ROOT))

from preflight import ADAPTER, MOVIE, alignment, factory, folder, load_data
from data.schottdorf_lee_multirecording import load_schottdorf_movie_drive
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    torch.set_num_threads(2)
    assert not (OUT / "primary_replay.json").exists()
    manifest = json.loads((OUT / "evidence_manifest.initial.json").read_text())
    source = json.loads((PRIMARY / "results.json").read_text())
    rows = []
    raw = {}
    print("Decoding frozen original movie for mandatory replay", flush=True)
    movie = load_schottdorf_movie_drive(MOVIE, ADAPTER)
    for cell in source["cells"]:
        cid = cell["cell_id"]
        data = load_data(cid, movie)
        path = folder(cid, PRIMARY) / "model-trained.pt"
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == manifest["input_sha256"][path.relative_to(ROOT).as_posix()]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["model_name"] == "Canonical V1"
        assert checkpoint["schema"] == "schottdorf_canonical_v1_shared_bc_development"
        assert checkpoint["revision"] == 4 and checkpoint["stage"] == "trained"
        assert checkpoint["steps"] == cell["best_step"] == checkpoint["training_contract"]["fresh_full_train_refit_steps"]
        assert checkpoint["model_config"]["causal_contract"] == "h1-shared-bc-direct-broad-ac"
        assert checkpoint["model_config"]["spatial_contract"] == "bc-central-disk_ac-overlapping-full-disk"
        assert not checkpoint["training_contract"]["original_validation_used_for_selection"]
        assert torch.equal(checkpoint["cell_positions_degs"], alignment(cid))
        assert torch.equal(checkpoint["cone_positions_degs"], data.cone_positions_degs)
        model = factory(cid, data, checkpoint["cell_positions_degs"])
        model.load_state_dict(checkpoint["model"], strict=True)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 33
        metrics, logits = evaluate_retinal_model(model, data.validation)
        saved = torch.load(folder(cid, PRIMARY) / "validation-predictions.pt", map_location="cpu", weights_only=True)
        checks = {
            "target_exact": torch.equal(saved["target"], data.validation.spike_events),
            "mask_exact": torch.equal(saved["valid_mask"], data.validation.valid_mask),
            "source_order_exact": tuple(saved["source_image_ids"]) == data.validation.source_image_ids,
            "trial_order_exact": tuple(saved["trial_indices"]) == data.validation.trial_indices,
            "logits_bitwise_exact": torch.equal(saved["logits_trained"], logits),
            "nll_exact": metrics.population_nll == cell["validation_nll_trained"],
            "checkpoint_state_unchanged": all(torch.equal(v, checkpoint["model"][k]) for k, v in model.state_dict().items()),
            "parameter_grads_none": all(p.grad is None for p in model.parameters()),
        }
        passed = all(checks.values())
        rows.append({"cell_id": cid, "recording_ids": list(data.recording_ids), "checks": checks,
            "strict_load": True, "final_refit_steps": checkpoint["steps"], "trainable_scalars": 33,
            "nll": metrics.population_nll, "saved_nll": cell["validation_nll_trained"],
            "max_logit_error": float((saved["logits_trained"] - logits).abs().max()),
            "target_sha256": tensor_hash(data.validation.spike_events),
            "mask_sha256": tensor_hash(data.validation.valid_mask), "logits_sha256": tensor_hash(logits),
            "passed": passed})
        (OUT / "primary_replay.json").write_text(json.dumps({"all_passed": False, "training_runs": 0,
            "cells_completed": len(rows), "cells": rows}, indent=2), encoding="utf-8")
        assert passed, f"STOP: exact primary replay failed for {cid}"
        raw[cid] = {"cones": data.validation.cone_drive, "events": data.validation.spike_events,
            "mask": data.validation.valid_mask, "logits": logits,
            "cone_positions": data.cone_positions_degs, "cell_positions": checkpoint["cell_positions_degs"],
            "source_image_ids": data.validation.source_image_ids, "trial_indices": data.validation.trial_indices}
        print(f"PASS {len(rows)}/22 {cid}: NLL={metrics.population_nll:.12f}", flush=True)
    assert len(rows) == 22 and sum(len(r["recording_ids"]) for r in rows) == 37
    torch.save(raw, OUT / "validation-replay.pt")
    (OUT / "primary_replay.json").write_text(json.dumps({"all_passed": True, "training_runs": 0,
        "cells_completed": 22, "cells": rows}, indent=2), encoding="utf-8")
    print("PRIMARY EXACT REPLAY GATE PASSED", flush=True)


if __name__ == "__main__":
    main()
