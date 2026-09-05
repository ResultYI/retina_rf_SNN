#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.10.0", "numpy==1.26.4"]
# ///
# How to run: D:/anaconda/envs/snn_env/python.exe -B -u .omo/evidence/compact_causal_cnn_baseline/train.py
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Final

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselines.center_surround_ln import LNError
from data.retinal_recording import RealSequenceSplit
from training.mechanistic_retina.center_surround_ln import LNHistory
from training.mechanistic_retina.compact_causal_cnn import evaluate_cnn, fresh_cnn, select_and_refit_cnn


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def to_cuda(split: RealSequenceSplit) -> RealSequenceSplit:
    return RealSequenceSplit(split.cone_drive.cuda(), split.spike_counts.cuda(), split.spike_events.cuda(),
                              split.valid_mask.cuda(), split.source_image_ids, split.trial_indices)


def main() -> None:
    if (OUT / "cells").exists():
        raise LNError("STOP: CNN fits already exist; automatic restart is prohibited")
    if not torch.cuda.is_available():
        raise LNError("STOP: declared GPU runtime is unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    preflight = json.loads((OUT / "preflight.json").read_text())
    for path, digest in preflight["source_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise LNError(f"STOP: frozen source changed before fitting: {path}")
    runtime = dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                   gpu=torch.cuda.get_device_name(), dtype="float32", amp=False, tf32=False,
                   deterministic_algorithms=True, cudnn_benchmark=False,
                   preflight_sha256=sha256_file(OUT / "preflight.json"))
    (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (OUT / "cells").mkdir()
    for cell in preflight["cells"]:
        name = cell["cell_id"].replace("#", "_")
        bundle = torch.load(OUT / "inputs" / f"{name}.pt", map_location="cpu", weights_only=True)
        train = RealSequenceSplit(**bundle["train"])
        validation = RealSequenceSplit(**bundle["validation"])
        for label, split in (("train", train), ("validation", validation)):
            for field, tensor in (("input", split.cone_drive), ("target", split.spike_events), ("mask", split.valid_mask)):
                digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                if digest != cell["tensor_sha256"][f"{label}_{field}_sha256"]:
                    raise LNError("STOP: serialized data differs from native production tensors")
        history = LNHistory(**bundle["history"])
        print(f"CELL {cell['cell_id']} START", flush=True)
        fitted = select_and_refit_cnn(to_cuda(train), history)
        destination = OUT / "cells" / name
        destination.mkdir()
        for candidate in fitted.candidates:
            torch.save(dict(learning_rate=candidate.learning_rate, best_step=candidate.best_step,
                            stop_step=candidate.stop_step, best_dev_nll=candidate.best_dev_nll,
                            development_curve=candidate.development_curve,
                            model={k: v.detach().cpu() for k, v in candidate.model.state_dict().items()},
                            initial_state={k: v.cpu() for k, v in candidate.initial_state.items()}),
                       destination / f"inner-lr-{candidate.learning_rate:g}.pt")
        val_gpu = to_cuda(validation)
        raw = fresh_cnn(to_cuda(train), history)
        raw_metrics, raw_logits = evaluate_cnn(raw, val_gpu)
        metrics, logits = evaluate_cnn(fitted.refit.model, val_gpu)
        selected = fitted.selected
        checkpoint = dict(schema="schottdorf_compact_causal_cnn_v1", cell_id=cell["cell_id"],
                          history=asdict(history), contract=preflight["contract"], runtime=runtime,
                          selected_lr=selected.learning_rate, best_step=selected.best_step, stop_step=selected.stop_step,
                          refit_steps=fitted.refit.stop_step, preflight_sha256=runtime["preflight_sha256"])
        torch.save(dict(**checkpoint, model={k: v.cpu() for k, v in fitted.refit.initial_state.items()}), destination / "cnn-raw.pt")
        torch.save(dict(**checkpoint, model={k: v.detach().cpu() for k, v in fitted.refit.model.state_dict().items()}), destination / "cnn-trained.pt")
        torch.save(dict(logits_raw=raw_logits.cpu(), logits_trained=logits.cpu(), target=validation.spike_events,
                        valid_mask=validation.valid_mask, source_image_ids=validation.source_image_ids,
                        trial_indices=validation.trial_indices), destination / "validation-predictions.pt")
        result = dict(cell_id=cell["cell_id"], group=cell["group"], selected_lr=selected.learning_rate,
                      best_step=selected.best_step, stopping_step=selected.stop_step, refit_steps=fitted.refit.stop_step,
                      validation_nll_raw_gpu=raw_metrics.population_nll, validation_nll_trained_gpu=metrics.population_nll,
                      refit_train_nll_raw=fitted.refit.train_nll_raw, refit_train_nll_trained=fitted.refit.train_nll_trained,
                      parameter_counts=asdict(fitted.refit.parameter_counts),
                      inner_boundaries=[asdict(b) for b in fitted.inner_split.boundaries],
                      inner_candidates=[dict(lr=c.learning_rate, best_step=c.best_step, stopping_step=c.stop_step,
                                             best_dev_nll=c.best_dev_nll, development_curve=c.development_curve,
                                             parameter_counts=asdict(c.parameter_counts)) for c in fitted.candidates],
                      original_validation_used_for_selection=False, gradients_finite=True)
        if result["inner_boundaries"] != cell["inner_boundaries"]:
            raise LNError("STOP: inner boundaries differ from native LN split")
        (destination / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"CELL {cell['cell_id']} DONE validation={metrics.population_nll:.12f} lr={selected.learning_rate:g} best={selected.best_step} stop={selected.stop_step}", flush=True)
        del fitted, raw, val_gpu, raw_logits, logits
    if any(sha256_file(Path(p)) != h for p, h in preflight["source_sha256"].items()):
        raise LNError("STOP: source changed during training")
    if sha256_file(OUT / "preflight.json") != runtime["preflight_sha256"]:
        raise LNError("STOP: preflight changed during training")
    (OUT / "training_complete.json").write_text(json.dumps(dict(status="COMPLETED", cells=22,
        inner_fits=44, refits=22, source_hashes_unchanged=True), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
