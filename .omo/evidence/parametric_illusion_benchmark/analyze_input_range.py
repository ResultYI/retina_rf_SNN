from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Final

import numpy as np
import torch


OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
CNN: Final = ROOT / ".omo/evidence/compact_causal_cnn_baseline"
sys.path.insert(0, str(OUT))
Scalar = str | int | float | bool


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def percentile(values: torch.Tensor, value: float) -> float:
    return float((values <= value).double().mean() * 100)


def write_csv(path: Path, rows: list[dict[str, Scalar]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.tensor((0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0))
    result = torch.quantile(values.float(), quantiles)
    return {f"q{float(q):g}": float(v) for q, v in zip(quantiles, result, strict=True)}


def main() -> None:
    preflight = json.loads((CNN / "preflight.json").read_text(encoding="utf-8"))
    members: dict[str, list[str]] = {}
    representatives: dict[str, str] = {}
    for cell in preflight["cells"]:
        digest = cell["tensor_sha256"]["train_input_sha256"]
        members.setdefault(digest, []).append(cell["cell_id"])
        representatives.setdefault(digest, cell["cell_id"])
    scalar_blocks, frame_blocks = [], []
    source_rows: list[dict[str, Scalar]] = []
    for digest, cell_ids in members.items():
        cell_id = representatives[digest]
        bundle = torch.load(CNN / "inputs" / f"{cell_id.replace('#', '_')}.pt", weights_only=True)
        values = bundle["train"]["cone_drive"].float()
        assert tensor_sha256(values) == digest
        flat = values.flatten()
        frame = values.reshape(-1, values.shape[-1])
        scalar_blocks.append(flat)
        frame_blocks.append(torch.stack((frame.mean(1), frame.std(1, unbiased=False),
                                         frame.square().mean(1).sqrt(), frame.amin(1), frame.amax(1)), dim=1))
        source_rows.append({"tensor_sha256": digest, "representative_cell": cell_id,
                            "member_cells": ";".join(cell_ids), "member_count": len(cell_ids),
                            "sequences": values.shape[0], "frames": frame.shape[0], "scalar_values": flat.numel(),
                            "minimum": float(flat.min()), "maximum": float(flat.max())})
    scalars, frames = torch.cat(scalar_blocks), torch.cat(frame_blocks)
    metrics = ("frame_mean", "frame_std", "frame_rms", "frame_min", "frame_max")
    range_summary = {
        "scope": "three unique frozen full-training stimulus tensors; duplicated cell presentations removed",
        "unique_tensor_count": len(members), "represented_cells": sum(len(value) for value in members.values()),
        "frames": frames.shape[0], "scalar_values": scalars.numel(), "scalar": summary(scalars),
        "frame_metrics": {name: summary(frames[:, index]) for index, name in enumerate(metrics)},
    }
    stimuli = torch.load(OUT / "stimuli.pt", weights_only=True)
    probe_rows: list[dict[str, Scalar]] = []
    for name, patch in zip(stimuli["names"], stimuli["patches"], strict=True):
        values = patch.flatten()
        measurements = (float(values.mean()), float(values.std(unbiased=False)),
                        float(values.square().mean().sqrt()), float(values.min()), float(values.max()))
        row: dict[str, Scalar] = {"name": name, "scalar_min": measurements[3], "scalar_max": measurements[4],
                                 "scalar_min_percentile": percentile(scalars, measurements[3]),
                                 "scalar_max_percentile": percentile(scalars, measurements[4]),
                                 "outside_training_scalar_minmax": measurements[3] < float(scalars.min())
                                 or measurements[4] > float(scalars.max())}
        for index, (metric, value) in enumerate(zip(metrics, measurements, strict=True)):
            row[metric] = value
            row[f"{metric}_percentile"] = percentile(frames[:, index], value)
        probe_rows.append(row)
    levels = sorted({float(value) for patch in stimuli["patches"] for value in torch.unique(patch)})
    level_rows = [{"input_level": level, "training_scalar_percentile": percentile(scalars, level),
                   "inside_training_minmax": float(scalars.min()) <= level <= float(scalars.max())}
                  for level in levels]
    write_csv(OUT / "natural_training_sources.csv", source_rows)
    write_csv(OUT / "probe_input_position.csv", probe_rows)
    write_csv(OUT / "probe_scalar_levels.csv", level_rows)
    (OUT / "natural_training_range.json").write_text(json.dumps(range_summary, indent=2), encoding="utf-8")
    print(json.dumps(range_summary, indent=2))


if __name__ == "__main__":
    main()
