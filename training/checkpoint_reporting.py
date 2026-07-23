from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from training.checkpointing import load_checkpoint


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    filename: str
    exists: bool
    optimizer_step: int | None
    reference_energy: float | None
    target_budget: float | None
    best_reconstruction_mse: float | None
    best_feasible_mse: float | None


def write_checkpoint_summaries(
    output_dir: Path,
    selected_checkpoint: Path,
    device: torch.device,
) -> None:
    names = (
        "checkpoint_last.pt",
        "checkpoint_best_reconstruction.pt",
        "checkpoint_best_feasible.pt",
    )
    for name in names:
        summary = checkpoint_summary(output_dir / name, device)
        (output_dir / name.replace(".pt", "_summary.json")).write_text(
            json.dumps(asdict(summary), indent=2),
            encoding="utf-8",
        )
    selected_payload = load_checkpoint(selected_checkpoint, device)
    (output_dir / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "filename": selected_checkpoint.name,
                "reason": (
                    "best_feasible"
                    if selected_checkpoint.name == "checkpoint_best_feasible.pt"
                    else "best_reconstruction"
                ),
                "optimizer_step": int(selected_payload["optimizer_step"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def checkpoint_summary(
    path: Path,
    device: torch.device,
) -> CheckpointSummary:
    if not path.exists():
        return CheckpointSummary(path.name, False, None, None, None, None, None)
    payload = load_checkpoint(path, device)
    energy = payload["energy_state"]
    validation = payload["validation_state"]
    return CheckpointSummary(
        filename=path.name,
        exists=True,
        optimizer_step=int(payload["optimizer_step"]),
        reference_energy=energy["reference_energy"],
        target_budget=energy["target_budget"],
        best_reconstruction_mse=validation["best_reconstruction_mse"],
        best_feasible_mse=validation["best_feasible_mse"],
    )


def write_training_row(
    output_dir: Path,
    optimizer_step: int,
    metrics: dict[str, float | int | bool | None],
) -> None:
    row = {"optimizer_step": optimizer_step, **metrics}
    with (output_dir / "training.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


__all__ = [
    "CheckpointSummary",
    "checkpoint_summary",
    "write_checkpoint_summaries",
    "write_training_row",
]
