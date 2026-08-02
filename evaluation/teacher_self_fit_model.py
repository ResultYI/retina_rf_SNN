from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class TeacherSelfFitModelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TeacherSelfFitShape:
    seed_count: int
    cell_count: int
    lag_count: int
    cone_count: int


@dataclass(frozen=True, slots=True)
class TeacherSelfFitBatch:
    normalized_cones: torch.Tensor
    context_basis: torch.Tensor
    history: torch.Tensor


class TeacherSelfFitModel(nn.Module):
    def __init__(self, shape: TeacherSelfFitShape) -> None:
        super().__init__()
        kernel_shape = (
            shape.seed_count,
            shape.cell_count,
            shape.lag_count,
            shape.cone_count,
        )
        self.base_kernel = nn.Parameter(torch.zeros(kernel_shape))
        self.context_kernel = nn.Parameter(torch.zeros(kernel_shape))
        self.history_gain = nn.Parameter(
            torch.full((shape.seed_count, shape.cell_count), -1.0)
        )
        self.bias = nn.Parameter(torch.zeros(shape.seed_count, shape.cell_count))

    def forward(self, batch: TeacherSelfFitBatch) -> torch.Tensor:
        seed_count = self.base_kernel.shape[0]
        stimulus_count, time_count = batch.normalized_cones.shape[:2]
        cell_count = self.base_kernel.shape[1]
        drive = self.bias[:, None, None, :].expand(
            seed_count,
            stimulus_count,
            time_count,
            cell_count,
        ).clone()
        context_drive = torch.zeros_like(drive)
        for lag in range(self.base_kernel.shape[2]):
            if lag >= time_count:
                break
            cones = batch.normalized_cones[:, : time_count - lag]
            drive[:, :, lag:] += torch.einsum(
                "stq,mcq->mstc",
                cones,
                self.base_kernel[:, :, lag],
            )
            context_drive[:, :, lag:] += torch.einsum(
                "stq,mcq->mstc",
                cones,
                self.context_kernel[:, :, lag],
            )
        drive += batch.context_basis[None, :, :, None] * context_drive
        return (
            drive[:, :, None]
            + batch.history * self.history_gain[:, None, None, None, :]
        )


def history_trace(spikes: np.ndarray, *, decay: float) -> np.ndarray:
    values = np.asarray(spikes, dtype=np.float32)
    if values.ndim < 2 or not 0 <= decay <= 1:
        raise TeacherSelfFitModelError(
            "History trace requires [...,time,cell] spikes and valid decay"
        )
    trace = np.zeros_like(values, dtype=np.float32)
    state = np.zeros(values.shape[:-2] + (values.shape[-1],), dtype=np.float32)
    for time in range(values.shape[-2]):
        trace[..., time, :] = state
        state = np.float32(decay) * state + values[..., time, :]
    return trace


def context_recovery_basis(
    context_ids: tuple[str, ...],
    time_count: int,
) -> np.ndarray:
    if time_count < 1 or len(context_ids) < 1:
        raise TeacherSelfFitModelError("Context basis requires contexts and time bins")
    basis = np.zeros((len(context_ids), time_count), dtype=np.float32)
    start = max(1, time_count - min(64, time_count // 2))
    recovery = np.exp(-np.arange(time_count - start, dtype=np.float32) / 30.0)
    for stimulus, context in enumerate(context_ids):
        if context == "high":
            basis[stimulus, start:] = recovery
        elif context != "low":
            raise TeacherSelfFitModelError("Teacher self-fit requires low/high contexts")
    return basis


__all__ = [
    "TeacherSelfFitBatch",
    "TeacherSelfFitModel",
    "TeacherSelfFitModelError",
    "TeacherSelfFitShape",
    "context_recovery_basis",
    "history_trace",
]
