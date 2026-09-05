from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina

if TYPE_CHECKING:
    from training.mechanistic_retina.stages import MechanisticSeedData


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    model: MechanisticGraphTemporalRetina
    data: MechanisticSeedData
    clamps: frozenset[PathwayClamp]
    h1: bool


@dataclass(frozen=True, slots=True)
class ProbeInput:
    preregistered_name: str
    stimulus: torch.Tensor
    history: torch.Tensor
    cell: int


def set_teacher_parameters(model: MechanisticGraphTemporalRetina) -> None:
    with torch.no_grad():
        bc = torch.full_like(model.bipolar.raw_weights, 0.08)
        bc[:, 0, 0] = bc.new_tensor((0.24, 0.15, 0.08))
        bc[:, 0, 1] = bc.new_tensor((0.16, 0.10, 0.05))
        bc[:, 1, 0] = bc.new_tensor((0.12, 0.18, 0.10))
        bc[:, 1, 1] = bc.new_tensor((0.08, 0.12, 0.07))
        model.bipolar.raw_weights.copy_(torch.log(torch.expm1(bc)))


def matched_bias(
    logits: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    low = torch.full_like(target, -12.0)
    high = torch.full_like(target, 12.0)
    for _ in range(64):
        middle = (low + high) * 0.5
        rate = _masked_rate(torch.sigmoid(logits + middle), mask)
        low = torch.where(rate < target, middle, low)
        high = torch.where(rate >= target, middle, high)
    return (low + high) * 0.5


def probe_effects(
    request: ProbeRequest,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    probes = preregistered_probe_inputs(request.model, request.data, request.h1)
    effects = []
    for probe in probes:
        with torch.no_grad():
            full = request.model.forward_sequence(
                probe.stimulus, observed_counts=probe.history
            ).logits
            ablated = request.model.forward_sequence(
                probe.stimulus,
                observed_counts=probe.history,
                clamps=request.clamps,
            ).logits
        effects.append(float((full - ablated).abs().mean()))
    return tuple(probe.preregistered_name for probe in probes), tuple(effects)


def preregistered_probe_inputs(
    model: MechanisticGraphTemporalRetina,
    data: MechanisticSeedData,
    h1: bool,
) -> tuple[ProbeInput, ...]:
    names = (
        (
            "diagnostic-h1-01",
            "diagnostic-h1-02",
            "diagnostic-h1-03",
            "diagnostic-h1-04",
            "diagnostic-h1-05",
        )
        if h1
        else (
            "diagnostic-ac-01",
            "diagnostic-ac-02",
            "diagnostic-ac-03",
            "diagnostic-ac-04",
            "diagnostic-ac-05",
        )
    )
    masks = model.feature_bank.supports
    annulus = masks.ac * (1 - masks.bc)
    far = (masks.h1 - masks.bc - annulus).clamp_min(0)
    cell = (
        int(far.sum(dim=1).argmax())
        if h1
        else int(annulus.sum(dim=1).argmax())
    )
    center = masks.bc[cell]
    surround = far[cell] if h1 else annulus[cell]
    time = torch.linspace(0, 1, 32)
    patterns = (time, time, time, torch.ones_like(time), torch.sin(4 * torch.pi * time))
    spatial = (
        center,
        (center + surround).clamp_max(1),
        surround,
        torch.ones_like(center),
        surround,
    )
    probes = []
    for name, temporal, support in zip(names, patterns, spatial, strict=True):
        stimulus = temporal.view(1, -1, 1) * support.view(1, 1, -1)
        history = torch.zeros((1, stimulus.shape[1], len(data.cell_ids)))
        probes.append(ProbeInput(name, stimulus, history, cell))
    return tuple(probes)


def _masked_rate(probability: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (probability * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp_min(1)


__all__ = [
    "ProbeInput",
    "ProbeRequest",
    "matched_bias",
    "preregistered_probe_inputs",
    "probe_effects",
    "set_teacher_parameters",
]
