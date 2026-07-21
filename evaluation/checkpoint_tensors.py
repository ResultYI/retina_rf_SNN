from __future__ import annotations

import torch

from models.cells.rgc import RGCOutput, RGCPopulationTensors


def last_output(history: RGCOutput) -> RGCOutput:
    return RGCOutput(
        spikes=_last_populations(history.spikes),
        rates=_last_populations(history.rates),
    )


def slice_output(output: RGCOutput, count: int) -> RGCOutput:
    return RGCOutput(
        spikes=_slice_populations(output.spikes, count),
        rates=_slice_populations(output.rates, count),
    )


def concat_outputs(outputs: list[RGCOutput]) -> RGCOutput:
    return RGCOutput(
        spikes=_concat_populations(tuple(item.spikes for item in outputs)),
        rates=_concat_populations(tuple(item.rates for item in outputs)),
    )


def _last_populations(values: RGCPopulationTensors) -> RGCPopulationTensors:
    return RGCPopulationTensors(
        values.midget[:, -1],
        values.parasol[:, -1],
    )


def _slice_populations(
    values: RGCPopulationTensors,
    count: int,
) -> RGCPopulationTensors:
    return RGCPopulationTensors(
        values.midget[:count].detach(),
        values.parasol[:count].detach(),
    )


def _concat_populations(
    values: tuple[RGCPopulationTensors, ...],
) -> RGCPopulationTensors:
    return RGCPopulationTensors(
        torch.cat(tuple(item.midget for item in values)),
        torch.cat(tuple(item.parasol for item in values)),
    )
