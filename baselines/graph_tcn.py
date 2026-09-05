from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def graph_tcn_parameter_count(
    width: int, cell_count: int, *, history_lags: int = 4
) -> int:
    return 2 * width * width + 12 * width + cell_count * width + cell_count * (
        history_lags + 1
    )


def select_hidden_width(target_parameters: int, cell_count: int) -> int:
    return min(
        range(1, 65),
        key=lambda width: (
            abs(graph_tcn_parameter_count(width, cell_count) - target_parameters),
            width,
        ),
    )


class _CausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.depthwise = nn.Conv1d(
            width, width, 3, dilation=dilation, groups=width
        )
        self.pointwise = nn.Conv1d(width, width, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padded = F.pad(values, (2 * self.dilation, 0))
        return values + F.gelu(self.pointwise(self.depthwise(padded)))


class GraphTCN(nn.Module):
    def __init__(
        self,
        cone_positions: torch.Tensor,
        cell_positions: torch.Tensor,
        width: int,
        *,
        history_lags: int = 4,
    ) -> None:
        super().__init__()
        cone_graph = _local_normalized_graph(cone_positions, 0.11)
        cell_pool, cell_support = _cell_pool(
            cone_positions, cell_positions, 0.18
        )
        self.input_projection = nn.Linear(1, width)
        self.blocks = nn.ModuleList((_CausalBlock(width, 1), _CausalBlock(width, 7)))
        self.readout = nn.Parameter(0.03 * torch.randn(cell_positions.shape[0], width))
        self.bias = nn.Parameter(torch.full((cell_positions.shape[0],), -2.0))
        self.history = nn.Parameter(
            torch.zeros(cell_positions.shape[0], history_lags)
        )
        self.register_buffer("cone_graph", cone_graph)
        self.register_buffer("cell_pool", cell_pool)
        self.register_buffer("cell_support", cell_support)

    @property
    def receptive_field_steps(self) -> int:
        return 17

    @property
    def width(self) -> int:
        return self.readout.shape[1]

    def forward(
        self,
        cones: torch.Tensor,
        observed_counts: torch.Tensor,
        *,
        ablate_channel: int | None = None,
    ) -> torch.Tensor:
        hidden = self.features(cones)
        if ablate_channel is not None:
            hidden = hidden.clone()
            hidden[..., ablate_channel] = 0
        logits = (hidden * self.readout[None, None]).sum(dim=-1) + self.bias
        return _add_history(logits, observed_counts, self.history)

    def features(self, cones: torch.Tensor) -> torch.Tensor:
        graph = torch.einsum("btc,cd->btd", cones, self.cone_graph)
        local = torch.einsum("btc,nc->btn", graph, self.cell_pool)
        hidden = self.input_projection(local.unsqueeze(-1))
        batch, time, cells, width = hidden.shape
        sequence = hidden.permute(0, 2, 3, 1).reshape(batch * cells, width, time)
        for block in self.blocks:
            sequence = block(sequence)
        return sequence.reshape(batch, cells, width, time).permute(0, 3, 1, 2)

    def gradient_attribution(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.readout.detach().abs().mean(dim=0))


def _local_normalized_graph(positions: torch.Tensor, radius: float) -> torch.Tensor:
    distance = torch.cdist(positions.float(), positions.float())
    support = distance <= radius
    support.fill_diagonal_(True)
    values = torch.exp(-0.5 * (distance / radius) ** 2) * support
    return values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _cell_pool(
    cones: torch.Tensor, cells: torch.Tensor, radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    distance = torch.cdist(cells.float(), cones.float())
    support = distance <= radius
    nearest = distance.argmin(dim=1)
    support[torch.arange(cells.shape[0]), nearest] = True
    values = torch.exp(-0.5 * (distance / radius) ** 2) * support
    return values / values.sum(dim=1, keepdim=True).clamp_min(1e-12), support


def _add_history(
    logits: torch.Tensor,
    observed_counts: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    result = logits.clone()
    for lag in range(1, history.shape[1] + 1):
        if lag >= observed_counts.shape[1]:
            break
        result[:, lag:] += observed_counts[:, :-lag] * history[:, lag - 1]
    return result


__all__ = ["GraphTCN", "graph_tcn_parameter_count", "select_hidden_width"]
