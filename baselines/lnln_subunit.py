from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def lnln_parameter_count(
    subunits_per_cell: int,
    cell_count: int,
    group_count: int,
    *,
    lag_steps: int = 16,
    history_lags: int = 4,
) -> int:
    return (
        group_count * subunits_per_cell * lag_steps
        + cell_count * subunits_per_cell
        + cell_count * history_lags
        + cell_count
    )


def select_subunit_count(
    target_parameters: int,
    cell_count: int,
    group_count: int,
) -> int:
    return min(
        range(1, 33),
        key=lambda count: (
            abs(lnln_parameter_count(count, cell_count, group_count) - target_parameters),
            count,
        ),
    )


class LNPNLSubunit(nn.Module):
    def __init__(
        self,
        cone_positions: torch.Tensor,
        cell_positions: torch.Tensor,
        cell_types: tuple[str, ...],
        polarities: tuple[str, ...],
        subunits_per_cell: int,
        *,
        lag_steps: int = 16,
        history_lags: int = 4,
    ) -> None:
        super().__init__()
        if subunits_per_cell < 1:
            raise ValueError("LN-LN requires at least one subunit per cell")
        group_index, group_count = _group_indices(cell_types, polarities)
        spatial, support = _local_kernels(
            cone_positions, cell_positions, subunits_per_cell
        )
        sharing = _sharing_matrix(cell_positions, group_index)
        self.temporal = nn.Parameter(
            0.03 * torch.randn(group_count, subunits_per_cell, lag_steps)
        )
        self.readout = nn.Parameter(
            0.03 * torch.randn(cell_positions.shape[0], subunits_per_cell)
        )
        self.history = nn.Parameter(
            torch.zeros(cell_positions.shape[0], history_lags)
        )
        self.bias = nn.Parameter(torch.full((cell_positions.shape[0],), -2.0))
        self.register_buffer("group_index", group_index)
        self.register_buffer("spatial_kernels", spatial)
        self.register_buffer("local_support", support)
        self.register_buffer("sharing", sharing)

    @property
    def lag_steps(self) -> int:
        return self.temporal.shape[-1]

    @property
    def shared_subunit_fraction(self) -> float:
        nonzero = self.sharing > 0
        off_diagonal = nonzero & ~torch.eye(
            nonzero.shape[0], dtype=torch.bool, device=nonzero.device
        )
        return float(off_diagonal.sum() / nonzero.sum().clamp_min(1))

    def forward(
        self, cones: torch.Tensor, observed_counts: torch.Tensor
    ) -> torch.Tensor:
        local = torch.einsum("btc,nsc->btns", cones, self.spatial_kernels)
        shared = torch.einsum("ij,btjs->btis", self.sharing, local)
        padded = F.pad(shared, (0, 0, 0, 0, self.lag_steps - 1, 0))
        windows = padded.unfold(1, self.lag_steps, 1)
        filters = self.temporal[self.group_index]
        drive = (windows * filters[None, None]).sum(dim=-1)
        subunits = F.softplus(drive) - math.log(2.0)
        logits = (subunits * self.readout[None, None]).sum(dim=-1) + self.bias
        return _add_history(logits, observed_counts, self.history)

    def subunit_kernels(self) -> torch.Tensor:
        spatial = torch.einsum(
            "ij,jsc->isc", self.sharing, self.spatial_kernels
        )
        temporal = self.temporal[self.group_index]
        return torch.einsum("isl,isc->islc", temporal, spatial)


def _group_indices(
    cell_types: tuple[str, ...], polarities: tuple[str, ...]
) -> tuple[torch.Tensor, int]:
    keys = tuple(zip(cell_types, polarities, strict=True))
    groups = tuple(dict.fromkeys(keys))
    return torch.tensor(tuple(groups.index(key) for key in keys)), len(groups)


def _local_kernels(
    cones: torch.Tensor, cells: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(cells.float(), cones.float())
    radii = torch.linspace(0.07, 0.15, count)
    rows = []
    masks = []
    for radius in radii:
        support = distances <= radius
        nearest = distances.argmin(dim=1)
        support[torch.arange(cells.shape[0]), nearest] = True
        values = torch.exp(-0.5 * (distances / radius.clamp_min(1e-6)) ** 2) * support
        rows.append(values / values.sum(dim=1, keepdim=True).clamp_min(1e-12))
        masks.append(support)
    return torch.stack(rows, dim=1), torch.stack(masks, dim=1)


def _sharing_matrix(cells: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    distance = torch.cdist(cells.float(), cells.float())
    support = (distance <= 0.08) & (groups[:, None] == groups[None, :])
    support.fill_diagonal_(True)
    values = support.float()
    return values / values.sum(dim=1, keepdim=True).clamp_min(1)


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


__all__ = [
    "LNPNLSubunit",
    "lnln_parameter_count",
    "select_subunit_count",
]
