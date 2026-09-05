from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SharedSubunitError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class SharedSubunitLayout:
    cell_positions: torch.Tensor
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]
    edge_index: torch.Tensor | None = None


class SharedSubunitMixer(nn.Module):
    def __init__(
        self,
        layout: SharedSubunitLayout,
        *,
        radius_deg: float,
        trainable: bool,
    ) -> None:
        super().__init__()
        cell_count = layout.cell_positions.shape[0]
        if len(layout.cell_types) != cell_count or len(layout.polarities) != cell_count:
            raise SharedSubunitError("shared-subunit metadata lengths differ")
        same_group = torch.tensor(
            [
                [
                    layout.cell_types[target] == layout.cell_types[source]
                    and layout.polarities[target] == layout.polarities[source]
                    for source in range(cell_count)
                ]
                for target in range(cell_count)
            ],
            device=layout.cell_positions.device,
        )
        if layout.edge_index is None:
            distances = torch.cdist(
                layout.cell_positions.float(), layout.cell_positions.float()
            )
            support = same_group & (distances <= radius_deg)
            support.fill_diagonal_(True)
            edges = support.nonzero(as_tuple=False).T.contiguous()
        else:
            supplied = layout.edge_index
            valid_shape = supplied.ndim == 2 and supplied.shape[0] == 2
            valid_dtype = supplied.dtype in (torch.int32, torch.int64)
            if not valid_shape or not valid_dtype or supplied.shape[1] < cell_count:
                raise SharedSubunitError("explicit shared-subunit edges must be integer [2,E]")
            edges = supplied.to(
                device=layout.cell_positions.device, dtype=torch.long
            ).contiguous()
            in_bounds = bool((edges >= 0).all() and (edges < cell_count).all())
            linear = edges[0] * cell_count + edges[1]
            if not in_bounds or torch.unique(linear).numel() != edges.shape[1]:
                raise SharedSubunitError("explicit shared-subunit edges are invalid or duplicated")
            support = torch.zeros(
                (cell_count, cell_count),
                dtype=torch.bool,
                device=layout.cell_positions.device,
            )
            support[edges[0], edges[1]] = True
            valid_support = (
                bool(torch.diagonal(support).all())
                and torch.equal(support, support.T)
                and bool(same_group[edges[0], edges[1]].all())
            )
            if not valid_support:
                raise SharedSubunitError(
                    "explicit shared-subunit support must be symmetric, self-complete, and class-local"
                )
        initial = torch.where(
            edges[0] == edges[1],
            torch.ones(edges.shape[1]),
            torch.full((edges.shape[1],), 0.10),
        )
        raw = torch.log(torch.expm1(initial.clamp_min(1e-6)))
        row_degree = support.sum(dim=1)
        mixing_edges = row_degree[edges[0]] > 1
        if cell_count == 1:
            self.register_buffer("raw_connections", raw)
        elif not bool(mixing_edges.any()):
            self.register_buffer("raw_connections", raw[mixing_edges])
        else:
            self.raw_connections = nn.Parameter(raw[mixing_edges], requires_grad=trainable)
        self.register_buffer("edge_index", edges)
        self.register_buffer("cell_order", torch.arange(cell_count))

    def connection_matrix(self) -> torch.Tensor:
        if self.cell_order.numel() == 1:
            return self.raw_connections.new_ones((1, 1))
        matrix = self.raw_connections.new_zeros(
            (self.cell_order.numel(), self.cell_order.numel())
        )
        row_degree = torch.bincount(self.edge_index[0], minlength=self.cell_order.numel())
        self_only_rows = (row_degree == 1).nonzero().flatten()
        matrix[self_only_rows, self_only_rows] = 1.0
        mixing_edges = self.edge_index[:, row_degree[self.edge_index[0]] > 1]
        matrix[mixing_edges[0], mixing_edges[1]] = F.softplus(
            self.raw_connections
        )
        return matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,btjpsr->btipsr", self.connection_matrix(), features)

    def mix_kernels(self, kernels: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "ij,jpsrlc->ipsrlc",
            self.connection_matrix(),
            kernels,
        )


__all__ = ["SharedSubunitError", "SharedSubunitLayout", "SharedSubunitMixer"]
