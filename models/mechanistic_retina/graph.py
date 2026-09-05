from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class GraphBuildError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


class ConeGraph(nn.Module):
    def __init__(self, positions: torch.Tensor, radius_deg: float) -> None:
        super().__init__()
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise GraphBuildError("cone positions must be [cone,2]")
        if radius_deg <= 0 or not bool(torch.isfinite(positions).all()):
            raise GraphBuildError("cone graph inputs must be finite and positive")
        distances = torch.cdist(positions.float(), positions.float())
        support = distances <= radius_deg
        indices = support.nonzero(as_tuple=False).T.contiguous()
        weights = torch.exp(-0.5 * (distances[support] / (0.5 * radius_deg)) ** 2)
        row_sum = torch.zeros(positions.shape[0], dtype=weights.dtype)
        row_sum.scatter_add_(0, indices[0], weights)
        weights = weights / row_sum[indices[0]]
        self.register_buffer("edge_index", indices)
        self.register_buffer("edge_weight", weights)
        self.register_buffer("node_order", torch.arange(positions.shape[0]))

    @property
    def edge_count(self) -> int:
        return self.edge_weight.numel()

    def sparse_kernel(self) -> torch.Tensor:
        size = self.node_order.numel()
        return torch.sparse_coo_tensor(
            self.edge_index,
            self.edge_weight,
            (size, size),
            device=self.edge_weight.device,
        ).coalesce()

    def dense_kernel(self) -> torch.Tensor:
        return self.sparse_kernel().to_dense()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shape = values.shape
        flattened = values.reshape(-1, shape[-1])
        applied = torch.sparse.mm(self.sparse_kernel(), flattened.T).T
        return applied.reshape(shape)

    def transpose_apply(self, values: torch.Tensor) -> torch.Tensor:
        shape = values.shape
        flattened = values.reshape(-1, shape[-1])
        applied = torch.sparse.mm(self.sparse_kernel().transpose(0, 1), flattened.T).T
        return applied.reshape(shape)


__all__ = ["ConeGraph", "GraphBuildError"]
