from __future__ import annotations

import numpy as np
import torch

PositionArray = np.ndarray | torch.Tensor


class GeometryError(ValueError):
    pass


def local_gaussian_weights(
    source_positions: PositionArray,
    target_positions: PositionArray,
    radius_degs: float,
    sigma_degs: float,
    allow_empty_rows: bool = False,
) -> torch.Tensor:
    if radius_degs <= 0 or sigma_degs <= 0:
        raise GeometryError("radius_degs and sigma_degs must be positive")

    source = _as_positions("source_positions", source_positions)
    target = _as_positions("target_positions", target_positions)
    distances = torch.cdist(target, source)
    mask = distances <= radius_degs
    weights = torch.exp(-0.5 * torch.square(distances / sigma_degs)) * mask
    row_sum = weights.sum(dim=1, keepdim=True)
    if not allow_empty_rows and torch.any(row_sum <= 0):
        raise GeometryError("Every target must have at least one source inside radius_degs")

    weights = torch.where(row_sum > 0, weights / row_sum.clamp_min(1e-12), weights)
    indices = torch.nonzero(weights > 0, as_tuple=False).t().contiguous()
    values = weights[weights > 0].contiguous()
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(target.shape[0], source.shape[0]),
    ).coalesce()


def nearest_one_to_one_weights(
    source_positions: PositionArray,
    target_positions: PositionArray,
) -> torch.Tensor:
    source = _as_positions("source_positions", source_positions)
    target = _as_positions("target_positions", target_positions)
    nearest = torch.argmin(torch.cdist(target, source), dim=1)
    rows = torch.arange(target.shape[0], dtype=torch.long)
    indices = torch.stack([rows, nearest.to(torch.long)], dim=0)
    values = torch.ones(target.shape[0], dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(target.shape[0], source.shape[0]),
    ).coalesce()


def _as_positions(name: str, positions: PositionArray) -> torch.Tensor:
    tensor = torch.as_tensor(positions, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise GeometryError(f"{name} must have shape [N,2]")
    if not torch.isfinite(tensor).all():
        raise GeometryError(f"{name} must contain finite values")
    return tensor
