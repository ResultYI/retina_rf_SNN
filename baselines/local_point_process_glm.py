from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class LocalGLMError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


class LocalPointProcessGLM(nn.Module):
    def __init__(
        self,
        cone_positions: torch.Tensor,
        cell_positions: torch.Tensor,
        radius_deg: float | None,
        temporal_lags: int,
        history_lags: int = 4,
        *,
        support_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if temporal_lags < 1 or history_lags < 1:
            raise LocalGLMError("GLM support and lag settings must be positive")
        if cone_positions.ndim != 2 or cell_positions.ndim != 2:
            raise LocalGLMError("GLM geometry must be [node,coordinate]")
        if support_mask is None:
            if radius_deg is None or radius_deg <= 0:
                raise LocalGLMError("radial GLM support requires a positive radius")
            declared_support = torch.cdist(
                cell_positions, cone_positions
            ) <= radius_deg
        else:
            expected = (cell_positions.shape[0], cone_positions.shape[0])
            if radius_deg is not None or support_mask.shape != expected:
                raise LocalGLMError("explicit GLM support must match cells and cones")
            declared_support = support_mask.to(dtype=torch.bool)
        supports = tuple(
            torch.nonzero(declared_support[cell], as_tuple=False).flatten()
            for cell in range(cell_positions.shape[0])
        )
        if any(support.numel() == 0 for support in supports):
            raise LocalGLMError("every cell needs at least one local cone")
        maximum = max(support.numel() for support in supports)
        padded = torch.zeros(
            cell_positions.shape[0], maximum, dtype=torch.long
        )
        support_mask = torch.zeros(
            cell_positions.shape[0], maximum, dtype=torch.bool
        )
        for cell, support in enumerate(supports):
            padded[cell, : support.numel()] = support
            support_mask[cell, : support.numel()] = True
        self.register_buffer("support_indices", padded)
        self.register_buffer("support_mask", support_mask)
        self.kernels = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    temporal_lags,
                    support.numel(),
                    dtype=cone_positions.dtype,
                    device=cone_positions.device,
                )
            )
            for support in supports
        )
        self.history = nn.Parameter(
            torch.zeros(
                cell_positions.shape[0],
                history_lags,
                dtype=cone_positions.dtype,
                device=cone_positions.device,
            )
        )
        self.bias = nn.Parameter(
            torch.zeros(
                cell_positions.shape[0],
                dtype=cone_positions.dtype,
                device=cone_positions.device,
            )
        )

    @property
    def support_counts(self) -> tuple[int, ...]:
        return tuple(int(mask.sum()) for mask in self.support_mask)

    @property
    def temporal_lags(self) -> int:
        return self.kernels[0].shape[0]

    def forward(
        self,
        cones: torch.Tensor,
        observed_counts: torch.Tensor,
    ) -> torch.Tensor:
        if cones.ndim != 3 or observed_counts.ndim != 3:
            raise LocalGLMError("GLM inputs must be [batch,time,feature]")
        local = cones[:, :, self.support_indices]
        kernel = self.padded_kernel()
        logits = self.bias.view(1, 1, -1).expand(
            cones.shape[0], cones.shape[1], -1
        ).clone()
        for lag in range(self.temporal_lags):
            if lag >= cones.shape[1]:
                break
            logits[:, lag:] += (
                local[:, : cones.shape[1] - lag]
                * kernel[:, lag].view(1, 1, *kernel[:, lag].shape)
            ).sum(dim=-1)
        return self.add_history(logits, observed_counts)

    def static_flash_logits(
        self,
        spatial_by_source: torch.Tensor,
        active_time: torch.Tensor,
    ) -> torch.Tensor:
        local = spatial_by_source[:, self.support_indices]
        projections = torch.einsum(
            "ins,nls->inl", local, self.padded_kernel()
        )
        time_basis = active_time.new_zeros(
            active_time.shape[0], self.temporal_lags
        )
        for lag in range(self.temporal_lags):
            if lag >= active_time.shape[0]:
                break
            time_basis[lag:, lag] = active_time[: active_time.shape[0] - lag]
        return torch.einsum("inl,tl->itn", projections, time_basis)

    def add_history(
        self,
        logits: torch.Tensor,
        observed_counts: torch.Tensor,
    ) -> torch.Tensor:
        for lag in range(1, self.history.shape[1] + 1):
            if lag >= observed_counts.shape[1]:
                break
            logits[:, lag:] += (
                observed_counts[:, : observed_counts.shape[1] - lag]
                * self.history[:, lag - 1]
            )
        return logits

    def padded_kernel(self) -> torch.Tensor:
        maximum = self.support_indices.shape[1]
        return torch.stack(
            tuple(
                F.pad(kernel, (0, maximum - kernel.shape[1]))
                for kernel in self.kernels
            )
        )


__all__ = ["LocalGLMError", "LocalPointProcessGLM"]
