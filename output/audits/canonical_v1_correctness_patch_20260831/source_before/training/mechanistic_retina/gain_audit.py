from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PathwayGainAudit:
    name: str
    all_gradient_nonzero: bool
    all_best_updated: bool
    min_peak_abs_gradient: float
    max_peak_abs_gradient: float
    best_update_norm: float
    best: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CellSpecificGainAudit:
    pathways: tuple[PathwayGainAudit, ...]


__all__ = ["CellSpecificGainAudit", "PathwayGainAudit"]
