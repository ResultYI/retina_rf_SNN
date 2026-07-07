from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from models.cells.amacrine import A2AmacrineConfig
from models.cells.bipolar import BipolarConfig
from models.cells.horizontal import H1HorizontalConfig
from models.cells.rgc import RGCConfig
from models.decoder.local_decoder import LocalDecoderConfig

_TIME_AXIS_CV_MAX: Final[float] = 1e-3


class PhysiologyProfileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PhysiologyProfile:
    name: str
    species_priority: tuple[str, ...]
    h1: H1HorizontalConfig
    bipolar: BipolarConfig
    a2: A2AmacrineConfig
    rgc: RGCConfig
    decoder: LocalDecoderConfig


def dt_ms_from_time_axis_seconds(time_axis_seconds: np.ndarray) -> float:
    axis = np.asarray(time_axis_seconds, dtype=np.float64).reshape(-1)
    if axis.size < 2:
        raise PhysiologyProfileError("time_axis_seconds needs at least two frames")
    if not np.isfinite(axis).all():
        raise PhysiologyProfileError("time_axis_seconds must be finite")
    intervals = np.diff(axis)
    if np.any(intervals <= 0):
        raise PhysiologyProfileError("time_axis_seconds must be strictly increasing")
    interval_mean = float(intervals.mean())
    interval_cv = float(intervals.std() / (interval_mean + 1e-12))
    if interval_cv > _TIME_AXIS_CV_MAX:
        raise PhysiologyProfileError("time_axis_seconds has unstable frame interval")
    return interval_mean * 1000.0


def human_macaque_v1(*, dt_ms: float, horizon_count: int) -> PhysiologyProfile:
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise PhysiologyProfileError("dt_ms must be positive and finite")
    if horizon_count < 1:
        raise PhysiologyProfileError("horizon_count must be positive")
    return PhysiologyProfile(
        name="human_macaque_v1",
        species_priority=("human", "macaque", "marmoset"),
        h1=H1HorizontalConfig(
            radius_degs=0.16,
            sigma_degs=0.10,
            feedback_radius_degs=0.21,
            feedback_sigma_degs=0.12,
            h1_spacing_degs=0.20,
            dt_ms=dt_ms,
            initial_tau_ms=50.0,
            tau_min_ms=10.0,
            tau_max_ms=200.0,
            initial_gain=0.01,
            gain_max=0.20,
        ),
        bipolar=BipolarConfig(
            dt_ms=dt_ms,
            initial_tau_sustained_ms=80.0,
            tau_sustained_min_ms=60.0,
            tau_sustained_max_ms=200.0,
            initial_tau_transient_ms=20.0,
            tau_transient_min_ms=5.0,
            tau_transient_max_ms=40.0,
            initial_g_ab_sustained=0.01,
            g_ab_sustained_max=0.10,
            initial_g_ab_transient=0.01,
            g_ab_transient_max=0.30,
        ),
        a2=A2AmacrineConfig(
            radius_degs=0.16,
            sigma_degs=0.10,
            dt_ms=dt_ms,
            initial_tau_sustained_ms=100.0,
            tau_sustained_min_ms=40.0,
            tau_sustained_max_ms=250.0,
            initial_tau_transient_ms=40.0,
            tau_transient_min_ms=15.0,
            tau_transient_max_ms=100.0,
            initial_g_ba_sustained=0.03,
            g_ba_sustained_max=0.30,
            initial_g_ba_transient=0.05,
            g_ba_transient_max=0.50,
        ),
        rgc=RGCConfig(
            parasol_radius_degs=0.16,
            parasol_sigma_degs=0.10,
            residual_radius_degs=0.25,
            residual_sigma_degs=0.12,
            dt_ms=dt_ms,
            membrane_tau_ms=20.0,
            adaptation_tau_ms=80.0,
            rate_tau_ms=50.0,
            threshold=0.20,
            surrogate_slope=5.0,
            adaptation_strength=0.10,
            initial_g_ag_midget=0.01,
            g_ag_midget_max=0.10,
            initial_g_ag_parasol=0.03,
            g_ag_parasol_max=0.30,
            initial_g_ag_residual=0.01,
            g_ag_residual_max=0.10,
            residual_drive_scale=0.25,
        ),
        decoder=LocalDecoderConfig(
            horizon_count=horizon_count,
            fine_radius_degs=0.16,
            fine_sigma_degs=0.08,
            coarse_radius_degs=0.21,
            coarse_sigma_degs=0.12,
            residual_weight_max=0.10,
        ),
    )
