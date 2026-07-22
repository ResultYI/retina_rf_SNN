from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from models.cells.amacrine import LocalAmacrineConfig
from models.cells.bipolar import BipolarConfig
from models.cells.horizontal import H1HorizontalConfig

_TIME_AXIS_CV_MAX: Final[float] = 1e-3


class PhysiologyProfileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PhysiologyProfile:
    name: str
    species_priority: tuple[str, ...]
    cone_spacing_deg: float
    eccentricity_deg: float
    h1: H1HorizontalConfig
    bipolar: BipolarConfig
    amacrine: LocalAmacrineConfig


def dt_ms_from_time_axis_seconds(time_axis_seconds: np.ndarray) -> float:
    axis = np.asarray(time_axis_seconds, dtype=np.float64).reshape(-1)
    if axis.size < 2 or not np.isfinite(axis).all():
        raise PhysiologyProfileError("time_axis_seconds needs finite frames")
    intervals = np.diff(axis)
    if np.any(intervals <= 0):
        raise PhysiologyProfileError("time_axis_seconds must be strictly increasing")
    interval_median = float(np.median(intervals))
    interval_cv = float(intervals.std() / (interval_median + 1e-12))
    if interval_cv > _TIME_AXIS_CV_MAX:
        raise PhysiologyProfileError("time_axis_seconds has unstable frame interval")
    return interval_median * 1000.0


def human_macaque(
    *,
    dt_ms: float,
    cone_spacing_deg: float,
    eccentricity_deg: float,
    debug_checks: bool = True,
) -> PhysiologyProfile:
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise PhysiologyProfileError("dt_ms must be positive and finite")
    if not math.isfinite(cone_spacing_deg) or cone_spacing_deg <= 0:
        raise PhysiologyProfileError("cone_spacing_deg must be positive and finite")
    if not math.isfinite(eccentricity_deg) or eccentricity_deg < 0:
        raise PhysiologyProfileError("eccentricity_deg must be finite and non-negative")
    return PhysiologyProfile(
        name="human_macaque",
        species_priority=("human", "macaque", "marmoset"),
        cone_spacing_deg=cone_spacing_deg,
        eccentricity_deg=eccentricity_deg,
        h1=H1HorizontalConfig(
            radius_degs=1.75 * cone_spacing_deg,
            sigma_degs=0.90 * cone_spacing_deg,
            feedback_radius_degs=1.75 * cone_spacing_deg,
            feedback_sigma_degs=0.90 * cone_spacing_deg,
            h1_spacing_degs=1.45 * cone_spacing_deg,
            dt_ms=dt_ms,
            initial_tau_ms=50.0,
            tau_min_ms=10.0,
            tau_max_ms=200.0,
            initial_gain=0.01,
            gain_max=0.20,
            debug_checks=debug_checks,
        ),
        bipolar=BipolarConfig(
            dt_ms=dt_ms,
            initial_tau_sustained_ms=80.0,
            tau_sustained_min_ms=20.0,
            tau_sustained_max_ms=200.0,
            initial_tau_transient_ms=20.0,
            tau_transient_min_ms=5.0,
            tau_transient_max_ms=120.0,
            initial_g_ab_sustained=0.01,
            g_ab_sustained_max=0.10,
            initial_g_ab_transient=0.01,
            g_ab_transient_max=0.30,
            initial_polarity_gain_on=1.0,
            initial_polarity_gain_off=1.0,
            polarity_gain_min=0.25,
            polarity_gain_max=4.0,
            initial_polarity_threshold_on=0.0,
            initial_polarity_threshold_off=0.0,
            polarity_threshold_min=-1.0,
            polarity_threshold_max=1.0,
            initial_rectifier_softness=0.05,
            rectifier_softness_min=0.01,
            rectifier_softness_max=0.50,
            debug_checks=debug_checks,
        ),
        amacrine=LocalAmacrineConfig(
            radius_degs=3.60 * cone_spacing_deg,
            sigma_degs=1.80 * cone_spacing_deg,
            dt_ms=dt_ms,
            initial_tau_sustained_ms=100.0,
            tau_sustained_min_ms=20.0,
            tau_sustained_max_ms=250.0,
            initial_tau_transient_ms=40.0,
            tau_transient_min_ms=15.0,
            tau_transient_max_ms=180.0,
            initial_g_ba_sustained=0.03,
            g_ba_sustained_max=0.30,
            initial_g_ba_transient=0.05,
            g_ba_transient_max=0.50,
            debug_checks=debug_checks,
        ),
    )


__all__ = [
    "PhysiologyProfile",
    "PhysiologyProfileError",
    "dt_ms_from_time_axis_seconds",
    "human_macaque",
]
