from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch


HUMRET_MICRONS_PER_DEGREE: Final = 266.0
HUMRET_SPATIAL_PERIODS_UM: Final = (
    100.0,
    200.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
)
HUMRET_TEMPORAL_FREQUENCIES_HZ: Final = (1.0, 2.0, 4.0, 8.0)
HUMRET_FREQUENCY_CHIRP_DURATION_S: Final = 8.0
HUMRET_CONTRAST_CHIRP_DURATION_S: Final = 8.0
HUMRET_GRATING_DURATION_S: Final = 12.0
HUMRET_FLASH_PHASE_DURATION_S: Final = 2.0


@dataclass(frozen=True, slots=True)
class HumRetDataError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class HumRetGratingCondition:
    spatial_period_um: float
    temporal_frequency_hz: float

    def __post_init__(self) -> None:
        values = (self.spatial_period_um, self.temporal_frequency_hz)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise HumRetDataError("Grating condition values must be positive")

    @property
    def spatial_frequency_cpd(self) -> float:
        return HUMRET_MICRONS_PER_DEGREE / self.spatial_period_um


@dataclass(frozen=True, slots=True)
class HumRetReference:
    grating_f1_normalized: torch.Tensor
    grating_cell_ids: torch.Tensor
    chirp_frequency_hz: torch.Tensor
    chirp_modulation_normalized: torch.Tensor


@dataclass(frozen=True, slots=True)
class HumRetPopulationAgreement:
    mean_tuning_cosine_similarity: float
    spatial_preference_total_variation: float
    temporal_preference_total_variation: float


def humret_grating_conditions() -> tuple[HumRetGratingCondition, ...]:
    return tuple(
        HumRetGratingCondition(period, frequency)
        for period in HUMRET_SPATIAL_PERIODS_UM
        for frequency in HUMRET_TEMPORAL_FREQUENCIES_HZ
    )


def build_humret_frequency_chirp(
    cone_count: int,
    dt_ms: float,
) -> torch.Tensor:
    time_s = _time_axis(HUMRET_FREQUENCY_CHIRP_DURATION_S, dt_ms)
    waveform = torch.sin(torch.pi * (time_s.square() + time_s / 10.0))
    return _full_field(waveform, cone_count)


def build_humret_contrast_chirp(
    cone_count: int,
    dt_ms: float,
) -> torch.Tensor:
    time_s = _time_axis(HUMRET_CONTRAST_CHIRP_DURATION_S, dt_ms)
    contrast = time_s / HUMRET_CONTRAST_CHIRP_DURATION_S
    waveform = contrast * torch.sin(4.0 * torch.pi * time_s)
    return _full_field(waveform, cone_count)


def build_humret_flash_steps(cone_count: int, dt_ms: float) -> torch.Tensor:
    phase_steps = _step_count(HUMRET_FLASH_PHASE_DURATION_S, dt_ms)
    levels = torch.tensor((0.0, -1.0, 0.0, 1.0, 0.0))
    waveform = torch.repeat_interleave(levels, phase_steps)
    return _full_field(waveform, cone_count)


def build_humret_drifting_grating(
    positions_degs: torch.Tensor,
    dt_ms: float,
    condition: HumRetGratingCondition,
    direction_degrees: float = 0.0,
) -> torch.Tensor:
    positions = torch.as_tensor(positions_degs, dtype=torch.float32)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 1:
        raise HumRetDataError("positions_degs must have shape [N,2]")
    if not torch.isfinite(positions).all() or not math.isfinite(direction_degrees):
        raise HumRetDataError("Grating geometry must be finite")
    angle = math.radians(direction_degrees)
    direction = positions.new_tensor((math.cos(angle), math.sin(angle)))
    spatial_period_degs = condition.spatial_period_um / HUMRET_MICRONS_PER_DEGREE
    spatial_phase = 2.0 * torch.pi * (positions @ direction) / spatial_period_degs
    time_s = _time_axis(HUMRET_GRATING_DURATION_S, dt_ms).to(positions.device)
    temporal_phase = 2.0 * torch.pi * condition.temporal_frequency_hz * time_s
    return torch.cos(temporal_phase[:, None] + spatial_phase[None, :])


def parse_humret_reference(
    norm_peaks: np.ndarray,
    grating_cell_ids: np.ndarray,
    chirp_mod: np.ndarray,
) -> HumRetReference:
    peaks = np.asarray(norm_peaks)
    expected_conditions = len(humret_grating_conditions())
    if peaks.ndim != 3 or peaks.shape[1] != expected_conditions or peaks.shape[2] < 2:
        raise HumRetDataError("normPeaks must have shape [cell,24,harmonic]")
    f1 = np.asarray(peaks[:, :, 1], dtype=np.float32)
    if not np.isfinite(f1).all() or np.any(f1 < 0):
        raise HumRetDataError(
            "HumRet grating F1 responses must be finite and non-negative"
        )
    ids = np.asarray(grating_cell_ids).reshape(-1)
    if ids.shape[0] != f1.shape[0] or not np.isfinite(ids).all():
        raise HumRetDataError("HumRet grating cell IDs do not match normPeaks")
    rounded_ids = np.rint(ids)
    if not np.allclose(ids, rounded_ids):
        raise HumRetDataError("HumRet grating cell IDs must be integers")

    frequency_hz, modulation = _parse_chirp_mod(chirp_mod)
    return HumRetReference(
        grating_f1_normalized=torch.from_numpy(
            f1.reshape(f1.shape[0], len(HUMRET_SPATIAL_PERIODS_UM), -1)
        ),
        grating_cell_ids=torch.from_numpy(rounded_ids.astype(np.int64)),
        chirp_frequency_hz=torch.from_numpy(frequency_hz),
        chirp_modulation_normalized=torch.from_numpy(modulation),
    )


def load_humret_reference(root: str | Path) -> HumRetReference:
    data_root = Path(root)
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise HumRetDataError("SciPy is required to read HumRet MAT files") from error
    try:
        grating = loadmat(data_root / "h_normPeaks.mat", squeeze_me=True)
        chirp = loadmat(data_root / "h_chirp.mat", squeeze_me=True)
        return parse_humret_reference(
            grating["normPeaks"],
            grating["togo"],
            chirp["chirpMod"],
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        detail = f"Cannot parse HumRet data in {data_root}: {error}"
        raise HumRetDataError(detail) from error


def compare_humret_grating_population(
    model_tuning: torch.Tensor,
    reference: HumRetReference,
) -> HumRetPopulationAgreement:
    model = _normalized_tuning(model_tuning, "model_tuning")
    human = _normalized_tuning(
        reference.grating_f1_normalized,
        "reference.grating_f1_normalized",
    ).to(model.device)
    model_mean = model.mean(dim=0).flatten()
    human_mean = human.mean(dim=0).flatten()
    cosine = torch.nn.functional.cosine_similarity(model_mean, human_mean, dim=0)
    model_spatial, model_temporal = _preference_histograms(model)
    human_spatial, human_temporal = _preference_histograms(human)
    return HumRetPopulationAgreement(
        mean_tuning_cosine_similarity=cosine.detach().item(),
        spatial_preference_total_variation=(
            0.5 * torch.abs(model_spatial - human_spatial).sum()
        ).detach().item(),
        temporal_preference_total_variation=(
            0.5 * torch.abs(model_temporal - human_temporal).sum()
        ).detach().item(),
    )


def smoothed_spike_probability_to_hz(
    rate_per_bin: torch.Tensor,
    dt_ms: float,
) -> torch.Tensor:
    rate = torch.as_tensor(rate_per_bin)
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise HumRetDataError("dt_ms must be positive and finite")
    if not torch.isfinite(rate).all() or torch.any((rate < 0) | (rate > 1)):
        raise HumRetDataError("Smoothed spike probability must lie in [0,1]")
    return rate * (1000.0 / dt_ms)


def _time_axis(duration_s: float, dt_ms: float) -> torch.Tensor:
    steps = _step_count(duration_s, dt_ms)
    return torch.arange(steps, dtype=torch.float32) * (dt_ms / 1000.0)


def _step_count(duration_s: float, dt_ms: float) -> int:
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise HumRetDataError("dt_ms must be positive and finite")
    exact_steps = duration_s * 1000.0 / dt_ms
    steps = round(exact_steps)
    if steps < 1 or not math.isclose(exact_steps, steps, abs_tol=1e-6):
        raise HumRetDataError("dt_ms must exactly tile the HumRet stimulus duration")
    return steps


def _full_field(waveform: torch.Tensor, cone_count: int) -> torch.Tensor:
    if cone_count < 1:
        raise HumRetDataError("HumRet probes need at least one cone")
    return waveform[:, None].expand(-1, cone_count).clone()


def _parse_chirp_mod(chirp_mod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(chirp_mod, dtype=object)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise HumRetDataError("chirpMod must have shape [cell,2]")
    frequencies: list[np.ndarray] = []
    modulations: list[np.ndarray] = []
    for modulation_raw, frequency_raw in cells:
        modulation = _object_vector(modulation_raw)
        frequency = _object_vector(frequency_raw)
        if modulation.size == 0 and frequency.size == 0:
            continue
        if modulation.size < 2 or modulation.shape != frequency.shape:
            raise HumRetDataError("HumRet chirp modulation/frequency vectors mismatch")
        if not np.isfinite(modulation).all() or not np.isfinite(frequency).all():
            raise HumRetDataError("HumRet chirp vectors must be finite")
        scale = float(np.max(np.abs(modulation)))
        if scale <= 0:
            raise HumRetDataError("HumRet chirp modulation cannot be all zero")
        frequencies.append(frequency.astype(np.float32))
        modulations.append((modulation / scale).astype(np.float32))
    if not frequencies:
        raise HumRetDataError("HumRet chirpMod contains no paired responses")
    frequency_hz = frequencies[0]
    if any(
        item.shape != frequency_hz.shape or not np.allclose(item, frequency_hz)
        for item in frequencies[1:]
    ):
        raise HumRetDataError("HumRet chirp frequency grids are inconsistent")
    return frequency_hz, np.stack(modulations)


def _object_vector(value: object) -> np.ndarray:
    array = np.asarray(value)
    while array.dtype == object and array.size == 1:
        array = np.asarray(array.item())
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _normalized_tuning(tuning: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(tuning, dtype=torch.float32)
    expected_shape = (
        len(HUMRET_SPATIAL_PERIODS_UM),
        len(HUMRET_TEMPORAL_FREQUENCIES_HZ),
    )
    if value.ndim != 3 or tuple(value.shape[1:]) != expected_shape:
        raise HumRetDataError(f"{name} must have shape [cell,6,4]")
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise HumRetDataError(f"{name} must be finite and non-negative")
    maximum = value.flatten(1).amax(dim=1)
    if torch.any(maximum <= 0):
        raise HumRetDataError(f"{name} contains a cell without grating response")
    return value / maximum[:, None, None]


def _preference_histograms(
    tuning: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    preferred = tuning.flatten(1).argmax(dim=1)
    temporal_count = len(HUMRET_TEMPORAL_FREQUENCIES_HZ)
    spatial = torch.bincount(
        preferred // temporal_count,
        minlength=len(HUMRET_SPATIAL_PERIODS_UM),
    ).float()
    temporal = torch.bincount(
        preferred % temporal_count,
        minlength=temporal_count,
    ).float()
    return spatial / spatial.sum(), temporal / temporal.sum()
