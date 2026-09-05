from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from benchmarks.teacher_population import TeacherPopulationConfig, build_teacher_population
from evaluation.v4_identity_endpoint import CellIdentityMetadata


@dataclass(frozen=True, slots=True)
class StaticTeacherConfig:
    candidate_index: int
    seed: int
    lag_steps: int
    dt_ms: float
    spatial_family: str
    temporal_family: str
    midget_center_radius: float
    parasol_center_radius: float
    midget_surround_radius: float
    parasol_surround_radius: float
    midget_surround_ratio: float
    parasol_surround_ratio: float
    midget_primary_tau_ms: float
    parasol_primary_tau_ms: float
    midget_secondary_tau_ms: float
    parasol_secondary_tau_ms: float
    midget_secondary_gain: float
    parasol_secondary_gain: float
    center_shift_bound: float
    radius_jitter_fraction: float
    ratio_jitter_fraction: float
    latency_jitter_ms: float


@dataclass(frozen=True, slots=True)
class StaticTeacherCellParameter:
    cell_id: str
    center_shift_x: float
    center_shift_y: float
    center_radius: float
    surround_radius: float
    surround_ratio: float
    peak_latency_shift_ms: float


@dataclass(frozen=True, slots=True)
class StaticTeacherCandidate:
    config: StaticTeacherConfig
    metadata: tuple[CellIdentityMetadata, ...]
    cell_parameters: tuple[StaticTeacherCellParameter, ...]
    rf: torch.Tensor


@dataclass(frozen=True, slots=True)
class TypeKernelParameters:
    center_radius: float
    surround_radius: float
    surround_ratio: float
    primary_tau_ms: float
    secondary_tau_ms: float
    secondary_gain: float


@dataclass(frozen=True, slots=True)
class CellParameterRequest:
    cell_id: str
    type_id: str
    replicate: str
    base: TypeKernelParameters
    config: StaticTeacherConfig


@dataclass(frozen=True, slots=True)
class StaticTeacherError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def build_static_teacher_candidate(
    cone_positions: np.ndarray,
    candidate_index: int,
) -> StaticTeacherCandidate:
    configs = static_teacher_configs()
    if not 0 <= candidate_index < len(configs):
        raise StaticTeacherError("candidate index must be zero or one")
    config = configs[candidate_index]
    population = build_teacher_population(
        cone_positions,
        TeacherPopulationConfig(4),
        generation_seed=19,
    )
    type_parameters = _type_parameters(config)
    parameters_by_key: dict[tuple[str, str], StaticTeacherCellParameter] = {}
    kernels = []
    realized = []
    for cell_id, type_id, polarity, position, replicate in zip(
        population.cells.ids,
        population.cells.type_ids,
        population.cells.polarities,
        population.cells.positions_degs,
        population.replicate_ids,
        strict=True,
    ):
        key = (type_id, replicate)
        parameter = parameters_by_key.get(key)
        if parameter is None:
            parameter = _cell_parameter(
                CellParameterRequest(
                    cell_id,
                    type_id,
                    replicate,
                    type_parameters[type_id],
                    config,
                )
            )
            parameters_by_key[key] = parameter
        spatial = _spatial_kernel(cone_positions, position, parameter)
        temporal = _temporal_kernel(parameter, type_parameters[type_id], config)
        sign = 1.0 if int(polarity) == 0 else -1.0
        kernels.append(sign * temporal[:, None] * spatial[None, :])
        realized.append(replace(parameter, cell_id=cell_id))
    metadata = tuple(
        CellIdentityMetadata(
            cell_id,
            type_id,
            "ON" if int(polarity) == 0 else "OFF",
            float(position[0]),
            float(position[1]),
            replicate,
        )
        for cell_id, type_id, polarity, position, replicate in zip(
            population.cells.ids,
            population.cells.type_ids,
            population.cells.polarities,
            population.cells.positions_degs,
            population.replicate_ids,
            strict=True,
        )
    )
    rf = torch.from_numpy(np.stack(kernels)).unsqueeze(0).unsqueeze(0).float()
    return StaticTeacherCandidate(config, metadata, tuple(realized), rf)


def static_teacher_configs() -> tuple[StaticTeacherConfig, ...]:
    return (_teacher_config(0, 74001), _teacher_config(1, 74002))


def _teacher_config(index: int, seed: int) -> StaticTeacherConfig:
    return StaticTeacherConfig(
        index, seed, 16, 5.0,
        "difference_of_normalized_gaussians",
        "difference_of_gamma_modes",
        0.060, 0.085, 0.140, 0.200, 0.42, 0.52,
        55.0, 20.0, 18.0, 60.0, 0.20, 0.55,
        0.006, 0.06, 0.05, 5.0,
    )


def _type_parameters(
    config: StaticTeacherConfig,
) -> dict[str, TypeKernelParameters]:
    return {
        "midget": TypeKernelParameters(
            config.midget_center_radius,
            config.midget_surround_radius,
            config.midget_surround_ratio,
            config.midget_primary_tau_ms,
            config.midget_secondary_tau_ms,
            config.midget_secondary_gain,
        ),
        "parasol": TypeKernelParameters(
            config.parasol_center_radius,
            config.parasol_surround_radius,
            config.parasol_surround_ratio,
            config.parasol_primary_tau_ms,
            config.parasol_secondary_tau_ms,
            config.parasol_secondary_gain,
        ),
    }


def _cell_parameter(request: CellParameterRequest) -> StaticTeacherCellParameter:
    type_offset = {"midget": 0, "parasol": 100}[request.type_id]
    replicate_index = int(request.replicate.removeprefix("r"))
    rng = np.random.default_rng(
        request.config.seed + type_offset + replicate_index
    )
    radius = request.config.radius_jitter_fraction
    ratio = request.config.ratio_jitter_fraction
    shift = request.config.center_shift_bound
    return StaticTeacherCellParameter(
        request.cell_id,
        float(rng.uniform(-shift, shift)),
        float(rng.uniform(-shift, shift)),
        request.base.center_radius * float(rng.uniform(1 - radius, 1 + radius)),
        request.base.surround_radius * float(rng.uniform(1 - radius, 1 + radius)),
        request.base.surround_ratio * float(rng.uniform(1 - ratio, 1 + ratio)),
        float(
            rng.uniform(
                -request.config.latency_jitter_ms,
                request.config.latency_jitter_ms,
            )
        ),
    )


def _spatial_kernel(
    cone_positions: np.ndarray,
    base_position: np.ndarray,
    parameter: StaticTeacherCellParameter,
) -> np.ndarray:
    center_position = base_position + np.asarray(
        (parameter.center_shift_x, parameter.center_shift_y), dtype=np.float32
    )
    distance = np.linalg.norm(cone_positions - center_position, axis=1)
    center = np.exp(-0.5 * np.square(distance / parameter.center_radius))
    surround = np.exp(-0.5 * np.square(distance / parameter.surround_radius))
    center /= center.sum()
    surround /= surround.sum()
    spatial = center - parameter.surround_ratio * surround
    return spatial / max(np.linalg.norm(spatial), 1e-12)


def _temporal_kernel(
    parameter: StaticTeacherCellParameter,
    base: TypeKernelParameters,
    config: StaticTeacherConfig,
) -> np.ndarray:
    elapsed = np.arange(config.lag_steps - 1, -1, -1) * config.dt_ms
    shifted = np.maximum(elapsed - parameter.peak_latency_shift_ms, 0.0)
    primary = _gamma_mode(shifted, base.primary_tau_ms)
    secondary = _gamma_mode(shifted, base.secondary_tau_ms)
    temporal = primary - base.secondary_gain * secondary
    return temporal / max(np.linalg.norm(temporal), 1e-12)


def _gamma_mode(elapsed_ms: np.ndarray, tau_ms: float) -> np.ndarray:
    scaled = elapsed_ms / tau_ms
    return scaled * np.exp(1.0 - scaled)


__all__ = [
    "StaticTeacherCandidate",
    "StaticTeacherCellParameter",
    "StaticTeacherConfig",
    "build_static_teacher_candidate",
    "static_teacher_configs",
]
