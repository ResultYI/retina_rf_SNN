from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Final

import numpy as np
import torch

from benchmarks.point_process_teacher import (
    SyntheticTeacherResult,
    TeacherPopulationConfig,
    generate_teacher_responses,
)
from data.input_identity import InputIdentity
from data.rgc_response import RGCResponseSession, load_rgc_response
from data.synthetic_teacher import (
    TeacherInputNormalization,
    fit_teacher_input_normalization,
)
from evaluation.phase1_benchmark import (
    BudgetCondition,
    NestedStimulusBank,
    generate_nested_stimulus_bank,
)
from training.response_config import ResponseExperimentConfig, load_response_config
from training.response_data import PreparedResponseData, ResponseSplit, prepare_response_data


ROOT: Final = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT: Final = ROOT / ".omo/evidence/multi-seed-short-validation/benchmark"
RUNS_ROOT: Final = ROOT / ".omo/evidence/canonical-v2-stage05-checkpoint-rerun/runs"
FINAL_TEST_BOUNDARY: Final = (
    "TEST_SPLIT_ACCESSED_FOR_IDENTITY_ONLY",
    "TEST_EXAMPLES_NOT_USED_FOR_INFERENCE_METRICS_MODEL_SELECTION_OR_CONCLUSIONS",
    "FINAL_TEST_SCIENTIFIC_EVALUATION_NOT_CONSUMED",
)


class Phase1DataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AssetRequest:
    surrogate_seed: int
    total_source_count: int
    teacher_seed: int
    validation_seed: int
    validation_trials: int


@dataclass(frozen=True, slots=True)
class Phase1Assets:
    response_config: ResponseExperimentConfig
    canonical_data: PreparedResponseData
    master_bank: NestedStimulusBank
    normalization: TeacherInputNormalization
    train_identity: InputIdentity
    validation_result: SyntheticTeacherResult
    validation_split: ResponseSplit
    teacher_grid: torch.Tensor
    test_path: Path


@dataclass(frozen=True, slots=True)
class ConditionData:
    data: PreparedResponseData
    teacher_probabilities: torch.Tensor
    total_spikes_per_cell: tuple[int, ...]
    train_teacher_probabilities: torch.Tensor


def prepare_phase1_assets(request: AssetRequest) -> Phase1Assets:
    response_config = load_response_config(RUNS_ROOT / "seed-19-type_blind/config.yaml")
    canonical_data = prepare_response_data(response_config.data)
    train_session = load_rgc_response(BENCHMARK_ROOT / "train.h5")
    validation_session = load_rgc_response(BENCHMARK_ROOT / "validation.h5")
    train_cones, train_sources = _unique_context_sources(train_session)
    validation_cones, validation_sources = _unique_context_sources(validation_session)
    _validate_geometry(train_session, validation_session)
    bank = generate_nested_stimulus_bank(
        train_cones,
        train_sources,
        request.total_source_count,
        seed=request.surrogate_seed,
    )
    normalization = fit_teacher_input_normalization(bank.cone_sequences)
    validation_identity = validation_session.input_identity.with_sources(
        tuple(_sha256_array(sequence) for sequence in validation_cones),
        generator_name="phase1-fixed-validation",
        generator_revision="1",
    )
    validation_result = generate_teacher_responses(
        validation_cones,
        validation_session.cone_positions_degs,
        validation_sources,
        validation_session.time_axis_seconds,
        trials=request.validation_trials,
        seed=request.validation_seed,
        adaptive=True,
        teacher_normalization=normalization,
        input_identity=validation_identity,
        population_config=TeacherPopulationConfig(4),
        teacher_seed=request.teacher_seed,
    )
    if validation_result.session.cells.ids != canonical_data.cells.ids:
        raise Phase1DataError("phase-one teacher cells differ from canonical cells")
    teacher_grid = torch.stack(
        (
            torch.from_numpy(validation_result.kernels["context_kernel_low"]),
            torch.from_numpy(validation_result.kernels["context_kernel_high"]),
        ),
        dim=0,
    ).unsqueeze(0).expand(len(validation_sources), -1, -1, -1, -1).clone()
    return Phase1Assets(
        response_config,
        canonical_data,
        bank,
        normalization,
        train_session.input_identity,
        validation_result,
        _response_split(validation_result, normalization),
        teacher_grid.float(),
        BENCHMARK_ROOT / "test.h5",
    )


def build_condition_data(
    assets: Phase1Assets,
    condition: BudgetCondition,
    *,
    bank_seed: int,
    teacher_seed: int,
) -> ConditionData:
    source_count = condition.source_count
    source_hashes = assets.master_bank.source_sha256[:source_count]
    identity = assets.train_identity.with_sources(
        source_hashes,
        generator_name=assets.master_bank.generator,
        generator_revision="1",
    )
    generated = generate_teacher_responses(
        assets.master_bank.cone_sequences[:source_count],
        assets.canonical_data.cone_positions_degs,
        assets.master_bank.source_ids[:source_count],
        assets.canonical_data.time_axis_seconds,
        trials=condition.trials_per_stimulus,
        seed=bank_seed,
        adaptive=True,
        teacher_normalization=assets.normalization,
        input_identity=identity,
        population_config=TeacherPopulationConfig(4),
        teacher_seed=teacher_seed,
    )
    train = _response_split(generated, assets.normalization)
    fingerprint = hashlib.sha256(
        f"{condition.label}:{bank_seed}:{':'.join(source_hashes)}".encode()
    ).hexdigest()
    data = PreparedResponseData(
        train,
        assets.validation_split,
        assets.canonical_data.test,
        assets.canonical_data.cells,
        assets.canonical_data.cone_positions_degs,
        assets.canonical_data.time_axis_seconds,
        assets.canonical_data.target_kind,
        assets.normalization.input_mean,
        assets.normalization.input_std,
        fingerprint,
        identity,
    )
    spikes = train.spike_counts.sum(dim=(0, 1, 2))
    probabilities = torch.from_numpy(
        assets.validation_result.conditional_probabilities.astype(np.float32)
    )
    train_probabilities = torch.from_numpy(
        generated.conditional_probabilities.astype(np.float32)
    )
    return ConditionData(
        data,
        probabilities,
        tuple(int(value) for value in spikes),
        train_probabilities,
    )


def _response_split(
    result: SyntheticTeacherResult,
    normalization: TeacherInputNormalization,
) -> ResponseSplit:
    session = result.session
    cones = (session.cone_response - normalization.input_mean) / normalization.input_std
    return ResponseSplit(
        torch.from_numpy(cones.astype(np.float32)),
        torch.from_numpy(session.spike_counts.astype(np.float32)),
        torch.from_numpy(session.valid_mask),
        session.source_ids,
        session.context_ids,
    )


def _unique_context_sources(
    session: RGCResponseSession,
) -> tuple[np.ndarray, tuple[str, ...]]:
    return recover_base_context_sources(
        session.cone_response,
        session.source_ids,
        session.context_ids,
    )


def recover_base_context_sources(
    cone_response: np.ndarray,
    source_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if cone_response.shape[0] % 2:
        raise Phase1DataError("canonical stimuli must form low/high pairs")
    paired_sources = tuple(zip(source_ids[0::2], source_ids[1::2], strict=True))
    paired_contexts = tuple(zip(context_ids[0::2], context_ids[1::2], strict=True))
    if any(low != high for low, high in paired_sources):
        raise Phase1DataError("canonical context pairs must share source ids")
    if any(pair != ("low", "high") for pair in paired_contexts):
        raise Phase1DataError("canonical context pairs must be ordered low then high")
    low = cone_response[0::2].copy()
    high = cone_response[1::2].copy()
    context_steps = max(1, cone_response.shape[1] - min(64, cone_response.shape[1] // 2))
    low[:, :context_steps] /= 0.5
    high[:, :context_steps] /= 1.5
    if not np.allclose(low, high, rtol=2e-6, atol=2e-5):
        raise Phase1DataError("canonical contexts do not invert to shared base stimuli")
    return ((low + high) * 0.5).astype(np.float32), source_ids[0::2]


def _validate_geometry(
    train: RGCResponseSession,
    validation: RGCResponseSession,
) -> None:
    if not np.array_equal(train.cone_positions_degs, validation.cone_positions_degs):
        raise Phase1DataError("train and validation cone geometry differs")
    if not np.array_equal(train.time_axis_seconds, validation.time_axis_seconds):
        raise Phase1DataError("train and validation time axes differ")


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


__all__ = [
    "AssetRequest",
    "BENCHMARK_ROOT",
    "ConditionData",
    "FINAL_TEST_BOUNDARY",
    "Phase1Assets",
    "Phase1DataError",
    "RUNS_ROOT",
    "build_condition_data",
    "prepare_phase1_assets",
    "recover_base_context_sources",
]
