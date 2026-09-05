from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from evaluation.mechanistic_retina.artifacts import FINAL_TEST_BOUNDARY
from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.mechanistic_retina.rf_base import load_candidate0
from evaluation.mechanistic_retina.spike_banks import (
    generate_nested_spike_bank,
    slice_spike_bank,
)
from evaluation.model_comparison.baseline_runs import (
    run_bias,
    run_glm,
    run_graph_tcn,
    run_lnln,
)
from evaluation.model_comparison.mechanistic_run import run_mechanistic
from evaluation.model_comparison.parameters import parameter_inventory
from evaluation.model_comparison.run_data import BankRunData, ProgressCallback
from evaluation.model_comparison.sample_efficiency_profiles import (
    MECHANISTIC_OPTIMIZER_LISTED,
    MECHANISTIC_TOTAL,
    format_parameter_report,
    profile_sets,
)
from evaluation.model_comparison.sample_efficiency_protocol import (
    SampleEfficiencyDataSlice,
    SampleEfficiencyProtocol,
    SampleEfficiencySubset,
    build_sample_efficiency_slices,
    load_sample_efficiency_protocol,
)
from evaluation.model_comparison.sample_efficiency_reporting import MetricRow, Profile
from evaluation.model_comparison.sample_efficiency_rows import (
    ProfiledRunResult,
    canonical_metric_rows,
    metric_rows_from_profiled,
    profiled_run,
)
from evaluation.model_comparison.types import ProgressEvent, RunResult
from evaluation.mechanistic_retina.mechanism_runtime import build_student
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.stages import MechanisticSeedData, build_seed_data


BiasRunner = Callable[[BankRunData], RunResult]
SeedRunner = Callable[[BankRunData, int], RunResult]


@dataclass(frozen=True, slots=True)
class SampleEfficiencyRunSet:
    bias: BiasRunner = run_bias
    glm: BiasRunner = run_glm
    lnln: SeedRunner = run_lnln
    graph_tcn: SeedRunner = run_graph_tcn
    mechanistic: SeedRunner = run_mechanistic


@dataclass(frozen=True, slots=True)
class PreparedSampleEfficiency:
    protocol: SampleEfficiencyProtocol
    data: MechanisticSeedData
    candidate_rf_sha256: str
    slices: tuple[SampleEfficiencyDataSlice, ...]
    match_target: int


@dataclass(frozen=True, slots=True)
class SampleEfficiencyExperimentError(Exception):
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


DEFAULT_RUN_SET = SampleEfficiencyRunSet()


def prepare_sample_efficiency(root: Path, config_path: Path) -> PreparedSampleEfficiency:
    protocol = load_sample_efficiency_protocol(config_path)
    config = protocol.canonical_config
    candidate = load_candidate0(
        root / config.candidate0_path,
        usage=config.candidate_teacher_usage,
        reference_candidate_index=config.candidate_teacher_reference_index,
    )
    if candidate.rf_sha256 != config.candidate0_rf_sha256:
        raise SampleEfficiencyExperimentError(
            "BENCHMARK_IDENTITY_MISMATCH",
            "Candidate0 RF",
        )
    data = build_seed_data(config.data_seed, candidate)
    banks = tuple(
        slice_spike_bank(
            generate_nested_spike_bank(
                data.train_probability[:, 0],
                data.validation_probability[:, 0],
                seed=bank_seed,
                max_trials=64,
            ),
            config.trials,
        )
        for bank_seed in config.bank_seeds
    )
    reference = build_student(data, config.model_seeds[0])
    inventory = parameter_inventory(reference, phase1_parameters(reference))
    return PreparedSampleEfficiency(
        protocol,
        data,
        candidate.rf_sha256,
        build_sample_efficiency_slices(protocol, data, banks),
        inventory.requires_grad,
    )


def validation_contract(root: Path, config_path: Path) -> Mapping[str, JsonValue]:
    prepared = prepare_sample_efficiency(root, config_path)
    sets = profile_sets()
    config = prepared.protocol.canonical_config
    return {
        "candidate0_rf_sha256": prepared.candidate_rf_sha256,
        "candidate0_source_sha256": hashlib.sha256((root / config.candidate0_path).read_bytes()).hexdigest(),
        "sample_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "canonical_config_sha256": hashlib.sha256(prepared.protocol.canonical_config_path.read_bytes()).hexdigest(),
        "data_seed": config.data_seed,
        "trials": config.trials,
        "steps": config.steps,
        "fractions": [subset.fraction for subset in prepared.protocol.subsets],
        "train_counts": [subset.train_count for subset in prepared.protocol.subsets],
        "selection_seed": prepared.protocol.selection_seed,
        "model_seeds": list(config.model_seeds),
        "bank_seeds": list(config.bank_seeds),
        "mechanistic_parameters": {
            "total": sets.mechanistic.total,
            "requires_grad": sets.mechanistic.requires_grad,
            "optimizer_listed": sets.mechanistic.optimizer_listed,
            "nonzero_gradient": sets.mechanistic.nonzero_gradient,
            "actually_updated": sets.mechanistic.actually_updated,
        },
        "parameter_report": format_parameter_report(sets),
        "rf_estimand": "conditional total-dynamic logit RF, 16 lags",
        "fraction_hashes": {
            str(item.fraction): {
                "indices": list(item.indices),
                "train_cone_sha256": item.train_cone_sha256,
                "train_probability_sha256": item.train_probability_sha256,
                "train_mask_sha256": item.train_mask_sha256,
                "validation_cone_sha256": item.validation_cone_sha256,
                "validation_probability_sha256": item.validation_probability_sha256,
                "validation_mask_sha256": item.validation_mask_sha256,
                "banks": [{"seed": bank.seed, "trials": bank.trials, "train_sha256": bank.train_sha256, "validation_sha256": bank.validation_sha256} for bank in item.banks],
            }
            for item in prepared.slices
        },
        "source_sha256": {
            str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in (Path("evaluation/model_comparison/sample_efficiency_experiment.py"), Path("evaluation/model_comparison/sample_efficiency_runner.py"), Path("evaluation/model_comparison/sample_efficiency_rows.py"), Path("evaluation/model_comparison/sample_efficiency_state.py"), Path("scripts/run_sample_efficiency.py"))
        },
        "training_rows_by_fraction": {"0.25": 51, "0.5": 51, "1.0": 18},
        "final_test_boundary": list(FINAL_TEST_BOUNDARY),
    }


def run_fraction(
    root: Path,
    prepared: PreparedSampleEfficiency,
    subset: SampleEfficiencySubset,
    run_set: SampleEfficiencyRunSet,
    progress: ProgressCallback,
) -> tuple[MetricRow, ...]:
    data_slice = next(item for item in prepared.slices if item.fraction == subset.fraction)
    if subset.fraction == 1.0:
        reused = canonical_metric_rows(root)
        active = _run_active_controls(root, prepared, data_slice, run_set, progress)
        return reused + active
    return _run_all_models(root, prepared, data_slice, run_set, progress)


def _run_all_models(
    root: Path,
    prepared: PreparedSampleEfficiency,
    data_slice: SampleEfficiencyDataSlice,
    run_set: SampleEfficiencyRunSet,
    progress: ProgressCallback,
) -> tuple[MetricRow, ...]:
    profiled: list[ProfiledRunResult] = []
    for request in _bank_requests(root, prepared, data_slice, MECHANISTIC_TOTAL, progress):
        profiled.extend(
            (
                profiled_run(run_set.bias(request), Profile.SHARED_REFERENCE),
                profiled_run(run_set.glm(request), Profile.SHARED_REFERENCE),
            )
        )
        for seed in prepared.protocol.canonical_config.model_seeds:
            profiled.extend(
                (
                    profiled_run(run_set.lnln(request, seed), Profile.ARCHITECTURE_SIZE),
                    profiled_run(run_set.graph_tcn(request, seed), Profile.ARCHITECTURE_SIZE),
                    profiled_run(run_set.mechanistic(request, seed), Profile.ARCHITECTURE_SIZE),
                )
            )
    profiled.extend(_active_profiled_runs(root, prepared, data_slice, run_set, progress))
    return metric_rows_from_profiled(data_slice.fraction, data_slice.train_count, tuple(profiled))


def _run_active_controls(
    root: Path,
    prepared: PreparedSampleEfficiency,
    data_slice: SampleEfficiencyDataSlice,
    run_set: SampleEfficiencyRunSet,
    progress: ProgressCallback,
) -> tuple[MetricRow, ...]:
    profiled = _active_profiled_runs(root, prepared, data_slice, run_set, progress)
    return metric_rows_from_profiled(data_slice.fraction, data_slice.train_count, profiled)


def _active_profiled_runs(
    root: Path,
    prepared: PreparedSampleEfficiency,
    data_slice: SampleEfficiencyDataSlice,
    run_set: SampleEfficiencyRunSet,
    progress: ProgressCallback,
) -> tuple[ProfiledRunResult, ...]:
    rows: list[ProfiledRunResult] = []
    for request in _bank_requests(
        root,
        prepared,
        data_slice,
        MECHANISTIC_OPTIMIZER_LISTED,
        progress,
    ):
        for seed in prepared.protocol.canonical_config.model_seeds:
            rows.append(profiled_run(run_set.lnln(request, seed), Profile.ACTIVE_DOF))
            rows.append(profiled_run(run_set.graph_tcn(request, seed), Profile.ACTIVE_DOF))
    return tuple(rows)


def _bank_requests(
    root: Path,
    prepared: PreparedSampleEfficiency,
    data_slice: SampleEfficiencyDataSlice,
    match_target: int,
    progress: ProgressCallback,
) -> tuple[BankRunData, ...]:
    config = replace(
        prepared.protocol.canonical_config,
        output_dir=prepared.protocol.output_dir,
        run_dir=prepared.protocol.run_dir / f"fraction-{int(data_slice.fraction * 100):03d}",
    )
    sliced_data = replace(
        prepared.data,
        train_cones=data_slice.train_cones,
        train_probability=data_slice.train_probability,
        train_mask=data_slice.train_mask,
    )
    return tuple(
        BankRunData(
            root,
            config,
            sliced_data,
            load_candidate0(
                root / config.candidate0_path,
                usage=config.candidate_teacher_usage,
                reference_candidate_index=config.candidate_teacher_reference_index,
            ),
            bank.seed,
            match_target,
            bank.train_spikes,
            bank.validation_spikes,
            data_slice.train_mask[:, 0, None].expand_as(bank.train_spikes).clone(),
            data_slice.validation_mask[:, 0, None].expand_as(bank.validation_spikes).clone(),
            progress,
        )
        for bank in data_slice.banks
    )

__all__ = [
    "DEFAULT_RUN_SET",
    "PreparedSampleEfficiency",
    "SampleEfficiencyRunSet",
    "SampleEfficiencyExperimentError",
    "prepare_sample_efficiency",
    "run_fraction",
    "validation_contract",
]
