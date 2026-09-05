from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from evaluation.mechanistic_retina.clean_sampled_teacher import (
    configure_clean_teacher,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    PathwayClamp,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)


@dataclass(frozen=True, slots=True)
class CleanBenchmarkConfig:
    train_stimuli: int = 32
    validation_stimuli: int = 12
    time_steps: int = 64
    trials: int = 4
    steps: int = 400
    checkpoint_steps: tuple[int, ...] = (0, 50, 100, 200, 400)
    learning_rate: float = 0.03
    batch_size: int = 8
    stimulus_seed: int = 51_001
    teacher_seed: int = 52_001
    student_seed: int = 53_001
    spike_seed: int = 54_001
    training_seed: int = 55_001

    def __post_init__(self) -> None:
        positive = (
            self.train_stimuli,
            self.validation_stimuli,
            self.time_steps,
            self.trials,
            self.steps,
            self.batch_size,
        )
        if any(value < 1 for value in positive):
            raise CleanBenchmarkError("clean benchmark sizes must be positive")
        if self.time_steps < 16:
            raise CleanBenchmarkError("time_steps must cover the fixed 16-lag model RF")
        if not self.checkpoint_steps or self.checkpoint_steps[0] != 0:
            raise CleanBenchmarkError("checkpoint schedule must start at zero")
        if self.checkpoint_steps[-1] != self.steps:
            raise CleanBenchmarkError("checkpoint schedule must end at the final step")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise CleanBenchmarkError(
                "checkpoint schedule must be unique and increasing"
            )


@dataclass(frozen=True, slots=True)
class CleanBenchmarkError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class CleanBenchmarkState:
    config: CleanBenchmarkConfig
    teacher: MechanisticGraphTemporalRetina
    student: MechanisticGraphTemporalRetina
    cone_positions: torch.Tensor
    cell_positions: torch.Tensor
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]
    train_cones: torch.Tensor
    validation_cones: torch.Tensor
    train_spikes: torch.Tensor
    validation_spikes: torch.Tensor
    teacher_effects: dict[str, float]


def build_clean_state(
    config: CleanBenchmarkConfig,
    *,
    verify_teacher_pathways: bool = True,
) -> CleanBenchmarkState:
    cone_positions, cell_positions, cell_types, polarities = _geometry()
    model_config = MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
        cell_specific_gains=True,
    )
    teacher = _build_model(
        model_config,
        cone_positions,
        cell_positions,
        cell_types,
        polarities,
        config.teacher_seed,
    )
    configure_clean_teacher(teacher)
    student = _build_model(
        model_config,
        cone_positions,
        cell_positions,
        cell_types,
        polarities,
        config.student_seed,
    )
    train_cones, validation_cones = _cone_drives(config, cone_positions.shape[0])
    train_spikes = _sample_spikes(
        teacher, train_cones, config.trials, config.spike_seed
    )
    validation_spikes = _sample_spikes(
        teacher, validation_cones, config.trials, config.spike_seed + 1_000_003
    )
    effects = (
        _teacher_effects(teacher, validation_cones[:2])
        if verify_teacher_pathways
        else {}
    )
    if any(
        value <= 0 or not torch.isfinite(torch.tensor(value))
        for value in effects.values()
    ):
        raise CleanBenchmarkError(
            "teacher H1/BC/AC pathways must have finite nonzero effects"
        )
    return CleanBenchmarkState(
        config,
        teacher,
        student,
        cone_positions,
        cell_positions,
        cell_types,
        polarities,
        train_cones,
        validation_cones,
        train_spikes,
        validation_spikes,
        effects,
    )


def nested_budget_state(
    master: CleanBenchmarkState, repeat_budget: int
) -> CleanBenchmarkState:
    if repeat_budget < 1 or repeat_budget > master.config.trials:
        raise CleanBenchmarkError(
            "repeat budget must be within the master sampled-spike bank"
        )
    config = replace(master.config, trials=repeat_budget)
    student = _build_model(
        master.student.config,
        master.cone_positions,
        master.cell_positions,
        master.cell_types,
        master.polarities,
        config.student_seed,
    )
    return CleanBenchmarkState(
        config,
        master.teacher,
        student,
        master.cone_positions,
        master.cell_positions,
        master.cell_types,
        master.polarities,
        master.train_cones,
        master.validation_cones,
        master.train_spikes[:, :repeat_budget].clone(),
        master.validation_spikes[:, :repeat_budget].clone(),
        master.teacher_effects,
    )


def _build_model(
    config: MechanisticRetinaConfig,
    cone_positions: torch.Tensor,
    cell_positions: torch.Tensor,
    cell_types: tuple[str, ...],
    polarities: tuple[str, ...],
    seed: int,
) -> MechanisticGraphTemporalRetina:
    torch.manual_seed(seed)
    return build_mechanistic_retina(
        config, cone_positions, cell_positions, cell_types, polarities
    )


def _geometry() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    coordinates = torch.linspace(-0.075, 0.075, 4)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    cones = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    anchors = torch.tensor([[-0.04, -0.04], [0.04, -0.04], [-0.04, 0.04], [0.04, 0.04]])
    cells = torch.cat((anchors - 0.008, anchors + 0.008), dim=0)
    types = ("midget", "midget", "parasol", "parasol") * 2
    polarities = ("ON", "OFF", "ON", "OFF") * 2
    return cones.float(), cells.float(), types, polarities


def _cone_drives(
    config: CleanBenchmarkConfig, cone_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(config.stimulus_seed)
    total = config.train_stimuli + config.validation_stimuli
    innovations = torch.randn(total, config.time_steps, cone_count, generator=generator)
    common = torch.randn(total, config.time_steps, 1, generator=generator)
    values = innovations.mul(0.75).add(common.mul(0.25))
    state = torch.zeros(total, cone_count)
    drives = []
    for time in range(config.time_steps):
        state = 0.82 * state + 0.35 * values[:, time]
        drives.append(state)
    cones = torch.stack(drives, dim=1)
    return cones[: config.train_stimuli], cones[config.train_stimuli :]


def _sample_spikes(
    teacher: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    trials: int,
    seed: int,
) -> torch.Tensor:
    stimuli, time_steps, _ = cones.shape
    cells = teacher.rgc.response_bias.numel()
    flat_cones = cones[:, None].expand(-1, trials, -1, -1).flatten(0, 1)
    observed = torch.zeros(stimuli * trials, time_steps, cells)
    with torch.no_grad():
        base_logits = teacher.forward_sequence(
            flat_cones, observed_counts=observed
        ).logits
        history_state = torch.zeros(stimuli * trials, cells)
        previous = torch.zeros_like(history_state)
        generator = torch.Generator().manual_seed(seed)
        for time in range(time_steps):
            history_state = (
                teacher.rgc.history_decay * history_state
                + (1 - teacher.rgc.history_decay) * previous
            )
            logits = base_logits[:, time] - (
                teacher.gates.history * teacher.rgc.history_gain * history_state
            )
            previous = (
                torch.rand(logits.shape, generator=generator) < torch.sigmoid(logits)
            ).float()
            observed[:, time] = previous
    return observed.reshape(stimuli, trials, time_steps, cells)


def _teacher_effects(
    teacher: MechanisticGraphTemporalRetina, cones: torch.Tensor
) -> dict[str, float]:
    history = torch.zeros(cones.shape[0], cones.shape[1], 8)
    with torch.no_grad():
        full = teacher.forward_sequence(cones, observed_counts=history).logits
        clamped = {
            "H1": frozenset({PathwayClamp.H1}),
            "direct_BC": frozenset(
                {PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}
            ),
            "AC": frozenset(
                {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
            ),
        }
        return {
            name: float(
                (
                    full
                    - teacher.forward_sequence(
                        cones, observed_counts=history, clamps=clamps
                    ).logits
                )
                .abs()
                .mean()
            )
            for name, clamps in clamped.items()
        }


__all__ = [
    "CleanBenchmarkConfig",
    "CleanBenchmarkError",
    "CleanBenchmarkState",
    "build_clean_state",
    "nested_budget_state",
]
