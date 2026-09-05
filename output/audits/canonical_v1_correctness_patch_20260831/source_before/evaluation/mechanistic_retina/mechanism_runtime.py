from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    PathwayClamp,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)
from evaluation.mechanistic_retina.rf_base import CandidateTeacherUsage
from training.mechanistic_retina.stages import MechanisticSeedData


@dataclass(frozen=True, slots=True)
class MechanismRunConfig:
    candidate0_path: Path
    candidate_teacher_usage: CandidateTeacherUsage
    candidate_teacher_reference_index: int | None
    output_dir: Path
    seeds: tuple[int, ...]
    bank_seed: int
    steps: int
    checkpoints: tuple[int, ...]
    learning_rate: float
    batch_size: int
    monitor_interval_seconds: float
    stall_intervals: int


@dataclass(frozen=True, slots=True)
class MechanismConfigError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class ClampedRFRequest:
    model: MechanisticGraphTemporalRetina
    cones: torch.Tensor
    observed_counts: torch.Tensor
    clamps: frozenset[PathwayClamp]


def load_mechanism_config(path: Path) -> MechanismRunConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference_index = payload.get("candidate_teacher_reference_index")
    if reference_index is not None and (
        not isinstance(reference_index, int)
        or isinstance(reference_index, bool)
        or reference_index < 0
    ):
        raise MechanismConfigError("Candidate reference index is invalid")
    config = MechanismRunConfig(
        Path(payload["candidate0_path"]),
        CandidateTeacherUsage(payload["candidate_teacher_usage"]),
        reference_index,
        Path(payload["output_dir"]),
        tuple(int(value) for value in payload["seeds"]),
        int(payload["bank_seed"]),
        int(payload["steps"]),
        tuple(int(value) for value in payload["checkpoints"]),
        float(payload["learning_rate"]),
        int(payload["batch_size"]),
        float(payload["monitor_interval_seconds"]),
        int(payload["stall_intervals"]),
    )
    if config.seeds != (19, 20, 21):
        raise MechanismConfigError("mechanism seeds must remain 19/20/21")
    if config.steps != 400 or config.checkpoints != (0, 50, 100, 200, 400):
        raise MechanismConfigError("mechanism schedule must remain fixed at step 400")
    return config


def build_student(
    data: MechanisticSeedData,
    seed: int,
) -> MechanisticGraphTemporalRetina:
    torch.manual_seed(seed)
    config = MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
    )
    return build_mechanistic_retina(
        config,
        data.cone_positions,
        data.cell_positions,
        data.cell_types,
        data.polarities,
    )


def pathway_rfs(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    observed_counts: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    no_h1 = frozenset({PathwayClamp.H1})
    no_optional = frozenset(
        {
            PathwayClamp.H1,
            PathwayClamp.AMACRINE_LOCAL,
            PathwayClamp.AMACRINE_TRANSIENT,
        }
    )
    full = _effective_rf(ClampedRFRequest(model, cones, observed_counts, frozenset()))
    h1_off = _effective_rf(ClampedRFRequest(model, cones, observed_counts, no_h1))
    bc = _effective_rf(ClampedRFRequest(model, cones, observed_counts, no_optional))
    return {"BC": bc, "AC": h1_off - bc, "H1": full - h1_off}


def _effective_rf(request: ClampedRFRequest) -> torch.Tensor:
    stimulus = request.cones.detach().clone().requires_grad_(True)
    logits = request.model.forward_sequence(
        stimulus,
        observed_counts=request.observed_counts,
        clamps=request.clamps,
    ).logits[:, -1]
    rows = []
    for cell in range(logits.shape[-1]):
        gradient = torch.autograd.grad(
            logits[:, cell].sum(),
            stimulus,
            retain_graph=cell + 1 < logits.shape[-1],
        )[0]
        rows.append(gradient[:, -request.model.config.lag_steps :])
    return torch.stack(rows, dim=1).detach()


__all__ = [
    "MechanismConfigError",
    "MechanismRunConfig",
    "build_student",
    "load_mechanism_config",
    "pathway_rfs",
]
