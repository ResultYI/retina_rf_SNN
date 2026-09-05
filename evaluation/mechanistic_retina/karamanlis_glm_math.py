from __future__ import annotations

from dataclasses import dataclass

import torch

from baselines.local_point_process_glm import LocalPointProcessGLM
from data.retinal_recording import RealSequenceSplit


@dataclass(frozen=True, slots=True)
class StaticFlashDesign:
    spatial_by_source: torch.Tensor
    source_indices: torch.Tensor
    active_time: torch.Tensor


@dataclass(frozen=True, slots=True)
class GLMMathError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def build_static_flash_design(split: RealSequenceSplit) -> StaticFlashDesign:
    source_ids = tuple(dict.fromkeys(split.source_image_ids))
    first_by_source = {
        source_id: split.source_image_ids.index(source_id)
        for source_id in source_ids
    }
    first_indices = torch.tensor(
        tuple(first_by_source[source_id] for source_id in source_ids),
        dtype=torch.long,
    )
    representatives = split.cone_drive[first_indices]
    active = representatives.ne(0).any(dim=-1)
    if not bool(active.any(dim=1).all()):
        raise GLMMathError("every source needs a nonzero flash frame")
    if not all(torch.equal(row, active[0]) for row in active):
        raise GLMMathError("source images do not share one flash timing")
    first_active = active.to(dtype=torch.int64).argmax(dim=1)
    spatial = representatives[
        torch.arange(representatives.shape[0]), first_active
    ]
    if not torch.equal(spatial[:, None] * active[:, :, None], representatives):
        raise GLMMathError("stimulus is not a static image flash sequence")
    source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
    source_indices = torch.tensor(
        tuple(source_lookup[source_id] for source_id in split.source_image_ids),
        dtype=torch.long,
    )
    for sequence, source_index in enumerate(source_indices):
        if not torch.equal(
            split.cone_drive[sequence], representatives[int(source_index)]
        ):
            raise GLMMathError("repeated source image stimuli differ")
    return StaticFlashDesign(spatial, source_indices, active[0].to(spatial.dtype))


def local_static_flash_logits(
    model: LocalPointProcessGLM,
    design: StaticFlashDesign,
    observed_counts: torch.Tensor,
) -> torch.Tensor:
    unique = model.static_flash_logits(
        design.spatial_by_source,
        design.active_time,
    )
    logits = unique[design.source_indices] + model.bias.view(1, 1, -1)
    return model.add_history(logits, observed_counts)


__all__ = [
    "GLMMathError",
    "StaticFlashDesign",
    "build_static_flash_design",
    "local_static_flash_logits",
]
