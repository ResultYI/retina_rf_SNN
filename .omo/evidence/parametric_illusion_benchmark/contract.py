from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from evaluation.mechanistic_retina.temporal_center_surround import _optional_box


DT_MS: Final = 1000 / 150
PITCH_DEG: Final = 4.6 / 256 * 3
ONSET_MS: Final = 300.0
DURATION_MS: Final = 100.0
TOTAL_MS: Final = 1000.0
CONTRASTS: Final = (0.0, 0.0625, 0.125, 0.25, 0.375, 0.5)
SBC_RADII_PX: Final = (0, 2, 4, 6, 8)
MACH_WIDTHS_PX: Final = (0, 2, 4, 8, 12, 16)
TIME_MS: Final = torch.arange(round(TOTAL_MS / DT_MS), dtype=torch.float32) * DT_MS
ACTIVE: Final = (TIME_MS >= ONSET_MS) & (TIME_MS < ONSET_MS + DURATION_MS)


@dataclass(frozen=True, slots=True)
class Comparison:
    family: str
    signature: str
    contrast: float
    extent_px: int
    a: int
    b: int


@dataclass(frozen=True, slots=True)
class StimulusBank:
    names: tuple[str, ...]
    patches: torch.Tensor
    drive: torch.Tensor
    history: torch.Tensor
    comparisons: tuple[Comparison, ...]


def _coordinates(radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.arange(radius, -radius - 1, -1),
        torch.arange(-radius, radius + 1),
        indexing="ij",
    )
    return x, y


def _crop(scene: torch.Tensor, x: int) -> torch.Tensor:
    return scene[24:41, 32 + x - 8 : 32 + x + 9].clone()


def build_bank() -> StimulusBank:
    patches: list[torch.Tensor] = [torch.zeros(17, 17), torch.zeros(17, 17)]
    names = ["SBC_zero_context_A", "SBC_zero_context_B"]
    rows: list[Comparison] = []
    local_x, local_y = _coordinates(8)
    for contrast in CONTRASTS:
        for radius in SBC_RADII_PX:
            if contrast == 0 or radius == 0:
                a, b = 0, 1
            else:
                context = (torch.maximum(local_x.abs(), local_y.abs()) <= radius) & (
                    torch.maximum(local_x.abs(), local_y.abs()) > 1
                )
                a, b = len(patches), len(patches) + 1
                patches.extend((context.float() * contrast, context.float() * -contrast))
                names.extend(
                    (f"SBC_C{contrast:g}_R{radius}_bright", f"SBC_C{contrast:g}_R{radius}_dark")
                )
            rows.append(Comparison("SBC", "bright_minus_dark", contrast, radius, a, b))

    mach_zero = len(patches)
    patches.extend(torch.zeros(17, 17) for _ in range(4))
    names.extend(("Mach_zero_dark", "Mach_zero_dark_uniform", "Mach_zero_bright", "Mach_zero_bright_uniform"))
    scene_x, _ = _coordinates(32)
    for contrast in CONTRASTS:
        for width in MACH_WIDTHS_PX:
            if contrast == 0:
                indices = (mach_zero, mach_zero + 1, mach_zero + 2, mach_zero + 3)
            else:
                scene = (
                    torch.where(scene_x < 0, -contrast, contrast).float()
                    if width == 0
                    else contrast * (scene_x.float() / (width / 2)).clamp(-1, 1)
                )
                position = 1 if width == 0 else width // 2
                dark, bright = _crop(scene, -position), _crop(scene, position)
                indices = tuple(range(len(patches), len(patches) + 4))
                patches.extend((dark, torch.full_like(dark, float(dark[8, 8])), bright,
                                torch.full_like(bright, float(bright[8, 8]))))
                names.extend((f"Mach_C{contrast:g}_W{width}_dark", f"Mach_C{contrast:g}_W{width}_dark_uniform",
                              f"Mach_C{contrast:g}_W{width}_bright", f"Mach_C{contrast:g}_W{width}_bright_uniform"))
            rows.extend((
                Comparison("Mach", "dark_ramp_minus_uniform", contrast, width, indices[0], indices[1]),
                Comparison("Mach", "bright_ramp_minus_uniform", contrast, width, indices[2], indices[3]),
            ))

    patch_tensor = torch.stack(patches)
    pulse = _optional_box(TIME_MS, ONSET_MS, DURATION_MS, DT_MS)
    drive = pulse[None, :, None] * patch_tensor.flatten(1)[:, None, :]
    return StimulusBank(tuple(names), patch_tensor, drive,
                        torch.zeros(drive.shape[0], drive.shape[1], 1), tuple(rows))


__all__ = [
    "ACTIVE", "CONTRASTS", "DT_MS", "DURATION_MS", "MACH_WIDTHS_PX",
    "ONSET_MS", "PITCH_DEG", "SBC_RADII_PX", "TIME_MS", "Comparison",
    "StimulusBank", "build_bank",
]
