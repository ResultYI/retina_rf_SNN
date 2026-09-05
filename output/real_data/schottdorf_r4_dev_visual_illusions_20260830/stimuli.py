from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Final

import torch

PITCH_DEG: Final = 4.6 / 256 * 3
DT_MS: Final = 1000 / 150
ONSET_MS: Final = 300.0
DURATION_MS: Final = 100.0
TOTAL_MS: Final = 1000.0


@dataclass(frozen=True, slots=True)
class Pair:
    family: str
    name: str
    a: int
    b: int
    control: bool = False
    x_pixels: int | None = None


@dataclass(frozen=True, slots=True)
class Stimuli:
    names: tuple[str, ...]
    patches: torch.Tensor
    pairs: tuple[Pair, ...]
    scenes: dict[str, torch.Tensor]
    crop_centers_pixels: dict[str, tuple[tuple[int, int], ...]]


def _coordinates(radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(torch.arange(radius, -radius - 1, -1),
                          torch.arange(-radius, radius + 1), indexing="ij")
    return x, y


def _crop(scene: torch.Tensor, x: int, y: int = 0) -> torch.Tensor:
    return scene[32 - y - 8:32 - y + 9, 32 + x - 8:32 + x + 9].clone()


def build_stimuli() -> Stimuli:
    x, y = _coordinates(32)
    mach = 0.25 * (x.float() / 4).clamp(-1, 1)
    sbc = torch.where(x < 0, -0.25, 0.25)
    for center in (-16, 16):
        sbc[(abs(x - center) <= 1) & (abs(y) <= 1)] = 0
    horizontal = abs((y + 8) % 16 - 8) <= 1
    vertical = abs((x + 8) % 16 - 8) <= 1
    hermann = torch.where(horizontal | vertical, 0.25, -0.25)
    corridor_control = torch.where(horizontal, 0.25, -0.25)
    white = torch.where(torch.div(x + 1, 3, rounding_mode="floor") % 2 == 0, 0.25, -0.25)
    for center in (-18, 21):
        white[(abs(x - center) <= 1) & (abs(y) <= 3)] = 0
    neutral = torch.zeros_like(mach)
    patches = [_crop(mach, position) for position in range(-12, 13)]
    names = [f"Mach_x{position:+03d}" for position in range(-12, 13)]
    patches.extend(torch.full((17, 17), float(patch[8, 8])) for patch in patches.copy())
    names.extend(f"Mach_uniform_x{position:+03d}" for position in range(-12, 13))
    pairs = [Pair("Mach_bands", f"ramp_minus_matched_uniform_x{position:+03d}",
                  position + 12, position + 37, x_pixels=position) for position in range(-12, 13)]
    scenes = {"Mach_bands": mach, "SBC": sbc, "SBC_control": neutral,
              "Hermann": hermann, "Hermann_control": corridor_control,
              "White": white, "White_control": neutral}
    centers = {"Mach_bands": ((-4, 0), (4, 0)), "SBC": ((16, 0), (-16, 0)),
               "Hermann": ((0, 0), (8, 0)), "White": ((-18, 0), (21, 0))}
    for family, label_a, label_b in (
        ("SBC", "bright_surround", "dark_surround"),
        ("Hermann", "intersection", "corridor"),
        ("White", "on_bright_bar", "on_dark_bar"),
    ):
        start = len(patches)
        for control in (False, True):
            scene = scenes[f"{family}_control" if control else family]
            for (cx, cy), label in zip(centers[family], (label_a, label_b), strict=True):
                patches.append(_crop(scene, cx, cy))
                names.append(f"{family}_{'control_' if control else ''}{label}")
        pairs.extend((Pair(family, f"{label_a}_minus_{label_b}", start, start + 1),
                      Pair(family, f"control_{label_a}_minus_{label_b}", start + 2, start + 3, True)))
    names.append("neutral_blank")
    patches.append(torch.zeros(17, 17))
    return Stimuli(tuple(names), torch.stack(patches), tuple(pairs), scenes, centers)


def save_stimuli(bank: Stimuli, out: Path) -> None:
    contract = {
        "checkpoint_root": "schottdorf_r4_development_22cell_20260830_verified",
        "input": "Calibrated L+M Weber drive (L+M-background)/background; no RGB gamma or synthetic frontend",
        "relative_LM_levels": [0.75, 1.0, 1.25], "absolute_luminance_cd_m2": None,
        "dt_ms": DT_MS, "pitch_deg": PITCH_DEG, "patch_shape": [17, 17],
        "scene_shape": [65, 65], "onset_ms": ONSET_MS, "duration_ms": DURATION_MS,
        "total_ms": TOTAL_MS, "spike_history": "Identical all-zero observed counts; conditional inference, not free-running spikes",
        "polarity": "Same physical luminance for ON and OFF; no polarity-dependent stimulus inversion",
        "sampling": "Local crops translated onto the saved cone grid; x right/y up; no invented cell positions",
        "Mach_bands": "Continuous ramp from -0.25 at x=-4 pixels to +0.25 at x=+4; flat plateaus outside; scan -12..12. Control at each position is spatially uniform, matching the central pixel luminance.",
        "SBC": "Identical 3x3 neutral targets on -0.25/+0.25 surrounds; control removes surround difference to neutral background.",
        "Hermann": "Bright grid width 3 pixels, period 16, on dark background; compare intersection to horizontal corridor 8 pixels away. Control removes vertical bars, leaving identical horizontal corridors.",
        "White": "Alternating vertical bars width 3 pixels; identical 3x7 neutral rectangles replace sections of bright/dark bars. Control removes grating to neutral background.",
        "control_matching": "Target luminance, target pixels, local center position and timing are matched. Surround luminance/histogram is intentionally not matched.",
        "metrics": "Raw logit/probability; within-mode blank subtraction; A-B physical-target contrast; clamp-minus-normal; stimulus-on 300<=t<400ms mean; peak/integral and onset/offset time courses. No perceptual decoder or illusion score.",
        "Mach_boundary_regions_pixels": [[-6, -2], [2, 6]],
        "Mach_overshoot": "Positive excursion above max of the two remote-plateau responses; negative excursion below their min; computed from stimulus-on response means in each fixed boundary region, in original response units. Uniform matched-control profile evaluated identically.",
        "time_quantities": "Learned tau and explicit pathway delay frozen in ms. No RF lag window evaluated. RGC history uses existing strictly-past one-bin shift with zero observed counts.",
        "references": ["https://doi.org/10.1016/0042-6989(95)00341-X",
                       "https://michaelbach.de/ot/lum-herGrid/", "https://michaelbach.de/ot/lum-white/"],
        "names": bank.names, "pairs": [asdict(pair) for pair in bank.pairs],
        "crop_centers_pixels": bank.crop_centers_pixels,
        "protocol_fixed_before_inference": True,
    }
    (out / "stimulus-contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    torch.save({"patches": bank.patches, "names": bank.names, "scenes": bank.scenes,
                "cone_positions_degs": torch.stack(_coordinates(8), dim=-1).reshape(-1, 2) * PITCH_DEG},
               out / "stimuli.pt")
