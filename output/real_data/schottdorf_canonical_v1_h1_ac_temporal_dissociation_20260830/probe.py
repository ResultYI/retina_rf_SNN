#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0"]
# ///
# How to run: D:/anaconda/python.exe -B probe.py (frozen repository environment).
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

DT_MS: Final = 1000.0 / 150.0
ONSET_MS: Final = 300.0
ACTIVE_MS: Final = 2000.0
TOTAL_MS: Final = 3000.0
CONTRAST: Final = 0.25
EVENT_MS: Final = 50.0
NAMES: Final = ("slow_step", "slow_1Hz", "slow_2Hz", "rapid_10Hz", "rapid_20Hz", "transient_50ms", "large_field_50ms")
REFERENCES: Final = (7, 7, 7, 7, 7, 7, 8)
POLARITY: Final = {"ON": 1.0, "OFF": -1.0}


@dataclass(frozen=True, slots=True)
class Interval:
    onset_ms: float
    duration_ms: float
    sign: float


@dataclass(frozen=True, slots=True)
class ProbeBank:
    drive: torch.Tensor
    time_ms: torch.Tensor
    waveforms: torch.Tensor
    intervals: tuple[tuple[Interval, ...], ...]
    center: torch.Tensor
    annulus: torch.Tensor


def box(interval: Interval, time_ms: torch.Tensor) -> torch.Tensor:
    """Exact duration overlap with each native 150-Hz input bin."""
    left = torch.maximum(time_ms, time_ms.new_tensor(interval.onset_ms))
    right = torch.minimum(time_ms + DT_MS, time_ms.new_tensor(interval.onset_ms + interval.duration_ms))
    return interval.sign * ((right - left) / DT_MS).clamp(0, 1)


def build_bank(supports: tuple[torch.Tensor, torch.Tensor], polarity: str) -> ProbeBank:
    """Freeze temporal variants on the existing center and stimulus annulus."""
    center, ac_disk = (x.bool().flatten() for x in supports)
    annulus = ac_disk & ~center
    assert center.any() and annulus.any() and (center <= ac_disk).all()
    time = torch.arange(round(TOTAL_MS / DT_MS), dtype=torch.float64) * DT_MS
    intervals = [(Interval(ONSET_MS, 1000.0, 1.0), Interval(ONSET_MS + 1000, 1000.0, -1.0))]
    for frequency in (1, 2, 10, 20):
        half_period = 500.0 / frequency
        intervals.append(tuple(Interval(ONSET_MS + j * half_period, half_period, (-1.0) ** j)
                               for j in range(4 * frequency)))
    transient = (Interval(ONSET_MS, 50.0, 1.0), Interval(ONSET_MS + 1000, 50.0, -1.0))
    intervals.extend((transient, transient))
    waveforms = torch.stack([sum((box(i, time) for i in events), torch.zeros_like(time)) for events in intervals])
    center_envelope = box(Interval(ONSET_MS, ACTIVE_MS, 1.0), time)
    sign = POLARITY[polarity]
    center_drive = sign * CONTRAST * center_envelope[:, None] * center
    drive = [center_drive + sign * CONTRAST * wave[:, None] * annulus for wave in waveforms[:6]]
    drive.extend((sign * CONTRAST * waveforms[6, :, None] * torch.ones_like(center),
                  center_drive, sign * CONTRAST * waveforms[6, :, None] * center,
                  torch.zeros_like(center_drive)))
    return ProbeBank(torch.stack(drive).float(), time, waveforms, tuple(intervals), center, annulus)


def verify_bank(bank: ProbeBank) -> None:
    """Check temporal matching independently of any checkpoint response."""
    assert bank.drive.shape == (10, 450, bank.center.numel())
    assert torch.isfinite(bank.drive).all()
    assert bank.waveforms.mean(1).abs().max() < 1e-12
    assert torch.allclose(bank.waveforms.max(1).values, torch.ones(7, dtype=torch.float64))
    assert torch.allclose(bank.waveforms.min(1).values, -torch.ones(7, dtype=torch.float64))
    assert torch.count_nonzero(bank.center & bank.annulus) == 0
    assert torch.allclose(bank.drive[:6].double().mean(1), bank.drive[7].double().mean(0).expand(6, -1), atol=1e-8, rtol=0)
    assert all(torch.equal(bank.drive[j, :, bank.center], bank.drive[7, :, bank.center]) for j in range(6))
    assert (bank.waveforms[:5].abs().sum(1) * DT_MS <= ACTIVE_MS + 1e-9).all()
    for events in bank.intervals:
        assert sum(e.sign * e.duration_ms for e in events) == 0
    assert abs(float(box(bank.intervals[5][0], bank.time_ms).sum()) * DT_MS - 50) < 1e-9
    assert len(bank.intervals[1]) == 4 and len(bank.intervals[4]) == 80


if __name__ == "__main__":
    fixture = build_bank((torch.tensor([1, 0, 0]), torch.tensor([1, 1, 0])), "ON")
    verify_bank(fixture)
    print("probe mean/contrast/geometry/native-bin contracts PASS")
