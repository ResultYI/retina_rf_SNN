from __future__ import annotations

from dataclasses import dataclass

import torch

from data.schottdorf_lee_2021 import SchottdorfDataError
from data.schottdorf_lee_catalog import RecordingKind, SchottdorfRecording


_SPIKE_RESOLUTION_MS = 0.1
_ONE_MINUTE_MS = 60_000.0
_TEN_MINUTES_MS = 600_000.0


@dataclass(frozen=True, slots=True)
class RecordingSpikeTrials:
    total_spikes: int
    video_start_ticks: int
    resolution_ms: float
    live_times_ms_by_trial: tuple[torch.Tensor, ...]
    duplicate_payload_removed: bool


def parse_recording_spike_trials(
    recording: SchottdorfRecording,
) -> RecordingSpikeTrials:
    text = recording.path.read_text(encoding="utf-8")
    payload, duplicate_removed = _first_payload(text)
    lines = payload.splitlines()
    total_spikes = _header_integer(lines, "Total spikes")
    if recording.recording_kind is RecordingKind.TEN_MINUTE:
        video_start = _header_integer(lines, "Video Start")
        times = _single_trial_times(lines, total_spikes)
        live = (_live_times(times, _TEN_MINUTES_MS),)
    else:
        video_start = _header_integer(lines, "Video Starts")
        trial_times = _repeated_trial_times(lines, total_spikes)
        live = tuple(_live_times(times, _ONE_MINUTE_MS) for times in trial_times[:6])
        if len(live) != 6:
            raise SchottdorfDataError("6x1min recording must contain six video trials")
    return RecordingSpikeTrials(
        total_spikes=total_spikes,
        video_start_ticks=video_start,
        resolution_ms=_SPIKE_RESOLUTION_MS,
        live_times_ms_by_trial=live,
        duplicate_payload_removed=duplicate_removed,
    )


def _first_payload(text: str) -> tuple[str, bool]:
    starts = []
    cursor = 0
    while (start := text.find("Total spikes", cursor)) >= 0:
        starts.append(start)
        cursor = start + len("Total spikes")
    if len(starts) == 1:
        return text, False
    if not starts:
        return text, False
    payloads = tuple(
        text[start:stop]
        for start, stop in zip(starts, starts[1:] + [len(text)], strict=True)
    )
    if any(payload != payloads[0] for payload in payloads[1:]):
        raise SchottdorfDataError("duplicated spike payloads disagree")
    return payloads[0], True


def _header_integer(lines: list[str], name: str) -> int:
    prefix = f"{name}\t"
    for line in lines:
        if line.startswith(prefix):
            try:
                return int(line.removeprefix(prefix))
            except ValueError as error:
                raise SchottdorfDataError(f"invalid {name} metadata") from error
    raise SchottdorfDataError(f"spike table lacks {name} metadata")


def _single_trial_times(lines: list[str], total_spikes: int) -> torch.Tensor:
    try:
        start = lines.index("No\tTime") + 1
    except ValueError as error:
        raise SchottdorfDataError("10min spike table lacks No/Time header") from error
    recorded = []
    for line in lines[start:]:
        fields = line.split("\t")
        if len(fields) == 2 and fields[1].strip():
            recorded.append(_integer_time(fields[1]))
    if len(recorded) != total_spikes:
        raise SchottdorfDataError("10min spike table count disagrees with metadata")
    return torch.tensor(recorded, dtype=torch.float64) * _SPIKE_RESOLUTION_MS


def _repeated_trial_times(
    lines: list[str], total_spikes: int
) -> tuple[torch.Tensor, ...]:
    try:
        count_header = lines.index("Spikes per rpt") + 1
        table_start = lines.index("Spikes times") + 1
    except ValueError as error:
        raise SchottdorfDataError("6x1min spike table lacks repeat metadata") from error
    count_line = next((line for line in lines[count_header:] if line.strip()), None)
    if count_line is None:
        raise SchottdorfDataError("6x1min spike table lacks repeat counts")
    try:
        expected = tuple(
            int(field) for field in count_line.split("\t") if field.strip()
        )
    except ValueError as error:
        raise SchottdorfDataError(
            "6x1min repeat counts contain a non-integer value"
        ) from error
    if len(expected) != 7 or sum(expected) != total_spikes:
        raise SchottdorfDataError("6x1min repeat counts disagree with metadata")
    recorded: list[list[int]] = [[] for _ in expected]
    for line in lines[table_start:]:
        fields = line.split("\t")
        for index, value in enumerate(fields[1 : len(expected) + 1]):
            if value.strip():
                recorded[index].append(_integer_time(value))
    if tuple(map(len, recorded)) != expected:
        raise SchottdorfDataError("6x1min spike columns disagree with repeat counts")
    return tuple(
        torch.tensor(times, dtype=torch.float64) * _SPIKE_RESOLUTION_MS
        for times in recorded
    )


def _integer_time(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise SchottdorfDataError("spike table contains a non-integer time") from error


def _live_times(times_ms: torch.Tensor, duration_ms: float) -> torch.Tensor:
    return times_ms[(times_ms >= 0) & (times_ms < duration_ms)]


__all__ = ["RecordingSpikeTrials", "parse_recording_spike_trials"]
