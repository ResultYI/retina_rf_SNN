from __future__ import annotations

from pathlib import Path

import pytest
import torch

from data.schottdorf_lee_2021 import SchottdorfDataError
from data.schottdorf_lee_catalog import (
    RecordingKind,
    SchottdorfRecording,
    mc_pc_recordings,
)
from data.schottdorf_lee_spikes import parse_recording_spike_trials


_REPOSITORY = Path("data/real/schottdorf_lee_2021_repository")
_AVAILABLE = _REPOSITORY.is_dir()


def _synthetic_recording(path: Path, kind: RecordingKind) -> SchottdorfRecording:
    return SchottdorfRecording(
        recording_id=path.stem,
        path=path,
        cell_id="fixture#1",
        recorded_cell_class="MC on",
        retinal_class="MC",
        canonical_cell_type="parasol",
        polarity="ON",
        recording_kind=kind,
        catalog_recording_kind=kind,
        eccentricity_deg=5.0,
    )


def test_parser_accepts_only_exact_duplicate_payload(tmp_path: Path) -> None:
    payload = "Total spikes\t1\nVideo Start\t0\nNo\tTime\n1\t10\n"
    path = tmp_path / "exact.txt"
    path.write_text(payload + payload, encoding="utf-8")

    spikes = parse_recording_spike_trials(
        _synthetic_recording(path, RecordingKind.TEN_MINUTE)
    )

    assert spikes.duplicate_payload_removed
    assert spikes.total_spikes == 1


def test_parser_rejects_conflicting_duplicate_payload(tmp_path: Path) -> None:
    first = "Total spikes\t1\nVideo Start\t0\nNo\tTime\n1\t10\n"
    second = "Total spikes\t2\nVideo Start\t0\nNo\tTime\n1\t10\n2\t20\n"
    path = tmp_path / "conflict.txt"
    path.write_text(first + second, encoding="utf-8")

    with pytest.raises(SchottdorfDataError, match="payloads disagree"):
        parse_recording_spike_trials(
            _synthetic_recording(path, RecordingKind.TEN_MINUTE)
        )


def test_parser_rejects_whitespace_mismatch_between_payloads(
    tmp_path: Path,
) -> None:
    payload = "Total spikes\t1\nVideo Start\t0\nNo\tTime\n1\t10\n"
    path = tmp_path / "whitespace-mismatch.txt"
    path.write_text(payload + payload + "   \n", encoding="utf-8")

    with pytest.raises(SchottdorfDataError, match="payloads disagree"):
        parse_recording_spike_trials(
            _synthetic_recording(path, RecordingKind.TEN_MINUTE)
        )


def test_parser_wraps_invalid_repeat_count_metadata(tmp_path: Path) -> None:
    path = tmp_path / "repeat.txt"
    path.write_text(
        "Total spikes\t7\n"
        "Video Starts\t0\n"
        "Spikes per rpt\n"
        "1\t1\t1\t1\t1\t1\tbad\n"
        "Spikes times\n",
        encoding="utf-8",
    )

    with pytest.raises(SchottdorfDataError, match="non-integer value"):
        parse_recording_spike_trials(
            _synthetic_recording(path, RecordingKind.REPEATED_ONE_MINUTE)
        )


@pytest.mark.skipif(not _AVAILABLE, reason="public macaque dataset is unavailable")
def test_repeated_recording_parser_preserves_six_real_trials() -> None:
    recording = next(
        item
        for item in mc_pc_recordings(_REPOSITORY / "data")
        if item.recording_id == "lSS01254"
    )

    spikes = parse_recording_spike_trials(recording)

    assert spikes.total_spikes == 19_912
    assert spikes.resolution_ms == pytest.approx(0.1)
    assert len(spikes.live_times_ms_by_trial) == 6
    assert all(torch.all(times >= 0) for times in spikes.live_times_ms_by_trial)
    assert all(torch.all(times < 60_000.0) for times in spikes.live_times_ms_by_trial)


@pytest.mark.skipif(not _AVAILABLE, reason="public macaque dataset is unavailable")
def test_parser_ignores_exactly_repeated_source_payload() -> None:
    recording = next(
        item
        for item in mc_pc_recordings(_REPOSITORY / "data")
        if item.recording_id == "lSS01159"
    )

    spikes = parse_recording_spike_trials(recording)

    assert spikes.total_spikes == 13_708
    assert len(spikes.live_times_ms_by_trial) == 6
    assert spikes.duplicate_payload_removed
