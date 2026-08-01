from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from data.input_identity import InputIdentityError, validate_experiment_input
from data.rgc_response import (
    RGCResponseContractError,
    load_rgc_response,
    validate_response_splits,
)


def test_v2_response_loads_input_identity(tmp_path: Path) -> None:
    path = tmp_path / "response.h5"
    _write_v2_response(path, source="train", species="macaque")

    session = load_rgc_response(path)

    assert session.input_identity.species == "macaque"
    assert session.input_identity.mosaic_fingerprint == "mosaic-sha256"
    assert session.input_identity.stimulus_source_fingerprints == (
        "train-content-0",
        "train-content-1",
    )


def test_split_rejects_input_identity_mismatch(tmp_path: Path) -> None:
    train = tmp_path / "train.h5"
    validation = tmp_path / "validation.h5"
    _write_v2_response(train, source="train", species="macaque")
    _write_v2_response(validation, source="validation", species="human")

    with pytest.raises(RGCResponseContractError, match="input identity"):
        validate_response_splits(
            (load_rgc_response(train),),
            (load_rgc_response(validation),),
        )


def test_real_canonical_input_rejects_human_provenance(tmp_path: Path) -> None:
    path = tmp_path / "human.h5"
    _write_v2_response(path, source="human", species="human")

    with pytest.raises(InputIdentityError, match="macaque"):
        validate_experiment_input(load_rgc_response(path).input_identity, 5.0)


def _write_v2_response(path: Path, *, source: str, species: str) -> None:
    text = h5py.string_dtype(encoding="utf-8")
    spikes = np.zeros((2, 1, 6, 2), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "format_version",
            data=np.frombuffer(b"retina-rgc-response-v2", dtype=np.uint8),
        )
        handle.attrs["response_target_kind"] = "bernoulli"
        handle.create_dataset("cone_response", data=np.ones((2, 6, 3), dtype=np.float32))
        handle.create_dataset("spike_counts", data=spikes)
        handle.create_dataset("valid_mask", data=np.ones_like(spikes, dtype=np.uint8))
        handle.create_dataset("time_axis_seconds", data=np.arange(6) * 0.005)
        handle.create_dataset("cone/position_degs", data=np.zeros((3, 2)))
        handle.create_dataset("cell/id", data=np.asarray(["c0", "c1"], dtype=text))
        handle.create_dataset(
            "cell/type_id",
            data=np.asarray(["midget", "parasol"], dtype=text),
        )
        handle.create_dataset("cell/polarity", data=np.asarray([0, 1], dtype=np.uint8))
        handle.create_dataset("cell/position_degs", data=np.zeros((2, 2)))
        handle.create_dataset("cell/eccentricity_deg", data=np.asarray([4.0, 4.0]))
        handle.create_dataset(
            "stimulus/source_id",
            data=np.asarray([f"{source}-0", f"{source}-1"], dtype=text),
        )
        handle.create_dataset(
            "stimulus/context_id",
            data=np.asarray(["low", "high"], dtype=text),
        )
        handle.create_dataset(
            "stimulus/source_content_sha256",
            data=np.asarray(
                [f"{source}-content-0", f"{source}-content-1"],
                dtype=text,
            ),
        )
        identity = handle.create_group("input")
        for key, value in {
            "dataset_kind": "real_recording",
            "species": species,
            "optics_species": "macaque",
            "mosaic_species": "macaque",
            "photoreceptor_mode": "cone_only",
            "chromatic_mode": "achromatic",
            "light_level": "photopic",
            "response_units": "isomerizations_per_integration_time",
            "cone_mosaic_id": "mosaic-1",
            "cone_mosaic_fingerprint": "mosaic-sha256",
            "generator_name": "recording-exporter",
            "generator_revision": "1",
            "cone_bin_reference": "interval_end",
            "spike_bin_reference": "interval_end",
        }.items():
            identity.create_dataset(key, data=np.frombuffer(value.encode(), dtype=np.uint8))
        identity.create_dataset("mean_luminance_cd_m2", data=100.0)
        identity.create_dataset("cone_type", data=np.asarray([1, 2, 1], dtype=np.uint8))
        identity.create_dataset("stimulus_to_spike_offset_bins", data=0)
