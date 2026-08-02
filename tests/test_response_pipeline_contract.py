from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from data.synthetic_teacher import load_teacher_rf_metadata


def test_synthetic_rf_metadata_loader_reads_kernels_and_recovery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic.h5"
    with h5py.File(path, "w") as handle:
        teacher = handle.create_group("teacher")
        teacher.create_dataset("static_kernel", data=np.ones((2, 3, 1), dtype=np.float32))
        teacher.create_dataset("context_kernel_low", data=np.ones((2, 3, 1), dtype=np.float32))
        teacher.create_dataset(
            "context_kernel_high",
            data=np.ones((2, 3, 1), dtype=np.float32) * 2,
        )
        teacher.create_dataset(
            "context_gain_envelope",
            data=np.ones((4, 5, 2), dtype=np.float32),
        )

    metadata = load_teacher_rf_metadata(path)

    assert metadata is not None
    assert metadata.static_kernel.shape == (2, 3, 1)
    assert metadata.context_gain_envelope is not None
    assert metadata.context_gain_envelope.shape == (4, 5, 2)
