from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from data.karamanlis_2024 import (
    CellSelection,
    KaramanlisAdapterConfig,
    load_marmoset_imagesequence,
)
from evaluation.mechanistic_retina.karamanlis_real_run import (
    KaramanlisRealRunConfig,
    run_karamanlis_real_training,
)
from training.mechanistic_retina.real_sampled import spike_prediction_metrics


_SESSION = Path(
    "data/real/karamanlis_2024/sessions/20220301_252MEA_marmoset_left_n1"
)


def test_per_cell_metrics_preserve_population_likelihood_contract() -> None:
    # Given: two cells with respectively accurate and inaccurate predictions.
    logits = torch.tensor([[[-2.0, 2.0], [2.0, -2.0]]])
    events = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    mask = torch.ones_like(events, dtype=torch.bool)

    # When: population and per-cell likelihoods are summarized.
    metrics = spike_prediction_metrics(logits, events, mask)

    # Then: the population mean and both cell-specific losses are retained.
    assert metrics.population_nll == pytest.approx(1.126928, abs=1e-6)
    assert metrics.per_cell_nll == pytest.approx((0.126928, 2.126928), abs=1e-6)


@pytest.mark.skipif(not _SESSION.exists(), reason="Karamanlis session not downloaded")
def test_population_adapter_uses_all_quality_one_target_cells() -> None:
    # Given: a bounded real-data fixture using every target cell and four images.
    config = KaramanlisAdapterConfig(
        train_image_count=2,
        validation_image_count=2,
        cell_selection=CellSelection.ALL_QUALITY_1_TARGETS,
        crop_pixels=153,
        pool_factor=3,
    )

    # When: the population adapter reads the measured session.
    data = load_marmoset_imagesequence(_SESSION, config)

    # Then: all 64 target cells are retained and the cone grid covers them.
    assert Counter(zip(data.polarities, data.cell_types, strict=True)) == {
        ("ON", "parasol"): 29,
        ("ON", "midget"): 14,
        ("OFF", "parasol"): 16,
        ("OFF", "midget"): 5,
    }
    assert data.train.cone_drive.shape[-1] == 51 * 51
    assert torch.all(data.cell_positions_degs.amin(0) >= data.cone_positions_degs.amin(0))
    assert torch.all(data.cell_positions_degs.amax(0) <= data.cone_positions_degs.amax(0))


@pytest.mark.skipif(not _SESSION.exists(), reason="Karamanlis session not downloaded")
def test_population_run_saves_per_cell_and_type_results(tmp_path: Path) -> None:
    # Given: a one-step population run over a small source-disjoint split.
    config = KaramanlisRealRunConfig(
        session_dir=_SESSION,
        output_dir=tmp_path / "population-run",
        steps=1,
        learning_rate=0.03,
        batch_size=1,
        seed=202_603_02,
        adapter=KaramanlisAdapterConfig(
            train_image_count=1,
            validation_image_count=1,
            cell_selection=CellSelection.ALL_QUALITY_1_TARGETS,
            crop_pixels=153,
            pool_factor=3,
        ),
    )

    # When: the canonical real-data training entry point is executed.
    result = run_karamanlis_real_training(config)

    # Then: auditable per-cell and four-class validation results are saved.
    payload = json.loads((result.artifact_dir / "results.json").read_text())
    assert len(payload["validation_prediction"]["per_cell"]) == 64
    assert set(payload["validation_prediction"]["by_cell_class"]) == {
        "ON parasol",
        "ON midget",
        "OFF parasol",
        "OFF midget",
    }
