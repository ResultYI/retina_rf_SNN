from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from baselines.local_point_process_glm import LocalPointProcessGLM
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.schottdorf_baseline_run import (
    SchottdorfBaselineRunConfig,
    run_schottdorf_prediction_baselines,
)
from evaluation.mechanistic_retina.schottdorf_prediction_baselines import (
    DynamicGLMTrainingRequest,
    evaluate_dynamic_glm,
    fit_dynamic_glm,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    sha256_file,
)


_REPOSITORY = Path("data/real/schottdorf_lee_2021_repository")
_MOVIE = Path("data/real/schottdorf_lee_2021_macaque/1x10_256.mpg")
_RETINAL = Path(
    "output/real_data/"
    "schottdorf_lee_2021_22cell_canonical_v1_revision4_fresh_20260829"
)
_AVAILABLE = _REPOSITORY.is_dir() and _MOVIE.is_file() and _RETINAL.is_dir()


def test_dynamic_glm_stimulus_and_history_are_causal() -> None:
    model = LocalPointProcessGLM(
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        radius_deg=None,
        temporal_lags=3,
        history_lags=2,
        support_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    with torch.no_grad():
        model.kernels[0].fill_(0.5)
        model.history.copy_(torch.tensor([[0.7, -0.2]]))
    cones = torch.zeros(1, 7, 2)
    spikes = torch.zeros(1, 7, 1)
    future_cones = cones.clone()
    future_spikes = spikes.clone()
    future_cones[:, 5] = 3.0
    future_spikes[:, 5] = 1.0

    reference = model(cones, spikes)
    changed = model(future_cones, future_spikes)

    torch.testing.assert_close(changed[:, :5], reference[:, :5], rtol=0, atol=0)
    assert not torch.equal(changed[:, 5:], reference[:, 5:])
    torch.testing.assert_close(changed[:, 5], reference[:, 5] + 3.0)


def test_dynamic_glm_fit_is_train_only_and_reduces_train_nll() -> None:
    cones = torch.tensor(
        [
            [[0.0, 0.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0]],
            [[0.0, 0.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]],
        ]
    )
    events = torch.tensor(
        [
            [[0.0], [1.0], [1.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )
    train = RealSequenceSplit(
        cone_drive=cones,
        spike_counts=events.to(dtype=torch.int64),
        spike_events=events,
        valid_mask=torch.ones_like(events, dtype=torch.bool),
        source_image_ids=("train-a", "train-b"),
        trial_indices=(0, 0),
    )

    fitted = fit_dynamic_glm(
        DynamicGLMTrainingRequest(
            train=train,
            cone_positions=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            cell_positions=torch.tensor([[0.0, 0.0]]),
            temporal_lags=2,
            history_lags=2,
            max_iterations=20,
            seed=7,
        )
    )
    metrics, logits = evaluate_dynamic_glm(fitted.model, train)

    assert fitted.train_nll_trained < fitted.train_nll_initial
    assert fitted.gradients_finite
    assert fitted.actually_updated
    assert fitted.solver_evaluations >= fitted.solver_iterations
    assert torch.isfinite(torch.tensor(fitted.final_gradient_max))
    assert logits.shape == events.shape
    assert metrics.population_nll == pytest.approx(fitted.train_nll_trained)
    assert sum(parameter.numel() for parameter in fitted.model.parameters()) == 7


@pytest.mark.skipif(not _AVAILABLE, reason="macaque baseline sources unavailable")
def test_runner_replays_retinal_without_modifying_checkpoint(tmp_path: Path) -> None:
    checkpoint = _RETINAL / "cells" / "67_4" / "model-trained.pt"
    before = sha256_file(checkpoint)

    result = run_schottdorf_prediction_baselines(
        SchottdorfBaselineRunConfig(
            repository_dir=_REPOSITORY,
            movie_path=_MOVIE,
            retinal_artifact_dir=_RETINAL,
            output_dir=tmp_path / "matched-baselines",
            cell_ids=("67#4",),
            glm_max_iterations=1,
        )
    )
    payload = json.loads((result.artifact_dir / "results.json").read_text())

    assert sha256_file(checkpoint) == before
    assert payload["cell_count"] == 1
    assert payload["recording_count"] == 1
    assert payload["feature_contract"] == {
        "history": "strictly_past_same_cell_spike_events",
        "history_lags": 4,
        "stimulus": "current_and_past_l_plus_m_luminance",
        "stimulus_lags": 16,
        "stimulus_features": 289,
    }
    cell = payload["cells"][0]
    assert cell["retinal_nll_replay_error"] < 1e-7
    assert cell["source_checkpoint_sha256_before"] == before
    assert cell["source_checkpoint_sha256_after"] == before
    assert cell["time_segment_disjoint"] is True


def test_runner_rejects_output_inside_source_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source-retinal"

    with pytest.raises(RuntimeError, match="must not overwrite or contain"):
        run_schottdorf_prediction_baselines(
            SchottdorfBaselineRunConfig(
                repository_dir=tmp_path / "repository",
                movie_path=tmp_path / "movie.mpg",
                retinal_artifact_dir=source,
                output_dir=source / "baseline-output",
            )
        )
