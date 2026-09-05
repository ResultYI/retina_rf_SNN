from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from baselines.graph_tcn import GraphTCN
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.schottdorf_neural_baseline import (
    CompactNeuralTrainingRequest,
    compact_neural_logits,
    fit_compact_neural_baseline,
    graph_tcn_spatial_drive,
)
from evaluation.mechanistic_retina.schottdorf_final_benchmark_run import (
    run_final_prediction_benchmark,
)
from evaluation.mechanistic_retina.schottdorf_final_benchmark_cell import (
    _verify_recording_hashes,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    sha256_file,
)
from evaluation.mechanistic_retina.schottdorf_final_benchmark_types import (
    FinalBenchmarkConfig,
)


def _split() -> RealSequenceSplit:
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
    return RealSequenceSplit(
        cone_drive=cones,
        spike_counts=events.to(dtype=torch.int64),
        spike_events=events,
        valid_mask=torch.ones_like(events, dtype=torch.bool),
        source_image_ids=("train-a", "train-b"),
        trial_indices=(0, 0),
    )


def test_cached_compact_neural_forward_matches_graph_tcn_and_is_causal() -> None:
    cones_pos = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
    cells_pos = torch.tensor([[0.0, 0.0]])
    model = GraphTCN(cones_pos, cells_pos, width=3, history_lags=2)
    cones = torch.randn(1, 8, 2)
    spikes = torch.zeros(1, 8, 1)
    local = graph_tcn_spatial_drive(model, cones)

    torch.testing.assert_close(
        compact_neural_logits(model, local, spikes),
        model(cones, spikes),
        rtol=0,
        atol=1e-7,
    )
    changed_cones = cones.clone()
    changed_spikes = spikes.clone()
    changed_cones[:, 5] += 3.0
    changed_spikes[:, 5] = 1.0
    changed = compact_neural_logits(
        model, graph_tcn_spatial_drive(model, changed_cones), changed_spikes
    )
    reference = compact_neural_logits(model, local, spikes)
    torch.testing.assert_close(changed[:, :5], reference[:, :5], rtol=0, atol=1e-7)


def test_compact_neural_fit_uses_training_split_and_reduces_train_nll() -> None:
    train = _split()
    fitted = fit_compact_neural_baseline(
        CompactNeuralTrainingRequest(
            train=train,
            cone_positions=torch.tensor([[0.0, 0.0], [0.1, 0.0]]),
            cell_positions=torch.tensor([[0.0, 0.0]]),
            target_parameters=148,
            seed=7,
            maximum_steps=80,
            patience=20,
        )
    )

    assert fitted.train_nll_trained < fitted.train_nll_initial
    assert fitted.gradients_finite
    assert fitted.actually_updated
    assert fitted.validation_used is False
    assert fitted.model.receptive_field_steps == 17


def test_final_runner_rejects_output_inside_frozen_source_tree(
    tmp_path,
) -> None:
    retinal = tmp_path / "retinal-source"
    glm = tmp_path / "glm-source"

    for source, output in (
        (retinal, retinal / "nested-output"),
        (glm, glm / "nested-output"),
    ):
        with pytest.raises(RuntimeError, match="outside frozen source artifact trees"):
            run_final_prediction_benchmark(
                FinalBenchmarkConfig(
                    repository_dir=tmp_path / "repository",
                    movie_path=tmp_path / "movie.mpg",
                    retinal_artifact_dir=retinal,
                    glm_artifact_dir=glm,
                    output_dir=output,
                )
            )


def test_final_runner_rejects_mutated_spike_recording(tmp_path) -> None:
    spike_path = tmp_path / "recording.txt"
    spike_path.write_text("original", encoding="utf-8")
    recording = SimpleNamespace(path=spike_path)
    source = {"source_sha256": {spike_path.name: sha256_file(spike_path)}}
    _verify_recording_hashes((recording,), source)

    spike_path.write_text("mutated", encoding="utf-8")
    with pytest.raises(RuntimeError, match="spike/source artifact hash mismatch"):
        _verify_recording_hashes((recording,), source)
