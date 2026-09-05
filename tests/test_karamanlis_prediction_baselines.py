from __future__ import annotations

import pytest
import torch

from baselines.local_point_process_glm import LocalPointProcessGLM
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import (
    PopulationGLMTrainingRequest,
    constant_rate_logits,
    fit_population_glm,
    grouped_nll,
    winner_counts,
)


def test_constant_rate_uses_only_training_bins() -> None:
    # Given: train events and a distinct validation tensor with one invalid bin.
    train_events = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])
    train_mask = torch.ones_like(train_events, dtype=torch.bool)
    validation_events = torch.zeros(1, 3, 2)
    validation_mask = torch.tensor(
        [[[True, True], [True, False], [True, True]]]
    )

    # When: constant-rate logits are fitted and expanded over validation.
    logits = constant_rate_logits(
        train_events,
        train_mask,
        validation_events,
        validation_mask,
    )

    # Then: rates are train-only and invalid validation bins do not affect fitting.
    expected = torch.logit(torch.tensor([0.5, 0.99999]))
    torch.testing.assert_close(logits[0, 0], expected)
    torch.testing.assert_close(logits[0, 2], expected)


def test_local_glm_history_is_strictly_causal() -> None:
    model = LocalPointProcessGLM(
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        radius_deg=2.0,
        temporal_lags=3,
    )
    with torch.no_grad():
        model.history.copy_(torch.tensor([[0.7, -0.3, 0.2, 0.1]]))
    cones = torch.arange(12.0).reshape(1, 6, 2) / 10
    history = torch.zeros(1, 6, 1)
    changed_current = history.clone()
    changed_current[:, 4] = 1.0

    reference = model(cones, history)
    changed = model(cones, changed_current)

    torch.testing.assert_close(changed[:, :5], reference[:, :5], rtol=0, atol=0)
    assert not torch.equal(changed[:, 5:], reference[:, 5:])


def test_group_and_winner_summaries_cover_every_cell() -> None:
    # Given: three models over four cells spanning the four requested RGC classes.
    per_model = {
        "constant_rate": (0.4, 0.3, 0.4, 0.3),
        "glm": (0.3, 0.4, 0.2, 0.2),
        "retinal": (0.2, 0.2, 0.3, 0.1),
    }
    polarities = ("ON", "ON", "OFF", "OFF")
    cell_types = ("midget", "parasol", "midget", "parasol")

    # When: winner counts and four-class means are summarized.
    winners = winner_counts(per_model)
    groups = grouped_nll(per_model, polarities, cell_types)

    # Then: all cells have one winner and every class retains all model values.
    assert winners == {"constant_rate": 0, "glm": 1, "retinal": 3}
    assert sum(winners.values()) == 4
    assert set(groups) == {
        "ON midget",
        "ON parasol",
        "OFF midget",
        "OFF parasol",
    }
    assert groups["OFF parasol"]["retinal"] == pytest.approx(0.1)


def test_local_glm_full_objective_fit_does_not_explode_on_first_fit() -> None:
    # Given: two local cells, two static flashes, measured events, and no validation.
    images = torch.tensor([[0.8, -0.2, 0.1], [-0.3, 0.4, 0.9]])
    cones = torch.zeros(4, 6, 3)
    cones[:2, 1:4] = images[0]
    cones[2:, 1:4] = images[1]
    events = torch.tensor(
        [
            [[0, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [0, 0], [0, 0]],
        ],
        dtype=torch.float32,
    )
    train = RealSequenceSplit(
        cone_drive=cones,
        spike_counts=events.to(dtype=torch.int64),
        spike_events=events,
        valid_mask=torch.ones_like(events, dtype=torch.bool),
        source_image_ids=("a", "a", "b", "b"),
        trial_indices=(0, 1, 2, 3),
    )

    # When: the train-only local GLM is fit with its full convex objective.
    result = fit_population_glm(
        PopulationGLMTrainingRequest(
            train=train,
            cone_positions=torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            cell_positions=torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
            graph_radius_deg=0.2,
            temporal_lags=2,
            steps=8,
            seed=17,
        )
    )

    # Then: strong-Wolfe fitting reduces train NLL with exactly the local capacity.
    assert result.train_nll_trained <= result.train_nll_initial
    assert result.gradients_finite
    assert sum(parameter.numel() for parameter in result.model.parameters()) == 14


def test_local_glm_explicit_rf_mask_excludes_undeclared_cones() -> None:
    # Given: two cells with measured RF masks over three shared cone features.
    support = torch.tensor([[True, False, True], [False, True, False]])
    model = LocalPointProcessGLM(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        None,
        temporal_lags=2,
        support_mask=support,
    )
    with torch.no_grad():
        for kernel in model.kernels:
            kernel.fill_(1.0)
    reference = torch.zeros(1, 5, 3)
    changed = reference.clone()
    changed[:, 2, 1] = 7.0
    history = torch.zeros(1, 5, 2)

    # When: an undeclared cone for cell zero changes inside the same stimulus tensor.
    reference_logits = model(reference, history)
    changed_logits = model(changed, history)

    # Then: cell zero is invariant and capacity follows exactly the declared masks.
    assert torch.equal(reference_logits[..., 0], changed_logits[..., 0])
    assert model.support_counts == (2, 1)
    assert sum(parameter.numel() for parameter in model.parameters()) == 16
