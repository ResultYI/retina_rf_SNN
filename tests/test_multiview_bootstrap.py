from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training import training_batch
from training.augmentation import augment_clip_pair
from training.bootstrap import (
    MultiViewBootstrapContext,
    MultiViewBootstrapRuntime,
    MultiViewReadouts,
    apply_multiview_bootstrap,
    calibrate_view_consistency_weight,
)
from training.config import load_config
from training.data import PreparedClip
from training.state import BootstrapState
from training.training_batch import (
    TrainingBatchRequest,
    TrainingGenerators,
)


ROOT = Path(__file__).resolve().parents[1]


def test_noise_views_share_context_and_target_but_not_noise() -> None:
    # Given: one clean sequence and a deterministic augmentation generator.
    config = load_config(ROOT / "configs" / "experiment.yaml")
    clip = PreparedClip(
        clean=torch.linspace(
            -1.0,
            1.0,
            config.data.sequence_steps * 2,
        ).reshape(config.data.sequence_steps, 2),
        source_id="source",
    )

    # When: two training views are generated together.
    first, second = augment_clip_pair(
        clip,
        config.data,
        torch.Generator().manual_seed(19),
    )

    # Then: only the sampled noise realization differs.
    assert torch.equal(first.clean_target, second.clean_target)
    assert first.metadata == second.metadata
    assert not torch.equal(first.noisy_input, second.noisy_input)


def test_bootstrap_batch_keeps_paired_views_in_matching_halves() -> None:
    # Given: five sources and a four-source bootstrap batch.
    config = load_config(ROOT / "configs" / "experiment.yaml")
    sources = tuple(
        PreparedClip(
            clean=torch.full(
                (config.data.sequence_steps, 1),
                float(index),
            ),
            source_id=str(index),
        )
        for index in range(5)
    )

    # When: one training batch is assembled.
    batch = training_batch.build_training_batch(
        TrainingBatchRequest(
            sources=sources,
            config=config,
            device=torch.device("cpu"),
            generators=TrainingGenerators(
                sampling=torch.Generator().manual_seed(1),
                augmentation=torch.Generator().manual_seed(2),
            ),
            optimizer_step=0,
        )
    )

    # Then: both halves contain the same sources and clean targets.
    assert len(batch) == 2 * config.training.batch_size
    for first, second in zip(
        batch[: config.training.batch_size],
        batch[config.training.batch_size :],
        strict=True,
    ):
        assert first.metadata["source_id"] == second.metadata["source_id"]
        assert torch.equal(first.clean_target, second.clean_target)


def test_view_consistency_weight_targets_quarter_core_gradient() -> None:
    # Given: primary and consistency losses with a known gradient ratio.
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    primary_loss = (2.0 * parameter).square()
    consistency_loss = parameter.square()

    # When: the consistency weight is calibrated once.
    weight = calibrate_view_consistency_weight(
        primary_loss,
        consistency_loss,
        (parameter,),
    )

    # Then: its weighted gradient is one quarter of the primary gradient.
    assert weight == pytest.approx(1.0)


def test_multiview_bootstrap_uses_persistent_reconstruction_and_variance_guard() -> None:
    # Given: two sources with two generator views and one shared target per source.
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    first = parameter * torch.tensor(
        [[[[0.0]], [[1.0]]], [[[1.0]], [[2.0]]]]
    )
    second = parameter * torch.tensor(
        [[[[0.5]], [[1.5]]], [[[1.5]], [[2.5]]]]
    )
    generator = torch.cat((first, second))
    target = torch.zeros(4, 2, 1)
    prediction = parameter.expand_as(target)
    state = BootstrapState()

    # When: the multi-view bootstrap objective is applied.
    application = apply_multiview_bootstrap(
        MultiViewReadouts(
            generator_readout=generator,
            target=target,
            persistent_prediction=prediction,
        ),
        MultiViewBootstrapContext(
            reconstruction_scale=1.0,
            view_consistency_scale=1.0,
            generator_variance_weight=0.1,
        ),
        MultiViewBootstrapRuntime(
            state=state,
            parameters=(parameter,),
        ),
    )

    # Then: reconstruction stays in the calibrated decoder coordinates and
    # the variance reference is captured for later collapse prevention.
    assert torch.equal(application.prediction, prediction)
    assert application.auxiliary_loss.requires_grad
    assert state.initial_generator_variance_reference is not None
    assert state.view_consistency_base_weight is not None
    assert state.generator_variance_retention == pytest.approx(1.0)
