from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from loss.retina import RetinaLosses
from models.cells.rgc_types import RGCOutput
from training.augmentation import AugmentedClip
from training.config import load_config
from training.trainer import RetinaTrainer


ROOT = Path(__file__).resolve().parents[1]


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tau = torch.nn.Parameter(torch.tensor(1.0))


def _fake_result(
    model: TinyModel,
    decoder: torch.nn.Linear,
    batch_size: int,
) -> tuple[RetinaLosses, RGCOutput, None]:
    total = model.tau.square() + decoder.weight.square().sum()
    zero = total * 0.0
    output_tensor = torch.zeros(batch_size, 1, 2, 1)
    output = RGCOutput(
        hard_spikes=output_tensor,
        surrogate_spikes=output_tensor,
        spike_probability=output_tensor,
        rates=output_tensor,
        generator_potential=output_tensor,
    )
    return (
        RetinaLosses(
            total=total,
            reconstruction=total,
            normalized_reconstruction=total,
            energy=zero,
            budget_energy=zero,
            energy_penalty=zero,
            energy_violation=zero,
            wiring=zero,
            variance_floor=zero,
            phenotype_repulsion=zero,
            homeostasis=zero,
        ),
        output,
        None,
    )


def _clip() -> AugmentedClip:
    return AugmentedClip(
        noisy_input=torch.zeros(1, 2, 1),
        clean_target=torch.zeros(1, 2, 1),
        metadata={"source_id": "test"},
    )


def test_trainer_constructs_with_true_batch_configuration() -> None:
    # Given: the canonical experiment configuration.
    config = load_config(ROOT / "configs" / "experiment.yaml")

    # When: a trainer is constructed.
    trainer = RetinaTrainer(
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
        object(),
        config,
        1.0,
    )

    # Then: one optimizer step expects a four-source batch.
    assert trainer.config.training.batch_size == 4


def test_optimizer_step_uses_one_true_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a bootstrap trainer and four augmented clips.
    config = load_config(ROOT / "configs" / "experiment.yaml")
    model = TinyModel()
    decoder = torch.nn.Linear(1, 1, bias=False)
    trainer = RetinaTrainer(model, decoder, object(), config, 1.0)
    observed_batch_sizes: list[int] = []

    def fake_forward(
        noisy_input: torch.Tensor,
        clean_target: torch.Tensor,
        *,
        checkpointed: bool,
    ) -> tuple[RetinaLosses, RGCOutput, None]:
        del clean_target, checkpointed
        observed_batch_sizes.append(noisy_input.shape[0])
        return _fake_result(model, decoder, noisy_input.shape[0])

    monkeypatch.setattr(trainer, "forward_clip", fake_forward)

    # When: one optimizer step is executed.
    result = trainer.train_optimizer_step(
        (_clip(),) * config.training.batch_size
    )

    # Then: the core sees one true batch while the persistent decoder is frozen.
    assert observed_batch_sizes == [config.training.batch_size]
    assert result.gradient_norm > 0
    assert result.temporal_gradient_norm > 0
    assert result.metrics["model_gradient_norm"] > 0
    assert result.metrics["decoder_gradient_norm"] == 0
    assert result.metrics["h1_gradient_norm"] == 0


def test_decoder_is_frozen_for_configured_optimizer_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a one-step decoder freeze after cross-fit bootstrap is disabled.
    config = load_config(ROOT / "configs" / "experiment.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            decoder_freeze_steps=1,
            reconstruction_bootstrap_steps=0,
        ),
    )
    model = TinyModel()
    decoder = torch.nn.Linear(1, 1, bias=False)
    trainer = RetinaTrainer(model, decoder, object(), config, 1.0)

    def fake_forward(
        noisy_input: torch.Tensor,
        clean_target: torch.Tensor,
        *,
        checkpointed: bool,
    ) -> tuple[RetinaLosses, RGCOutput, None]:
        del clean_target, checkpointed
        return _fake_result(model, decoder, noisy_input.shape[0])

    monkeypatch.setattr(trainer, "forward_clip", fake_forward)
    initial_decoder = decoder.weight.detach().clone()

    # When: two optimizer steps are executed.
    trainer.train_optimizer_step((_clip(),) * config.training.batch_size)
    frozen_decoder = decoder.weight.detach().clone()
    trainer.train_optimizer_step((_clip(),) * config.training.batch_size)

    # Then: the decoder changes only after the configured freeze expires.
    assert torch.equal(frozen_decoder, initial_decoder)
    assert not torch.equal(decoder.weight.detach(), frozen_decoder)
