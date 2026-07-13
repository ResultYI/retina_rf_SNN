from __future__ import annotations

import torch

from loss.retina import RetinaLossConfig, RetinaObjective
from models.decoder.local_decoder import LocalDecoderOutput
from training.hybrid import (
    HybridRetinaTrainer,
    HybridTrainingConfig,
    RetinaTargets,
    RetinaTrainingBatch,
    TrainingStage,
)
from test_local_decoder import _decoder
from test_retina_snn import _core, _state_tensors


def _batch() -> RetinaTrainingBatch:
    generator = torch.Generator().manual_seed(7)
    return RetinaTrainingBatch(
        x_cone=torch.randn((2, 5, 4), generator=generator),
        targets=RetinaTargets(
            fine=torch.randn((2, 3, 4), generator=generator),
            coarse=torch.randn((2, 3, 2), generator=generator),
        ),
    )


def _objective() -> RetinaObjective:
    return RetinaObjective(
        RetinaLossConfig(
            fine_weight=1.0,
            coarse_weight=0.5,
            fine_prediction_scale=4.0,
            coarse_prediction_scale=0.25,
            rate_weight=0.01,
            homeostasis_weight=0.02,
            residual_activity_weight=0.03,
            residual_decoder_weight=0.04,
            homeostasis_rate_min=0.01,
            homeostasis_rate_max=0.2,
        )
    )


def _trainer() -> HybridRetinaTrainer:
    core = _core()
    decoder = _decoder()
    optimizer = torch.optim.SGD(
        (*core.parameters(), *decoder.parameters()),
        lr=0.05,
    )
    return HybridRetinaTrainer(
        core,
        decoder,
        _objective(),
        optimizer,
        HybridTrainingConfig(t_bptt=2, grad_clip_norm=1.0),
    )


def test_primary_loss_excludes_decorrelation_and_uses_homeostasis_band() -> None:
    # Given
    config = RetinaLossConfig()

    # When / Then
    assert not hasattr(config, "decorrelation_weight")
    assert config.homeostasis_rate_min < config.homeostasis_rate_max


def test_retina_objective_combines_only_declared_prediction_and_regularizers() -> None:
    # Given
    trainer = _trainer()
    batch = _batch()
    rgc_history, _ = trainer.core.forward_sequence(batch.x_cone)
    decoded = trainer.decoder(rgc_history)
    prediction = LocalDecoderOutput(
        decoded.target_fine[:, -1],
        decoded.target_coarse[:, -1],
    )
    residual_penalty = trainer.decoder.residual_weight_penalty()

    # When
    losses = trainer.objective(
        prediction,
        batch.targets,
        rgc_history,
        residual_penalty,
    )

    # Then
    config = trainer.objective.config
    expected = (
        config.fine_weight
        * losses.prediction_fine
        / config.fine_prediction_scale
        + config.coarse_weight
        * losses.prediction_coarse
        / config.coarse_prediction_scale
        + config.rate_weight * losses.rate_regularization
        + config.homeostasis_weight * losses.homeostasis
        + config.residual_activity_weight * losses.residual_activity
        + config.residual_decoder_weight * losses.residual_decoder_weight
    )
    torch.testing.assert_close(losses.total, expected)
    assert losses.prediction_fine > 0
    assert losses.prediction_coarse > 0
    assert losses.residual_activity >= 0
    assert not hasattr(losses, "rf_target")


def test_decoder_warmup_updates_decoder_without_changing_core() -> None:
    # Given
    trainer = _trainer()
    core_before = tuple(
        parameter.detach().clone() for parameter in trainer.core.parameters()
    )
    decoder_before = tuple(
        parameter.detach().clone() for parameter in trainer.decoder.parameters()
    )

    # When
    result = trainer.train_batch(_batch(), TrainingStage.DECODER_WARMUP)

    # Then
    assert all(
        torch.equal(before, after)
        for before, after in zip(core_before, trainer.core.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            decoder_before,
            trainer.decoder.parameters(),
            strict=True,
        )
    )
    assert torch.isfinite(result.losses.total)
    assert all(not tensor.requires_grad for tensor in _state_tensors(result.state))


def test_core_finetune_updates_core_through_only_last_truncated_window() -> None:
    # Given
    trainer = _trainer()
    with torch.no_grad():
        trainer.decoder.fine_midget.raw_weight.fill_(0.05)
        trainer.decoder.coarse_parasol.raw_weight.fill_(0.05)
    core_before = tuple(
        parameter.detach().clone() for parameter in trainer.core.parameters()
    )

    # When
    result = trainer.train_batch(_batch(), TrainingStage.CORE_FINETUNE)

    # Then
    assert any(
        not torch.equal(before, after)
        for before, after in zip(core_before, trainer.core.parameters(), strict=True)
    )
    assert torch.isfinite(result.losses.total)
    assert all(not tensor.requires_grad for tensor in _state_tensors(result.state))


def test_evaluate_batch_preserves_parameters_and_reports_diagnostics() -> None:
    trainer = _trainer()
    before = tuple(
        parameter.detach().clone()
        for parameter in (*trainer.core.parameters(), *trainer.decoder.parameters())
    )

    result = trainer.evaluate_batch(_batch())

    after = (*trainer.core.parameters(), *trainer.decoder.parameters())
    assert all(torch.equal(first, second) for first, second in zip(before, after))
    assert torch.isfinite(result.losses.total)
    assert "h1" in result.core_diagnostics
    assert "amacrine" in result.core_diagnostics
    assert "rgc_midget_rate_mean" in result.core_diagnostics["rgc"]
