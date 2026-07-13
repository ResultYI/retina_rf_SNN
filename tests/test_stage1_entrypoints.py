from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from evaluation.residual_ablation import (
    population_ablation_report,
    residual_ablation_report,
)
from evaluation.rf_probe import (
    GradientRFRequest,
    RGCPopulationName,
    WhiteNoiseSTARequest,
    gradient_rf,
    white_noise_sta,
)
from models.cells.rgc import RGCOutput, RGCPopulationTensors
from loss.retina import RetinaLossConfig, RetinaObjective
from scripts.train_stage1 import (
    TrainStage1Config,
    TrainStage1Error,
    _loss_config_from_train_metrics,
    _parse_horizons,
    _restore_checkpoint,
    _validate_clip_fractions,
)
from training.hybrid import HybridRetinaTrainer, HybridTrainingConfig, TrainingStage
from training.stage1 import (
    MidgetSamplingMode,
    Stage1BuildConfig,
    Stage1OptimizerConfig,
    build_stage1_components,
    build_stage1_optimizer,
)


def _positions() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0],
         [0.4, 0.0], [0.5, 0.0], [0.6, 0.0], [0.7, 0.0]]
    )


def _median_row_count(matrix: torch.Tensor) -> float:
    matrix = matrix.coalesce()
    counts = torch.bincount(matrix.indices()[0], minlength=matrix.shape[0])
    return float(counts.float().median())


def _median_column_count(matrix: torch.Tensor) -> float:
    matrix = matrix.coalesce()
    counts = torch.bincount(matrix.indices()[1], minlength=matrix.shape[1])
    return float(counts.float().median())


def test_stage1_factory_builds_core_decoder_and_row_normalized_pools() -> None:
    # Given
    components = build_stage1_components(
        _positions(),
        Stage1BuildConfig(
            dt_ms=5.0,
            horizon_count=3,
            eccentricity_deg=2.5,
            midget_sampling=MidgetSamplingMode.CONVERGENT,
        ),
    )

    # When
    optimizer = build_stage1_optimizer(
        components.core,
        components.decoder,
        Stage1OptimizerConfig(core_lr=1e-4, decoder_lr=1e-3),
    )

    # Then
    assert components.profile.name == "human_macaque_v1"
    assert components.profile.eccentricity_deg == 2.5
    assert components.target_pools.fine.shape == (8, 8)
    assert components.target_pools.coarse.shape[1] == 8
    torch.testing.assert_close(
        torch.sparse.sum(components.target_pools.fine, dim=1).to_dense(),
        torch.ones(8),
    )
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-3


def test_stage1_factory_uses_spatially_distributed_population_positions() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0], [0.0, 0.1], [0.0, 0.2], [0.0, 0.3],
            [0.1, 0.0], [0.1, 0.1], [0.1, 0.2], [0.1, 0.3],
            [0.2, 0.0], [0.2, 0.1], [0.2, 0.2], [0.2, 0.3],
            [0.3, 0.0], [0.3, 0.1], [0.3, 0.2], [0.3, 0.3],
        ]
    )

    components = build_stage1_components(
        positions,
        Stage1BuildConfig(dt_ms=5.0, horizon_count=1),
    )

    parasol = components.mosaic.parasol_positions_degs
    residual = components.mosaic.residual_positions_degs
    assert parasol.shape[0] == 4
    assert torch.unique(parasol[:, 0]).numel() == 2
    assert torch.unique(parasol[:, 1]).numel() == 2
    assert residual.shape[0] <= parasol.shape[0]


def test_stage1_geometry_stays_local_at_training_mosaic_scale() -> None:
    axis = torch.linspace(-0.1, 0.1, 29)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    positions = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)

    components = build_stage1_components(
        positions,
        Stage1BuildConfig(dt_ms=5.0, horizon_count=1),
    )

    h1 = components.core.h1
    rgc = components.core.rgc
    assert 100 < h1.h1_positions_degs.shape[0] < positions.shape[0]
    assert 3 <= _median_column_count(h1.cone_to_h1) <= 6
    assert 20 <= _median_row_count(rgc.parasol_pool) <= 60
    assert 3 <= _median_row_count(components.decoder.fine_midget.local_mask) <= 12
    assert _median_row_count(rgc.residual_pool) < positions.shape[0] / 4


def test_decoder_warmup_checkpoint_initializes_core_finetune(tmp_path: Path) -> None:
    warmup = build_stage1_components(
        _positions(),
        Stage1BuildConfig(dt_ms=5.0, horizon_count=1),
    )
    with torch.no_grad():
        warmup.decoder.fine_midget.raw_weight.fill_(0.3)
    checkpoint = tmp_path / "warmup.pt"
    torch.save(
        {
            "epoch": 2,
            "step": 17,
            "core": warmup.core.state_dict(),
            "decoder": warmup.decoder.state_dict(),
            "optimizer": {},
            "stage": TrainingStage.DECODER_WARMUP.value,
        },
        checkpoint,
    )
    finetune = build_stage1_components(
        _positions(),
        Stage1BuildConfig(dt_ms=5.0, horizon_count=1),
    )
    trainer = HybridRetinaTrainer(
        finetune.core,
        finetune.decoder,
        RetinaObjective(RetinaLossConfig()),
        build_stage1_optimizer(
            finetune.core,
            finetune.decoder,
            Stage1OptimizerConfig(),
        ),
        HybridTrainingConfig(),
    )
    config = TrainStage1Config(
        train_h5=(Path("train.h5"),),
        val_h5=(),
        output_dir=tmp_path,
        epochs=1,
        batch_size=1,
        input_steps=2,
        horizons=(1,),
        stage=TrainingStage.CORE_FINETUNE,
        device=torch.device("cpu"),
        seed=7,
        t_bptt=1,
        lr_core=1e-4,
        lr_decoder=1e-3,
        num_workers=0,
        max_clip_fraction=0.01,
        resume=checkpoint,
    )

    progress = _restore_checkpoint(config, trainer)

    assert (progress.epoch, progress.step) == (0, 0)
    assert math.isinf(progress.best_loss)
    torch.testing.assert_close(
        trainer.decoder.fine_midget.raw_weight,
        warmup.decoder.fine_midget.raw_weight,
    )


def test_clip_fraction_gate_rejects_overclipped_export(tmp_path: Path) -> None:
    # Given
    export = tmp_path / "overclipped.h5"

    # When / Then
    with pytest.raises(TrainStage1Error, match="clip_fraction"):
        _validate_clip_fractions(((export, 0.02),), maximum=0.01)


def test_prediction_scales_come_from_train_zero_change_mse() -> None:
    config = _loss_config_from_train_metrics(
        {
            "zero_change_mse_fine": 4.0,
            "zero_change_mse_coarse": 0.25,
        }
    )

    assert config.fine_prediction_scale == 4.0
    assert config.coarse_prediction_scale == 0.25


def test_zero_variance_prediction_target_blocks_training() -> None:
    with pytest.raises(TrainStage1Error, match="zero-change MSE"):
        _loss_config_from_train_metrics(
            {
                "zero_change_mse_fine": 1.0,
                "zero_change_mse_coarse": 0.0,
            }
        )


def test_parse_horizons_accepts_positive_integer_steps() -> None:
    assert _parse_horizons("1,2,4") == (1, 2, 4)


def test_rf_probe_reports_gradient_rf_and_white_noise_sta_shapes() -> None:
    # Given
    components = build_stage1_components(
        _positions(),
        Stage1BuildConfig(dt_ms=5.0, horizon_count=1),
    )
    x_cone = torch.randn(2, 5, 8)

    # When
    gradient = gradient_rf(
        components.core,
        GradientRFRequest(x_cone, RGCPopulationName.MIDGET, 0, 0),
    )
    sta = white_noise_sta(
        components.core,
        WhiteNoiseSTARequest(8, 5, 4, RGCPopulationName.MIDGET, 0, 0),
    )

    # Then
    assert gradient.gradient.shape == (5, 8)
    assert gradient.response.shape == (2,)
    assert sta.sta.shape == (5, 8)
    assert torch.isfinite(sta.response_mean)


def test_residual_ablation_reports_residual_decoder_contribution() -> None:
    # Given
    components = build_stage1_components(
        _positions(),
        Stage1BuildConfig(dt_ms=5.0, horizon_count=1),
    )
    with torch.no_grad():
        components.decoder.fine_midget.raw_weight.fill_(0.5)
        components.decoder.coarse_midget.raw_weight.fill_(0.5)
        components.decoder.fine_parasol.raw_weight.fill_(0.5)
        components.decoder.coarse_parasol.raw_weight.fill_(0.5)
        components.decoder.fine_residual.raw_weight.fill_(0.5)
        components.decoder.coarse_residual.raw_weight.fill_(0.5)
    output = RGCOutput(
        spikes=RGCPopulationTensors(
            midget=torch.ones(1, 2, 8),
            parasol=torch.ones(1, 2, 2),
            residual=torch.ones(1, 2, 1),
        ),
        rates=RGCPopulationTensors(
            midget=torch.ones(1, 2, 8),
            parasol=torch.ones(1, 2, 2),
            residual=torch.ones(1, 2, 1),
        ),
    )

    # When
    report = residual_ablation_report(components.decoder, output)
    midget = population_ablation_report(components.decoder, output, "midget")
    parasol = population_ablation_report(components.decoder, output, "parasol")

    # Then
    assert report.fine_residual_contribution > 0
    assert report.coarse_residual_contribution > 0
    assert midget.fine_contribution > 0
    assert parasol.coarse_contribution > 0
