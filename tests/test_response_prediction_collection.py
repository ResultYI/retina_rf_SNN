from __future__ import annotations

import math

from evaluation.response_metrics import compute_response_metrics
from evaluation.response_predictions import (
    ResponsePredictionRequest,
    collect_response_predictions,
)
from tests.calibration_fixture import calibration_trainer


def test_trainer_evaluate_characterization_is_finite_and_cell_aligned() -> None:
    # Given
    trainer = calibration_trainer()

    # When
    metrics = trainer.evaluate(trainer.data.validation)

    # Then
    assert math.isfinite(metrics.nll)
    assert len(metrics.per_cell_nll) == 2


def test_collected_predictions_reproduce_trainer_metrics() -> None:
    # Given
    trainer = calibration_trainer()

    # When
    predictions = collect_response_predictions(
        ResponsePredictionRequest(
            trainer.model,
            trainer.data.validation,
            trainer.config.training.burn_in_steps,
            trainer.device,
            "observed",
        )
    )
    collected_metrics = compute_response_metrics(
        predictions.logits,
        predictions.targets,
        predictions.valid_mask,
        trainer.data.target_kind,
        trainer.baseline_rates,
    )

    # Then
    assert predictions.generator_potential.shape == predictions.logits.shape
    assert collected_metrics.nll == trainer.evaluate(trainer.data.validation).nll


def test_prediction_collection_restores_model_training_mode() -> None:
    trainer = calibration_trainer()
    trainer.model.train()

    collect_response_predictions(
        ResponsePredictionRequest(
            trainer.model,
            trainer.data.validation,
            trainer.config.training.burn_in_steps,
            trainer.device,
            "observed",
        )
    )

    assert trainer.model.training
