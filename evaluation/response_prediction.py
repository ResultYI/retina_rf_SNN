from __future__ import annotations

from evaluation.response_report_schema import ResponsePredictionEvidence
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit
from training.response_trainer import ResponseTrainer


def evaluate_response_prediction(
    trainer: ResponseTrainer,
    split: ResponseSplit,
    initialized_model: ResponseRetinaModel,
) -> ResponsePredictionEvidence:
    return ResponsePredictionEvidence(
        conditional=trainer.evaluate(split),
        initialized_conditional=trainer.evaluate(split, model=initialized_model),
        zero_history=trainer.evaluate(split, history_mode="zero"),
        shuffled_history=(
            trainer.evaluate(split, history_mode="shuffled")
            if split.spike_counts.shape[1] >= 2
            else None
        ),
        free_running=trainer.evaluate(split, history_mode="free_running"),
    )


__all__ = ["evaluate_response_prediction"]
