from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from torch.nn import functional as F

from evaluation.global_probe import GlobalReadoutGeometry
from evaluation.representation_diagnostics import (
    DecoderExamples,
    collect_decoder_examples,
)
from evaluation.representation_selection import (
    RepresentationSelectionInputs,
    RepresentationSelectionLog,
    RepresentationSelectionSnapshot,
    evaluate_representation_selection,
    selection_log,
)
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel
from training.augmentation import AugmentedClip
from training.state import ValidationState


@dataclass(frozen=True, slots=True)
class RepresentationSelectionRequest:
    model: RetinaModel
    decoder: TiedLocalDecoder
    training_clips: Sequence[AugmentedClip]
    validation_clips: Sequence[AugmentedClip]
    supervised_steps: int


class RepresentationSelector:
    def __init__(self, request: RepresentationSelectionRequest) -> None:
        self._request = request
        train_examples, validation_examples = self._collect_examples()
        geometry = self._geometry()
        fixed_prediction = request.decoder(
            validation_examples.rates,
            geometry.spatial_weights,
        )
        self._baseline = evaluate_representation_selection(
            RepresentationSelectionInputs(
                train_examples=train_examples,
                validation_examples=validation_examples,
                geometry=geometry,
                fixed_validation_mse=float(
                    F.mse_loss(
                        fixed_prediction,
                        validation_examples.target,
                    ).detach()
                ),
            ),
            None,
        )

    @property
    def baseline(self) -> RepresentationSelectionSnapshot:
        return self._baseline

    def evaluate(
        self,
        fixed_validation_mse: float,
    ) -> RepresentationSelectionSnapshot:
        train_examples, validation_examples = self._collect_examples()
        return evaluate_representation_selection(
            RepresentationSelectionInputs(
                train_examples=train_examples,
                validation_examples=validation_examples,
                geometry=self._geometry(),
                fixed_validation_mse=fixed_validation_mse,
            ),
            self._baseline,
        )

    def write_baseline(self, output_dir: Path) -> None:
        (output_dir / "representation_selector_initial.json").write_text(
            json.dumps(asdict(self._baseline), indent=2),
            encoding="utf-8",
        )

    def observe(
        self,
        state: ValidationState,
        fixed_validation_mse: float,
    ) -> tuple[bool, RepresentationSelectionLog]:
        snapshot = self.evaluate(fixed_validation_mse)
        event = state.observe_representation(snapshot.metrics)
        return event, selection_log(snapshot, event)

    def _collect_examples(
        self,
    ) -> tuple[DecoderExamples, DecoderExamples]:
        return (
            collect_decoder_examples(
                self._request.model,
                self._request.training_clips,
                self._request.supervised_steps,
            ),
            collect_decoder_examples(
                self._request.model,
                self._request.validation_clips,
                self._request.supervised_steps,
            ),
        )

    def _geometry(self) -> GlobalReadoutGeometry:
        return GlobalReadoutGeometry(
            spatial_weights=(
                self._request.model.rgc.compute_spatial_weights()
            ),
            gain_max=self._request.decoder.gain_max,
        )


__all__ = [
    "RepresentationSelectionRequest",
    "RepresentationSelector",
]
