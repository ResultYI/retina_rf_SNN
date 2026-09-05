from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from evaluation.mechanistic_retina.metrics import JsonValue


FINAL_TEST_BOUNDARY = (
    "TEST_SPLIT_ACCESSED_FOR_IDENTITY_ONLY",
    "TEST_EXAMPLES_NOT_USED_FOR_INFERENCE_METRICS_MODEL_SELECTION_OR_CONCLUSIONS",
    "FINAL_TEST_SCIENTIFIC_EVALUATION_NOT_CONSUMED",
)


def write_json(
    path: Path, payload: JsonValue | Mapping[str, JsonValue]
) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "FINAL_TEST_BOUNDARY",
    "write_json",
]
